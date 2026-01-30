use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use crate::config::Config;
use crate::dictionary_io::{load_dict, save_dict};
use crate::document_state::DocumentState;
use crate::git_commit_parser::GitCommitParser;
use crate::ignored_lints_io::{load_ignored_lints, save_ignored_lints};
use crate::io_utils::fileify_path;
use anyhow::{Context, Result, anyhow};
use futures::future::join;
use harper_comments::CommentParser;
use harper_core::linting::{LintGroup, LintGroupConfig};
use harper_core::parsers::{
    CollapseIdentifiers, IsolateEnglish, Markdown, OrgMode, Parser, PlainEnglish,
};
use harper_core::spell::{
    Dictionary, FstDictionary, MergedDictionary, MutableDictionary,
};
use harper_core::keyboard_distance::avg_keyboard_distance;
use harper_core::{Dialect, DictWordMetadata, Document, IgnoredLints};
use harper_html::HtmlParser;
use harper_ink::InkParser;
use harper_jjdescription::JJDescriptionParser;
use harper_literate_haskell::LiterateHaskellParser;
use harper_python::PythonParser;
use harper_stats::{Record, Stats};
use harper_typst::Typst;
use serde_json::Value;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{Duration, Instant, sleep};
use tower_lsp_server::jsonrpc::Result as JsonResult;
use tower_lsp_server::lsp_types::notification::PublishDiagnostics;
use tower_lsp_server::lsp_types::{
    CodeActionOrCommand, CodeActionParams, CodeActionProviderCapability, CodeActionResponse,
    CompletionItem, CompletionItemKind, CompletionOptions, CompletionParams, CompletionResponse,
    ConfigurationItem, Diagnostic, DidChangeConfigurationParams, DidChangeTextDocumentParams,
    DidChangeWatchedFilesParams, DidChangeWatchedFilesRegistrationOptions,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, ExecuteCommandOptions,
    ExecuteCommandParams, FileChangeType, FileSystemWatcher, GlobPattern, InitializeParams,
    InitializeResult, InitializedParams, MessageType, PublishDiagnosticsParams, Range,
    Registration, ServerCapabilities, ServerInfo, TextDocumentSyncCapability, TextDocumentSyncKind,
    TextDocumentSyncOptions, TextDocumentSyncSaveOptions, Uri, WatchKind,
};
use tower_lsp_server::{Client, LanguageServer, UriExt};
use tracing::{error, info, warn};

/// Return harper-ls version
pub fn ls_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Cache entry for dictionaries with their modification times
#[derive(Clone)]
struct DictCacheEntry {
    dict: Arc<MergedDictionary>,
    user_dict_mtime: Option<SystemTime>,
    workspace_dict_mtime: Option<SystemTime>,
    file_dict_mtime: Option<SystemTime>,
}

pub struct Backend {
    client: Client,
    root: RwLock<PathBuf>,
    config: RwLock<Config>,
    stats: RwLock<Stats>,
    doc_state: Mutex<HashMap<Uri, DocumentState>>,
    pending_changes: RwLock<HashMap<Uri, Instant>>,
    dict_cache: RwLock<HashMap<Uri, DictCacheEntry>>,
}

/// Calculate a completion score for ranking suggestions.
/// Higher scores are better.
/// Uses multiple signals to determine relevance:
/// - Edit distance (primary signal - exact matches win)
/// - First letter match (strong signal - users rarely mistype first letter)
/// - Prefix match length (longer common prefixes = better matches)
/// - Character overlap at same positions (more matching chars = better)
/// - Position of first difference (later errors more forgivable)
/// - Substring containment (query contained in candidate)
/// - Transposition detection (adjacent character swaps are common typos)
/// - Common word bonus (scaled by length to avoid overwhelming proper nouns)
/// - Length similarity (prefer words closer to query length)
/// - Character frequency similarity (similar character distributions)
/// - Last character match (small bonus for matching endings)
fn calculate_completion_score(
    query: &[char],
    candidate: &[char],
    edit_distance: u8,
    is_common: bool,
    is_transposition: bool,
    avg_kbd_distance: f32,
) -> f32 {
    let mut score = 100.0;

    // === PRIMARY SIGNALS ===

    // Progressive edit distance penalty - create bigger gaps between distance levels
    // Exact matches (distance=0) should dominate, followed by single typos (distance=1)
    score -= match edit_distance {
        0 => 0.0,
        1 => 20.0,
        2 => 50.0,
        _ => 100.0,
    };

    // First letter match - very strong signal of intent
    // Users rarely mistype the first letter
    if !query.is_empty() && !candidate.is_empty() {
        if query[0].eq_ignore_ascii_case(&candidate[0]) {
            score += 50.0;
        } else {
            // First letter mismatch is a very bad sign for completions
            score -= 30.0;
        }
    }

    // === SECONDARY SIGNALS (break ties for same edit distance) ===

    // Prefix match length - longer matching prefixes indicate better completions
    // This is a critical tie-breaker for same edit distance
    let prefix_match_len = query
        .iter()
        .zip(candidate.iter())
        .take_while(|(q, c)| q.eq_ignore_ascii_case(c))
        .count();
    score += (prefix_match_len as f32) * 8.0;

    // Character overlap at same positions
    // Count how many characters match at their exact positions
    let char_overlap = query
        .iter()
        .zip(candidate.iter())
        .filter(|(q, c)| q.eq_ignore_ascii_case(c))
        .count();
    score += (char_overlap as f32) * 3.0;

    // Position of first difference - errors at the end are more forgivable
    // Users may still be typing, so later errors are more acceptable
    if edit_distance > 0 {
        let first_diff_pos = query
            .iter()
            .zip(candidate.iter())
            .position(|(q, c)| !q.eq_ignore_ascii_case(c))
            .unwrap_or_else(|| query.len().min(candidate.len()));

        let position_ratio = first_diff_pos as f32 / query.len().max(1) as f32;
        // Exponential bonus: 0.0 -> 0 points, 0.5 -> 3.75, 1.0 -> 15 points
        score += position_ratio * position_ratio * 15.0;
    }

    // Substring containment - does candidate contain query as substring?
    // Example: "his" is contained in "this" starting at position 1
    // Only award this bonus if edit_distance > 0 (not an exact prefix match)
    if edit_distance > 0 && candidate.len() >= query.len() {
        let query_lower: Vec<char> = query.iter().map(|c| c.to_ascii_lowercase()).collect();
        let candidate_lower: Vec<char> = candidate.iter().map(|c| c.to_ascii_lowercase()).collect();

        if candidate_lower
            .windows(query.len())
            .any(|window| window == query_lower.as_slice())
        {
            score += 25.0;
        }
    }

    // Progressive length difference penalty
    // Prefer candidates with similar length to the query
    // Balanced to handle both "myy"→"my" and "tthi"→"this" cases
    let len_diff = (candidate.len() as i32 - query.len() as i32).abs();
    score -= match len_diff {
        0 => 0.0,
        1 => 8.0,   // Moderate penalty for 1-char difference
        2 => 25.0,  // Stronger penalty for 2-char difference
        3 => 40.0,  // Even stronger for 3-char difference
        _ => (len_diff as f32) * 15.0,
    };

    // Transposition bonus (adjacent letter swaps are very common typos)
    // Example: "teh" -> "the"
    if is_transposition {
        score += 30.0;
    }

    // Common word bonus - scaled by word length
    // Short common words (to, the, of) get big bonus
    // Longer common words get smaller bonus so they don't dominate proper nouns
    if is_common {
        if candidate.len() <= 3 {
            score += 60.0; // Reduced from 80 - still prioritized but not overwhelming
        } else if candidate.len() <= 5 {
            score += 35.0; // Reduced from 40
        } else {
            score += 15.0; // Reduced from 20
        }
    }

    // === TERTIARY SIGNALS (fine-grained tie-breakers) ===

    // Character frequency similarity - do the words use similar character distributions?
    // This helps differentiate words with same edit distance
    // Example: "ths" vs "tab" - "ths" has more similar character distribution to "this"
    if edit_distance >= 2 {
        let freq_similarity = calculate_char_frequency_similarity(query, candidate);
        score += freq_similarity * 5.0;
    }

    // Last character match - small bonus for matching endings
    if !query.is_empty() && !candidate.is_empty() {
        if query
            .last()
            .unwrap()
            .eq_ignore_ascii_case(candidate.last().unwrap())
        {
            score += 3.0;
        }
    }

    // Keyboard distance penalty (only for edit distance > 0)
    // Lower keyboard distance = better score (keys physically closer)
    // Conservative weighting: 8.0 points per unit distance
    if edit_distance > 0 {
        // avg_kbd_distance ranges from 0.0 (same key) to ~1.0 (max normalized)
        // Penalty ranges from 0 (same keys) to ~8 (far keys)
        score -= avg_kbd_distance * 8.0;
    }

    score
}

/// Calculate similarity between two words based on character frequency distribution.
/// Returns a value between 0.0 (completely different) and 1.0 (identical distribution).
fn calculate_char_frequency_similarity(a: &[char], b: &[char]) -> f32 {
    let mut freq_a = [0u8; 26];
    let mut freq_b = [0u8; 26];

    // Build frequency histogram for word a
    for &c in a {
        if c.is_ascii_alphabetic() {
            let idx = (c.to_ascii_lowercase() as u8 - b'a') as usize;
            if idx < 26 {
                freq_a[idx] = freq_a[idx].saturating_add(1);
            }
        }
    }

    // Build frequency histogram for word b
    for &c in b {
        if c.is_ascii_alphabetic() {
            let idx = (c.to_ascii_lowercase() as u8 - b'a') as usize;
            if idx < 26 {
                freq_b[idx] = freq_b[idx].saturating_add(1);
            }
        }
    }

    // Calculate similarity as inverse of sum of absolute differences
    let diff_sum: u32 = freq_a
        .iter()
        .zip(freq_b.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs() as u32)
        .sum();

    // Normalize to 0-1 range
    let max_diff = (a.len() + b.len()) as f32;
    if max_diff == 0.0 {
        return 0.0;
    }
    1.0 - (diff_sum as f32 / max_diff).min(1.0)
}

impl Backend {
    pub fn new(client: Client, config: Config) -> Self {
        Self {
            client,
            root: RwLock::new(".".into()),
            stats: RwLock::new(Stats::new()),
            config: RwLock::new(config),
            doc_state: Mutex::new(HashMap::new()),
            pending_changes: RwLock::new(HashMap::new()),
            dict_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Load a specific file's dictionary
    async fn load_file_dictionary(&self, uri: &Uri) -> anyhow::Result<MutableDictionary> {
        // VS Code's unsaved documents have "untitled" scheme
        if uri
            .scheme()
            .is_some_and(|scheme| scheme.eq_lowercase("untitled"))
        {
            return Ok(MutableDictionary::new());
        }

        let path = self
            .get_file_dict_path(uri)
            .await
            .context("Unable to get the file path.")?;

        load_dict(path, self.config.read().await.dialect)
            .await
            .map_err(|err| info!("{err}"))
            .or(Ok(MutableDictionary::new()))
    }

    /// Compute the location of the ignored lint's store.
    async fn get_ignored_lints_path(&self, uri: &Uri) -> anyhow::Result<PathBuf> {
        let config = self.config.read().await;

        Ok(config.ignored_lints_path.join(fileify_path(uri)?))
    }

    async fn save_ignored_lints(&self, uri: &Uri, ignored_lints: &IgnoredLints) -> Result<()> {
        save_ignored_lints(
            self.get_ignored_lints_path(uri)
                .await
                .context("Unable to get ignored lints path.")?,
            ignored_lints,
        )
        .await
        .context("Unable to save ignored lints to path.")
    }

    async fn load_ignored_lints(&self, uri: &Uri) -> Result<IgnoredLints> {
        // VS Code's unsaved documents have "untitled" scheme
        if uri
            .scheme()
            .is_some_and(|scheme| scheme.eq_lowercase("untitled"))
        {
            return Ok(IgnoredLints::new());
        }

        Ok(load_ignored_lints(
            self.get_ignored_lints_path(uri)
                .await
                .context("Unable to get ignored lints path.")?,
        )
        .await
        .map_err(|err| info!("{err}"))
        .unwrap_or(IgnoredLints::new()))
    }

    /// Compute the location of the file's specific dictionary
    async fn get_file_dict_path(&self, uri: &Uri) -> anyhow::Result<PathBuf> {
        let config = self.config.read().await;

        Ok(config.file_dict_path.join(fileify_path(uri)?))
    }

    async fn save_file_dictionary(&self, uri: &Uri, dict: impl Dictionary) -> Result<()> {
        save_dict(
            self.get_file_dict_path(uri)
                .await
                .context("Unable to get the file path.")?,
            dict,
        )
        .await
        .context("Unable to save the dictionary to path.")
    }

    async fn load_user_dictionary(&self, ) -> MutableDictionary {
        let config = self.config.read().await;

        load_dict(&config.user_dict_path, self.config.read().await.dialect)
            .await
            .map_err(|err| info!("{err}"))
            .unwrap_or(MutableDictionary::new())
    }

    async fn save_user_dictionary(&self, dict: impl Dictionary) -> Result<()> {
        let config = self.config.read().await;

        save_dict(&config.user_dict_path, dict)
            .await
            .map_err(|err| anyhow!("Unable to save the dictionary to file: {err}"))
    }

    async fn load_workspace_dictionary(&self) -> MutableDictionary {
        let config = self.config.read().await;
        load_dict(
            &config.workspace_dict_path,
            self.config.read().await.dialect,
        )
        .await
        .map_err(|err| info!("{err}"))
        .unwrap_or(MutableDictionary::new())
    }

    async fn save_workspace_dictionary(&self, dict: impl Dictionary) -> Result<()> {
        let config = self.config.read().await;
        save_dict(&config.workspace_dict_path, dict)
            .await
            .map_err(|err| anyhow!("Unable to save the dictionary to file: {err}"))
    }

    async fn save_stats(&self) -> Result<()> {
        let (config, stats) = join(self.config.read(), self.stats.read()).await;

        if let Some(parent) = config.stats_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let mut writer = BufWriter::new(
            OpenOptions::new()
                .read(true)
                .append(true)
                .create(true)
                .open(&config.stats_path)?,
        );
        stats.write(&mut writer)?;
        writer.flush()?;

        Ok(())
    }

    /// Get modification time of a file, returning None if file doesn't exist or on error
    async fn get_file_mtime(&self, path: PathBuf) -> Option<SystemTime> {
        tokio::fs::metadata(path)
            .await
            .ok()
            .and_then(|m| m.modified().ok())
    }

    async fn generate_global_dictionary(&self) -> Result<MergedDictionary> {
        let mut dict = MergedDictionary::new();
        dict.add_dictionary(FstDictionary::curated());
        let user_dict = self.load_user_dictionary().await;
        dict.add_dictionary(Arc::new(user_dict));
        let ws_dict = self.load_workspace_dictionary().await;
        dict.add_dictionary(Arc::new(ws_dict));
        Ok(dict)
    }

    async fn generate_file_dictionary(&self, uri: &Uri) -> Result<MergedDictionary> {
        // Check cache first
        let cache_entry = {
            let cache = self.dict_cache.read().await;
            cache.get(uri).cloned()
        };

        // Get dictionary paths from config
        let (user_dict_path, workspace_dict_path) = {
            let config = self.config.read().await;
            (config.user_dict_path.clone(), config.workspace_dict_path.clone())
        };
        let file_dict_path = self.get_file_dict_path(uri).await.ok();

        // Get current modification times
        let current_user_mtime = self.get_file_mtime(user_dict_path).await;
        let current_workspace_mtime = self.get_file_mtime(workspace_dict_path).await;
        let current_file_mtime = if let Some(path) = file_dict_path {
            self.get_file_mtime(path).await
        } else {
            None
        };

        // Check if cache is still valid
        if let Some(entry) = cache_entry {
            if entry.user_dict_mtime == current_user_mtime
                && entry.workspace_dict_mtime == current_workspace_mtime
                && entry.file_dict_mtime == current_file_mtime
            {
                // Cache hit! Return cached dictionary
                return Ok((*entry.dict).clone());
            }
        }

        // Cache miss or invalidated - reload dictionaries
        let (global_dictionary, file_dictionary) = tokio::join!(
            self.generate_global_dictionary(),
            self.load_file_dictionary(uri)
        );

        let mut global_dictionary =
            global_dictionary.context("Unable to load the user dictionary.")?;
        global_dictionary.add_dictionary(Arc::new(
            file_dictionary.context("Unable to load the file dictionary.")?,
        ));

        // Update cache
        let cache_entry = DictCacheEntry {
            dict: Arc::new(global_dictionary.clone()),
            user_dict_mtime: current_user_mtime,
            workspace_dict_mtime: current_workspace_mtime,
            file_dict_mtime: current_file_mtime,
        };

        {
            let mut cache = self.dict_cache.write().await;
            cache.insert(uri.clone(), cache_entry);
        }

        Ok(global_dictionary)
    }

    async fn update_document_from_file(&self, uri: &Uri, language_id: Option<&str>) -> Result<()> {
        let content = tokio::fs::read_to_string(
            uri.to_file_path()
                .ok_or_else(|| anyhow!("Unable to convert URL to file path."))?,
        )
        .await
        .with_context(|| format!("Unable to read from file {uri:?}"))?;

        self.update_document(uri, &content, language_id).await
    }

    async fn update_document(
        &self,
        uri: &Uri,
        text: &str,
        language_id: Option<&str>,
    ) -> Result<()> {
        self.pull_config().await;

        // Copy necessary configuration to avoid holding lock.
        let (
            lint_config,
            markdown_options,
            isolate_english,
            dialect,
            max_file_length,
            exclude_patterns,
        ) = {
            let config = self.config.read().await;
            (
                config.lint_config.clone(),
                config.markdown_options,
                config.isolate_english,
                config.dialect,
                config.max_file_length,
                config.exclude_patterns.clone(),
            )
        };

        let mut doc_lock = self.doc_state.lock().await;

        // Check exclude patterns - but only for URIs that can be converted to file paths
        // Unsaved files might have URIs that don't map to real paths
        if !exclude_patterns.is_empty() {
            match uri.to_file_path() {
                Some(path) if exclude_patterns.is_match(&path) => {
                    eprintln!("HARPER: Excluding file due to pattern match: {:?}", uri);
                    doc_lock.remove(uri);
                    return Ok(());
                }
                None => {
                    eprintln!("HARPER: URI has no file path (unsaved file?): {:?}", uri);
                    // Continue processing - don't exclude unsaved files
                }
                _ => {}
            }
        }

        let ignored_lints = self.load_ignored_lints(uri).await.unwrap_or_default();

        let dict = Arc::new(
            self.generate_file_dictionary(uri)
                .await
                .context("Unable to generate the file dictionary.")?,
        );

        let doc_state = doc_lock.entry(uri.clone()).or_insert_with(|| {
            info!("Constructing new LintGroup for new document.");

            DocumentState {
                ignored_lints,
                linter: LintGroup::new_curated(dict.clone(), dialect)
                    .with_lint_config(lint_config.clone()),
                language_id: language_id.map(|v| v.to_string()),
                dict: dict.clone(),
                uri: uri.clone(),
                ..Default::default()
            }
        });

        if doc_state.dict != dict {
            doc_state.dict = dict.clone();
            info!("Constructing new linter because of modified dictionary.");
            doc_state.linter =
                LintGroup::new_curated(dict.clone(), dialect).with_lint_config(lint_config.clone());
        }

        let Some(language_id) = &doc_state.language_id else {
            eprintln!("HARPER: No language_id for document, removing: {:?}", uri);
            doc_lock.remove(uri);
            return Ok(())
        };

        async fn use_ident_dict<'a>(
            backend: &'a Backend,
            new_dict: Arc<MutableDictionary>,
            parser: impl Parser + 'static,
            uri: &'a Uri,
            doc_state: &'a mut DocumentState,
            lint_config: &LintGroupConfig,
            dialect: Dialect,
        ) -> Result<Box<dyn Parser>> {
            if doc_state.ident_dict != new_dict {
                info!("Constructing new linter because of modified ident dictionary.");
                doc_state.ident_dict = new_dict.clone();

                let mut merged = backend.generate_file_dictionary(uri).await?;
                merged.add_dictionary(new_dict);
                let merged = Arc::new(merged);

                doc_state.linter = LintGroup::new_curated(merged.clone(), dialect)
                    .with_lint_config(lint_config.clone());
                doc_state.dict = merged.clone();
            }

            Ok(Box::new(CollapseIdentifiers::new(
                Box::new(parser),
                Box::new(doc_state.dict.clone()),
            )))
        }

        let source: Vec<char> = text.chars().collect();
        let ts_parser = CommentParser::new_from_language_id(language_id, markdown_options);
        let parser: Option<Box<dyn Parser>> = match language_id.as_str() {
            _ if ts_parser.is_some() => {
                let ts_parser = ts_parser.unwrap();

                if let Some(new_dict) = ts_parser.create_ident_dict(&Arc::new(source)) {
                    Some(
                        use_ident_dict(
                            self,
                            Arc::new(new_dict),
                            ts_parser,
                            uri,
                            doc_state,
                            &lint_config,
                            dialect,
                        )
                        .await?,
                    )
                } else {
                    Some(Box::new(ts_parser))
                }
            }
            "git-commit" | "gitcommit" => {
                Some(Box::new(GitCommitParser::new_markdown(markdown_options)))
            }
            "html" => Some(Box::new(HtmlParser::default())),
            "ink" => Some(Box::new(InkParser::default())),
            "jj-commit" | "jjdescription" => {
                Some(Box::new(JJDescriptionParser::new(markdown_options)))
            }
            "lhaskell" | "literate haskell" => {
                let parser = LiterateHaskellParser::new_markdown(markdown_options);

                if let Some(new_dict) =
                    parser.create_ident_dict(&Arc::new(source), markdown_options)
                {
                    Some(
                        use_ident_dict(
                            self,
                            Arc::new(new_dict),
                            parser,
                            uri,
                            doc_state,
                            &lint_config,
                            dialect,
                        )
                        .await?,
                    )
                } else {
                    Some(Box::new(parser))
                }
            }
            "mail" => Some(Box::new(PlainEnglish)),
            "markdown" => Some(Box::new(Markdown::new(markdown_options))),
            "org" => Some(Box::new(OrgMode)),
            "plaintext" | "text" => Some(Box::new(PlainEnglish)),
            "python" => Some(Box::new(PythonParser::default())),
            "typst" => Some(Box::new(Typst)),
            _ => None,
        };

        match parser {
            None => {
                doc_lock.remove(uri);
            }
            Some(mut parser) => {
                if isolate_english {
                    parser = Box::new(IsolateEnglish::new(parser, doc_state.dict.clone()));
                }

                // Don't lint on documents larger than the configured maximum length.
                if text.len() <= max_file_length {
                    doc_state.document = Document::new(text, &parser, &doc_state.dict);
                } else {
                    // Ensures that existing lints are cleared when we stop linting the file.
                    // Otherwise, prior lints will remain, and they will quickly fall out of sync
                    // with the document when it is edited.
                    doc_state.document = Document::default();
                }

                // Build line index for O(1) position conversions
                let source: Vec<char> = doc_state.document.get_source().iter().copied().collect();
                doc_state.line_index = crate::pos_conv::LineIndex::new(&source);
            }
        }

        Ok(())
    }

    async fn generate_code_actions(
        &self,
        uri: &Uri,
        range: Range,
    ) -> JsonResult<Vec<CodeActionOrCommand>> {
        let (config, mut doc_states) = tokio::join!(self.config.read(), self.doc_state.lock());
        let Some(doc_state) = doc_states.get_mut(uri) else {
            return Ok(Vec::new());
        };

        Ok(doc_state.generate_code_actions(range, &config.code_action_config))
    }

    async fn generate_diagnostics(&self, uri: &Uri) -> Vec<Diagnostic> {
        // Copy necessary configuration to avoid holding lock.
        let diagnostic_severity = {
            let config = self.config.read().await;
            config.diagnostic_severity
        };

        let mut doc_states = self.doc_state.lock().await;
        let Some(doc_state) = doc_states.get_mut(uri) else {
            return Vec::new();
        };

        doc_state.generate_diagnostics(diagnostic_severity)
    }

    async fn publish_diagnostics(&self, uri: &Uri) {
        let diagnostics = self.generate_diagnostics(uri).await;

        // Check if diagnostics have changed compared to last publish
        let should_publish = {
            let mut doc_states = self.doc_state.lock().await;
            if let Some(doc_state) = doc_states.get_mut(uri) {
                // Compare with last published diagnostics
                if diagnostics == doc_state.last_diagnostics {
                    // Identical diagnostics - skip publishing
                    false
                } else {
                    // Different diagnostics - update cache and publish
                    doc_state.last_diagnostics = diagnostics.clone();
                    true
                }
            } else {
                // Document not found - publish anyway
                true
            }
        };

        if !should_publish {
            // Skip publishing identical diagnostics
            return;
        }

        let result = PublishDiagnosticsParams {
            uri: uri.clone(),
            diagnostics,
            version: None,
        };

        self.client
            .send_notification::<PublishDiagnostics>(result)
            .await;
    }

    /// Generate completion suggestions based on the current cursor position
    async fn generate_completions(
        &self,
        uri: &Uri,
        position: tower_lsp_server::lsp_types::Position,
    ) -> JsonResult<Vec<CompletionItem>> {
        // Check if completion is enabled
        let completion_config = {
            let config = self.config.read().await;
            config.completion_config.clone()
        };

        if !completion_config.enabled {
            return Ok(Vec::new());
        }

        // Copy needed data while holding lock, then release it before expensive operations
        // This follows the pattern from commits 6760b477 and e4251ac2
        // Also get lints to filter out misspelled words from completions
        let (source, dict, line_index, misspelled_words) = {
            let doc_states = self.doc_state.lock().await;
            eprintln!("  Document states count: {}, looking for uri: {:?}", doc_states.len(), uri);
            let Some(doc_state) = doc_states.get(uri) else {
                eprintln!("  ERROR: Document state not found for uri!");
                return Ok(Vec::new());
            };

            // Extract misspelled words from diagnostics
            // These are words that currently have spelling errors in the document
            use tower_lsp_server::lsp_types::NumberOrString;
            let doc_source = doc_state.document.get_source();
            let misspelled: std::collections::HashSet<String> = doc_state
                .last_diagnostics
                .iter()
                .filter(|diag| {
                    diag.code
                        .as_ref()
                        .map_or(false, |code| match code {
                            NumberOrString::String(s) => s.contains("Spelling"),
                            _ => false,
                        })
                })
                .filter_map(|diag| {
                    let span = crate::pos_conv::range_to_span(doc_source, diag.range);
                    let word: String = span.get_content(doc_source).iter().collect();
                    Some(word.to_lowercase())
                })
                .collect();

            (
                doc_source.iter().copied().collect::<Vec<char>>(),
                doc_state.dict.clone(),
                doc_state.line_index.clone(),
                misspelled,
            )
        }; // Lock released here

        // Convert LSP position to character index using line index (O(1) instead of O(N))
        let cursor_index = line_index.position_to_index(&source, position);

        // Find the word being typed by looking backwards from cursor
        let word_start = source[..cursor_index]
            .iter()
            .rposition(|c| !c.is_alphanumeric() && *c != '\'' && *c != '-')
            .map(|i| i + 1)
            .unwrap_or(0);

        // Extract the prefix being typed
        let prefix: Vec<char> = source[word_start..cursor_index].to_vec();

        // Log what we're completing with context
        use tracing::info;
        let prefix_str: String = prefix.iter().collect();
        let context_before: String = source[word_start.saturating_sub(10)..word_start].iter().collect();
        let context_after: String = source[cursor_index..cursor_index.saturating_add(10).min(source.len())].iter().collect();

        info!("Completion request: prefix='{}', len={}", prefix_str, prefix.len());
        eprintln!("  Completing prefix: '{}' (len={})", prefix_str, prefix.len());
        eprintln!("  Context: ...{:?}[{}]{:?}...", context_before, prefix_str, context_after);
        eprintln!("  Position: cursor_index={}, word_start={}, source_len={}", cursor_index, word_start, source.len());

        // Don't show completions for very short prefixes
        if prefix.len() < completion_config.min_prefix_length {
            info!("Prefix too short, skipping");
            eprintln!("  Prefix too short (min={})", completion_config.min_prefix_length);
            return Ok(Vec::new());
        }

        // Perform expensive fuzzy matching and scoring WITHOUT holding the lock
        // This is the same logic as generate_completion_list but without needing DocumentState
        // Use adaptive edit distance based on prefix length for better matching on longer words
        // Very short (2 chars): distance 1 - very strict to avoid noise
        // Short words (3-5 chars): distance 2 - allow common typos like doubled letters (tthi→this)
        // Medium words (6-9 chars): distance 3 - balanced
        // Long words (10+ chars): distance 4 - accommodate multiple typos
        let max_edit_distance = match prefix.len() {
            0..=2 => 1,
            3..=5 => 2,
            6..=9 => 3,
            _ => 4,
        };
        eprintln!("  Using edit distance {} for prefix length {}", max_edit_distance, prefix.len());
        let fuzzy_completions = dict.fuzzy_match(&prefix, max_edit_distance, 200);
        eprintln!("  Fuzzy match returned {} candidates", fuzzy_completions.len());

        // Helper function to check if word is a simple transposition of prefix
        let is_transposition = |word: &[char]| -> bool {
            if word.len() != prefix.len() {
                return false;
            }
            let mut diff_positions = Vec::new();
            for (i, (p, w)) in prefix.iter().zip(word.iter()).enumerate() {
                if !p.eq_ignore_ascii_case(w) {
                    diff_positions.push(i);
                    if diff_positions.len() > 2 {
                        return false;
                    }
                }
            }
            if diff_positions.len() == 2 {
                let i = diff_positions[0];
                let j = diff_positions[1];
                if j == i + 1 {
                    return prefix[i].eq_ignore_ascii_case(&word[j])
                        && prefix[j].eq_ignore_ascii_case(&word[i]);
                }
            }
            false
        };

        eprintln!("  Misspelled words in document: {:?}", misspelled_words.iter().take(10).collect::<Vec<_>>());

        let mut completions: Vec<(String, f32)> = fuzzy_completions
            .into_iter()
            .filter_map(|fuzzy_match| {
                let word_string: String = fuzzy_match.word.iter().collect();

                // Filter out words that are currently marked as misspelled in the document
                // This prevents suggesting "tthis" when the user typed it earlier and it has an error
                if misspelled_words.contains(&word_string.to_lowercase()) {
                    eprintln!("    FILTERED OUT: '{}' (marked as misspelled)", word_string);
                    return None;
                }

                let is_common = fuzzy_match.metadata.common;
                let is_trans = is_transposition(fuzzy_match.word);

                let avg_kbd_dist = avg_keyboard_distance(&prefix, fuzzy_match.word);

                let score = calculate_completion_score(
                    &prefix,
                    fuzzy_match.word,
                    fuzzy_match.edit_distance,
                    is_common,
                    is_trans,
                    avg_kbd_dist,
                );
                Some((word_string, score))
            })
            .collect();

        completions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        info!("Returning {} completions for '{}'", completions.len(), prefix_str);
        eprintln!("  Found {} fuzzy matches, returning top {} completions",
            completions.len(), completion_config.max_results.min(completions.len()));
        if !completions.is_empty() {
            eprintln!("  Top 5: {:?}",
                completions.iter().take(5).map(|(w, s)| format!("{}({:.1})", w, s)).collect::<Vec<_>>());
        }

        // Calculate the start position of the word being completed
        let word_start_position = line_index.index_to_position(&source, word_start);

        // Convert prefix to string - we'll use this as filterText
        // This tells Helix that all our completions match the user's input
        let prefix_string: String = prefix.iter().collect();

        eprintln!("  text_edit range: start={}:{}, end={}:{}, prefix='{}'",
            word_start_position.line, word_start_position.character,
            position.line, position.character,
            prefix_string);

        // Helper function to apply smart casing based on user's input pattern
        // Preserves dictionary word casing while respecting user's capitalization intent
        let apply_prefix_casing = |prefix: &[char], word: &str| -> String {
            if prefix.is_empty() {
                return word.to_string();
            }

            let word_chars: Vec<char> = word.chars().collect();

            // Check the casing pattern of the prefix
            let first_is_upper = prefix[0].is_uppercase();
            let all_upper = prefix.iter().all(|c| !c.is_alphabetic() || c.is_uppercase());

            // If all typed characters are uppercase, return all uppercase
            if all_upper && prefix.iter().any(|c| c.is_alphabetic()) {
                return word.to_uppercase();
            }

            // If only first character is uppercase, capitalize first letter of word
            if first_is_upper {
                let mut result = String::new();
                for (i, ch) in word_chars.iter().enumerate() {
                    if i == 0 {
                        result.push(ch.to_uppercase().next().unwrap_or(*ch));
                    } else {
                        result.push(*ch);
                    }
                }
                return result;
            }

            // Otherwise (all lowercase or mixed), keep original word casing
            word.to_string()
        };

        // Convert to LSP completion items
        // Use text_edit to specify exact replacement range - this tells Helix what to replace
        // Set filter_text to prefix so items aren't filtered out while typing
        let completion_items: Vec<CompletionItem> = completions
            .into_iter()
            .take(completion_config.max_results)
            .enumerate()
            .map(|(idx, (word_string, _))| {
                use tower_lsp_server::lsp_types::{Range, TextEdit, CompletionTextEdit};

                // Apply the casing from the user's prefix to the completion
                let completion_text = apply_prefix_casing(&prefix, &word_string);

                CompletionItem {
                    label: word_string.clone(),
                    kind: Some(CompletionItemKind::TEXT),
                    detail: Some("Harper".to_string()),
                    // text_edit specifies the exact range to replace
                    text_edit: Some(CompletionTextEdit::Edit(TextEdit {
                        range: Range {
                            start: word_start_position,
                            end: position,
                        },
                        new_text: completion_text,
                    })),
                    // filter_text must match what user typed for Helix to show the item
                    filter_text: Some(prefix_string.clone()),
                    sort_text: Some(format!("{:05}", idx)),
                    ..Default::default()
                }
            })
            .collect();

        Ok(completion_items)
    }

    /// Update the configuration of the server and publish document updates that
    /// match it.
    async fn update_config_from_obj(&self, json_obj: Value) {
        if let Ok(new_config) = Config::from_lsp_config(&self.root.read().await, json_obj)
            .map_err(|err| error!("{err}"))
        {
            let mut config = self.config.write().await;
            *config = new_config;
        }
    }

    async fn pull_config(&self) {
        let mut new_config = self
            .client
            .configuration(vec![ConfigurationItem {
                scope_uri: None,
                section: None,
            }])
            .await
            .unwrap();

        if let Some(first) = new_config.pop() {
            self.update_config_from_obj(first).await;
        }
    }
}

impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> JsonResult<InitializeResult> {
        if let Some(root) = params
            .workspace_folders
            .as_ref()
            // We take the first workspace folder
            .and_then(|v| v.first())
            .map(|f| &f.uri)
            // Or failing that, the root_uri (which is deprecated in favour of workspace_folders)
            .or(
                #[allow(deprecated)]
                params.root_uri.as_ref(),
            )
            .and_then(|u| u.to_file_path().map(PathBuf::from))
            // Or failing that, the root_path (which is deprecated in favour of root_uri)
            .or(
                #[allow(deprecated)]
                params.root_path.as_deref().map(PathBuf::from),
            )
        {
            // Save the workspace root away for use during the configuration step
            *self.root.write().await = root;
        }

        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "harper-ls".to_owned(),
                version: Some(ls_version().to_owned()),
            }),
            capabilities: ServerCapabilities {
                code_action_provider: Some(CodeActionProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    // Trigger on all letters so completions appear as user types
                    trigger_characters: Some(
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            .chars()
                            .map(String::from)
                            .collect(),
                    ),
                    all_commit_characters: None,
                    work_done_progress_options: Default::default(),
                    completion_item: None,
                }),
                execute_command_provider: Some(ExecuteCommandOptions {
                    commands: vec![
                        "HarperRecordLint".to_owned(),
                        "HarperAddToUserDict".to_owned(),
                        "HarperAddToWSDict".to_owned(),
                        "HarperAddToFileDict".to_owned(),
                        "HarperOpen".to_owned(),
                        "HarperIgnoreLint".to_owned(),
                    ],
                    ..Default::default()
                }),
                text_document_sync: Some(TextDocumentSyncCapability::Options(
                    TextDocumentSyncOptions {
                        open_close: Some(true),
                        change: Some(TextDocumentSyncKind::FULL),
                        will_save: None,
                        will_save_wait_until: None,
                        save: Some(TextDocumentSyncSaveOptions::Supported(true)),
                    },
                )),
                ..Default::default()
            },
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Server initialized!")
            .await;

        self.pull_config().await;

        let did_change_watched_files = Registration {
            id: "workspace/didChangeWatchedFiles".to_owned(),
            method: "workspace/didChangeWatchedFiles".to_owned(),
            register_options: Some(
                serde_json::to_value(DidChangeWatchedFilesRegistrationOptions {
                    watchers: vec![FileSystemWatcher {
                        glob_pattern: GlobPattern::String("**/*".to_owned()),
                        kind: Some(WatchKind::Delete),
                    }],
                })
                .unwrap(),
            ),
        };
        if let Err(err) = self
            .client
            .register_capability(vec![did_change_watched_files])
            .await
        {
            warn!("Unable to register watch file capability: {}", err);
        }
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.update_document(
            &params.text_document.uri,
            &params.text_document.text,
            Some(&params.text_document.language_id),
        )
        .await
        .map_err(|err| error!("{err}"))
        .err();

        self.publish_diagnostics(&params.text_document.uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let change_start = Instant::now();
        let Some(last) = params.content_changes.last() else {
            return;
        };

        let uri = params.text_document.uri.clone();
        let text = last.text.clone();

        eprintln!("HARPER DID_CHANGE: version={:?}, text_len={}, last_40_chars={:?}",
            params.text_document.version,
            text.len(),
            text.chars().rev().take(40).collect::<Vec<_>>().iter().rev().collect::<String>());

        // IMPORTANT: Update document content immediately (without debounce)
        // This ensures completions have access to the latest text while typing
        if let Err(err) = self.update_document(&uri, &text, None).await {
            error!("{err}")
        }

        let change_elapsed = change_start.elapsed();
        eprintln!("HARPER DID_CHANGE COMPLETE: version={:?}, took {:?}", params.text_document.version, change_elapsed);

        // Record this change with current timestamp for debounced diagnostics
        let now = Instant::now();
        {
            let mut pending = self.pending_changes.write().await;
            pending.insert(uri.clone(), now);
        }

        // Debounce: wait 300ms before publishing diagnostics
        // This prevents excessive diagnostic updates while typing
        sleep(Duration::from_millis(300)).await;

        // Check if this is still the latest change for this URI
        let should_process = {
            let pending = self.pending_changes.read().await;
            pending.get(&uri).map_or(false, |&timestamp| timestamp == now)
        };

        if !should_process {
            // A newer change came in, skip publishing diagnostics
            return;
        }

        // Publish diagnostics (debounced)
        self.publish_diagnostics(&uri).await;

        // Clear from pending
        {
            let mut pending = self.pending_changes.write().await;
            pending.remove(&uri);
        }
    }

    async fn did_close(&self, _params: DidCloseTextDocumentParams) {
        let uri = _params.text_document.uri;
        let mut doc_lock = self.doc_state.lock().await;
        doc_lock.remove(&uri);

        self.client
            .send_notification::<PublishDiagnostics>(PublishDiagnosticsParams {
                uri: uri.clone(),
                diagnostics: vec![],
                version: None,
            })
            .await;
    }

    async fn did_change_watched_files(&self, params: DidChangeWatchedFilesParams) {
        let mut doc_lock = self.doc_state.lock().await;
        let mut uris_to_clear = Vec::new();

        for change in &params.changes {
            if change.typ != FileChangeType::DELETED {
                continue;
            }

            doc_lock.retain(|uri, _| {
                // `change.uri` could be a directory so use `starts_with` instead of `==`.
                let to_remove = uri.as_str().starts_with(change.uri.as_str());

                if to_remove {
                    uris_to_clear.push(uri.clone());
                }

                !to_remove
            });
        }

        for uri in &uris_to_clear {
            self.client
                .send_notification::<PublishDiagnostics>(PublishDiagnosticsParams {
                    uri: uri.clone(),
                    diagnostics: vec![],
                    version: None,
                })
                .await;
        }
    }

    async fn execute_command(&self, params: ExecuteCommandParams) -> JsonResult<Option<Value>> {
        let mut string_args = params
            .arguments
            .iter()
            .map(|v| serde_json::from_value::<String>(v.clone()).unwrap());

        let Some(first) = string_args.next() else {
            return Ok(None);
        };

        info!("Received command: \"{}\"", params.command.as_str());

        match params.command.as_str() {
            "HarperRecordLint" => {
                let Ok(kind) = serde_json::from_str(&first) else {
                    error!("Unable to deserialize RecordKind.");
                    return Ok(None);
                };

                let record = Record::now(kind);

                let mut stats = self.stats.write().await;
                stats.records.push(record);
            }
            "HarperAddToUserDict" => {
                let word = &first.chars().collect::<Vec<_>>();

                let Some(second) = string_args.next() else {
                    return Ok(None);
                };

                let file_uri = second.parse().unwrap();

                let mut dict = self.load_user_dictionary().await;
                dict.append_word(word, DictWordMetadata::default());
                self.save_user_dictionary(dict)
                    .await
                    .map_err(|err| error!("{err}"))
                    .err();
                self.update_document_from_file(&file_uri, None)
                    .await
                    .map_err(|err| error!("{err}"))
                    .err();
                self.publish_diagnostics(&file_uri).await;
            }
            "HarperAddToWSDict" => {
                let word = &first.chars().collect::<Vec<_>>();

                let Some(second) = string_args.next() else {
                    return Ok(None);
                };

                let file_uri = second.parse().unwrap();

                let mut dict = self.load_workspace_dictionary().await;
                dict.append_word(word, DictWordMetadata::default());
                self.save_workspace_dictionary(dict)
                    .await
                    .map_err(|err| error!("{err}"))
                    .err();
                self.update_document_from_file(&file_uri, None)
                    .await
                    .map_err(|err| error!("{err}"))
                    .err();
                self.publish_diagnostics(&file_uri).await;
            }
            "HarperAddToFileDict" => {
                let word = &first.chars().collect::<Vec<_>>();

                let Some(second) = string_args.next() else {
                    return Ok(None);
                };

                let file_uri = second.parse().unwrap();

                let mut dict = match self
                    .load_file_dictionary(&file_uri)
                    .await
                    .map_err(|err| error!("{err}"))
                {
                    Ok(dict) => dict,
                    Err(_) => {
                        return Ok(None);
                    }
                };
                dict.append_word(word, DictWordMetadata::default());

                self.save_file_dictionary(&file_uri, dict)
                    .await
                    .map_err(|err| error!("{err}"))
                    .err();
                self.update_document_from_file(&file_uri, None)
                    .await
                    .map_err(|err| error!("{err}"))
                    .err();
                self.publish_diagnostics(&file_uri).await;
            }
            "HarperOpen" => match open::that(&first) {
                Ok(()) => {
                    let message = format!(r#"Opened \"{{first}}\""#);

                    self.client.log_message(MessageType::INFO, &message).await;

                    info!("{}", message);
                }
                Err(err) => {
                    self.client
                        .log_message(MessageType::ERROR, "Unable to open URL")
                        .await;
                    error!("Unable to open URL: {}", err);
                }
            },
            "HarperIgnoreLint" => {
                let Ok(uri) = first.parse() else {
                    error!("Unable to parse URL from command: {first}");
                    return Ok(None);
                };

                let Some(second) = params.arguments.into_iter().nth(1) else {
                    error!("Not enough arguments to HarperIgnoreLint");
                    return Ok(None);
                };

                let Ok(lint) = serde_json::from_value(second) else {
                    error!("Unable to parse lint.");
                    return Ok(None);
                };

                let mut doc_lock = self.doc_state.lock().await;
                let Some(doc_state) = doc_lock.get_mut(&uri) else {
                    error!("Requested document has not been loaded.");
                    return Ok(None);
                };

                doc_state.ignore_lint(&lint);
                if let Err(_err) = self
                    .save_ignored_lints(&uri, &doc_state.ignored_lints)
                    .await
                {
                    error!("Unable to save ignored lints.");
                    return Ok(None);
                }

                drop(doc_lock);

                self.publish_diagnostics(&uri).await;
            }
            _ => (),
        }

        Ok(None)
    }

    async fn did_change_configuration(&self, params: DidChangeConfigurationParams) {
        self.update_config_from_obj(params.settings).await;

        let uris: Vec<Uri> = {
            let mut doc_lock = self.doc_state.lock().await;
            let config_lock = self.config.read().await;

            for doc in doc_lock.values_mut() {
                info!("Constructing new LintGroup for updated configuration.");
                doc.linter = LintGroup::new_curated(doc.dict.clone(), config_lock.dialect)
                    .with_lint_config(config_lock.lint_config.clone());
            }

            doc_lock.keys().cloned().collect()
        };

        for uri in uris {
            self.update_document_from_file(&uri, None)
                .await
                .map_err(|err| error!("{err}"))
                .err();
            self.publish_diagnostics(&uri).await;
        }
    }

    async fn code_action(
        &self,
        params: CodeActionParams,
    ) -> JsonResult<Option<CodeActionResponse>> {
        let actions = self
            .generate_code_actions(&params.text_document.uri, params.range)
            .await?;

        Ok(Some(actions))
    }

    async fn completion(
        &self,
        params: CompletionParams,
    ) -> JsonResult<Option<CompletionResponse>> {
        let start_time = Instant::now();
        eprintln!("\n========== HARPER COMPLETION START ==========");
        eprintln!("HARPER COMPLETION CALLED: pos={:?}, trigger={:?}",
            params.text_document_position.position,
            params.context.as_ref().map(|c| c.trigger_kind));

        // Check if cursor position might be ahead of document content
        // This happens when Helix sends completion before didChange
        // We need to check BEFORE position_to_index clamps the values
        let (needs_wait, debug_info) = {
            let doc_states = self.doc_state.lock().await;
            if let Some(doc_state) = doc_states.get(&params.text_document_position.text_document.uri) {
                let source = doc_state.document.get_source();
                let position = params.text_document_position.position;

                // Check if the requested position is beyond what the document currently contains
                let out_of_bounds = doc_state.line_index.is_position_out_of_bounds(source, position);

                let info = format!("pos={}:{}, source_len={}, out_of_bounds={}",
                    position.line, position.character, source.len(), out_of_bounds);
                (out_of_bounds, info)
            } else {
                (false, "no_doc_state".to_string())
            }
        };

        eprintln!("HARPER RACE CHECK: {}, needs_wait={}", debug_info, needs_wait);

        // If we're racing with didChange, wait for the update to arrive
        // Use short waits (10ms) and retry up to 5 times (max 50ms total)
        // This prevents timeouts in editors like Helix that have short completion timeouts
        if needs_wait {
            use tokio::time::{sleep, Duration};
            for attempt in 0..5 {
                sleep(Duration::from_millis(10)).await;

                // Check if document has caught up
                let doc_states = self.doc_state.lock().await;
                if let Some(doc_state) = doc_states.get(&params.text_document_position.text_document.uri) {
                    let source = doc_state.document.get_source();
                    let position = params.text_document_position.position;

                    // Check if position is still out of bounds
                    let still_out_of_bounds = doc_state.line_index.is_position_out_of_bounds(source, position);

                    if !still_out_of_bounds {
                        eprintln!("HARPER RACE RESOLVED after {}ms", (attempt + 1) * 10);
                        break;
                    }
                }
            }
        }

        let completions = self
            .generate_completions(
                &params.text_document_position.text_document.uri,
                params.text_document_position.position,
            )
            .await?;

        let elapsed = start_time.elapsed();
        eprintln!("HARPER COMPLETION RESULT: {} items", completions.len());
        if !completions.is_empty() {
            eprintln!("  First 10 items: {:?}",
                completions.iter().take(10).map(|c| &c.label).collect::<Vec<_>>());
        }
        eprintln!("========== HARPER COMPLETION END (took {:?}) ==========\n", elapsed);

        if completions.is_empty() {
            eprintln!("HARPER RETURNING: None (empty)");
            Ok(None)
        } else {
            use tower_lsp_server::lsp_types::CompletionList;
            eprintln!("HARPER RETURNING: List with {} items, is_incomplete=true", completions.len());
            // Set is_incomplete=true to prevent Helix from doing its own client-side filtering
            // We've already done fuzzy matching, so Helix should show our results as-is
            Ok(Some(CompletionResponse::List(CompletionList {
                is_incomplete: true,
                items: completions,
            })))
        }
    }

    async fn shutdown(&self) -> JsonResult<()> {
        let doc_states = self.doc_state.lock().await;

        // Clears the diagnostics for open buffers.
        for uri in doc_states.keys() {
            let result = PublishDiagnosticsParams {
                uri: uri.clone(),
                diagnostics: vec![],
                version: None,
            };

            self.client
                .send_notification::<PublishDiagnostics>(result)
                .await;
        }

        if self.save_stats().await.is_err() {
            error!("Unable to save stats.")
        }

        Ok(())
    }
}

#[cfg(test)]
mod completion_scoring_tests {
    use super::*;

    /// Helper to convert string to Vec<char> for testing
    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn test_ths_completion_ordering() {
        // When typing "ths", "the" and "this" should rank higher than "tab", "tag"
        let query = chars("ths");

        // "the" - edit distance 1, common word
        let score_the = calculate_completion_score(&query, &chars("the"), 1, true, false, 0.0);

        // "this" - edit distance 1, common word
        let score_this = calculate_completion_score(&query, &chars("this"), 1, true, false, 0.0);

        // "tab" - edit distance 2, not common
        let score_tab = calculate_completion_score(&query, &chars("tab"), 2, false, false, 0.0);

        // "tag" - edit distance 2, not common
        let score_tag = calculate_completion_score(&query, &chars("tag"), 2, false, false, 0.0);

        // "than" - edit distance 2, common word, better prefix match
        let score_than = calculate_completion_score(&query, &chars("than"), 2, true, false, 0.0);

        // Verify ordering: the/this > than > tab/tag
        assert!(
            score_the > score_tab,
            "Expected 'the' ({}) to rank higher than 'tab' ({})",
            score_the,
            score_tab
        );
        assert!(
            score_this > score_tab,
            "Expected 'this' ({}) to rank higher than 'tab' ({})",
            score_this,
            score_tab
        );
        assert!(
            score_the > score_tag,
            "Expected 'the' ({}) to rank higher than 'tag' ({})",
            score_the,
            score_tag
        );
        assert!(
            score_this > score_tag,
            "Expected 'this' ({}) to rank higher than 'tag' ({})",
            score_this,
            score_tag
        );

        // "than" has better prefix match than "tab"/"tag", so should rank higher
        assert!(
            score_than > score_tab,
            "Expected 'than' ({}) with 2-char prefix to rank higher than 'tab' ({})",
            score_than,
            score_tab
        );
        assert!(
            score_than > score_tag,
            "Expected 'than' ({}) with 2-char prefix to rank higher than 'tag' ({})",
            score_than,
            score_tag
        );
    }

    #[test]
    fn test_transposition_bonus() {
        // "teh" -> "the" should rank very high due to transposition
        let query = chars("teh");

        // "the" with transposition detected
        let score_transposition = calculate_completion_score(&query, &chars("the"), 1, true, true, 0.0);

        // "tea" without transposition (also edit distance 1)
        let score_no_transposition =
            calculate_completion_score(&query, &chars("tea"), 1, true, false, 0.0);

        assert!(
            score_transposition > score_no_transposition,
            "Expected transposition 'teh'->'the' ({}) to rank higher than 'teh'->'tea' ({})",
            score_transposition,
            score_no_transposition
        );
    }

    #[test]
    fn test_prefix_match_bonus() {
        // Longer prefix matches should score higher
        let query = chars("hel");

        // "hello" - 3 char prefix match
        let score_hello = calculate_completion_score(&query, &chars("hello"), 2, false, false, 0.0);

        // "help" - 3 char prefix match
        let score_help = calculate_completion_score(&query, &chars("help"), 1, false, false, 0.0);

        // "hal" - 1 char prefix match
        let score_hal = calculate_completion_score(&query, &chars("hal"), 2, false, false, 0.0);

        // help should rank highest (edit distance 1)
        // hello should rank higher than hal (same edit distance, but better prefix)
        assert!(
            score_help > score_hello,
            "Expected 'help' with ED=1 ({}) to rank higher than 'hello' with ED=2 ({})",
            score_help,
            score_hello
        );
        assert!(
            score_hello > score_hal,
            "Expected 'hello' with 3-char prefix ({}) to rank higher than 'hal' with 1-char prefix ({})",
            score_hello,
            score_hal
        );
    }

    #[test]
    fn test_substring_match() {
        // Query contained as substring should get bonus
        // "his" is a substring of "this"
        let query = chars("his");
        let score_substring = calculate_completion_score(&query, &chars("this"), 1, true, false, 0.0);

        let query2 = chars("wor");
        let score_word = calculate_completion_score(&query2, &chars("word"), 1, false, false, 0.0);

        // These should get substring bonus (25 points) and have positive scores
        // Just verify they're positive scores with reasonable values
        assert!(
            score_substring > 100.0,
            "Expected substring match 'his' in 'this' to have high score, got {}",
            score_substring
        );
        assert!(
            score_word > 100.0,
            "Expected substring match 'wor' in 'word' to have high score, got {}",
            score_word
        );
    }

    #[test]
    fn test_exact_match_wins() {
        // Exact matches (edit distance 0) should always have the highest scores
        let query = chars("test");

        let score_exact = calculate_completion_score(&query, &chars("test"), 0, false, false, 0.0);
        let score_ed1 = calculate_completion_score(&query, &chars("text"), 1, false, false, 0.0);
        let score_ed2 = calculate_completion_score(&query, &chars("best"), 1, false, false, 0.0);

        assert!(
            score_exact > score_ed1,
            "Exact match ({}) should beat ED=1 ({})",
            score_exact,
            score_ed1
        );
        assert!(
            score_exact > score_ed2,
            "Exact match ({}) should beat ED=1 with first letter mismatch ({})",
            score_exact,
            score_ed2
        );
        assert!(
            score_ed1 > score_ed2,
            "ED=1 with first letter match ({}) should beat ED=1 with first letter mismatch ({})",
            score_ed1,
            score_ed2
        );
    }

    #[test]
    fn test_first_letter_mismatch_penalty() {
        // Words with mismatched first letter should rank much lower
        let query = chars("test");

        let score_match = calculate_completion_score(&query, &chars("test"), 0, false, false, 0.0);
        let score_mismatch = calculate_completion_score(&query, &chars("best"), 1, false, false, 0.0);

        // First letter mismatch gives -30 penalty, so even with lower edit distance,
        // mismatched first letter should be heavily penalized
        assert!(
            score_match > score_mismatch,
            "First letter match ({}) should beat mismatch ({})",
            score_match,
            score_mismatch
        );
    }

    #[test]
    fn test_common_word_bonus_scaling() {
        // Common words should get bonuses, but scaled by length
        let query = chars("th");

        // Short common word (len=3)
        let score_the = calculate_completion_score(&query, &chars("the"), 1, true, false, 0.0);

        // Medium common word (len=5)
        let score_there = calculate_completion_score(&query, &chars("there"), 1, true, false, 0.0);

        // Long common word (len=7)
        let score_through = calculate_completion_score(&query, &chars("through"), 1, true, false, 0.0);

        // Non-common word
        let score_thy = calculate_completion_score(&query, &chars("thy"), 1, false, false, 0.0);

        // All common words should rank higher than non-common
        assert!(
            score_the > score_thy,
            "Common word 'the' ({}) should rank higher than non-common 'thy' ({})",
            score_the,
            score_thy
        );
        assert!(
            score_there > score_thy,
            "Common word 'there' ({}) should rank higher than non-common 'thy' ({})",
            score_there,
            score_thy
        );
        assert!(
            score_through > score_thy,
            "Common word 'through' ({}) should rank higher than non-common 'thy' ({})",
            score_through,
            score_thy
        );

        // Shorter common words get bigger bonuses
        assert!(
            score_the > score_through,
            "Short common word 'the' ({}) should rank higher than long common word 'through' ({})",
            score_the,
            score_through
        );
    }

    #[test]
    fn test_character_frequency_similarity() {
        // Test the character frequency similarity helper function
        let word_a = chars("hello");
        let word_b = chars("olleh"); // Anagram - should have similarity 1.0
        let word_c = chars("world"); // Different chars - should have low similarity

        let sim_anagram = calculate_char_frequency_similarity(&word_a, &word_b);
        let sim_different = calculate_char_frequency_similarity(&word_a, &word_c);

        assert!(
            sim_anagram > 0.9,
            "Anagrams should have very high similarity, got {}",
            sim_anagram
        );
        assert!(
            sim_different < sim_anagram,
            "Different words ({}) should have lower similarity than anagrams ({})",
            sim_different,
            sim_anagram
        );
    }

    #[test]
    fn test_position_of_first_difference() {
        // Errors later in the word should be more forgivable
        let query = chars("test");

        // Difference at position 0 (beginning)
        let score_beginning = calculate_completion_score(&query, &chars("best"), 1, false, false, 0.0);

        // Difference at position 3 (end)
        let score_end = calculate_completion_score(&query, &chars("text"), 1, false, false, 0.0);

        assert!(
            score_end > score_beginning,
            "Error at end ({}) should be more forgivable than at beginning ({})",
            score_end,
            score_beginning
        );
    }

    #[test]
    fn test_length_difference_penalty() {
        // Test that length penalties exist by comparing words of different lengths
        // with the same edit distance and other characteristics
        let query = chars("abc");

        // Test with ED=2 to avoid substring bonus complications
        // Same length (ED=2)
        let score_same_len = calculate_completion_score(&query, &chars("xyz"), 2, false, false, 0.0);
        // +3 length (ED=2)
        let score_len_plus_3 = calculate_completion_score(&query, &chars("xyzdef"), 2, false, false, 0.0);
        // +6 length (ED=2)
        let score_len_plus_6 = calculate_completion_score(&query, &chars("xyzdefghi"), 2, false, false, 0.0);

        // Longer words should be penalized when other factors are similar
        assert!(
            score_same_len > score_len_plus_3,
            "Same length ({}) should beat +3 length ({})",
            score_same_len,
            score_len_plus_3
        );
        assert!(
            score_len_plus_3 > score_len_plus_6,
            "+3 length ({}) should beat +6 length ({})",
            score_len_plus_3,
            score_len_plus_6
        );
    }
}

