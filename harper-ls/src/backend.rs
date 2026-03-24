use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
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
use harper_core::keyboard_distance::avg_keyboard_distance;
use harper_core::linting::{LintGroup, LintGroupConfig};
use harper_core::parsers::{
    CollapseIdentifiers, IsolateEnglish, Markdown, OrgMode, Parser, PlainEnglish,
};
use harper_core::spell::{Dictionary, FstDictionary, MergedDictionary, MutableDictionary};
use harper_core::{Dialect, DictWordMetadata, Document, IgnoredLints, TokenKind};
use harper_html::HtmlParser;
use harper_ink::InkParser;
use harper_jjdescription::JJDescriptionParser;
use harper_literate_haskell::LiterateHaskellParser;
use harper_python::PythonParser;
use harper_stats::{Record, Stats};
use harper_typst::Typst;
use serde_json::Value;
use tokio::sync::{Mutex, Notify, RwLock};
use tokio::time::{Duration, Instant, sleep, timeout};
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

#[derive(Clone)]
struct InFlightChange {
    notify: Arc<Notify>,
}

const MISSPELLED_REFRESH_DEBOUNCE_MS: u64 = 75;
const DIAGNOSTICS_PUBLISH_DEBOUNCE_MS: u64 = 300;

pub struct Backend {
    client: Client,
    root: RwLock<PathBuf>,
    config: RwLock<Config>,
    stats: RwLock<Stats>,
    doc_state: Mutex<HashMap<Uri, DocumentState>>,
    pending_changes: RwLock<HashMap<Uri, Instant>>,
    dict_cache: RwLock<HashMap<Uri, DictCacheEntry>>,
    in_flight_changes: RwLock<HashMap<Uri, InFlightChange>>,
}

#[derive(Debug, Clone)]
struct CompletionSortRank {
    edit_distance: u8,
    first_letter_match: bool,
    prefix_match_len: usize,
    is_transposition: bool,
    is_common: bool,
    avg_kbd_distance: f32,
    len_diff: usize,
}

fn first_letter_matches(query: &[char], candidate: &[char]) -> bool {
    !query.is_empty() && !candidate.is_empty() && query[0].eq_ignore_ascii_case(&candidate[0])
}

fn common_prefix_len(query: &[char], candidate: &[char]) -> usize {
    query
        .iter()
        .zip(candidate.iter())
        .take_while(|(q, c)| q.eq_ignore_ascii_case(c))
        .count()
}

fn is_simple_transposition(query: &[char], candidate: &[char]) -> bool {
    if query.len() != candidate.len() {
        return false;
    }

    let mut diff_positions = Vec::new();
    for (i, (q, c)) in query.iter().zip(candidate.iter()).enumerate() {
        if !q.eq_ignore_ascii_case(c) {
            diff_positions.push(i);
            if diff_positions.len() > 2 {
                return false;
            }
        }
    }

    if diff_positions.len() != 2 {
        return false;
    }

    let i = diff_positions[0];
    let j = diff_positions[1];
    j == i + 1
        && query[i].eq_ignore_ascii_case(&candidate[j])
        && query[j].eq_ignore_ascii_case(&candidate[i])
}

fn completion_sort_rank(
    query: &[char],
    candidate: &[char],
    edit_distance: u8,
    is_common: bool,
) -> CompletionSortRank {
    CompletionSortRank {
        edit_distance,
        first_letter_match: first_letter_matches(query, candidate),
        prefix_match_len: common_prefix_len(query, candidate),
        is_transposition: is_simple_transposition(query, candidate),
        // frequency proxy (best currently available metadata signal)
        is_common,
        avg_kbd_distance: if edit_distance > 0 {
            avg_keyboard_distance(query, candidate)
        } else {
            0.0
        },
        len_diff: candidate.len().abs_diff(query.len()),
    }
}

fn compare_completion_ranks(a: &CompletionSortRank, b: &CompletionSortRank) -> Ordering {
    a.edit_distance
        .cmp(&b.edit_distance)
        .then_with(|| b.first_letter_match.cmp(&a.first_letter_match))
        .then_with(|| b.prefix_match_len.cmp(&a.prefix_match_len))
        .then_with(|| b.is_transposition.cmp(&a.is_transposition))
        .then_with(|| b.is_common.cmp(&a.is_common))
        .then_with(|| {
            a.avg_kbd_distance
                .partial_cmp(&b.avg_kbd_distance)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| a.len_diff.cmp(&b.len_diff))
}

/// Apply casing intent from the typed prefix to a completion candidate.
///
/// Behavior:
/// - All-uppercase prefix with at least 2 letters => uppercase completion.
/// - First-letter uppercase prefix => title-case completion for normal words.
/// - Preserve mixed-case words that start lowercase with internal capitals
///   (e.g. "iPhone", "eBay") when prefix is first-letter uppercase.
/// - Otherwise keep original completion casing.
fn apply_prefix_casing(prefix: &[char], word: &str) -> String {
    if prefix.is_empty() {
        return word.to_string();
    }

    let word_chars: Vec<char> = word.chars().collect();
    if word_chars.is_empty() {
        return word.to_string();
    }

    let first_is_upper = prefix[0].is_uppercase();
    let all_upper = prefix
        .iter()
        .all(|c| !c.is_alphabetic() || c.is_uppercase());
    let alphabetic_count = prefix.iter().filter(|c| c.is_alphabetic()).count();

    // Only treat as CAPS intent when there are 2+ alphabetic characters.
    if all_upper && alphabetic_count >= 2 {
        return word.to_uppercase();
    }

    if first_is_upper {
        let has_internal_upper = word_chars.iter().skip(1).any(|c| c.is_uppercase());
        let starts_lowercase = word_chars[0].is_lowercase();

        // Preserve branded/mixed-case forms like iPhone and eBay.
        if starts_lowercase && has_internal_upper {
            return word.to_string();
        }

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

    word.to_string()
}

fn extract_misspelled_words(
    source: &[char],
    line_index: &crate::pos_conv::LineIndex,
    diagnostics: &[Diagnostic],
) -> HashSet<String> {
    use tower_lsp_server::lsp_types::NumberOrString;

    diagnostics
        .iter()
        .filter(|diag| {
            diag.code.as_ref().is_some_and(|code| match code {
                NumberOrString::String(s) => s.contains("Spelling"),
                _ => false,
            })
        })
        .filter_map(|diag| {
            let span = line_index.range_to_span(source, diag.range);
            if span.start >= span.end || span.end > source.len() {
                return None;
            }

            let word: String = span.get_content(source).iter().collect();
            if word.is_empty() {
                None
            } else {
                Some(word.to_lowercase())
            }
        })
        .collect()
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
            in_flight_changes: RwLock::new(HashMap::new()),
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

    async fn load_user_dictionary(&self) -> MutableDictionary {
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
            (
                config.user_dict_path.clone(),
                config.workspace_dict_path.clone(),
            )
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
                completion_dict: dict.clone(),
                lint_dict: dict.clone(),
                uri: uri.clone(),
                ..Default::default()
            }
        });

        if doc_state.completion_dict != dict {
            doc_state.completion_dict = dict.clone();
            let mut merged = (*doc_state.completion_dict).clone();
            merged.add_dictionary(doc_state.ident_dict.clone());
            doc_state.lint_dict = Arc::new(merged);
            info!("Constructing new linter because of modified dictionary.");
            doc_state.linter = LintGroup::new_curated(doc_state.lint_dict.clone(), dialect)
                .with_lint_config(lint_config.clone());
        }

        let Some(language_id) = &doc_state.language_id else {
            eprintln!("HARPER: No language_id for document, removing: {:?}", uri);
            doc_lock.remove(uri);
            return Ok(());
        };

        fn use_ident_dict(
            new_dict: Arc<MutableDictionary>,
            parser: impl Parser + 'static,
            doc_state: &mut DocumentState,
            lint_config: &LintGroupConfig,
            dialect: Dialect,
        ) -> Result<Box<dyn Parser>> {
            if doc_state.ident_dict != new_dict {
                info!("Constructing new linter because of modified ident dictionary.");
                doc_state.ident_dict = new_dict.clone();

                let mut merged = (*doc_state.completion_dict).clone();
                merged.add_dictionary(new_dict);
                let merged = Arc::new(merged);

                doc_state.linter = LintGroup::new_curated(merged.clone(), dialect)
                    .with_lint_config(lint_config.clone());
                doc_state.lint_dict = merged.clone();
            }

            Ok(Box::new(CollapseIdentifiers::new(
                Box::new(parser),
                Box::new(doc_state.lint_dict.clone()),
            )))
        }

        let source: Vec<char> = text.chars().collect();
        let ts_parser = CommentParser::new_from_language_id(language_id, markdown_options);
        let parser: Option<Box<dyn Parser>> = match language_id.as_str() {
            _ if ts_parser.is_some() => {
                let ts_parser = ts_parser.unwrap();

                if let Some(new_dict) = ts_parser.create_ident_dict(&Arc::new(source)) {
                    Some(use_ident_dict(
                        Arc::new(new_dict),
                        ts_parser,
                        doc_state,
                        &lint_config,
                        dialect,
                    )?)
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
                    Some(use_ident_dict(
                        Arc::new(new_dict),
                        parser,
                        doc_state,
                        &lint_config,
                        dialect,
                    )?)
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
                    parser = Box::new(IsolateEnglish::new(parser, doc_state.lint_dict.clone()));
                }

                // Don't lint on documents larger than the configured maximum length.
                if text.len() <= max_file_length {
                    doc_state.document = Document::new(text, &parser, &doc_state.lint_dict);
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
                    // Different diagnostics - update cached diagnostics and misspelled words
                    let source = doc_state.document.get_source();
                    doc_state.misspelled_words = Arc::new(extract_misspelled_words(
                        source,
                        &doc_state.line_index,
                        &diagnostics,
                    ));
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

    async fn refresh_misspelled_words_cache(&self, uri: &Uri) {
        let diagnostics = self.generate_diagnostics(uri).await;

        let mut doc_states = self.doc_state.lock().await;
        let Some(doc_state) = doc_states.get_mut(uri) else {
            return;
        };

        let source = doc_state.document.get_source();
        doc_state.misspelled_words = Arc::new(extract_misspelled_words(
            source,
            &doc_state.line_index,
            &diagnostics,
        ));
    }

    async fn wait_for_in_flight_change(&self, uri: &Uri) {
        // Wait for currently running didChange processing to finish.
        // Wakeups are notify-driven (no fixed sleep), with a timeout as fallback.
        for _ in 0..3 {
            let notify = {
                let in_flight = self.in_flight_changes.read().await;
                in_flight.get(uri).map(|state| state.notify.clone())
            };

            let Some(notify) = notify else {
                return;
            };

            if timeout(Duration::from_millis(250), notify.notified())
                .await
                .is_err()
            {
                return;
            }
        }
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

        // Copy needed data while holding lock, then release it before expensive operations.
        let (source, dict, line_index, misspelled_words, cursor_index) = {
            let doc_states = self.doc_state.lock().await;
            let Some(doc_state) = doc_states.get(uri) else {
                return Ok(Vec::new());
            };
            let doc_source = doc_state.document.get_source();
            let source: Vec<char> = doc_source.iter().copied().collect();

            // Convert LSP position to character index while holding the lock so we can
            // check the token kind before releasing it.
            let cursor_index = doc_state.line_index.position_to_index(&source, position);

            // Only complete inside lintable regions (comments/docstrings).
            // Code regions are marked Unlintable by the language parser — skip them.
            // If no token exists at the cursor (e.g. empty file), also skip.
            let in_lintable_region = doc_state
                .document
                .get_token_at_char_index(cursor_index)
                .is_some_and(|t| !matches!(t.kind, TokenKind::Unlintable));
            if !in_lintable_region {
                return Ok(Vec::new());
            }

            (
                source,
                doc_state.completion_dict.clone(),
                doc_state.line_index.clone(),
                doc_state.misspelled_words.clone(),
                cursor_index,
            )
        }; // Lock released here

        // Find the word being typed by looking backwards from cursor
        let word_start = source[..cursor_index]
            .iter()
            .rposition(|c| !c.is_alphanumeric() && *c != '\'' && *c != '-')
            .map(|i| i + 1)
            .unwrap_or(0);

        // Extract the prefix being typed
        let prefix: Vec<char> = source[word_start..cursor_index].to_vec();

        // Don't show completions for very short prefixes
        if prefix.len() < completion_config.min_prefix_length {
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
        let fuzzy_completions = dict.fuzzy_match(&prefix, max_edit_distance, 200);

        let mut completions: Vec<(String, CompletionSortRank)> = fuzzy_completions
            .into_iter()
            .filter_map(|fuzzy_match| {
                let word_string: String = fuzzy_match.word.iter().collect();

                // Filter out words that are currently marked as misspelled in the document
                // This prevents suggesting "tthis" when the user typed it earlier and it has an error
                if misspelled_words.contains(&word_string.to_lowercase()) {
                    return None;
                }

                let rank = completion_sort_rank(
                    &prefix,
                    fuzzy_match.word,
                    fuzzy_match.edit_distance,
                    fuzzy_match.metadata.common,
                );
                Some((word_string, rank))
            })
            .collect();

        completions
            .sort_by(|a, b| compare_completion_ranks(&a.1, &b.1).then_with(|| a.0.cmp(&b.0)));

        // Calculate the start position of the word being completed
        let word_start_position = line_index.index_to_position(&source, word_start);
        let cursor_position = line_index.index_to_position(&source, cursor_index);

        // Convert prefix to string - we'll use this as filterText
        // This tells Helix that all our completions match the user's input
        let prefix_string: String = prefix.iter().collect();

        // Convert to LSP completion items
        // Use text_edit to specify exact replacement range - this tells Helix what to replace
        // Set filter_text to prefix so items aren't filtered out while typing
        let completion_items: Vec<CompletionItem> = completions
            .into_iter()
            .take(completion_config.max_results)
            .enumerate()
            .map(|(idx, (word_string, _))| {
                use tower_lsp_server::lsp_types::{CompletionTextEdit, Range, TextEdit};

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
                            end: cursor_position,
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
                    // Space auto-accepts first suggestion for natural spell-checker flow
                    all_commit_characters: Some(vec![" ".to_string()]),
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
        let change_notify = Arc::new(Notify::new());

        eprintln!(
            "HARPER DID_CHANGE: version={:?}, text_len={}, last_40_chars={:?}",
            params.text_document.version,
            text.len(),
            text.chars()
                .rev()
                .take(40)
                .collect::<Vec<_>>()
                .iter()
                .rev()
                .collect::<String>()
        );

        // Mark this URI as actively updating so completion requests can wait on a notification
        // instead of sleeping for a fixed duration.
        if let Some(previous_notify) = {
            let mut in_flight = self.in_flight_changes.write().await;
            in_flight
                .insert(
                    uri.clone(),
                    InFlightChange {
                        notify: change_notify.clone(),
                    },
                )
                .map(|old| old.notify)
        } {
            previous_notify.notify_waiters();
        }

        // IMPORTANT: Update document content immediately (without debounce)
        // This ensures completions have access to the latest text while typing
        if let Err(err) = self.update_document(&uri, &text, None).await {
            error!("{err}")
        }

        // Signal completion requests waiting on this update.
        if let Some(done_notify) = {
            let mut in_flight = self.in_flight_changes.write().await;
            if in_flight
                .get(&uri)
                .is_some_and(|state| Arc::ptr_eq(&state.notify, &change_notify))
            {
                in_flight.remove(&uri).map(|state| state.notify)
            } else {
                None
            }
        } {
            done_notify.notify_waiters();
        }

        let change_elapsed = change_start.elapsed();
        eprintln!(
            "HARPER DID_CHANGE COMPLETE: version={:?}, took {:?}",
            params.text_document.version, change_elapsed
        );

        // Record this change with current timestamp for debounced diagnostics
        let now = Instant::now();
        {
            let mut pending = self.pending_changes.write().await;
            pending.insert(uri.clone(), now);
        }

        // Refresh misspelled-word filtering sooner than full diagnostics publish.
        sleep(Duration::from_millis(MISSPELLED_REFRESH_DEBOUNCE_MS)).await;

        let should_refresh_misspellings = {
            let pending = self.pending_changes.read().await;
            pending.get(&uri).is_some_and(|&timestamp| timestamp == now)
        };

        if should_refresh_misspellings {
            self.refresh_misspelled_words_cache(&uri).await;
        } else {
            // A newer change came in, skip this refresh cycle.
            return;
        }

        let publish_wait_ms =
            DIAGNOSTICS_PUBLISH_DEBOUNCE_MS.saturating_sub(MISSPELLED_REFRESH_DEBOUNCE_MS);
        if publish_wait_ms > 0 {
            sleep(Duration::from_millis(publish_wait_ms)).await;
        }

        // Check if this is still the latest change for this URI
        let should_process = {
            let pending = self.pending_changes.read().await;
            pending
                .get(&uri)
                .map_or(false, |&timestamp| timestamp == now)
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
        drop(doc_lock);

        {
            let mut pending = self.pending_changes.write().await;
            pending.remove(&uri);
        }

        if let Some(change) = {
            let mut in_flight = self.in_flight_changes.write().await;
            in_flight.remove(&uri)
        } {
            change.notify.notify_waiters();
        }

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
                doc.linter = LintGroup::new_curated(doc.lint_dict.clone(), config_lock.dialect)
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

    async fn completion(&self, params: CompletionParams) -> JsonResult<Option<CompletionResponse>> {
        self.wait_for_in_flight_change(&params.text_document_position.text_document.uri)
            .await;

        let completions = self
            .generate_completions(
                &params.text_document_position.text_document.uri,
                params.text_document_position.position,
            )
            .await?;

        if completions.is_empty() {
            Ok(None)
        } else {
            use tower_lsp_server::lsp_types::CompletionList;
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
mod completion_ranking_tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    fn rank(
        query: &str,
        candidate: &str,
        edit_distance: u8,
        is_common: bool,
    ) -> CompletionSortRank {
        completion_sort_rank(
            chars(query).as_slice(),
            chars(candidate).as_slice(),
            edit_distance,
            is_common,
        )
    }

    fn assert_better(query: &str, better: (&str, u8, bool), worse: (&str, u8, bool), reason: &str) {
        let better_rank = rank(query, better.0, better.1, better.2);
        let worse_rank = rank(query, worse.0, worse.1, worse.2);

        assert_eq!(
            compare_completion_ranks(&better_rank, &worse_rank),
            Ordering::Less,
            "{}",
            reason
        );
    }

    #[test]
    fn test_priority_edit_distance() {
        assert_better(
            "hello",
            ("hello", 0, false),
            ("hallo", 1, false),
            "Lower edit distance must rank first",
        );
    }

    #[test]
    fn test_priority_first_letter_match() {
        assert_better(
            "test",
            ("text", 1, false),
            ("best", 1, false),
            "First-letter match must beat mismatch when edit distance is tied",
        );
    }

    #[test]
    fn test_priority_prefix_length() {
        assert_better(
            "ths",
            ("this", 1, false),
            ("tos", 1, false),
            "Longer common prefix must rank first when higher priorities tie",
        );
    }

    #[test]
    fn test_priority_transposition() {
        assert_better(
            "teh",
            ("the", 1, false),
            ("tah", 1, false),
            "Transposition should beat non-transposition when higher priorities tie",
        );
    }

    #[test]
    fn test_tiebreak_common_word_proxy() {
        assert_better(
            "hel",
            ("help", 1, true),
            ("helm", 1, false),
            "Common words (frequency proxy) should win tie-breaks",
        );
    }

    #[test]
    fn test_tiebreak_keyboard_distance() {
        assert_better(
            "tge",
            ("the", 1, false),
            ("toe", 1, false),
            "Lower keyboard distance should win tie-breaks",
        );
    }

    #[test]
    fn test_tiebreak_length_difference() {
        assert_better(
            "abc",
            ("abx", 1, false),
            ("abxy", 1, false),
            "Smaller length difference should win final tie-break",
        );
    }

    #[test]
    fn test_apply_prefix_casing_preserves_mixed_case_words() {
        let prefix = chars("Ip");
        assert_eq!(apply_prefix_casing(&prefix, "iPhone"), "iPhone");

        let prefix2 = chars("Eb");
        assert_eq!(apply_prefix_casing(&prefix2, "eBay"), "eBay");
    }

    #[test]
    fn test_apply_prefix_casing_caps_intent_requires_two_letters() {
        let one_letter_caps = chars("I");
        assert_eq!(apply_prefix_casing(&one_letter_caps, "iphone"), "Iphone");

        let two_letter_caps = chars("IP");
        assert_eq!(apply_prefix_casing(&two_letter_caps, "iphone"), "IPHONE");
    }

    #[test]
    fn test_transposition_detection() {
        assert!(is_simple_transposition(&chars("teh"), &chars("the")));
        assert!(!is_simple_transposition(&chars("teh"), &chars("ten")));
    }
}
