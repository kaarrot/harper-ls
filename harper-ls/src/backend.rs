use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use crate::config::{CompletionCommitWithSpace, Config};
use crate::dictionary_io::{load_dict, save_dict};
use crate::document_state::DocumentState;
use crate::git_commit_parser::GitCommitParser;
use crate::ignored_lints_io::{load_ignored_lints, save_ignored_lints};
use crate::io_utils::fileify_path;
use anyhow::{Context, Result, anyhow};
use futures::future::join;
use harper_comments::CommentParser;
use harper_core::keyboard_distance::{
    INSERT_DELETE_COST, keyboard_substitution_cost, weighted_damerau_distance,
};
use harper_core::linting::{LintGroup, LintGroupConfig};
use harper_core::parsers::{
    CollapseIdentifiers, IsolateEnglish, Markdown, OrgMode, Parser, PlainEnglish,
};
use harper_core::spell::{Dictionary, FstDictionary, MergedDictionary, MutableDictionary};
use harper_core::word_frequency::{bigram_rank, frequency_rank, most_frequent_with_prefix};
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
    CompletionTextEdit, ConfigurationItem, Diagnostic, DidChangeConfigurationParams,
    DidChangeTextDocumentParams, DidChangeWatchedFilesParams,
    DidChangeWatchedFilesRegistrationOptions, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, ExecuteCommandOptions, ExecuteCommandParams, FileChangeType,
    FileSystemWatcher, GlobPattern, InitializeParams, InitializeResult, InitializedParams,
    MessageType, PublishDiagnosticsParams, Range, Registration, ServerCapabilities, ServerInfo,
    TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions,
    TextDocumentSyncSaveOptions, TextEdit, Uri, WatchKind,
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
const MIN_COMPLETION_CANDIDATE_POOL: usize = 64;
const MAX_EXACT_PREFIX_CANDIDATES: usize = 256;
const MAX_FUZZY_COMPLETION_CANDIDATES: usize = 200;
const MAX_NOISY_PREFIX_EXPANSION_CANDIDATES: usize = 128;
/// Bottom-row phone keyboard keys adjacent to the spacebar; pressing one of
/// these instead of space fuses two words into one token (e.g. "thenworld").
const SPACE_ADJACENT_KEYS: &[char] = &['v', 'b', 'n', 'm'];

#[derive(Clone)]
pub struct Backend {
    client: Client,
    root: Arc<RwLock<PathBuf>>,
    config: Arc<RwLock<Config>>,
    stats: Arc<RwLock<Stats>>,
    doc_state: Arc<Mutex<HashMap<Uri, DocumentState>>>,
    pending_changes: Arc<RwLock<HashMap<Uri, Instant>>>,
    dict_cache: Arc<RwLock<HashMap<Uri, DictCacheEntry>>>,
    in_flight_changes: Arc<RwLock<HashMap<Uri, InFlightChange>>>,
}

#[derive(Debug, Clone)]
struct CompletionSortRank {
    /// Primary ranking key: keyboard-edit cost blended with frequency and context
    /// (lower is better). See [`combined_completion_cost`].
    combined_cost: i32,
    distance: u16,
    exact_prefix: bool,
    typed_word: bool,
    first_letter_match: bool,
    is_common: bool,
    remaining_ambiguity: usize,
    replacement_len: usize,
}

#[derive(Debug, Clone)]
struct CompletionPrefixScore {
    distance: u16,
    exact_prefix: bool,
    matched_prefix_len: usize,
}

#[derive(Debug, Clone)]
struct RankedCompletion {
    word: String,
    rank: CompletionSortRank,
}

/// Document-wide word statistics for completion ranking, built once per parsed
/// document revision (in `update_document`, riding the full reparse that already
/// happens on every change) and shared with completion requests via `Arc`.
/// Counts include every word token — the word currently being typed is removed
/// at query time through [`CursorExclusions`].
#[derive(Debug, Default)]
pub(crate) struct CompletionStats {
    word_counts: HashMap<String, usize>,
    /// Bigram counts keyed by previous word, then following word.
    bigram_counts: HashMap<String, HashMap<String, usize>>,
}

/// The cursor word's contribution to [`CompletionStats`], subtracted at query
/// time so the half-typed word doesn't boost itself. Each entry is one excluded
/// occurrence.
#[derive(Debug, Clone, Default)]
struct CursorExclusions {
    words: Vec<String>,
    pairs: Vec<(String, String)>,
}

#[derive(Debug, Clone, Default)]
struct CompletionContext {
    stats: Arc<CompletionStats>,
    exclusions: CursorExclusions,
    previous_word: Option<String>,
}

impl CompletionContext {
    /// How often `word` appears in the document, excluding the word being typed.
    fn local_count(&self, word: &str) -> usize {
        let total = self.stats.word_counts.get(word).copied().unwrap_or(0);
        let excluded = self
            .exclusions
            .words
            .iter()
            .filter(|excluded| excluded.as_str() == word)
            .count();
        total.saturating_sub(excluded)
    }

    /// How often `word` follows `previous_word` in the document, excluding pairs
    /// involving the word being typed.
    fn pair_count(&self, previous_word: &str, word: &str) -> usize {
        let total = self
            .stats
            .bigram_counts
            .get(previous_word)
            .and_then(|following| following.get(word))
            .copied()
            .unwrap_or(0);
        let excluded = self
            .exclusions
            .pairs
            .iter()
            .filter(|(prev, next)| prev == previous_word && next == word)
            .count();
        total.saturating_sub(excluded)
    }

    /// How often `word` follows the word preceding the cursor.
    fn previous_pair_count(&self, word: &str) -> usize {
        self.previous_word
            .as_deref()
            .map(|previous_word| self.pair_count(previous_word, word))
            .unwrap_or(0)
    }
}

fn first_letter_matches(query: &[char], candidate: &[char]) -> bool {
    !query.is_empty() && !candidate.is_empty() && query[0].eq_ignore_ascii_case(&candidate[0])
}

fn chars_equal_ignore_case(a: char, b: char) -> bool {
    a.eq_ignore_ascii_case(&b)
}

fn starts_with_ignore_case(candidate: &[char], prefix: &[char]) -> bool {
    candidate.len() >= prefix.len()
        && candidate
            .iter()
            .zip(prefix.iter())
            .all(|(candidate_char, prefix_char)| {
                chars_equal_ignore_case(*candidate_char, *prefix_char)
            })
}

fn same_word_ignore_case(a: &[char], b: &[char]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(a_char, b_char)| chars_equal_ignore_case(*a_char, *b_char))
}

fn allowed_noisy_prefix_edits(prefix_len: usize) -> usize {
    match prefix_len {
        0..=2 => 1,
        3..=5 => 2,
        6..=9 => 3,
        _ => 4,
    }
}

fn max_noisy_prefix_distance(prefix_len: usize) -> u16 {
    (allowed_noisy_prefix_edits(prefix_len) as u16).saturating_mul(INSERT_DELETE_COST)
}

fn best_completion_prefix_score(
    query: &[char],
    candidate: &[char],
) -> Option<CompletionPrefixScore> {
    if query.is_empty() || candidate.is_empty() {
        return None;
    }

    if starts_with_ignore_case(candidate, query) {
        // Exact-prefix matches carry no correction distance — the user typed a real
        // prefix of the candidate, not a typo. Completion length is preferred only as
        // a low-priority tiebreaker (`remaining_ambiguity`), which sits below global
        // frequency, so a frequent longer word (e.g. "world") still beats a rare
        // shorter one (e.g. "worm").
        return Some(CompletionPrefixScore {
            distance: 0,
            exact_prefix: true,
            matched_prefix_len: query.len(),
        });
    }

    let allowed_edits = allowed_noisy_prefix_edits(query.len());
    if candidate.len() + allowed_edits < query.len() {
        return None;
    }

    let lower_prefix_len = query.len().saturating_sub(allowed_edits).max(1);
    let upper_prefix_len = candidate.len().min(query.len() + allowed_edits);

    if lower_prefix_len > upper_prefix_len {
        return None;
    }

    let max_distance = max_noisy_prefix_distance(query.len());

    (lower_prefix_len..=upper_prefix_len)
        .map(|prefix_len| CompletionPrefixScore {
            distance: weighted_damerau_distance(query, &candidate[..prefix_len]),
            exact_prefix: false,
            matched_prefix_len: prefix_len,
        })
        .filter(|score| score.distance <= max_distance)
        .min_by(|a, b| {
            a.distance
                .cmp(&b.distance)
                .then_with(|| b.matched_prefix_len.cmp(&a.matched_prefix_len))
        })
}

fn should_score_noisy_prefix_candidate(query: &[char], candidate: &[char]) -> bool {
    if query.is_empty() || candidate.is_empty() {
        return false;
    }

    if chars_equal_ignore_case(query[0], candidate[0])
        || keyboard_substitution_cost(query[0], candidate[0]) <= INSERT_DELETE_COST
    {
        return true;
    }

    if query.len() > 1 && chars_equal_ignore_case(query[1], candidate[0]) {
        return true;
    }

    candidate.len() > 1 && chars_equal_ignore_case(query[0], candidate[1])
}

fn normalized_completion_word(chars: &[char]) -> String {
    chars.iter().collect::<String>().to_lowercase()
}

fn lowercase_chars(chars: &[char]) -> Vec<char> {
    chars
        .iter()
        .map(|c| c.to_lowercase().next().unwrap_or(*c))
        .collect()
}

/// Weight on the log-frequency penalty. A candidate's frequency penalty is
/// `FREQ_LOG_WEIGHT * ln(rank + 1)`, so the penalty grows slowly among common
/// words (small gaps near the top of the list) but separates common words from
/// rare ones — letting a much more frequent word reachable by a small keyboard
/// edit outrank a rare exact match without flipping two similarly-common words.
const FREQ_LOG_WEIGHT: f32 = 30.0;
/// Penalty for words absent from the bundled 10k frequency list. Larger than the
/// penalty of the least-frequent listed word (≈368), so any listed word is treated
/// as more likely than an unlisted one, but small enough that a 2-3 edit gap can
/// still overcome it.
const UNKNOWN_FREQUENCY_PENALTY: i32 = 450;
/// Cost shaved off the word the user actually typed. Keeps a *common* typed word
/// on top of a slightly-more-common neighbor (so "do" isn't replaced by "to"),
/// while still letting a far-more-common neighbor win when the typed word is rare
/// (so "ruse" yields "rise").
const TYPED_WORD_BONUS: i32 = 50;
/// Per-occurrence cost shaved off a word already used in the document, capped so a
/// heavily-repeated word can't override a large keyboard-distance gap.
const LOCAL_USE_BONUS_PER_COUNT: i32 = 60;
const MAX_LOCAL_USE_BONUS: i32 = 240;
/// Per-occurrence cost shaved off a word seen following the previous word in the
/// document (document-local bigram context), capped like the local-use bonus.
const PREVIOUS_WORD_BONUS_PER_COUNT: i32 = 60;
const MAX_PREVIOUS_WORD_BONUS: i32 = 240;
/// Strongest cost shaved off a word that forms a known English collocation with
/// the previous word (e.g. "new york"). Scaled down for rarer pairs.
const GLOBAL_BIGRAM_BONUS_MAX: f32 = 200.0;
const GLOBAL_BIGRAM_BONUS_MIN: i32 = 50;
const GLOBAL_BIGRAM_LIST_LEN: f32 = 30_000.0;
/// Extra cost for a correction that changes the first letter of what was typed.
/// Phone users rarely fat-finger the first keystroke, so a same-first-letter
/// substitution ("ruse"->"rise") is preferred over a first-letter-changing edit
/// ("ruse"->"use") even when the latter is a more common word. Sized from real data
/// to sit above ~97 (so "use", rank 60 and one deletion away, can't beat "rise" for
/// "ruse") and below ~148 (so "and" still beats the first-letter-preserving "send"
/// for "snd").
const FIRST_LETTER_MISMATCH_PENALTY: i32 = 120;
/// Bonus for a "high-confidence short correction": a short typed string (<=3 chars)
/// one keyboard edit away from a same-length *common* word that isn't a prefix of
/// what was typed (e.g. "teh"->"the", "adn"->"and"). This nudges a confident short
/// fat-finger fix up the list, but is intentionally small so a clearly more frequent
/// word still wins — it sits *below* frequency, not above it as the old tier did.
/// Kept under the protections (<~76) so a common typed word isn't flipped to a
/// same-length neighbor (e.g. "do" stays above "to").
const HIGH_CONFIDENCE_SHORT_CORRECTION_BONUS: i32 = 40;
/// `combined_cost` for an apostrophe-restored contraction ("im"->"I'm") when the
/// typed letters are NOT themselves a word — the contraction is almost certainly
/// the intent, so rank it at the top (well below the cost of any fuzzy neighbor).
const CONTRACTION_NONWORD_PREFIX_COST: i32 = 30;
/// `combined_cost` for an apostrophe-restored contraction when the typed letters
/// ARE also a valid word ("were"->"we're", "id"->"I'd", "cant"->"can't"). The
/// contraction is offered but priced so the real typed word stays competitive.
const CONTRACTION_WORD_PREFIX_COST: i32 = 200;

/// Frequency penalty in keyboard-edit-cost units. `u32::MAX` marks an unlisted word.
fn frequency_penalty(rank: u32) -> i32 {
    if rank == u32::MAX {
        UNKNOWN_FREQUENCY_PENALTY
    } else {
        (FREQ_LOG_WEIGHT * ((rank as f32) + 1.0).ln()).round() as i32
    }
}

fn local_use_bonus(local_frequency: usize) -> i32 {
    (local_frequency as i32)
        .saturating_mul(LOCAL_USE_BONUS_PER_COUNT)
        .min(MAX_LOCAL_USE_BONUS)
}

fn previous_word_bonus(previous_word_frequency: usize) -> i32 {
    (previous_word_frequency as i32)
        .saturating_mul(PREVIOUS_WORD_BONUS_PER_COUNT)
        .min(MAX_PREVIOUS_WORD_BONUS)
}

fn global_bigram_bonus(global_bigram_rank: u32) -> i32 {
    if global_bigram_rank == u32::MAX {
        return 0;
    }
    let scaled =
        GLOBAL_BIGRAM_BONUS_MAX * (1.0 - (global_bigram_rank as f32 / GLOBAL_BIGRAM_LIST_LEN));
    (scaled.round() as i32).max(GLOBAL_BIGRAM_BONUS_MIN)
}

/// Blended primary ranking cost (lower is better): keyboard-edit distance plus a
/// frequency penalty, minus bonuses for the typed word and document/collocation
/// context. This is the Gboard-style trade-off of "how likely is this word"
/// (frequency + context) against "how far is it from what was typed" (edit cost),
/// so a common word a fat-finger away can beat a rare exact or prefix match.
#[allow(clippy::too_many_arguments)]
fn combined_completion_cost(
    distance: u16,
    typed_word: bool,
    first_letter_match: bool,
    high_confidence_short_correction: bool,
    global_frequency_rank: u32,
    local_frequency: usize,
    previous_word_frequency: usize,
    global_bigram_rank: u32,
) -> i32 {
    (distance as i32)
        + frequency_penalty(global_frequency_rank)
        + if first_letter_match {
            0
        } else {
            FIRST_LETTER_MISMATCH_PENALTY
        }
        - if typed_word { TYPED_WORD_BONUS } else { 0 }
        - if high_confidence_short_correction {
            HIGH_CONFIDENCE_SHORT_CORRECTION_BONUS
        } else {
            0
        }
        - local_use_bonus(local_frequency)
        - previous_word_bonus(previous_word_frequency)
        - global_bigram_bonus(global_bigram_rank)
}

fn completion_sort_rank(
    query: &[char],
    candidate: &[char],
    prefix_score: CompletionPrefixScore,
    is_common: bool,
    context: &CompletionContext,
) -> CompletionSortRank {
    let normalized_candidate = normalized_completion_word(candidate);
    let previous_word_frequency = context.previous_pair_count(&normalized_candidate);

    let typed_word = same_word_ignore_case(query, candidate);
    let first_letter_match = first_letter_matches(query, candidate);
    // A confident short fat-finger fix: <=3 typed chars, a same-length common word
    // one edit away that isn't a prefix of what was typed (e.g. "teh"->"the").
    let high_confidence_short_correction = query.len() <= 3
        && candidate.len() == query.len()
        && !prefix_score.exact_prefix
        && prefix_score.distance <= INSERT_DELETE_COST
        && is_common;
    let local_frequency = context.local_count(&normalized_candidate);
    let global_bigram_rank = context
        .previous_word
        .as_ref()
        .and_then(|previous_word| bigram_rank(previous_word, &normalized_candidate))
        .unwrap_or(u32::MAX);
    let global_frequency_rank = frequency_rank(&normalized_candidate).unwrap_or(u32::MAX);

    CompletionSortRank {
        combined_cost: combined_completion_cost(
            prefix_score.distance,
            typed_word,
            first_letter_match,
            high_confidence_short_correction,
            global_frequency_rank,
            local_frequency,
            previous_word_frequency,
            global_bigram_rank,
        ),
        distance: prefix_score.distance,
        exact_prefix: prefix_score.exact_prefix,
        typed_word,
        first_letter_match,
        is_common,
        remaining_ambiguity: candidate
            .len()
            .saturating_sub(prefix_score.matched_prefix_len),
        replacement_len: candidate.len(),
    }
}

fn compare_completion_ranks(a: &CompletionSortRank, b: &CompletionSortRank) -> Ordering {
    // Primary key: the blended keyboard-edit-cost-plus-frequency score, which already
    // folds in the typed-word bonus and document/collocation context. This is what
    // lets a common word a fat-finger away (e.g. "rise" for "ruse", "response" for
    // "respine") outrank a rare exact or prefix match. The remaining comparisons only
    // break ties between candidates of equal blended cost.
    a.combined_cost
        .cmp(&b.combined_cost)
        // Prefer a true prefix of what was typed over an equal-cost fuzzy correction.
        .then_with(|| b.exact_prefix.cmp(&a.exact_prefix))
        .then_with(|| b.first_letter_match.cmp(&a.first_letter_match))
        .then_with(|| b.is_common.cmp(&a.is_common))
        // Prefer the shorter completion (less remaining ambiguity) at equal cost.
        .then_with(|| a.remaining_ambiguity.cmp(&b.remaining_ambiguity))
        .then_with(|| a.replacement_len.cmp(&b.replacement_len))
}

/// Builds the document-wide [`CompletionStats`] in one pass over the tokens.
/// Runs in `update_document`, so completion requests never rescan the document.
pub(crate) fn build_completion_stats(source: &[char], document: &Document) -> CompletionStats {
    let mut stats = CompletionStats::default();
    let mut previous_word: Option<String> = None;

    for token in document.tokens() {
        if !token.kind.is_word() {
            continue;
        }

        let normalized_word = normalized_completion_word(token.span.get_content(source));
        *stats
            .word_counts
            .entry(normalized_word.clone())
            .or_default() += 1;

        if let Some(previous_word) = previous_word.take() {
            *stats
                .bigram_counts
                .entry(previous_word)
                .or_default()
                .entry(normalized_word.clone())
                .or_default() += 1;
        }

        previous_word = Some(normalized_word);
    }

    stats
}

/// Computes the cursor word's contribution to the cached [`CompletionStats`]
/// (to be subtracted at query time) along with the word preceding the cursor.
/// Equivalent to excluding tokens overlapping `word_start..word_end` during a
/// full document scan, but costs O(log n) instead of O(document).
fn cursor_word_exclusions(
    source: &[char],
    document: &Document,
    word_start: usize,
    word_end: usize,
) -> (CursorExclusions, Option<String>) {
    let tokens = document.get_tokens();

    // Tokens are ordered and non-overlapping, so `span.end` is monotonic: every
    // token before this index ends at or before the cursor word.
    let run_start = tokens.partition_point(|token| token.span.end <= word_start);

    let previous_word = tokens[..run_start]
        .iter()
        .rev()
        .find(|token| token.kind.is_word())
        .map(|token| normalized_completion_word(token.span.get_content(source)));

    // Word tokens overlapping the cursor word (more than one for e.g. a
    // hyphenated word the tokenizer splits apart).
    let mut overlapping = Vec::new();
    let mut index = run_start;
    while let Some(token) = tokens.get(index) {
        if token.span.start >= word_end {
            break;
        }
        if token.kind.is_word() {
            overlapping.push(normalized_completion_word(token.span.get_content(source)));
        }
        index += 1;
    }

    // The full-document stats count the cursor word's occurrences and every
    // bigram into, within, and out of its token run; the old per-request scan
    // counted none of those, so they are exactly the exclusions.
    let mut exclusions = CursorExclusions::default();
    if let (Some(first), Some(last)) = (overlapping.first(), overlapping.last()) {
        if let Some(previous_word) = &previous_word {
            exclusions
                .pairs
                .push((previous_word.clone(), first.clone()));
        }

        let next_word = tokens[index..]
            .iter()
            .find(|token| token.kind.is_word())
            .map(|token| normalized_completion_word(token.span.get_content(source)));
        if let Some(next_word) = next_word {
            exclusions.pairs.push((last.clone(), next_word));
        }

        for pair in overlapping.windows(2) {
            exclusions.pairs.push((pair[0].clone(), pair[1].clone()));
        }

        exclusions.words = overlapping;
    }

    (exclusions, previous_word)
}

/// Checks whether `prefix` contains a space-adjacent key (v/b/n/m) that the
/// user may have pressed instead of the spacebar. Returns every position where
/// the characters before the split form a valid dictionary word and there are
/// at least `min_right_len` characters after it to complete.
fn find_missed_space_splits(
    prefix: &[char],
    dict: &dyn Dictionary,
    min_right_len: usize,
) -> Vec<(Vec<char>, Vec<char>)> {
    let mut splits = Vec::new();
    for i in 1..prefix.len() {
        let ch = prefix[i].to_lowercase().next().unwrap_or(prefix[i]);
        if !SPACE_ADJACENT_KEYS.contains(&ch) {
            continue;
        }
        let left = &prefix[..i];
        let right = &prefix[i + 1..];
        if right.len() < min_right_len {
            continue;
        }
        let left_lower = lowercase_chars(left);
        if dict.contains_exact_word(&left_lower) {
            splits.push((left.to_vec(), right.to_vec()));
        }
    }
    splits
}

/// Surfaces contractions typed without their apostrophe ("im"->"I'm",
/// "weve"->"we've", "dont"->"don't"). Phone users routinely skip the apostrophe,
/// and these words are absent from the unigram frequency list, so the normal
/// pipeline charges an apostrophe insertion and buries them. For each interior
/// position, inserts an apostrophe and asks the dictionary for the canonical
/// (correctly-capitalized) word; matches are scored cheaply so they rank near the
/// top — unless the typed letters are themselves a word, in which case the real
/// word is kept competitive.
fn add_contraction_completion_candidates(
    dict: &dyn Dictionary,
    prefix: &[char],
    misspelled_words: &HashSet<String>,
    seen: &mut HashSet<String>,
    completions: &mut Vec<RankedCompletion>,
    max_results: usize,
) {
    // The user already typed an apostrophe, or the token is too short to split.
    if prefix.len() < 2 || prefix.contains(&'\'') {
        return;
    }

    let lower = lowercase_chars(prefix);
    let prefix_is_word = dict.contains_exact_word(&lower);
    let cost = if prefix_is_word {
        CONTRACTION_WORD_PREFIX_COST
    } else {
        CONTRACTION_NONWORD_PREFIX_COST
    };

    for i in 1..lower.len() {
        let mut variant = Vec::with_capacity(lower.len() + 1);
        variant.extend_from_slice(&lower[..i]);
        variant.push('\'');
        variant.extend_from_slice(&lower[i..]);

        let Some(canonical) = dict
            .get_correct_capitalization_of(&variant)
            .map(<[char]>::to_vec)
        else {
            continue;
        };

        // Guard against a hash lookup returning a non-apostrophe word.
        if !canonical.contains(&'\'') {
            continue;
        }

        let normalized = normalized_completion_word(&canonical);
        if !seen.insert(normalized.clone()) || misspelled_words.contains(&normalized) {
            continue;
        }

        let is_common = dict
            .get_word_metadata(&canonical)
            .is_some_and(|metadata| metadata.common);

        insert_ranked_completion(
            completions,
            RankedCompletion {
                word: canonical.iter().collect(),
                rank: CompletionSortRank {
                    combined_cost: cost,
                    distance: 0,
                    exact_prefix: false,
                    typed_word: false,
                    first_letter_match: first_letter_matches(prefix, &canonical),
                    is_common,
                    remaining_ambiguity: 0,
                    replacement_len: canonical.len(),
                },
            },
            max_results,
        );
    }
}

fn rank_completion_candidates(
    dict: &dyn Dictionary,
    prefix: &[char],
    context: &CompletionContext,
    misspelled_words: &HashSet<String>,
    max_results: usize,
) -> Vec<RankedCompletion> {
    if max_results == 0 {
        return Vec::new();
    }

    let mut seen = HashSet::new();
    let mut completions = Vec::new();
    let exact_prefix_limit = exact_prefix_candidate_limit(max_results);
    let noisy_expansion_limit = noisy_prefix_expansion_candidate_limit(max_results);
    let noisy_expansion_per_prefix_limit = noisy_prefix_expansion_per_prefix_limit(max_results);
    let mut noisy_expansion_count = 0;

    add_exact_prefix_completion_candidates(
        dict,
        prefix,
        prefix,
        context,
        misspelled_words,
        &mut seen,
        &mut completions,
        max_results,
        exact_prefix_limit,
    );

    let lowercase_prefix = lowercase_chars(prefix);
    if lowercase_prefix != prefix {
        add_exact_prefix_completion_candidates(
            dict,
            &lowercase_prefix,
            prefix,
            context,
            misspelled_words,
            &mut seen,
            &mut completions,
            max_results,
            exact_prefix_limit,
        );
    }

    add_frequent_prefix_completion_candidates(
        dict,
        prefix,
        context,
        misspelled_words,
        &mut seen,
        &mut completions,
        max_results,
        frequent_prefix_candidate_limit(max_results),
    );

    add_contraction_completion_candidates(
        dict,
        prefix,
        misspelled_words,
        &mut seen,
        &mut completions,
        max_results,
    );

    for fuzzy_match in dict.fuzzy_match(
        prefix,
        allowed_noisy_prefix_edits(prefix.len()) as u8,
        fuzzy_completion_candidate_limit(max_results),
    ) {
        if !starts_with_ignore_case(fuzzy_match.word, prefix)
            && !should_score_noisy_prefix_candidate(prefix, fuzzy_match.word)
        {
            continue;
        }

        score_completion_candidate(
            dict,
            prefix,
            context,
            misspelled_words,
            &mut seen,
            &mut completions,
            fuzzy_match.word,
            max_results,
        );

        if !starts_with_ignore_case(fuzzy_match.word, prefix)
            && noisy_expansion_count < noisy_expansion_limit
        {
            let remaining_limit = noisy_expansion_limit - noisy_expansion_count;
            noisy_expansion_count += add_noisy_prefix_expansion_candidates(
                dict,
                prefix,
                fuzzy_match.word,
                context,
                misspelled_words,
                &mut seen,
                &mut completions,
                max_results,
                noisy_expansion_per_prefix_limit.min(remaining_limit),
            );
        }
    }

    completions
}

fn exact_prefix_candidate_limit(max_results: usize) -> usize {
    max_results
        .saturating_mul(32)
        .clamp(MIN_COMPLETION_CANDIDATE_POOL, MAX_EXACT_PREFIX_CANDIDATES)
}

fn fuzzy_completion_candidate_limit(max_results: usize) -> usize {
    max_results.saturating_mul(24).clamp(
        MIN_COMPLETION_CANDIDATE_POOL,
        MAX_FUZZY_COMPLETION_CANDIDATES,
    )
}

fn noisy_prefix_expansion_candidate_limit(max_results: usize) -> usize {
    max_results.saturating_mul(16).clamp(
        MIN_COMPLETION_CANDIDATE_POOL / 2,
        MAX_NOISY_PREFIX_EXPANSION_CANDIDATES,
    )
}

fn noisy_prefix_expansion_per_prefix_limit(max_results: usize) -> usize {
    max_results.saturating_mul(2).clamp(4, 16)
}

fn frequent_prefix_candidate_limit(max_results: usize) -> usize {
    max_results.saturating_mul(2).clamp(32, 128)
}

fn add_exact_prefix_completion_candidates(
    dict: &dyn Dictionary,
    prefix_lookup: &[char],
    ranking_prefix: &[char],
    context: &CompletionContext,
    misspelled_words: &HashSet<String>,
    seen: &mut HashSet<String>,
    completions: &mut Vec<RankedCompletion>,
    max_results: usize,
    candidate_limit: usize,
) {
    for word in dict.find_words_with_prefix_limited(prefix_lookup, candidate_limit) {
        if !starts_with_ignore_case(word.as_ref(), ranking_prefix) {
            continue;
        }

        score_completion_candidate(
            dict,
            ranking_prefix,
            context,
            misspelled_words,
            seen,
            completions,
            word.as_ref(),
            max_results,
        );
    }
}

fn add_frequent_prefix_completion_candidates(
    dict: &dyn Dictionary,
    prefix: &[char],
    context: &CompletionContext,
    misspelled_words: &HashSet<String>,
    seen: &mut HashSet<String>,
    completions: &mut Vec<RankedCompletion>,
    max_results: usize,
    candidate_limit: usize,
) {
    let prefix_string: String = prefix.iter().collect();

    for word in most_frequent_with_prefix(&prefix_string, candidate_limit) {
        let word_chars: Vec<char> = word.chars().collect();

        // Only surface words present in the active dictionary so the frequency list
        // never introduces non-words (e.g. "http") as completions.
        if !dict.contains_exact_word(&word_chars) {
            continue;
        }

        score_completion_candidate(
            dict,
            prefix,
            context,
            misspelled_words,
            seen,
            completions,
            &word_chars,
            max_results,
        );
    }
}

fn add_noisy_prefix_expansion_candidates(
    dict: &dyn Dictionary,
    ranking_prefix: &[char],
    corrected_prefix: &[char],
    context: &CompletionContext,
    misspelled_words: &HashSet<String>,
    seen: &mut HashSet<String>,
    completions: &mut Vec<RankedCompletion>,
    max_results: usize,
    candidate_limit: usize,
) -> usize {
    let mut considered = 0;

    for word in dict.find_words_with_prefix_limited(corrected_prefix, candidate_limit) {
        considered += 1;

        if same_word_ignore_case(word.as_ref(), corrected_prefix) {
            continue;
        }

        score_completion_candidate(
            dict,
            ranking_prefix,
            context,
            misspelled_words,
            seen,
            completions,
            word.as_ref(),
            max_results,
        );
    }

    considered
}

fn score_completion_candidate(
    dict: &dyn Dictionary,
    prefix: &[char],
    context: &CompletionContext,
    misspelled_words: &HashSet<String>,
    seen: &mut HashSet<String>,
    completions: &mut Vec<RankedCompletion>,
    word: &[char],
    max_results: usize,
) {
    let normalized_word = normalized_completion_word(word);
    if !seen.insert(normalized_word.clone()) {
        return;
    }

    if misspelled_words.contains(&normalized_word) {
        return;
    }

    let Some(prefix_score) = best_completion_prefix_score(prefix, word) else {
        return;
    };

    let is_common = dict
        .get_word_metadata(word)
        .is_some_and(|metadata| metadata.common);
    let rank = completion_sort_rank(prefix, word, prefix_score, is_common, context);

    insert_ranked_completion(
        completions,
        RankedCompletion {
            word: word.iter().collect(),
            rank,
        },
        max_results,
    );
}

fn compare_ranked_completion(a: &RankedCompletion, b: &RankedCompletion) -> Ordering {
    compare_completion_ranks(&a.rank, &b.rank).then_with(|| a.word.cmp(&b.word))
}

fn insert_ranked_completion(
    completions: &mut Vec<RankedCompletion>,
    completion: RankedCompletion,
    max_results: usize,
) {
    let insert_at = completions
        .binary_search_by(|existing| compare_ranked_completion(existing, &completion))
        .unwrap_or_else(|idx| idx);

    if insert_at >= max_results {
        return;
    }

    completions.insert(insert_at, completion);
    completions.truncate(max_results);
}

fn should_commit_space(completions: &[RankedCompletion]) -> bool {
    let Some(top) = completions.first() else {
        return false;
    };

    if top.rank.typed_word || top.rank.distance > INSERT_DELETE_COST {
        return false;
    }

    let Some(runner_up) = completions.get(1) else {
        return true;
    };

    top.rank.distance.saturating_add(INSERT_DELETE_COST / 2) < runner_up.rank.distance
}

fn should_item_commit_space(
    item_index: usize,
    completions: &[RankedCompletion],
    commit_with_space: CompletionCommitWithSpace,
) -> bool {
    match commit_with_space {
        CompletionCommitWithSpace::Never => false,
        CompletionCommitWithSpace::Confident => item_index == 0 && should_commit_space(completions),
        CompletionCommitWithSpace::Always => item_index < completions.len(),
    }
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

fn is_completion_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '\'' || c == '-'
}

fn find_completion_word_bounds(source: &[char], cursor_index: usize) -> (usize, usize) {
    let cursor_index = cursor_index.min(source.len());

    let word_start = source[..cursor_index]
        .iter()
        .rposition(|c| !is_completion_word_char(*c))
        .map(|i| i + 1)
        .unwrap_or(0);

    let suffix_len = source[cursor_index..]
        .iter()
        .position(|c| !is_completion_word_char(*c))
        .unwrap_or(source.len() - cursor_index);
    let word_end = cursor_index + suffix_len;

    (word_start, word_end)
}

/// Token at the completion cursor.
///
/// LSP positions sit at the *exclusive* end of the token being typed. Harper
/// spans are also exclusive, so `get_token_at_char_index(cursor)` returns
/// `None` at EOF and at the end of a word. Fall back to the character
/// immediately before the cursor in that case.
fn token_at_completion_cursor(
    document: &Document,
    cursor_index: usize,
) -> Option<&harper_core::Token> {
    document.get_token_at_char_index(cursor_index).or_else(|| {
        cursor_index
            .checked_sub(1)
            .and_then(|index| document.get_token_at_char_index(index))
    })
}

fn is_cursor_in_lintable_region(document: &Document, cursor_index: usize) -> bool {
    token_at_completion_cursor(document, cursor_index)
        .is_some_and(|token| !matches!(token.kind, TokenKind::Unlintable))
}

struct SplitRankedItem {
    left_chars: Vec<char>,
    right_prefix: Vec<char>,
    right_word: String,
}

fn rank_completions_and_splits(
    dict: &dyn Dictionary,
    prefix: &[char],
    context: &CompletionContext,
    misspelled_words: &HashSet<String>,
    max_results: usize,
    split_min_right: usize,
) -> (Vec<RankedCompletion>, Vec<SplitRankedItem>) {
    let ranking_limit = max_results.max(2);
    let completions =
        rank_completion_candidates(dict, prefix, context, misspelled_words, ranking_limit);

    let mut splits = Vec::new();
    'splits: for (left_chars, right_prefix) in
        find_missed_space_splits(prefix, dict, split_min_right)
    {
        let left_lower: String = lowercase_chars(&left_chars).iter().collect();
        let split_context = CompletionContext {
            stats: context.stats.clone(),
            exclusions: context.exclusions.clone(),
            previous_word: Some(left_lower),
        };
        let right_completions = rank_completion_candidates(
            dict,
            &right_prefix,
            &split_context,
            misspelled_words,
            ranking_limit,
        );
        for ranked in right_completions.into_iter().take(max_results) {
            if completions.len() + splits.len() >= 2 * max_results {
                break 'splits;
            }
            splits.push(SplitRankedItem {
                left_chars: left_chars.clone(),
                right_prefix: right_prefix.clone(),
                right_word: ranked.word,
            });
        }
    }

    (completions, splits)
}

impl Backend {
    pub fn new(client: Client, config: Config) -> Self {
        Self {
            client,
            root: Arc::new(RwLock::new(".".into())),
            stats: Arc::new(RwLock::new(Stats::new())),
            config: Arc::new(RwLock::new(config)),
            doc_state: Arc::new(Mutex::new(HashMap::new())),
            pending_changes: Arc::new(RwLock::new(HashMap::new())),
            dict_cache: Arc::new(RwLock::new(HashMap::new())),
            in_flight_changes: Arc::new(RwLock::new(HashMap::new())),
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

                // Build the line index for O(1) position conversions, and refresh the
                // shared source snapshot + completion stats so completion requests
                // don't copy or rescan the document.
                let source: Arc<Vec<char>> =
                    Arc::new(doc_state.document.get_source().iter().copied().collect());
                doc_state.line_index = crate::pos_conv::LineIndex::new(source.as_slice());
                doc_state.completion_stats = Arc::new(build_completion_stats(
                    source.as_slice(),
                    &doc_state.document,
                ));
                doc_state.source = source;
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

    async fn run_debounced_diagnostics(&self, uri: Uri, now: Instant) {
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

        let should_process = {
            let pending = self.pending_changes.read().await;
            pending.get(&uri).is_some_and(|&timestamp| timestamp == now)
        };

        if !should_process {
            return;
        }

        self.publish_diagnostics(&uri).await;

        let mut pending = self.pending_changes.write().await;
        if pending.get(&uri).is_some_and(|&timestamp| timestamp == now) {
            pending.remove(&uri);
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
        let (source, dict, line_index, misspelled_words, word_start, word_end, prefix, context) = {
            let doc_states = self.doc_state.lock().await;
            let Some(doc_state) = doc_states.get(uri) else {
                return Ok(Vec::new());
            };
            // Snapshot maintained by `update_document`; cloning the Arc avoids copying
            // the document text on every request.
            let source = doc_state.source.clone();

            // Convert LSP position to character index while holding the lock so we can
            // check the token kind before releasing it.
            let cursor_index = doc_state
                .line_index
                .position_to_index(source.as_slice(), position);

            // Only complete inside lintable regions (comments/docstrings/prose).
            // Code is Unlintable. Cursor usually sits at the exclusive end of the
            // word being typed (EOF on the last line), so look at the previous
            // character when there is no token *at* the cursor.
            if !is_cursor_in_lintable_region(&doc_state.document, cursor_index) {
                return Ok(Vec::new());
            }

            let (word_start, word_end) =
                find_completion_word_bounds(source.as_slice(), cursor_index);
            let prefix: Vec<char> = source[word_start..cursor_index].to_vec();
            let (exclusions, previous_word) = cursor_word_exclusions(
                source.as_slice(),
                &doc_state.document,
                word_start,
                word_end,
            );
            let context = CompletionContext {
                stats: doc_state.completion_stats.clone(),
                exclusions,
                previous_word,
            };

            (
                source,
                doc_state.completion_dict.clone(),
                doc_state.line_index.clone(),
                doc_state.misspelled_words.clone(),
                word_start,
                word_end,
                prefix,
                context,
            )
        }; // Lock released here

        // Don't show completions for very short prefixes
        if prefix.len() < completion_config.min_prefix_length {
            return Ok(Vec::new());
        }

        if completion_config.max_results == 0 {
            return Ok(Vec::new());
        }

        // Rank off the async worker so fuzzy matching cannot stall didChange.
        let split_min_right = completion_config.min_prefix_length.max(2);
        let max_results = completion_config.max_results;
        let commit_with_space = completion_config.commit_with_space;
        let ranking_dict = dict.clone();
        let ranking_prefix = prefix.clone();
        let ranking_context = context.clone();
        let ranking_misspelled = misspelled_words.clone();
        let (completions, splits) = tokio::task::spawn_blocking(move || {
            rank_completions_and_splits(
                ranking_dict.as_ref(),
                &ranking_prefix,
                &ranking_context,
                ranking_misspelled.as_ref(),
                max_results,
                split_min_right,
            )
        })
        .await
        .map_err(|err| {
            error!("completion ranking task failed: {err}");
            tower_lsp_server::jsonrpc::Error::internal_error()
        })?;

        let space_commit_flags: Vec<bool> = (0..max_results.min(completions.len()))
            .map(|idx| should_item_commit_space(idx, &completions, commit_with_space))
            .collect();

        // Calculate the start position of the word being completed
        let word_start_position = line_index.index_to_position(source.as_slice(), word_start);
        let word_end_position = line_index.index_to_position(source.as_slice(), word_end);

        // Convert to LSP completion items
        // Use text_edit to specify exact replacement range - this tells Helix what to replace
        // filter_text is the candidate (not the typed prefix). Helix fuzzy-matches the
        // growing insert-mode filter against filter_text; using the prefix made every
        // item vanish as soon as the user typed one more character.
        let completion_items: Vec<CompletionItem> = completions
            .into_iter()
            .take(max_results)
            .enumerate()
            .map(|(idx, completion)| {
                // Apply the casing from the user's prefix to the completion
                let word_string = completion.word;
                let completion_text = apply_prefix_casing(&prefix, &word_string);

                CompletionItem {
                    label: word_string.clone(),
                    kind: Some(CompletionItemKind::TEXT),
                    detail: Some("Harper".to_string()),
                    // Replace the full word under/around cursor.
                    text_edit: Some(CompletionTextEdit::Edit(TextEdit {
                        range: Range {
                            start: word_start_position,
                            end: word_end_position,
                        },
                        new_text: completion_text,
                    })),
                    filter_text: Some(word_string),
                    sort_text: Some(format!("{:05}", idx)),
                    commit_characters: space_commit_flags[idx].then(|| vec![" ".to_string()]),
                    ..Default::default()
                }
            })
            .collect();

        // Missed-space split completions: v/b/n/m pressed instead of space fuses
        // two words into one token (e.g. "thenworld" → offer "the world").
        // These are appended after normal completions so they don't displace exact
        // single-word completions, but they surface as an option the user can pick.
        let normal_count = completion_items.len();
        let split_items: Vec<CompletionItem> = splits
            .into_iter()
            .enumerate()
            .map(|(split_idx, split)| {
                let left_lower: String = lowercase_chars(&split.left_chars).iter().collect();
                let left_display =
                    apply_prefix_casing(&prefix[..split.left_chars.len()], &left_lower);
                let right_display = apply_prefix_casing(&split.right_prefix, &split.right_word);
                let combined_text = format!("{left_display} {right_display}");
                let combined_label = format!("{left_lower} {}", split.right_word);
                // Concatenate without space so Helix's filter ("thenworld") still
                // matches the two-word candidate ("theworld").
                let filter_text = format!("{left_lower}{}", split.right_word);
                let sort_idx = normal_count + split_idx;
                CompletionItem {
                    label: combined_label,
                    kind: Some(CompletionItemKind::TEXT),
                    detail: Some("Harper".to_string()),
                    text_edit: Some(CompletionTextEdit::Edit(TextEdit {
                        range: Range {
                            start: word_start_position,
                            end: word_end_position,
                        },
                        new_text: combined_text,
                    })),
                    filter_text: Some(filter_text),
                    sort_text: Some(format!("{:05}", sort_idx)),
                    ..Default::default()
                }
            })
            .collect();

        let mut all_items = completion_items;
        all_items.extend(split_items);
        Ok(all_items)
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

        // Record this change with current timestamp for debounced diagnostics.
        // Sleep/lint off the notification handler so completions are not queued
        // behind 300ms of debounce per keystroke.
        let now = Instant::now();
        {
            let mut pending = self.pending_changes.write().await;
            pending.insert(uri.clone(), now);
        }

        let backend = self.clone();
        tokio::spawn(async move {
            backend.run_debounced_diagnostics(uri, now).await;
        });
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

        // Always an incomplete list. Helix maps `None` / a complete empty list
        // to "stop asking", which kills the popup after a single miss (EOF,
        // unlintable, short prefix).
        use tower_lsp_server::lsp_types::CompletionList;
        Ok(Some(CompletionResponse::List(CompletionList {
            is_incomplete: true,
            items: completions,
        })))
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

    fn dict_with_words(words: &[(&str, bool)]) -> MutableDictionary {
        let mut dict = MutableDictionary::new();
        for (word, common) in words {
            let mut metadata = DictWordMetadata::default();
            metadata.common = *common;
            dict.append_word_str(word, metadata);
        }
        dict
    }

    fn ranked_words(
        query: &str,
        dict: &dyn Dictionary,
        context: &CompletionContext,
    ) -> Vec<String> {
        let misspelled_words = HashSet::new();
        rank_completion_candidates(dict, &chars(query), context, &misspelled_words, 50)
            .into_iter()
            .map(|completion| completion.word)
            .collect()
    }

    #[test]
    fn test_completion_word_bounds_include_suffix_after_cursor() {
        let source = chars("hello worlxd!");
        let cursor_index = "hello wo".chars().count();

        let (word_start, word_end) = find_completion_word_bounds(&source, cursor_index);

        assert_eq!(word_start, 6);
        assert_eq!(word_end, 12);
    }

    #[test]
    fn test_completion_word_bounds_at_word_end() {
        let source = chars("hello wo!");
        let cursor_index = "hello wo".chars().count();

        let (word_start, word_end) = find_completion_word_bounds(&source, cursor_index);

        assert_eq!(word_start, 6);
        assert_eq!(word_end, 8);
    }

    #[test]
    fn test_completion_word_bounds_include_apostrophe_and_dash() {
        let source = chars("it's re-entering.");
        let cursor_index = "it's re".chars().count();

        let (word_start, word_end) = find_completion_word_bounds(&source, cursor_index);

        assert_eq!(word_start, 5);
        assert_eq!(word_end, 16);
    }

    #[test]
    fn cursor_at_eof_after_last_word_is_lintable() {
        let document = Document::new_plain_english_curated("hello world");
        let eof = "hello world".chars().count();

        assert!(is_cursor_in_lintable_region(&document, eof));
        // Exclusive end of "world" is also EOF here.
        assert!(token_at_completion_cursor(&document, eof).is_some());
    }

    #[test]
    fn cursor_at_exclusive_end_of_word_before_newline_is_lintable() {
        let document = Document::new_plain_english_curated("hello\n");
        let word_end = "hello".chars().count();

        assert!(is_cursor_in_lintable_region(&document, word_end));
    }

    #[test]
    fn empty_document_is_not_lintable() {
        let document = Document::new_plain_english_curated("");

        assert!(!is_cursor_in_lintable_region(&document, 0));
        assert!(token_at_completion_cursor(&document, 0).is_none());
    }

    #[test]
    fn unlintable_markdown_code_span_is_skipped() {
        let document = Document::new_markdown_default_curated("see `abc` now");
        let inside_code = "see `ab".chars().count();

        assert!(!is_cursor_in_lintable_region(&document, inside_code));
    }

    #[test]
    fn pred_prefers_prefix_completions_over_short_fuzzy_words() {
        let dict = dict_with_words(&[
            ("prod", true),
            ("predict", true),
            ("prediction", true),
            ("prey", true),
        ]);
        let words = ranked_words("pred", &dict, &CompletionContext::default());
        let position = |target: &str| words.iter().position(|word| word == target).unwrap();

        // Exact-prefix completions rank ahead of fuzzy matches like "prod"/"prey".
        assert!(position("predict") < position("prod"));
        assert!(position("prediction") < position("prod"));
        assert!(position("predict") < position("prey"));
        assert!(position("prediction") < position("prey"));
        // Between the two completions, global frequency decides: "prediction" is more
        // frequent than "predict" in the bundled list, so it ranks first.
        assert!(position("prediction") < position("predict"));
    }

    #[test]
    fn uppercase_prefix_uses_exact_prefix_candidates() {
        let dict = dict_with_words(&[("predict", true), ("prediction", true), ("prod", true)]);
        let words = ranked_words("Pred", &dict, &CompletionContext::default());
        let position = |target: &str| words.iter().position(|word| word == target).unwrap();

        // The uppercase prefix still surfaces the exact-prefix completions ahead of
        // the fuzzy "prod"; frequency orders "prediction" before "predict".
        assert!(position("predict") < position("prod"));
        assert!(position("prediction") < position("prod"));
        assert!(position("prediction") < position("predict"));
    }

    #[test]
    fn ranking_limits_returned_candidates() {
        let dict = dict_with_words(&[
            ("abacus", true),
            ("abandon", true),
            ("abate", true),
            ("abbey", true),
        ]);
        let misspelled_words = HashSet::new();
        let completions = rank_completion_candidates(
            &dict,
            &chars("ab"),
            &CompletionContext::default(),
            &misspelled_words,
            2,
        );

        assert_eq!(completions.len(), 2);
    }

    #[test]
    fn transposed_teh_ranks_the_first() {
        let dict = dict_with_words(&[
            ("ten", true),
            ("the", true),
            ("tech", true),
            ("Teheran", false),
        ]);
        let words = ranked_words("teh", &dict, &CompletionContext::default());

        assert_eq!(words[0], "the");
    }

    #[test]
    fn nearby_first_letter_typo_ranks_common_short_word_above_prefix_expansions() {
        let dict = dict_with_words(&[
            ("Rhea", false),
            ("rheum", false),
            ("rhetoric", false),
            ("the", true),
        ]);
        let words = ranked_words("rhe", &dict, &CompletionContext::default());

        assert_eq!(words[0], "the");
    }

    #[test]
    fn whie_ranks_while_first() {
        let dict = dict_with_words(&[("white", true), ("while", true), ("whale", true)]);
        let words = ranked_words("whie", &dict, &CompletionContext::default());

        assert_eq!(words[0], "while");
    }

    #[test]
    fn rigth_ranks_right_first() {
        let dict = dict_with_words(&[("rigid", true), ("right", true), ("righteous", true)]);
        let words = ranked_words("rigth", &dict, &CompletionContext::default());

        assert_eq!(words[0], "right");
    }

    #[test]
    fn noisy_prefix_includes_long_prefix_expansions() {
        let dict = dict_with_words(&[("right", true), ("righteous", true), ("rightful", true)]);
        let words = ranked_words("rigth", &dict, &CompletionContext::default());

        assert!(words.iter().any(|word| word == "righteous"));
        assert!(words.iter().any(|word| word == "rightful"));
    }

    #[test]
    fn nearby_key_typo_beats_distant_key_alternative() {
        let dict = dict_with_words(&[("hello", false), ("pello", false)]);
        let words = ranked_words("gello", &dict, &CompletionContext::default());

        assert_eq!(words[0], "hello");
    }

    #[test]
    fn valid_typed_word_stays_on_top_and_does_not_commit_space() {
        let dict = dict_with_words(&[("in", true), ("inn", true), ("inside", true)]);
        let misspelled_words = HashSet::new();
        let completions = rank_completion_candidates(
            &dict,
            &chars("in"),
            &CompletionContext::default(),
            &misspelled_words,
            50,
        );

        assert_eq!(completions[0].word, "in");
        assert!(!should_commit_space(&completions));
    }

    #[test]
    fn local_frequency_boosts_repeated_document_words() {
        let dict = dict_with_words(&[("food", false), ("fool", false)]);
        let mut stats = CompletionStats::default();
        stats.word_counts.insert("food".to_string(), 3);
        let context = CompletionContext {
            stats: Arc::new(stats),
            ..Default::default()
        };

        let words = ranked_words("foo", &dict, &context);

        assert_eq!(words[0], "food");
    }

    #[test]
    fn previous_word_context_breaks_close_ties() {
        let dict = dict_with_words(&[("food", false), ("fool", false)]);
        let mut stats = CompletionStats::default();
        stats
            .bigram_counts
            .entry("eat".to_string())
            .or_default()
            .insert("fool".to_string(), 2);
        let context = CompletionContext {
            stats: Arc::new(stats),
            previous_word: Some("eat".to_string()),
            ..Default::default()
        };

        let words = ranked_words("foo", &dict, &context);

        assert_eq!(words[0], "fool");
    }

    /// The old per-request context scan, kept as the behavioral reference for
    /// the cached-stats + cursor-exclusions path.
    #[allow(clippy::type_complexity)]
    fn reference_context(
        source: &[char],
        document: &Document,
        current_word_start: usize,
        current_word_end: usize,
    ) -> (
        HashMap<String, usize>,
        HashMap<(String, String), usize>,
        Option<String>,
    ) {
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        let mut previous_word = None;
        let mut previous_non_current_word: Option<String> = None;

        for token in document.tokens() {
            if !token.kind.is_word() {
                continue;
            }

            let normalized_word = normalized_completion_word(token.span.get_content(source));
            if token.span.end <= current_word_start {
                previous_word = Some(normalized_word.clone());
            }

            let overlaps_current_word =
                token.span.start < current_word_end && current_word_start < token.span.end;
            if overlaps_current_word {
                previous_non_current_word = None;
                continue;
            }

            *word_counts.entry(normalized_word.clone()).or_default() += 1;

            if let Some(previous) = &previous_non_current_word {
                *pair_counts
                    .entry((previous.clone(), normalized_word.clone()))
                    .or_default() += 1;
            }

            previous_non_current_word = Some(normalized_word);
        }

        (word_counts, pair_counts, previous_word)
    }

    /// Asserts the cached stats + exclusions produce the same effective counts
    /// and previous word as a full rescan that skips the cursor word.
    fn assert_cached_context_matches_rescan(text: &str, cursor_index: usize) {
        let document = Document::new_plain_english_curated(text);
        let source: Vec<char> = text.chars().collect();
        let (word_start, word_end) = find_completion_word_bounds(&source, cursor_index);

        let stats = Arc::new(build_completion_stats(&source, &document));
        let (exclusions, previous_word) =
            cursor_word_exclusions(&source, &document, word_start, word_end);
        let context = CompletionContext {
            stats,
            exclusions,
            previous_word,
        };

        let (reference_words, reference_pairs, reference_previous) =
            reference_context(&source, &document, word_start, word_end);

        assert_eq!(
            context.previous_word, reference_previous,
            "previous word for {text:?}"
        );

        // The reference counts are a subset of the full-document stats, so
        // iterating the stats' support compares every count in either map.
        for word in context.stats.word_counts.keys() {
            assert_eq!(
                context.local_count(word),
                reference_words.get(word).copied().unwrap_or(0),
                "unigram count for {word:?} in {text:?}"
            );
        }

        for (previous, following) in &context.stats.bigram_counts {
            for word in following.keys() {
                assert_eq!(
                    context.pair_count(previous, word),
                    reference_pairs
                        .get(&(previous.clone(), word.clone()))
                        .copied()
                        .unwrap_or(0),
                    "pair count for ({previous:?}, {word:?}) in {text:?}"
                );
            }
        }
    }

    #[test]
    fn cached_stats_match_rescan_while_typing_at_document_end() {
        let text = "the quick fox ate the food. The fox ate qui";
        assert_cached_context_matches_rescan(text, text.chars().count());
    }

    #[test]
    fn cached_stats_match_rescan_with_cursor_mid_document() {
        let text = "alpha beta gamma delta beta gamma";
        // Cursor inside the first "gamma".
        assert_cached_context_matches_rescan(text, "alpha beta gam".chars().count());
    }

    #[test]
    fn cached_stats_match_rescan_for_multi_token_cursor_word() {
        // The completion word bounds treat "re-entering" as one word, but the
        // tokenizer may split it into several word tokens — the exclusion logic
        // must remove the whole run and its boundary bigrams.
        let text = "we keep re-entering the zone before re-entering";
        assert_cached_context_matches_rescan(text, text.chars().count());
    }

    #[test]
    fn cursor_word_does_not_boost_itself() {
        let text = "food is good food foo";
        let document = Document::new_plain_english_curated(text);
        let source: Vec<char> = text.chars().collect();
        let (word_start, word_end) = find_completion_word_bounds(&source, source.len());

        let stats = Arc::new(build_completion_stats(&source, &document));
        let (exclusions, previous_word) =
            cursor_word_exclusions(&source, &document, word_start, word_end);
        let context = CompletionContext {
            stats,
            exclusions,
            previous_word,
        };

        assert_eq!(context.local_count("foo"), 0);
        assert_eq!(context.local_count("food"), 2);
        assert_eq!(context.previous_word.as_deref(), Some("food"));
    }

    #[test]
    fn global_frequency_orders_equal_distance_completions() {
        // work/word/worn/worm are all equal-distance exact-prefix matches of "wor"
        // and all flagged common, so only global usage frequency separates them.
        let dict = dict_with_words(&[
            ("work", true),
            ("word", true),
            ("worm", true),
            ("worn", true),
        ]);
        let words = ranked_words("wor", &dict, &CompletionContext::default());
        let position = |target: &str| words.iter().position(|word| word == target).unwrap();

        assert_eq!(words[0], "work");
        assert!(position("work") < position("word"));
        assert!(position("word") < position("worn"));
        assert!(position("worn") < position("worm"));
    }

    #[test]
    fn frequent_longer_word_outranks_rare_shorter_word() {
        // "world" (5 letters, extremely common) must beat the shorter "worm"/"worn"
        // (4 letters, rare). Before P2 the per-character length penalty buried the
        // longer word; now frequency wins over the shortest-first bias.
        let dict = dict_with_words(&[
            ("worm", true),
            ("worn", true),
            ("work", true),
            ("world", true),
        ]);
        let words = ranked_words("wor", &dict, &CompletionContext::default());
        let position = |target: &str| words.iter().position(|word| word == target).unwrap();

        assert_eq!(words[0], "world");
        assert!(position("world") < position("worm"));
        assert!(position("world") < position("worn"));
    }

    #[test]
    fn global_bigram_boosts_context_word_over_more_frequent_unigram() {
        let dict = dict_with_words(&[
            ("you", true),
            ("your", true),
            ("york", true),
            ("young", true),
            ("youth", true),
        ]);

        // After "new", the strong global bigram "new york" lifts the rarer unigram
        // "york" to the top for prefix "yo".
        let after_new = CompletionContext {
            previous_word: Some("new".to_string()),
            ..Default::default()
        };
        let words = ranked_words("yo", &dict, &after_new);
        assert_eq!(words[0], "york");

        // Without that context, plain unigram frequency leads with "you" — proving the
        // reordering comes from the bigram signal, not from "york" itself.
        let no_context = ranked_words("yo", &dict, &CompletionContext::default());
        assert_eq!(no_context[0], "you");
    }

    #[test]
    fn fat_finger_adjacent_key_typos_rank_intended_word_first() {
        // Characterization corpus for adjacent-key "fat-finger" typos: one key replaced
        // by a physical neighbor on the phone QWERTY layout. Pins current behavior so a
        // future keyboard cost-model tweak (keyboard_distance.rs) can't silently regress
        // it. Each typo should make its intended word the top suggestion.
        let dict = FstDictionary::curated();
        let cases = [
            ("tge", "the"),         // g <-> h
            ("wprk", "work"),       // o <-> p
            ("wirk", "work"),       // i <-> o
            ("abiut", "about"),     // o <-> i
            ("tjis", "this"),       // h <-> j
            ("wuth", "with"),       // i <-> u
            ("fimd", "find"),       // n <-> m
            ("yhe", "the"),         // t <-> y
            ("soace", "space"),     // o <-> p
            ("befpre", "before"),   // o <-> p
            ("vould", "could"),     // c <-> v
            ("shoukd", "should"),   // l <-> k
            ("wjat", "what"),       // h <-> j
            ("snd", "and"),         // a <-> s
            ("becauae", "because"), // s <-> a
        ];

        let mut failures = Vec::new();
        for (typo, expected) in cases {
            let words = ranked_words(typo, dict.as_ref(), &CompletionContext::default());
            if words.first().map(String::as_str) != Some(expected) {
                let top5: Vec<&String> = words.iter().take(5).collect();
                failures.push(format!(
                    "{typo:?} -> expected {expected:?} first, got {top5:?}"
                ));
            }
        }

        assert!(
            failures.is_empty(),
            "adjacent-key typos not ranked #1:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn frequency_seeding_surfaces_common_words_past_the_alphabetical_window() {
        // In the real dictionary, "could"/"come" sort far past the start of the "co"
        // words, so the bounded alphabetical fetch alone misses them. Frequency seeding
        // pulls these very common words into the candidate pool for prefix "co".
        let dict = FstDictionary::curated();
        let words = ranked_words("co", dict.as_ref(), &CompletionContext::default());

        assert!(words.iter().any(|word| word == "could"));
        assert!(words.iter().any(|word| word == "come"));
    }

    #[test]
    fn common_neighbor_outranks_rare_typed_word() {
        // "ruse" is a valid but rare word; "rise" is one near-key edit away (u <-> i)
        // and far more common. Blended scoring surfaces the common neighbor first while
        // keeping the typed word (and the "rues" transposition) lower in the list.
        let dict = FstDictionary::curated();
        let words = ranked_words("ruse", dict.as_ref(), &CompletionContext::default());
        let top: Vec<&String> = words.iter().take(6).collect();
        let pos = |t: &str| words.iter().position(|w| w == t);

        assert_eq!(
            words.first().map(String::as_str),
            Some("rise"),
            "expected 'rise' first, got {top:?}"
        );
        assert!(
            pos("rise") < pos("ruse"),
            "'rise' should outrank the rarer typed 'ruse': {top:?}"
        );
        assert!(
            pos("rise") < pos("rues"),
            "'rise' should outrank the transposition 'rues': {top:?}"
        );
        // The first-letter-mismatch penalty keeps the same-first-letter "rise" ahead
        // of the more common but first-letter-changing "use" correction.
        if let (Some(rise), Some(use_)) = (pos("rise"), pos("use")) {
            assert!(
                rise < use_,
                "'rise' ({rise}) should outrank first-letter-changed 'use' ({use_}): {top:?}"
            );
        }
    }

    #[test]
    fn common_word_reachable_by_small_edit_is_listed_and_beats_rare_prefix() {
        // "respine" -> the common "response" (sub i->o then insert s) must be listed
        // and rank ahead of the rare exact-prefix continuation "respined", which the
        // old distance-first comparator pinned on top.
        let dict = FstDictionary::curated();
        let words = ranked_words("respine", dict.as_ref(), &CompletionContext::default());
        let head: Vec<&String> = words.iter().take(8).collect();
        let pos = |t: &str| words.iter().position(|w| w == t);

        assert!(
            words.iter().any(|w| w == "response"),
            "'response' should be listed for 'respine': {head:?}"
        );
        if let (Some(response), Some(respined)) = (pos("response"), pos("respined")) {
            assert!(
                response < respined,
                "'response' ({response}) should outrank 'respined' ({respined}): {head:?}"
            );
        }
    }

    #[test]
    fn short_prefix_surfaces_real_words_over_two_letter_noise() {
        // Typing "im" should surface real words ("in", "image", "important") above
        // two-letter fuzzy noise ("mi", "um", "km", "om"). The old high-confidence
        // short-correction *tier* promoted this noise to the very top; demoting that
        // signal to a small bonus (below frequency) keeps the real words on top.
        let dict = FstDictionary::curated();
        let words = ranked_words("im", dict.as_ref(), &CompletionContext::default());
        let head: Vec<&String> = words.iter().take(8).collect();
        let pos = |t: &str| words.iter().position(|w| w == t);

        for noise in ["mi", "um", "km", "om"] {
            if let (Some(real), Some(junk)) = (pos("in"), pos(noise)) {
                assert!(
                    real < junk,
                    "real word 'in' ({real}) should rank above two-letter noise '{noise}' ({junk}): {head:?}"
                );
            }
        }
        if let (Some(image), Some(mi)) = (pos("image"), pos("mi")) {
            assert!(
                image < mi,
                "prefix word 'image' ({image}) should rank above 'mi' ({mi}): {head:?}"
            );
        }
    }

    #[test]
    fn high_confidence_short_correction_helps_but_stays_below_frequency() {
        // Args: distance, typed_word, first_letter_match, high_confidence,
        //       global_frequency_rank, local_frequency, previous_word_frequency,
        //       global_bigram_rank.
        // The bonus lowers the cost of a confident short correction...
        let with = combined_completion_cost(82, false, true, true, 500, 0, 0, u32::MAX);
        let without = combined_completion_cost(82, false, true, false, 500, 0, 0, u32::MAX);
        assert_eq!(without - with, HIGH_CONFIDENCE_SHORT_CORRECTION_BONUS);

        // ...but it sits *below* frequency: a much more common same-distance word
        // (rank 50) still beats a high-confidence correction to a rare word (rank 5000).
        let confident_correction =
            combined_completion_cost(82, false, true, true, 5000, 0, 0, u32::MAX);
        let more_frequent_plain =
            combined_completion_cost(82, false, true, false, 50, 0, 0, u32::MAX);
        assert!(
            more_frequent_plain < confident_correction,
            "frequency ({more_frequent_plain}) should outweigh the short-correction bonus ({confident_correction})"
        );
    }

    #[test]
    fn apostrophe_less_contraction_surfaces_canonical_form() {
        // Typed without the apostrophe (and these letters aren't words), the
        // contraction is the clear intent and should lead with correct casing.
        let dict = FstDictionary::curated();
        for (typed, expected) in [
            ("im", "I'm"),
            ("ive", "I've"),
            ("weve", "we've"),
            ("dont", "don't"),
        ] {
            let words = ranked_words(typed, dict.as_ref(), &CompletionContext::default());
            let head: Vec<&String> = words.iter().take(6).collect();
            assert_eq!(
                words.first().map(String::as_str),
                Some(expected),
                "typing {typed:?} should surface {expected:?} first, got {head:?}"
            );
        }
    }

    #[test]
    fn contraction_is_offered_but_keeps_a_common_typed_word_on_top() {
        // When the typed letters are themselves a common word, the contraction is
        // still offered but priced so the real word stays ahead.
        let dict = FstDictionary::curated();
        for (typed, contraction) in [("were", "we're"), ("id", "I'd")] {
            let words = ranked_words(typed, dict.as_ref(), &CompletionContext::default());
            let pos = |t: &str| words.iter().position(|w| w == t);
            assert!(
                words.iter().any(|w| w == contraction),
                "{contraction:?} should be offered for {typed:?}"
            );
            if let (Some(real), Some(contr)) = (pos(typed), pos(contraction)) {
                assert!(
                    real < contr,
                    "real word {typed:?} ({real}) should stay above {contraction:?} ({contr})"
                );
            }
        }
    }

    #[test]
    fn confident_top_candidate_commits_on_space() {
        let dict = dict_with_words(&[("toy", true), ("the", true)]);
        let misspelled_words = HashSet::new();
        let completions = rank_completion_candidates(
            &dict,
            &chars("teh"),
            &CompletionContext::default(),
            &misspelled_words,
            50,
        );

        assert_eq!(completions[0].word, "the");
        assert!(should_commit_space(&completions));
    }

    #[test]
    fn commit_with_space_never_disables_space_commit() {
        let dict = dict_with_words(&[("toy", true), ("the", true)]);
        let misspelled_words = HashSet::new();
        let completions = rank_completion_candidates(
            &dict,
            &chars("teh"),
            &CompletionContext::default(),
            &misspelled_words,
            50,
        );

        assert!(!should_item_commit_space(
            0,
            &completions,
            CompletionCommitWithSpace::Never
        ));
    }

    #[test]
    fn commit_with_space_always_enables_every_item() {
        let dict = dict_with_words(&[("toy", true), ("the", true)]);
        let misspelled_words = HashSet::new();
        let completions = rank_completion_candidates(
            &dict,
            &chars("teh"),
            &CompletionContext::default(),
            &misspelled_words,
            50,
        );

        assert!(should_item_commit_space(
            0,
            &completions,
            CompletionCommitWithSpace::Always
        ));
        assert!(should_item_commit_space(
            1,
            &completions,
            CompletionCommitWithSpace::Always
        ));
    }

    #[test]
    fn prefix_score_checks_candidate_prefix_windows() {
        let score = best_completion_prefix_score(&chars("whie"), &chars("while")).unwrap();

        assert_eq!(score.distance, INSERT_DELETE_COST);
        assert_eq!(score.matched_prefix_len, 5);
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

    // --- Missed-space split tests ---

    #[test]
    fn missed_space_n_splits_the_from_world() {
        // "thenworld": user pressed 'n' instead of space → left="the", right="world"
        let dict = dict_with_words(&[("the", true)]);
        let splits = find_missed_space_splits(&chars("thenworld"), &dict, 2);
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].0.iter().collect::<String>(), "the");
        assert_eq!(splits[0].1.iter().collect::<String>(), "world");
    }

    #[test]
    fn missed_space_split_requires_left_word_in_dict() {
        // "xhe" is not a word — no split for "xhenworld"
        let dict = dict_with_words(&[("world", true)]);
        assert!(find_missed_space_splits(&chars("xhenworld"), &dict, 2).is_empty());
    }

    #[test]
    fn missed_space_split_all_four_adjacent_keys() {
        let dict = dict_with_words(&[("in", true)]);
        // v: "invery" → left="in", right="ery"
        assert!(!find_missed_space_splits(&chars("invery"), &dict, 2).is_empty());
        // b: "inbest" → left="in", right="est"
        assert!(!find_missed_space_splits(&chars("inbest"), &dict, 2).is_empty());
        // n: "inname" → split at second 'n', left="in", right="ame"
        assert!(!find_missed_space_splits(&chars("inname"), &dict, 2).is_empty());
        // m: "inmore" → left="in", right="ore"
        assert!(!find_missed_space_splits(&chars("inmore"), &dict, 2).is_empty());
    }

    #[test]
    fn missed_space_split_min_right_len_respected() {
        let dict = dict_with_words(&[("the", true)]);
        // "thenw": right part is 1 char — below min_right_len=2
        assert!(find_missed_space_splits(&chars("thenw"), &dict, 2).is_empty());
        // "thenwo": right part is 2 chars — exactly at min_right_len=2
        assert!(!find_missed_space_splits(&chars("thenwo"), &dict, 2).is_empty());
    }

    #[test]
    fn missed_space_split_curated_dict_the_world() {
        // End-to-end check: the curated dictionary has "the", so "thenworld" splits.
        let dict = FstDictionary::curated();
        let splits = find_missed_space_splits(&chars("thenworld"), dict.as_ref(), 2);
        assert!(
            splits.iter().any(|(l, r)| {
                l.iter().collect::<String>() == "the" && r.iter().collect::<String>() == "world"
            }),
            "expected to find split (the, world) in 'thenworld'"
        );
    }

    #[test]
    fn missed_space_split_v_key_also_detected() {
        let dict = FstDictionary::curated();
        let splits = find_missed_space_splits(&chars("thevworld"), dict.as_ref(), 2);
        assert!(
            splits.iter().any(|(l, r)| {
                l.iter().collect::<String>() == "the" && r.iter().collect::<String>() == "world"
            }),
            "expected to find split (the, world) in 'thevworld'"
        );
    }
}
