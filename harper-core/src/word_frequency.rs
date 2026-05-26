//! Bundled unigram word-frequency data used to rank completion suggestions.
//!
//! The list is the public-domain `google-10000-english` frequency-ordered word
//! list, derived from the Google Web Trillion Word Corpus. Words are stored most-
//! frequent first, one lowercase word per line, and a word's line position is used
//! as its frequency rank (0 = most frequent). Words absent from the list have no
//! rank and fall back to coarser signals (e.g. the dictionary's `common` flag).
//!
//! Source: <https://github.com/first20hours/google-10000-english> (Unlicense)

use std::collections::HashMap;
use std::sync::OnceLock;

/// Frequency-ordered word list: one lowercase word per line, most frequent first.
const RAW_FREQUENCY_LIST: &str = include_str!("../english_word_frequency.txt");

/// Frequency-ordered bigram list: one lowercase `first second` pair per line, most
/// frequent first. Same provenance family as the unigram list — Norvig's `count_2w`
/// (Google Web Trillion Word Corpus), trimmed to the most frequent clean word pairs.
const RAW_BIGRAM_LIST: &str = include_str!("../english_bigram_frequency.txt");

fn frequency_ranks() -> &'static HashMap<&'static str, u32> {
    static RANKS: OnceLock<HashMap<&'static str, u32>> = OnceLock::new();
    RANKS.get_or_init(|| {
        let mut ranks = HashMap::new();
        for (rank, line) in RAW_FREQUENCY_LIST.lines().enumerate() {
            let word = line.trim();
            if word.is_empty() {
                continue;
            }
            // Keep the first (most frequent) occurrence of any duplicated word.
            ranks.entry(word).or_insert(rank as u32);
        }
        ranks
    })
}

/// The frequency rank of `word`, where `0` is the most frequent word in English.
///
/// Returns `None` for words outside the bundled list. Lookup is case-insensitive.
/// A lower rank means a more frequent (and, for completions, higher-priority) word.
pub fn frequency_rank(word: &str) -> Option<u32> {
    if word.is_empty() {
        return None;
    }

    // The list is lowercase; completion candidates are normalized before lookup,
    // so the borrow-free fast path hits in the common case.
    if let Some(rank) = frequency_ranks().get(word).copied() {
        return Some(rank);
    }

    frequency_ranks().get(word.to_lowercase().as_str()).copied()
}

fn bigram_ranks() -> &'static HashMap<&'static str, u32> {
    static RANKS: OnceLock<HashMap<&'static str, u32>> = OnceLock::new();
    RANKS.get_or_init(|| {
        let mut ranks = HashMap::new();
        for (rank, line) in RAW_BIGRAM_LIST.lines().enumerate() {
            let pair = line.trim();
            if pair.is_empty() {
                continue;
            }
            // Keep the first (most frequent) occurrence of any duplicated pair.
            ranks.entry(pair).or_insert(rank as u32);
        }
        ranks
    })
}

/// The frequency rank of the bigram `previous_word word`, where `0` is the most
/// frequent word pair in English.
///
/// Returns `None` when the pair is not in the bundled list. Lookup is case-insensitive.
/// A lower rank means a more common collocation (e.g. "new york"), which lets a
/// context-appropriate completion outrank a more frequent but contextually weaker word.
pub fn bigram_rank(previous_word: &str, word: &str) -> Option<u32> {
    if previous_word.is_empty() || word.is_empty() {
        return None;
    }

    // The list is lowercase; callers normalize before lookup, so this usually hits.
    let key = format!("{previous_word} {word}");
    if let Some(rank) = bigram_ranks().get(key.as_str()).copied() {
        return Some(rank);
    }

    bigram_ranks().get(key.to_lowercase().as_str()).copied()
}

/// The most frequent English words that start with `prefix`, in descending frequency
/// order, up to `limit` results.
///
/// Completion candidates are otherwise fetched from the dictionary in alphabetical
/// order and truncated, which can drop very common words that sort late within a
/// prefix (e.g. "could"/"come" sit far past the start of the "co" words). Seeding the
/// candidate pool from this list keeps those words in contention. Matching is
/// case-insensitive; returned words are lowercase slices into the bundled list.
pub fn most_frequent_with_prefix(prefix: &str, limit: usize) -> Vec<&'static str> {
    if prefix.is_empty() || limit == 0 {
        return Vec::new();
    }

    let prefix_lower = prefix.to_lowercase();
    let mut matches = Vec::new();
    for line in RAW_FREQUENCY_LIST.lines() {
        let word = line.trim();
        if word.starts_with(prefix_lower.as_str()) {
            matches.push(word);
            if matches.len() >= limit {
                break;
            }
        }
    }

    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_is_the_most_frequent_word() {
        assert_eq!(frequency_rank("the"), Some(0));
    }

    #[test]
    fn frequent_words_outrank_rare_words() {
        let work = frequency_rank("work").expect("work is in the list");
        let word = frequency_rank("word").expect("word is in the list");
        let worm = frequency_rank("worm").expect("worm is in the list");

        assert!(work < word, "work should be more frequent than word");
        assert!(word < worm, "word should be more frequent than worm");
    }

    #[test]
    fn lookup_is_case_insensitive() {
        assert_eq!(frequency_rank("The"), frequency_rank("the"));
        assert_eq!(frequency_rank("WORK"), frequency_rank("work"));
    }

    #[test]
    fn unknown_words_have_no_rank() {
        assert_eq!(frequency_rank("zzqqxnotaword"), None);
    }

    #[test]
    fn common_collocations_have_a_rank() {
        assert!(bigram_rank("thank", "you").is_some());
        assert!(bigram_rank("new", "york").is_some());
    }

    #[test]
    fn frequent_bigram_outranks_rarer_bigram() {
        let in_the = bigram_rank("in", "the").expect("\"in the\" is in the list");
        let new_york = bigram_rank("new", "york").expect("\"new york\" is in the list");
        assert!(
            in_the < new_york,
            "\"in the\" is far more frequent than \"new york\""
        );
    }

    #[test]
    fn bigram_lookup_is_case_insensitive() {
        assert_eq!(bigram_rank("New", "York"), bigram_rank("new", "york"));
    }

    #[test]
    fn unknown_bigram_has_no_rank() {
        assert_eq!(bigram_rank("zzqqx", "wwvvqq"), None);
    }

    #[test]
    fn most_frequent_with_prefix_is_frequency_ordered() {
        let words = most_frequent_with_prefix("th", 5);
        assert_eq!(words.first(), Some(&"the"));
        assert!(words.len() <= 5);
        assert!(words.iter().all(|word| word.starts_with("th")));
    }

    #[test]
    fn most_frequent_with_prefix_surfaces_alphabetically_late_words() {
        // "could"/"come" sort late within the "co" words but are very frequent.
        let words = most_frequent_with_prefix("co", 100);
        assert!(words.contains(&"could"));
        assert!(words.contains(&"come"));
    }

    #[test]
    fn most_frequent_with_prefix_respects_limit_and_case() {
        assert!(most_frequent_with_prefix("th", 3).len() <= 3);
        assert_eq!(
            most_frequent_with_prefix("TH", 3),
            most_frequent_with_prefix("th", 3)
        );
        assert!(most_frequent_with_prefix("", 5).is_empty());
    }
}
