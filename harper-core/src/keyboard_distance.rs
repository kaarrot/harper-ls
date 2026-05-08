/// Cost scale used by keyboard-aware edit distance helpers.
pub const KEYBOARD_DISTANCE_SCALE: u16 = 100;

/// Cost for inserting or deleting one character.
pub const INSERT_DELETE_COST: u16 = KEYBOARD_DISTANCE_SCALE;

/// Cost for swapping two adjacent characters.
pub const ADJACENT_TRANSPOSITION_COST: u16 = 50;

const MIN_SUBSTITUTION_COST: u16 = 60;
const MAX_SUBSTITUTION_COST: u16 = 260;

/// Maximum distance on the staggered phone QWERTY keyboard, measured from q to m.
const MAX_PHONE_QWERTY_DISTANCE: f32 = 8.902247;

/// Phone touchscreen QWERTY keyboard layout coordinates with staggered rows.
///
/// Row 0: Q W E R T Y U I O P
/// Row 1:  A S D F G H J K L
/// Row 2:    Z X C V B N M
fn phone_qwerty_coord(c: char) -> Option<(f32, f32)> {
    match normalized_key(c) {
        'q' => Some((0.0, 0.0)),
        'w' => Some((1.0, 0.0)),
        'e' => Some((2.0, 0.0)),
        'r' => Some((3.0, 0.0)),
        't' => Some((4.0, 0.0)),
        'y' => Some((5.0, 0.0)),
        'u' => Some((6.0, 0.0)),
        'i' => Some((7.0, 0.0)),
        'o' => Some((8.0, 0.0)),
        'p' => Some((9.0, 0.0)),
        'a' => Some((0.5, 1.0)),
        's' => Some((1.5, 1.0)),
        'd' => Some((2.5, 1.0)),
        'f' => Some((3.5, 1.0)),
        'g' => Some((4.5, 1.0)),
        'h' => Some((5.5, 1.0)),
        'j' => Some((6.5, 1.0)),
        'k' => Some((7.5, 1.0)),
        'l' => Some((8.5, 1.0)),
        'z' => Some((1.5, 2.0)),
        'x' => Some((2.5, 2.0)),
        'c' => Some((3.5, 2.0)),
        'v' => Some((4.5, 2.0)),
        'b' => Some((5.5, 2.0)),
        'n' => Some((6.5, 2.0)),
        'm' => Some((7.5, 2.0)),
        _ => None,
    }
}

fn normalized_key(c: char) -> char {
    c.to_lowercase().next().unwrap_or(c)
}

fn chars_equal(a: char, b: char) -> bool {
    normalized_key(a) == normalized_key(b)
}

/// Calculate the normalized keyboard distance between two characters on a phone QWERTY keyboard.
///
/// Returns a value between 0.0 (same key) and 1.0 (maximum distance). Non-alphabetic
/// characters return 1.0 unless they are equal.
pub fn keyboard_distance(char1: char, char2: char) -> f32 {
    if chars_equal(char1, char2) {
        return 0.0;
    }

    match (phone_qwerty_coord(char1), phone_qwerty_coord(char2)) {
        (Some((x1, y1)), Some((x2, y2))) => {
            let dx = x1 - x2;
            let dy = y1 - y2;
            ((dx * dx + dy * dy).sqrt() / MAX_PHONE_QWERTY_DISTANCE).min(1.0)
        }
        _ => 1.0,
    }
}

/// Cost of replacing `from` with `to` in a keyboard-aware edit distance.
pub fn keyboard_substitution_cost(from: char, to: char) -> u16 {
    if chars_equal(from, to) {
        return 0;
    }

    let weighted = MIN_SUBSTITUTION_COST as f32
        + keyboard_distance(from, to) * (MAX_SUBSTITUTION_COST - MIN_SUBSTITUTION_COST) as f32;

    weighted.round() as u16
}

/// Calculate a weighted optimal-string-alignment Damerau distance.
///
/// Insertions and deletions have a fixed cost, substitutions are weighted by phone-key
/// proximity, and adjacent transpositions are cheaper than delete+insert.
pub fn weighted_damerau_distance(query: &[char], candidate: &[char]) -> u16 {
    let rows = query.len() + 1;
    let cols = candidate.len() + 1;
    let mut dp = vec![0u16; rows * cols];

    let cell = |i: usize, j: usize| i * cols + j;

    for i in 0..=query.len() {
        dp[cell(i, 0)] = (i as u16).saturating_mul(INSERT_DELETE_COST);
    }
    for j in 0..=candidate.len() {
        dp[cell(0, j)] = (j as u16).saturating_mul(INSERT_DELETE_COST);
    }

    for i in 1..=query.len() {
        for j in 1..=candidate.len() {
            let deletion = dp[cell(i - 1, j)].saturating_add(INSERT_DELETE_COST);
            let insertion = dp[cell(i, j - 1)].saturating_add(INSERT_DELETE_COST);
            let substitution = dp[cell(i - 1, j - 1)]
                .saturating_add(keyboard_substitution_cost(query[i - 1], candidate[j - 1]));

            let mut best = deletion.min(insertion).min(substitution);

            if i > 1
                && j > 1
                && chars_equal(query[i - 1], candidate[j - 2])
                && chars_equal(query[i - 2], candidate[j - 1])
            {
                best = best.min(dp[cell(i - 2, j - 2)].saturating_add(ADJACENT_TRANSPOSITION_COST));
            }

            dp[cell(i, j)] = best;
        }
    }

    dp[cell(query.len(), candidate.len())]
}

/// Calculate the average keyboard distance between two character sequences.
///
/// This compares characters at matching positions and returns the average distance.
/// Only counts positions where characters differ.
pub fn avg_keyboard_distance(query: &[char], candidate: &[char]) -> f32 {
    let min_len = query.len().min(candidate.len());

    if min_len == 0 {
        return 1.0;
    }

    let mut total_distance = 0.0;
    let mut count = 0;

    for i in 0..min_len {
        let dist = keyboard_distance(query[i], candidate[i]);
        if !chars_equal(query[i], candidate[i]) {
            total_distance += dist;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    total_distance / count as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn test_same_character() {
        assert_eq!(keyboard_distance('a', 'a'), 0.0);
        assert_eq!(keyboard_distance('q', 'q'), 0.0);
    }

    #[test]
    fn test_case_insensitive() {
        let dist_lower = keyboard_distance('a', 's');
        let dist_upper = keyboard_distance('A', 'S');
        let dist_mixed = keyboard_distance('a', 'S');
        assert_eq!(dist_lower, dist_upper);
        assert_eq!(dist_lower, dist_mixed);
    }

    #[test]
    fn nearby_substitution_beats_far_substitution() {
        assert!(
            keyboard_substitution_cost('g', 'h') < keyboard_substitution_cost('g', 'p'),
            "nearby key substitutions should cost less than distant substitutions"
        );
    }

    #[test]
    fn transposition_beats_unrelated_two_edit_change() {
        assert!(
            weighted_damerau_distance(&chars("teh"), &chars("the"))
                < weighted_damerau_distance(&chars("teh"), &chars("toy")),
            "adjacent transposition should beat unrelated two-edit changes"
        );
    }

    #[test]
    fn insertion_does_not_misalign_rest_of_word() {
        assert_eq!(
            weighted_damerau_distance(&chars("whie"), &chars("while")),
            INSERT_DELETE_COST
        );
    }

    #[test]
    fn deletion_does_not_misalign_rest_of_word() {
        assert_eq!(
            weighted_damerau_distance(&chars("tthis"), &chars("this")),
            INSERT_DELETE_COST
        );
    }

    #[test]
    fn test_non_alphabetic() {
        assert_eq!(keyboard_distance('1', '2'), 1.0);
        assert_eq!(keyboard_distance('a', '1'), 1.0);
    }

    #[test]
    fn test_avg_distance_same_strings() {
        let query: Vec<char> = "hello".chars().collect();
        let candidate: Vec<char> = "hello".chars().collect();
        assert_eq!(avg_keyboard_distance(&query, &candidate), 0.0);
    }

    #[test]
    fn test_avg_distance_empty() {
        let query: Vec<char> = vec![];
        let candidate: Vec<char> = "hello".chars().collect();
        assert_eq!(avg_keyboard_distance(&query, &candidate), 1.0);
    }
}
