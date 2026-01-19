use std::collections::HashMap;

/// Phone touchscreen QWERTY keyboard layout coordinates
/// Row 0: Q W E R T Y U I O P
/// Row 1: A S D F G H J K L
/// Row 2: Z X C V B N M
fn get_phone_qwerty_coords() -> HashMap<char, (f32, f32)> {
    let mut m = HashMap::new();
    // Row 0
    m.insert('q', (0.0, 0.0));
    m.insert('w', (1.0, 0.0));
    m.insert('e', (2.0, 0.0));
    m.insert('r', (3.0, 0.0));
    m.insert('t', (4.0, 0.0));
    m.insert('y', (5.0, 0.0));
    m.insert('u', (6.0, 0.0));
    m.insert('i', (7.0, 0.0));
    m.insert('o', (8.0, 0.0));
    m.insert('p', (9.0, 0.0));
    // Row 1
    m.insert('a', (0.0, 1.0));
    m.insert('s', (1.0, 1.0));
    m.insert('d', (2.0, 1.0));
    m.insert('f', (3.0, 1.0));
    m.insert('g', (4.0, 1.0));
    m.insert('h', (5.0, 1.0));
    m.insert('j', (6.0, 1.0));
    m.insert('k', (7.0, 1.0));
    m.insert('l', (8.0, 1.0));
    // Row 2
    m.insert('z', (0.0, 2.0));
    m.insert('x', (1.0, 2.0));
    m.insert('c', (2.0, 2.0));
    m.insert('v', (3.0, 2.0));
    m.insert('b', (4.0, 2.0));
    m.insert('n', (5.0, 2.0));
    m.insert('m', (6.0, 2.0));
    m
}

/// Maximum normalized distance on the phone QWERTY keyboard (Q to M ≈ 9.49)
const MAX_DISTANCE: f32 = 10.0;

/// Calculate the normalized keyboard distance between two characters on a phone QWERTY keyboard.
/// Returns a value between 0.0 (same key) and 1.0 (maximum distance).
/// Non-alphabetic characters return 1.0 (default maximum distance).
pub fn keyboard_distance(char1: char, char2: char) -> f32 {
    let coords = get_phone_qwerty_coords();

    let c1 = char1.to_lowercase().next().unwrap_or(char1);
    let c2 = char2.to_lowercase().next().unwrap_or(char2);

    // If same character, distance is 0
    if c1 == c2 {
        return 0.0;
    }

    // Get coordinates for both characters
    let pos1 = coords.get(&c1);
    let pos2 = coords.get(&c2);

    match (pos1, pos2) {
        (Some(&(x1, y1)), Some(&(x2, y2))) => {
            // Calculate Euclidean distance
            let dx = x1 - x2;
            let dy = y1 - y2;
            let distance = (dx * dx + dy * dy).sqrt();

            // Normalize to 0.0-1.0 range
            (distance / MAX_DISTANCE).min(1.0)
        }
        _ => {
            // If either character is not on the keyboard, return maximum distance
            1.0
        }
    }
}

/// Calculate the average keyboard distance between two character sequences.
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
        // Only count positions where characters differ
        if query[i].to_lowercase().next() != candidate[i].to_lowercase().next() {
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
    fn test_adjacent_horizontal() {
        // A and S are horizontally adjacent (distance = 1.0)
        let dist = keyboard_distance('a', 's');
        assert!((dist - 0.1).abs() < 0.01, "Expected ~0.1, got {}", dist);
    }

    #[test]
    fn test_adjacent_vertical() {
        // Q and A are vertically adjacent (distance = 1.0)
        let dist = keyboard_distance('q', 'a');
        assert!((dist - 0.1).abs() < 0.01, "Expected ~0.1, got {}", dist);
    }

    #[test]
    fn test_diagonal() {
        // Q and S are diagonal (distance = sqrt(2) ≈ 1.414)
        let dist = keyboard_distance('q', 's');
        assert!((dist - 0.141).abs() < 0.01, "Expected ~0.141, got {}", dist);
    }

    #[test]
    fn test_far_keys() {
        // Q and M are far apart (max distance)
        let dist = keyboard_distance('q', 'm');
        assert!(dist > 0.6, "Expected distance > 0.6, got {}", dist);
    }

    #[test]
    fn test_non_alphabetic() {
        // Non-alphabetic characters return default maximum distance
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
    fn test_avg_distance_one_diff() {
        // "hello" vs "hallo" - one character different (e→a)
        let query: Vec<char> = "hello".chars().collect();
        let candidate: Vec<char> = "hallo".chars().collect();
        let avg_dist = avg_keyboard_distance(&query, &candidate);

        // e(2,0) to a(0,1): distance = sqrt(4+1) = sqrt(5) ≈ 2.236, normalized ≈ 0.224
        assert!(avg_dist > 0.15 && avg_dist < 0.3, "Expected ~0.224, got {}", avg_dist);
    }

    #[test]
    fn test_avg_distance_empty() {
        let query: Vec<char> = vec![];
        let candidate: Vec<char> = "hello".chars().collect();
        assert_eq!(avg_keyboard_distance(&query, &candidate), 1.0);
    }
}
