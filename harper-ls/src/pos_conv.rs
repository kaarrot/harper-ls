//! This module includes various conversions from the index-based [`Span`]s that
//! Harper uses, and the Ranges that the LSP uses.

use harper_core::Span;
use tower_lsp_server::lsp_types::{Position, Range};

/// Pre-built index of line start positions for O(1) position conversion.
/// Avoids O(N) scanning of the entire document on every position lookup.
#[derive(Debug, Clone, Default)]
pub struct LineIndex {
    /// Indices of the first character of each line.
    /// line_starts[0] is always 0 (first line starts at index 0).
    /// line_starts[1] is the index after the first '\n', etc.
    line_starts: Vec<usize>,
}

impl LineIndex {
    /// Build a new LineIndex from source characters.
    /// Scans the document once in O(N) to find all line boundaries.
    pub fn new(source: &[char]) -> Self {
        let mut line_starts = vec![0]; // First line always starts at 0

        for (idx, &ch) in source.iter().enumerate() {
            if ch == '\n' {
                // Next line starts after this newline
                line_starts.push(idx + 1);
            }
        }

        Self { line_starts }
    }

    /// Convert a character index to an LSP Position.
    /// O(log N) complexity using binary search on line starts.
    #[allow(dead_code)]
    pub fn index_to_position(&self, source: &[char], index: usize) -> Position {
        // Binary search to find which line contains this index
        let line = match self.line_starts.binary_search(&index) {
            Ok(line) => line,                    // Exact match - index is at start of line
            Err(line) => line.saturating_sub(1), // Index is within previous line
        };

        let line_start = self.line_starts.get(line).copied().unwrap_or(0);

        // Calculate column as UTF-16 code units from line start to index
        let cols: usize = source[line_start..index.min(source.len())]
            .iter()
            .map(|c| c.len_utf16())
            .sum();

        Position {
            line: line as u32,
            character: cols as u32,
        }
    }

    /// Convert an LSP Position to a character index.
    /// O(line length) complexity after O(1) line lookup.
    pub fn position_to_index(&self, source: &[char], position: Position) -> usize {
        // Find target line start index
        let line_idx = position.line as usize;

        // Get the slice for the target line
        let Some(&line_start) = self.line_starts.get(line_idx) else {
            // Requested line doesn't exist - return last character
            return source.len().saturating_sub(1);
        };

        // Find where this line ends (at next line start or end of source)
        let line_end = self
            .line_starts
            .get(line_idx + 1)
            .copied()
            .unwrap_or(source.len());

        position_to_index_in_line(source, line_start, line_end, position.character)
    }

    /// Convert a Span to an LSP Range using this line index.
    #[allow(dead_code)]
    pub fn span_to_range(&self, source: &[char], span: Span<char>) -> Range {
        let start = self.index_to_position(source, span.start);
        let end = self.index_to_position(source, span.end);
        Range { start, end }
    }

    /// Convert an LSP Range to a Span using this line index.
    #[allow(dead_code)]
    pub fn range_to_span(&self, source: &[char], range: Range) -> Span<char> {
        let start = self.position_to_index(source, range.start);
        let end = self.position_to_index(source, range.end);
        Span::new(start, end)
    }

    /// Check if a position is beyond the current document bounds.
    /// Returns true if the position would be clamped by position_to_index.
    /// Useful for detecting race conditions where completion requests arrive
    /// before the corresponding didChange notification.
    #[allow(dead_code)]
    pub fn is_position_out_of_bounds(&self, source: &[char], position: Position) -> bool {
        let line_idx = position.line as usize;

        // Check if line exists
        if let Some(&line_start) = self.line_starts.get(line_idx) {
            let line_end = self
                .line_starts
                .get(line_idx + 1)
                .copied()
                .unwrap_or(source.len());

            let content_end = line_content_end(source, line_start, line_end);
            let line_len_utf16: usize = source[line_start..content_end]
                .iter()
                .map(|c| c.len_utf16())
                .sum();

            // Position is out of bounds if character is beyond line content length.
            // Character N means "after N UTF-16 code units", so for a line with N
            // code units, position N is at the end and still valid.
            position.character as usize > line_len_utf16
        } else {
            // Line doesn't exist - definitely out of bounds
            true
        }
    }
}

fn line_content_end(source: &[char], line_start: usize, line_end: usize) -> usize {
    if line_end > line_start && source.get(line_end - 1) == Some(&'\n') {
        line_end - 1
    } else {
        line_end
    }
}

fn position_to_index_in_line(
    source: &[char],
    line_start: usize,
    line_end: usize,
    utf16_character: u32,
) -> usize {
    let content_end = line_content_end(source, line_start, line_end);
    let target_utf16 = utf16_character as usize;
    let mut seen_utf16 = 0;

    for (offset, ch) in source[line_start..content_end].iter().enumerate() {
        if seen_utf16 == target_utf16 {
            return line_start + offset;
        }

        let next_seen_utf16 = seen_utf16 + ch.len_utf16();
        if target_utf16 < next_seen_utf16 {
            return line_start + offset;
        }

        seen_utf16 = next_seen_utf16;
    }

    if target_utf16 <= seen_utf16 {
        content_end
    } else {
        line_end.saturating_sub(1)
    }
}

pub fn span_to_range(source: &[char], span: Span<char>) -> Range {
    let start = index_to_position(source, span.start);
    let end = index_to_position(source, span.end);

    Range { start, end }
}

fn index_to_position(source: &[char], index: usize) -> Position {
    let before = &source[0..index];
    let newline_indices: Vec<_> = before
        .iter()
        .enumerate()
        .filter_map(|(idx, c)| if *c == '\n' { Some(idx + 1) } else { None })
        .collect();

    let lines = newline_indices.len();

    let last_newline_idx = newline_indices.last().copied().unwrap_or(0);

    let cols: usize = source[last_newline_idx..index]
        .iter()
        .map(|c| c.len_utf16())
        .sum();

    Position {
        line: lines as u32,
        character: cols as u32,
    }
}

/// Converts a position to a (zero-based) character index within the source character array.
///
/// The position is converted to an index using saturating arithmetic. If the requested line index
/// is too high, the index of the last character in the source is returned. If the line is
/// in-bounds but the requested character isn't, the last character of that line is returned.
pub fn position_to_index(source: &[char], position: Position) -> usize {
    let mut line_start = 0;
    for _ in 0..position.line {
        let Some(newline_offset) = source[line_start..].iter().position(|char| *char == '\n')
        else {
            // Requested line index is too high.
            // Return the last char in `source' as the closest approximation.
            // Uses `saturating_sub` to avoid underflow when `source` is empty.
            return source.len().saturating_sub(1);
        };

        line_start += newline_offset + 1;
    }

    if line_start > source.len() {
        // Requested line index is too high.
        // Return the last char in `source' as the closest approximation.
        // Uses `saturating_sub` to avoid underflow when `source` is empty.
        return source.len().saturating_sub(1);
    }

    let line_end = source[line_start..]
        .iter()
        .position(|char| *char == '\n')
        .map(|newline_offset| line_start + newline_offset + 1)
        .unwrap_or(source.len());

    position_to_index_in_line(source, line_start, line_end, position.character)
}

pub fn range_to_span(source: &[char], range: Range) -> Span<char> {
    let start = position_to_index(source, range.start);
    let end = position_to_index(source, range.end);

    Span::new(start, end)
}

#[cfg(test)]
mod tests {
    use tower_lsp_server::lsp_types::{Position, Range};

    use super::{LineIndex, index_to_position, position_to_index, range_to_span};

    #[test]
    fn first_line_correct() {
        let source: Vec<_> = "Hello there.".chars().collect();

        let start = Position {
            line: 0,
            character: 4,
        };

        let i = position_to_index(&source, start);

        assert_eq!(i, 4);

        let p = index_to_position(&source, i);

        assert_eq!(p, start)
    }

    #[test]
    fn reversible_position_conv() {
        let source: Vec<_> = "There was a man,\n his voice had timbre,\n unlike a boy."
            .chars()
            .collect();

        let a = Position {
            line: 1,
            character: 2,
        };

        let b = position_to_index(&source, a);

        assert_eq!(b, 19);

        let c = index_to_position(&source, b);

        let d = position_to_index(&source, a);

        assert_eq!(a, c);
        assert_eq!(b, d);
    }

    #[test]
    fn end_of_line() {
        let source: Vec<_> = "This is a short test\n".chars().collect();

        let a = Position {
            line: 0,
            character: 20,
        };

        assert_eq!(position_to_index(&source, a), 20);
    }

    #[test]
    fn end_of_file() {
        let source: Vec<_> = "This is a short test".chars().collect();

        let a = Position {
            line: 0,
            character: 19,
        };

        assert_eq!(position_to_index(&source, a), 19);
    }

    #[test]
    fn issue_250() {
        let source: Vec<_> = "Hello thur\n".chars().collect();

        let range = Range {
            start: Position {
                line: 0,
                character: 9,
            },
            end: Position {
                line: 0,
                character: 10,
            },
        };

        let out = range_to_span(&source, range);
        assert_eq!(out.start, 9);
        assert_eq!(out.end, 10);
    }

    /// Ensures that `position_to_index` does not produce an incorrect index of 0 for an input
    /// `Position` of `{ line: 1, character: 0 }`.
    /// Related to: https://github.com/Automattic/harper/issues/1253
    #[test]
    fn pos_to_index_correct_for_l1_c0() {
        let source: Vec<_> = ". one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five twenty-six twenty-seven twenty-eight twenty-nine thirty thirty-one\n".chars().collect();
        let position = Position {
            line: 1,
            character: 0,
        };

        let out_index = position_to_index(&source, position);
        assert_ne!(out_index, 0);
    }

    /// Ensures `position_to_index` produces the correct result when indexing line 0 character 0.
    #[test]
    fn pos_to_index_off_by_one_check_l0_c0() {
        let source: Vec<_> = "abc\ndef\nghi\njkl".chars().collect();
        let position = Position {
            line: 0,
            character: 0,
        };

        let out_index = position_to_index(&source, position);
        assert_eq!(source[out_index], 'a');
    }

    /// Ensures `position_to_index` produces the correct result when indexing a non-zero line and
    /// character.
    #[test]
    fn pos_to_index_off_by_one_check_l2_c1() {
        let source: Vec<_> = "abc\ndef\nghi\njkl".chars().collect();
        let position = Position {
            line: 2,
            character: 1,
        };

        let out_index = position_to_index(&source, position);
        assert_eq!(source[out_index], 'h');
    }

    /// Ensures `position_to_index` produces an index of 0 when indexing line 0 character 0 of
    /// a source that contains only a newline (`\n`).
    #[test]
    fn pos_to_index_newline_only_l0_c0() {
        let source: Vec<_> = "\n".chars().collect();
        let position = Position {
            line: 0,
            character: 0,
        };

        let out_index = position_to_index(&source, position);
        assert_eq!(out_index, 0);
    }

    /// Ensures `position_to_index` produces the last character index when indexing an out of
    /// bounds line in a source that contains only newlines (`\n`).
    #[test]
    fn pos_to_index_newlines_only_l7_c0() {
        let source: Vec<_> = "\n\n\n".chars().collect();
        let position = Position {
            line: 7,
            character: 0,
        };

        let out_index = position_to_index(&source, position);
        assert_eq!(out_index, 2);
    }

    /// Ensures `position_to_index` gives the last character of the line when indexing an out of
    /// bounds character.
    #[test]
    fn pos_to_index_out_of_bounds_char() {
        let source: Vec<_> = "abc\ndef\nghi\njkl".chars().collect();
        let position = Position {
            line: 3, // "jkl"
            character: 8,
        };

        let out_index = position_to_index(&source, position);
        assert_eq!(source[out_index], 'l');
    }

    #[test]
    fn line_index_position_to_index_uses_utf16_columns() {
        let source: Vec<_> = "😀 pred".chars().collect();
        let line_index = LineIndex::new(&source);
        let position = Position {
            line: 0,
            character: 7,
        };

        let out_index = line_index.position_to_index(&source, position);

        assert_eq!(out_index, source.len());
    }

    #[test]
    fn line_index_position_after_non_bmp_character() {
        let source: Vec<_> = "a😀b".chars().collect();
        let line_index = LineIndex::new(&source);
        let position = Position {
            line: 0,
            character: 3,
        };

        let out_index = line_index.position_to_index(&source, position);

        assert_eq!(source[out_index], 'b');
    }

    #[test]
    fn position_to_index_uses_utf16_columns() {
        let source: Vec<_> = "a😀b".chars().collect();
        let position = Position {
            line: 0,
            character: 3,
        };

        let out_index = position_to_index(&source, position);

        assert_eq!(source[out_index], 'b');
    }
}
