/// This module implements parsing of file paths (Unix and Windows style).
/// Recognizes patterns like:
/// - Unix absolute: /path/to/file.txt
/// - Unix relative: path/to/file.txt, ./path/to/file.txt, ../path/to/file.txt
/// - Windows: C:\path\to\file.txt, \\network\path
use super::FoundToken;
use crate::TokenKind;

pub fn lex_file_path(source: &[char]) -> Option<FoundToken> {
    // Try Unix-style path first (more common in markdown/documentation)
    if let Some(token) = lex_unix_path(source) {
        return Some(token);
    }

    // Try Windows-style path
    lex_windows_path(source)
}

fn lex_unix_path(source: &[char]) -> Option<FoundToken> {
    if source.is_empty() {
        return None;
    }

    let mut cursor = 0;
    let mut slash_count = 0;

    // Absolute path starting with /
    let is_absolute = source[0] == '/';
    if is_absolute {
        cursor += 1;
        slash_count += 1;
    }

    // Check for relative path indicators: ./ or ../
    let is_dot_relative = !is_absolute && source[0] == '.' && source.len() > 1 && source[1] == '/';
    let is_dotdot_relative = !is_absolute && source.len() > 2
        && source[0] == '.' && source[1] == '.' && source[2] == '/';

    if is_dotdot_relative {
        cursor = 3;
        slash_count += 1;
    } else if is_dot_relative {
        cursor = 2;
        slash_count += 1;
    } else if !is_absolute {
        // For non-absolute, non-dot-relative paths, require at least 2 slashes
        // This prevents false positives like "a.m" from being treated as paths
    }

    // Consume path components separated by /
    let start_cursor = cursor;
    let mut component_start = cursor;

    while cursor < source.len() {
        let c = source[cursor];

        if c == '/' {
            // Don't allow empty components (consecutive slashes)
            if cursor == component_start {
                break;
            }
            slash_count += 1;
            cursor += 1;
            component_start = cursor;
            continue;
        }

        // Valid path characters: alphanumeric, underscore, hyphen, dot
        if c.is_alphanumeric() || matches!(c, '_' | '-' | '.') {
            cursor += 1;
        } else {
            break;
        }
    }

    // Require meaningful path structure
    let path_length = cursor - start_cursor;

    // Must have at least one slash and reasonable content
    if slash_count >= 1 && path_length > 0 {
        // For non-absolute paths without ./ or ../, require at least 2 slashes
        // This prevents "file.txt" from being treated as a path
        if !is_absolute && !is_dot_relative && !is_dotdot_relative && slash_count < 2 {
            return None;
        }

        return Some(FoundToken {
            next_index: cursor,
            token: TokenKind::FilePath,
        });
    }

    None
}

fn lex_windows_path(source: &[char]) -> Option<FoundToken> {
    if source.len() < 3 {
        return None;
    }

    // Check for drive letter: C:\ or network path: \\
    let is_drive = source[0].is_ascii_alphabetic()
        && source[1] == ':'
        && source[2] == '\\';

    let is_network = source[0] == '\\' && source[1] == '\\';

    if !is_drive && !is_network {
        return None;
    }

    let mut cursor = if is_drive { 3 } else { 2 };

    // Consume path components separated by \
    while cursor < source.len() {
        let c = source[cursor];

        if c == '\\' {
            cursor += 1;
            continue;
        }

        // Valid Windows path characters
        if c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | ' ') {
            cursor += 1;
        } else {
            break;
        }
    }

    if cursor > 3 {
        Some(FoundToken {
            next_index: cursor,
            token: TokenKind::FilePath,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_consumes_full(path: &str) {
        assert_consumes_part(path, path.len());
    }

    fn assert_consumes_part(path: &str, len: usize) {
        let path_chars: Vec<_> = path.chars().collect();
        let result = lex_file_path(&path_chars);
        assert!(result.is_some(), "Failed to parse path: {}", path);
        assert_eq!(result.unwrap().next_index, len, "Wrong length for path: {}", path);
    }

    fn assert_doesnt_consume(path: &str) {
        let path_chars: Vec<_> = path.chars().collect();
        assert!(lex_file_path(&path_chars).is_none(), "Incorrectly parsed: {}", path);
    }

    #[test]
    fn consumes_unix_absolute_path() {
        assert_consumes_full("/home/user/file.txt");
    }

    #[test]
    fn consumes_unix_relative_path() {
        assert_consumes_full("path/to/file.txt");
    }

    #[test]
    fn consumes_dot_relative_path() {
        assert_consumes_full("./path/to/file.txt");
    }

    #[test]
    fn consumes_dotdot_relative_path() {
        assert_consumes_full("../path/to/file.txt");
    }

    #[test]
    fn consumes_python_path() {
        assert_consumes_full("extensions/houdini/tools/python/cshoudini/validate/sanity/test/corrupted_fetch_test.py");
    }

    #[test]
    fn consumes_path_with_underscores() {
        assert_consumes_full("my_project/src/test_file.rs");
    }

    #[test]
    fn consumes_windows_drive_path() {
        assert_consumes_full("C:\\Users\\Documents\\file.txt");
    }

    #[test]
    fn consumes_windows_network_path() {
        assert_consumes_full("\\\\server\\share\\file.txt");
    }

    #[test]
    fn stops_at_whitespace() {
        assert_consumes_part("path/to/file.txt and more text", 16);
    }

    #[test]
    fn doesnt_consume_single_word() {
        assert_doesnt_consume("word");
    }

    #[test]
    fn doesnt_consume_just_slash() {
        assert_doesnt_consume("/");
    }

    #[test]
    fn doesnt_consume_simple_file_with_extension() {
        // Single files without path separators should not be treated as paths
        // to avoid false positives like "a.m." being treated as a path
        assert_doesnt_consume("readme.md");
    }

    #[test]
    fn doesnt_consume_abbreviated_time() {
        // Ensure we don't match things like "4.a.m." or "p.m."
        assert_doesnt_consume("a.m.");
        assert_doesnt_consume("p.m.");
    }
}
