use harper_core::parsers::{self, Parser, PlainEnglish};
use harper_core::{Token, TokenKind};
use harper_tree_sitter::TreeSitterMasker;
use tree_sitter::Node;

pub struct PythonParser {
    /// Used to grab the text nodes.
    inner: parsers::Mask<TreeSitterMasker, PlainEnglish>,
}

impl PythonParser {
    fn node_condition(n: &Node) -> bool {
        // Skip error nodes - when Tree-sitter can't parse correctly
        if n.is_error() || n.is_missing() || n.kind() == "ERROR" {
            return false;
        }

        if n.kind().contains("comment") {
            return true;
        }
        if n.kind() == "string_content"
            && let Some(expr_stmt) = parent_is_expression_statement(n)
            && (is_module_level_docstring(&expr_stmt)
                || is_fn_or_class_docstrings(&expr_stmt)
                || is_attribute_docstring(&expr_stmt))
        {
            return true;
        }
        false
    }
}

impl Default for PythonParser {
    fn default() -> Self {
        Self {
            inner: parsers::Mask::new(
                TreeSitterMasker::new(tree_sitter_python::LANGUAGE.into(), Self::node_condition),
                PlainEnglish,
            ),
        }
    }
}

impl Parser for PythonParser {
    fn parse(&self, source: &[char]) -> Vec<Token> {
        let mut tokens = self.inner.parse(source);

        let mut prev_kind: Option<&TokenKind> = None;

        for token in &mut tokens {
            if let TokenKind::Space(v) = &mut token.kind {
                if let Some(TokenKind::Newline(_)) = &prev_kind {
                    // Lines in multiline docstrings are indented with spaces to match the current level.
                    // We need to remove such spaces to avoid triggering French spaces rule.
                    *v = 0;
                } else {
                    *v = (*v).clamp(0, 1);
                }
            }

            prev_kind = Some(&token.kind);
        }

        // Detect bullet points and make them act as sentence boundaries
        // Look for patterns like: "- text\n" or "* text\n"
        for i in 0..tokens.len() {
            if let TokenKind::Punctuation(punct) = tokens[i].kind {
                if matches!(punct, harper_core::Punctuation::Hyphen | harper_core::Punctuation::Star) {
                    // Look ahead to find the newline that ends this bullet point
                    for j in (i + 1)..tokens.len() {
                        if matches!(tokens[j].kind, TokenKind::Newline(_)) {
                            // Convert the newline after the bullet point to a ParagraphBreak
                            tokens[j].kind = TokenKind::ParagraphBreak;
                            break;
                        }
                        // If we hit another bullet or paragraph break, stop looking
                        if matches!(tokens[j].kind, TokenKind::ParagraphBreak) {
                            break;
                        }
                        if let TokenKind::Punctuation(p) = tokens[j].kind {
                            if matches!(p, harper_core::Punctuation::Hyphen | harper_core::Punctuation::Star) {
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Make newlines act as sentence boundaries in Python
        // This treats each line as a separate sentence, which is more natural for
        // Python code comments and docstrings
        for token in &mut tokens {
            if matches!(token.kind, TokenKind::Newline(_)) {
                token.kind = TokenKind::ParagraphBreak;
            }
        }

        tokens
    }
}

fn parent_is_expression_statement<'a>(node: &Node<'a>) -> Option<Node<'a>> {
    node.parent()
        .filter(|n| n.kind() == "string")
        .and_then(|string_node| string_node.parent())
        .filter(|n| n.kind() == "expression_statement")
}

#[inline]
fn is_module_level_docstring(expr_stmt: &Node) -> bool {
    // (module . (expression_statement (string)))
    expr_stmt.parent().is_some_and(|n| n.kind() == "module")
}

#[inline]
fn is_fn_or_class_docstrings(expr_stmt: &Node) -> bool {
    // (class/func_definition body: (block . (expression_statement (string))))
    expr_stmt
        .parent()
        .filter(|n| n.kind() == "block")
        .and_then(|n| n.parent())
        .is_some_and(|n| n.kind() == "function_definition" || n.kind() == "class_definition")
}

#[inline]
fn is_attribute_docstring(expr_stmt: &Node) -> bool {
    // ((expression_statement (assignment)) . (expression_statement (string)))
    expr_stmt
        .prev_sibling()
        .filter(|s| s.kind() == "expression_statement")
        .and_then(|s| s.child(0))
        .is_some_and(|c| c.kind() == "assignment")
}
