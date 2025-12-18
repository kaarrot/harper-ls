use harper_core::parsers::{Markdown, StrParser};

#[test]
fn deeply_indented_file_performs_well() {
    // Create a file with deep indentation similar to the issue
    let mut content = String::new();
    
    // Create 100 lines with progressively deeper indentation
    for i in 0..100 {
        let indent = "\t".repeat(i);
        content.push_str(&format!("{}impl DocumentState {{\n", indent));
    }
    
    let parser = Markdown::default();
    
    let start = std::time::Instant::now();
    let _tokens = parser.parse_str(&content);
    let elapsed = start.elapsed();
    
    // This should complete quickly (under 100ms even on slow machines)
    // The old implementation would take much longer due to large vector allocation
    assert!(elapsed.as_millis() < 100, 
        "Parsing took too long: {:?}. This suggests the byte_to_char optimization isn't working.", 
        elapsed);
}
