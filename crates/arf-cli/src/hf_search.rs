//! `arf hf-search <query>` (hidden) — query the HuggingFace model search API
//! and print matching repo ids, one per line. Powers `<TAB>` completion of
//! `arf pull <partial>`. Best-effort: any failure prints nothing, exits 0, so
//! a completion call can never hang or error the user's shell.

use std::time::Duration;

/// Extract repo ids (`"id":"org/model"`) from an HF `/api/models` JSON array.
/// Order-preserving, deduplicated. Tolerant of whitespace; a tiny scan, no deps.
pub fn extract_model_ids(json: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let bytes = json.as_bytes();
    let pat = b"\"id\"";
    let mut i = 0;
    while let Some(pos) = find(&bytes[i..], pat) {
        let mut j = i + pos + pat.len();
        // skip spaces and the colon
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b':') {
            j += 1;
        }
        if j < bytes.len() && bytes[j] == b'"' {
            j += 1;
            let start = j;
            while j < bytes.len() && bytes[j] != b'"' {
                j += 1;
            }
            if j <= bytes.len() {
                let id = &json[start..j];
                if !id.is_empty() && seen.insert(id.to_string()) {
                    out.push(id.to_string());
                }
            }
        }
        i = j.max(i + pos + pat.len());
    }
    out
}

/// Substring search (byte). Small helper for extract_model_ids.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Query HF and return repo ids (best-effort; empty on any failure / short query).
pub fn search(query: &str) -> Vec<String> {
    if query.chars().count() < 2 {
        return Vec::new();
    }
    let url = format!(
        "https://huggingface.co/api/models?search={}&limit=15&sort=downloads&direction=-1",
        urlencode(query)
    );
    // Bound BOTH the connect and the overall request so a hung network can never
    // block completion forever. `.timeout` is the whole-request cap (ureq 2.x);
    // `.timeout_connect` covers a stalled TCP/TLS handshake (not covered by it).
    let agent = ureq::builder()
        .timeout_connect(Duration::from_secs(2))
        .timeout(Duration::from_secs(2))
        .build();
    match agent.get(&url).call() {
        Ok(resp) => match resp.into_string() {
            Ok(body) => extract_model_ids(&body),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    }
}

/// Minimal percent-encoding for the query (alnum, -, _, ., / pass through).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'/' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extracts_ids_in_order_deduped() {
        let json = r#"[{"_id":"x","id":"google/gemma-3-4b-it","likes":5},{"id":"unsloth/gemma-2-2b-it"},{"id":"google/gemma-3-4b-it"}]"#;
        let ids = extract_model_ids(json);
        assert_eq!(ids, vec!["google/gemma-3-4b-it", "unsloth/gemma-2-2b-it"]);
    }
    #[test]
    fn empty_or_garbage_json_yields_nothing() {
        assert!(extract_model_ids("").is_empty());
        assert!(extract_model_ids("not json").is_empty());
        assert!(extract_model_ids("[]").is_empty());
    }
    #[test]
    fn short_query_does_not_search() {
        assert!(search("g").is_empty()); // <2 chars: no network call, empty
    }
}
