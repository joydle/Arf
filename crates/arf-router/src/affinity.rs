//! Affinity-key extraction from request bodies.
//!
//! The affinity key is a stable hash of the *shared prefix* of a request —
//! the part that is likely already resident in a replica's KV cache:
//!
//! - `/v1/chat/completions`: `model` + concatenation of all messages
//!   **except the last** (the conversation history is the cached prefix;
//!   only the newest user turn is new), capped to 4 KiB to bound hashing cost.
//! - `/v1/completions`: `model` + first 512 bytes of the `prompt` string
//!   (the shared preamble; the tail is the unique suffix).
//!
//! If the body doesn't parse as expected JSON, returns `None` → the caller
//! falls back to pure least-outstanding-requests routing.  Affinity failures
//! are silent; they never cause a request to be rejected.
//!
//! ## Hash stability
//!
//! Uses `std::collections::hash_map::DefaultHasher` with explicit mixing
//! (hash each field separately) to produce a `u64` key. `DefaultHasher`'s
//! seed is randomised per-process by std (since Rust 1.36), so affinity
//! assignments are consistent *within a router process lifetime* but reset
//! on restart — which is correct: after a restart the router's in-memory
//! key→replica mapping resets anyway, and replicas' KV caches are still warm
//! so the new mapping just takes a short warm-up period to re-converge.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use bytes::Bytes;

/// Maximum bytes of the conversation history to include in the affinity hash.
/// Caps the cost of hashing a very long conversation while still capturing
/// the system prompt (the dominant shared prefix in dev fleets).
const CHAT_PREFIX_CAP: usize = 4 * 1024; // 4 KiB

/// Maximum bytes of a `/v1/completions` prompt to include in the hash.
/// The shared preamble in most fleets fits comfortably in 512 chars.
const COMPLETIONS_PROMPT_CAP: usize = 512;

/// Path discriminant — determines which extraction path to take.
enum RequestKind {
    ChatCompletions,
    Completions,
    /// Any other path: no affinity.
    Other,
}

impl From<&str> for RequestKind {
    fn from(path: &str) -> Self {
        if path == "/v1/chat/completions" {
            RequestKind::ChatCompletions
        } else if path == "/v1/completions" {
            RequestKind::Completions
        } else {
            RequestKind::Other
        }
    }
}

/// Extract an affinity key from the request body.
///
/// Returns `None` if:
/// - The path has no affinity (not a completions endpoint), OR
/// - The body is not valid JSON, OR
/// - The expected fields are absent / wrong type.
///
/// The caller must treat `None` as "no affinity" (use LOR).
pub fn extract_affinity_key(path: &str, body: &Bytes) -> Option<u64> {
    match RequestKind::from(path) {
        RequestKind::ChatCompletions => extract_chat_key(body),
        RequestKind::Completions => extract_completions_key(body),
        RequestKind::Other => None,
    }
}

/// Extract the `model` field from a completions request body.
///
/// Used by `proxy.rs` to scope replica selection to the model pool.
/// Returns `None` if the body isn't valid JSON or has no `model` string field —
/// the caller falls back to all-healthy routing in that case (back-compat).
pub fn extract_model(body: &Bytes) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v.get("model")?.as_str().map(|s| s.to_string())
}

/// `/v1/chat/completions`: hash `model` + all-but-last message contents,
/// capped to `CHAT_PREFIX_CAP` bytes.
fn extract_chat_key(body: &Bytes) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let model = v.get("model")?.as_str()?;
    let messages = v.get("messages")?.as_array()?;

    let mut hasher = DefaultHasher::new();
    model.hash(&mut hasher);

    // All messages except the last form the "conversation history" prefix.
    // The last message (the new user turn) varies per request.
    let prefix_messages = if messages.len() > 1 {
        &messages[..messages.len() - 1]
    } else {
        // Only one message (system prompt only, or single-turn): hash it anyway —
        // it IS the shared prefix if all requests start with the same system prompt.
        &messages[..]
    };

    let mut total_bytes = 0usize;
    'outer: for msg in prefix_messages {
        // Hash role + content for each message.
        if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
            role.hash(&mut hasher);
        }
        // Content can be a string or an array of content parts (vision).
        match msg.get("content") {
            Some(serde_json::Value::String(s)) => {
                // Hash a byte prefix — slicing &str by a raw cap could land
                // mid-UTF-8-char and panic (e.g. a prompt with emoji/CJK crossing
                // the cap); bytes have no char-boundary requirement and hashing
                // only needs a stable prefix, not valid UTF-8.
                let take = s.len().min(CHAT_PREFIX_CAP.saturating_sub(total_bytes));
                s.as_bytes()[..take].hash(&mut hasher);
                total_bytes += take;
                if total_bytes >= CHAT_PREFIX_CAP {
                    break 'outer;
                }
            }
            Some(serde_json::Value::Array(parts)) => {
                for part in parts {
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        let take = text.len().min(CHAT_PREFIX_CAP.saturating_sub(total_bytes));
                        text.as_bytes()[..take].hash(&mut hasher);
                        total_bytes += take;
                        if total_bytes >= CHAT_PREFIX_CAP {
                            break 'outer;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Some(hasher.finish())
}

/// `/v1/completions`: hash `model` + first `COMPLETIONS_PROMPT_CAP` bytes of `prompt`.
fn extract_completions_key(body: &Bytes) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let model = v.get("model")?.as_str()?;
    let prompt = v.get("prompt")?.as_str()?;

    let mut hasher = DefaultHasher::new();
    model.hash(&mut hasher);

    // Byte prefix (see the chat path): &str[..cap] could split a UTF-8 char and
    // panic on a non-ASCII prompt crossing the cap; bytes are panic-safe + stable.
    let take = prompt.len().min(COMPLETIONS_PROMPT_CAP);
    prompt.as_bytes()[..take].hash(&mut hasher);

    Some(hasher.finish())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn b(s: &str) -> Bytes {
        Bytes::copy_from_slice(s.as_bytes())
    }

    /// Regression: non-ASCII content whose multi-byte chars straddle the cap
    /// boundary must NOT panic (raw `&str[..cap]` would split a UTF-8 char).
    /// Real conversations contain emoji / CJK / accents; this exercises both the
    /// chat history cap (4 KiB) and the completions prompt cap (512 B).
    #[test]
    fn non_ascii_crossing_the_cap_does_not_panic() {
        // Chat: a system message of CJK that overflows the 4 KiB cap mid-char.
        let big_cjk: String = "你好世界".repeat(2000); // ~24 KB of 3-byte chars
        let chat = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": big_cjk},
                {"role": "user", "content": "hi"}
            ]
        })
        .to_string();
        assert!(extract_affinity_key("/v1/chat/completions", &b(&chat)).is_some());

        // Completions: a prompt of emoji (4-byte chars) crossing the 512 B cap.
        let big_emoji: String = "🚀".repeat(500); // 2000 bytes of 4-byte chars
        let comp = serde_json::json!({ "model": "m", "prompt": big_emoji }).to_string();
        assert!(extract_affinity_key("/v1/completions", &b(&comp)).is_some());

        // Exact boundary: 511 ASCII + a 2-byte char at byte 511..513, cap 512.
        let edge = format!("{}é", "a".repeat(511));
        let comp2 = serde_json::json!({ "model": "m", "prompt": edge }).to_string();
        assert!(extract_affinity_key("/v1/completions", &b(&comp2)).is_some());
    }

    /// Same system prompt + different last turn → same affinity key.
    #[test]
    fn chat_same_prefix_different_last_turn_yields_same_key() {
        let body_a = b(r#"{
            "model": "qwen3-30b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ]
        }"#);
        let body_b = b(r#"{
            "model": "qwen3-30b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum entanglement in detail."}
            ]
        }"#);

        let key_a = extract_affinity_key("/v1/chat/completions", &body_a);
        let key_b = extract_affinity_key("/v1/chat/completions", &body_b);

        assert!(key_a.is_some());
        assert_eq!(
            key_a, key_b,
            "same system prompt + different last turn must yield the same affinity key"
        );
    }

    /// Different system prompts → different affinity keys.
    #[test]
    fn chat_different_system_prompt_yields_different_key() {
        let body_a = b(r#"{
            "model": "qwen3-30b",
            "messages": [
                {"role": "system", "content": "You are a pirate. Arrr!"},
                {"role": "user", "content": "Hello"}
            ]
        }"#);
        let body_b = b(r#"{
            "model": "qwen3-30b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
        }"#);

        let key_a = extract_affinity_key("/v1/chat/completions", &body_a);
        let key_b = extract_affinity_key("/v1/chat/completions", &body_b);

        assert_ne!(
            key_a, key_b,
            "different system prompts must yield different affinity keys"
        );
    }

    /// Different models → different keys (even with same messages).
    #[test]
    fn chat_different_model_yields_different_key() {
        let body_a = b(r#"{"model":"model-a","messages":[{"role":"user","content":"hi"}]}"#);
        let body_b = b(r#"{"model":"model-b","messages":[{"role":"user","content":"hi"}]}"#);

        assert_ne!(
            extract_affinity_key("/v1/chat/completions", &body_a),
            extract_affinity_key("/v1/chat/completions", &body_b)
        );
    }

    /// Unparseable body → None (must not panic or return an error).
    #[test]
    fn chat_unparseable_body_yields_none() {
        let bad = b("not json at all {{{{");
        assert!(extract_affinity_key("/v1/chat/completions", &bad).is_none());
    }

    /// Missing `messages` field → None.
    #[test]
    fn chat_missing_messages_yields_none() {
        let body = b(r#"{"model":"x"}"#);
        assert!(extract_affinity_key("/v1/chat/completions", &body).is_none());
    }

    /// `/v1/completions`: same model + same prompt prefix → same key.
    #[test]
    fn completions_same_prefix_same_key() {
        let shared_prefix = "The following is a list of countries: France, Germany, Spain";
        let body_a = format!(
            r#"{{"model":"qwen3-30b","prompt":"{shared_prefix} and their capitals: Paris,"}}"#
        );
        let body_b = format!(
            r#"{{"model":"qwen3-30b","prompt":"{shared_prefix} and their GDP per capita:"}}"#
        );

        // Both prompts start with the same COMPLETIONS_PROMPT_CAP bytes (they're short),
        // so the tail doesn't affect the key if the prefix is identical.
        // Here the full prompt fits in 512 chars but differs after the prefix,
        // so we use two prompts where the first 512 chars are identical.
        let long_prefix: String = "A".repeat(COMPLETIONS_PROMPT_CAP);
        let body_c = format!(r#"{{"model":"qwen3-30b","prompt":"{long_prefix}SUFFIX_C"}}"#);
        let body_d = format!(r#"{{"model":"qwen3-30b","prompt":"{long_prefix}SUFFIX_D"}}"#);

        let key_c = extract_affinity_key("/v1/completions", &b(&body_c));
        let key_d = extract_affinity_key("/v1/completions", &b(&body_d));
        assert_eq!(key_c, key_d, "same first-512-bytes must yield same key");

        // Sanity: short distinct prompts give different keys.
        let key_a = extract_affinity_key("/v1/completions", &b(&body_a));
        let key_b_val = extract_affinity_key("/v1/completions", &b(&body_b));
        assert_ne!(key_a, key_b_val);
    }

    /// `/v1/completions` unparseable body → None.
    #[test]
    fn completions_unparseable_yields_none() {
        assert!(extract_affinity_key("/v1/completions", &b("garbage")).is_none());
    }

    /// Unknown path → None.
    #[test]
    fn unknown_path_yields_none() {
        let body = b(r#"{"model":"x","messages":[]}"#);
        assert!(extract_affinity_key("/v1/embeddings", &body).is_none());
    }

    /// Determinism: calling the same function twice returns the same value.
    #[test]
    fn key_extraction_is_deterministic() {
        let body = b(r#"{
            "model": "qwen3-30b",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Tell me a joke."}
            ]
        }"#);

        let k1 = extract_affinity_key("/v1/chat/completions", &body);
        let k2 = extract_affinity_key("/v1/chat/completions", &body);
        assert_eq!(k1, k2, "key extraction must be deterministic");
    }
}
