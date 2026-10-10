//! Stop strings: OpenAI `stop` (`/v1/chat/completions`, `/v1/completions`) and Anthropic
//! `stop_sequences` (`/v1/messages`). Until then both were accepted and ignored.
//!
//! Two halves, because the scheduler holds token ids and the reply is text:
//! - [`StopStrings::check`] is what the scheduler runs after every generated token
//!   (`arf_core::scheduler::StopCheck`). When the newest token's text completes a stop string the
//!   sequence retires right there — on the stop-token path, so a speculative run is truncated at
//!   that token and the KV freed — with `FinishReason::StopString`, and that token is still sent.
//! - The HTTP side cuts the text: [`StopStrings::cut`] on a finished reply, [`StopFilter`] on a
//!   stream, which holds back the last `longest stop - 1` bytes so a stop string split across
//!   chunks is never sent, and flushes them when the reply ends without one.
//!
//! The matched stop string is not part of the reply. A match is searched for only where the newly
//! decoded text could complete one — the new text plus `len - 1` bytes of what came before, as
//! vLLM's `check_stop_strings` does. When one step appends several tokens (speculation), the
//! scheduler still asks token by token, and within one piece of text the EARLIEST-ending match
//! wins (on a tie, the one that starts first), so the reply is the one token-by-token decoding
//! gives. The text matched is the decoded text (`skip_special_tokens`), reasoning included: a stop
//! string that occurs in the reasoning ends the reply there, as vLLM's does.

use std::sync::Arc;

use arf_core::scheduler::StopCheck;
use arf_core::Tokenizer;

/// OpenAI's limit on `stop`: up to four sequences.
pub const OPENAI_MAX_STOP: usize = 4;

/// Tokens of context the scheduler-side check decodes before the newest token, beyond the
/// longest stop string's length in bytes — a token decodes to at least a byte except a lone
/// special token, so this always covers the `len - 1` bytes a match can start back.
const CHECK_SLACK_TOKENS: usize = 16;

/// A request's stop strings: non-empty, deduplicated, in the order sent. Empty = none.
#[derive(Clone, Debug, Default)]
pub struct StopStrings(Arc<[String]>);

impl StopStrings {
    /// The non-empty strings of `stops`, each once (an empty string would match everywhere;
    /// OpenAI and vLLM ignore it).
    pub fn new(stops: impl IntoIterator<Item = String>) -> Self {
        let mut v: Vec<String> = Vec::new();
        for s in stops {
            if !s.is_empty() && !v.contains(&s) {
                v.push(s);
            }
        }
        StopStrings(v.into())
    }

    /// OpenAI's `stop`: absent, `null`, a string, or an array of strings — at most
    /// [`OPENAI_MAX_STOP`] once empty strings are dropped. Anything else is the caller's 400.
    pub fn from_openai(stop: Option<&serde_json::Value>) -> Result<Self, String> {
        use serde_json::Value;
        let list: Vec<String> = match stop {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or("`stop` must be a string or an array of strings")?,
            Some(_) => return Err("`stop` must be a string or an array of strings".into()),
        };
        let stops = Self::new(list);
        if stops.0.len() > OPENAI_MAX_STOP {
            return Err(format!(
                "`stop` takes at most {OPENAI_MAX_STOP} sequences, got {}",
                stops.0.len()
            ));
        }
        Ok(stops)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The longest stop string, in bytes.
    fn max_len(&self) -> usize {
        self.0.iter().map(String::len).max().unwrap_or(0)
    }

    /// The earliest stop-string match in `text` that ends after byte `from` (where the new text
    /// starts; everything before it was searched already): `(start byte, index into the
    /// strings)`. Earliest END wins — the point where the text first contains a stop string —
    /// and on a tie the match that starts first.
    pub fn find(&self, text: &str, from: usize) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, usize)> = None; // (end, start, index)
        for (i, s) in self.0.iter().enumerate() {
            // The first byte a match ending after `from` can start at.
            let mut lo = (from + 1).saturating_sub(s.len()).min(text.len());
            while !text.is_char_boundary(lo) {
                lo -= 1;
            }
            while let Some(at) = text[lo..].find(s.as_str()) {
                let start = lo + at;
                let end = start + s.len();
                if end > from {
                    if best.is_none_or(|(e, st, _)| (end, start) < (e, st)) {
                        best = Some((end, start, i));
                    }
                    break;
                }
                // Already searched (only reachable when `from` was rounded down): look on.
                lo = start + text[start..].chars().next().map_or(1, char::len_utf8);
            }
        }
        best.map(|(_, start, i)| (start, i))
    }

    /// A finished reply cut at its first stop string: the text before it, and the string matched.
    pub fn cut<'a>(&self, text: &'a str) -> (&'a str, Option<&str>) {
        match self.find(text, 0) {
            Some((start, i)) => (&text[..start], Some(self.0[i].as_str())),
            None => (text, None),
        }
    }

    /// The scheduler-side check (`Request::with_stop_check`): did the newest generated token's
    /// text complete a stop string? `None` when there are none, so a request without stop strings
    /// costs nothing. Stateless: it decodes a window of the last tokens twice, without and with
    /// the newest, and searches only where the second adds to the first — an incomplete UTF-8
    /// character (U+FFFD) is completed by a later token and searched then.
    pub fn check(&self, tok: Arc<Tokenizer>) -> Option<StopCheck> {
        if self.is_empty() {
            return None;
        }
        let stops = self.clone();
        let window = stops.max_len() + CHECK_SLACK_TOKENS;
        Some(StopCheck::new(move |out: &[u32]| {
            let Some(newest) = out.len().checked_sub(1) else {
                return false;
            };
            let first = newest.saturating_sub(window);
            let before = decode(&tok, &out[first..newest]);
            let now = decode(&tok, &out[first..]);
            let mut from = before
                .bytes()
                .zip(now.bytes())
                .take_while(|(a, b)| a == b)
                .count();
            while !now.is_char_boundary(from) {
                from -= 1;
            }
            stops.find(&now, from).is_some()
        }))
    }
}

fn decode(tok: &Tokenizer, ids: &[u32]) -> String {
    tok.decode(ids, true).unwrap_or_default()
}

/// A streamed reply's text through the stop strings: what is safe to send now, held back while
/// it could still be the start of a stop string, cut at the first one.
///
/// Each piece pushed carries a tag `K` (which block or field it belongs to — reasoning, answer, a
/// `</think>` marker), and the text comes back out with the tags it went in with, in order: the
/// held-back bytes of a reasoning token are still reasoning when a later token releases them. A
/// piece of empty text is kept as a marker and released in its place.
///
/// Without stop strings every push is returned at once, as it came — the stream is unchanged.
pub struct StopFilter<K = ()> {
    stops: StopStrings,
    /// Text not yet released.
    held: String,
    /// The pieces of `held`: the byte offset each ends at, and its tag.
    tags: Vec<(usize, K)>,
    /// The index of the stop string matched; nothing more is released after it.
    matched: Option<usize>,
}

impl<K: Copy> StopFilter<K> {
    pub fn new(stops: StopStrings) -> Self {
        StopFilter {
            stops,
            held: String::new(),
            tags: Vec::new(),
            matched: None,
        }
    }

    /// Add the next piece of text; returns what may be sent now.
    pub fn push(&mut self, text: &str, tag: K) -> Vec<(String, K)> {
        if self.matched.is_some() {
            return Vec::new();
        }
        if self.stops.is_empty() {
            return vec![(text.to_string(), tag)];
        }
        let from = self.held.len();
        self.held.push_str(text);
        self.tags.push((self.held.len(), tag));
        if let Some((start, i)) = self.stops.find(&self.held, from) {
            self.matched = Some(i);
            let out = self.release(start);
            self.held.clear();
            self.tags.clear();
            return out;
        }
        // Hold back `longest - 1` bytes: the most of a stop string that can precede its end.
        let mut cut = self.held.len().saturating_sub(self.stops.max_len() - 1);
        while !self.held.is_char_boundary(cut) {
            cut -= 1;
        }
        self.release(cut)
    }

    /// The reply ended: everything still held (nothing once a stop string matched).
    pub fn finish(&mut self) -> Vec<(String, K)> {
        if self.matched.is_some() {
            return Vec::new();
        }
        self.release(self.held.len())
    }

    /// The stop string the reply was cut at, if any.
    pub fn matched(&self) -> Option<&str> {
        self.matched.map(|i| self.stops.0[i].as_str())
    }

    /// Release `held[..cut]` piece by piece with its tags; keep the rest.
    fn release(&mut self, cut: usize) -> Vec<(String, K)> {
        let mut out = Vec::new();
        let mut start = 0;
        let mut kept = Vec::new();
        for &(end, tag) in &self.tags {
            if end <= cut {
                out.push((self.held[start..end].to_string(), tag));
            } else {
                if start < cut {
                    out.push((self.held[start..cut].to_string(), tag));
                }
                kept.push((end - cut, tag));
            }
            start = end;
        }
        self.held.drain(..cut);
        self.tags = kept;
        out
    }
}

/// The text of released pieces, tags dropped.
pub fn text_of<K>(pieces: Vec<(String, K)>) -> String {
    pieces.into_iter().map(|(t, _)| t).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(s: &[&str]) -> StopStrings {
        StopStrings::new(s.iter().map(|s| s.to_string()))
    }

    /// Push `pieces` one at a time; the text sent per push, and what `finish` flushed.
    fn run(st: &StopStrings, pieces: &[&str]) -> (Vec<String>, String, Option<String>) {
        let mut f = StopFilter::new(st.clone());
        let sent: Vec<String> = pieces.iter().map(|p| text_of(f.push(p, ()))).collect();
        let tail = text_of(f.finish());
        (sent, tail, f.matched().map(str::to_string))
    }

    #[test]
    fn openai_stop_parses_a_string_a_list_and_null_and_refuses_the_rest() {
        let v = |j: serde_json::Value| StopStrings::from_openai(Some(&j));
        assert_eq!(&*v(serde_json::json!("\n")).unwrap().0, ["\n".to_string()]);
        assert_eq!(
            v(serde_json::json!(["a", "", "b", "a"])).unwrap().0.len(),
            2
        );
        assert!(v(serde_json::json!(null)).unwrap().is_empty());
        assert!(StopStrings::from_openai(None).unwrap().is_empty());
        assert!(v(serde_json::json!(["a", "b", "c", "d"])).is_ok());
        assert!(v(serde_json::json!(["a", "b", "c", "d", "e"])).is_err());
        // Empty strings do not count against the limit — they are ignored.
        assert!(v(serde_json::json!(["a", "b", "c", "d", ""])).is_ok());
        assert!(v(serde_json::json!([1])).is_err());
        assert!(v(serde_json::json!(7)).is_err());
    }

    /// (a) A stop string inside one piece: the text before it is sent, the rest never is.
    #[test]
    fn a_stop_string_inside_one_piece() {
        let (sent, tail, m) = run(&stops(&["."]), &["Hello", " world. More", " text"]);
        assert_eq!(sent.concat(), "Hello world");
        assert_eq!(tail, "");
        assert_eq!(m.as_deref(), Some("."));
        let st = stops(&["world"]);
        assert_eq!(st.cut("Hello world. More"), ("Hello ", Some("world")));
        assert_eq!(st.cut("Hello"), ("Hello", None));
    }

    /// (b) Split across two pieces: the first half is held, never sent.
    #[test]
    fn a_stop_string_split_across_two_pieces() {
        let (sent, tail, m) = run(&stops(&["STOP"]), &["abc ST", "OP def"]);
        // "abc ST": three bytes held (a stop string's longest proper prefix), " ST" among them.
        assert_eq!(sent, vec!["abc".to_string(), " ".to_string()]);
        assert_eq!(tail, "");
        assert_eq!(m.as_deref(), Some("STOP"));
    }

    /// (c) Several tokens appended at once (a speculative run) with two candidate stops: the one
    /// whose match ENDS first wins, whatever order the strings were given in — the result
    /// token-by-token decoding gives.
    #[test]
    fn earliest_completion_wins_in_one_multi_token_piece() {
        let st = stops(&["cd", "b"]);
        // "b" ends at byte 2, "cd" at 4: "b" wins although "cd" is listed first.
        let (sent, _, m) = run(&st, &["abcde"]);
        assert_eq!((sent.concat(), m.as_deref()), ("a".to_string(), Some("b")));
        // Token by token the same.
        let (sent, _, m) = run(&st, &["a", "b", "c", "d", "e"]);
        assert_eq!((sent.concat(), m.as_deref()), ("a".to_string(), Some("b")));
        // Same end: the longer (earlier-starting) match.
        let st = stops(&["b", "ab"]);
        assert_eq!(st.cut("xab"), ("x", Some("ab")));
        // Overlapping occurrences: the search starts `len - 1` back, not after a stale match.
        assert_eq!(stops(&["aa"]).find("aaa", 2), Some((1, 0)));
        // A match wholly inside already-searched text is not reported again.
        assert_eq!(stops(&["ab"]).find("ab-x", 3), None);
    }

    /// (d) Streaming holdback: no part of a stop string is ever sent, and a reply that ends
    /// without a match flushes what was held.
    #[test]
    fn holdback_never_sends_a_partial_stop_and_flushes_on_end() {
        let st = stops(&["</answer>", "\n\n"]);
        // Completed piece by piece: what has been sent is at every step a prefix of the reply
        // before the stop string — never a byte of "</answer>".
        let pieces = ["The", " end", " <", "/", "ans", "wer", ">", " more"];
        let (sent, tail, m) = run(&st, &pieces);
        let mut so_far = String::new();
        for s in &sent {
            so_far.push_str(s);
            assert!("The end ".starts_with(so_far.as_str()), "{so_far:?}");
        }
        assert_eq!((so_far.as_str(), tail.as_str()), ("The end ", ""));
        assert_eq!(m.as_deref(), Some("</answer>"));
        // Never completed: everything arrives, the held part flushed by `finish`.
        let pieces = ["The", " end <", "/ans", "wer", " is", " near\n"];
        let (sent, tail, m) = run(&st, &pieces);
        assert_eq!(m, None);
        assert_eq!(tail, "is near\n");
        assert_eq!(sent.concat() + &tail, pieces.concat());
        // Multi-byte text is held and released on character boundaries.
        let (sent, tail, _) = run(&stops(&["ab"]), &["x\u{e9}", "y"]);
        assert_eq!(sent, vec!["x".to_string(), "\u{e9}".to_string()]);
        assert_eq!(tail, "y");
    }

    /// Without stop strings every push comes straight back, empty ones included: the stream is
    /// exactly what it was before stop strings existed.
    #[test]
    fn no_stop_strings_is_a_pass_through() {
        let mut f = StopFilter::new(StopStrings::default());
        assert_eq!(f.push("", 1u8), vec![(String::new(), 1)]);
        assert_eq!(f.push("ab", 2), vec![("ab".to_string(), 2)]);
        assert!(f.finish().is_empty());
    }

    /// Tags come out with the text they went in with, across a release that splits a piece; an
    /// empty marker is released in its place, and not after the stop.
    #[test]
    fn tags_follow_their_text() {
        let mut f = StopFilter::new(stops(&["XYZ"]));
        let mut out = f.push("think", 'r');
        out.extend(f.push("", 'm'));
        out.extend(f.push("an", 'c'));
        out.extend(f.push("sXYZ", 'c'));
        assert_eq!(
            out,
            vec![
                ("thi".to_string(), 'r'),
                ("nk".to_string(), 'r'),
                (String::new(), 'm'),
                ("an".to_string(), 'c'),
                ("s".to_string(), 'c'),
            ]
        );
        assert_eq!(f.matched(), Some("XYZ"));
        assert!(f.push("more", 'c').is_empty());
    }
}
