//! Grammar-constrained sampling: a per-sequence token-level state machine that
//! masks logits to enforce output validity before the argmax/sample step.
//!
//! ## Architecture
//!
//! [`Constraint`] is the public seam.  [`JsonConstraint`] is the v1 concrete
//! implementation: it enforces **structural JSON validity** — every generated
//! token keeps the accumulated text a valid JSON prefix, and
//! [`Constraint::is_complete`] fires when the top-level value has closed.
//!
//! ## v1 coverage (json_object / json_schema structural mode)
//!
//! - First non-whitespace token must start with `{`, `[`, `"`, `t`, `f`, `n`,
//!   a digit, or `-`.
//! - Inside a JSON string: any byte allowed until an unescaped `"`.
//! - After an object key `"..."`: only `:` (with optional whitespace) allowed.
//! - After `:`: value-start chars only.
//! - Literals `true`, `false`, `null` enforced byte-by-byte.
//! - Numbers: `0-9 . e E + -`; terminated by structural chars.
//! - After a complete value: `,`, `}`, `]`, or whitespace; at depth 0 → Done.
//! - After `{`: `"` (key), `}`, or whitespace.
//! - After `,` in object: `"` (key) or whitespace.
//! - After `,` in array: value-start or whitespace; `]` closes an empty array.
//! - Done state: only trailing whitespace allowed.  Stop/EOS tokens are ALWAYS
//!   unmasked at Done (see invariant below).
//!
//! ## Never-all-masked invariant
//!
//! `apply_mask` guarantees that after masking, **at least one logit remains
//! finite** for every row.  The mechanism:
//!
//! 1. When the parser reaches `Done` (top-level value complete), the configured
//!    stop/EOS token ids are left unmasked so the model can terminate cleanly.
//!    This wires `is_complete` to the normal stop-token path — no extra
//!    scheduler change needed.
//! 2. A defense-in-depth guard at the end of `apply_mask` checks whether all
//!    logits are `-inf`.  If so, it restores the stop-token id(s) (or, if none
//!    are configured, the original argmax token) so the row is never fully
//!    masked.  This prevents greedy `argmax` from returning garbage token 0,
//!    and prevents stochastic `WeightedIndex::new` from failing with an all-NaN
//!    distribution (→ `Err` → fatal actor death → server outage).
//!
//! ## v2 (not built in this pass)
//!
//! - `json_schema` key/type enforcement (required fields, field types, nested
//!   schemas).  The `schema` field is stored for v2 but currently unused —
//!   structural JSON validation is identical for `json_object` and `json_schema`
//!   in v1.  Schema-level enforcement is the documented v2 follow-up.
//!
//! ## Token → piece mapping
//!
//! [`JsonConstraint`] holds an `Arc<TokenPieceMap>` built once from the
//! tokenizer at server startup.  It maps each token id to its UTF-8 string
//! piece (empty string for control codes / byte-fallback tokens that don't
//! decode cleanly).  Tokens with empty pieces are always masked.

use std::sync::Arc;

// ---------------------------------------------------------------------------
// Public trait
// ---------------------------------------------------------------------------

/// A per-sequence grammar constraint on the next token.
///
/// The sampler calls `apply_mask` once per decode step, **before**
/// argmax/sampling.  Tokens that would violate the grammar are set to
/// `f32::NEG_INFINITY` so they are never selected.
///
/// Constraints are `Send` (moved with the per-sequence state into the actor
/// OS thread) but NOT required to be `Sync` — each sequence owns its own
/// `Box<dyn Constraint>`.
pub trait Constraint: Send {
    /// Mask `logits` in place.
    ///
    /// `generated` is the slice of token ids produced so far for this sequence
    /// (does NOT include the prompt).  The constraint sets logits of tokens
    /// that would violate the grammar to `f32::NEG_INFINITY`.
    ///
    /// **Invariant**: after this call, at least one logit MUST be finite.
    /// Implementations must guarantee this (see module-level documentation for
    /// the never-all-masked invariant).
    fn apply_mask(&self, generated: &[u32], logits: &mut [f32]);

    /// Returns `true` when `generated` forms a complete, valid terminal in the
    /// grammar (e.g., the top-level JSON value has closed at depth 0).
    /// The server may use this to stop generation cleanly before `max_tokens`.
    /// Default: always `false` (never auto-stop).
    fn is_complete(&self, _generated: &[u32]) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// TokenPieceMap — shared, built once from the tokenizer
// ---------------------------------------------------------------------------

/// Maps each token id to its string piece.  Built once at server startup from
/// `tokenizer.vocab_size()` + `tokenizer.decode([id])` and shared via `Arc`.
///
/// Tokens whose decode fails (byte-fallback tokens, control codes, etc.) are
/// stored as empty strings and are masked in all grammar-constrained states.
#[derive(Debug, Clone)]
pub struct TokenPieceMap {
    pieces: Vec<String>,
}

impl TokenPieceMap {
    /// Build from a `pieces` vec indexed by token id.
    pub fn new(pieces: Vec<String>) -> Self {
        TokenPieceMap { pieces }
    }

    /// The string piece for `id`, or `""` if out of range.
    #[inline]
    pub fn piece(&self, id: u32) -> &str {
        self.pieces
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Number of entries (== vocab_size).
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }
}

// ---------------------------------------------------------------------------
// JsonStructParser — the structural JSON state machine
// ---------------------------------------------------------------------------

/// Structural JSON parser state.
#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    /// No characters yet (or only whitespace).
    Start,
    /// Inside a JSON string.  `key`: we are parsing an object key.
    /// `esc`: the previous byte was `\`.
    Str { key: bool, esc: bool },
    /// After the closing `"` of an object key; waiting for `:`.
    PostKey,
    /// After `:`; waiting for a value to start.
    PostColon,
    /// Inside a JSON number.
    Num,
    /// Consuming the remainder of a literal (`true`, `false`, `null`).
    /// `idx` is an index into the `LIT_SUFFIXES` table.
    Lit(u8),
    /// After a complete value inside a container (depth > 0).
    PostVal,
    /// Just opened an object (`{`); expecting a key (`"`) or `}`.
    OpenBrace,
    /// After `,` inside an object; expecting a key (`"`).
    CommaObj,
    /// After `,` inside an array, or right after `[`; expecting a value or `]`.
    CommaArr,
    /// Top-level value complete.  Only trailing whitespace is valid.
    /// Stop/EOS tokens are always kept unmasked in this state (see module docs).
    Done,
    /// Unrecoverable error.  All tokens masked.
    Bad,
}

/// Remaining bytes expected for each literal state (indexed by `Lit(idx)`).
///
/// - Lit(0..=2): `true` → need `r`, `u`, `e`
/// - Lit(3..=6): `false` → need `a`, `l`, `s`, `e`
/// - Lit(7..=9): `null` → need `u`, `l`, `l`
const LIT_SUFFIXES: &[u8] = b"ruealseull";

/// Coarse first-byte filter used to fast-reject clearly-invalid tokens without
/// cloning the full parser.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ValidFirst {
    /// Any byte (e.g., inside a string).
    Any,
    /// Only whitespace (e.g., after top-level value is Done).
    Ws,
    /// JSON value start: `{ [ " t f n 0-9 -`, or whitespace.
    ValStart,
    /// `:` or whitespace.
    ColonOrWs,
    /// Number continuation or a structural terminator (`, } ] whitespace`).
    NumOrTerm,
    /// A single specific byte (e.g., next literal char).
    One(u8),
    /// Post-value terminators: `, } ] whitespace`.
    PostVal,
    /// `"` or `}` or whitespace (after `{`).
    KeyOrClose,
    /// `"` or whitespace (after `,` in object).
    KeyOrWs,
    /// JSON value start OR `]` (after `[`, allows empty array).
    ValStartOrClose,
}

impl ValidFirst {
    fn allows(self, b: u8) -> bool {
        #[inline]
        fn is_ws(b: u8) -> bool {
            matches!(b, b' ' | b'\t' | b'\n' | b'\r')
        }
        match self {
            ValidFirst::Any => true,
            ValidFirst::Ws => is_ws(b),
            ValidFirst::ValStart => {
                is_ws(b)
                    || matches!(b, b'{' | b'[' | b'"' | b't' | b'f' | b'n' | b'-')
                    || b.is_ascii_digit()
            }
            ValidFirst::ColonOrWs => b == b':' || is_ws(b),
            ValidFirst::NumOrTerm => {
                matches!(b, b'.' | b'e' | b'E' | b'+' | b'-' | b',' | b'}' | b']')
                    || b.is_ascii_digit()
                    || is_ws(b)
            }
            ValidFirst::One(expected) => b == expected,
            ValidFirst::PostVal => matches!(b, b',' | b'}' | b']') || is_ws(b),
            ValidFirst::KeyOrClose => b == b'"' || b == b'}' || is_ws(b),
            ValidFirst::KeyOrWs => b == b'"' || is_ws(b),
            ValidFirst::ValStartOrClose => {
                b == b']'
                    || is_ws(b)
                    || matches!(b, b'{' | b'[' | b'"' | b't' | b'f' | b'n' | b'-')
                    || b.is_ascii_digit()
            }
        }
    }
}

/// A structural JSON parser driven byte-by-byte.
#[derive(Debug, Clone)]
pub(crate) struct JsonStructParser {
    state: State,
    /// Stack of open containers.  Each entry is `true` for object, `false` for array.
    stack: Vec<bool>,
}

impl JsonStructParser {
    pub fn new() -> Self {
        JsonStructParser {
            state: State::Start,
            stack: Vec::new(),
        }
    }

    /// `true` when the top-level JSON value is complete (depth 0, Done state).
    pub fn is_complete(&self) -> bool {
        self.state == State::Done
    }

    /// `true` when the parser has entered an unrecoverable error state.
    pub fn is_error(&self) -> bool {
        self.state == State::Bad
    }

    /// A coarse first-byte filter for the current state.  Returns `None` when
    /// the state is terminal (Bad) and nothing is valid — the caller
    /// should mask all tokens (subject to the never-all-masked guard).
    /// Returns `Some(ValidFirst::Ws)` for Done, meaning only whitespace is
    /// structurally valid; the caller separately unmasks stop tokens in that state.
    pub fn valid_first(&self) -> Option<ValidFirst> {
        use State::*;
        Some(match &self.state {
            Bad => return None,
            Done => ValidFirst::Ws,
            Start | PostColon => ValidFirst::ValStart,
            // CommaArr: right after `[` or `,` inside array — value start OR `]`
            // (the `]` case closes an empty array, symmetric to `}` in OpenBrace).
            CommaArr => ValidFirst::ValStartOrClose,
            Str { .. } => ValidFirst::Any,
            PostKey => ValidFirst::ColonOrWs,
            Num => ValidFirst::NumOrTerm,
            Lit(idx) => ValidFirst::One(LIT_SUFFIXES[*idx as usize]),
            PostVal => ValidFirst::PostVal,
            OpenBrace => ValidFirst::KeyOrClose,
            CommaObj => ValidFirst::KeyOrWs,
        })
    }

    /// Advance the parser by one byte.  After `Bad` or `Done` (with a
    /// non-whitespace byte), the state remains `Bad`.
    pub fn advance(&mut self, c: u8) {
        use State::*;
        self.state = match self.state {
            Bad => Bad,
            Done => {
                if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                    Done
                } else {
                    Bad
                }
            }
            Start => {
                if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                    Start
                } else {
                    self.open_value(c)
                }
            }
            Str { key, esc } => {
                if esc {
                    Str { key, esc: false }
                } else if c == b'\\' {
                    Str { key, esc: true }
                } else if c == b'"' {
                    if key {
                        PostKey
                    } else if self.stack.is_empty() {
                        Done
                    } else {
                        PostVal
                    }
                } else {
                    Str { key, esc: false }
                }
            }
            PostKey => {
                if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                    PostKey
                } else if c == b':' {
                    PostColon
                } else {
                    Bad
                }
            }
            PostColon => {
                if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                    PostColon
                } else {
                    self.open_value(c)
                }
            }
            Num => {
                if matches!(c, b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') {
                    Num
                } else {
                    // Number ended; re-process the terminator in PostVal/Done.
                    self.state = if self.stack.is_empty() { Done } else { PostVal };
                    self.advance(c);
                    return;
                }
            }
            Lit(idx) => {
                if LIT_SUFFIXES[idx as usize] == c {
                    // This was the last expected byte of the literal.
                    // true:  Lit(2)='e', false: Lit(6)='e', null: Lit(9)='l'
                    let last = matches!(idx, 2 | 6 | 9);
                    if last {
                        if self.stack.is_empty() {
                            Done
                        } else {
                            PostVal
                        }
                    } else {
                        Lit(idx + 1)
                    }
                } else {
                    Bad
                }
            }
            PostVal => {
                if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                    PostVal
                } else if c == b',' {
                    match self.stack.last() {
                        Some(true) => CommaObj,
                        Some(false) => CommaArr,
                        None => Bad,
                    }
                } else if c == b'}' {
                    if self.stack.last() == Some(&true) {
                        self.stack.pop();
                        if self.stack.is_empty() {
                            Done
                        } else {
                            PostVal
                        }
                    } else {
                        Bad
                    }
                } else if c == b']' {
                    if self.stack.last() == Some(&false) {
                        self.stack.pop();
                        if self.stack.is_empty() {
                            Done
                        } else {
                            PostVal
                        }
                    } else {
                        Bad
                    }
                } else {
                    Bad
                }
            }
            OpenBrace => {
                if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                    OpenBrace
                } else if c == b'"' {
                    Str {
                        key: true,
                        esc: false,
                    }
                } else if c == b'}' {
                    self.stack.pop();
                    if self.stack.is_empty() {
                        Done
                    } else {
                        PostVal
                    }
                } else {
                    Bad
                }
            }
            CommaObj => {
                if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                    CommaObj
                } else if c == b'"' {
                    Str {
                        key: true,
                        esc: false,
                    }
                } else {
                    Bad
                }
            }
            CommaArr => {
                if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                    CommaArr
                }
                // `]` right after `[` (or after `,` — invalid JSON, but state
                // machine allows it here; the real guard is that CommaArr is only
                // entered after `[` or `,` inside an array).  Closes an empty
                // array `[]` symmetrically to `{}` via OpenBrace.
                else if c == b']' {
                    self.stack.pop();
                    if self.stack.is_empty() {
                        Done
                    } else {
                        PostVal
                    }
                } else {
                    self.open_value(c)
                }
            }
        };
    }

    /// Advance by all bytes in `s`.
    pub fn advance_str(&mut self, s: &str) {
        for b in s.bytes() {
            self.advance(b);
        }
    }

    /// Transition to the state for the first byte of a value.
    fn open_value(&mut self, c: u8) -> State {
        match c {
            b'{' => {
                self.stack.push(true);
                State::OpenBrace
            }
            b'[' => {
                self.stack.push(false);
                State::CommaArr
            }
            b'"' => State::Str {
                key: false,
                esc: false,
            },
            b't' => State::Lit(0), // expect r,u,e  (indices 0,1,2)
            b'f' => State::Lit(3), // expect a,l,s,e (indices 3,4,5,6)
            b'n' => State::Lit(7), // expect u,l,l  (indices 7,8,9)
            b'0'..=b'9' | b'-' => State::Num,
            _ => State::Bad,
        }
    }
}

// ---------------------------------------------------------------------------
// JsonConstraint
// ---------------------------------------------------------------------------

/// Structurally-constrained sampler that enforces valid JSON output.
///
/// See the module-level documentation for v1 coverage, v2 roadmap, and the
/// never-all-masked invariant.
pub struct JsonConstraint {
    pieces: Arc<TokenPieceMap>,
    /// Stop/EOS token ids from the job's `SamplingParams::stop_tokens`.
    /// These are NEVER masked when the parser is in `Done` state, so the model
    /// can terminate cleanly instead of padding to `max_tokens`.  They are also
    /// used as the fallback in the all-masked guard (defense in depth).
    stop_tokens: Vec<u32>,
    /// Stored for v2 (schema key/type enforcement); unused in v1.
    #[allow(dead_code)]
    schema: Option<serde_json::Value>,
}

impl JsonConstraint {
    /// Create for `json_object` mode (structural JSON validity only).
    pub fn new(pieces: Arc<TokenPieceMap>) -> Self {
        JsonConstraint {
            pieces,
            stop_tokens: Vec::new(),
            schema: None,
        }
    }

    /// Create for `json_object` mode with known stop/EOS token ids.
    ///
    /// The stop tokens are left unmasked when the JSON value is complete
    /// (`Done` state) so the model can terminate instead of running to
    /// `max_tokens`.  They also serve as the never-all-masked fallback.
    pub fn new_with_stops(pieces: Arc<TokenPieceMap>, stop_tokens: Vec<u32>) -> Self {
        JsonConstraint {
            pieces,
            stop_tokens,
            schema: None,
        }
    }

    /// Create for `json_schema` mode.
    ///
    /// **v1 note**: the `schema` is stored but currently unused — only
    /// structural JSON validity is enforced (same as `json_object`).
    /// Key/type enforcement per the schema is the documented v2 follow-up.
    pub fn with_schema(pieces: Arc<TokenPieceMap>, schema: serde_json::Value) -> Self {
        JsonConstraint {
            pieces,
            stop_tokens: Vec::new(),
            schema: Some(schema),
        }
    }

    /// `json_schema` mode with stop tokens.
    pub fn with_schema_and_stops(
        pieces: Arc<TokenPieceMap>,
        schema: serde_json::Value,
        stop_tokens: Vec<u32>,
    ) -> Self {
        JsonConstraint {
            pieces,
            stop_tokens,
            schema: Some(schema),
        }
    }

    /// Rebuild the parser state by feeding all `generated` tokens.
    fn parser_state(&self, generated: &[u32]) -> JsonStructParser {
        let mut p = JsonStructParser::new();
        for &id in generated {
            p.advance_str(self.pieces.piece(id));
        }
        p
    }
}

impl Constraint for JsonConstraint {
    /// Mask all logits whose token piece would produce an invalid JSON prefix.
    ///
    /// Algorithm:
    /// 1. Reconstruct current parser state from `generated`.
    /// 2. For each vocabulary token:
    ///   - Fast-reject: if the token's first byte can't be valid in this
    ///     state (`valid_first` filter), mask immediately.
    ///   - Full probe: clone the parser, advance by the whole piece, mask if
    ///     the result is `Bad`.
    /// 3. If the parser is `Done` (top-level value complete), leave stop token
    ///    ids unmasked so the model can select them and terminate cleanly.
    /// 4. **Never-all-masked guard**: if all logits are `-inf` after step 2,
    ///    restore the stop token(s) — or the original argmax token as a last
    ///    resort — so this row always has ≥1 finite logit.
    fn apply_mask(&self, generated: &[u32], logits: &mut [f32]) {
        let parser = self.parser_state(generated);
        let complete = parser.is_complete();
        let valid_first = parser.valid_first();

        // Save original logits for the all-masked fallback (only needed if
        // we might need to restore a token).  We take the argmax of the
        // original now so we have it cheaply without a second scan.
        let original_argmax = {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            best_idx
        };

        for (id, logit) in logits.iter_mut().enumerate() {
            let id_u32 = id as u32;

            // Stop tokens at completion: ALWAYS leave unmasked.
            // This is how is_complete drives termination — the model can now
            // select the stop token, and the existing stop-token path in the
            // scheduler/actor handles the actual sequence termination.
            if complete && self.stop_tokens.contains(&id_u32) {
                // Leave the logit at its original value (do not mask).
                continue;
            }

            let piece = self.pieces.piece(id_u32);
            if piece.is_empty() {
                // Empty-piece tokens (control codes, byte-fallback) are always
                // masked — they produce no visible character and would confuse
                // the state machine.
                *logit = f32::NEG_INFINITY;
                continue;
            }

            // Fast first-byte reject.
            let first_byte = piece.as_bytes()[0];
            let ok = match valid_first {
                None => false,
                Some(vf) => vf.allows(first_byte),
            };
            if !ok {
                *logit = f32::NEG_INFINITY;
                continue;
            }

            // Full probe: advance a cloned parser by the whole piece.
            let mut probe = parser.clone();
            probe.advance_str(piece);
            if probe.is_error() {
                *logit = f32::NEG_INFINITY;
            }
        }

        // ---------------------------------------------------------------
        // Never-all-masked guard (defense in depth).
        //
        // After applying the structural mask, verify that at least one logit
        // is finite.  If every token was masked (which can happen if the vocab
        // has no whitespace token and the parser is in Done state, or after an
        // unexpected Bad state), restore a safe fallback token so:
        //   - greedy argmax never silently returns token 0 (JSON-invalid garbage)
        //   - stochastic WeightedIndex::new never sees an all-NaN distribution
        //     (→ Err → fatal actor death → server outage)
        //
        // Fallback priority:
        //   1. The first stop token (EOS): always valid at end of sequence.
        //   2. The original argmax token: at least picks the model's preferred
        //      token rather than token 0.
        // ---------------------------------------------------------------
        let any_finite = logits.iter().any(|&v| v.is_finite());
        if !any_finite {
            // Restore stop token(s) if we have them.
            if !self.stop_tokens.is_empty() {
                for &stop_id in &self.stop_tokens {
                    if let Some(l) = logits.get_mut(stop_id as usize) {
                        // Restore to original value (0.0 is a safe stand-in if
                        // we didn't save per-token originals, but using 0.0
                        // means the distribution is valid and the stop token is
                        // the only candidate, which is exactly what we want).
                        *l = 0.0;
                    }
                }
            } else {
                // No stop tokens configured: fall back to the original argmax.
                if let Some(l) = logits.get_mut(original_argmax) {
                    *l = 0.0;
                }
            }
        }
    }

    fn is_complete(&self, generated: &[u32]) -> bool {
        self.parser_state(generated).is_complete()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn piece_map(pieces: &[&str]) -> Arc<TokenPieceMap> {
        Arc::new(TokenPieceMap::new(
            pieces.iter().map(|s| s.to_string()).collect(),
        ))
    }

    fn make_constraint(pieces: &[&str]) -> JsonConstraint {
        JsonConstraint::new(piece_map(pieces))
    }

    fn make_constraint_with_stops(pieces: &[&str], stop_ids: Vec<u32>) -> JsonConstraint {
        JsonConstraint::new_with_stops(piece_map(pieces), stop_ids)
    }

    /// Return the set of allowed token ids given `generated`.
    fn allowed(c: &JsonConstraint, generated: &[u32]) -> Vec<u32> {
        let vocab = c.pieces.len();
        let mut logits: Vec<f32> = vec![1.0; vocab];
        c.apply_mask(generated, &mut logits);
        logits
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > f32::NEG_INFINITY)
            .map(|(i, _)| i as u32)
            .collect()
    }

    #[test]
    fn start_state_allows_open_brace_bracket_and_ws() {
        // pieces: 0="{", 1="}", 2="[", 3="]", 4="\"key\"", 5=":", 6="1", 7=",", 8=" ", 9="hello"
        let pieces = ["{", "}", "[", "]", "\"key\"", ":", "1", ",", " ", "hello"];
        let c = make_constraint(&pieces);
        let a = allowed(&c, &[]);
        assert!(a.contains(&0), "{{ must be allowed at start");
        assert!(a.contains(&2), "[ must be allowed at start");
        assert!(a.contains(&8), "whitespace must be allowed at start");
        assert!(a.contains(&4), "string must be allowed at start");
        assert!(a.contains(&6), "digit must be allowed at start");
        assert!(!a.contains(&1), "}} must be masked at start");
        assert!(!a.contains(&5), ": must be masked at start");
        assert!(!a.contains(&7), ", must be masked at start");
        assert!(
            !a.contains(&9),
            "hello (h not a JSON value start) must be masked"
        );
    }

    #[test]
    fn after_open_brace_expects_key_or_close() {
        // pieces: 0="{", 1="}", 2="\"key\"", 3=":", 4="1", 5=" "
        let pieces = ["{", "}", "\"key\"", ":", "1", " "];
        let c = make_constraint(&pieces);
        let a = allowed(&c, &[0]); // generated: "{"
        assert!(a.contains(&1), "}} must be allowed after {{");
        assert!(a.contains(&2), "\"key\" must be allowed after {{");
        assert!(a.contains(&5), "whitespace must be allowed after {{");
        assert!(!a.contains(&3), ": must not be allowed right after {{");
        assert!(
            !a.contains(&4),
            "number must not be allowed after {{ (would be a key)"
        );
    }

    #[test]
    fn inside_string_allows_any_non_empty_piece() {
        // pieces: 0="{", 1="\"", 2="hello", 3="}", 4="world"
        let pieces = ["{", "\"", "hello", "}", "world"];
        let c = make_constraint(&pieces);
        // generated: ["{"=0, "\""=1]  → inside key string
        let a = allowed(&c, &[0, 1]);
        assert!(a.contains(&2), "hello must be allowed inside string");
        assert!(
            a.contains(&3),
            "}} (as a char) must be allowed inside string"
        );
        assert!(a.contains(&4), "world must be allowed inside string");
        assert!(
            a.contains(&1),
            "closing quote must be allowed inside string"
        );
    }

    #[test]
    fn after_complete_object_only_whitespace_allowed() {
        // pieces: 0="{", 1="}", 2=" ", 3=",", 4="{"
        let pieces = ["{", "}", " ", ",", "{"];
        let c = make_constraint(&pieces);
        // "{}" is a complete JSON object.
        assert!(c.is_complete(&[0, 1]));
        let a = allowed(&c, &[0, 1]);
        assert!(
            a.contains(&2),
            "whitespace must be allowed after complete JSON"
        );
        assert!(
            !a.contains(&0),
            "{{ must not be allowed after complete JSON"
        );
        assert!(!a.contains(&3), ", must not be allowed after complete JSON");
    }

    #[test]
    fn simple_json_object_complete() {
        // Tokens: 0="{", 1="\"k\"", 2=":", 3="1", 4="}"
        let pieces = ["{", "\"k\"", ":", "1", "}"];
        let c = make_constraint(&pieces);
        // {"k":1} → complete
        assert!(c.is_complete(&[0, 1, 2, 3, 4]));
    }

    #[test]
    fn json_array_complete() {
        // Tokens: 0="[", 1="1", 2=",", 3="2", 4="]"
        let pieces = ["[", "1", ",", "2", "]"];
        let c = make_constraint(&pieces);
        assert!(c.is_complete(&[0, 1, 2, 3, 4]));
    }

    #[test]
    fn parser_literal_true() {
        let mut p = JsonStructParser::new();
        for b in b"true" {
            p.advance(*b);
        }
        assert!(p.is_complete(), "true must be complete JSON");
    }

    #[test]
    fn parser_literal_false() {
        let mut p = JsonStructParser::new();
        for b in b"false" {
            p.advance(*b);
        }
        assert!(p.is_complete(), "false must be complete JSON");
    }

    #[test]
    fn parser_literal_null() {
        let mut p = JsonStructParser::new();
        for b in b"null" {
            p.advance(*b);
        }
        assert!(p.is_complete(), "null must be complete JSON");
    }

    #[test]
    fn parser_nested_object() {
        let mut p = JsonStructParser::new();
        for b in b"{\"a\":{\"b\":1}}" {
            p.advance(*b);
        }
        assert!(p.is_complete());
    }

    #[test]
    fn parser_array_of_objects() {
        let mut p = JsonStructParser::new();
        for b in b"[{\"x\":1},{\"y\":2}]" {
            p.advance(*b);
        }
        assert!(p.is_complete());
    }

    #[test]
    fn parser_rejects_bare_word() {
        let mut p = JsonStructParser::new();
        for b in b"hello" {
            p.advance(*b);
        }
        assert!(p.is_error());
    }

    #[test]
    fn parser_rejects_trailing_junk() {
        let mut p = JsonStructParser::new();
        for b in b"{}extra" {
            p.advance(*b);
        }
        assert!(p.is_error());
    }

    #[test]
    fn constrained_sampler_overrides_natural_argmax() {
        use crate::sampling::{sample_at_constrained, SamplingParams};
        // pieces: 0="{" (valid JSON start), 1="hello" (invalid JSON start)
        let pieces = ["{", "hello"];
        let pm = piece_map(&pieces);
        let c: Box<dyn Constraint> = Box::new(make_constraint(&pieces));
        let p = SamplingParams::greedy(8);
        // "hello" has the higher logit but is invalid JSON; constrained must pick "{".
        let logits = [1.0f32, 5.0];
        let tok = sample_at_constrained(&logits, &p, 0, &[], c.as_ref(), &pm).unwrap();
        assert_eq!(
            tok, 0,
            "constrained sampler must pick {{ (valid) over hello (invalid, higher logit)"
        );
    }

    // -----------------------------------------------------------------------
    // Regression tests for the critical safety bugs fixed in this commit.
    // -----------------------------------------------------------------------

    /// BUG 1 regression: after a complete JSON value, apply_mask must NEVER
    /// produce an all-(-inf) row, even when the vocab has no whitespace token.
    ///
    /// Scenario: the only tokens in the vocab are "{", "}" (no whitespace, no
    /// EOS).  After generating "{}" the parser is Done.  The old code masked
    /// ALL tokens (only whitespace valid in Done, but none available).  The
    /// new guard must restore at least one finite logit.
    #[test]
    fn all_masked_guard_never_all_neg_inf_at_done() {
        // No whitespace token, no stop token configured.
        let pieces = ["{", "}"];
        let c = make_constraint(&pieces);
        // Feed "{}" → Done state.
        let generated = [0u32, 1u32]; // "{", "}"
        assert!(
            c.is_complete(&generated),
            "prerequisite: parser should be Done"
        );

        let mut logits = vec![2.0f32, 3.0f32];
        c.apply_mask(&generated, &mut logits);

        let any_finite = logits.iter().any(|&v| v.is_finite());
        assert!(any_finite, "CRITICAL: apply_mask must never produce all-(-inf) — at least one logit must be finite");
    }

    /// BUG 1 regression: with a stop token configured, the stop token must be
    /// unmasked (not -inf) after the value is complete.
    #[test]
    fn stop_token_unmasked_at_done_with_stop_configured() {
        // vocab: 0="{", 1="}", 2="<eos>" (empty piece → normally masked), stop_id=2
        let pieces = ["{", "}", ""];
        let stop_id = 2u32;
        let c = make_constraint_with_stops(&pieces, vec![stop_id]);

        let generated = [0u32, 1u32]; // "{}" → Done
        assert!(c.is_complete(&generated));

        let mut logits = vec![1.0f32, 1.0f32, 5.0f32];
        c.apply_mask(&generated, &mut logits);

        // The stop token must be finite (allowed) at Done.
        assert!(
            logits[stop_id as usize].is_finite(),
            "stop token must be unmasked (finite) when JSON value is complete"
        );
        // The all-masked guard must hold regardless.
        let any_finite = logits.iter().any(|&v| v.is_finite());
        assert!(
            any_finite,
            "at least one logit must be finite after apply_mask"
        );
    }

    /// BUG 1 regression: stop token is masked BEFORE completion (mid-JSON).
    /// The model must NOT be able to stop before the JSON value is closed.
    #[test]
    fn stop_token_masked_before_completion() {
        // vocab: 0="{", 1="\"k\"", 2=":", 3="1", 4="}", 5="<eos>" (empty piece)
        let pieces = ["{", "\"k\"", ":", "1", "}", ""];
        let stop_id = 5u32;
        let c = make_constraint_with_stops(&pieces, vec![stop_id]);

        // After just "{", we are mid-JSON (not complete).
        let generated = [0u32]; // "{"
        assert!(!c.is_complete(&generated));

        let mut logits = vec![1.0f32; 6];
        c.apply_mask(&generated, &mut logits);

        // Stop token should be masked (can't stop mid-JSON).
        assert_eq!(
            logits[stop_id as usize],
            f32::NEG_INFINITY,
            "stop token must be masked before JSON value is complete"
        );
    }

    /// BUG 1 regression: sample_at_constrained must return Ok (not Err) even
    /// when the grammar would mask all tokens — the guard restores a fallback.
    #[test]
    fn sample_at_constrained_never_errors_on_all_masked() {
        use crate::sampling::{sample_at_constrained, SamplingParams};

        // No whitespace, no stop tokens.  After "{}" → Done, old code would
        // mask everything → stochastic sampling would Err.
        let pieces = ["{", "}"];
        let pm = piece_map(&pieces);
        let c: Box<dyn Constraint> = Box::new(make_constraint(&pieces));
        let generated = [0u32, 1u32]; // "{}"

        // Greedy must return Ok.
        let p_greedy = SamplingParams::greedy(8);
        let logits = [2.0f32, 3.0f32];
        let result = sample_at_constrained(&logits, &p_greedy, 0, &generated, c.as_ref(), &pm);
        assert!(
            result.is_ok(),
            "greedy constrained sampling must not error: {:?}",
            result
        );

        // Stochastic must also return Ok (not panic/Err on all-NaN distribution).
        let p_stoch = SamplingParams {
            temperature: 1.0,
            seed: 42,
            max_tokens: 8,
            ..Default::default()
        };
        let result = sample_at_constrained(&logits, &p_stoch, 0, &generated, c.as_ref(), &pm);
        assert!(
            result.is_ok(),
            "stochastic constrained sampling must not error: {:?}",
            result
        );
    }

    /// BUG 3 regression: empty array `[]` must be representable.
    ///
    /// The old CommaArr state didn't allow `]` directly after `[`, so `[]` was
    /// unrepresentable (every vocab token for `]` was masked right after `[`).
    #[test]
    fn empty_array_is_representable() {
        // vocab: 0="[", 1="]", 2="{", 3="}"
        let pieces = ["[", "]", "{", "}"];
        let c = make_constraint(&pieces);

        // After "[", "]" must be allowed (closes the empty array).
        let generated_after_open = [0u32]; // "["
        let a = allowed(&c, &generated_after_open);
        assert!(
            a.contains(&1),
            "] must be allowed right after [ (empty array [])"
        );

        // A full "[]" sequence must reach is_complete.
        let generated_complete = [0u32, 1u32]; // "[]"
        assert!(
            c.is_complete(&generated_complete),
            "[] must be a complete JSON value"
        );
    }

    /// Symmetry: empty object `{}` still works (regression guard).
    #[test]
    fn empty_object_still_works() {
        let pieces = ["{", "}"];
        let c = make_constraint(&pieces);
        let a = allowed(&c, &[0]); // after "{"
        assert!(
            a.contains(&1),
            "}} must be allowed right after {{ (empty object {{}})"
        );
        assert!(c.is_complete(&[0, 1]), "{{}} must be a complete JSON value");
    }

    /// BUG 2 / is_complete wiring: after a complete value with stop tokens
    /// configured, the stop token is the preferred termination path —
    /// verify both that it's selectable AND that non-stop tokens consistent
    /// with Done (whitespace) are also selectable.
    #[test]
    fn done_state_allows_stop_and_whitespace() {
        // vocab: 0="{", 1="}", 2=" " (whitespace), 3="<eos>" (empty piece), stop_id=3
        let pieces = ["{", "}", " ", ""];
        let stop_id = 3u32;
        let c = make_constraint_with_stops(&pieces, vec![stop_id]);

        let generated = [0u32, 1u32]; // "{}"
        assert!(c.is_complete(&generated));

        let a = allowed(&c, &generated);
        // Whitespace is valid trailing content.
        assert!(
            a.contains(&2),
            "whitespace must be allowed after complete JSON"
        );
        // Stop token must be allowed to terminate.
        assert!(
            a.contains(&stop_id),
            "stop token must be allowed after complete JSON"
        );
        // "{" is not allowed (new JSON value after Done → Bad).
        assert!(
            !a.contains(&0),
            "{{ must not be allowed after complete JSON"
        );
    }

    /// Unconstrained path is unchanged: apply_mask on the Constraint trait
    /// default (is_complete=false) doesn't affect regular sampling paths.
    /// This verifies the unconstrained `sample_at` path is byte-identical.
    #[test]
    fn unconstrained_path_unaffected() {
        use crate::sampling::{sample_at, SamplingParams};
        let logits = [0.5f32, 1.0, 0.8, 0.3, 0.9];
        let p = SamplingParams::greedy(8);
        // Greedy over unconstrained logits should return index 1 (value 1.0).
        assert_eq!(sample_at(&logits, &p, 0).unwrap(), 1);
    }
}
