//! Build a `tokenizers::Tokenizer` from a GGUF's embedded `tokenizer.ggml.*`
//! metadata. Two families are supported, both BPE-with-merges — they differ only
//! in how a space is spelled:
//!
//!   * `gpt2` — byte-level BPE (Qwen2/3, Llama-3). Space is `Ġ`; the ByteLevel
//!     pre-tokenizer's built-in GPT-2 regex does the split.
//!   * `gemma4` / `llama` — SentencePiece-style BPE (Gemma 3/4). Space is `▁`
//!     (U+2581) and the split is Metaspace, not byte-level.
//!
//! L320: the `gemma4` case used to be rejected outright, which made
//! `arf run` on a Gemma-4 QAT GGUF fail with "only gpt2 BPE is supported"
//! even though the file carries a complete vocab AND 514,906 merges. The
//! rejection was a check on the model NAME, not on any missing capability.
//!
//! A tokenizer model with no merges (true unigram SentencePiece) is still
//! rejected — never silent approximate tokenization.

use crate::error::{ArfError, Result};
use tokenizers::decoders::byte_level::ByteLevel as DecByteLevel;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as PreByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

/// The raw tokenizer fields captured from a GGUF header. Owned (Strings/Vecs), so
/// it outlives the mmap cursor. Optional fields are `None` when the key is absent.
#[derive(Debug, Clone, Default)]
pub struct GgufTokenizerMeta {
    /// `tokenizer.ggml.model` — e.g. "gpt2". Build only supports "gpt2".
    pub model: Option<String>,
    /// `tokenizer.ggml.pre` — e.g. "qwen2". Informational only; NOT checked. The
    /// ByteLevel pre-tokenizer's built-in GPT-2 regex covers the gpt2/qwen2/llama3
    /// byte-level split, so we don't transcribe a per-variant pattern.
    pub pre: Option<String>,
    /// `tokenizer.ggml.tokens` — vocab; id == index.
    pub tokens: Vec<String>,
    /// `tokenizer.ggml.merges` — BPE merges, each `"a b"`.
    pub merges: Vec<String>,
    /// `tokenizer.ggml.token_type` — per-token type (3 == CONTROL in ggml).
    pub token_type: Vec<i32>,
    /// `tokenizer.ggml.add_space_prefix` — SentencePiece only; whether a leading
    /// space is prepended to the first word. Gemma sets it true.
    pub add_space_prefix: Option<bool>,
    /// Special-token ids: `tokenizer.ggml.{bos,eos,padding}_token_id` (None if absent).
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub padding_token_id: Option<i64>,
}

impl GgufTokenizerMeta {
    /// Error if this meta cannot be built into a tokenizer (caller should then
    /// require `--tokenizer`).
    pub fn require_buildable(&self) -> Result<()> {
        match self.model.as_deref() {
            Some(m)
                if Self::is_supported(m) && !self.tokens.is_empty() && !self.merges.is_empty() =>
            {
                Ok(())
            }
            Some(m) if Self::is_supported(m) && self.tokens.is_empty() => {
                Err(ArfError::Tokenizer(format!(
                    "GGUF has a {m} tokenizer but tokenizer.ggml.tokens is empty — \
                     pass --tokenizer tokenizer.json"
                )))
            }
            Some(m) if Self::is_supported(m) => Err(ArfError::Tokenizer(format!(
                "GGUF has a {m} tokenizer but no merges (unigram, not BPE) — \
                 pass --tokenizer tokenizer.json"
            ))),
            Some(other) => Err(ArfError::Tokenizer(format!(
                "GGUF embeds a `{other}` tokenizer; supported: gpt2 (byte-level BPE) \
                 and gemma4/llama (SentencePiece BPE) — pass --tokenizer tokenizer.json"
            ))),
            None => Err(ArfError::Tokenizer(
                "GGUF has no embedded tokenizer (tokenizer.ggml.model absent) — \
                 pass --tokenizer tokenizer.json"
                    .into(),
            )),
        }
    }

    /// Tokenizer models this module can build. Both are BPE with merges; the
    /// difference is the space convention, handled in `build_tokenizer`.
    fn is_supported(model: &str) -> bool {
        matches!(model, "gpt2" | "gemma4" | "llama")
    }

    /// True when spaces are SentencePiece's `▁` rather than byte-level `Ġ`.
    fn is_sentencepiece(&self) -> bool {
        matches!(self.model.as_deref(), Some("gemma4") | Some("llama"))
    }
}

/// ggml token type for control/special tokens.
const TOKEN_TYPE_CONTROL: i32 = 3;

/// ggml token type for USER_DEFINED tokens — real vocab entries a template addresses by name,
/// but not part of the model's control set. gemma-4 splits its brackets across BOTH types:
/// the channel pair is USER_DEFINED (`<|channel>` id 100, `<channel|>` id 101, type 4) while
/// the turn pair is CONTROL (`<|turn>` 105, `<turn|>` 106, type 3). Seven tokens in a
/// 262144-entry vocab, and registering only type 3 left the other pair as ordinary text.
///
/// They must be ADDED TOKENS or the encoder never matches them: measured against the running
/// daemon, `<|turn>` encoded to 1 token while `<|channel>` encoded to 4 and `<channel|>` to 3 —
/// the same count as the nonexistent `<|nonsense>`, i.e. spelled out character by character. A
/// chat template that emits them then hands the model junk tokens where its own generation cue
/// belongs, the model does not recognise the cue, and it emits the real token itself — which is
/// how `<|channel>thought` ended up inside user-visible replies.
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

/// The pre-tokenizer split regex for a `tokenizer.ggml.pre` tag, as the models' own
/// `tokenizer.json` files define it. `None` = use ByteLevel's built-in GPT-2 regex.
/// The bool is whether the family NFC-normalises first (Qwen does; Llama-3's normalizer is null).
fn split_pattern(pre: Option<&str>) -> Option<(&'static str, bool)> {
    const QWEN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let pre = pre?.to_ascii_lowercase();
    if pre.starts_with("qwen") {
        Some((QWEN, true))
    } else if matches!(
        pre.as_str(),
        "llama-bpe" | "llama3" | "llama-v3" | "smaug-bpe"
    ) {
        // NOT "llama4": it uses a different (case-aware) pattern, and no reference tokenizer.json
        // for it was on hand to check one against — it stays on the fallback, UNVERIFIED.
        Some((LLAMA3, false))
    } else {
        None
    }
}

/// Build a `tokenizers::Tokenizer` from gpt2-BPE GGUF metadata.
///
/// Pre-tokenizer is `ByteLevel { use_regex: true }`, which applies the canonical
/// GPT-2/Qwen2 split pattern internally — so we never transcribe the regex (the
/// main correctness risk). `add_prefix_space=false`, `trim_offsets=false` match
/// Qwen2's `tokenizer.json`.
pub fn build_tokenizer(meta: &GgufTokenizerMeta) -> Result<HfTokenizer> {
    meta.require_buildable()?;

    // vocab: token -> id (id is the array index).
    let vocab: tokenizers::models::bpe::Vocab = meta
        .tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();

    // merges: each "a b" -> (a, b). A malformed entry is an error, not a silent skip.
    let mut merges: tokenizers::models::bpe::Merges = Vec::with_capacity(meta.merges.len());
    for (line_no, m) in meta.merges.iter().enumerate() {
        let mut it = m.splitn(2, ' ');
        match (it.next(), it.next()) {
            (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => {
                merges.push((a.to_string(), b.to_string()))
            }
            _ => {
                return Err(ArfError::Tokenizer(format!(
                    "GGUF merge {line_no} malformed: {m:?} (expected \"a b\")"
                )))
            }
        }
    }

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| ArfError::Tokenizer(format!("BPE build: {e:?}")))?;

    let mut tok = HfTokenizer::new(bpe);
    if meta.is_sentencepiece() {
        // L320 — Gemma 3/4: spaces are `▁` (U+2581), so the split is Metaspace, not
        // byte-level. `prepend_scheme` follows `tokenizer.ggml.add_space_prefix`
        // (Gemma sets it): a leading space is added to the first word so "hi" and
        // " hi" tokenize identically, which is what the model was trained on.
        let scheme = if meta.add_space_prefix.unwrap_or(true) {
            PrependScheme::Always
        } else {
            PrependScheme::Never
        };
        let ms = Metaspace::new('▁', scheme, true);
        tok.with_pre_tokenizer(Some(ms.clone()));
        tok.with_decoder(Some(ms));
    } else {
        // 🔴 THE SPLIT PATTERN IS PER FAMILY (fixed 2026-09-20). This used ByteLevel's BUILT-IN
        // GPT-2 regex for every byte-level model, on the stated belief that it "covers
        // gpt2/qwen2/llama3". It does not. GPT-2 splits ` ?\p{L}+`; Qwen and Llama-3 split
        // `[^\r\n\p{L}\p{N}]?\p{L}+` — a run of letters may carry ONE leading non-letter — so
        // `_text`, `.trim`, `(path` are single pre-tokens there and were being cut in two here.
        // MEASURED against Qwen3.8's own tokenizer.json (`tok_canonical_check`): prose identical,
        // but 4 of 4 code lines different, 28 tokens for 22 — every CODE prompt was reaching the
        // model in a segmentation it was not trained on, ~25% longer than it should be. It was
        // found because prompt-lookup drafts copied from the prompt kept being rejected at single
        // tokens where the text was identical.
        match split_pattern(meta.pre.as_deref()) {
            Some((pattern, nfc)) => {
                use tokenizers::pre_tokenizers::sequence::Sequence;
                use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
                use tokenizers::pre_tokenizers::PreTokenizerWrapper;
                use tokenizers::SplitDelimiterBehavior;
                let split = Split::new(
                    SplitPattern::Regex(pattern.to_string()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .map_err(|e| ArfError::Tokenizer(format!("split pattern: {e:?}")))?;
                tok.with_pre_tokenizer(Some(Sequence::new(vec![
                    PreTokenizerWrapper::Split(split),
                    // use_regex = false: the Split above already did the splitting
                    PreTokenizerWrapper::ByteLevel(PreByteLevel::new(false, false, false)),
                ])));
                if nfc {
                    tok.with_normalizer(Some(tokenizers::normalizers::unicode::NFC))
                        .map_err(|e| ArfError::Tokenizer(format!("NFC normalizer: {e:?}")))?;
                }
            }
            // Unknown or genuinely GPT-2: the built-in regex, as before.
            None => {
                tok.with_pre_tokenizer(Some(PreByteLevel::new(false, false, true)));
            }
        }
        tok.with_decoder(Some(DecByteLevel::new(false, false, true)));
    }

    // Register control tokens (ggml token_type == 3) as special added tokens so they
    // encode atomically and `token_to_id` resolves chat stop tokens. An EMPTY
    // token_type (some minimal GGUFs omit it) is allowed — no specials registered.
    // A NON-empty token_type whose length disagrees with tokens means a malformed
    // file / parser bug: error rather than silently drop stop tokens (which would
    // make chat never stop).
    if !meta.token_type.is_empty() {
        if meta.token_type.len() != meta.tokens.len() {
            return Err(ArfError::Tokenizer(format!(
                "GGUF token_type len {} != tokens len {} — malformed tokenizer metadata",
                meta.token_type.len(),
                meta.tokens.len()
            )));
        }
        let specials: Vec<AddedToken> = meta
            .tokens
            .iter()
            .zip(&meta.token_type)
            .filter(|(_, &ty)| ty == TOKEN_TYPE_CONTROL)
            .map(|(t, _)| AddedToken::from(t.clone(), true))
            .collect();
        if !specials.is_empty() {
            tok.add_special_tokens(specials)
                .map_err(|e| ArfError::Tokenizer(format!("add_special_tokens: {e}")))?;
        }

        // USER_DEFINED goes through `add_tokens`, NOT `add_special_tokens`, and the difference
        // is load-bearing rather than stylistic. Both make the encoder match the token
        // atomically, which is the bug being fixed. Only `special` also makes DECODE drop it,
        // because `arf-serve`'s `decode` passes `skip_special_tokens: true`.
        //
        // Dropping them would be actively worse here. A thought block arrives as
        // `<|channel>thought\n{thinking}<channel|>{answer}`, and `strip_thought` removes the
        // whole block by finding those two brackets. Silence them at decode and the brackets
        // vanish while the THINKING TEXT REMAINS — leaving `thought\n{thinking}{answer}` glued
        // together, with nothing left to cut on. Visible-but-strippable beats invisible.
        let user_defined: Vec<AddedToken> = meta
            .tokens
            .iter()
            .zip(&meta.token_type)
            .filter(|(_, &ty)| ty == TOKEN_TYPE_USER_DEFINED)
            .map(|(t, _)| AddedToken::from(t.clone(), false))
            .collect();
        if !user_defined.is_empty() {
            tok.add_tokens(user_defined)
                .map_err(|e| ArfError::Tokenizer(format!("add_tokens: {e}")))?;
        }
    }

    Ok(tok)
}

#[cfg(test)]
mod split_pattern_tests {
    use super::split_pattern;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::{PreTokenizedString, PreTokenizer, SplitDelimiterBehavior};

    fn pieces(pre: &str, text: &str) -> Vec<String> {
        let (pat, _) = split_pattern(Some(pre)).expect("family has a pattern");
        let split = Split::new(
            SplitPattern::Regex(pat.to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .unwrap();
        let mut p = PreTokenizedString::from(text);
        split.pre_tokenize(&mut p).unwrap();
        p.get_splits(
            tokenizers::OffsetReferential::Original,
            tokenizers::OffsetType::Byte,
        )
        .into_iter()
        .map(|(s, _, _)| s.to_string())
        .collect()
    }

    /// The bug of 2026-09-20: GPT-2's rule cut `.trim`, `_text` and `(path` in two. In the Qwen
    /// and Llama-3 rule a run of letters carries ONE leading non-letter.
    #[test]
    fn a_letter_run_keeps_one_leading_punctuation_mark() {
        for pre in ["qwen2", "qwen35", "llama-bpe"] {
            assert_eq!(pieces(pre, "line.trim()"), ["line", ".trim", "()"], "{pre}");
            assert_eq!(pieces(pre, "raw_text"), ["raw", "_text"], "{pre}");
            assert_eq!(pieces(pre, "f(path"), ["f", "(path"], "{pre}");
        }
    }

    #[test]
    fn digits_group_by_family() {
        assert_eq!(pieces("qwen35", "1234"), ["1", "2", "3", "4"]);
        assert_eq!(pieces("llama-bpe", "1234"), ["123", "4"]);
    }

    /// Families with no VERIFIED pattern stay on ByteLevel's built-in regex rather than a guess.
    #[test]
    fn unverified_families_fall_back() {
        assert!(split_pattern(Some("llama4")).is_none());
        assert!(split_pattern(Some("gpt-2")).is_none());
        assert!(split_pattern(None).is_none());
    }
}
