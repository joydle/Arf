//! Tokenizer construction from GGUF metadata. Synthetic-vocab tests pin the wiring
//! (BPE + ByteLevel + control tokens). An env-gated real-blob parity test is added
//! in a later task.

use arf_core::model::gguf_tokenizer::{build_tokenizer, GgufTokenizerMeta};

// A tiny byte-level BPE: ascii 'h','i' map to themselves under ByteLevel, plus the
// merged "hi" and a control token. All chars the inputs produce must be in vocab.
fn tiny_meta() -> GgufTokenizerMeta {
    let tokens = vec![
        "h".to_string(),             // id 0
        "i".to_string(),             // id 1
        "hi".to_string(),            // id 2
        "<|endoftext|>".to_string(), // id 3 (control)
    ];
    let token_type = vec![1, 1, 1, 3]; // 1=normal, 3=control
    GgufTokenizerMeta {
        model: Some("gpt2".into()),
        pre: Some("qwen2".into()),
        tokens,
        merges: vec!["h i".to_string()],
        token_type,
        add_space_prefix: None, // gpt2: byte-level, no Metaspace prefix
        bos_token_id: None,
        eos_token_id: Some(3),
        padding_token_id: Some(3),
    }
}

#[test]
fn builds_and_merges() {
    let tok = build_tokenizer(&tiny_meta()).expect("build");
    let enc = tok.encode("hi", false).expect("encode");
    assert_eq!(enc.get_ids(), &[2]); // "h"+"i" merge to id 2
}

#[test]
fn control_token_resolves_and_is_atomic() {
    let tok = build_tokenizer(&tiny_meta()).expect("build");
    assert_eq!(tok.token_to_id("<|endoftext|>"), Some(3));
    let enc = tok.encode("<|endoftext|>", false).expect("encode");
    assert_eq!(enc.get_ids(), &[3]); // atomic, not split into bytes
}

/// USER_DEFINED (ggml token_type 4) must encode ATOMICALLY, exactly as CONTROL does.
///
/// This is gemma-4's real shape: its turn brackets are CONTROL (`<|turn>` 105, `<turn|>` 106)
/// but its channel brackets are USER_DEFINED (`<|channel>` 100, `<channel|>` 101) — 7 such
/// tokens in a 262144-entry vocab. Registering only CONTROL left the channel brackets to be
/// spelled out as ordinary text. MEASURED against the real 12B GGUF through the daemon:
/// `<|turn>` encoded to 1 token, `<|channel>` to 4 and `<channel|>` to 3 — the same counts as
/// a `<|nonsense>` string that is not in the vocab at all.
///
/// The consequence was not a tokenizer curiosity. The chat template's generation cue ends
/// `<|channel>thought\n<channel|>`, so the model was handed junk tokens where two control
/// tokens belonged, did not recognise its own cue, and emitted the real tokens itself — which
/// is how `<|channel>thought` reached users mid-reply.
#[test]
fn user_defined_token_is_atomic() {
    let mut meta = tiny_meta();
    meta.tokens.push("<|channel>".to_string()); // id 4
    meta.token_type.push(4); // USER_DEFINED
    let tok = build_tokenizer(&meta).expect("build");
    assert_eq!(tok.token_to_id("<|channel>"), Some(4));
    let enc = tok.encode("<|channel>", false).expect("encode");
    assert_eq!(
        enc.get_ids(),
        &[4],
        "USER_DEFINED must encode atomically, not spelled out as bytes"
    );
}

/// USER_DEFINED is registered NON-special on purpose, and the distinction is load-bearing.
///
/// `arf-serve` decodes with `skip_special_tokens: true`, and a gemma-4 thought block is
/// removed by locating its `<|channel>` / `<channel|>` brackets. Mark those special and decode
/// deletes the BRACKETS while leaving the thinking TEXT between them — the reply would arrive
/// as `thought\n{thinking}{answer}` glued together with nothing left to cut on. Keeping them
/// visible is what lets the strip remove the block whole.
#[test]
fn user_defined_survives_skip_special_tokens() {
    let mut meta = tiny_meta();
    meta.tokens.push("<|channel>".to_string()); // id 4, USER_DEFINED
    meta.token_type.push(4);
    let tok = build_tokenizer(&meta).expect("build");

    let kept = tok.decode(&[4], true).expect("decode user_defined");
    assert!(
        kept.contains("<|channel>"),
        "USER_DEFINED must survive skip_special_tokens; got {kept:?}"
    );
    // Contrast: a CONTROL token (id 3) IS dropped by the same call.
    let dropped = tok.decode(&[3], true).expect("decode control");
    assert!(
        !dropped.contains("<|endoftext|>"),
        "CONTROL must still be skipped; got {dropped:?}"
    );
}

#[test]
fn rejects_unknown_tokenizer_model() {
    let mut meta = tiny_meta();
    meta.model = Some("bert".into());
    assert!(build_tokenizer(&meta).is_err());
}

/// L320 — SentencePiece BPE (Gemma 3/4, `llama`) is supported: the vocab and merges
/// are the same shape as gpt2's, only the space convention differs (`▁` vs `Ġ`), so
/// the build succeeds and swaps ByteLevel for Metaspace. This used to be rejected on
/// the model NAME alone, which made a Gemma-4 GGUF unrunnable despite carrying a
/// complete tokenizer.
#[test]
fn accepts_sentencepiece_bpe() {
    let mut meta = tiny_meta();
    meta.model = Some("gemma4".into());
    assert!(build_tokenizer(&meta).is_ok());
    meta.model = Some("llama".into());
    assert!(build_tokenizer(&meta).is_ok());
}

/// A tokenizer with a vocab but NO merges is unigram, not BPE — still rejected,
/// because building a BPE from it would silently mis-tokenize.
#[test]
fn rejects_bpe_without_merges() {
    let mut meta = tiny_meta();
    meta.model = Some("gemma4".into());
    meta.merges.clear();
    assert!(build_tokenizer(&meta).is_err());
}

#[test]
fn rejects_malformed_merge() {
    let mut meta = tiny_meta();
    meta.merges = vec!["nospace".to_string()]; // not "a b"
    assert!(build_tokenizer(&meta).is_err());
}

#[test]
fn errors_on_token_type_length_mismatch() {
    let mut meta = tiny_meta();
    meta.token_type = vec![1, 1]; // shorter than tokens (len 4)
    assert!(build_tokenizer(&meta).is_err());
}

#[test]
fn empty_token_type_builds_without_specials() {
    let mut meta = tiny_meta();
    meta.token_type = vec![]; // omitted entirely — allowed, just no specials
    let tok = build_tokenizer(&meta).expect("build with no token_type");
    // The control token is now a plain vocab entry, not registered special.
    // It still has id 3 in the vocab, but won't necessarily encode atomically.
    assert_eq!(tok.encode("hi", false).unwrap().get_ids(), &[2]);
}

#[test]
fn wrapper_from_gguf_meta() {
    let meta = tiny_meta();
    let tok = arf_core::Tokenizer::from_gguf_meta(&meta).expect("wrap");
    assert_eq!(tok.encode("hi", false).unwrap(), vec![2]);
    assert_eq!(tok.token_to_id("<|endoftext|>"), Some(3));
}

/// Exact-id parity against the LIVE Qwen3-Coder-30B GGUF blob. Skips unless
/// `ARF_GGUF_BLOB` points at the blob; also `#[ignore]` so plain `cargo test`
/// never runs it. Run with:
///   ARF_GGUF_BLOB=$HOME/.ollama/models/blobs/sha256-1194192cf2... \
///   cargo test -p arf-core --test gguf_tokenizer real_blob -- --ignored --nocapture
///
/// The `cases` table holds expected ids LOCKED against the live blob (each id
/// cross-verified to the GGUF's own vocab — see the inline note on the table), so
/// the test asserts exact-id parity. A row with an empty `&[]` would fall back to
/// RECORD MODE (print our ids, skip that row's assert) — useful when adding new
/// cases. The lossless round-trip and the <|im_end|>=151645 stop id are asserted
/// unconditionally.
#[test]
#[ignore]
fn real_blob_known_ids() {
    let Ok(blob) = std::env::var("ARF_GGUF_BLOB") else {
        eprintln!("ARF_GGUF_BLOB not set; skipping real-blob parity test");
        return;
    };
    let tok = arf_core::Tokenizer::from_gguf_path(std::path::Path::new(&blob))
        .expect("build tokenizer from blob");

    // (text, expected ids). LOCKED against the live Qwen3-Coder-30B GGUF: each id
    // was cross-checked to the GGUF's own `tokenizer.ggml.tokens` vocab and
    // reconstructs the input under byte-level BPE — e.g. 9707="Hello", 11=",",
    // 1879="Ġworld" (space+world), 0="!"; 750="def", 912="Ġadd"; 101059="æĹ¥æľ¬"=日本.
    // An empty &[] would put a row in RECORD MODE (print our ids, skip the assert).
    let cases: &[(&str, &[u32])] = &[
        ("Hello, world!", &[9707, 11, 1879, 0]),
        (
            "def add(a, b):\n    return a + b\n",
            &[
                750, 912, 7, 64, 11, 293, 1648, 198, 262, 470, 264, 488, 293, 198,
            ],
        ),
        (
            "日本語のテスト 🚀",
            &[101059, 102819, 15767, 140592, 11162, 248, 222],
        ),
    ];
    let recording = cases.iter().any(|(_, ids)| ids.is_empty());
    for (text, expected) in cases {
        let got = tok.encode(text, false).expect("encode");
        if expected.is_empty() {
            eprintln!("[RECORD] {text:?} -> {got:?}");
        } else {
            assert_eq!(&got[..], *expected, "token-id drift for {text:?}");
        }
    }
    if recording {
        eprintln!(
            "[RECORD MODE] fill the `cases` expected ids from the reference \
             tokenizer to lock exact-id parity (see the test doc comment)."
        );
    }

    // These run regardless of record mode — they need no reference:
    // 1) lossless round-trip for ascii / multibyte / emoji.
    for (text, _) in cases {
        let ids = tok.encode(text, false).expect("encode");
        let back = tok.decode(&ids, false).expect("decode");
        assert_eq!(&back, text, "decode(encode) not lossless for {text:?}");
    }
    // 2) the Qwen3 end-of-turn stop token resolves to its known id.
    assert_eq!(
        tok.token_to_id("<|im_end|>"),
        Some(151645),
        "Qwen3 <|im_end|> should resolve to 151645"
    );
}

/// The USER_DEFINED fix against the REAL gemma-4 vocab, not a synthetic one — the synthetic
/// test proves the wiring, this proves the file actually has the shape the wiring assumes.
/// Skips unless `ARF_GEMMA4_BLOB` points at a gemma-4 GGUF; `#[ignore]` so plain
/// `cargo test` never runs it. Run with:
///   ARF_GEMMA4_BLOB=/path/to/gemma-4-12b-it-Q4_K_M.gguf \
///   cargo test -p arf-core --test gguf_tokenizer real_gemma4 -- --ignored --nocapture
///
/// Both bracket pairs are asserted TOGETHER because the bug was that they behave differently:
/// the turn pair is CONTROL and always worked, the channel pair is USER_DEFINED and did not.
/// Asserting only the channel pair would pass on a build that broke CONTROL registration.
#[test]
#[ignore]
fn real_gemma4_control_and_channel_brackets_are_atomic() {
    let Ok(blob) = std::env::var("ARF_GEMMA4_BLOB") else {
        eprintln!("ARF_GEMMA4_BLOB not set; skipping real gemma-4 bracket test");
        return;
    };
    let tok = arf_core::Tokenizer::from_gguf_path(std::path::Path::new(&blob))
        .expect("build tokenizer from gemma-4 blob");

    for tag in ["<|turn>", "<turn|>", "<|channel>", "<channel|>"] {
        let ids = tok.encode(tag, false).expect("encode");
        assert_eq!(
            ids.len(),
            1,
            "{tag} must encode to ONE token; got {ids:?}. More than one means it was spelled \
             out as text, which is what put `<|channel>thought` into user-visible replies."
        );
    }

    // The generation cue the chat template ends with must cost exactly its 4 real tokens
    // (`<|turn>`, `model`+`\n`, `<|channel>`, `thought`+`\n`, `<channel|>`) rather than
    // decomposing. Asserted as an upper bound so a vocab revision that merges differently
    // does not fail the build for the wrong reason.
    let cue = tok
        .encode("<|turn>model\n<|channel>thought\n<channel|>", false)
        .expect("encode cue");
    assert!(
        cue.len() <= 8,
        "the generation cue should cost a handful of tokens, got {} ({cue:?}) — that many \
         means the brackets decomposed again",
        cue.len()
    );
}
