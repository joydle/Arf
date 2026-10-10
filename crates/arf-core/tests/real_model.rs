//! End-to-end generation against REAL Llama-3.2-1B weights.
//!
//! Gated on `ARF_MODEL_PATH` (a directory with `model.safetensors` and
//! `tokenizer.json`, e.g. produced by `arf pull`). `#[ignore]`d so CI —
//! which is network-free and ships no weights — never runs it. Run locally with:
//!
//! ```sh
//! # Use an ABSOLUTE path: cargo runs this test with CWD = crates/arf-core,
//! # so a relative path would resolve there, not at the repo root.
//! ARF_MODEL_PATH="$PWD/models/llama-3.2-1b" \
//!   cargo test -p arf-gpu --test real_model -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use arf_core::config::{EngineConfig, ModelConfig};
use arf_core::engine::LlmEngine;
use arf_core::model::weights;
use arf_core::sampling::SamplingParams;
use arf_core::Tokenizer;

#[test]
#[ignore = "requires ARF_MODEL_PATH with real weights"]
fn generates_coherent_text_from_real_weights() {
    let dir = match std::env::var("ARF_MODEL_PATH") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            eprintln!("ARF_MODEL_PATH not set; skipping");
            return;
        }
    };
    // Cargo runs integration tests with CWD = the package dir (crates/arf),
    // so a *relative* ARF_MODEL_PATH resolves there, not at the workspace
    // root. Fail with an actionable message instead of a cryptic NotFound.
    assert!(
        dir.is_dir(),
        "ARF_MODEL_PATH={} is not a directory (cwd is {}). \
         Use an ABSOLUTE path — relative paths resolve from crates/arf-core, not the repo root.",
        dir.display(),
        std::env::current_dir().unwrap().display(),
    );
    let model_file = dir.join("model.safetensors");
    let tok_file = dir.join("tokenizer.json");

    let cfg = ModelConfig::llama_3_2_1b();
    let paths: [&Path; 1] = [model_file.as_path()];
    let model = weights::load_safetensors(&cfg, &paths, 8192).expect("load real safetensors");
    let tokenizer = Tokenizer::from_file(&tok_file).expect("load tokenizer");

    let prompt = "The capital of France is";
    let ids = tokenizer.encode(prompt, true).expect("encode");

    let mut engine = LlmEngine::new(model, EngineConfig::default()).expect("engine");
    let out = engine
        .generate(ids, SamplingParams::greedy(8))
        .expect("generate");

    let text = tokenizer.decode(&out, true).expect("decode");
    eprintln!("PROMPT: {prompt}\nCONTINUATION: {text}");

    assert!(!out.is_empty(), "model produced no tokens");
    assert!(
        text.to_lowercase().contains("paris"),
        "greedy continuation of {prompt:?} should mention Paris; got {text:?}"
    );
}
