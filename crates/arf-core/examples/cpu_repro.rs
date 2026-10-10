//! CPU repro for Qwen3-Coder-30B coherence debugging. Builds a CPU Q4 model from
//! the GGUF and greedily decodes a few tokens. If this prints the same garbage the
//! GPU path does, the bug is in shared model math (not a GPU kernel).
//!
//! Usage: cargo run --release -p arf-core --example cpu_repro -- <file.gguf>

use arf_core::config::{EngineConfig, ModelConfig, Quant};
use arf_core::engine::LlmEngine;
use arf_core::model::gguf::load_gguf;
use arf_core::model::weights;
use arf_core::sampling::SamplingParams;
use arf_core::{Device, Tokenizer};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: cpu_repro <file.gguf>");
    let p = std::path::Path::new(&path);
    let cfg = ModelConfig::qwen3_coder_30b();

    let w = load_gguf(p, &cfg).expect("load_gguf");
    eprintln!("building CPU Q4 model (30B, this takes a bit)...");
    let model = weights::build_quant(&cfg, &w, 8192, &Device::Cpu, Quant::Q4).expect("build");
    let tok = Tokenizer::from_gguf_path(p).expect("tokenizer");

    let prompt = "<|im_start|>user\nWrite a Rust function that adds two numbers.<|im_end|>\n<|im_start|>assistant\n";
    let ids = tok.encode(prompt, false).expect("encode");
    eprintln!("prompt ids ({}): {:?}", ids.len(), ids);

    let mut engine = LlmEngine::new(model, EngineConfig::default()).expect("engine");
    // `generate` returns ONLY the newly generated tokens (not the prompt).
    let new = engine
        .generate(ids.clone(), SamplingParams::greedy(64))
        .expect("generate");
    eprintln!("new ids ({}): {new:?}", new.len());
    eprintln!(
        "CONTINUATION: {:?}",
        tok.decode(&new, false).unwrap_or_default()
    );
}
