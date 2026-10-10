//! int8 quantization quality: int8 logits track the bf16 baseline on the
//! synthetic model (CI-safe), and greedy decode matches on real weights (gated
//! on `ARF_MODEL_PATH`).

mod common;

use arf_core::config::{EngineConfig, Quant};
use arf_core::engine::LlmEngine;
use arf_core::sampling::SamplingParams;

#[test]
fn int8_greedy_tracks_bf16_on_synthetic_model() {
    let prompt = vec![3u32, 7, 1];
    let mut e_bf16 = LlmEngine::new(
        common::tiny_model_quant(Quant::None),
        EngineConfig::default(),
    )
    .unwrap();
    let mut e_q8 = LlmEngine::new(
        common::tiny_model_quant(Quant::Int8),
        EngineConfig::default(),
    )
    .unwrap();

    let a = e_bf16
        .generate(prompt.clone(), SamplingParams::greedy(12))
        .unwrap();
    let b = e_q8.generate(prompt, SamplingParams::greedy(12)).unwrap();

    // The tiny model is random, so int8 may diverge late; require a shared prefix.
    let agree = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    assert!(
        agree >= 4,
        "int8 greedy should track bf16 for >=4 tokens; got {agree} (bf16={a:?} int8={b:?})"
    );
}

#[test]
#[ignore = "requires ARF_MODEL_PATH with real weights"]
fn int8_greedy_matches_bf16_on_real_weights() {
    use arf_core::model::weights::load_safetensors_quant;
    use arf_core::Tokenizer;
    use std::path::{Path, PathBuf};

    let Ok(dir) = std::env::var("ARF_MODEL_PATH") else {
        eprintln!("ARF_MODEL_PATH not set; skipping");
        return;
    };
    let dir = PathBuf::from(dir);
    assert!(
        dir.is_dir(),
        "use an ABSOLUTE ARF_MODEL_PATH; cwd is {}",
        std::env::current_dir().unwrap().display()
    );
    let model_file = dir.join("model.safetensors");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let cfg = arf_core::config::ModelConfig::llama_3_2_1b();
    let ids = tok.encode("The capital of France is", true).unwrap();
    let paths: [&Path; 1] = [model_file.as_path()];
    let dev = arf_core::Device::cpu();

    let m_bf16 = load_safetensors_quant(&cfg, &paths, 8192, &dev, Quant::None).unwrap();
    let m_q8 = load_safetensors_quant(&cfg, &paths, 8192, &dev, Quant::Int8).unwrap();
    let mut e_bf16 = LlmEngine::new(m_bf16, EngineConfig::default()).unwrap();
    let mut e_q8 = LlmEngine::new(m_q8, EngineConfig::default()).unwrap();

    let a = e_bf16
        .generate(ids.clone(), SamplingParams::greedy(20))
        .unwrap();
    let b = e_q8.generate(ids, SamplingParams::greedy(20)).unwrap();
    eprintln!(
        "bf16: {}\nint8: {}",
        tok.decode(&a, true).unwrap(),
        tok.decode(&b, true).unwrap()
    );

    let agree = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    assert!(
        agree >= 15,
        "int8 greedy should match bf16 for >=15/20 tokens; got {agree}"
    );
}
