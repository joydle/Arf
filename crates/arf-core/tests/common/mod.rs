//! Shared helpers: a tiny, deterministic Llama for fast end-to-end tests.
//!
//! Each integration test is its own binary, so `mod common;` compiles this file once PER test
//! target — and a target that uses one helper gets `dead_code` for the rest. Even `tiny_model`,
//! which eleven call sites use, warns in the binaries that do not. That is what the allow is for;
//! it is not covering unused code.
#![allow(dead_code)]

use std::collections::HashMap;

use arf_core::config::{ModelConfig, RopeScaling};
use arf_core::model::weights::{build, Weights};
use arf_core::model::Llama;
use arf_core::Tensor;

/// A small but architecturally complete config (GQA with 2 layers).
fn tiny_config() -> ModelConfig {
    ModelConfig {
        nextn_layers: 0,
        vocab_size: 48,
        hidden_size: 16,
        intermediate_size: 32,
        num_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_scaling: RopeScaling::None,
        max_position_embeddings: 256,
        tie_word_embeddings: true,
        mlp: arf_core::config::MlpKind::Dense,
        qk_norm: false,
        norm_style: arf_core::config::NormStyle::Llama,
        attn: arf_core::config::AttnKind::Causal,
        embedding_scale: None,
        query_pre_attn_scalar: None,
        final_logit_softcap: None,
        value_norm: false,
        gate_act: arf_core::config::GateAct::Silu,
    }
}

/// A tiny but architecturally complete Gemma config: 4-norm blocks, QK-norm,
/// √hidden embedding scale, and hybrid local/global attention with a small
/// sliding window (so the windowing path is actually exercised) and two RoPE
/// bases. `global_every = 2` makes layers 0,2 local and layer 1 global.
fn tiny_gemma_config() -> ModelConfig {
    ModelConfig {
        nextn_layers: 0,
        vocab_size: 48,
        hidden_size: 16,
        intermediate_size: 32,
        num_layers: 3,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        rope_scaling: RopeScaling::None,
        max_position_embeddings: 256,
        tie_word_embeddings: true,
        mlp: arf_core::config::MlpKind::Dense,
        qk_norm: true,
        norm_style: arf_core::config::NormStyle::Gemma,
        attn: arf_core::config::AttnKind::HybridLocalGlobal {
            window: 3,
            global_every: 2,
            local_theta: 10_000.0,
            global_theta: 1_000_000.0,
            // Gemma-3-style: global layers share the sliding geometry, full rotary.
            global_head_dim: 4,
            global_kv_heads: 2,
            partial_rotary_factor: 1.0,
        },
        embedding_scale: Some((16.0f32).sqrt()),
        query_pre_attn_scalar: None,
        final_logit_softcap: None,
        value_norm: false,
        gate_act: arf_core::config::GateAct::Silu,
    }
}

/// Build a tiny Gemma [`Llama`] with fixed deterministic weights.
pub fn tiny_gemma_model() -> Llama {
    let cfg = tiny_gemma_config();
    build(&cfg, &tiny_gemma_weights(&cfg), cfg.max_position_embeddings).unwrap()
}

/// Deterministic synthetic weights for the tiny Gemma model — the four per-layer
/// norms plus the per-head q/k norms that the Llama tiny model omits.
fn tiny_gemma_weights(cfg: &ModelConfig) -> Weights {
    let h = cfg.hidden_size;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut seed = 100u64;
    let mut rand = |rows: usize, cols: usize| {
        seed += 1;
        Tensor::from_vec(fill(rows * cols, seed, 0.1), vec![rows, cols])
    };

    t.insert("model.embed_tokens.weight".into(), rand(cfg.vocab_size, h));
    for i in 0..cfg.num_layers {
        let p = format!("model.layers.{i}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            t.insert(format!("{p}.{norm}.weight"), Tensor::zeros(vec![h]));
        }
        t.insert(format!("{p}.self_attn.q_proj.weight"), rand(q_dim, h));
        t.insert(format!("{p}.self_attn.k_proj.weight"), rand(kv_dim, h));
        t.insert(format!("{p}.self_attn.v_proj.weight"), rand(kv_dim, h));
        t.insert(format!("{p}.self_attn.o_proj.weight"), rand(h, q_dim));
        t.insert(
            format!("{p}.self_attn.q_norm.weight"),
            Tensor::zeros(vec![cfg.head_dim]),
        );
        t.insert(
            format!("{p}.self_attn.k_norm.weight"),
            Tensor::zeros(vec![cfg.head_dim]),
        );
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand(cfg.intermediate_size, h),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand(cfg.intermediate_size, h),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand(h, cfg.intermediate_size),
        );
    }
    t.insert("model.norm.weight".into(), Tensor::zeros(vec![h]));

    Weights::from_map(t)
}

/// Deterministic pseudo-random values in roughly `[-scale, scale]`.
fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Build a tiny [`Llama`] with fixed deterministic weights.
pub fn tiny_model() -> Llama {
    let cfg = tiny_config();
    build(&cfg, &tiny_weights(&cfg), cfg.max_position_embeddings).unwrap()
}

/// The same tiny model, built with weight quantization `quant`.
pub fn tiny_model_quant(quant: arf_core::config::Quant) -> Llama {
    use arf_core::device::Device;
    use arf_core::model::weights::build_quant;
    let cfg = tiny_config();
    build_quant(
        &cfg,
        &tiny_weights(&cfg),
        cfg.max_position_embeddings,
        &Device::Cpu,
        quant,
    )
    .unwrap()
}

/// Deterministic synthetic weights for the tiny model.
fn tiny_weights(cfg: &ModelConfig) -> Weights {
    let h = cfg.hidden_size;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut seed = 1u64;
    let mut rand = |rows: usize, cols: usize| {
        seed += 1;
        Tensor::from_vec(fill(rows * cols, seed, 0.1), vec![rows, cols])
    };

    t.insert("model.embed_tokens.weight".into(), rand(cfg.vocab_size, h));
    for i in 0..cfg.num_layers {
        let p = format!("model.layers.{i}");
        t.insert(format!("{p}.input_layernorm.weight"), Tensor::ones(vec![h]));
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            Tensor::ones(vec![h]),
        );
        t.insert(format!("{p}.self_attn.q_proj.weight"), rand(q_dim, h));
        t.insert(format!("{p}.self_attn.k_proj.weight"), rand(kv_dim, h));
        t.insert(format!("{p}.self_attn.v_proj.weight"), rand(kv_dim, h));
        t.insert(format!("{p}.self_attn.o_proj.weight"), rand(h, q_dim));
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand(cfg.intermediate_size, h),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand(cfg.intermediate_size, h),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand(h, cfg.intermediate_size),
        );
    }
    t.insert("model.norm.weight".into(), Tensor::ones(vec![h]));

    Weights::from_map(t)
}
