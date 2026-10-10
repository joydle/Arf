//! GPU decode-step throughput on a small resident model.
//!
//! The metric of interest is per-token overhead now that the decode path draws
//! its transients (the ctx-dependent storage scratch and every per-pass uniform
//! block) from two `GpuArena` sub-allocations — one `wgpu::Buffer` each — rather
//! than allocating a fresh buffer per value (hundreds per token at 16 layers).
//!
//! Skips cleanly when no GPU adapter is present. Run with:
//!   cargo bench -p arf-gpu --features gpu --bench gpu_decode
//!
//! To compare against the pre-arena per-buffer path, save a baseline on the
//! parent commit and compare on this one:
//!   git checkout HEAD~1 && cargo bench --features gpu --bench gpu_decode -- --save-baseline per_buffer
//!   git checkout -      && cargo bench --features gpu --bench gpu_decode -- --baseline per_buffer

use std::collections::HashMap;
use std::hint::black_box;

use arf_core::config::{ModelConfig, RopeScaling};
use arf_core::model::weights::Weights;
use arf_core::Tensor;
use arf_gpu::gpu::GpuContext;
use arf_gpu::weights::build_gpu;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn config() -> ModelConfig {
    ModelConfig {
        vocab_size: 1000,
        hidden_size: 512,
        intermediate_size: 1024,
        num_layers: 8,
        num_attention_heads: 8,
        num_kv_heads: 2,
        head_dim: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_scaling: RopeScaling::None,
        max_position_embeddings: 4096,
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
        nextn_layers: 0,
    }
}

/// Zeroed weights — we measure the decode machinery (allocation + dispatch),
/// not generation quality.
fn zeros_weights(cfg: &ModelConfig) -> Weights {
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mat = |t: &mut HashMap<String, Tensor>, name: String, rows: usize, cols: usize| {
        t.insert(name, Tensor::zeros(vec![rows, cols]));
    };
    mat(
        &mut t,
        "model.embed_tokens.weight".into(),
        cfg.vocab_size,
        h,
    );
    for i in 0..cfg.num_layers {
        let p = format!("model.layers.{i}");
        t.insert(format!("{p}.input_layernorm.weight"), Tensor::ones(vec![h]));
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            Tensor::ones(vec![h]),
        );
        mat(&mut t, format!("{p}.self_attn.q_proj.weight"), q, h);
        mat(&mut t, format!("{p}.self_attn.k_proj.weight"), kv, h);
        mat(&mut t, format!("{p}.self_attn.v_proj.weight"), kv, h);
        mat(&mut t, format!("{p}.self_attn.o_proj.weight"), h, q);
        mat(
            &mut t,
            format!("{p}.mlp.gate_proj.weight"),
            cfg.intermediate_size,
            h,
        );
        mat(
            &mut t,
            format!("{p}.mlp.up_proj.weight"),
            cfg.intermediate_size,
            h,
        );
        mat(
            &mut t,
            format!("{p}.mlp.down_proj.weight"),
            h,
            cfg.intermediate_size,
        );
    }
    t.insert("model.norm.weight".into(), Tensor::ones(vec![h]));
    Weights::from_map(t)
}

fn gpu_decode(c: &mut Criterion) {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping gpu_decode bench (no adapter): {e}");
            return;
        }
    };
    eprintln!("gpu_decode on {}", ctx.info());

    let cfg = config();
    let w = zeros_weights(&cfg);
    let model = build_gpu(&cfg, &w, cfg.max_position_embeddings, &ctx).expect("build_gpu");
    let prompt: Vec<u32> = (1..=8).collect();

    let mut g = c.benchmark_group("gpu_decode");
    for new_tokens in [16usize, 64] {
        g.bench_with_input(
            BenchmarkId::from_parameter(new_tokens),
            &new_tokens,
            |b, &n| b.iter(|| black_box(model.generate(black_box(&prompt), n))),
        );
    }
    g.finish();
}

criterion_group!(benches, gpu_decode);
criterion_main!(benches);
