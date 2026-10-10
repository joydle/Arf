//! Macro-benchmarks: prefill throughput and decode-step latency vs batch size.
//!
//! Uses a small model with zeroed weights — we measure the engine/cache/op
//! machinery, not generation quality.

use std::collections::HashMap;

use arf_core::config::{EngineConfig, ModelConfig, RopeScaling};
use arf_core::engine::LlmEngine;
use arf_core::model::weights::{build, Weights};
use arf_core::sampling::SamplingParams;
use arf_core::Tensor;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

fn small_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 1000,
        hidden_size: 512,
        intermediate_size: 1024,
        num_layers: 4,
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

fn build_engine() -> LlmEngine {
    let cfg = small_config();
    let model = build(&cfg, &zeros_weights(&cfg), cfg.max_position_embeddings).unwrap();
    let ecfg = EngineConfig {
        block_size: 16,
        num_blocks: 512,
        max_batch_size: 64,
        max_prefill_tokens: 8192,
        ..Default::default()
    };
    LlmEngine::new(model, ecfg).unwrap()
}

fn bench_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefill");
    for &prompt_len in &[16usize, 64, 256] {
        group.bench_with_input(
            BenchmarkId::from_parameter(prompt_len),
            &prompt_len,
            |b, &len| {
                b.iter(|| {
                    let mut e = build_engine();
                    e.submit(vec![1u32; len], SamplingParams::greedy(1))
                        .unwrap();
                    black_box(e.step().unwrap());
                });
            },
        );
    }
    group.finish();
}

fn bench_decode_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_step");
    for &batch in &[1usize, 8, 32] {
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &n| {
            b.iter_batched(
                || {
                    let mut e = build_engine();
                    for _ in 0..n {
                        e.submit(vec![1u32; 32], SamplingParams::greedy(128))
                            .unwrap();
                    }
                    e.step().unwrap(); // prefill all
                    e
                },
                |mut e| {
                    black_box(e.step().unwrap()); // one decode step
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_prefill, bench_decode_step);
criterion_main!(benches);
