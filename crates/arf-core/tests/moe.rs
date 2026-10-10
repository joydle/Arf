//! End-to-end CPU test for a Qwen3-style MoE model: a tiny config with a router,
//! several experts, and a shared expert builds into a `Llama` and produces finite,
//! routing-dependent logits. The MoE routing/weight-sum math itself is unit-tested
//! in `model::moe`; this proves the whole forward path (embed → attn+qk-norm →
//! MoE MLP → norm → lm_head) wires together and runs.

use std::collections::HashMap;

use arf_core::cache::{slots_for, write_runs, PagedKvCache};
use arf_core::config::{AttnKind, MlpKind, ModelConfig, NormStyle, RopeScaling};
use arf_core::model::weights::{build, Weights};
use arf_core::model::{ForwardBatch, Llama, SeqAttn};
use arf_core::Tensor;

/// A tiny but complete MoE config: 4 experts, top-2, 1 shared expert, qk-norm on.
fn moe_cfg() -> ModelConfig {
    ModelConfig {
        nextn_layers: 0,
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32, // unused for MoE; experts use moe_intermediate
        num_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_scaling: RopeScaling::None,
        max_position_embeddings: 64,
        tie_word_embeddings: true,
        mlp: MlpKind::Moe {
            num_experts: 4,
            top_k: 2,
            shared_experts: 1,
            moe_intermediate: 8,
            norm_topk: true,
        },
        qk_norm: true,
        norm_style: NormStyle::Llama,
        attn: AttnKind::Causal,
        embedding_scale: None,
        query_pre_attn_scalar: None,
        final_logit_softcap: None,
        value_norm: false,
        gate_act: arf_core::config::GateAct::Silu,
    }
}

/// Deterministic pseudo-random in roughly `[-0.5, 0.5]`.
fn gen(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
        })
        .collect()
}

fn moe_weights(cfg: &ModelConfig) -> Weights {
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let (n_exp, moe_inter, shared) = match cfg.mlp {
        MlpKind::Moe {
            num_experts,
            moe_intermediate,
            shared_experts,
            ..
        } => (num_experts, moe_intermediate, shared_experts),
        _ => unreachable!(),
    };
    let mut s = 7u64;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    // Insert a `[rows, cols]` random tensor.
    fn mat(t: &mut HashMap<String, Tensor>, name: String, rows: usize, cols: usize, s: &mut u64) {
        t.insert(
            name,
            Tensor::from_vec(gen(rows * cols, s), vec![rows, cols]),
        );
    }
    mat(
        &mut t,
        "model.embed_tokens.weight".into(),
        cfg.vocab_size,
        h,
        &mut s,
    );
    for i in 0..cfg.num_layers {
        let p = format!("model.layers.{i}");
        t.insert(format!("{p}.input_layernorm.weight"), Tensor::ones(vec![h]));
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            Tensor::ones(vec![h]),
        );
        mat(&mut t, format!("{p}.self_attn.q_proj.weight"), q, h, &mut s);
        mat(
            &mut t,
            format!("{p}.self_attn.k_proj.weight"),
            kv,
            h,
            &mut s,
        );
        mat(
            &mut t,
            format!("{p}.self_attn.v_proj.weight"),
            kv,
            h,
            &mut s,
        );
        mat(&mut t, format!("{p}.self_attn.o_proj.weight"), h, q, &mut s);
        // q/k norm weights are 1-D [head_dim] (HF layout).
        t.insert(
            format!("{p}.self_attn.q_norm.weight"),
            Tensor::from_vec(gen(cfg.head_dim, &mut s), vec![cfg.head_dim]),
        );
        t.insert(
            format!("{p}.self_attn.k_norm.weight"),
            Tensor::from_vec(gen(cfg.head_dim, &mut s), vec![cfg.head_dim]),
        );
        // Router and experts.
        mat(&mut t, format!("{p}.mlp.gate.weight"), n_exp, h, &mut s);
        for e in 0..n_exp {
            let ep = format!("{p}.mlp.experts.{e}");
            mat(
                &mut t,
                format!("{ep}.gate_proj.weight"),
                moe_inter,
                h,
                &mut s,
            );
            mat(&mut t, format!("{ep}.up_proj.weight"), moe_inter, h, &mut s);
            mat(
                &mut t,
                format!("{ep}.down_proj.weight"),
                h,
                moe_inter,
                &mut s,
            );
        }
        for _ in 0..shared {
            let sp = format!("{p}.mlp.shared_expert");
            mat(
                &mut t,
                format!("{sp}.gate_proj.weight"),
                moe_inter,
                h,
                &mut s,
            );
            mat(&mut t, format!("{sp}.up_proj.weight"), moe_inter, h, &mut s);
            mat(
                &mut t,
                format!("{sp}.down_proj.weight"),
                h,
                moe_inter,
                &mut s,
            );
        }
    }
    t.insert("model.norm.weight".into(), Tensor::ones(vec![h]));
    Weights::from_map(t)
}

fn last_logits(model: &Llama, tokens: &[u32]) -> Vec<f32> {
    let mut cache = PagedKvCache::new(
        model.num_layers(),
        4,
        4,
        model.config().num_kv_heads,
        model.config().head_dim,
    );
    let q_len = tokens.len();
    let block_table = [0u32, 1];
    let batch = ForwardBatch {
        positions: (0..q_len as u32).collect(),
        seqs: vec![SeqAttn {
            stream_id: None,
            q_start: 0,
            q_len,
            past_len: 0,
            slots: slots_for(&block_table, 4, q_len),
            write_runs: write_runs(&block_table, 4, 0, q_len),
            image_spans: Vec::new(),
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let hidden = model.forward(tokens, &batch, &mut cache);
    model.logits_last(&hidden, &batch).into_vec()
}

#[test]
fn moe_model_builds_and_produces_finite_logits() {
    let cfg = moe_cfg();
    let model = build(&cfg, &moe_weights(&cfg), cfg.max_position_embeddings).unwrap();
    let logits = last_logits(&model, &[3, 1, 4, 1, 5]);
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "MoE logits must be finite"
    );
    // Not all-equal: routing + experts produce a non-degenerate distribution.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(max - min > 1e-4, "MoE logits should vary across the vocab");
}

#[test]
fn different_tokens_route_to_different_outputs() {
    let cfg = moe_cfg();
    let model = build(&cfg, &moe_weights(&cfg), cfg.max_position_embeddings).unwrap();
    let a = last_logits(&model, &[1, 2, 3]);
    let b = last_logits(&model, &[10, 20, 30]);
    let diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(diff > 1e-4, "distinct prompts should yield distinct logits");
}
