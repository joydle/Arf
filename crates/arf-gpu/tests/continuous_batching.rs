//! Invariant (spec §5 #4): N concurrent greedy requests through the
//! scheduler-driven batched loop produce the SAME tokens as N sequential
//! single-stream runs. Runs on the real Metal device; skips when none is
//! present (CI).

use std::collections::HashMap;

use arf_core::backend::BatchedBackend;
use arf_core::config::{EngineConfig, ModelConfig, RopeScaling};
use arf_core::engine::build_forward;
use arf_core::model::weights::Weights;
use arf_core::sampling::{SamplingParams, SeqSampling};
use arf_core::scheduler::{Request, Scheduler};
use arf_core::Tensor;
use arf_gpu::gpu::GpuContext;
use arf_gpu::WgpuBatched;

/// The tiny model config (copied from tests/gpu.rs `cfg`).
fn cfg() -> ModelConfig {
    ModelConfig {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_scaling: RopeScaling::None,
        max_position_embeddings: 64,
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

/// Deterministic tiny weights (copied from tests/gpu.rs `weights`).
fn weights(cfg: &ModelConfig) -> Weights {
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let mut seed = 7u64;
    let mut gen = |shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
            })
            .collect();
        Tensor::from_vec(data, shape)
    };
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        gen(vec![cfg.vocab_size, h]),
    );
    for i in 0..cfg.num_layers {
        let p = format!("model.layers.{i}");
        t.insert(format!("{p}.input_layernorm.weight"), Tensor::ones(vec![h]));
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            Tensor::ones(vec![h]),
        );
        t.insert(format!("{p}.self_attn.q_proj.weight"), gen(vec![q, h]));
        t.insert(format!("{p}.self_attn.k_proj.weight"), gen(vec![kv, h]));
        t.insert(format!("{p}.self_attn.v_proj.weight"), gen(vec![kv, h]));
        t.insert(format!("{p}.self_attn.o_proj.weight"), gen(vec![h, q]));
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            gen(vec![cfg.intermediate_size, h]),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            gen(vec![cfg.intermediate_size, h]),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            gen(vec![h, cfg.intermediate_size]),
        );
    }
    t.insert("model.norm.weight".into(), Tensor::ones(vec![h]));
    Weights::from_map(t)
}

#[test]
fn scheduler_loop_matches_single_stream_greedy() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip continuous-batching parity: {e}");
            return;
        }
    };
    let c = cfg();
    let w = weights(&c);
    let max_tokens = 6;
    let prompts: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];

    // Reference: single-stream generation, one prompt at a time (default pool
    // sizing — its identity slot mapping needs exactly max_positions coverage).
    let expected: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            let m = arf_gpu::weights::build_gpu(&c, &w, c.max_position_embeddings, &ctx).unwrap();
            m.generate_stop(p, max_tokens, &[])
        })
        .collect();

    // Server loop: all three concurrently through the scheduler. The pool
    // geometry (built with default sizing) IS the scheduler config — so the
    // BlockManager only hands out ids the GPU pool covers.
    //
    // Run under BOTH attention kernels: the per-(head,row) `attention_batched`
    // (ARF_GQA=0) and the GQA-grouped one (ARF_GQA=1). The tiny cfg has
    // num_heads 4 / kv_heads 2 (group=2), so both paths are exercised; each must
    // be bit-equal to the single-stream reference. (env var is read per forward;
    // this test binary is its own process so the var doesn't leak to other tests.)
    let run_loop = || -> Vec<Vec<u32>> {
        let model = arf_gpu::weights::build_gpu(&c, &w, c.max_position_embeddings, &ctx).unwrap();
        let backend = WgpuBatched(model);
        let (nb, bs) = backend.kv_geometry();
        let mut sched = Scheduler::new(EngineConfig {
            block_size: bs,
            num_blocks: nb,
            max_batch_size: 8,
            max_prefill_tokens: 64,
            ..Default::default()
        });
        for (i, p) in prompts.iter().enumerate() {
            sched.add(Request::new(
                i as u64,
                p.clone(),
                SamplingParams::greedy(max_tokens),
            ));
        }
        let mut got: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
        while sched.has_unfinished() {
            let Some(plan) = sched.schedule().unwrap() else {
                break;
            };
            let (ids, batch) = build_forward(&plan, sched.block_size());
            let g = SamplingParams::greedy(max_tokens);
            let samp: Vec<SeqSampling> = plan
                .seqs
                .iter()
                .map(|sp| SeqSampling {
                    params: &g,
                    position: sp.past_len + sp.q_len,
                    generated: &[],
                })
                .collect();
            let tokens = backend.step(&ids, &batch, &samp).unwrap();
            for out in sched.commit_tokens(&tokens).unwrap() {
                got[out.id as usize].push(out.token);
            }
        }
        got
    };

    for forced in ["0", "1"] {
        std::env::set_var("ARF_GQA", forced);
        let got = run_loop();
        assert_eq!(
            got, expected,
            "batched server loop (ARF_GQA={forced}) must equal single-stream greedy"
        );
    }
    std::env::remove_var("ARF_GQA");
}

/// Invariant: reusing the KV of a shared prompt prefix
/// must NOT change the output. Three requests sharing a multi-block prefix are
/// run through the scheduler loop with prefix caching ON vs OFF; the greedy
/// tokens must be bit-identical, and the cache must actually have been hit.
#[test]
fn prefix_cache_matches_no_cache_greedy() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip prefix-cache parity: {e}");
            return;
        }
    };
    let c = cfg();
    let w = weights(&c);
    let max_tokens = 6;

    let probe = arf_gpu::weights::build_gpu(&c, &w, c.max_position_embeddings, &ctx).unwrap();
    let (_, bs) = WgpuBatched(probe).kv_geometry();

    // A shared prefix spanning three full blocks, then a per-request tail — so
    // the second/third requests reuse the leading blocks the first registered.
    let shared: Vec<u32> = (0..(3 * bs)).map(|i| (i % 7 + 1) as u32).collect();
    let mk = |tail: &[u32]| [shared.clone(), tail.to_vec()].concat();
    let prompts: Vec<Vec<u32>> = vec![mk(&[10, 11]), mk(&[12]), mk(&[13, 14, 15])];

    let run = |enable_prefix_cache: bool| -> (Vec<Vec<u32>>, u64) {
        let model = arf_gpu::weights::build_gpu(&c, &w, c.max_position_embeddings, &ctx).unwrap();
        let backend = WgpuBatched(model);
        let (nb, bs) = backend.kv_geometry();
        let mut sched = Scheduler::new(EngineConfig {
            block_size: bs,
            num_blocks: nb,
            max_batch_size: 8,
            max_prefill_tokens: 64,
            enable_prefix_cache,
            ..Default::default()
        });
        for (i, p) in prompts.iter().enumerate() {
            sched.add(Request::new(
                i as u64,
                p.clone(),
                SamplingParams::greedy(max_tokens),
            ));
        }
        let mut got: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
        while sched.has_unfinished() {
            let Some(plan) = sched.schedule().unwrap() else {
                break;
            };
            let (ids, batch) = build_forward(&plan, sched.block_size());
            let g = SamplingParams::greedy(max_tokens);
            let samp: Vec<SeqSampling> = plan
                .seqs
                .iter()
                .map(|sp| SeqSampling {
                    params: &g,
                    position: sp.past_len + sp.q_len,
                    generated: &[],
                })
                .collect();
            let tokens = backend.step(&ids, &batch, &samp).unwrap();
            for out in sched.commit_tokens(&tokens).unwrap() {
                got[out.id as usize].push(out.token);
            }
        }
        (got, sched.prefix_cache_reused_tokens())
    };

    let (base, base_reused) = run(false);
    let (cached, cached_reused) = run(true);
    assert_eq!(base_reused, 0, "caching off must report zero reuse");
    assert!(
        cached_reused > 0,
        "the shared multi-block prefix should have been reused"
    );
    assert_eq!(
        cached, base,
        "prefix-cached greedy output must be bit-identical to the no-cache run"
    );
}
