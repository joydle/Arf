//! Gemma-4 batched parity: the batched/serving path (`forward_batch` via
//! `WgpuBatched::step`) must produce the SAME greedy tokens as the single-stream
//! path (`generate_stop`) for a gemma-4-SHAPED tiny model — i.e. one with a GLOBAL
//! layer whose `head_dim` differs from the sliding layers (gemma-4 uses 512 vs 256),
//! plus qk_norm, weightless V-norm, embedding scale, GeGLU, layer_scalar, four-norm
//! blocks, and final-logit softcap.
//!
//! The generic `continuous_batching.rs` parity test uses a Llama-shaped config
//! (uniform head_dim, no global layers), so it never exercised the gemma-4
//! global-layer path — which is where batched gemma-4 was found to DRIFT from
//! single-stream over many layers. This test
//! reproduces that drift deterministically in seconds (no 6.5GB GGUF).
//!
//! Skips when no Metal device is present (CI).

use std::collections::HashMap;
use std::sync::Arc;

use arf_core::backend::BatchedBackend;
use arf_core::config::{
    AttnKind, EngineConfig, GateAct, MlpKind, ModelConfig, NormStyle, RopeScaling,
};
use arf_core::engine::build_forward;
use arf_core::model::weights::Weights;
use arf_core::sampling::{SamplingParams, SeqSampling};
use arf_core::scheduler::{Request, Scheduler};
use arf_core::Tensor;
use arf_gpu::gpu::GpuContext;
use arf_gpu::WgpuBatched;

// Use the REAL gemma-4-12B head dims: sliding 256, global 512. The 512-wide global
// head is what engages attention_batched's DPL=2 (dims-per-lane) path — the suspect
// for the batched-vs-single-stream drift. A small head_dim (<256) never hits DPL=2,
// so the bug hides (an earlier 8-wide config passed parity).
// Gemma-4-SHAPED but TINY: a global layer (head_dim 512) interleaved with sliding
// layers (head_dim 256), qk_norm, V-norm, GeGLU, layer_scalar, softcap — enough to
// validate the batched serving path STRUCTURALLY matches single-stream. (A separate
// scale sweep during the bug hunt confirmed the real-model fix; this stays a fast,
// deterministic correctness gate, not a numerical-stress test.)
const SLIDING_HD: usize = 256;
const GLOBAL_HD: usize = 512; // gemma-4: global head_dim ≠ sliding (512 vs 256)
const NHEADS: usize = 2;
const SLIDING_KV: usize = 2;
const GLOBAL_KV: usize = 1; // gemma-4-12B: global has 1 kv head
const NLAYERS: usize = 12; // global layers at i=2,5,8,11

fn envn(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// gemma-4-SHAPED tiny config: hybrid local/global with a distinct global head_dim.
/// ARF_T_LAYERS / _HEADS / _HIDDEN override scale for bisecting the threshold.
fn cfg() -> ModelConfig {
    let hidden = envn("ARF_T_HIDDEN", 64);
    ModelConfig {
        vocab_size: 32,
        hidden_size: hidden,
        intermediate_size: hidden * 2,
        num_layers: envn("ARF_T_LAYERS", NLAYERS),
        num_attention_heads: envn("ARF_T_HEADS", NHEADS),
        num_kv_heads: SLIDING_KV,
        head_dim: SLIDING_HD,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_scaling: RopeScaling::None,
        max_position_embeddings: 64,
        tie_word_embeddings: true,
        mlp: MlpKind::Dense,
        // Per-feature toggles to bisect WHICH gemma op causes the magnitude-sensitive
        // batched-vs-single-stream divergence. ARF_T_* unset = full gemma-4.
        qk_norm: std::env::var_os("ARF_T_NO_QKNORM").is_none(),
        norm_style: NormStyle::Gemma,
        attn: if std::env::var_os("ARF_T_CAUSAL").is_some() {
            AttnKind::Causal // uniform head_dim, no global layers
        } else {
            AttnKind::HybridLocalGlobal {
                window: 8,
                global_every: 3,
                local_theta: 10_000.0,
                global_theta: 1_000_000.0,
                global_head_dim: GLOBAL_HD,
                global_kv_heads: GLOBAL_KV,
                partial_rotary_factor: 0.25,
            }
        },
        embedding_scale: Some((hidden as f32).sqrt()),
        query_pre_attn_scalar: Some(1.0),
        final_logit_softcap: if std::env::var_os("ARF_T_NO_SOFTCAP").is_some() {
            None
        } else {
            Some(30.0)
        },
        value_norm: std::env::var_os("ARF_T_NO_VNORM").is_none(),
        gate_act: GateAct::GeluTanh,
        nextn_layers: 0,
    }
}

/// Deterministic tiny gemma weights. Per-layer geometry follows `cfg.layer_geometry`
/// (the global layer at i=2 has GLOBAL_HD/GLOBAL_KV). Norm weights are random (gemma
/// `(1+w)` makes ones() ≈ doubling — random exercises the real path).
fn weights(cfg: &ModelConfig) -> Weights {
    let h = cfg.hidden_size;
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
        let (hd, kvh, _) = cfg.layer_geometry(i);
        let q = NHEADS * hd;
        let kv = kvh * hd;
        // four gemma norms (small random so (1+w) stays ~1)
        for nm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            t.insert(format!("{p}.{nm}.weight"), gen(vec![h]));
        }
        t.insert(format!("{p}.self_attn.q_proj.weight"), gen(vec![q, h]));
        t.insert(format!("{p}.self_attn.k_proj.weight"), gen(vec![kv, h]));
        t.insert(format!("{p}.self_attn.v_proj.weight"), gen(vec![kv, h]));
        t.insert(format!("{p}.self_attn.o_proj.weight"), gen(vec![h, q]));
        // qk_norm (head_dim sized, per-layer)
        t.insert(format!("{p}.self_attn.q_norm.weight"), gen(vec![hd]));
        t.insert(format!("{p}.self_attn.k_norm.weight"), gen(vec![hd]));
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
    t.insert("model.norm.weight".into(), gen(vec![h]));
    Weights::from_map(t)
}

/// Run the parity check for a given config + quant: single-stream (one prompt at a
/// time) vs batched (all concurrent through the scheduler). Returns (got, expected).
///
/// `coop=false` pins `ARF_COOP_MAX_K=0` so EVERY matmul takes the bit-exact
/// sub-block GEMV — the kernel these tests were written to verify against the m=1
/// oracle. (Must be below the test's K=256; the coop guard is `kdim <= cap`, so a
/// cap of 256 would let the K=256 down_proj engage coop and break bit-identity.)
/// `coop=true` leaves the default (coop GEMM on): coop reassociates the dot product,
/// so it is NOT bit-identical to the GEMV; assert well-formedness, not equality.
fn run_parity(
    ctx: &Arc<GpuContext>,
    c: &ModelConfig,
    w: &Weights,
    quant: Option<arf_core::config::Quant>,
    coop: bool,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    // `ARF_COOP_MAX_K` is a PROCESS-GLOBAL env var that `coop_max_k()` reads per
    // dispatch. cargo runs tests in this binary on parallel threads, so the coop and
    // bit-exact tests would race on it (one sets "0", another removes it) mid-run —
    // an intermittent failure. Serialize the env-sensitive section with a static lock
    // held for the whole build+run, so each test sees a stable value end-to-end.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    if coop {
        std::env::remove_var("ARF_COOP_MAX_K");
    } else {
        std::env::set_var("ARF_COOP_MAX_K", "0");
    }
    let max_tokens = 6;
    // ARF_T_ONE=1 → a SINGLE prompt (isolates multi-token-prefill m>1 from
    // multi-sequence batching: a single seq still prefills m=len at once).
    let prompts: Vec<Vec<u32>> = if let Some(seed) = std::env::var("ARF_T_SEED")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        vec![(0..4u32)
            .map(|i| (seed.wrapping_mul(7).wrapping_add(i * 5)) % 30 + 1)
            .collect()] // a len-4 prompt varied by seed
    } else if let Some(len) = std::env::var("ARF_T_LEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        vec![(1..=len as u32).map(|i| (i * 3 + 1) % 30 + 1).collect()] // single prompt of length `len`
    } else if std::env::var_os("ARF_T_ONE3").is_some() {
        vec![vec![1, 2, 3]]
    } else if std::env::var_os("ARF_T_ONE").is_some() {
        vec![vec![6, 7, 8, 9]]
    } else if std::env::var_os("ARF_T_TWINS").is_some() {
        // Two IDENTICAL prompts: their outputs MUST be identical to each other (and
        // to single-stream). If they differ → cross-row contamination in the decode
        // step; if both wrong-but-equal → a multi-row op systematically off.
        vec![vec![6, 7, 8, 9], vec![6, 7, 8, 9]]
    } else {
        vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]]
    };
    let build = |ctx: &Arc<GpuContext>| match quant {
        Some(q) => {
            arf_gpu::weights::build_gpu_quant(c, w, c.max_position_embeddings, ctx, q).unwrap()
        }
        None => arf_gpu::weights::build_gpu(c, w, c.max_position_embeddings, ctx).unwrap(),
    };

    let expected: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| build(ctx).generate_stop(p, max_tokens, &[]))
        .collect();

    let backend = WgpuBatched(build(ctx));
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
    (got, expected)
}

#[test]
fn gemma4_batched_matches_single_stream_f32() {
    let ctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("no Metal device — skipping");
            return;
        }
    };
    let c = cfg();
    let w = weights(&c);
    let (got, expected) = run_parity(&ctx, &c, &w, None, false);
    for (s, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let first = g.iter().zip(e).position(|(a, b)| a != b);
        eprintln!(
            "[SEQ {s}] match={} first_diff_tok={:?} got={g:?} exp={e:?}",
            g == e,
            first
        );
    }
    assert_eq!(
        got, expected,
        "f32 batched gemma-4 (GEMV path) diverges from single-stream"
    );
}

/// THE REAL-MODEL CASE: Q4_K weights. Single-stream decode uses `matmul_vec_q4k`
/// (m=1); batched prefill uses `matmul_vec_q4k_batch` (m>1) — DIFFERENT kernels. The
/// real gemma-4-12B GGUF is Q4_K and garbages batched while f32 may pass. All
/// quantized dims must be multiples of 256.
#[test]
fn gemma4_batched_matches_single_stream_q4k() {
    let ctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("no Metal device — skipping");
            return;
        }
    };
    let c = cfg_q4k();
    let w = weights(&c);
    let (got, expected) = run_parity(&ctx, &c, &w, Some(arf_core::config::Quant::Q4K), false);
    assert_eq!(
        got, expected,
        "Q4K batched gemma-4 (GEMV path) diverges from single-stream"
    );
}

/// THE COOP-GEMM CASE: with `ARF_COOP_MAX_K` at its default (8192), gemma's
/// batched prefill/decode (m≥8) runs on the cooperative-matrix GEMM, not the GEMV.
/// Coop reassociates the dot product, so it is NOT bit-identical to the single-stream
/// GEMV oracle — but on real weights it is token-IDENTICAL (57/57 trajectories in
/// `examples/coop_gemma_match`). This synthetic tiny config uses random weights that
/// exaggerate softcap near-ties, so we don't demand exact token equality; instead we
/// assert the coop path RUNS and produces well-formed output of the right shape (a
/// smoke + shape guard). Real-weight token-match is covered by the example sweep and
/// the CLI `--chat` checks; this guards against coop crashing or emitting garbage
/// lengths once it's the default batched path.
#[test]
fn gemma4_batched_coop_runs_and_is_well_formed() {
    let ctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("no Metal device — skipping");
            return;
        }
    };
    let c = cfg_q4k();
    let w = weights(&c);
    let (got, _expected) = run_parity(&ctx, &c, &w, Some(arf_core::config::Quant::Q4K), true);
    // Coop path must produce one token stream per prompt, each of the requested
    // length, with valid vocab ids — i.e. it ran without crashing or collapsing.
    assert_eq!(got.len(), 3, "coop batched gemma-4 lost a sequence");
    for (s, g) in got.iter().enumerate() {
        assert_eq!(
            g.len(),
            6,
            "[SEQ {s}] coop produced {} tokens, expected 6",
            g.len()
        );
        for &t in g {
            assert!(
                (t as usize) < c.vocab_size,
                "[SEQ {s}] coop emitted out-of-range token {t}"
            );
        }
    }
}

/// Like `cfg()` but every quantized dim is a multiple of 256 (Q4_K requirement):
/// hidden 256, intermediate 256, sliding head_dim 256, global 512.
fn cfg_q4k() -> ModelConfig {
    ModelConfig {
        hidden_size: 256,
        intermediate_size: 256,
        embedding_scale: Some((256.0f32).sqrt()),
        ..cfg()
    }
}
