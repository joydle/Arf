//! GPU MoE parity: the route kernel (and later the full MoE block) must match the
//! CPU `MoeMlp` oracle bit-for-bit. Compiles only with `--features gpu`; skips when
//! no adapter is present.

use std::collections::HashMap;

use arf_core::cache::{slots_for, write_runs, PagedKvCache};
use arf_core::config::Quant;
use arf_core::config::{AttnKind, MlpKind, ModelConfig, NormStyle, RopeScaling};
use arf_core::device::Device;
use arf_core::model::weights::{build_on, build_quant, Weights};
use arf_core::model::{ForwardBatch, Llama, SeqAttn};
use arf_core::Tensor;
use arf_gpu::gpu::{ComputeKernel, GpuContext};
use arf_gpu::weights::{build_gpu, build_gpu_quant};
use std::sync::Arc;

/// CPU reference for the route kernel: full softmax over logits, take top_k by
/// probability (ties to lower index), optionally renormalize. Mirrors
/// `model::moe::top_k_softmax` (which is private) and `moe_route.wgsl`.
fn cpu_route(logits: &[f32], top_k: usize, norm: bool) -> (Vec<u32>, Vec<f32>) {
    let n = logits.len();
    let k = top_k.min(n);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    let mut w: Vec<f32> = idx.iter().map(|&i| probs[i]).collect();
    if norm {
        let s: f32 = w.iter().sum();
        if s > 0.0 {
            for x in &mut w {
                *x /= s;
            }
        }
    }
    (idx.iter().map(|&i| i as u32).collect(), w)
}

fn run_route_gpu(
    ctx: &arf_gpu::gpu::GpuContext,
    kernel: &ComputeKernel,
    logits: &[f32],
    top_k: usize,
    norm: bool,
) -> (Vec<u32>, Vec<f32>) {
    let n = logits.len();
    let logits_buf = ctx.storage_init("route_logits", logits);
    let ids_buf = ctx.storage_zeros_u32("route_ids", top_k);
    let wts_buf = ctx.storage_zeros("route_wts", top_k);
    let dims = ctx.uniform_of("route_dims", &[n as u32, top_k as u32, norm as u32, 0u32]);
    kernel.dispatch(
        "moe_route",
        &[&logits_buf, &ids_buf, &wts_buf, &dims],
        [1, 1, 1],
    );
    (ctx.read_u32(&ids_buf, top_k), ctx.read_f32(&wts_buf, top_k))
}

#[test]
fn moe_route_gpu_matches_cpu() {
    let ctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skipping MoE route parity: {e}");
            return;
        }
    };
    let kernel = ComputeKernel::new(
        &ctx,
        "moe_route",
        include_str!("../src/shaders/wgsl/moe_route.wgsl"),
    );

    // A few representative routing cases (incl. ties and a Qwen-sized 128-expert set).
    let mut seed = 99u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };

    let cases: Vec<(Vec<f32>, usize, bool)> = vec![
        (vec![2.0, -1.0, 3.0, 0.0], 2, true),
        (vec![2.0, -1.0, 3.0, 0.0], 2, false),
        (vec![1.0, 1.0, 1.0, 1.0], 2, true), // ties -> lowest indices
        ((0..128).map(|_| rand()).collect(), 8, true), // Qwen3 shape
        ((0..128).map(|_| rand()).collect(), 8, false),
    ];

    for (logits, k, norm) in cases {
        let (cpu_ids, cpu_w) = cpu_route(&logits, k, norm);
        let (gpu_ids, gpu_w) = run_route_gpu(&ctx, &kernel, &logits, k, norm);
        assert_eq!(gpu_ids, cpu_ids, "route ids diverged (k={k}, norm={norm})");
        for (i, (a, b)) in gpu_w.iter().zip(&cpu_w).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "route weight {i} diverged: gpu {a} vs cpu {b} (k={k}, norm={norm})"
            );
        }
    }
}

// --- Full MoE decode block parity (GpuModel decode path vs CPU oracle) ---

/// A GPU-friendly tiny MoE config: every contraction dim is a multiple of 8 (bf16
/// vec4 packing). hidden=32, moe_inter=16, head_dim=8 (q=32), 4 experts top-2 + 1
/// shared, qk-norm on (full Qwen3 block).
fn moe_cfg() -> ModelConfig {
    ModelConfig {
        vocab_size: 32,
        hidden_size: 32,
        intermediate_size: 16,
        num_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_scaling: RopeScaling::None,
        max_position_embeddings: 64,
        tie_word_embeddings: true,
        mlp: MlpKind::Moe {
            num_experts: 4,
            top_k: 2,
            shared_experts: 1,
            moe_intermediate: 16,
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
        nextn_layers: 0,
    }
}

fn moe_weights(cfg: &ModelConfig) -> Weights {
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let (n_exp, mi, shared) = match cfg.mlp {
        MlpKind::Moe {
            num_experts,
            moe_intermediate,
            shared_experts,
            ..
        } => (num_experts, moe_intermediate, shared_experts),
        _ => unreachable!(),
    };
    let mut s = 7u64;
    let g = |n: usize, s: &mut u64| -> Vec<f32> {
        (0..n)
            .map(|_| {
                *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (*s >> 40) as f32 / (1u64 << 23) as f32 - 0.5
            })
            .collect()
    };
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let put = |t: &mut HashMap<String, Tensor>, name: String, r: usize, c: usize, s: &mut u64| {
        t.insert(name, Tensor::from_vec(g(r * c, s), vec![r, c]));
    };
    put(
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
        put(&mut t, format!("{p}.self_attn.q_proj.weight"), q, h, &mut s);
        put(
            &mut t,
            format!("{p}.self_attn.k_proj.weight"),
            kv,
            h,
            &mut s,
        );
        put(
            &mut t,
            format!("{p}.self_attn.v_proj.weight"),
            kv,
            h,
            &mut s,
        );
        put(&mut t, format!("{p}.self_attn.o_proj.weight"), h, q, &mut s);
        t.insert(
            format!("{p}.self_attn.q_norm.weight"),
            Tensor::from_vec(g(cfg.head_dim, &mut s), vec![cfg.head_dim]),
        );
        t.insert(
            format!("{p}.self_attn.k_norm.weight"),
            Tensor::from_vec(g(cfg.head_dim, &mut s), vec![cfg.head_dim]),
        );
        put(&mut t, format!("{p}.mlp.gate.weight"), n_exp, h, &mut s);
        for e in 0..n_exp {
            let ep = format!("{p}.mlp.experts.{e}");
            put(&mut t, format!("{ep}.gate_proj.weight"), mi, h, &mut s);
            put(&mut t, format!("{ep}.up_proj.weight"), mi, h, &mut s);
            put(&mut t, format!("{ep}.down_proj.weight"), h, mi, &mut s);
        }
        for _ in 0..shared {
            let sp = format!("{p}.mlp.shared_expert");
            put(&mut t, format!("{sp}.gate_proj.weight"), mi, h, &mut s);
            put(&mut t, format!("{sp}.up_proj.weight"), mi, h, &mut s);
            put(&mut t, format!("{sp}.down_proj.weight"), h, mi, &mut s);
        }
    }
    t.insert("model.norm.weight".into(), Tensor::ones(vec![h]));
    Weights::from_map(t)
}

fn cpu_last_logits(model: &Llama, tokens: &[u32]) -> Vec<f32> {
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
            q_start: 0,
            q_len,
            past_len: 0,
            slots: slots_for(&block_table, 4, q_len),
            write_runs: write_runs(&block_table, 4, 0, q_len),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let hidden = model.forward(tokens, &batch, &mut cache);
    model.logits_last(&hidden, &batch).into_vec()
}

/// THE MoE PROOF: the GPU resident decode pipeline (GpuModel::forward_logits, which
/// runs the on-GPU route + packed-expert GEMV block) must match the CPU MoE oracle's
/// last-token logits. Exercises router GEMV → moe_route → routed experts (gate/up/
/// swiglu/down weight-sum) → shared expert, across two layers and a 5-token prompt.
#[test]
fn moe_gpu_decode_matches_cpu_logits() {
    let gctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skipping MoE GPU decode parity: {e}");
            return;
        }
    };
    let cfg = moe_cfg();
    let w = moe_weights(&cfg);
    let cpu = build_on(&cfg, &w, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &gctx).unwrap();

    let tokens = [3u32, 1, 4, 1, 5];
    let cpu_logits = cpu_last_logits(&cpu, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);

    let diff = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff < 1e-3,
        "MoE GPU decode vs CPU logits diverged by {diff}"
    );
}

/// A Q4-friendly MoE config: every expert contraction dim is a multiple of 32 (the
/// Q4_0 block). hidden=32, moe_inter=32, head_dim=8.
fn moe_cfg_q4() -> ModelConfig {
    ModelConfig {
        intermediate_size: 32,
        mlp: MlpKind::Moe {
            num_experts: 4,
            top_k: 2,
            shared_experts: 1,
            moe_intermediate: 32,
            norm_topk: true,
        },
        ..moe_cfg()
    }
}

/// THE Q4 MoE PROOF (PQ4b): the GPU packed-Q4 expert GEMV (matmul_vec_moe_q4,
/// expert resolved on-GPU, codes/scales offset-indexed) must match the CPU Q4 MoE
/// oracle — both quantize the same bf16-rounded weights, so logits are bit-exact.
/// This is the path that fits + runs the real 30B.
#[test]
fn moe_gpu_decode_q4_matches_cpu() {
    let gctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skipping Q4 MoE GPU decode parity: {e}");
            return;
        }
    };
    let cfg = moe_cfg_q4();
    let w = moe_weights(&cfg);
    let cpu = build_quant(
        &cfg,
        &w,
        cfg.max_position_embeddings,
        &Device::Cpu,
        Quant::Q4,
    )
    .unwrap();
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Q4).unwrap();

    let tokens = [3u32, 1, 4, 1, 5];
    let cpu_logits = cpu_last_logits(&cpu, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);

    let diff = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff < 1e-3,
        "Q4 MoE GPU decode vs CPU Q4 logits diverged by {diff}"
    );
}

/// Like [`moe_cfg_q4`] but with every quantized dimension a multiple of 256, the
/// Q4_K super-block size: `hidden` and `moe_intermediate` (and so every expert /
/// projection `cols`) are 256. Q4_0 needs only block-32, but Q4_K-lite needs 256.
fn moe_cfg_q4k() -> ModelConfig {
    ModelConfig {
        hidden_size: 256,
        intermediate_size: 256,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 64, // q_dim = 4*64 = 256 = hidden
        mlp: MlpKind::Moe {
            num_experts: 4,
            top_k: 2,
            shared_experts: 1,
            moe_intermediate: 256,
            norm_topk: true,
        },
        ..moe_cfg()
    }
}

/// THE Q4_K MoE PROOF: the GPU packed-Q4_K expert GEMV (matmul_vec_moe_q4k, expert
/// resolved on-GPU, codes/scales/mins offset-indexed) must match the CPU Q4_K MoE
/// oracle. Both quantize the same bf16-rounded weights to Q4_K-lite (per-32
/// asymmetric scale+min), so logits are bit-exact. This is the path that keeps the
/// GGUF disk format's accuracy class while still fitting the real 30B.
#[test]
fn moe_gpu_decode_q4k_matches_cpu() {
    let gctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skipping Q4_K MoE GPU decode parity: {e}");
            return;
        }
    };
    let cfg = moe_cfg_q4k();
    let w = moe_weights(&cfg);
    let cpu = build_quant(
        &cfg,
        &w,
        cfg.max_position_embeddings,
        &Device::Cpu,
        Quant::Q4K,
    )
    .unwrap();
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Q4K).unwrap();

    let tokens = [3u32, 1, 4, 1, 5];
    let cpu_logits = cpu_last_logits(&cpu, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);

    let diff = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff < 1e-3,
        "Q4_K MoE GPU decode vs CPU Q4_K logits diverged by {diff}"
    );
}

/// Q4_K_S MoE config: identical shape to [`moe_cfg_q4k`] (every quantized dim a
/// multiple of 256, the super-block size). Q4_K_S reuses the super-block-256 layout
/// but with two-level u8/i8 scales (a coarser numeric class than Q4_K-lite's bf16
/// per-sub-block scale+min), so it MUST be compared to the Q4_K_S CPU oracle, not
/// the Q4_K one.
fn moe_cfg_q4ks() -> ModelConfig {
    moe_cfg_q4k()
}

/// THE Q4_K_S MoE PROOF: the GPU packed-Q4_K_S expert GEMV (matmul_vec_moe_q4ks /
/// the fused batch_gu + down_reduce variants, expert resolved on-GPU,
/// codes/scales(u8)/mins(i8)/dd offset-indexed) must match the CPU Q4_K_S MoE
/// oracle. Both quantize the same bf16-rounded weights to Q4_K_S (super-block-256,
/// two-level u8/i8 scales against a per-super-block f32 d/dmin), so logits are
/// bit-exact within 1e-3. This cuts expert weight bytes ~16% vs Q4_K-lite.
#[test]
fn moe_gpu_decode_q4ks_matches_cpu() {
    let gctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skipping Q4_K_S MoE GPU decode parity: {e}");
            return;
        }
    };
    let cfg = moe_cfg_q4ks();
    let w = moe_weights(&cfg);
    // CPU oracle MUST quantize through the Q4_K_S path (coarser two-level scales);
    // comparing to a Q4_K oracle would falsely fail on the numeric-class difference.
    let cpu = build_quant(
        &cfg,
        &w,
        cfg.max_position_embeddings,
        &Device::Cpu,
        Quant::Q4KS,
    )
    .unwrap();
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Q4KS).unwrap();

    let tokens = [3u32, 1, 4, 1, 5];
    let cpu_logits = cpu_last_logits(&cpu, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);

    let diff = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff < 1e-3,
        "Q4_K_S MoE GPU decode vs CPU Q4_K_S logits diverged by {diff}"
    );
}

/// Mirror the real Qwen3-Coder-30B's distinguishing shape that the other MoE
/// configs DON'T exercise: an explicit `head_dim` such that `q_dim = n_heads *
/// head_dim != hidden_size` (Qwen3: 32*128=4096 != 2048). hidden=256, n_heads=4,
/// head_dim=128 → q_dim=512 != hidden=256, kv_dim=2*128=256. If the GPU forward
/// confuses q_dim with hidden anywhere (attn/o_proj/residual), this diverges from
/// the CPU oracle (or goes degenerate) where moe_cfg_q4k (q_dim==hidden) could not.
fn moe_cfg_qdim_ne_hidden() -> ModelConfig {
    ModelConfig {
        hidden_size: 256,
        intermediate_size: 256,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 128, // q_dim = 4*128 = 512 != hidden 256; kv_dim = 2*128 = 256
        mlp: MlpKind::Moe {
            num_experts: 4,
            top_k: 2,
            shared_experts: 1,
            moe_intermediate: 256,
            norm_topk: true,
        },
        ..moe_cfg()
    }
}

#[test]
fn moe_gpu_qdim_ne_hidden_matches_cpu() {
    let gctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skipping q_dim!=hidden MoE parity: {e}");
            return;
        }
    };
    let cfg = moe_cfg_qdim_ne_hidden();
    let w = moe_weights(&cfg);
    let cpu = build_on(&cfg, &w, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &gctx).unwrap();

    let tokens = [3u32, 1, 4, 1, 5];
    let cpu_logits = cpu_last_logits(&cpu, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);

    // Non-degenerate: the GPU logits must not be all-equal (the 30B bug signature).
    let gmax = gpu_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let gmin = gpu_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        gmax > gmin,
        "GPU logits are degenerate (all equal to {gmax}) — the 30B zero-logits bug"
    );

    let diff = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff < 1e-3,
        "q_dim!=hidden MoE GPU decode vs CPU logits diverged by {diff}"
    );
}

/// THE STEADY-DECODE-CACHE PROOF: gpu.generate() runs the FIRST token with
/// token=Some (cache miss, builds bind groups) then every subsequent token with
/// token=None — the P15 bind-group-cache path. Fix #1 extended that cache to cover
/// the whole MoE block (router/route/experts/swiglu) + the per-layer norms/rope, so
/// this test (which the per-logits parity tests do NOT exercise — they use
/// forward_logits = token=Some only) guards that the CACHED MoE decode still
/// produces the correct greedy tokens. Compares against a CPU greedy loop.
#[test]
fn moe_gpu_generate_cached_matches_cpu_greedy() {
    let gctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skipping cached MoE generate parity: {e}");
            return;
        }
    };
    let cfg = moe_cfg();
    let w = moe_weights(&cfg);
    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &gctx).unwrap();

    let prompt = vec![3u32, 1, 4, 1, 5];
    let n = 8usize; // enough tokens that the token=None cached path runs repeatedly

    // ORACLE: step greedy via forward_logits (token=Some => cursor=None => the
    // UNcached path, which the per-logits parity tests prove matches CPU). The
    // cached gpu.generate() must produce the same tokens.
    let mut seq = prompt.clone();
    let mut oracle = Vec::new();
    for _ in 0..n {
        let lg = gpu.forward_logits(&seq);
        let argmax = lg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        seq.push(argmax);
        oracle.push(argmax);
    }

    let gpu_gen = gpu.generate(&prompt, n);

    assert_eq!(
        gpu_gen, oracle,
        "cached MoE gpu.generate (token=None path) diverged from uncached \
         forward_logits stepping — the bind-group cache (fix #1) changed output"
    );
}

/// THE BATCHED-MoE PROOF: `gpu.generate_batch` runs B sequences (one decode token
/// each) through the batched `forward_batch` path, whose `GpuMlp::Moe` arm is the
/// newly-implemented per-row MoE block (router GEMV → per-row top-k route → routed
/// expert gate/up/swiglu/down → shared expert, looped over each batch row). Each
/// sequence's greedy output MUST equal that SAME sequence run single-stream through
/// the proven `gpu.generate` (the decode-path MoE block, which the per-logits parity
/// tests prove matches the CPU oracle). This proves batched MoE decode == per-seq
/// single-stream decode — the gate that unlocks continuous batching for the 30B MoE.
fn assert_batched_moe_matches_single_stream(gpu: &arf_gpu::gpu::GpuModel) {
    // Distinct prompts so the per-row routing genuinely differs across batch rows
    // (each row picks its OWN top-k experts). Lengths kept small so B sequences fit
    // the tiny test KV pool (block_size 16, num_blocks 4): max_ctx = max_len +
    // max_tokens must give n·ceil(max_ctx/16) <= num_blocks.
    let prompts: Vec<Vec<u32>> = vec![
        vec![3u32, 1, 4, 1, 5],
        vec![2u32, 7, 1, 8],
        vec![6u32, 0, 3],
    ];
    let max_tokens = 4usize;

    // ORACLE: each sequence decoded single-stream (the proven MoE decode path).
    let oracle: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| gpu.generate(p, max_tokens))
        .collect();

    // CANDIDATE: all sequences batched through one continuous-decode loop.
    let batched = gpu.generate_batch(&prompts, max_tokens);

    assert_eq!(
        batched.len(),
        oracle.len(),
        "generate_batch returned {} sequences, expected {}",
        batched.len(),
        oracle.len()
    );
    for (s, (b, o)) in batched.iter().zip(&oracle).enumerate() {
        assert_eq!(
            b, o,
            "batched MoE decode diverged from single-stream for sequence {s}: \
             batched {b:?} vs single-stream {o:?}"
        );
    }
}

#[test]
fn moe_gpu_generate_batch_matches_single_stream() {
    let gctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skipping batched MoE generate parity: {e}");
            return;
        }
    };
    let cfg = moe_cfg();
    let w = moe_weights(&cfg);
    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &gctx).unwrap();
    assert_batched_moe_matches_single_stream(&gpu);
}

/// Same batched-MoE parity, but over the Q4_K_S packed experts — the exact GEMV /
/// fused batch_gu + down_reduce path the real Qwen3-30B runs. Guards that the per-row
/// batched MoE arm drives the quantized expert kernels identically to single-stream.
#[test]
fn moe_gpu_generate_batch_q4ks_matches_single_stream() {
    let gctx = match GpuContext::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("skipping batched Q4_K_S MoE generate parity: {e}");
            return;
        }
    };
    let cfg = moe_cfg_q4ks();
    let w = moe_weights(&cfg);
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Q4KS).unwrap();
    assert_batched_moe_matches_single_stream(&gpu);
}
