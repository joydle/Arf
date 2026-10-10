//! GPU backend parity: a model built on the GPU must produce the same logits as
//! the same model on the CPU. Compiles only with `--features gpu`, and skips
//! gracefully when no GPU adapter is present (e.g. CI).

use std::collections::HashMap;

use arf_core::cache::{slots_for, write_runs, PagedKvCache};
use arf_core::config::{ModelConfig, RopeScaling};
use arf_core::device::Device;
use arf_core::model::weights::{build_on, Weights};
use arf_core::model::{ForwardBatch, Llama, SeqAttn};
use arf_core::Tensor;
use arf_gpu::gpu::GpuContext;

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

/// A Q4-friendly tiny config: every matmul's `cols` (the contraction dim) is a
/// multiple of 32, the Q4_0 block size. h=32 (q/k/v/gate/up/lm_head cols), q=32
/// (o_proj cols), intermediate=64 (down_proj cols). `weights()` is dim-agnostic.
fn cfg_q4() -> ModelConfig {
    // Smaller than `cfg()`: Q4_0 quantizes in 32-weight blocks, so the dims need only divide 32.
    ModelConfig {
        hidden_size: 32,
        intermediate_size: 64,
        head_dim: 8,
        ..cfg()
    }
}

/// A Q4_K-friendly tiny config: every matmul's contraction dim is a multiple of
/// 256 (the Q4_K super-block). h=256 (q/k/v/gate/up/lm_head cols), q=256 (o_proj),
/// intermediate=512 (down_proj). `weights()` is dim-agnostic.
fn cfg_q4k() -> ModelConfig {
    // Dims chosen so `cols % 256 == 0` — Q4_K/Q4_K_S quantize in 256-weight super-blocks.
    ModelConfig {
        vocab_size: 64,
        hidden_size: 256,
        intermediate_size: 512,
        head_dim: 64,
        ..cfg()
    }
}

/// `cfg()` plus per-head q/k RMSNorm (Qwen3-style). Same dims, qk_norm on.
fn cfg_qknorm() -> ModelConfig {
    ModelConfig {
        qk_norm: true,
        ..cfg()
    }
}

/// An architecturally complete tiny Gemma config: four-norm blocks, QK-norm,
/// √hidden embedding scale, and hybrid local/global attention with a small
/// sliding window (`global_every = 2` → layers 0,2 local, layer 1 global) and two
/// RoPE bases. Exercises every Gemma-specific branch of the GPU decode path.
fn cfg_gemma() -> ModelConfig {
    ModelConfig {
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
        nextn_layers: 0,
    }
}

/// `weights()` extended with Gemma's two feed-forward norms; the shared `weights`
/// generator already emits the per-head q/k norms when `cfg.qk_norm`.
fn gemma_weights(cfg: &ModelConfig) -> Weights {
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let mut seed = 11u64;
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
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            t.insert(format!("{p}.{norm}.weight"), gen(vec![h]));
        }
        t.insert(format!("{p}.self_attn.q_proj.weight"), gen(vec![q, h]));
        t.insert(format!("{p}.self_attn.k_proj.weight"), gen(vec![kv, h]));
        t.insert(format!("{p}.self_attn.v_proj.weight"), gen(vec![kv, h]));
        t.insert(format!("{p}.self_attn.o_proj.weight"), gen(vec![h, q]));
        t.insert(
            format!("{p}.self_attn.q_norm.weight"),
            gen(vec![cfg.head_dim]),
        );
        t.insert(
            format!("{p}.self_attn.k_norm.weight"),
            gen(vec![cfg.head_dim]),
        );
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

fn weights(cfg: &ModelConfig) -> Weights {
    weights_with(cfg, false)
}

/// `weights()` plus a dedicated `lm_head.weight` holding the SAME values as the embedding table —
/// for exercising the untied path without changing what the weights contain.
fn weights_untied(cfg: &ModelConfig) -> Weights {
    weights_with(cfg, true)
}

/// Deterministic test weights for `cfg`. With `untied`, also emits `lm_head.weight` carrying the
/// embedding table's values, so the tied and untied paths differ only in WHERE the lm_head weight
/// comes from — never in what is in it.
fn weights_with(cfg: &ModelConfig, untied: bool) -> Weights {
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
    let embed = gen(vec![cfg.vocab_size, h]);
    if untied {
        t.insert("lm_head.weight".into(), embed.clone());
    }
    t.insert("model.embed_tokens.weight".into(), embed);
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
        if cfg.qk_norm {
            t.insert(
                format!("{p}.self_attn.q_norm.weight"),
                gen(vec![cfg.head_dim]),
            );
            t.insert(
                format!("{p}.self_attn.k_norm.weight"),
                gen(vec![cfg.head_dim]),
            );
        }
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

#[test]
fn gpu_matches_cpu_logits() {
    let device = match arf_gpu::eager_device() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping GPU parity test: {e}");
            return;
        }
    };
    eprintln!("running on {}", device.describe());

    let cfg = cfg();
    let w = weights(&cfg);
    let cpu = build_on(&cfg, &w, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let gpu = build_on(&cfg, &w, cfg.max_position_embeddings, &device).unwrap();

    let tokens = [3u32, 1, 4, 1, 5];
    let a = last_logits(&cpu, &tokens);
    let b = last_logits(&gpu, &tokens);
    let max_diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "GPU vs CPU logits diverged by {max_diff}");
}

/// QK-norm parity: a model with per-head q/k RMSNorm (Qwen3) must match CPU↔GPU.
/// Exercises both the decode (single-token) and prefill (multi-token) qk_norm
/// dispatches, since `last_logits` runs a 5-token prompt through forward_batch.
#[test]
fn gpu_matches_cpu_logits_qk_norm() {
    let device = match arf_gpu::eager_device() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping QK-norm GPU parity test: {e}");
            return;
        }
    };
    let cfg = cfg_qknorm();
    let w = weights(&cfg);
    let cpu = build_on(&cfg, &w, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let gpu = build_on(&cfg, &w, cfg.max_position_embeddings, &device).unwrap();

    let tokens = [3u32, 1, 4, 1, 5];
    let a = last_logits(&cpu, &tokens);
    let b = last_logits(&gpu, &tokens);
    let max_diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "QK-norm GPU vs CPU logits diverged by {max_diff}"
    );
}

/// Foundation smoke test: the ComputeKernel dispatch + resident-buffer helpers
/// round-trip correctly (upload -> run a trivial doubling shader -> read back).
#[test]
fn compute_kernel_dispatch_roundtrips() {
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU foundation test: {e}");
            return;
        }
    };

    const DOUBLE_WGSL: &str = include_str!("../src/shaders/wgsl/double_test.wgsl");
    let kernel = ComputeKernel::new(&ctx, "double", DOUBLE_WGSL);

    let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let in_buf = ctx.storage_init("in", &input);
    let out_buf = ctx.storage_zeros("out", input.len());

    // 100 elements, workgroup_size 64 -> 2 workgroups.
    let groups = (input.len() as u32).div_ceil(64);
    kernel.dispatch("double", &[&in_buf, &out_buf], [groups, 1, 1]);

    let out = ctx.read_f32(&out_buf, input.len());
    for (i, (&got, &x)) in out.iter().zip(&input).enumerate() {
        assert_eq!(got, 2.0 * x, "lane {i}");
    }
}

/// Chaining two dispatches into ONE command buffer (the decode-step primitive):
/// double then double again (in -> tmp -> out) = ×4, with a single submit.
#[test]
fn command_pass_chains_dispatches_in_one_submit() {
    use arf_gpu::gpu::{CommandPass, ComputeKernel};

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU CommandPass test: {e}");
            return;
        }
    };

    const DOUBLE_WGSL: &str = include_str!("../src/shaders/wgsl/double_test.wgsl");
    let kernel = ComputeKernel::new(&ctx, "double", DOUBLE_WGSL);

    let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let in_buf = ctx.storage_init("in", &input);
    let tmp_buf = ctx.storage_zeros("tmp", input.len());
    let out_buf = ctx.storage_zeros("out", input.len());
    let groups = (input.len() as u32).div_ceil(64);

    let mut cp = CommandPass::new(&ctx);
    cp.add(&kernel, "double-1", &[&in_buf, &tmp_buf], [groups, 1, 1]);
    cp.add(&kernel, "double-2", &[&tmp_buf, &out_buf], [groups, 1, 1]);
    cp.submit(); // both passes, one command buffer, one submit

    let out = ctx.read_f32(&out_buf, input.len());
    for (i, (&got, &x)) in out.iter().zip(&input).enumerate() {
        assert_eq!(got, 4.0 * x, "lane {i} (doubled twice)");
    }
}

/// GpuModel::new (via build_gpu) uploads every weight + scratch into resident
/// buffers without error, on the synthetic config.
#[test]
fn gpu_model_new_uploads_resident_weights() {
    use arf_gpu::weights::build_gpu;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GpuModel::new test: {e}");
            return;
        }
    };

    let cfg = cfg();
    let w = weights(&cfg);
    let model = build_gpu(&cfg, &w, cfg.max_position_embeddings, &ctx).expect("build_gpu");
    assert_eq!(model.layers.len(), cfg.num_layers);
}

/// Pack a `Dims { tokens, hidden, eps, _pad }` uniform: same 16-byte layout as
/// the WGSL struct, with `eps` carried as its f32 bit pattern.
fn dims4(tokens: u32, hidden: u32, eps: f32) -> [u32; 4] {
    [tokens, hidden, eps.to_bits(), 0]
}

/// Max absolute difference between two equal-length slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn rmsnorm_gpu_matches_cpu() {
    use arf_core::model::rms_norm::RmsNorm;
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping rmsnorm parity: {e}");
            return;
        }
    };
    const RMSNORM_WGSL: &str = include_str!("../src/shaders/wgsl/rmsnorm.wgsl");
    let kernel = ComputeKernel::new(&ctx, "rmsnorm", RMSNORM_WGSL);

    let (tokens, hidden, eps) = (3usize, 2048usize, 1e-5f32);
    let mut seed = 11u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let x: Vec<f32> = (0..tokens * hidden).map(|_| rand()).collect();
    let weight: Vec<f32> = (0..hidden).map(|_| 0.5 + rand()).collect();

    // CPU oracle.
    let cpu = RmsNorm::new(Tensor::from_vec(weight.clone(), vec![hidden]), eps as f64);
    let expected = cpu.forward(&Tensor::from_vec(x.clone(), vec![tokens, hidden]));

    // GPU: one workgroup per row.
    let x_buf = ctx.storage_init("x", &x);
    let w_buf = ctx.storage_init("w", &weight);
    let y_buf = ctx.storage_zeros("y", tokens * hidden);
    let dims = ctx.uniform_of("dims", &dims4(tokens as u32, hidden as u32, eps));
    kernel.dispatch(
        "rmsnorm",
        &[&x_buf, &w_buf, &y_buf, &dims],
        [tokens as u32, 1, 1],
    );
    let got = ctx.read_f32(&y_buf, tokens * hidden);

    let diff = max_abs_diff(&got, expected.as_slice());
    assert!(diff < 1e-5, "rmsnorm GPU vs CPU diverged by {diff}");
}
#[test]
fn swiglu_gpu_matches_cpu() {
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping swiglu parity: {e}");
            return;
        }
    };
    const SWIGLU_WGSL: &str = include_str!("../src/shaders/wgsl/swiglu.wgsl");
    let kernel = ComputeKernel::new(&ctx, "swiglu", SWIGLU_WGSL);

    let n = 512usize;
    let mut seed = 13u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 * 10.0 - 5.0
    };
    let gate: Vec<f32> = (0..n).map(|_| rand()).collect();
    let up: Vec<f32> = (0..n).map(|_| rand()).collect();

    // CPU oracle: silu(gate) * up where silu(x) = x / (1 + exp(-x))
    let gate_t = Tensor::from_vec(gate.clone(), vec![n]);
    let up_t = Tensor::from_vec(up.clone(), vec![n]);
    let expected = gate_t.silu().mul(&up_t);

    // GPU
    let gate_buf = ctx.storage_init("gate", &gate);
    let up_buf = ctx.storage_init("up", &up);
    let out_buf = ctx.storage_zeros("out", n);
    let dims = ctx.uniform_of("dims", &[n as u32, 0, 0, 0]);
    let groups = (n as u32).div_ceil(256);
    kernel.dispatch(
        "swiglu",
        &[&gate_buf, &up_buf, &out_buf, &dims],
        [groups, 1, 1],
    );
    let got = ctx.read_f32(&out_buf, n);

    // exp() differs slightly between GPU and CPU libm; 1e-4 is the right bar for
    // a transcendental (the per-element error is uniformly ~1e-5).
    let diff = max_abs_diff(&got, expected.as_slice());
    assert!(diff < 1e-4, "swiglu GPU vs CPU diverged by {diff}");
}
/// Pack a `Dims { tokens, heads, head_dim, _pad }` uniform.
fn rope_dims(tokens: u32, heads: u32, head_dim: u32) -> [u32; 4] {
    [tokens, heads, head_dim, 0]
}

#[test]
fn rope_gpu_matches_cpu() {
    use arf_core::model::rope::Rope;
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping rope parity: {e}");
            return;
        }
    };
    const ROPE_WGSL: &str = include_str!("../src/shaders/wgsl/rope.wgsl");
    let kernel = ComputeKernel::new(&ctx, "rope", ROPE_WGSL);

    // Small test config: head_dim 4 (half = 2).
    let mut cfg = ModelConfig::llama_3_2_1b();
    cfg.head_dim = 4;
    cfg.rope_theta = 10000.0;
    cfg.rope_scaling = RopeScaling::None;

    let (tokens, heads, head_dim) = (3usize, 2usize, 4usize);

    // Build RoPE tables.
    let rope = Rope::new(&cfg, 64);

    // Random input x[tokens, heads, head_dim].
    let mut seed = 13u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let x: Vec<f32> = (0..tokens * heads * head_dim).map(|_| rand()).collect();

    // Positions per token.
    let positions = vec![0u32, 5u32, 12u32];

    // CPU oracle.
    let x_tensor = Tensor::from_vec(x.clone(), vec![tokens, heads, head_dim]);
    let expected = rope.apply(&x_tensor, &positions);

    // GPU: upload tables and dispatch.
    let x_buf = ctx.storage_init("x", &x);
    let cos_buf = ctx.storage_init("cos", rope.cos_table());
    let sin_buf = ctx.storage_init("sin", rope.sin_table());
    let pos_buf = ctx.storage_init("positions", &positions);
    let dims = ctx.uniform_of(
        "dims",
        &rope_dims(tokens as u32, heads as u32, head_dim as u32),
    );

    // In-place: one thread per element (early-exits for i >= half).
    let total = (tokens * heads * head_dim) as u32;
    let groups = total.div_ceil(256);
    kernel.dispatch(
        "rope",
        &[&x_buf, &cos_buf, &sin_buf, &pos_buf, &dims],
        [groups, 1, 1],
    );

    let got = ctx.read_f32(&x_buf, tokens * heads * head_dim);

    let diff = max_abs_diff(&got, expected.as_slice());
    assert!(diff < 1e-5, "rope GPU vs CPU diverged by {diff}");
}

/// Suspect probe (gemma-4 incoherent output): `final_logit_softcap`.
///
/// GPU (gpu/decode.rs:737-750, shaders/wgsl/softcap.wgsl) applies
/// `logit[i] = cap * tanh(logit[i] / cap)` in place over the [vocab] logits
/// BEFORE argmax, whenever `mc.final_logit_softcap = Some(30.0)`. The CPU
/// forward (llama.rs logits/logits_last) applies NO softcap. So the CPU oracle
/// here must apply the same tanh transform explicitly — this probe checks that
/// the GPU softcap *kernel* itself computes the documented function correctly
/// (a wrong cap / wrong formula reshapes the argmax distribution → degenerate
/// tokens). Bar 1e-4: tanh is a transcendental (GPU vs libm).
#[test]
fn softcap_gpu_matches_cpu() {
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping softcap parity: {e}");
            return;
        }
    };
    const SOFTCAP_WGSL: &str = include_str!("../src/shaders/wgsl/softcap.wgsl");
    let kernel = ComputeKernel::new(&ctx, "softcap", SOFTCAP_WGSL);

    // gemma-4 cap. Use vocab that straddles a 256-workgroup boundary (300) so
    // we cover the grid-edge `if (i < n)` guard too.
    let cap = 30.0f32;
    let vocab = 300usize;

    // Logits spanning ±~120 — well past the cap so the squashing is exercised
    // on extreme values (|logit| >> cap → output saturates near ±cap), plus
    // near-origin values where the map is ~linear.
    let mut seed = 17u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 * 240.0 - 120.0
    };
    let logits: Vec<f32> = (0..vocab).map(|_| rand()).collect();

    // CPU oracle: exactly the documented transform.
    let expected: Vec<f32> = logits.iter().map(|&l| cap * (l / cap).tanh()).collect();

    // GPU: in-place over `logits`. Binding order matches the WGSL @binding
    // order: [logits (read_write), dims (uniform)]. Dims = { n, cap, _pad, _pad }
    // with cap carried as its f32 bit pattern (same packing as decode.rs).
    let logits_buf = ctx.storage_init("logits", &logits);
    let dims = ctx.uniform_of("dims", &[vocab as u32, cap.to_bits(), 0, 0]);
    let groups = (vocab as u32).div_ceil(256);
    kernel.dispatch("softcap", &[&logits_buf, &dims], [groups, 1, 1]);
    let got = ctx.read_f32(&logits_buf, vocab);

    // Where do they first differ (for the report)?
    let first = got
        .iter()
        .zip(&expected)
        .position(|(g, e)| (g - e).abs() >= 1e-4);
    let diff = max_abs_diff(&got, &expected);
    eprintln!(
        "softcap parity: max_abs_diff={diff:e}, first-divergent-index={:?}, \
         sample in[0..4]={:?} gpu[0..4]={:?} cpu[0..4]={:?}",
        first,
        &logits[0..4],
        &got[0..4],
        &expected[0..4],
    );
    assert!(diff < 1e-4, "softcap GPU vs CPU diverged by {diff}");
}

#[test]
fn rope_position_zero_is_identity() {
    use arf_core::model::rope::Rope;
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping rope position-zero test: {e}");
            return;
        }
    };
    const ROPE_WGSL: &str = include_str!("../src/shaders/wgsl/rope.wgsl");
    let kernel = ComputeKernel::new(&ctx, "rope", ROPE_WGSL);

    let mut cfg = ModelConfig::llama_3_2_1b();
    cfg.head_dim = 4;
    cfg.rope_theta = 10000.0;
    cfg.rope_scaling = RopeScaling::None;

    let rope = Rope::new(&cfg, 8);
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let positions = vec![0u32];

    let x_buf = ctx.storage_init("x", &x);
    let cos_buf = ctx.storage_init("cos", rope.cos_table());
    let sin_buf = ctx.storage_init("sin", rope.sin_table());
    let pos_buf = ctx.storage_init("positions", &positions);
    let dims = ctx.uniform_of("dims", &rope_dims(1, 1, 4));

    let total = x.len() as u32;
    let groups = total.div_ceil(256);
    kernel.dispatch(
        "rope-pos0",
        &[&x_buf, &cos_buf, &sin_buf, &pos_buf, &dims],
        [groups, 1, 1],
    );

    let got = ctx.read_f32(&x_buf, x.len());
    for (i, (&a, &b)) in got.iter().zip(&x).enumerate() {
        assert!((a - b).abs() < 1e-5, "position 0 identity: lane {i}");
    }
}
/// kv_scatter: write k/v into the physical pool at the block-table slots, and
/// confirm a paged gather reads back exactly what the CPU write produced.
/// Single-slot (decode) and a multi-slot run both covered; bitwise-exact.
#[test]
fn kv_scatter_writes_to_pool() {
    use arf_core::cache::{slots_for, write_runs, PagedKvCache};
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping kv_scatter test: {e}");
            return;
        }
    };
    const KV_SCATTER_WGSL: &str = include_str!("../src/shaders/wgsl/kv_scatter.wgsl");
    let kernel = ComputeKernel::new(&ctx, "kv_scatter", KV_SCATTER_WGSL);

    let (kv_heads, head_dim) = (2usize, 4usize);
    let row_floats = kv_heads * head_dim; // 8
    let (num_blocks, block_size) = (2usize, 4usize);
    let pool_size = num_blocks * block_size * row_floats;

    let mut seed = 42u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };

    // Run the GPU scatter and the CPU write for `q_len` tokens, then compare the
    // gathered rows bit-for-bit.
    let mut run = |q_len: usize, label: &str| {
        let new_k: Vec<f32> = (0..q_len * row_floats).map(|_| rand()).collect();
        let new_v: Vec<f32> = new_k.iter().map(|x| x + 10.0).collect();
        let block_table = vec![1u32, 3];
        let slots = slots_for(&block_table, block_size, q_len);

        // CPU oracle.
        let mut cache = PagedKvCache::new(1, num_blocks, block_size, kv_heads, head_dim);
        cache.write(
            0,
            &new_k,
            &new_v,
            &write_runs(&block_table, block_size, 0, q_len),
        );
        let (cpu_k, cpu_v) = cache.gather(0, &slots);

        // GPU scatter into zeroed pools, then read back the same slots.
        let nk = ctx.storage_init("new_k", &new_k);
        let nv = ctx.storage_init("new_v", &new_v);
        let sb = ctx.storage_init("slots", &slots);
        let kp = ctx.storage_zeros("key_pool", pool_size);
        let vp = ctx.storage_zeros("value_pool", pool_size);
        let dims = ctx.uniform_of("dims", &[q_len as u32, row_floats as u32, 0, 0]);
        let groups = (q_len as u32 * row_floats as u32).div_ceil(256);
        kernel.dispatch(label, &[&nk, &nv, &sb, &kp, &vp, &dims], [groups, 1, 1]);
        let key_pool = ctx.read_f32(&kp, pool_size);
        let value_pool = ctx.read_f32(&vp, pool_size);

        // Gather the GPU pool at the same slots and compare to the CPU gather.
        for (j, &slot) in slots.iter().enumerate() {
            for col in 0..row_floats {
                let pool_idx = slot as usize * row_floats + col;
                let want = j * row_floats + col;
                assert_eq!(
                    key_pool[pool_idx].to_bits(),
                    cpu_k[want].to_bits(),
                    "{label} key slot {slot} col {col}"
                );
                assert_eq!(
                    value_pool[pool_idx].to_bits(),
                    cpu_v[want].to_bits(),
                    "{label} value slot {slot} col {col}"
                );
            }
        }
    };

    run(1, "kv_scatter_decode"); // single slot
    run(3, "kv_scatter_run"); // multi-slot run across blocks
}
/// Parity test: attention.wgsl matches the CPU oracle (attend_sequence math).
#[test]
fn attention_gpu_matches_cpu() {
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping attention parity: {e}");
            return;
        }
    };
    const ATTENTION_WGSL: &str = include_str!("../src/shaders/wgsl/attention.wgsl");
    let kernel = ComputeKernel::new(&ctx, "attention", ATTENTION_WGSL);

    // Small dims for test speed. `seqlen` is the context length (named to avoid
    // shadowing the GPU context handle `ctx`).
    let q_len = 2usize;
    let seqlen = 4usize;
    let num_heads = 4usize;
    let kv_heads = 2usize;
    let head_dim = 4usize;
    let past_len = 1u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 42u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };

    // q: [q_len, num_heads, head_dim]
    let q: Vec<f32> = (0..q_len * num_heads * head_dim).map(|_| rand()).collect();
    // k_all, v_all: [seqlen, kv_heads, head_dim] (already gathered by the caller)
    let k_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();
    let v_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();

    // CPU oracle: manually replicate attend_sequence for this small case.
    let mut cpu_out = vec![0.0f32; q_len * num_heads * head_dim];
    for head in 0..num_heads {
        let kv_head = head / (num_heads / kv_heads);

        // Gather Q for this head: [q_len, head_dim]
        let mut q_h = vec![0.0f32; q_len * head_dim];
        for r in 0..q_len {
            let src_base = (r * num_heads + head) * head_dim;
            let dst_base = r * head_dim;
            q_h[dst_base..dst_base + head_dim].copy_from_slice(&q[src_base..src_base + head_dim]);
        }

        // Gather K for this head: [seqlen, head_dim]
        let mut k_h = vec![0.0f32; seqlen * head_dim];
        for j in 0..seqlen {
            let src_base = (j * kv_heads + kv_head) * head_dim;
            let dst_base = j * head_dim;
            k_h[dst_base..dst_base + head_dim]
                .copy_from_slice(&k_all[src_base..src_base + head_dim]);
        }

        // Gather V for this head: [seqlen, head_dim]
        let mut v_h = vec![0.0f32; seqlen * head_dim];
        for j in 0..seqlen {
            let src_base = (j * kv_heads + kv_head) * head_dim;
            let dst_base = j * head_dim;
            v_h[dst_base..dst_base + head_dim]
                .copy_from_slice(&v_all[src_base..src_base + head_dim]);
        }

        // Compute scores = scale * Q·Kᵀ: [q_len, seqlen]
        let mut scores = vec![0.0f32; q_len * seqlen];
        for i in 0..q_len {
            for j in 0..seqlen {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_h[i * head_dim + d] * k_h[j * head_dim + d];
                }
                scores[i * seqlen + j] = dot * scale;
            }
        }

        // Causal softmax.
        for i in 0..q_len {
            let row = &mut scores[i * seqlen..(i + 1) * seqlen];
            let last = (past_len as usize) + i;
            let max = row[..=last].iter().copied().fold(-3.4e38f32, f32::max);
            let mut sum = 0.0f32;
            for (j, v) in row.iter_mut().enumerate() {
                if j <= last {
                    *v = (*v - max).exp();
                    sum += *v;
                } else {
                    *v = 0.0;
                }
            }
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }

        // out = softmax · V: [q_len, head_dim]
        let mut out_h = vec![0.0f32; q_len * head_dim];
        for i in 0..q_len {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seqlen {
                    acc += scores[i * seqlen + j] * v_h[j * head_dim + d];
                }
                out_h[i * head_dim + d] = acc;
            }
        }

        // Scatter into output.
        for r in 0..q_len {
            let src_base = r * head_dim;
            let dst_base = (r * num_heads + head) * head_dim;
            cpu_out[dst_base..dst_base + head_dim]
                .copy_from_slice(&out_h[src_base..src_base + head_dim]);
        }
    }

    // GPU dispatch. attention.wgsl now reads the paged KV pool by slot; feeding
    // it the already-contiguous k_all/v_all with IDENTITY slots (slot[j] == j)
    // reproduces the old contiguous read (pool row stride == kv_heads*head_dim).
    let q_buf = ctx.storage_init("q", &q);
    let k_buf = ctx.storage_init("key_pool", &k_all);
    let v_buf = ctx.storage_init("value_pool", &v_all);
    let out_buf = ctx.storage_zeros("out", q_len * num_heads * head_dim);
    let slots: Vec<u32> = (0..seqlen as u32).collect();
    let slots_buf = ctx.storage_init("slots", &slots);

    let dims_packed = [
        q_len as u32,
        seqlen as u32,
        num_heads as u32,
        kv_heads as u32,
        head_dim as u32,
        past_len,
        scale.to_bits(),
        (num_heads / kv_heads) as u32, // group
        0,                             // q_start
        0,
        0,
        0,
    ];
    let dims_buf = ctx.uniform_of("dims", &dims_packed);

    kernel.dispatch(
        "attention",
        &[&q_buf, &k_buf, &v_buf, &out_buf, &slots_buf, &dims_buf],
        [num_heads as u32, q_len as u32, 1],
    );

    let gpu_out = ctx.read_f32(&out_buf, q_len * num_heads * head_dim);

    let diff = max_abs_diff(&gpu_out, &cpu_out);
    assert!(diff < 1e-4, "attention GPU vs CPU diverged by {diff}");
}

/// Split-KV (flash-decoding) attention parity: the two-pass split + combine
/// (attention_splitk.wgsl → attention_splitk_combine.wgsl) must match the CPU
/// oracle. Decode case (q_len=1). Uses a seqlen not divisible by splitk so the
/// uneven-chunk + empty-split paths are exercised.
#[test]
fn attention_splitk_gpu_matches_cpu() {
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping split-KV parity: {e}");
            return;
        }
    };
    let split_k = ComputeKernel::new(
        &ctx,
        "attention_splitk",
        include_str!("../src/shaders/wgsl/attention_splitk.wgsl"),
    );
    let combine_k = ComputeKernel::new(
        &ctx,
        "attention_splitk_combine",
        include_str!("../src/shaders/wgsl/attention_splitk_combine.wgsl"),
    );

    let q_len = 1usize; // decode
    let seqlen = 10usize;
    let num_heads = 4usize;
    let kv_heads = 2usize;
    let head_dim = 8usize;
    let splitk = 4u32; // 10 keys / 4 = chunks 3,3,3,1 (uneven; last splits partial)
    let past_len = (seqlen - q_len) as u32; // attend all 10 keys (row 0 → last=9)
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 99u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let q: Vec<f32> = (0..q_len * num_heads * head_dim).map(|_| rand()).collect();
    let k_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();
    let v_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();

    // CPU oracle (q_len=1, full causal range 0..=last where last = past_len).
    let mut cpu_out = vec![0.0f32; q_len * num_heads * head_dim];
    for head in 0..num_heads {
        let kv_head = head / (num_heads / kv_heads);
        let last = past_len as usize; // row 0
        let mut scores = vec![0.0f32; seqlen];
        for (j, sc) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for dd in 0..head_dim {
                dot += q[head * head_dim + dd] * k_all[(j * kv_heads + kv_head) * head_dim + dd];
            }
            *sc = dot * scale;
        }
        let max = scores[..=last].iter().copied().fold(-3.4e38f32, f32::max);
        let mut sum = 0.0f32;
        for (j, sv) in scores.iter_mut().enumerate() {
            if j <= last {
                *sv = (*sv - max).exp();
                sum += *sv;
            } else {
                *sv = 0.0;
            }
        }
        for sv in scores.iter_mut() {
            *sv /= sum;
        }
        for dd in 0..head_dim {
            let mut acc = 0.0f32;
            for (j, &a) in scores.iter().enumerate() {
                acc += a * v_all[(j * kv_heads + kv_head) * head_dim + dd];
            }
            cpu_out[head * head_dim + dd] = acc;
        }
    }

    let q_buf = ctx.storage_init("q", &q);
    let k_buf = ctx.storage_init("key_pool", &k_all);
    let v_buf = ctx.storage_init("value_pool", &v_all);
    let out_buf = ctx.storage_zeros("out", q_len * num_heads * head_dim);
    let slots: Vec<u32> = (0..seqlen as u32).collect();
    let slots_buf = ctx.storage_init("slots", &slots);
    // partials: [head * splitk * (head_dim+2)]
    let partials = ctx.storage_zeros("partials", num_heads * splitk as usize * (head_dim + 2));
    let dims_packed = [
        q_len as u32,
        seqlen as u32,
        num_heads as u32,
        kv_heads as u32,
        head_dim as u32,
        past_len,
        scale.to_bits(),
        (num_heads / kv_heads) as u32,
        0,
        0,
        splitk,
        0,
    ];
    let dims_buf = ctx.uniform_of("dims", &dims_packed);

    split_k.dispatch(
        "attention_splitk",
        &[&q_buf, &k_buf, &v_buf, &partials, &slots_buf, &dims_buf],
        [num_heads as u32, splitk, 1],
    );
    combine_k.dispatch(
        "attention_splitk_combine",
        &[&partials, &out_buf, &dims_buf],
        [num_heads as u32, 1, 1],
    );
    let gpu_out = ctx.read_f32(&out_buf, q_len * num_heads * head_dim);
    let diff = max_abs_diff(&gpu_out, &cpu_out);
    assert!(
        diff < 1e-4,
        "split-KV attention GPU vs CPU diverged by {diff}"
    );
}

/// Streamed-softmax parity at ctx > TILE (2048): the kernel must process the
/// context in multiple tiles and still match the CPU oracle. This is the whole
/// point of the online-softmax rewrite (the old kernel capped ctx at 2048).
#[test]
fn attention_gpu_matches_cpu_long_ctx() {
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping long-ctx attention parity: {e}");
            return;
        }
    };
    const ATTENTION_WGSL: &str = include_str!("../src/shaders/wgsl/attention.wgsl");
    let kernel = ComputeKernel::new(&ctx, "attention", ATTENTION_WGSL);

    // Decode-shape (q_len 1) over a context spanning >1 tile (3000 > 2048 → 2
    // tiles). past_len = seqlen-1 so the single query attends the whole context.
    let q_len = 1usize;
    let seqlen = 3000usize;
    let num_heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = 4usize;
    let past_len = (seqlen - q_len) as u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 99u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let q: Vec<f32> = (0..q_len * num_heads * head_dim).map(|_| rand()).collect();
    let k_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();
    let v_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();

    // CPU oracle (attend_sequence math) for q_len == 1.
    let mut cpu_out = vec![0.0f32; q_len * num_heads * head_dim];
    for head in 0..num_heads {
        let kv_head = head / (num_heads / kv_heads);
        let last = past_len as usize; // q_len==1, row 0
        let mut scores = vec![0.0f32; seqlen];
        for j in 0..seqlen {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[head * head_dim + d] * k_all[(j * kv_heads + kv_head) * head_dim + d];
            }
            scores[j] = dot * scale;
        }
        let max = scores[..=last].iter().copied().fold(-3.4e38f32, f32::max);
        let mut sum = 0.0f32;
        for (j, v) in scores.iter_mut().enumerate() {
            if j <= last {
                *v = (*v - max).exp();
                sum += *v;
            } else {
                *v = 0.0;
            }
        }
        for v in scores.iter_mut() {
            *v /= sum;
        }
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for j in 0..seqlen {
                acc += scores[j] * v_all[(j * kv_heads + kv_head) * head_dim + d];
            }
            cpu_out[head * head_dim + d] = acc;
        }
    }

    let q_buf = ctx.storage_init("q", &q);
    let k_buf = ctx.storage_init("key_pool", &k_all);
    let v_buf = ctx.storage_init("value_pool", &v_all);
    let out_buf = ctx.storage_zeros("out", q_len * num_heads * head_dim);
    let slots: Vec<u32> = (0..seqlen as u32).collect();
    let slots_buf = ctx.storage_init("slots", &slots);
    let dims_packed = [
        q_len as u32,
        seqlen as u32,
        num_heads as u32,
        kv_heads as u32,
        head_dim as u32,
        past_len,
        scale.to_bits(),
        (num_heads / kv_heads) as u32,
        0,
        0,
        0,
        0,
    ];
    let dims_buf = ctx.uniform_of("dims", &dims_packed);
    kernel.dispatch(
        "attention",
        &[&q_buf, &k_buf, &v_buf, &out_buf, &slots_buf, &dims_buf],
        [num_heads as u32, q_len as u32, 1],
    );
    let gpu_out = ctx.read_f32(&out_buf, q_len * num_heads * head_dim);
    let diff = max_abs_diff(&gpu_out, &cpu_out);
    assert!(
        diff < 1e-4,
        "long-ctx attention GPU vs CPU diverged by {diff}"
    );
}

/// SUSPECT PROBE (gemma-4 global layer geometry): head_dim 512 > WG 256 and
/// kv_heads 1 with group 16. This is the only path that triggers the strided
/// output fan-out in attention.wgsl (DPL = ceil(512/256) = 2: each lane owns
/// dims `lane` and `lane+256`, accumulated in `acc[0]` / `acc[1]`) AND the
/// group=16 GQA mapping `kv_head = head / 16`. No existing attention test hits
/// head_dim > 256, so the DPL=2 branch (lines 92-93, 177-187, 195-199 of the
/// shader) is entirely uncovered. Decode shape (q_len 1), small context, full
/// causal (no window). CPU oracle = the inlined attend_sequence math.
#[test]
fn attention_gpu_matches_cpu_gemma_global_hd512() {
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping gemma-global attention parity: {e}");
            return;
        }
    };
    const ATTENTION_WGSL: &str = include_str!("../src/shaders/wgsl/attention.wgsl");
    let kernel = ComputeKernel::new(&ctx, "attention", ATTENTION_WGSL);

    // Gemma-4 GLOBAL layer geometry: head_dim 512 (> WG 256 → DPL 2), 16 query
    // heads sharing 1 kv head (group 16). Decode: a single query attends the
    // whole (small) context. seqlen kept tiny so the strided V-accumulation, not
    // tiling, is what's exercised (single tile, exact softmax order).
    let q_len = 1usize;
    let seqlen = 6usize;
    let num_heads = 16usize;
    let kv_heads = 1usize;
    let head_dim = 512usize;
    let past_len = (seqlen - q_len) as u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 1234u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let q: Vec<f32> = (0..q_len * num_heads * head_dim).map(|_| rand()).collect();
    let k_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();
    let v_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();

    // CPU oracle (attend_sequence math), q_len == 1. group = num_heads/kv_heads.
    let group = num_heads / kv_heads;
    let mut cpu_out = vec![0.0f32; q_len * num_heads * head_dim];
    for head in 0..num_heads {
        let kv_head = head / group;
        let last = past_len as usize; // q_len==1, row 0
        let mut scores = vec![0.0f32; seqlen];
        for j in 0..seqlen {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[head * head_dim + d] * k_all[(j * kv_heads + kv_head) * head_dim + d];
            }
            scores[j] = dot * scale;
        }
        let max = scores[..=last].iter().copied().fold(-3.4e38f32, f32::max);
        let mut sum = 0.0f32;
        for (j, v) in scores.iter_mut().enumerate() {
            if j <= last {
                *v = (*v - max).exp();
                sum += *v;
            } else {
                *v = 0.0;
            }
        }
        for v in scores.iter_mut() {
            *v /= sum;
        }
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for j in 0..seqlen {
                acc += scores[j] * v_all[(j * kv_heads + kv_head) * head_dim + d];
            }
            cpu_out[head * head_dim + d] = acc;
        }
    }

    // GPU: identity slots (slot[j] == j) → contiguous pool read, pool row stride
    // == kv_heads*head_dim, exactly the cache layout the decode path produces.
    let q_buf = ctx.storage_init("q", &q);
    let k_buf = ctx.storage_init("key_pool", &k_all);
    let v_buf = ctx.storage_init("value_pool", &v_all);
    let out_buf = ctx.storage_zeros("out", q_len * num_heads * head_dim);
    let slots: Vec<u32> = (0..seqlen as u32).collect();
    let slots_buf = ctx.storage_init("slots", &slots);
    let dims_packed = [
        q_len as u32,
        seqlen as u32,
        num_heads as u32,
        kv_heads as u32,
        head_dim as u32,
        past_len,
        scale.to_bits(),
        group as u32,
        0, // q_start
        0, // window (0 = global/causal)
        0,
        0,
    ];
    let dims_buf = ctx.uniform_of("dims", &dims_packed);
    kernel.dispatch(
        "attention",
        &[&q_buf, &k_buf, &v_buf, &out_buf, &slots_buf, &dims_buf],
        [num_heads as u32, q_len as u32, 1],
    );
    let gpu_out = ctx.read_f32(&out_buf, q_len * num_heads * head_dim);

    // Report WHERE the first divergence occurs (head, dim), and whether it's in
    // the DPL slot-0 (dim < 256) or slot-1 (dim >= 256, the strided path) region.
    let mut first: Option<(usize, usize, f32, f32)> = None;
    for head in 0..num_heads {
        for d in 0..head_dim {
            let i = head * head_dim + d;
            if (gpu_out[i] - cpu_out[i]).abs() >= 1e-4 && first.is_none() {
                first = Some((head, d, gpu_out[i], cpu_out[i]));
            }
        }
    }
    let diff = max_abs_diff(&gpu_out, &cpu_out);
    if let Some((h, d, g, c)) = first {
        let region = if d < 256 {
            "DPL slot 0 (dim<256)"
        } else {
            "DPL slot 1 (dim>=256, strided)"
        };
        eprintln!(
            "FIRST DIVERGENCE head={h} dim={d} [{region}] gpu={g} cpu={c} (group={group}, kv_head={})",
            h / group
        );
    } else {
        eprintln!("no divergence >= 1e-4 anywhere; max_abs_diff={diff}");
    }
    assert!(
        diff < 1e-4,
        "gemma-global (hd=512, kv_heads=1, group=16) attention GPU vs CPU diverged by {diff}"
    );
}

/// Parity: the TurboQuant scatter (kv_scatter_tq.wgsl — normalize + Hadamard +
/// quantize + bit-pack on write) reconstructs the original K/V within the bit's
/// MSE bound, and its f32 norms match the true norms. Matches PagedKvCache::write
/// under Tq (TqKvBlock::set_vector); the GPU pool stores f32 norms (vs the CPU
/// block's bf16 — a benign per-vector divergence far inside the Tq error).
#[test]
fn kv_scatter_tq_gpu_reconstructs_within_bound() {
    use arf_core::tensor::{Codebook, Rotation, TqKvBlock};
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping kv_scatter_tq parity: {e}");
            return;
        }
    };
    const SRC: &str = include_str!("../src/shaders/wgsl/kv_scatter_tq.wgsl");
    let kernel = ComputeKernel::new(&ctx, "kv_scatter_tq", SRC);

    let (kv_heads, head_dim) = (2usize, 64usize);
    let seqlen = 5usize; // scatter all positions at once (q_len == seqlen)
    let n = seqlen * kv_heads;

    // Gaussian (Box-Muller): normalized, these match the codebook's target
    // distribution, so the round-trip meets the bit's MSE bound.
    let mut seed = 555u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 24) as f32 // [0, 1)
    };
    let mut gauss = || {
        let u1 = rand().max(1e-7);
        let u2 = rand();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    };
    let k_all: Vec<f32> = (0..n * head_dim).map(|_| gauss()).collect();
    let v_all: Vec<f32> = (0..n * head_dim).map(|_| gauss()).collect();

    for bits in [2u8, 3, 4] {
        let cb = Codebook::new(bits, head_dim);
        let rot = Rotation::default_for(head_dim, 0);
        let levels: Vec<f32> = cb.levels().to_vec();
        // CPU oracle: the same write PagedKvCache does under Tq.
        let cpu_block = TqKvBlock::from_f32_vectors(&k_all, n, head_dim, bits, &cb, &rot);
        let cpu_recon = cpu_block.to_f32(&cb, &rot);
        let words = (n * head_dim * bits as usize).div_ceil(32);

        let nk = ctx.storage_init("new_k", &k_all);
        let nv = ctx.storage_init("new_v", &v_all);
        let slots: Vec<u32> = (0..seqlen as u32).collect();
        let sb = ctx.storage_init("slots", &slots);
        let kc = ctx.storage_init("key_codes", &vec![0u32; words]);
        let vc = ctx.storage_init("value_codes", &vec![0u32; words]);
        let kn = ctx.storage_init("key_norms", &vec![0.0f32; n]);
        let vn = ctx.storage_init("value_norms", &vec![0.0f32; n]);
        let lv = ctx.storage_init("levels", &levels);
        let dims = ctx.uniform_of(
            "dims",
            &[
                seqlen as u32,
                kv_heads as u32,
                head_dim as u32,
                0,
                bits as u32,
                0,
                0,
                0,
            ],
        );
        let groups = (n as u32).div_ceil(64);
        kernel.dispatch(
            "kv_scatter_tq",
            &[&nk, &nv, &sb, &kc, &vc, &kn, &vn, &lv, &dims],
            [groups, 1, 1],
        );
        let kcodes = ctx.read_u32(&kc, words);
        let knorms = ctx.read_f32(&kn, n);

        // Reconstruct the GPU-packed keys (unpack code, norm·level, inverse-rotate)
        // and compare to the original within the bit's MSE bound.
        let unpack = |codes: &[u32], vidx: usize, coord: usize| -> usize {
            let bitpos = (vidx * head_dim + coord) * bits as usize;
            let word = bitpos / 32;
            let off = bitpos % 32;
            let mask = (1u32 << bits) - 1;
            let mut val = codes[word] >> off;
            if off + bits as usize > 32 {
                val |= codes[word + 1] << (32 - off);
            }
            (val & mask) as usize
        };
        let mut err_orig = 0.0f64; // GPU recon vs original  (quality)
        let mut sig = 0.0f64;
        let mut err_cpu = 0.0f64; // GPU recon vs CPU oracle (parity)
        for vidx in 0..n {
            let mut y = vec![0.0f32; head_dim];
            for (i, yi) in y.iter_mut().enumerate() {
                *yi = knorms[vidx] * levels[unpack(&kcodes, vidx, i)];
            }
            rot.apply_transpose(&mut y);
            for i in 0..head_dim {
                let o = k_all[vidx * head_dim + i] as f64;
                err_orig += (y[i] as f64 - o).powi(2);
                sig += o * o;
                err_cpu += (y[i] - cpu_recon[vidx * head_dim + i]).powi(2) as f64;
            }
            // The stored f32 norm matches the true norm.
            let true_norm: f32 = (0..head_dim)
                .map(|i| k_all[vidx * head_dim + i].powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(
                (knorms[vidx] - true_norm).abs() < 1e-3,
                "{bits}-bit norm mismatch: {} vs {true_norm}",
                knorms[vidx]
            );
        }
        let bound = match bits {
            4 => 0.009,
            3 => 0.036,
            2 => 0.117,
            _ => unreachable!(),
        };
        // Quality: reconstruction within the bit's MSE bound.
        assert!(
            err_orig / sig <= bound * 2.0,
            "{bits}-bit scatter reconstruction rel MSE {} over bound {bound}",
            err_orig / sig
        );
        // Parity: GPU scatter matches the CPU write oracle (only bf16-vs-f32 norm
        // rounding + rare borderline code flips differ).
        assert!(
            err_cpu / sig < 5e-3,
            "{bits}-bit GPU scatter diverged from CPU oracle by rel {}",
            err_cpu / sig
        );
        let _ = (vc, vn, v_all.len()); // value path exercises the same code; key suffices
    }
}

/// Parity: the TurboQuant fused attention (attention_tq.wgsl — inline dequant +
/// rotate-q + Rᵀ-output) matches the CPU oracle, which reconstructs the packed
/// K/V to f32 (TqKvBlock::to_f32) and runs the same attend_sequence math. This is
/// the GPU half of the rotation trick: keys/values stay packed-and-rotated; only
/// the query and the output are rotated in-kernel.
#[test]
fn attention_tq_gpu_matches_cpu() {
    use arf_core::tensor::{Codebook, Rotation, TqKvBlock};
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping attention_tq parity: {e}");
            return;
        }
    };
    const SRC: &str = include_str!("../src/shaders/wgsl/attention_tq.wgsl");
    let kernel = ComputeKernel::new(&ctx, "attention_tq", SRC);

    // head_dim 64: power of two (Hadamard) and head_dim*bits % 32 == 0 for all
    // bit widths (vectors stay u32-word-aligned, like real models).
    let q_len = 2usize;
    let seqlen = 5usize;
    let num_heads = 4usize;
    let kv_heads = 2usize;
    let head_dim = 64usize;
    let past_len = (seqlen - q_len) as u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 1234u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let q: Vec<f32> = (0..q_len * num_heads * head_dim).map(|_| rand()).collect();
    let k_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();
    let v_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();

    for bits in [2u8, 3, 4] {
        let cb = Codebook::new(bits, head_dim);
        let rot = Rotation::default_for(head_dim, 0); // Hadamard (head_dim is pow2)
                                                      // Pack as the cache does: vector index = slot*kv_heads + kv_head, identity
                                                      // slots so slot j == j. n_vectors == seqlen*kv_heads.
        let n_vectors = seqlen * kv_heads;
        let kblock = TqKvBlock::from_f32_vectors(&k_all, n_vectors, head_dim, bits, &cb, &rot);
        let vblock = TqKvBlock::from_f32_vectors(&v_all, n_vectors, head_dim, bits, &cb, &rot);

        // CPU oracle: reconstruct K/V to f32, then attend_sequence math.
        let k_rec = kblock.to_f32(&cb, &rot); // [n_vectors * head_dim] = [seqlen,kv,hd]
        let v_rec = vblock.to_f32(&cb, &rot);
        let mut cpu_out = vec![0.0f32; q_len * num_heads * head_dim];
        for head in 0..num_heads {
            let kv_head = head / (num_heads / kv_heads);
            for i in 0..q_len {
                let last = past_len as usize + i;
                let mut scores = vec![0.0f32; seqlen];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for dd in 0..head_dim {
                        let qv = q[(i * num_heads + head) * head_dim + dd];
                        let kv = k_rec[(j * kv_heads + kv_head) * head_dim + dd];
                        dot += qv * kv;
                    }
                    *sc = dot * scale;
                }
                let max = scores[..=last].iter().copied().fold(-3.4e38f32, f32::max);
                let mut sum = 0.0f32;
                for (j, sv) in scores.iter_mut().enumerate() {
                    if j <= last {
                        *sv = (*sv - max).exp();
                        sum += *sv;
                    } else {
                        *sv = 0.0;
                    }
                }
                for sv in scores.iter_mut() {
                    *sv /= sum;
                }
                for dd in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (j, &a) in scores.iter().enumerate() {
                        acc += a * v_rec[(j * kv_heads + kv_head) * head_dim + dd];
                    }
                    cpu_out[(i * num_heads + head) * head_dim + dd] = acc;
                }
            }
        }

        // GPU: upload packed codes + f32 (bf16-rounded) norms + codebook levels.
        let knorms: Vec<f32> = (0..n_vectors).map(|v| kblock.norm_at(v)).collect();
        let vnorms: Vec<f32> = (0..n_vectors).map(|v| vblock.norm_at(v)).collect();
        let levels: Vec<f32> = cb.levels().to_vec();

        let q_buf = ctx.storage_init("q", &q);
        let kc = ctx.storage_init("key_codes", kblock.codes());
        let vc = ctx.storage_init("value_codes", vblock.codes());
        let out_buf = ctx.storage_zeros("out", q_len * num_heads * head_dim);
        let slots: Vec<u32> = (0..seqlen as u32).collect();
        let slots_buf = ctx.storage_init("slots", &slots);
        let kn = ctx.storage_init("key_norms", &knorms);
        let vn = ctx.storage_init("value_norms", &vnorms);
        let lv = ctx.storage_init("levels", &levels);
        let dims_packed = [
            q_len as u32,
            seqlen as u32,
            num_heads as u32,
            kv_heads as u32,
            head_dim as u32,
            past_len,
            scale.to_bits(),
            (num_heads / kv_heads) as u32,
            0,
            0,
            bits as u32,
            0,
        ];
        let dims_buf = ctx.uniform_of("dims", &dims_packed);
        kernel.dispatch(
            "attention_tq",
            &[
                &q_buf, &kc, &vc, &out_buf, &slots_buf, &dims_buf, &kn, &vn, &lv,
            ],
            [num_heads as u32, q_len as u32, 1],
        );
        let gpu_out = ctx.read_f32(&out_buf, q_len * num_heads * head_dim);
        let diff = max_abs_diff(&gpu_out, &cpu_out);
        assert!(
            diff < 2e-3,
            "{bits}-bit attention_tq GPU vs CPU oracle diverged by {diff}"
        );
    }
}

/// Parity: the PARALLEL-Hadamard TurboQuant attention (attention_tq_par.wgsl —
/// the rotation butterfly spread across the workgroup instead of run on lane 0)
/// must match the CPU oracle to the same bound as the serial kernel. Same setup
/// as `attention_tq_gpu_matches_cpu`, only the shader source differs.
#[test]
fn attention_tq_par_gpu_matches_cpu() {
    use arf_core::tensor::{Codebook, Rotation, TqKvBlock};
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping attention_tq_par parity: {e}");
            return;
        }
    };
    const SRC: &str = include_str!("../src/shaders/wgsl/attention_tq_par.wgsl");
    let kernel = ComputeKernel::new(&ctx, "attention_tq_par", SRC);

    let q_len = 2usize;
    let seqlen = 5usize;
    let num_heads = 4usize;
    let kv_heads = 2usize;
    let head_dim = 64usize;
    let past_len = (seqlen - q_len) as u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut seed = 1234u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let q: Vec<f32> = (0..q_len * num_heads * head_dim).map(|_| rand()).collect();
    let k_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();
    let v_all: Vec<f32> = (0..seqlen * kv_heads * head_dim).map(|_| rand()).collect();

    for bits in [2u8, 3, 4] {
        let cb = Codebook::new(bits, head_dim);
        let rot = Rotation::default_for(head_dim, 0);
        let n_vectors = seqlen * kv_heads;
        let kblock = TqKvBlock::from_f32_vectors(&k_all, n_vectors, head_dim, bits, &cb, &rot);
        let vblock = TqKvBlock::from_f32_vectors(&v_all, n_vectors, head_dim, bits, &cb, &rot);
        let k_rec = kblock.to_f32(&cb, &rot);
        let v_rec = vblock.to_f32(&cb, &rot);
        let mut cpu_out = vec![0.0f32; q_len * num_heads * head_dim];
        for head in 0..num_heads {
            let kv_head = head / (num_heads / kv_heads);
            for i in 0..q_len {
                let last = past_len as usize + i;
                let mut scores = vec![0.0f32; seqlen];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for dd in 0..head_dim {
                        let qv = q[(i * num_heads + head) * head_dim + dd];
                        let kv = k_rec[(j * kv_heads + kv_head) * head_dim + dd];
                        dot += qv * kv;
                    }
                    *sc = dot * scale;
                }
                let max = scores[..=last].iter().copied().fold(-3.4e38f32, f32::max);
                let mut sum = 0.0f32;
                for (j, sv) in scores.iter_mut().enumerate() {
                    if j <= last {
                        *sv = (*sv - max).exp();
                        sum += *sv;
                    } else {
                        *sv = 0.0;
                    }
                }
                for sv in scores.iter_mut() {
                    *sv /= sum;
                }
                for dd in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (j, &a) in scores.iter().enumerate() {
                        acc += a * v_rec[(j * kv_heads + kv_head) * head_dim + dd];
                    }
                    cpu_out[(i * num_heads + head) * head_dim + dd] = acc;
                }
            }
        }

        let knorms: Vec<f32> = (0..n_vectors).map(|v| kblock.norm_at(v)).collect();
        let vnorms: Vec<f32> = (0..n_vectors).map(|v| vblock.norm_at(v)).collect();
        let levels: Vec<f32> = cb.levels().to_vec();
        let q_buf = ctx.storage_init("q", &q);
        let kc = ctx.storage_init("key_codes", kblock.codes());
        let vc = ctx.storage_init("value_codes", vblock.codes());
        let out_buf = ctx.storage_zeros("out", q_len * num_heads * head_dim);
        let slots: Vec<u32> = (0..seqlen as u32).collect();
        let slots_buf = ctx.storage_init("slots", &slots);
        let kn = ctx.storage_init("key_norms", &knorms);
        let vn = ctx.storage_init("value_norms", &vnorms);
        let lv = ctx.storage_init("levels", &levels);
        let dims_packed = [
            q_len as u32,
            seqlen as u32,
            num_heads as u32,
            kv_heads as u32,
            head_dim as u32,
            past_len,
            scale.to_bits(),
            (num_heads / kv_heads) as u32,
            0,
            0,
            bits as u32,
            0,
        ];
        let dims_buf = ctx.uniform_of("dims", &dims_packed);
        kernel.dispatch(
            "attention_tq_par",
            &[
                &q_buf, &kc, &vc, &out_buf, &slots_buf, &dims_buf, &kn, &vn, &lv,
            ],
            [num_heads as u32, q_len as u32, 1],
        );
        let gpu_out = ctx.read_f32(&out_buf, q_len * num_heads * head_dim);
        let diff = max_abs_diff(&gpu_out, &cpu_out);
        assert!(
            diff < 2e-3,
            "{bits}-bit attention_tq_par GPU vs CPU oracle diverged by {diff}"
        );
    }
}

/// END-TO-END: the full GPU decode pipeline wired through the
/// TurboQuant-packed KV pool (`build_gpu_kv_quant` → `kv_scatter_tq` +
/// `attention_tq` dispatched in place of the f32 pair) produces logits that
/// TRACK the exact f32 GPU pipeline, and the drift shrinks monotonically as the
/// bit-width grows. This proves the pool surgery + dispatch branching are correct:
/// same weights, same prompt, only the KV path differs (f32 vs packed). The
/// kernel-level parity tests (`attention_tq_gpu_matches_cpu`,
/// `kv_scatter_tq_gpu_reconstructs_within_bound`) already pin each kernel to its
/// CPU oracle; this pins their composition inside the real engine path.
#[test]
fn gpu_tq_forward_logits_tracks_f32() {
    use arf_core::config::{KvQuant, Quant};
    use arf_gpu::weights::{build_gpu, build_gpu_kv_quant};

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU Tq end-to-end parity: {e}");
            return;
        }
    };
    eprintln!("running GPU Tq end-to-end on {}", gctx.info());

    // head_dim 64: power-of-two (Hadamard) and 64·bits % 32 == 0 for bits 2/3/4,
    // so the packed pool vectors are u32-word-aligned (race-free scatter), exactly
    // as for real models (head_dim ∈ {64,128,256}).
    let cfg = cfg_q4k();
    let w = weights(&cfg);
    let max = cfg.max_position_embeddings;

    // Prefill past the first KV block (block_size 16) so the streamed-softmax /
    // packed-gather loop spans multiple blocks: a 20-token prompt.
    let tokens: Vec<u32> = (0..20u32)
        .map(|i| (i * 7 + 3) % cfg.vocab_size as u32)
        .collect();

    let f32_gpu = build_gpu(&cfg, &w, max, &gctx).unwrap();
    let f32_logits = f32_gpu.forward_logits(&tokens);

    let rel_l2 = |a: &[f32], b: &[f32]| -> f32 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            num += ((x - y) as f64).powi(2);
            den += (*y as f64).powi(2);
        }
        (num / den.max(1e-12)).sqrt() as f32
    };

    let mut prev_mad = f32::INFINITY;
    for bits in [2u8, 3, 4] {
        let tq_gpu = build_gpu_kv_quant(
            &cfg,
            &w,
            max,
            &gctx,
            Quant::None,
            KvQuant::Tq { bits, qjl: false },
        )
        .unwrap();
        let tq_logits = tq_gpu.forward_logits(&tokens);
        assert_eq!(tq_logits.len(), f32_logits.len());
        assert!(
            tq_logits.iter().all(|x| x.is_finite()),
            "{bits}-bit Tq logits contained non-finite values"
        );
        let drift = rel_l2(&tq_logits, &f32_logits);
        let mad = max_abs_diff(&tq_logits, &f32_logits);
        eprintln!("  {bits}-bit Tq vs f32 rel-L2 = {drift:.6}  max|Δ| = {mad:.6}");
        // The packed path tracks f32 closely (the real signal is the trend).
        assert!(
            drift < 0.05,
            "{bits}-bit Tq logits drifted too far from f32: rel-L2 {drift}"
        );
        if bits == 2 {
            // Proof the Tq kernels actually ran (not a silent fall-through to the
            // f32 pool): 2-bit KV quant MUST perturb the logits measurably.
            assert!(
                mad > 1e-3,
                "2-bit Tq produced ~identical logits to f32 — packed KV path likely inactive (max|Δ| {mad})"
            );
        }
        // More bits ⇒ closer to f32 (allow a hair of noise on the tie at high bits).
        assert!(
            mad <= prev_mad + 1e-3,
            "Tq error did not shrink with more bits: {bits}-bit max|Δ| {mad} > prev {prev_mad}"
        );
        prev_mad = mad;
    }
}

/// Greedy argmax (GPU) must match CPU oracle exactly (integer tie-break).
#[test]
fn sample_greedy_gpu_matches_cpu() {
    use arf_core::sampling::argmax;
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping sample parity: {e}");
            return;
        }
    };
    const SAMPLE_WGSL: &str = include_str!("../src/shaders/wgsl/sample.wgsl");
    let kernel = ComputeKernel::new(&ctx, "sample", SAMPLE_WGSL);

    // Test case 1: simple argmax.
    let vocab: usize = 256;
    let mut seed = 13u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 2.0
    };
    let logits: Vec<f32> = (0..vocab).map(|_| rand()).collect();

    // CPU oracle.
    let expected = argmax(&logits);

    // GPU: single workgroup reduction.
    let logits_buf = ctx.storage_init("logits", &logits);
    let out_buf = ctx.storage_zeros_u32("out", 1);
    let dims = ctx.uniform_of("dims", &[vocab as u32, 0, 0, 0]);
    kernel.dispatch("sample", &[&logits_buf, &out_buf, &dims], [1, 1, 1]);
    let got = ctx.read_u32(&out_buf, 1)[0];

    assert_eq!(
        got, expected,
        "sample GPU vs CPU diverged: got {}, expected {}",
        got, expected
    );

    // Test case 2: deliberate tie (indices 10 and 15 have same max value).
    let mut logits_tie = logits.clone();
    logits_tie[10] = 5.0;
    logits_tie[15] = 5.0;
    let expected_tie = argmax(&logits_tie);
    assert_eq!(
        expected_tie, 10,
        "CPU oracle tie-break selects lowest index"
    );

    let logits_tie_buf = ctx.storage_init("logits_tie", &logits_tie);
    let out_tie_buf = ctx.storage_zeros_u32("out_tie", 1);
    kernel.dispatch(
        "sample_tie",
        &[&logits_tie_buf, &out_tie_buf, &dims],
        [1, 1, 1],
    );
    let got_tie = ctx.read_u32(&out_tie_buf, 1)[0];
    assert_eq!(
        got_tie, expected_tie,
        "sample GPU tie-break matches CPU: got {}, expected {}",
        got_tie, expected_tie
    );
}

/// THE END-TO-END PROOF: the full GPU decode pipeline
/// (GpuModel::forward_logits) matches the CPU Llama's last-token logits.
#[test]
fn gpu_decode_step_matches_cpu_logits() {
    use arf_gpu::weights::build_gpu;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU decode-step parity: {e}");
            return;
        }
    };
    eprintln!("running full GPU decode on {}", gctx.info());

    let cfg = cfg();
    let w = weights(&cfg);
    let cpu = build_on(&cfg, &w, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &gctx).unwrap();

    let tokens = [3u32, 1, 4, 1, 5];
    let cpu_logits = last_logits(&cpu, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);

    let diff = max_abs_diff(&gpu_logits, &cpu_logits);
    assert!(
        diff < 1e-3,
        "full GPU decode vs CPU logits diverged by {diff}"
    );
}

/// GPU greedy generation produces the same token sequence as CPU greedy
/// generation — the engine generates on the GPU, matching the oracle.
#[test]
fn gpu_generate_matches_cpu_greedy() {
    use arf_core::config::EngineConfig;
    use arf_core::engine::LlmEngine;
    use arf_core::sampling::SamplingParams;
    use arf_gpu::weights::build_gpu;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU generate parity: {e}");
            return;
        }
    };

    let cfg = cfg();
    let w = weights(&cfg);
    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &gctx).unwrap();
    let cpu_model = build_on(&cfg, &w, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let mut engine = LlmEngine::new(cpu_model, EngineConfig::default()).unwrap();

    let prompt = vec![3u32, 1, 4];
    let cpu_out = engine
        .generate(prompt.clone(), SamplingParams::greedy(6))
        .unwrap();
    let gpu_out = gpu.generate(&prompt, 6);

    // Compare the shared greedy prefix (CPU may stop on a stop token).
    let n = cpu_out.len().min(gpu_out.len());
    assert!(n > 0, "no tokens generated");
    assert_eq!(
        &gpu_out[..n],
        &cpu_out[..n],
        "GPU greedy generation diverged from CPU"
    );
}

/// GEMMA END-TO-END PROOF: the full GPU decode pipeline (GpuModel::forward_logits
/// — four-norm blocks, dual RoPE, sliding-window local layers, √hidden embedding
/// scale) matches the CPU Llama Gemma oracle. The 6-token prompt exceeds the
/// window (3), so the local layers genuinely drop the oldest keys on both sides.
#[test]
fn gpu_gemma_decode_matches_cpu_logits() {
    use arf_gpu::weights::build_gpu;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping Gemma GPU parity: {e}");
            return;
        }
    };
    eprintln!("running Gemma GPU decode on {}", gctx.info());

    let cfg = cfg_gemma();
    let w = gemma_weights(&cfg);
    let cpu = build_on(&cfg, &w, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &gctx).unwrap();

    let tokens = [7u32, 2, 9, 1, 4, 3];
    let cpu_logits = last_logits(&cpu, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);

    let diff = max_abs_diff(&gpu_logits, &cpu_logits);
    assert!(diff < 1e-3, "Gemma GPU vs CPU logits diverged by {diff}");
}

/// Gemma GPU greedy generation matches CPU greedy: drives the decode loop
/// (on-GPU token feedback, `token = None`) through the Gemma four-norm block past
/// the sliding window, so steady-state decode is exercised, not just prefill.
#[test]
fn gpu_gemma_generate_matches_cpu_greedy() {
    use arf_core::config::EngineConfig;
    use arf_core::engine::LlmEngine;
    use arf_core::sampling::SamplingParams;
    use arf_gpu::weights::build_gpu;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping Gemma GPU generate parity: {e}");
            return;
        }
    };

    let cfg = cfg_gemma();
    let w = gemma_weights(&cfg);
    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &gctx).unwrap();
    let cpu_model = build_on(&cfg, &w, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let mut engine = LlmEngine::new(cpu_model, EngineConfig::default()).unwrap();

    let prompt = vec![7u32, 2, 9, 1];
    let cpu_out = engine
        .generate(prompt.clone(), SamplingParams::greedy(8))
        .unwrap();
    let gpu_out = gpu.generate(&prompt, 8);

    let n = cpu_out.len().min(gpu_out.len());
    assert!(n > 0, "no tokens generated");
    assert_eq!(
        gpu_out[..n],
        cpu_out[..n],
        "Gemma GPU greedy diverged from CPU greedy"
    );
}

/// GPU speculative decode (P19) must produce BIT-IDENTICAL output to GPU greedy —
/// it is a latency optimization, not an accuracy change. Every accepted draft
/// token must equal what plain greedy would have produced; the n-gram drafter only
/// changes HOW MANY forwards run, never WHAT is generated. Also sanity-checks the
/// reported stats (>= 1 forward, drafted >= accepted).
#[test]
fn gpu_speculative_matches_greedy_exactly() {
    use arf_core::config::Quant;
    use arf_gpu::weights::build_gpu_quant;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU spec-decode parity: {e}");
            return;
        }
    };

    let prompt = vec![3u32, 1, 4, 1, 5];
    let n = 24usize;
    // Verify across all decode precisions — Q4 only works because the lightweight
    // verify uses the m==1 decode path (Q4 has no batched GEMM). cfg_q4 (dims % 32)
    // is required for Q4; it's valid for int8/bf16 too, so use it for all three.
    for quant in [Quant::None, Quant::Int8, Quant::Q4] {
        let cfg = cfg_q4();
        let gpu = build_gpu_quant(&cfg, &weights(&cfg), 128, &gctx, quant).unwrap();
        let greedy = gpu.generate(&prompt, n);
        let (spec, fwd, drafted, accepted) = gpu.generate_speculative(&prompt, n, 4, 3);
        assert_eq!(
            spec, greedy,
            "GPU spec-decode ({quant:?}) diverged from greedy (must be bit-identical)"
        );
        assert!(fwd >= 1, "{quant:?}: no forward passes");
        assert!(drafted >= accepted, "{quant:?}: accepted > drafted");
    }
}

/// Decode PAST the ctx_len=65 boundary, where the idx-arena `pos_buf` offset used
/// to jump (the P15 reorder pins it). The other parity tests stop well before 64
/// tokens, so a cached `rope_qk` bind group capturing a drifting `pos_buf` would
/// pass them yet silently corrupt a long generation. This generates ~80 tokens and
/// requires GPU == CPU greedy across the whole run — the guard for the P15 cache.
#[test]
fn gpu_generate_matches_cpu_greedy_past_64() {
    use arf_gpu::weights::build_gpu;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping long-generate parity: {e}");
            return;
        }
    };

    let cfg = cfg();
    // Room for prompt + 80 tokens (well past the ctx_len=65 boundary).
    let max_pos = 256usize;
    let gpu = build_gpu(&cfg, &weights(&cfg), max_pos, &gctx).unwrap();
    let cpu = build_on(&cfg, &weights(&cfg), max_pos, &Device::Cpu).unwrap();

    let prompt = vec![3u32, 1, 4];
    let n = 80;
    let gpu_out = gpu.generate(&prompt, n);

    // CPU reference: plain greedy, one token per forward, over a fresh paged cache.
    let cpu_out = cpu_greedy(&cpu, &prompt, n);

    assert_eq!(gpu_out.len(), n);
    assert_eq!(
        gpu_out, cpu_out,
        "GPU greedy diverged from CPU past the ctx_len=65 boundary (P15 pos_buf reorder?)"
    );
}

/// Plain greedy reference (one token per forward) for the long-generate test.
fn cpu_greedy(model: &Llama, prompt: &[u32], n: usize) -> Vec<u32> {
    let bs = 16usize;
    let nb = (prompt.len() + n).div_ceil(bs);
    let mut cache = PagedKvCache::new(
        model.num_layers(),
        nb,
        bs,
        model.config().num_kv_heads,
        model.config().head_dim,
    );
    let bt: Vec<u32> = (0..nb as u32).collect();
    let run = |c: &mut PagedKvCache, toks: &[u32], past: usize| -> u32 {
        let ql = toks.len();
        let ctx = past + ql;
        let b = ForwardBatch {
            positions: (past..ctx).map(|p| p as u32).collect(),
            seqs: vec![SeqAttn {
                q_start: 0,
                q_len: ql,
                past_len: past,
                slots: slots_for(&bt, bs, ctx),
                write_runs: write_runs(&bt, bs, past, ql),
                image_spans: Vec::new(),
                stream_id: None,
            }],
            image_embeds: None,
            mrope_positions: None,
        };
        let h = model.forward(toks, &b, c);
        arf_core::sampling::argmax(&model.logits_last(&h, &b).into_vec())
    };
    let mut out = Vec::with_capacity(n);
    let mut t = run(&mut cache, prompt, 0);
    for pos in prompt.len()..prompt.len() + n {
        out.push(t);
        t = run(&mut cache, &[t], pos);
    }
    out
}

/// Batched decode: generate_batch must agree with single-stream generate.
/// Running N copies of one prompt through the batched trunk must yield N
/// identical sequences, each matching what `generate` produces alone — proving
/// the continuous-batching path is correct, not just fast. Two distinct prompts
/// in the same batch must each match their own single-stream run, confirming the
/// sequences don't cross-contaminate (KV block ranges are disjoint).
#[test]
fn gpu_generate_batch_matches_single_stream() {
    use arf_gpu::weights::build_gpu;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU batch parity: {e}");
            return;
        }
    };

    let cfg = cfg();
    let w = weights(&cfg);
    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &gctx).unwrap();

    let p0 = vec![3u32, 1, 4];
    let p1 = vec![5u32, 9, 2, 6];
    let single0 = gpu.generate(&p0, 6);
    let single1 = gpu.generate(&p1, 6);

    // Two distinct prompts in one batch, plus a duplicate of p0 to check that
    // identical inputs in different KV ranges stay identical.
    let batch = gpu.generate_batch(&[p0.clone(), p1.clone(), p0.clone()], 6);
    assert_eq!(batch.len(), 3);
    assert_eq!(
        batch[0], single0,
        "batch seq 0 diverged from single-stream p0"
    );
    assert_eq!(
        batch[1], single1,
        "batch seq 1 diverged from single-stream p1"
    );
    assert_eq!(batch[2], single0, "batch seq 2 (dup p0) diverged");
}

/// The batched GEMV (matmul_vec_batch.wgsl) must match single-stream across the
/// batch-size cap (MATMUL_VEC_BATCH_MAXM = 16): N below it (8), at it (16), and
/// above it (24, which falls through to the tiled GEMM). For N copies of one
/// prompt every output sequence must equal the single-stream `generate` of that
/// prompt — the per-row dot is bit-identical whether it runs through the m=1
/// GEMV, the batched GEMV, or the tiled fall-through. Larger KV pool so N>4 fits.
#[test]
fn gpu_batched_gemv_matches_single_stream_across_cap() {
    use arf_gpu::weights::build_gpu;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping batched-GEMV cap test: {e}");
            return;
        }
    };

    let cfg = cfg();
    let w = weights(&cfg);
    // 64 blocks (16 * block_size positions) so up to ~64 single-token seqs fit.
    let gpu = build_gpu(&cfg, &w, 16 * 64, &gctx).unwrap();

    let prompt = vec![3u32, 1, 4];
    let single = gpu.generate(&prompt, 5);

    // N straddling MATMUL_VEC_BATCH_MAXM (=16): below, at, and above (fall-through).
    for n in [8usize, 16, 24] {
        let prompts = vec![prompt.clone(); n];
        let batch = gpu.generate_batch(&prompts, 5);
        assert_eq!(batch.len(), n);
        for (s, seq) in batch.iter().enumerate() {
            assert_eq!(
                seq, &single,
                "batched N={n} seq {s} diverged from single-stream (kernel routing bug?)"
            );
        }
    }
}

/// Batched int8 (matmul_vec_q8_batch.wgsl) must match single-stream int8 across
/// the cap: N=8 (one batched dispatch), N=16 (at MAXM), N=24 (the chunked
/// over-MAXM path, two dispatches advancing row_off). The dequant (q · scale[col])
/// is identical to the m=1 int8 GEMV, so int8 `generate` is the oracle. This is
/// the path that combines int8 bandwidth with batched amortization — batched
/// decode runs int8 now, not the bf16 fallback.
#[test]
fn gpu_int8_batched_matches_single_stream_across_cap() {
    use arf_core::config::Quant;
    use arf_gpu::weights::build_gpu_quant;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping int8 batched cap test: {e}");
            return;
        }
    };

    let cfg = cfg();
    let w = weights(&cfg);
    let gpu = build_gpu_quant(&cfg, &w, 16 * 64, &gctx, Quant::Int8).unwrap();

    let prompt = vec![3u32, 1, 4];
    let single = gpu.generate(&prompt, 5); // int8 model -> int8 single-stream oracle

    for n in [8usize, 16, 24] {
        let prompts = vec![prompt.clone(); n];
        let batch = gpu.generate_batch(&prompts, 5);
        assert_eq!(batch.len(), n);
        for (s, seq) in batch.iter().enumerate() {
            assert_eq!(
                seq, &single,
                "int8 batched N={n} seq {s} diverged from int8 single-stream"
            );
        }
    }
}

/// int8 decode on the GPU matches int8 decode on the CPU token-for-token: the
/// GEMV's dequant (`w ≈ q · scale[row]`) is the same per-row symmetric int8 as
/// the CPU `matmul_nt_q8`, so the CPU int8 model is the exact oracle for the GPU
/// int8 path.
#[test]
fn gpu_int8_generate_matches_cpu_int8() {
    use arf_core::config::{EngineConfig, Quant};
    use arf_core::engine::LlmEngine;
    use arf_core::model::weights::build_quant;
    use arf_core::sampling::SamplingParams;
    use arf_gpu::weights::build_gpu_quant;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU int8 parity: {e}");
            return;
        }
    };

    let cfg = cfg();
    let w = weights(&cfg);
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Int8).unwrap();
    let cpu_model = build_quant(
        &cfg,
        &w,
        cfg.max_position_embeddings,
        &Device::Cpu,
        Quant::Int8,
    )
    .unwrap();
    let mut engine = LlmEngine::new(cpu_model, EngineConfig::default()).unwrap();

    let prompt = vec![3u32, 1, 4];
    let cpu_out = engine
        .generate(prompt.clone(), SamplingParams::greedy(6))
        .unwrap();
    let gpu_out = gpu.generate(&prompt, 6);

    let n = cpu_out.len().min(gpu_out.len());
    assert!(n > 0, "no tokens generated");
    assert_eq!(
        &gpu_out[..n],
        &cpu_out[..n],
        "GPU int8 generation diverged from CPU int8"
    );
}

/// Isolate the `matmul_vec_q4` kernel: dispatch ONE GEMV `c[n] = a[k] · Q4ᵀ` and
/// compare to the CPU `matmul_nt_q4` oracle on the same matrix. Pinpoints a kernel
/// bug separately from the full decode pipeline. n=5 rows, k=64 (2 blocks/row).
#[test]
fn gpu_q4_gemv_kernel_matches_oracle() {
    use arf_core::tensor::Q4Matrix;
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping Q4 kernel test: {e}");
            return;
        }
    };

    // k=32 is the REAL projection contraction dim (hidden_size) — 1 block/row,
    // the case the full-model logit test exercises; test it directly here.
    let (n, k) = (5usize, 32usize);
    // A weight matrix and an activation with values that exercise both signs.
    let mut wv = vec![0.0f32; n * k];
    for (i, v) in wv.iter_mut().enumerate() {
        *v = ((i * 13 % 31) as f32 - 15.0) * 0.21;
    }
    let q = Q4Matrix::from_f32(&wv, n, k);
    let a: Vec<f32> = (0..k).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3).collect();

    // Oracle: the CPU Q4 linear (which calls matmul_nt_q4).
    let want = q.linear(&arf_core::Tensor::from_vec(a.clone(), vec![1, k]));
    let want = want.as_slice();

    // GPU dispatch of matmul_vec_q4 with the same bindings the decode path uses.
    const SRC: &str = include_str!("../src/shaders/wgsl/matmul_vec_q4.wgsl");
    let kernel = ComputeKernel::new(&ctx, "matmul_vec_q4", SRC);
    let a_buf = ctx.storage_init("q4-a", &a);
    let codes_buf = ctx.storage_init("q4-codes", q.codes());
    let scales_u32: Vec<u32> = q.scales().iter().map(|&s| s as u32).collect();
    let scales_buf = ctx.storage_init("q4-scales", &scales_u32);
    let out_buf = ctx.storage_zeros("q4-out", n);
    let dims = ctx.uniform_of("q4-dims", &[1u32, k as u32, n as u32, 0u32]);
    kernel.dispatch(
        "matmul_vec_q4",
        &[&a_buf, &codes_buf, &out_buf, &dims, &scales_buf],
        [n as u32, 1, 1],
    );
    let got = ctx.read_f32(&out_buf, n);

    for (j, (&g, &wnt)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - wnt).abs() <= 1e-4 * wnt.abs().max(1.0),
            "row {j}: gpu {g} vs oracle {wnt}"
        );
    }
}

/// The Q4 oracle: GPU Q4_0 decode must match CPU Q4_0 decode token-for-token.
/// Q4 diverges from int8/bf16 by design (coarser quant), so the CPU Q4 path is
/// its OWN bit-exact oracle — the `matmul_vec_q4` kernel must reproduce
/// `matmul_nt_q4` exactly (same block scale × same integer sub-dot).
#[test]
fn gpu_q4_generate_matches_cpu_q4() {
    use arf_core::config::{EngineConfig, Quant};
    use arf_core::engine::LlmEngine;
    use arf_core::model::weights::build_quant;
    use arf_core::sampling::SamplingParams;
    use arf_gpu::weights::build_gpu_quant;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU Q4 parity: {e}");
            return;
        }
    };

    let cfg = cfg_q4();
    let w = weights(&cfg);
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Q4).unwrap();
    let cpu_model = build_quant(
        &cfg,
        &w,
        cfg.max_position_embeddings,
        &Device::Cpu,
        Quant::Q4,
    )
    .unwrap();
    let mut engine = LlmEngine::new(cpu_model, EngineConfig::default()).unwrap();

    let prompt = vec![3u32, 1, 4];
    let cpu_out = engine
        .generate(prompt.clone(), SamplingParams::greedy(6))
        .unwrap();
    let gpu_out = gpu.generate(&prompt, 6);

    let n = cpu_out.len().min(gpu_out.len());
    assert!(n > 0, "no tokens generated");
    assert_eq!(
        &gpu_out[..n],
        &cpu_out[..n],
        "GPU Q4 generation diverged from CPU Q4"
    );
}

/// Control: on the SAME cfg_q4 model, how tightly do GPU and CPU INT8 logits
/// agree? Tells us whether the Q4 logit gap is Q4-specific or just this model
/// amplifying any CPU/GPU reduction-order difference through its layers.
#[test]
fn gpu_int8_logits_match_cpu_int8_on_cfg_q4() {
    use arf_core::config::Quant;
    use arf_core::model::weights::build_quant;
    use arf_gpu::weights::build_gpu_quant;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping int8-on-cfg_q4 logits: {e}");
            return;
        }
    };
    let cfg = cfg_q4();
    let w = weights(&cfg);
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Int8).unwrap();
    let cpu = build_quant(
        &cfg,
        &w,
        cfg.max_position_embeddings,
        &Device::Cpu,
        Quant::Int8,
    )
    .unwrap();
    let tokens = vec![5u32, 2, 9, 1];
    let cl = last_logits(&cpu, &tokens);
    let gl = gpu.forward_logits(&tokens);
    let max_rel = cl
        .iter()
        .zip(&gl)
        .map(|(c, g)| (c - g).abs() / c.abs().max(1.0))
        .fold(0.0f32, f32::max);
    eprintln!("int8-on-cfg_q4 max relative logit diff: {max_rel}");
    assert!(max_rel <= 5e-2, "int8 logits diverged by {max_rel} rel");
}

/// GPU Q4 logits must match CPU Q4 logits (a tighter, per-value check than the
/// token match — catches a kernel that agrees on argmax but not the full vector).
#[test]
fn gpu_q4_logits_match_cpu_q4() {
    use arf_core::config::Quant;
    use arf_core::model::weights::build_quant;
    use arf_gpu::weights::build_gpu_quant;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU Q4 logits parity: {e}");
            return;
        }
    };

    let cfg = cfg_q4();
    let w = weights(&cfg);
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Q4).unwrap();
    let cpu = build_quant(
        &cfg,
        &w,
        cfg.max_position_embeddings,
        &Device::Cpu,
        Quant::Q4,
    )
    .unwrap();

    let tokens = vec![5u32, 2, 9, 1];
    let cpu_logits = last_logits(&cpu, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);
    assert_eq!(cpu_logits.len(), gpu_logits.len());
    // Both run the SAME Q4 arithmetic (block scale × int sub-dot); the only gap is
    // f32 reassociation across the GPU's lane reduction — well under a logit tie.
    for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
        assert!(
            (c - g).abs() <= 1e-2 * c.abs().max(1.0),
            "logit {i}: cpu {c} vs gpu {g}"
        );
    }
}

/// Isolate the `matmul_vec_q4k` kernel: one GEMV vs the CPU `matmul_nt_q4k` oracle.
/// k=256 (one super-block) and the asymmetric min-offset path. Pinpoints a kernel
/// bug apart from the full decode pipeline.
#[test]
fn gpu_q4k_gemv_kernel_matches_oracle() {
    use arf_core::tensor::Q4KMatrix;
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping Q4_K kernel test: {e}");
            return;
        }
    };

    let (n, k) = (5usize, 256usize); // one super-block
    let mut wv = vec![0.0f32; n * k];
    for (i, v) in wv.iter_mut().enumerate() {
        *v = ((i * 13 % 31) as f32 - 12.0) * 0.21; // skewed, exercises the min-offset
    }
    let q = Q4KMatrix::from_f32(&wv, n, k);
    let a: Vec<f32> = (0..k).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3).collect();
    let want = q.linear(&arf_core::Tensor::from_vec(a.clone(), vec![1, k]));
    let want = want.as_slice();

    const SRC: &str = include_str!("../src/shaders/wgsl/matmul_vec_q4k.wgsl");
    let kernel = ComputeKernel::new(&ctx, "matmul_vec_q4k", SRC);
    let a_buf = ctx.storage_init("q4k-a", &a);
    let codes_buf = ctx.storage_init("q4k-codes", q.codes());
    let scales_u32: Vec<u32> = q.scales().iter().map(|&s| s as u32).collect();
    let mins_u32: Vec<u32> = q.mins().iter().map(|&s| s as u32).collect();
    let scales_buf = ctx.storage_init("q4k-scales", &scales_u32);
    let mins_buf = ctx.storage_init("q4k-mins", &mins_u32);
    let out_buf = ctx.storage_zeros("q4k-out", n);
    let dims = ctx.uniform_of("q4k-dims", &[1u32, k as u32, n as u32, 0u32]);
    kernel.dispatch(
        "matmul_vec_q4k",
        &[&a_buf, &codes_buf, &out_buf, &dims, &scales_buf, &mins_buf],
        [n as u32, 1, 1],
    );
    let got = ctx.read_f32(&out_buf, n);
    for (j, (&g, &wnt)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - wnt).abs() <= 1e-4 * wnt.abs().max(1.0),
            "row {j}: gpu {g} vs oracle {wnt}"
        );
    }
}

/// GPU Q4_K decode == CPU Q4_K decode token-for-token (Q4_K is its own oracle).
#[test]
fn gpu_q4k_generate_matches_cpu_q4k() {
    use arf_core::config::{EngineConfig, Quant};
    use arf_core::engine::LlmEngine;
    use arf_core::model::weights::build_quant;
    use arf_core::sampling::SamplingParams;
    use arf_gpu::weights::build_gpu_quant;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU Q4_K parity: {e}");
            return;
        }
    };

    let cfg = cfg_q4k();
    let w = weights(&cfg);
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Q4K).unwrap();
    let cpu_model = build_quant(
        &cfg,
        &w,
        cfg.max_position_embeddings,
        &Device::Cpu,
        Quant::Q4K,
    )
    .unwrap();
    let mut engine = LlmEngine::new(cpu_model, EngineConfig::default()).unwrap();

    let prompt = vec![3u32, 1, 4];
    let cpu_out = engine
        .generate(prompt.clone(), SamplingParams::greedy(6))
        .unwrap();
    let gpu_out = gpu.generate(&prompt, 6);
    let n = cpu_out.len().min(gpu_out.len());
    assert!(n > 0, "no tokens generated");
    assert_eq!(
        &gpu_out[..n],
        &cpu_out[..n],
        "GPU Q4_K generation diverged from CPU Q4_K"
    );
}

/// GPU Q4_K logits match CPU Q4_K logits (per-value, tighter than token match).
#[test]
fn gpu_q4k_logits_match_cpu_q4k() {
    use arf_core::config::Quant;
    use arf_core::model::weights::build_quant;
    use arf_gpu::weights::build_gpu_quant;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU Q4_K logits parity: {e}");
            return;
        }
    };

    let cfg = cfg_q4k();
    let w = weights(&cfg);
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Q4K).unwrap();
    let cpu = build_quant(
        &cfg,
        &w,
        cfg.max_position_embeddings,
        &Device::Cpu,
        Quant::Q4K,
    )
    .unwrap();

    let tokens = vec![5u32, 2, 9, 1];
    let cpu_logits = last_logits(&cpu, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);
    assert_eq!(cpu_logits.len(), gpu_logits.len());
    for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
        assert!(
            (c - g).abs() <= 1e-2 * c.abs().max(1.0),
            "logit {i}: cpu {c} vs gpu {g}"
        );
    }
}

/// GPU Q4_K_S decode == CPU Q4_K_S decode token-for-token + logits per-value. The
/// two-level u8/i8 scale path is its own bit-exact oracle.
#[test]
fn gpu_q4ks_matches_cpu_q4ks() {
    use arf_core::config::{EngineConfig, Quant};
    use arf_core::engine::LlmEngine;
    use arf_core::model::weights::build_quant;
    use arf_core::sampling::SamplingParams;
    use arf_gpu::weights::build_gpu_quant;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU Q4_K_S parity: {e}");
            return;
        }
    };

    let cfg = cfg_q4k(); // dims % 256, valid for Q4_K_S
    let w = weights(&cfg);
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Q4KS).unwrap();
    let cpu_model = build_quant(
        &cfg,
        &w,
        cfg.max_position_embeddings,
        &Device::Cpu,
        Quant::Q4KS,
    )
    .unwrap();

    // Logits match (tighter than tokens).
    let tokens = vec![5u32, 2, 9, 1];
    let cpu_logits = last_logits(&cpu_model, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);
    assert_eq!(cpu_logits.len(), gpu_logits.len());
    for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
        assert!(
            (c - g).abs() <= 1e-2 * c.abs().max(1.0),
            "logit {i}: cpu {c} vs gpu {g}"
        );
    }

    // Token-for-token generation.
    let mut engine = LlmEngine::new(cpu_model, EngineConfig::default()).unwrap();
    let prompt = vec![3u32, 1, 4];
    let cpu_out = engine
        .generate(prompt.clone(), SamplingParams::greedy(6))
        .unwrap();
    let gpu_out = gpu.generate(&prompt, 6);
    let n = cpu_out.len().min(gpu_out.len());
    assert!(n > 0, "no tokens generated");
    assert_eq!(
        &gpu_out[..n],
        &cpu_out[..n],
        "GPU Q4_K_S generation diverged from CPU Q4_K_S"
    );
}

/// Batched Q4_K_S (m>1) prefill must match the m==1 Q4_K_S GEMV.
#[test]
fn forward_batch_q4ks_matches_forward_logits() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::config::Quant;
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu_quant;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip Q4_K_S batch test: {e}");
            return;
        }
    };
    let cfg = cfg_q4k();
    let w = weights(&cfg);
    let bs = 4usize;
    let tbl = [0u32, 1];
    let ids = [3u32, 7, 1];

    let g1 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4KS).unwrap();
    let logits_ref = g1.forward_logits(&ids);
    let g2 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4KS).unwrap();
    let batch = ForwardBatch {
        positions: vec![0, 1, 2],
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: 3,
            past_len: 0,
            slots: slots_for(&tbl, bs, 3),
            write_runs: write_runs(&tbl, bs, 0, 3),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let logits_batch = g2.forward_batch(&ids, &batch);
    let diff = max_abs_diff(&logits_batch[0], &logits_ref);
    assert!(
        diff < 1e-3,
        "batched Q4_K_S (m=3) vs m==1 Q4_K_S GEMV diverged by {diff}"
    );
}

/// GPU Q3_K decode == CPU Q3_K (logits + tokens). The 3-bit split-plane (ql/qh)
/// kernel must reproduce the CPU `matmul_nt_q3k` oracle. Q3_K is the danger zone for
/// ACCURACY (3-bit, small model) — this guards CORRECTNESS (GPU==CPU); the accuracy-
/// vs-bf16 trade is measured separately on the real model.
#[test]
fn gpu_q3k_matches_cpu_q3k() {
    use arf_core::config::{EngineConfig, Quant};
    use arf_core::engine::LlmEngine;
    use arf_core::model::weights::build_quant;
    use arf_core::sampling::SamplingParams;
    use arf_gpu::weights::build_gpu_quant;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping GPU Q3_K parity: {e}");
            return;
        }
    };

    let cfg = cfg_q4k(); // dims % 256, valid for Q3_K
    let w = weights(&cfg);
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Q3K).unwrap();
    let cpu_model = build_quant(
        &cfg,
        &w,
        cfg.max_position_embeddings,
        &Device::Cpu,
        Quant::Q3K,
    )
    .unwrap();

    let tokens = vec![5u32, 2, 9, 1];
    let cpu_logits = last_logits(&cpu_model, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);
    assert_eq!(cpu_logits.len(), gpu_logits.len());
    for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
        assert!(
            (c - g).abs() <= 1e-2 * c.abs().max(1.0),
            "logit {i}: cpu {c} vs gpu {g}"
        );
    }

    let mut engine = LlmEngine::new(cpu_model, EngineConfig::default()).unwrap();
    let prompt = vec![3u32, 1, 4];
    let cpu_out = engine
        .generate(prompt.clone(), SamplingParams::greedy(6))
        .unwrap();
    let gpu_out = gpu.generate(&prompt, 6);
    let n = cpu_out.len().min(gpu_out.len());
    assert!(n > 0, "no tokens generated");
    assert_eq!(
        &gpu_out[..n],
        &cpu_out[..n],
        "GPU Q3_K generation diverged from CPU Q3_K"
    );
}

/// Batched Q3_K (m>1) prefill must match the m==1 Q3_K GEMV.
#[test]
fn forward_batch_q3k_matches_forward_logits() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::config::Quant;
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu_quant;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip Q3_K batch test: {e}");
            return;
        }
    };
    let cfg = cfg_q4k();
    let w = weights(&cfg);
    let bs = 4usize;
    let tbl = [0u32, 1];
    let ids = [3u32, 7, 1];

    let g1 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q3K).unwrap();
    let logits_ref = g1.forward_logits(&ids);
    let g2 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q3K).unwrap();
    let batch = ForwardBatch {
        positions: vec![0, 1, 2],
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: 3,
            past_len: 0,
            slots: slots_for(&tbl, bs, 3),
            write_runs: write_runs(&tbl, bs, 0, 3),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let logits_batch = g2.forward_batch(&ids, &batch);
    let diff = max_abs_diff(&logits_batch[0], &logits_ref);
    assert!(
        diff < 1e-3,
        "batched Q3_K (m=3) vs m==1 Q3_K GEMV diverged by {diff}"
    );
}

/// PROTOTYPE: does batching amortize the weight stream? Times a resident
/// [m x 2048] x [8192 x 2048]ᵀ matmul (an MLP-shape projection) across batch
/// sizes; if per-row time drops as m grows, batch=1 is bandwidth-bound and
/// batched decode is the real GPU win. Ignored (a perf probe, not correctness).
#[test]
#[ignore = "perf probe; run with --ignored --nocapture"]
fn gpu_matmul_batch_scaling() {
    use std::time::Instant;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping batch-scaling probe: {e}");
            return;
        }
    };
    eprintln!("batch-scaling on {}", ctx.info());

    let (k, n) = (2048usize, 8192usize); // gate/up projection shape
    let weight: Vec<f32> = (0..n * k).map(|i| (i % 17) as f32 * 0.01 - 0.08).collect();
    let w = ctx.upload_weight(&weight, n, k);

    let iters = 20;
    for &m in &[1usize, 4, 8, 16, 32, 64] {
        let x: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.01).collect();
        let _ = w.linear(&x, m); // warm
        let t = Instant::now();
        for _ in 0..iters {
            let _ = w.linear(&x, m);
        }
        let per_call_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let per_row_us = per_call_ms * 1000.0 / m as f64;
        eprintln!("  m={m:<3} {per_call_ms:7.3} ms/call  {per_row_us:7.1} us/row",);
    }
}

/// GpuModel::forward_batch runs a ragged multi-sequence
/// batch and matches Llama::forward + logits_last per sequence. Two sequences,
/// different lengths, sharing one batched forward.
#[test]
fn forward_batch_matches_cpu_ragged() {
    use arf_core::cache::{slots_for, write_runs, PagedKvCache};
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping forward_batch parity: {e}");
            return;
        }
    };

    let cfg = cfg();
    let w = weights(&cfg);
    let bs = 4usize;

    // Two sequences: A in physical blocks [0,1], B in [2,3].
    let a_tbl = [0u32, 1];
    let b_tbl = [2u32, 3];

    // A ragged step: A decodes 1 token at past_len 2; B prefills 3 tokens at past 0.
    // (A is seeded with 2 tokens first so its decode has real context.)
    let seed_ids = [3u32, 7];
    let seed = ForwardBatch {
        positions: vec![0, 1],
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: 2,
            past_len: 0,
            slots: slots_for(&a_tbl, bs, 2),
            write_runs: write_runs(&a_tbl, bs, 0, 2),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let step_ids = [9u32, 2, 5, 1];
    let step = ForwardBatch {
        positions: vec![2, 0, 1, 2],
        seqs: vec![
            SeqAttn {
                q_start: 0,
                q_len: 1,
                past_len: 2,
                slots: slots_for(&a_tbl, bs, 3),
                write_runs: write_runs(&a_tbl, bs, 2, 1),
                image_spans: Vec::new(),
                stream_id: None,
            },
            SeqAttn {
                q_start: 1,
                q_len: 3,
                past_len: 0,
                slots: slots_for(&b_tbl, bs, 3),
                write_runs: write_runs(&b_tbl, bs, 0, 3),
                image_spans: Vec::new(),
                stream_id: None,
            },
        ],
        image_embeds: None,
        mrope_positions: None,
    };

    // CPU oracle.
    let cpu = build_on(&cfg, &w, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let mut cache = PagedKvCache::new(cfg.num_layers, 8, bs, cfg.num_kv_heads, cfg.head_dim);
    cpu.forward(&seed_ids, &seed, &mut cache);
    let h = cpu.forward(&step_ids, &step, &mut cache);
    let cpu_logits = cpu.logits_last(&h, &step); // [2, vocab]

    // GPU: same seed + same ragged step.
    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &ctx).unwrap();
    let _ = gpu.forward_batch(&seed_ids, &seed); // seeds A's KV into the pool
    let gpu_logits = gpu.forward_batch(&step_ids, &step); // Vec<Vec<f32>>, per seq

    assert_eq!(gpu_logits.len(), 2);
    for (s, gpu_row) in gpu_logits.iter().enumerate() {
        let diff = max_abs_diff(gpu_row, cpu_logits.row(s));
        assert!(diff < 1e-3, "seq {s} GPU vs CPU diverged by {diff}");
    }
}

/// Isolate the q_start>0 path: two DECODE sequences (q_len=1 each) at q_start 0
/// and 1, both seeded. If seq 1 (q_start=1) matches CPU, q_start handling is
/// correct and any forward_batch divergence is prefill-specific.
#[test]
fn forward_batch_two_decodes() {
    use arf_core::cache::{slots_for, write_runs, PagedKvCache};
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    let cfg = cfg();
    let w = weights(&cfg);
    let bs = 4usize;
    let a_tbl = [0u32, 1];
    let b_tbl = [2u32, 3];

    // Seed A (2 tok) and B (2 tok), then a batch of two decodes at past_len 2.
    let seed = |q_start, tbl: &[u32]| ForwardBatch {
        positions: vec![0, 1],
        seqs: vec![SeqAttn {
            q_start,
            q_len: 2,
            past_len: 0,
            slots: slots_for(tbl, bs, 2),
            write_runs: write_runs(tbl, bs, 0, 2),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let step = ForwardBatch {
        positions: vec![2, 2],
        seqs: vec![
            SeqAttn {
                q_start: 0,
                q_len: 1,
                past_len: 2,
                slots: slots_for(&a_tbl, bs, 3),
                write_runs: write_runs(&a_tbl, bs, 2, 1),
                image_spans: Vec::new(),
                stream_id: None,
            },
            SeqAttn {
                q_start: 1,
                q_len: 1,
                past_len: 2,
                slots: slots_for(&b_tbl, bs, 3),
                write_runs: write_runs(&b_tbl, bs, 2, 1),
                image_spans: Vec::new(),
                stream_id: None,
            },
        ],
        image_embeds: None,
        mrope_positions: None,
    };

    let cpu = build_on(&cfg, &w, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let mut cache = PagedKvCache::new(cfg.num_layers, 8, bs, cfg.num_kv_heads, cfg.head_dim);
    cpu.forward(&[3, 7], &seed(0, &a_tbl), &mut cache);
    cpu.forward(&[4, 8], &seed(0, &b_tbl), &mut cache); // B's 2 seed tokens into b_tbl slots
    let h = cpu.forward(&[9, 5], &step, &mut cache);
    let cpu_logits = cpu.logits_last(&h, &step);

    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &ctx).unwrap();
    let _ = gpu.forward_batch(&[3, 7], &seed(0, &a_tbl));
    let _ = gpu.forward_batch(&[4, 8], &seed(0, &b_tbl));
    let gpu_logits = gpu.forward_batch(&[9, 5], &step);

    for (s, gpu_row) in gpu_logits.iter().enumerate() {
        let diff = max_abs_diff(gpu_row, cpu_logits.row(s));
        assert!(diff < 1e-3, "decode seq {s} diverged by {diff}");
    }
}

/// Bisect: single-sequence forward_batch (B=1, q_start=0) must match the proven
/// single-stream forward_logits exactly. If this passes, the batched layout/
/// trunk is correct and any divergence is multi-sequence-specific.
#[test]
fn forward_batch_single_seq_matches_forward_logits() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    let cfg = cfg();
    let w = weights(&cfg);
    let bs = 4usize;
    let tbl = [0u32, 1];
    let ids = [3u32, 7, 1];

    // forward_logits prefills internally; forward_batch needs the matching batch.
    let g1 = build_gpu(&cfg, &w, cfg.max_position_embeddings, &ctx).unwrap();
    let logits_ref = g1.forward_logits(&ids); // proven path

    let g2 = build_gpu(&cfg, &w, cfg.max_position_embeddings, &ctx).unwrap();
    let batch = ForwardBatch {
        positions: vec![0, 1, 2],
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: 3,
            past_len: 0,
            slots: slots_for(&tbl, bs, 3),
            write_runs: write_runs(&tbl, bs, 0, 3),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let logits_batch = g2.forward_batch(&ids, &batch);

    let diff = max_abs_diff(&logits_batch[0], &logits_ref);
    assert!(
        diff < 1e-3,
        "single-seq forward_batch vs forward_logits diverged by {diff}"
    );
}

/// Batched Q4 GEMV (P20): a multi-token prefill drives `add_matmul_m` with m>1
/// through the new Q4 batch kernel; its last-token logits must match the m==1 Q4
/// GEMV (forward_logits). This is the only path that exercises matmul_vec_q4_batch
/// — generate_batch is 1 token/seq (m==1). Guards the batch CSE + per-block scale.
#[test]
fn forward_batch_q4_matches_forward_logits() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::config::Quant;
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu_quant;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip Q4 batch test: {e}");
            return;
        }
    };
    let cfg = cfg_q4(); // dims % 32 for Q4
    let w = weights(&cfg);
    let bs = 4usize;
    let tbl = [0u32, 1];
    let ids = [3u32, 7, 1];

    let g1 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4).unwrap();
    let logits_ref = g1.forward_logits(&ids); // m==1 Q4 GEMV, the proven path

    let g2 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4).unwrap();
    let batch = ForwardBatch {
        positions: vec![0, 1, 2],
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: 3, // m=3 -> the Q4 batch kernel
            past_len: 0,
            slots: slots_for(&tbl, bs, 3),
            write_runs: write_runs(&tbl, bs, 0, 3),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let logits_batch = g2.forward_batch(&ids, &batch);
    let diff = max_abs_diff(&logits_batch[0], &logits_ref);
    assert!(
        diff < 1e-3,
        "batched Q4 (m=3) vs m==1 Q4 GEMV diverged by {diff}"
    );
}

/// Cooperative-matrix Q4 GEMM (C4): a prefill with q_len ≥ 8 drives `add_matmul_m`
/// with m ≥ 8, routing every Q4 projection through `matmul_coop_q4` (dequant into
/// f32 staging, then mma) instead of the scalar Q4 batch GEMV. Its last-token
/// logits must match the m==1 Q4 GEMV — the proven scalar oracle. Pins the
/// per-element nibble unpack + per-block scale fold of the coop path. Skips when
/// the device has no cooperative-matrix feature (the dispatch falls back to scalar).
#[test]
fn forward_batch_q4_coop_matches_forward_logits() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::config::Quant;
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu_quant;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip Q4 coop test: {e}");
            return;
        }
    };
    if !ctx.has_coop() {
        eprintln!("skip Q4 coop test: device has no cooperative-matrix feature");
        return;
    }
    let cfg = cfg_q4(); // hidden/intermediate/head_dim all % 8 == 0
    let w = weights(&cfg);
    let bs = 4usize;
    let tbl = [0u32, 1, 2, 3]; // 4 blocks * bs=4 = 16 slots >= 9 tokens
                               // 9 tokens -> q_len=9 -> m=9 >= 8 -> the coop Q4 path on every projection.
    let ids = [3u32, 7, 1, 5, 9, 2, 8, 4, 6];

    let g1 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4).unwrap();
    let logits_ref = g1.forward_logits(&ids); // m==1 Q4 GEMV, the proven path

    let g2 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4).unwrap();
    let batch = ForwardBatch {
        positions: (0..ids.len() as u32).collect(),
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: ids.len(),
            past_len: 0,
            slots: slots_for(&tbl, bs, ids.len()),
            write_runs: write_runs(&tbl, bs, 0, ids.len()),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let logits_batch = g2.forward_batch(&ids, &batch);
    let diff = max_abs_diff(&logits_batch[0], &logits_ref);
    assert!(
        diff < 1e-3,
        "coop Q4 (m=9) vs m==1 Q4 GEMV diverged by {diff}"
    );
}

/// RB=2 register-blocked coop Q4 (C2 propagation): a prefill with q_len ≥ 16 routes
/// the Q4 projections through `matmul_coop_q4_rb2` (two stacked tiles, weight
/// unpack+dequant shared). m=18 also crosses a 16-row group boundary (2 groups, 18
/// valid rows) so the edge guard is exercised. Must match the m==1 Q4 GEMV oracle.
#[test]
fn forward_batch_q4_coop_rb2_matches_forward_logits() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::config::Quant;
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu_quant;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip Q4 coop RB2 test: {e}");
            return;
        }
    };
    if !ctx.has_coop() {
        eprintln!("skip Q4 coop RB2 test: no cooperative-matrix feature");
        return;
    }
    let cfg = cfg_q4();
    let w = weights(&cfg);
    let bs = 4usize;
    let tbl: Vec<u32> = (0..6).collect(); // 24 slots >= 18 tokens
    let ids: Vec<u32> = (0..18u32)
        .map(|i| (i * 7 + 3) % cfg.vocab_size as u32)
        .collect();

    let g1 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4).unwrap();
    let logits_ref = g1.forward_logits(&ids);
    let g2 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4).unwrap();
    let batch = ForwardBatch {
        positions: (0..ids.len() as u32).collect(),
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: ids.len(),
            past_len: 0,
            slots: slots_for(&tbl, bs, ids.len()),
            write_runs: write_runs(&tbl, bs, 0, ids.len()),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let logits_batch = g2.forward_batch(&ids, &batch);
    let diff = max_abs_diff(&logits_batch[0], &logits_ref);
    assert!(
        diff < 1e-3,
        "coop Q4 RB2 (m=18) vs m==1 GEMV diverged by {diff}"
    );
}

/// RB=2 register-blocked coop Q4_K (C2 propagation): q_len=18 → `matmul_coop_q4k_rb2`,
/// crossing a 16-row group boundary. Must match the m==1 Q4_K GEMV oracle.
#[test]
fn forward_batch_q4k_coop_rb2_matches_forward_logits() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::config::Quant;
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu_quant;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip Q4_K coop RB2 test: {e}");
            return;
        }
    };
    if !ctx.has_coop() {
        eprintln!("skip Q4_K coop RB2 test: no cooperative-matrix feature");
        return;
    }
    let cfg = cfg_q4k();
    let w = weights(&cfg);
    let bs = 4usize;
    let tbl: Vec<u32> = (0..6).collect();
    let ids: Vec<u32> = (0..18u32)
        .map(|i| (i * 5 + 1) % cfg.vocab_size as u32)
        .collect();

    let g1 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4K).unwrap();
    let logits_ref = g1.forward_logits(&ids);
    let g2 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4K).unwrap();
    let batch = ForwardBatch {
        positions: (0..ids.len() as u32).collect(),
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: ids.len(),
            past_len: 0,
            slots: slots_for(&tbl, bs, ids.len()),
            write_runs: write_runs(&tbl, bs, 0, ids.len()),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let logits_batch = g2.forward_batch(&ids, &batch);
    let diff = max_abs_diff(&logits_batch[0], &logits_ref);
    assert!(
        diff < 1e-3,
        "coop Q4_K RB2 (m=18) vs m==1 GEMV diverged by {diff}"
    );
}

/// RB=2×CB=4 coop bf16 (breadth port): a prefill with q_len=18 routes the bf16
/// projections through `matmul_coop_rb2_cb4` (16×32 tile, 8 mma/barrier; gy=ceil(n/32)),
/// crossing both a 16-row and a 32-col group boundary so the edge guards are exercised.
/// Must match the m==1 bf16 GEMV oracle.
#[test]
fn forward_batch_bf16_coop_cb4_matches_forward_logits() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip bf16 coop CB4 test: {e}");
            return;
        }
    };
    if !ctx.has_coop() {
        eprintln!("skip bf16 coop CB4 test: no cooperative-matrix feature");
        return;
    }
    let cfg = cfg_q4();
    let w = weights(&cfg);
    let bs = 4usize;
    let tbl: Vec<u32> = (0..6).collect();
    let ids: Vec<u32> = (0..18u32)
        .map(|i| (i * 3 + 2) % cfg.vocab_size as u32)
        .collect();

    let g1 = build_gpu(&cfg, &w, cfg.max_position_embeddings, &ctx).unwrap();
    let logits_ref = g1.forward_logits(&ids);
    let g2 = build_gpu(&cfg, &w, cfg.max_position_embeddings, &ctx).unwrap();
    let batch = ForwardBatch {
        positions: (0..ids.len() as u32).collect(),
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: ids.len(),
            past_len: 0,
            slots: slots_for(&tbl, bs, ids.len()),
            write_runs: write_runs(&tbl, bs, 0, ids.len()),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let logits_batch = g2.forward_batch(&ids, &batch);
    let diff = max_abs_diff(&logits_batch[0], &logits_ref);
    assert!(
        diff < 2e-3,
        "coop bf16 CB4 (m=18) vs m==1 GEMV diverged by {diff}"
    );
}

/// RB=2×CB=4 coop int8 (breadth port): q_len=18 → `matmul_coop_q8_rb2_cb4`, with the
/// per-column dequant scale applied per column-tile at store. Must match the m==1
/// int8 GEMV oracle.
#[test]
fn forward_batch_q8_coop_cb4_matches_forward_logits() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::config::Quant;
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu_quant;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip int8 coop CB4 test: {e}");
            return;
        }
    };
    if !ctx.has_coop() {
        eprintln!("skip int8 coop CB4 test: no cooperative-matrix feature");
        return;
    }
    let cfg = cfg_q4();
    let w = weights(&cfg);
    let bs = 4usize;
    let tbl: Vec<u32> = (0..6).collect();
    let ids: Vec<u32> = (0..18u32)
        .map(|i| (i * 7 + 5) % cfg.vocab_size as u32)
        .collect();

    let g1 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Int8).unwrap();
    let logits_ref = g1.forward_logits(&ids);
    let g2 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Int8).unwrap();
    let batch = ForwardBatch {
        positions: (0..ids.len() as u32).collect(),
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: ids.len(),
            past_len: 0,
            slots: slots_for(&tbl, bs, ids.len()),
            write_runs: write_runs(&tbl, bs, 0, ids.len()),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let logits_batch = g2.forward_batch(&ids, &batch);
    let diff = max_abs_diff(&logits_batch[0], &logits_ref);
    assert!(
        diff < 2e-3,
        "coop int8 CB4 (m=18) vs m==1 GEMV diverged by {diff}"
    );
}

/// Cooperative-matrix Q4_K GEMM (C4b): a prefill with q_len ≥ 8 routes every Q4_K
/// projection through `matmul_coop_q4k` (unsigned nibble unpack + per-sub-block
/// scale·nib+min folded into f32 staging, then mma) instead of the scalar Q4_K
/// batch GEMV. Last-token logits must match the m==1 Q4_K GEMV oracle. Skips when
/// the device has no cooperative-matrix feature.
#[test]
fn forward_batch_q4k_coop_matches_forward_logits() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::config::Quant;
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu_quant;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip Q4_K coop test: {e}");
            return;
        }
    };
    if !ctx.has_coop() {
        eprintln!("skip Q4_K coop test: device has no cooperative-matrix feature");
        return;
    }
    let cfg = cfg_q4k(); // dims % 256 for Q4_K, all % 8 == 0
    let w = weights(&cfg);
    let bs = 4usize;
    let tbl = [0u32, 1, 2, 3]; // 16 slots >= 9 tokens
    let ids = [3u32, 7, 1, 5, 9, 2, 8, 4, 6]; // 9 tokens -> m=9 -> coop

    let g1 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4K).unwrap();
    let logits_ref = g1.forward_logits(&ids); // m==1 Q4_K GEMV oracle

    let g2 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4K).unwrap();
    let batch = ForwardBatch {
        positions: (0..ids.len() as u32).collect(),
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: ids.len(),
            past_len: 0,
            slots: slots_for(&tbl, bs, ids.len()),
            write_runs: write_runs(&tbl, bs, 0, ids.len()),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let logits_batch = g2.forward_batch(&ids, &batch);
    let diff = max_abs_diff(&logits_batch[0], &logits_ref);
    assert!(
        diff < 1e-3,
        "coop Q4_K (m=9) vs m==1 Q4_K GEMV diverged by {diff}"
    );
}

/// Batched Q4_K GEMV (P22): a multi-token prefill drives add_matmul_m with m>1
/// through matmul_vec_q4k_batch; its last-token logits must match the m==1 Q4_K
/// GEMV (forward_logits). Guards the batch CSE + per-sub-block asymmetric scale+min.
#[test]
fn forward_batch_q4k_matches_forward_logits() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::config::Quant;
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu_quant;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip Q4_K batch test: {e}");
            return;
        }
    };
    let cfg = cfg_q4k(); // dims % 256 for Q4_K
    let w = weights(&cfg);
    let bs = 4usize;
    let tbl = [0u32, 1];
    let ids = [3u32, 7, 1];

    let g1 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4K).unwrap();
    let logits_ref = g1.forward_logits(&ids); // m==1 Q4_K GEMV, the proven path

    let g2 = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &ctx, Quant::Q4K).unwrap();
    let batch = ForwardBatch {
        positions: vec![0, 1, 2],
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: 3, // m=3 -> the Q4_K batch kernel
            past_len: 0,
            slots: slots_for(&tbl, bs, 3),
            write_runs: write_runs(&tbl, bs, 0, 3),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let logits_batch = g2.forward_batch(&ids, &batch);
    let diff = max_abs_diff(&logits_batch[0], &logits_ref);
    assert!(
        diff < 1e-3,
        "batched Q4_K (m=3) vs m==1 Q4_K GEMV diverged by {diff}"
    );
}

/// Attention kernel in the exact forward_batch B=1 prefill regime: q_len=3,
/// past_len=0, ctx=3, q_start=0. Pins whether the kernel itself is wrong there
/// (vs forward_batch's wiring). CPU oracle = attend_sequence math, recomputed.
#[test]
fn attention_prefill_q3_past0() {
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    const ATTENTION_WGSL: &str = include_str!("../src/shaders/wgsl/attention.wgsl");
    let kernel = ComputeKernel::new(&ctx, "attention", ATTENTION_WGSL);

    let (q_len, seqlen, nh, nkv, hd) = (3usize, 3usize, 4usize, 2usize, 4usize);
    let past_len = 0u32;
    let scale = 1.0 / (hd as f32).sqrt();
    let group = nh / nkv;
    let mut seed = 5u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let q: Vec<f32> = (0..q_len * nh * hd).map(|_| rnd()).collect();
    let k_all: Vec<f32> = (0..seqlen * nkv * hd).map(|_| rnd()).collect();
    let v_all: Vec<f32> = (0..seqlen * nkv * hd).map(|_| rnd()).collect();

    // CPU oracle.
    let mut cpu_out = vec![0.0f32; q_len * nh * hd];
    for head in 0..nh {
        let kvh = head / group;
        for r in 0..q_len {
            let last = past_len as usize + r;
            let mut scores = vec![0.0f32; seqlen];
            for j in 0..seqlen {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[(r * nh + head) * hd + d] * k_all[(j * nkv + kvh) * hd + d];
                }
                scores[j] = dot * scale;
            }
            let mx = scores[..=last].iter().copied().fold(-3.4e38f32, f32::max);
            let mut sum = 0.0f32;
            for (j, sc) in scores.iter_mut().enumerate() {
                if j <= last {
                    *sc = (*sc - mx).exp();
                    sum += *sc;
                } else {
                    *sc = 0.0;
                }
            }
            for sc in scores.iter_mut() {
                *sc /= sum;
            }
            for d in 0..hd {
                let mut acc = 0.0f32;
                for j in 0..seqlen {
                    acc += scores[j] * v_all[(j * nkv + kvh) * hd + d];
                }
                cpu_out[(r * nh + head) * hd + d] = acc;
            }
        }
    }

    // attention reads the pool by slot; identity slots (slot[j]==j) over the
    // already-contiguous k_all/v_all reproduce the old contiguous read.
    let qbuf = ctx.storage_init("q", &q);
    let kbuf = ctx.storage_init("key_pool", &k_all);
    let vbuf = ctx.storage_init("value_pool", &v_all);
    let obuf = ctx.storage_zeros("o", q_len * nh * hd);
    let slots: Vec<u32> = (0..seqlen as u32).collect();
    let sbuf = ctx.storage_init("slots", &slots);
    let dims = [
        q_len as u32,
        seqlen as u32,
        nh as u32,
        nkv as u32,
        hd as u32,
        past_len,
        scale.to_bits(),
        group as u32,
        0,
        0,
        0,
        0,
    ];
    let dbuf = ctx.uniform_of("d", &dims);
    kernel.dispatch(
        "attn",
        &[&qbuf, &kbuf, &vbuf, &obuf, &sbuf, &dbuf],
        [nh as u32, q_len as u32, 1],
    );
    let got = ctx.read_f32(&obuf, q_len * nh * hd);
    let diff = max_abs_diff(&got, &cpu_out);
    assert!(diff < 1e-4, "attention q3/past0 diverged by {diff}");
}

/// embed kernel writes to dst_row>0 correctly (the forward_batch multi-row path).
#[test]
fn embed_row_offset_writes_correct_row() {
    use arf_gpu::gpu::ComputeKernel;
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    const EMBED_WGSL: &str = include_str!("../src/shaders/wgsl/embed.wgsl");
    let kernel = ComputeKernel::new(&ctx, "embed", EMBED_WGSL);
    let (vocab, h) = (5usize, 4usize);
    let table: Vec<f32> = (0..vocab * h).map(|i| i as f32).collect(); // row r = [4r..4r+4)
    let tbuf = ctx.storage_init("t", &table);
    let obuf = ctx.storage_zeros("o", 3 * h); // 3 rows
                                              // Gather table row 2 into out row 1 (scale 1.0 in slot 3 = identity).
    let d = ctx.uniform_of("d", &[2u32, h as u32, 1u32, 1.0f32.to_bits()]);
    kernel.dispatch("e", &[&tbuf, &obuf, &d], [(h as u32).div_ceil(256), 1, 1]);
    let got = ctx.read_f32(&obuf, 3 * h);
    // out row 1 = table row 2 = [8,9,10,11]; rows 0,2 stay 0.
    assert_eq!(
        &got[h..2 * h],
        &[8.0, 9.0, 10.0, 11.0],
        "dst_row=1 from src_row=2"
    );
    assert_eq!(&got[0..h], &[0.0; 4]);
}

/// Bisect total: forward_batch with a SINGLE token (total=1) vs forward_logits
/// with one token. If this matches but the 3-token version doesn't, the bug is
/// total>1 (something accumulated across rows), not the per-row wiring.
#[test]
fn forward_batch_one_token() {
    use arf_core::cache::{slots_for, write_runs};
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::weights::build_gpu;
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    let cfg = cfg();
    let w = weights(&cfg);
    let tbl = [0u32, 1];

    let g1 = build_gpu(&cfg, &w, cfg.max_position_embeddings, &ctx).unwrap();
    let r1 = g1.forward_logits(&[3]);

    let g2 = build_gpu(&cfg, &w, cfg.max_position_embeddings, &ctx).unwrap();
    let batch = ForwardBatch {
        positions: vec![0],
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: 1,
            past_len: 0,
            slots: slots_for(&tbl, 4, 1),
            write_runs: write_runs(&tbl, 4, 0, 1),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let r2 = g2.forward_batch(&[3], &batch);
    let diff = max_abs_diff(&r2[0], &r1);
    assert!(
        diff < 1e-3,
        "forward_batch total=1 vs forward_logits diverged by {diff}"
    );
}

/// SUSPECT PROBE (V weight share): a Gemma-4 GLOBAL layer ships no attn_v.weight; both CPU and
/// GPU alias `v_proj = k_proj.clone()`. The worry is that on GPU the V matmul still
/// has to populate a DISTINCT `scratch.v` (and a distinct values pool) from the
/// aliased weight — i.e. that the alias means "K and V use the same WEIGHTS", NOT
/// "V reads the K buffer". This probe isolates exactly that:
///
///   (1) GPU(alias)   vs CPU(alias)            — the suspect path matches the oracle.
///   (2) GPU(alias)   vs GPU(explicit equal V) — alias == independent-but-equal V.
///   (3) CPU(alias)   vs CPU(explicit equal V) — oracle is self-consistent.
///
/// Config is plain Causal with softcap=None, no layer_scalar, qk_norm off — so the
/// CPU Llama oracle is valid and ONLY the k_eq_v alias path differs between arms.
#[cfg(test)]
fn cfg_keqv() -> ModelConfig {
    // Field-for-field identical to `cfg()` — this alias exists so the k==v tests read as their own
    // fixture, not because the config differs. It used to be 22 duplicated fields.
    cfg()
}

/// LCG-filled weights for `cfg_keqv`. When `emit_v` is false, NO `v_proj.weight` is
/// emitted (the alias must supply V from k_proj); when true, an explicit
/// `v_proj.weight` is emitted whose bytes EQUAL `k_proj.weight`. Same seed in both
/// so k_proj (and the explicit v_proj) are bit-identical.
#[cfg(test)]
fn weights_keqv(cfg: &ModelConfig, emit_v: bool) -> Weights {
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut seed = 23u64;
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
        // k_proj generated FIRST so the explicit v_proj (next) gets the very next
        // LCG draws — make them equal by cloning the just-generated k tensor.
        let k = gen(vec![kv, h]);
        if emit_v {
            // Explicit V == K bit-for-bit (clone the tensor, don't re-draw).
            t.insert(format!("{p}.self_attn.v_proj.weight"), k.clone());
        }
        t.insert(format!("{p}.self_attn.k_proj.weight"), k);
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
fn gpu_keqv_alias_matches_cpu_and_explicit_v() {
    use arf_gpu::weights::build_gpu;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping k_eq_v probe: {e}");
            return;
        }
    };
    eprintln!("running k_eq_v probe on {}", gctx.info());

    let mp = 64usize;

    // ARM A: k_eq_v=true, V aliases K (no v_proj.weight on disk).
    let cfg_alias = cfg_keqv();
    let w_alias = weights_keqv(&cfg_alias, false);

    // ARM B: k_eq_v=false, explicit v_proj.weight == k_proj.weight.
    let cfg_expl = ModelConfig {
        value_norm: false,
        ..cfg_keqv()
    };
    let w_expl = weights_keqv(&cfg_expl, true);

    let cpu_alias = build_on(&cfg_alias, &w_alias, mp, &Device::Cpu).unwrap();
    let gpu_alias = build_gpu(&cfg_alias, &w_alias, mp, &gctx).unwrap();
    let cpu_expl = build_on(&cfg_expl, &w_expl, mp, &Device::Cpu).unwrap();
    let gpu_expl = build_gpu(&cfg_expl, &w_expl, mp, &gctx).unwrap();

    let tokens = [7u32, 2, 9, 1, 4, 3];

    let l_cpu_alias = last_logits(&cpu_alias, &tokens);
    let l_gpu_alias = gpu_alias.forward_logits(&tokens);
    let l_cpu_expl = last_logits(&cpu_expl, &tokens);
    let l_gpu_expl = gpu_expl.forward_logits(&tokens);

    // (1) suspect path: GPU alias must match the CPU alias oracle.
    let d1 = max_abs_diff(&l_gpu_alias, &l_cpu_alias);
    // (2) alias path == independent-but-equal V (the headline claim).
    let d2 = max_abs_diff(&l_gpu_alias, &l_gpu_expl);
    // (3) oracle self-consistency.
    let d3 = max_abs_diff(&l_cpu_alias, &l_cpu_expl);
    // (4) GPU explicit vs CPU explicit (control: non-alias arm parity).
    let d4 = max_abs_diff(&l_gpu_expl, &l_cpu_expl);

    let first_div = l_gpu_alias
        .iter()
        .zip(&l_cpu_alias)
        .enumerate()
        .find(|(_, (a, b))| (**a - **b).abs() >= 1e-3)
        .map(|(i, (a, b))| format!("logit[{i}] gpu={a} cpu={b}"))
        .unwrap_or_else(|| "none".into());

    eprintln!(
        "k_eq_v probe: d1(gpu_alias vs cpu_alias)={d1:e}  d2(gpu_alias vs gpu_explicitV)={d2:e}  d3(cpu_alias vs cpu_explicit)={d3:e}  d4(gpu_expl vs cpu_expl)={d4:e}  first_div={first_div}"
    );

    assert!(
        d1 < 1e-3,
        "k_eq_v: GPU alias diverged from CPU oracle by {d1} (first: {first_div})"
    );
    assert!(
        d2 < 1e-3,
        "k_eq_v: GPU alias diverged from GPU explicit-equal-V by {d2}"
    );
    assert!(
        d3 < 1e-3,
        "k_eq_v: CPU alias diverged from CPU explicit-equal-V by {d3}"
    );
    assert!(
        d4 < 1e-3,
        "k_eq_v: GPU explicit diverged from CPU explicit by {d4}"
    );
}

/// Pack the `rope_qk.wgsl` Dims uniform:
/// `{ tokens, q_heads, k_heads, head_dim, rotary_dim, _p0, _p1, _p2 }`.
fn rope_qk_dims(
    tokens: u32,
    q_heads: u32,
    k_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
) -> [u32; 8] {
    [tokens, q_heads, k_heads, head_dim, rotary_dim, 0, 0, 0]
}

/// SUSPECT PROBE — global-layer RoPE table built via `Rope::with_theta_rotary`.
///
/// The Gemma-4 global layers build their RoPE table with `with_theta_rotary`
/// (own head_dim, partial rotary) on BOTH sides:
///   CPU  arf-core/src/model/weights.rs:654-667
///   GPU  arf-gpu/src/weights.rs:705-721
/// and decode reads it through `rope_qk.wgsl` with `rot_half = rotary_dim/2`.
///
/// This probe isolates that table + that kernel three ways:
///   1. FULL-rotary equivalence: `with_theta_rotary(maxpos,θ,hd,hd)` must produce
///      tables BIT-EQUAL to `with_theta(cfg_hd,maxpos,θ)` (the doc's claim that
///      full rotary reduces to the plain table).
///   2. FULL-rotary GPU kernel: dispatch `rope_qk.wgsl` with the with_theta_rotary
///      tables and compare q/k to the CPU `Rope::apply` oracle at < 1e-5.
///   3. PARTIAL-rotary GPU kernel: head_dim 8, rotary_dim 4 — verify the tail dims
///      4..8 pass through UNTOUCHED and the rotated dims 0..4 match CPU.
#[test]
fn global_rope_with_theta_rotary_gpu_matches_cpu() {
    use arf_core::model::rope::Rope;
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping global-rope with_theta_rotary parity: {e}");
            return;
        }
    };
    const ROPE_QK_WGSL: &str = include_str!("../src/shaders/wgsl/rope_qk.wgsl");
    let kernel = ComputeKernel::new(&ctx, "rope_qk", ROPE_QK_WGSL);

    let theta = 1_000_000.0f64; // Gemma global θ = 1e6
    let max_pos = 64usize;
    let positions = vec![0u32, 5u32, 12u32];
    let tokens = positions.len();

    let mut seed = 0xC0FFEEu64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };

    // ---- Case 1 + 2: FULL rotary, head_dim 8 (half = 4) ---------------------
    // global_head_dim 8 != base head_dim 4 forces the with_theta_rotary path; with
    // rotary_dim == head_dim it must equal the plain with_theta table bit-for-bit.
    let head_dim = 8usize;
    let rope_rot = Rope::with_theta_rotary(max_pos, theta, head_dim, head_dim);

    let mut cfg_hd8 = ModelConfig::llama_3_2_1b();
    cfg_hd8.head_dim = head_dim;
    cfg_hd8.rope_theta = theta;
    cfg_hd8.rope_scaling = RopeScaling::None;
    let rope_plain = Rope::with_theta(&cfg_hd8, max_pos, theta);

    // (1) bit-equal tables (full rotary == plain).
    assert_eq!(
        rope_rot.cos_table(),
        rope_plain.cos_table(),
        "with_theta_rotary(full) cos table must equal with_theta cos table"
    );
    assert_eq!(
        rope_rot.sin_table(),
        rope_plain.sin_table(),
        "with_theta_rotary(full) sin table must equal with_theta sin table"
    );

    // (2) Drive rope_qk.wgsl with the with_theta_rotary tables. global_kv_heads 1,
    // q_heads 4 (Gemma-12B global geometry: kv_heads collapse to 1).
    let q_heads = 4usize;
    let k_heads = 1usize;
    let q: Vec<f32> = (0..tokens * q_heads * head_dim).map(|_| rand()).collect();
    let k: Vec<f32> = (0..tokens * k_heads * head_dim).map(|_| rand()).collect();

    // CPU oracle: apply RoPE independently to q and k.
    let q_cpu = rope_rot.apply(
        &Tensor::from_vec(q.clone(), vec![tokens, q_heads, head_dim]),
        &positions,
    );
    let k_cpu = rope_rot.apply(
        &Tensor::from_vec(k.clone(), vec![tokens, k_heads, head_dim]),
        &positions,
    );

    let q_buf = ctx.storage_init("q", &q);
    let k_buf = ctx.storage_init("k", &k);
    let cos_buf = ctx.storage_init("cos", rope_rot.cos_table());
    let sin_buf = ctx.storage_init("sin", rope_rot.sin_table());
    let pos_buf = ctx.storage_init("positions", &positions);
    let dims = ctx.uniform_of(
        "dims",
        &rope_qk_dims(
            tokens as u32,
            q_heads as u32,
            k_heads as u32,
            head_dim as u32,
            head_dim as u32,
        ),
    );
    let q_total = (tokens * q_heads * head_dim) as u32;
    let k_total = (tokens * k_heads * head_dim) as u32;
    let groups = (q_total + k_total).div_ceil(256);
    kernel.dispatch(
        "rope_qk-full",
        &[&q_buf, &k_buf, &cos_buf, &sin_buf, &pos_buf, &dims],
        [groups, 1, 1],
    );
    let q_got = ctx.read_f32(&q_buf, q.len());
    let k_got = ctx.read_f32(&k_buf, k.len());
    let dq = max_abs_diff(&q_got, q_cpu.as_slice());
    let dk = max_abs_diff(&k_got, k_cpu.as_slice());
    eprintln!("[full-rotary hd8] q max_abs_diff={dq}  k max_abs_diff={dk}");
    assert!(dq < 1e-5, "FULL-rotary q GPU vs CPU diverged by {dq}");
    assert!(dk < 1e-5, "FULL-rotary k GPU vs CPU diverged by {dk}");

    // ---- Case 3: PARTIAL rotary, head_dim 8, rotary_dim 4 (rot_half = 2) -----
    // Gemma-4 global shape in miniature: only dims 0..4 rotate, dims 4..8 pass
    // through. Verifies rope_qk.wgsl's `i >= rot_half` guard + table stride
    // rot_half against the CPU `apply` (which uses half = rotary_dim/2).
    let rotary_dim = 4usize;
    let rope_part = Rope::with_theta_rotary(max_pos, theta, head_dim, rotary_dim);

    let qp: Vec<f32> = (0..tokens * q_heads * head_dim).map(|_| rand()).collect();
    let kp: Vec<f32> = (0..tokens * k_heads * head_dim).map(|_| rand()).collect();

    let qp_cpu = rope_part.apply(
        &Tensor::from_vec(qp.clone(), vec![tokens, q_heads, head_dim]),
        &positions,
    );
    let kp_cpu = rope_part.apply(
        &Tensor::from_vec(kp.clone(), vec![tokens, k_heads, head_dim]),
        &positions,
    );

    let qp_buf = ctx.storage_init("qp", &qp);
    let kp_buf = ctx.storage_init("kp", &kp);
    let cosp_buf = ctx.storage_init("cosp", rope_part.cos_table());
    let sinp_buf = ctx.storage_init("sinp", rope_part.sin_table());
    let posp_buf = ctx.storage_init("positionsp", &positions);
    let dimsp = ctx.uniform_of(
        "dimsp",
        &rope_qk_dims(
            tokens as u32,
            q_heads as u32,
            k_heads as u32,
            head_dim as u32,
            rotary_dim as u32,
        ),
    );
    kernel.dispatch(
        "rope_qk-partial",
        &[&qp_buf, &kp_buf, &cosp_buf, &sinp_buf, &posp_buf, &dimsp],
        [groups, 1, 1],
    );
    let qp_got = ctx.read_f32(&qp_buf, qp.len());
    let kp_got = ctx.read_f32(&kp_buf, kp.len());
    let dqp = max_abs_diff(&qp_got, qp_cpu.as_slice());
    let dkp = max_abs_diff(&kp_got, kp_cpu.as_slice());
    eprintln!("[partial-rotary hd8 rot4] q max_abs_diff={dqp}  k max_abs_diff={dkp}");

    // Tail dims [rotary_dim..head_dim) must be byte-for-byte the original input.
    for t in 0..tokens {
        for h in 0..q_heads {
            let base = (t * q_heads + h) * head_dim;
            assert_eq!(
                &qp_got[base + rotary_dim..base + head_dim],
                &qp[base + rotary_dim..base + head_dim],
                "partial-rotary q tail must pass through untouched (t={t} h={h})"
            );
        }
    }
    assert!(dqp < 1e-5, "PARTIAL-rotary q GPU vs CPU diverged by {dqp}");
    assert!(dkp < 1e-5, "PARTIAL-rotary k GPU vs CPU diverged by {dkp}");
}

// ---------------------------------------------------------------------------
// SUSPECT PROBE: Gemma-4 `layer_scalar` (per-layer learned output multiply).
//
// GPU applies `hidden *= layer_scalar` at the END of every decoder layer
// (decode.rs:693, batch.rs equivalent, scale_inplace.wgsl). CPU `Llama::forward`
// (llama.rs:81-85) applies NOTHING per-layer — `layer_scalar` is unreferenced in
// llama.rs/layer.rs. So the CPU model is NOT a valid oracle when layer_scalar is
// present, exactly as the suspect describes.
//
// Part 1 (ls_scale_inplace_gpu_matches_cpu): verifies the multiply KERNEL itself
//   is faithful — got == x*s, essentially exact.
// Part 2 (gemma_layer_scalar_divergence): the smallest END-TO-END check that
//   implicates this path. Build a whole-model probe (cfg_gemma: softcap None) whose
//   weights carry a non-1.0 layer_scalar; GPU applies it, CPU ignores it, so the
//   prefill logits MUST diverge far beyond the 1e-3 logit bar. CONTROL arm
//   (layer_scalar=1.0 == absent) proves the machinery: GPU and CPU agree there.
// ---------------------------------------------------------------------------

#[test]
fn ls_scale_inplace_gpu_matches_cpu() {
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping scale_inplace parity: {e}");
            return;
        }
    };
    const SCALE_WGSL: &str = include_str!("../src/shaders/wgsl/scale_inplace.wgsl");
    let kernel = ComputeKernel::new(&ctx, "scale_inplace", SCALE_WGSL);

    // 16000 is NOT a multiple of the 256 workgroup, so the `i < n` guard runs too.
    let n = 16_000usize;
    let scale = 0.6173f32; // arbitrary non-1.0, as a Gemma layer_scalar would be
    let mut seed = 4242u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let x: Vec<f32> = (0..n).map(|_| rand()).collect();

    // CPU oracle: plain elementwise multiply.
    let expected: Vec<f32> = x.iter().map(|v| v * scale).collect();

    // GPU: in-place x[i] *= scale. dims = [n, scale_bits, 0, 0].
    let x_buf = ctx.storage_init("x", &x);
    let dims = ctx.uniform_of("dims", &[n as u32, scale.to_bits(), 0, 0]);
    let groups = (n as u32).div_ceil(256);
    kernel.dispatch("scale_inplace", &[&x_buf, &dims], [groups, 1, 1]);
    let got = ctx.read_f32(&x_buf, n);

    // A single f32 multiply is bit-identical on GPU and CPU IEEE-754; 1e-6 leaves
    // headroom for any rounding-mode corner.
    let diff = max_abs_diff(&got, &expected);
    eprintln!("scale_inplace max_abs_diff = {diff:e}");
    assert!(diff < 1e-6, "scale_inplace GPU vs CPU diverged by {diff}");
}

/// `gemma_weights(cfg)` plus a per-layer `layer_scalar` tensor [1] = `scalar`.
/// Re-emits the exact tensors gemma_weights() builds (same seed 11) so the only
/// difference from the existing green Gemma probe is the added layer_scalar.
#[cfg(test)]
fn gemma_weights_with_layer_scalar(cfg: &ModelConfig, scalar: f32) -> Weights {
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let mut seed = 11u64;
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
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            t.insert(format!("{p}.{norm}.weight"), gen(vec![h]));
        }
        t.insert(format!("{p}.self_attn.q_proj.weight"), gen(vec![q, h]));
        t.insert(format!("{p}.self_attn.k_proj.weight"), gen(vec![kv, h]));
        t.insert(format!("{p}.self_attn.v_proj.weight"), gen(vec![kv, h]));
        t.insert(format!("{p}.self_attn.o_proj.weight"), gen(vec![h, q]));
        t.insert(
            format!("{p}.self_attn.q_norm.weight"),
            gen(vec![cfg.head_dim]),
        );
        t.insert(
            format!("{p}.self_attn.k_norm.weight"),
            gen(vec![cfg.head_dim]),
        );
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
        // The suspect tensor: per-layer learned output scalar [1].
        t.insert(
            format!("{p}.layer_scalar"),
            Tensor::from_vec(vec![scalar], vec![1]),
        );
    }
    t.insert("model.norm.weight".into(), gen(vec![h]));
    Weights::from_map(t)
}

#[test]
fn gemma_layer_scalar_divergence() {
    use arf_gpu::weights::build_gpu;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping Gemma layer_scalar probe: {e}");
            return;
        }
    };
    eprintln!("running Gemma layer_scalar probe on {}", gctx.info());

    let cfg = cfg_gemma(); // softcap None, qk_norm on, four-norm Gemma block
    let tokens = [7u32, 2, 9, 1, 4, 3];

    // CONTROL: layer_scalar == 1.0 (a no-op multiply == absent). GPU and CPU must
    // agree — proves the probe machinery is sound (matches the existing green path).
    let w1 = gemma_weights_with_layer_scalar(&cfg, 1.0);
    let cpu1 = build_on(&cfg, &w1, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let gpu1 = build_gpu(&cfg, &w1, cfg.max_position_embeddings, &gctx).unwrap();
    let control_diff = max_abs_diff(&gpu1.forward_logits(&tokens), &last_logits(&cpu1, &tokens));
    eprintln!("CONTROL (layer_scalar=1.0) GPU vs CPU max_abs_diff = {control_diff:e}");

    // SUSPECT: layer_scalar == 0.5 across all 3 layers. GPU applies it, CPU does
    // not (llama.rs:81-85 ignores it). The hidden state compounds through later
    // norms nonlinearly, so the logits MUST diverge well beyond the 1e-3 bar.
    let w2 = gemma_weights_with_layer_scalar(&cfg, 0.5);
    let cpu2 = build_on(&cfg, &w2, cfg.max_position_embeddings, &Device::Cpu).unwrap();
    let gpu2 = build_gpu(&cfg, &w2, cfg.max_position_embeddings, &gctx).unwrap();
    let gpu2_logits = gpu2.forward_logits(&tokens);
    let cpu2_logits = last_logits(&cpu2, &tokens);
    let suspect_diff = max_abs_diff(&gpu2_logits, &cpu2_logits);
    eprintln!("SUSPECT (layer_scalar=0.5) GPU vs CPU max_abs_diff = {suspect_diff:e}");

    let first_div = gpu2_logits
        .iter()
        .zip(&cpu2_logits)
        .enumerate()
        .find(|(_, (a, b))| (**a - **b).abs() >= 1e-3)
        .map(|(i, (a, b))| format!("logit[{i}] gpu={a} cpu={b}"))
        .unwrap_or_else(|| "none".into());
    eprintln!("first divergent logit (suspect): {first_div}");

    assert!(
        control_diff < 1e-3,
        "control (layer_scalar=1.0) should match the CPU oracle; got {control_diff}"
    );
    assert!(
        suspect_diff > 1e-3,
        "EXPECTED divergence: CPU ignores layer_scalar but logits did NOT diverge ({suspect_diff})"
    );
}

/// SUSPECT (gemma-4 incoherent output): the bf16 embed gather (`embed_bf16`)
/// must reproduce the CPU embedding row, including the √hidden Gemma scale.
///
/// This isolates the embed_bf16 path end-to-end at the kernel level:
///   * GPU build path (weights.rs:412-413): `storage_init_bf16(name,
///     &bf16_matrix(...).to_f32())` — a bf16→f32→bf16 round-trip (the f32 from
///     `to_f32()` is re-packed by `pack_bf16`, which TRUNCATES; the CPU
///     `from_f32` ROUNDS-to-nearest-even, so re-packing an already-bf16 f32 must
///     be a no-op for this to hold).
///   * decode.rs:244-260 dispatches `embed_bf16` with
///     edims = [id, hidden, 0, embed_scale.to_bits()].
///   * shader embed_bf16.wgsl:27-29 widens bf16→f32 (<<16) and multiplies by
///     scale_bits.
///   * CPU (llama.rs:73-78): `embed.forward` = `Bf16Matrix::gather` (widen <<16),
///     then `*= embedding_scale` (Gemma-4 = √5376).
///
/// Gemma-4-shaped: hidden = 5376 (the real embed width), √hidden scale, a vocab
/// subset large enough to gather a non-trivial row id.
#[test]
fn embed_bf16_gpu_matches_cpu_gemma_scale() {
    use arf_core::tensor::Bf16Matrix;
    use arf_gpu::gpu::ComputeKernel;

    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping embed_bf16 probe: {e}");
            return;
        }
    };
    eprintln!("running embed_bf16 probe on {}", ctx.info());

    const WGSL: &str = include_str!("../src/shaders/wgsl/embed_bf16.wgsl");
    let kernel = ComputeKernel::new(&ctx, "embed_bf16", WGSL);

    // Gemma-4 embed geometry: wide hidden, √hidden scale.
    let hidden = 5376usize;
    let vocab = 8usize; // small subset; row equality is per-row so any vocab works
    let scale = (hidden as f32).sqrt(); // Gemma-4 embedding_scale = √5376

    // LCG-filled f32 source matrix [vocab, hidden] (the engine-standard rng).
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let src: Vec<f32> = (0..vocab * hidden).map(|_| rand()).collect();

    // Replicate the GPU BUILD path's bf16 round-trip exactly:
    //   weights.rs: storage_init_bf16(name, &bf16_matrix(...).to_f32())
    // i.e. f32 src -> CPU round-to-nearest bf16 -> widen back to f32 -> upload
    // (pack_bf16 truncates that f32; must be a no-op because low 16 bits are 0).
    let cpu_bf16 = Bf16Matrix::from_f32(&src, vocab, hidden);
    let rounded_f32 = cpu_bf16.to_f32(); // the exact bytes weights.rs feeds the GPU
    let table = ctx.storage_init_bf16("embed_table", &rounded_f32);

    // Output is one gathered, scaled row.
    let out = ctx.storage_zeros("embed_out", hidden);

    // Sweep EVERY row (incl. 0 and the last) so row-equality is verified across
    // the whole table, not one lucky id. Track the worst diff and its location.
    let mut diff = 0.0f32;
    let mut worst_first_div = "none (bit-exact)".to_string();
    let mut worst_row = 0u32;
    for src_row in 0..vocab as u32 {
        let dims = ctx.uniform_of("edims", &[src_row, hidden as u32, 0u32, scale.to_bits()]);
        // Bind order MUST match WGSL @binding: 0=table, 1=out, 2=dims.
        kernel.dispatch(
            "embed_bf16",
            &[&table, &out, &dims],
            [(hidden as u32).div_ceil(256), 1, 1],
        );
        let got = ctx.read_f32(&out, hidden);

        // CPU oracle = exactly what llama.rs forward() produces for this row:
        //   Bf16Matrix::gather([id]) (widen <<16), then *= embedding_scale.
        let cpu_row = cpu_bf16.gather(&[src_row]);
        let expected: Vec<f32> = cpu_row.as_slice().iter().map(|&v| v * scale).collect();

        let d = max_abs_diff(&got, &expected);
        if d >= diff {
            diff = d;
            worst_row = src_row;
            worst_first_div = got
                .iter()
                .zip(&expected)
                .enumerate()
                .find(|(_, (a, b))| (**a - **b).abs() > 0.0)
                .map(|(i, (a, b))| {
                    format!(
                        "elem[{i}] gpu={a} cpu={b} (gpu_bits={:#x} cpu_bits={:#x})",
                        a.to_bits(),
                        b.to_bits()
                    )
                })
                .unwrap_or_else(|| "none (bit-exact)".into());
        }
    }
    let src_row = worst_row;
    let first_div = worst_first_div;

    eprintln!(
        "embed_bf16 probe: hidden={hidden} scale=√{hidden}={scale} swept rows 0..{vocab} (worst row={src_row})  max_abs_diff={diff:e}  first_div={first_div}"
    );

    // bf16 round-trip + scale: bar is bf16 precision (~1e-2 of magnitude). The
    // build path means it should actually be bit-exact, but we assert the stated
    // <1e-2 bar so a real round-trip/scale mismatch trips it.
    assert!(
        diff < 1e-2,
        "embed_bf16 GPU vs CPU diverged by {diff} (first: {first_div})"
    );
}

/// CONTROL for the four failing CPU/GPU quant-parity tests: same comparison with NO
/// quantization at all (bf16 both sides).
///
/// If this passes while the Q4/Q4K/Q4KS/Q3K versions fail, the divergence is in the
/// quantized weight path. If it FAILS too, quantization is exonerated and the fault is in
/// how the two full models compose — a norm, RoPE, or the attention assembly.
#[test]
fn gpu_bf16_logits_match_cpu_bf16_control() {
    use arf_gpu::weights::build_gpu;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping bf16 control: {e}");
            return;
        }
    };

    let cfg = cfg_q4();
    let w = weights(&cfg);
    let gpu = build_gpu(&cfg, &w, cfg.max_position_embeddings, &gctx).unwrap();
    let cpu = build_on(&cfg, &w, cfg.max_position_embeddings, &Device::Cpu).unwrap();

    let tokens = vec![5u32, 2, 9, 1];
    let cpu_logits = last_logits(&cpu, &tokens);
    let gpu_logits = gpu.forward_logits(&tokens);
    assert_eq!(cpu_logits.len(), gpu_logits.len());
    let worst = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .enumerate()
        .map(|(i, (c, g))| ((c - g).abs() / c.abs().max(1.0), i, *c, *g))
        .fold((0.0f32, 0, 0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
    eprintln!(
        "[bf16 control] worst relative gap {:.4}% at logit {} (cpu {} vs gpu {})",
        worst.0 * 100.0,
        worst.1,
        worst.2,
        worst.3
    );
    assert!(
        worst.0 <= 1e-2,
        "bf16 (unquantized) already diverges: {:.4}% at logit {}",
        worst.0 * 100.0,
        worst.1
    );
}

/// PROOF of the tied-lm_head asymmetry: with `tie_word_embeddings: false` the two paths
/// build the SAME lm_head, and the parity that fails for the tied config should hold.
#[test]
fn q4_parity_holds_when_lm_head_is_not_tied() {
    use arf_core::config::Quant;
    use arf_core::model::weights::build_quant;
    use arf_gpu::weights::build_gpu_quant;

    let gctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let cfg = ModelConfig {
        tie_word_embeddings: false,
        ..cfg_q4()
    };
    let w = weights_untied(&cfg);

    let tokens = vec![5u32, 2, 9, 1];
    let cpu = last_logits(
        &build_quant(
            &cfg,
            &w,
            cfg.max_position_embeddings,
            &Device::Cpu,
            Quant::Q4,
        )
        .unwrap(),
        &tokens,
    );
    let gpu = build_gpu_quant(&cfg, &w, cfg.max_position_embeddings, &gctx, Quant::Q4)
        .unwrap()
        .forward_logits(&tokens);
    let worst = cpu
        .iter()
        .zip(&gpu)
        .map(|(c, g)| (c - g).abs() / c.abs().max(1.0))
        .fold(0.0f32, f32::max);
    eprintln!("[untied lm_head] worst relative gap {:.4}%", worst * 100.0);
    assert!(
        worst <= 1e-2,
        "untied Q4 parity should hold; worst gap {:.4}%",
        worst * 100.0
    );
}

/// The `cfg_*` fixtures are spelled as `..cfg()` overrides. This pins what each one actually
/// changes, so a future edit to `cfg()` that silently alters a derived fixture fails here rather
/// than in whichever parity test happened to depend on the field.
#[test]
fn cfg_fixtures_override_only_what_they_claim() {
    let base = cfg();

    let q4 = cfg_q4();
    assert_eq!(
        (q4.hidden_size, q4.intermediate_size, q4.head_dim),
        (32, 64, 8)
    );
    assert_eq!(q4.vocab_size, base.vocab_size);
    assert_eq!(q4.num_layers, base.num_layers);
    assert_eq!(q4.num_attention_heads, base.num_attention_heads);
    assert_eq!(q4.num_kv_heads, base.num_kv_heads);

    let q4k = cfg_q4k();
    assert_eq!(
        (
            q4k.vocab_size,
            q4k.hidden_size,
            q4k.intermediate_size,
            q4k.head_dim
        ),
        (64, 256, 512, 64)
    );
    assert_eq!(q4k.num_layers, base.num_layers);
    // Q4_K quantizes in 256-weight super-blocks; every quantized dim must divide 256.
    assert_eq!(q4k.hidden_size % 256, 0);
    assert_eq!(q4k.intermediate_size % 256, 0);

    // cfg_keqv is a deliberate alias, not a variant.
    let keqv = cfg_keqv();
    assert_eq!(keqv.hidden_size, base.hidden_size);
    assert_eq!(keqv.head_dim, base.head_dim);
    assert!(!keqv.qk_norm);

    // cfg_qknorm changes exactly one thing.
    let qk = cfg_qknorm();
    assert!(qk.qk_norm && !base.qk_norm);
    assert_eq!(qk.hidden_size, base.hidden_size);
}
