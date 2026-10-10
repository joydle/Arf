//! GPU weight loading for the resident-decode path: upload every weight, the
//! RoPE tables, and the reused scratch into resident GPU buffers, quantizing the
//! matmul weights to the requested [`Quant`]. The backend-agnostic CPU loader and
//! the [`Weights`] store live in `arf-core`; this module is the GPU half of
//! the old `weights.rs`, split out when the wgpu backend became its own crate.

use std::path::Path;

use arf_core::config::{AttnKind, KvQuant, MlpKind, ModelConfig, NormStyle, Quant};

/// L139 — does layer `i` need a K/V cache?
///
/// The `layer_filter_cb` lesson from llama.cpp (llama-model.cpp:2284-2301): on a HYBRID arch the
/// recurrent (gated-delta-net) layers have no attention, so they have no K/V to cache — their
/// state is the island's resident conv/ssm pair, O(1) per token. Only the 1-in-`attn_every`
/// attention layers need a pool. For every non-hybrid arch this is unconditionally true, so the
/// dense/MoE paths are byte-for-byte unaffected.
///
/// Mirrors `ssm_qwen35::is_recurrent` inverted (and `weights.rs`'s own `is_gdn`): layer `i` is
/// recurrent iff `(i+1) % attn_every != 0`.
/// A model with recurrent layers (no KV pool) beside its attention layers — Qwen3.8's GDN trunk.
pub(crate) fn model_is_hybrid(cfg: &ModelConfig) -> bool {
    (0..cfg.num_layers).any(|i| !kv_layer_needed(&cfg.attn, i))
}

fn kv_layer_needed(attn: &AttnKind, i: usize) -> bool {
    // ARF_NO_KV_LAYER_FILTER=1 — allocate a full pool for EVERY layer (the pre-L139
    // behaviour). Diagnostic: isolates whether a failure is caused by the 16-byte placeholders
    // on recurrent layers. Costs the 75% of dead pool the filter exists to reclaim.
    if std::env::var_os("ARF_NO_KV_LAYER_FILTER").is_some() {
        return true;
    }
    match attn {
        AttnKind::HybridSsmAttn { attn_every, .. } => (i + 1).is_multiple_of((*attn_every).max(1)),
        _ => true,
    }
}
/// Bytes of f32 KV pool that `blocks` blocks cost for this model — K and V, 16 slots a block, only
/// the layers that actually cache (a hybrid model's recurrent layers do not). The same arithmetic
/// as `wired_memory_guard`, exposed so a server can choose a context default that fits BEFORE it
/// loads anything.
/// Group-64 weights loaded this process (2026-09-25) — nonzero means every matmul path must
/// route them (`Q4ksMtl::g64`); the serve binary disables the paths that cannot.
pub static G64_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Of those, the ones held at 3 bits (`arf_core::model::weights::g64_pack3`).
pub static G64_3BIT_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `ARF_MEGAKERNEL` as the decode paths must read it: OFF while any group-64 weight is loaded.
/// The m=1 megakernel (and the single-queue embed/collect that ride on it) reads Q4_K_S buffers,
/// which a group-64 weight has only as placeholders; single-token decode then goes through the
/// batched record, whose group-64 matmul pads to an 8-row lane.
pub fn megakernel_on() -> bool {
    std::env::var_os("ARF_MEGAKERNEL").is_some()
        && G64_COUNT.load(std::sync::atomic::Ordering::Relaxed) == 0
}

pub fn kv_pool_bytes(cfg: &ModelConfig, blocks: usize) -> u64 {
    let kv_layers = (0..cfg.num_layers)
        .filter(|&i| kv_layer_needed(&cfg.attn, i))
        .count() as u64;
    let slots = blocks as u64 * 16;
    if crate::kv_q8_for(cfg) {
        // int8 elements + one f32 scale per (slot, kv head), for K and V
        return 2
            * kv_layers
            * slots
            * ((cfg.num_kv_heads * cfg.head_dim) as u64 + 4 * cfg.num_kv_heads as u64);
    }
    2 * kv_layers * slots * (cfg.num_kv_heads * cfg.head_dim) as u64 * 4
}
use arf_core::error::{ArfError, Result};
use arf_core::model::weights::{read_weights_lazy, Weights};
use arf_core::model::Rope;

use crate::gpu::{
    ComputeKernel, GpuGdn, GpuKernels, GpuKvPool, GpuLayer, GpuMatWeight, GpuMlp, GpuModel,
    GpuScratch,
};

// ─── Pure helpers (module-level, no GPU context required) ────────────────────

/// Concatenate three bf16 weight row-buffers (q rows, then k, then v) for
/// fused QKV. Each element is a raw `u16` bf16 value; rows from q come first,
/// then k, then v — matching the fused weight layout `[q_rows + 2*kv_rows, in_]`.
///
/// NOTE: The `up_qkv` closure in `build_gpu_quant` intentionally does NOT call
/// this helper for the bf16 arm — that arm works at `f32` (concat three f32
/// row-blocks then `storage_init_bf16` packs once, which is bit-identical to
/// packing each then concatenating, since bf16 packing is per-element with no
/// cross-row state). This function documents and tests the u16-level row-stacking
/// invariant separately; use it when you already have packed `u16` bf16 buffers.
pub fn qkv_concat_bf16(q: &[u16], k: &[u16], v: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(q.len() + k.len() + v.len());
    out.extend_from_slice(q);
    out.extend_from_slice(k);
    out.extend_from_slice(v);
    out
}

/// Concatenate three Q4_0 matrices' packed sub-buffers (q rows, then k, then v).
/// Returns `(codes, scales)` — both are plain slices (codes: u32, scales: u16 bf16).
pub fn qkv_concat_q4(
    q: &arf_core::tensor::Q4Matrix,
    k: &arf_core::tensor::Q4Matrix,
    v: &arf_core::tensor::Q4Matrix,
) -> (Vec<u32>, Vec<u16>) {
    let codes = [q.codes(), k.codes(), v.codes()].concat();
    let scales = [q.scales(), k.scales(), v.scales()].concat();
    (codes, scales)
}

/// Concatenate three Q4_K-lite matrices' packed sub-buffers (q rows, then k, then v).
/// Returns `(codes, scales, mins)` — codes u32, scales/mins u16 bf16.
pub fn qkv_concat_q4k(
    q: &arf_core::tensor::Q4KMatrix,
    k: &arf_core::tensor::Q4KMatrix,
    v: &arf_core::tensor::Q4KMatrix,
) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    let codes = [q.codes(), k.codes(), v.codes()].concat();
    let scales = [q.scales(), k.scales(), v.scales()].concat();
    let mins = [q.mins(), k.mins(), v.mins()].concat();
    (codes, scales, mins)
}

/// Interleave a super-block's `d` and `dmin` scale arrays into one `[d, dmin]`
/// stream — the layout the GPU Q-K kernels expect. Shared by the Q4_K_S and
/// Q3_K qkv-concat paths (both store scales as parallel `d`/`dmin` slices).
fn interleave_d_dmin(d: &[f32], dmin: &[f32]) -> Vec<f32> {
    let mut dd = Vec::with_capacity(d.len() * 2);
    for (dv, dmv) in d.iter().zip(dmin) {
        dd.push(*dv);
        dd.push(*dmv);
    }
    dd
}

/// Concatenate three Q4_K_S matrices' packed sub-buffers (q rows, then k, then v).
/// Returns `(codes, scales, mins, dd)` — dd interleaved `[d, dmin]` per super-block.
pub fn qkv_concat_q4ks(
    q: &arf_core::tensor::Q4KSMatrix,
    k: &arf_core::tensor::Q4KSMatrix,
    v: &arf_core::tensor::Q4KSMatrix,
) -> (Vec<u32>, Vec<u8>, Vec<i8>, Vec<f32>) {
    let codes = [q.codes(), k.codes(), v.codes()].concat();
    let scales = [q.scales(), k.scales(), v.scales()].concat();
    let mins = [q.mins(), k.mins(), v.mins()].concat();
    let dd = [
        interleave_d_dmin(q.d(), q.dmin()),
        interleave_d_dmin(k.d(), k.dmin()),
        interleave_d_dmin(v.d(), v.dmin()),
    ]
    .concat();
    (codes, scales, mins, dd)
}

/// The five concatenated Q3_K sub-buffers: `(ql, qh, scales, mins, dd)` — dd
/// interleaved `[d, dmin]` per super-block.
pub type Q3kConcat = (Vec<u32>, Vec<u32>, Vec<u8>, Vec<i8>, Vec<f32>);

/// Concatenate three Q3_K matrices' packed sub-buffers (q rows, then k, then v).
pub fn qkv_concat_q3k(
    q: &arf_core::tensor::Q3KMatrix,
    k: &arf_core::tensor::Q3KMatrix,
    v: &arf_core::tensor::Q3KMatrix,
) -> Q3kConcat {
    let ql = [q.ql(), k.ql(), v.ql()].concat();
    let qh = [q.qh(), k.qh(), v.qh()].concat();
    let scales = [q.scales(), k.scales(), v.scales()].concat();
    let mins = [q.mins(), k.mins(), v.mins()].concat();
    let dd = [
        interleave_d_dmin(q.d(), q.dmin()),
        interleave_d_dmin(k.d(), k.dmin()),
        interleave_d_dmin(v.d(), v.dmin()),
    ]
    .concat();
    (ql, qh, scales, mins, dd)
}

/// Compile the full decode-pipeline kernel set ONCE. Extracted from the model loaders so a
/// self-contained module (e.g. the qwen35 hybrid generation path) can obtain the SAME proven
/// `matmul_vec_q4ks` / `matmul` GEMV kernels without re-loading a whole `GpuModel`. `is_gemma`
/// only sets `q4k_prefer_gemv` (route Gemma's prefill through the bit-exact GEMV); pass `false`
/// for non-Gemma arches. Behavior is byte-identical to the inline construction it replaced.
pub fn build_kernels(ctx: &std::sync::Arc<crate::gpu::GpuContext>, is_gemma: bool) -> GpuKernels {
    macro_rules! kern {
        ($name:literal) => {
            ComputeKernel::new(
                ctx,
                $name,
                include_str!(concat!("shaders/wgsl/", $name, ".wgsl")),
            )
        };
    }
    GpuKernels {
        // Gemma's massive-activation channels make the coop Q4_K GEMM's min-fold
        // reassociation flip tokens; route its prefill through the bit-exact GEMV.
        q4k_prefer_gemv: is_gemma,
        embed: kern!("embed"),
        embed_tok: kern!("embed_tok"),
        embed_bf16: kern!("embed_bf16"),
        embed_tok_bf16: kern!("embed_tok_bf16"),
        embed_q4k: kern!("embed_q4k"),
        embed_tok_q4k: kern!("embed_tok_q4k"),
        rmsnorm: kern!("rmsnorm"),
        matmul: kern!("matmul"),
        matmul_coop: if ctx.has_coop() {
            Some(kern!("matmul_coop"))
        } else {
            None
        },
        matmul_coop_rb2: if ctx.has_coop() {
            Some(kern!("matmul_coop_rb2"))
        } else {
            None
        },
        matmul_coop_rb2_cb4: if ctx.has_coop() {
            Some(kern!("matmul_coop_rb2_cb4"))
        } else {
            None
        },
        matmul_coop_rb2_cb4_dk2: if ctx.has_coop() {
            Some(kern!("matmul_coop_rb2_cb4_dk2"))
        } else {
            None
        },
        matmul_coop_q8: if ctx.has_coop() {
            Some(kern!("matmul_coop_q8"))
        } else {
            None
        },
        matmul_coop_q8_rb2: if ctx.has_coop() {
            Some(kern!("matmul_coop_q8_rb2"))
        } else {
            None
        },
        matmul_coop_q8_rb2_cb4: if ctx.has_coop() {
            Some(kern!("matmul_coop_q8_rb2_cb4"))
        } else {
            None
        },
        matmul_coop_q8_rb2_cb4_dk2: if ctx.has_coop() {
            Some(kern!("matmul_coop_q8_rb2_cb4_dk2"))
        } else {
            None
        },
        matmul_coop_q4: if ctx.has_coop() {
            Some(kern!("matmul_coop_q4"))
        } else {
            None
        },
        matmul_coop_q4_rb2: if ctx.has_coop() {
            Some(kern!("matmul_coop_q4_rb2"))
        } else {
            None
        },
        matmul_coop_q4_rb2_cb4: if ctx.has_coop() {
            Some(kern!("matmul_coop_q4_rb2_cb4"))
        } else {
            None
        },
        matmul_coop_q4_rb2_cb4_dk2: if ctx.has_coop() {
            Some(kern!("matmul_coop_q4_rb2_cb4_dk2"))
        } else {
            None
        },
        matmul_coop_q4k: if ctx.has_coop() {
            Some(kern!("matmul_coop_q4k"))
        } else {
            None
        },
        matmul_coop_q4ks: if ctx.has_coop() {
            Some(kern!("matmul_coop_q4ks"))
        } else {
            None
        },
        matmul_coop_q4k_rb2: if ctx.has_coop() {
            Some(kern!("matmul_coop_q4k_rb2"))
        } else {
            None
        },
        matmul_coop_q4k_rb2_cb2: if ctx.has_coop() {
            Some(kern!("matmul_coop_q4k_rb2_cb2"))
        } else {
            None
        },
        matmul_coop_q4k_rb2_cb8: if ctx.has_coop() {
            Some(kern!("matmul_coop_q4k_rb2_cb8"))
        } else {
            None
        },
        matmul_coop_q4k_rb2_cb4_dk2: if ctx.has_coop() {
            Some(kern!("matmul_coop_q4k_rb2_cb4_dk2"))
        } else {
            None
        },
        matmul_coop_q4k_rb2_cb4: if ctx.has_coop() {
            Some(kern!("matmul_coop_q4k_rb2_cb4"))
        } else {
            None
        },
        matmul_vec: kern!("matmul_vec"),
        matmul_vec_sg: if ctx.has_subgroups() {
            Some(kern!("matmul_vec_sg"))
        } else {
            None
        },
        matmul_vec_batch: kern!("matmul_vec_batch"),
        matmul_vec_q8: kern!("matmul_vec_q8"),
        matmul_vec_q8_sg: if ctx.has_subgroups() {
            Some(kern!("matmul_vec_q8_sg"))
        } else {
            None
        },
        matmul_vec_q8_batch: kern!("matmul_vec_q8_batch"),
        matmul_vec_q4: kern!("matmul_vec_q4"),
        matmul_vec_q4_mc: kern!("matmul_vec_q4_mc"),
        matmul_vec_q4_batch: kern!("matmul_vec_q4_batch"),
        matmul_vec_q4k: kern!("matmul_vec_q4k"),
        matmul_vec_q4k_mr: if std::env::var("ARF_MR").is_ok() {
            Some(kern!("matmul_vec_q4k_mr"))
        } else {
            None
        },
        matmul_vec_q4k_sg: if ctx.has_subgroups() {
            Some(kern!("matmul_vec_q4k_sg"))
        } else {
            None
        },
        matmul_vec_q4k_ggml: if ctx.has_subgroups() {
            Some(kern!("matmul_vec_q4k_ggml"))
        } else {
            None
        },
        matmul_vec_q4k_batch: kern!("matmul_vec_q4k_batch"),
        matmul_vec_q4ks: kern!("matmul_vec_q4ks"),
        matmul_vec_q4ks_batch: kern!("matmul_vec_q4ks_batch"),
        matmul_vec_q3k: kern!("matmul_vec_q3k"),
        matmul_vec_q3k_batch: kern!("matmul_vec_q3k_batch"),
        rope: kern!("rope"),
        rope_qk: kern!("rope_qk"),
        rope_qk_interleaved: kern!("rope_qk_interleaved"),
        qk_norm: kern!("qk_norm"),
        v_norm: kern!("v_norm"),
        qkv_norm: kern!("qkv_norm"),
        moe_route: kern!("moe_route"),
        matmul_vec_moe: kern!("matmul_vec_moe"),
        matmul_vec_moe_q4: kern!("matmul_vec_moe_q4"),
        matmul_vec_moe_q4k: kern!("matmul_vec_moe_q4k"),
        matmul_vec_moe_q4ks: kern!("matmul_vec_moe_q4ks"),
        matmul_vec_moe_batch_gu: kern!("matmul_vec_moe_batch_gu"),
        matmul_vec_moe_q4_batch_gu: kern!("matmul_vec_moe_q4_batch_gu"),
        matmul_vec_moe_q4k_batch_gu: kern!("matmul_vec_moe_q4k_batch_gu"),
        matmul_vec_moe_q4ks_batch_gu: kern!("matmul_vec_moe_q4ks_batch_gu"),
        matmul_vec_moe_down_reduce: kern!("matmul_vec_moe_down_reduce"),
        matmul_vec_moe_q4_down_reduce: kern!("matmul_vec_moe_q4_down_reduce"),
        matmul_vec_moe_q4k_down_reduce: kern!("matmul_vec_moe_q4k_down_reduce"),
        matmul_vec_moe_q4ks_down_reduce: kern!("matmul_vec_moe_q4ks_down_reduce"),
        moe_route_b: kern!("moe_route_b"),
        matmul_vec_moe_batch_gu_b: kern!("matmul_vec_moe_batch_gu_b"),
        matmul_vec_moe_q4ks_batch_gu_b: kern!("matmul_vec_moe_q4ks_batch_gu_b"),
        matmul_vec_moe_q4ks_batch_gu_b_v2: kern!("matmul_vec_moe_q4ks_batch_gu_b_v2"),
        matmul_vec_moe_down_reduce_b: kern!("matmul_vec_moe_down_reduce_b"),
        matmul_vec_moe_q4ks_down_reduce_b: kern!("matmul_vec_moe_q4ks_down_reduce_b"),
        kv_scatter: kern!("kv_scatter"),
        kv_scatter_tq: kern!("kv_scatter_tq"),
        // M11 Qwen3.6 gated-delta-net (single-queue WGSL twins of the island MSL kernels).
        gdn_conv1d: kern!("ssm_conv1d"),
        gdn_l2_norm: kern!("gdn_l2_norm"),
        attn_gate_mul: kern!("attn_gate_mul"),
        gdn_sigmoid_out: kern!("gdn_sigmoid_out"),
        gdn_g_decay: kern!("gdn_g_decay"),
        gdn_tile: kern!("gdn_tile"),
        gdn_recurrence: kern!("gated_delta_net"),
        // B-parallel (concurrency) twins — decode B seqs in lockstep, b-outer state layout.
        gdn_conv1d_b: kern!("ssm_conv1d_b"),
        gdn_l2_norm_b: kern!("gdn_l2_norm_b"),
        gdn_sigmoid_out_b: kern!("gdn_sigmoid_out_b"),
        gdn_g_decay_b: kern!("gdn_g_decay_b"),
        gdn_tile_b: kern!("gdn_tile_b"),
        gdn_recurrence_b: kern!("gated_delta_net_b"),
        attention: kern!("attention"),
        attention_batched: kern!("attention_batched"),
        attention_gqa: kern!("attention_gqa"),
        attention_tq: if std::env::var("ARF_TQ_SERIAL").is_ok() {
            kern!("attention_tq")
        } else {
            ComputeKernel::new(
                ctx,
                "attention_tq_par",
                include_str!("shaders/wgsl/attention_tq_par.wgsl"),
            )
        },
        attention_splitk: kern!("attention_splitk"),
        attention_splitk_combine: kern!("attention_splitk_combine"),
        swiglu: kern!("swiglu"),
        geglu: kern!("geglu"),
        add: kern!("add"),
        add_norm: kern!("add_norm"),
        rmsnorm_add: kern!("rmsnorm_add"),
        sample: kern!("sample"),
        sample_batched: kern!("sample_batched"),
        sample_stochastic: kern!("sample_stochastic"),
        softcap: kern!("softcap"),
        scale_inplace: kern!("scale_inplace"),
    }
}

/// Build a resident GPU model with bf16-packed matmul weights (the default,
/// lossless-to-checkpoint path). For int8 decode weights see [`build_gpu_quant`].
pub fn build_gpu(
    cfg: &ModelConfig,
    w: &Weights,
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
) -> Result<crate::gpu::GpuModel> {
    build_gpu_inner(cfg, w, max_positions, ctx, Quant::None, KvQuant::None, None)
}

/// Build a resident GPU model, quantizing the matmul weights to `quant`:
/// `Quant::None` packs them bf16; `Quant::Int8` stores per-row int8 (a quarter of
/// f32 bytes — decode is bandwidth-bound on these reads; the CPU int8 path is the
/// parity oracle). The embedding table and norm vectors stay f32 regardless.
pub fn build_gpu_quant(
    cfg: &ModelConfig,
    w: &Weights,
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
    quant: Quant,
) -> Result<crate::gpu::GpuModel> {
    build_gpu_inner(cfg, w, max_positions, ctx, quant, KvQuant::None, None)
}

/// Build a resident GPU model with an explicit weight `quant` AND KV-cache
/// `kv_quant`. `KvQuant::None` is the exact f32 KV pool (identical to
/// [`build_gpu_quant`]); `KvQuant::Tq` builds the TurboQuant-packed pool (codes +
/// norms) and uploads the codebook levels — the `*_tq` scatter/attention kernels
/// are then dispatched in place of the f32 pair.
pub fn build_gpu_kv_quant(
    cfg: &ModelConfig,
    w: &Weights,
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
    quant: Quant,
    kv_quant: KvQuant,
) -> Result<crate::gpu::GpuModel> {
    build_gpu_inner(cfg, w, max_positions, ctx, quant, kv_quant, None)
}

/// Like [`build_gpu_kv_quant`] but with an explicit KV pool block count for the
/// serving path. `kv_blocks` MUST equal the scheduler's `EngineConfig::num_blocks`
/// (the GPU pool and the `BlockManager` share the same 0..n block-id space — the
/// actor asserts this at spawn). `None` keeps the single-sequence sizing.
pub fn build_gpu_kv_quant_pooled(
    cfg: &ModelConfig,
    w: &Weights,
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
    quant: Quant,
    kv_quant: KvQuant,
    kv_blocks: Option<usize>,
) -> Result<crate::gpu::GpuModel> {
    build_gpu_inner(cfg, w, max_positions, ctx, quant, kv_quant, kv_blocks)
}

/// Refuse a load whose estimated WIRED footprint would freeze the machine. Metal weight
/// buffers are wired (unswappable) on Apple Silicon: past physical RAM the box doesn't
/// slow down — WindowServer starves and the machine hard-freezes (jetsam 2026-07-02:
/// `gpu_spec_timing` at 52 GB resident on a 36 GB box, a 30B GGUF force-expanded to
/// Int8). llama.cpp survives the same model because it checks the device's
/// recommendedMaxWorkingSetSize up front; this is our equivalent. Estimate = checkpoint
/// params (backing bytes ÷ native bytes/param) × target-quant bytes/param **plus the KV
/// pool**. Threshold 80% of physical RAM. `ARF_FORCE_LOAD=1` overrides for deliberate
/// over-commit experiments.
///
/// 🔴 L147 — THE KV POOL USED TO BE MISSING FROM THIS ESTIMATE, and that was the root cause of
/// the `--num-blocks 4096` corruption (L146). The doc used to say the 80% threshold "leaves
/// headroom for KV/scratch/OS" — but on qwen3-coder-30B the KV pool is 6.4 GB at 2048 blocks and
/// **12.9 GB at 4096**, which is not headroom, it is the largest allocation after the weights.
///
/// MEASURED on a 36 GB M4 Max (cap = 28.8 GB), same binary, only --num-blocks varying:
///
/// ```text
/// blocks 2048: weights 18.4 + KV  6.4 = 24.8 GB  -> CLEAN
/// blocks 3072: weights 18.4 + KV  9.7 = 28.1 GB  -> CLEAN   (banner: 28.08 GB resident)
/// blocks 4096: weights 18.4 + KV 12.9 = 31.3 GB  -> !!!!    (banner: 31.30 GB resident)
/// ```
///
/// The mechanism: weights are file-backed mmap (clean, evictable for free), while the KV pool is
/// anonymous Shared memory (dirty, must swap). Residency is lazy by default, so allocating a
/// 12.9 GB dirty pool on top forces the OS to reclaim exactly the clean weight pages — which then
/// read back ZEROED. Zero weights -> zero logits -> argmax picks token 0 -> `!` forever. The same
/// reclaim-to-zero mechanism is documented at the embed commit (~line 866) and in
/// a measured trap ("llama -3%, arf -78%" under identical pressure).
///
/// Note the pool is NOT the buffer that gets zeroed — it is the buffer whose allocation evicts
/// the weights. Making the pool itself resident does not help (tried, still `!!!!`); the fix is
/// to refuse the over-commit up front.
fn wired_memory_guard(
    w: &Weights,
    quant: Quant,
    cfg: &ModelConfig,
    kv_blocks: Option<usize>,
    max_positions: usize,
) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        if std::env::var_os("ARF_FORCE_LOAD").is_some() {
            return Ok(());
        }
        let backing = w.backing_bytes();
        let native_bpp = w
            .native_quant_hint()
            .map(|q| q.approx_bytes_per_param())
            .unwrap_or(2.0); // safetensors / full-precision GGUF ≈ bf16
        let est_weights = backing as f64 / native_bpp * quant.approx_bytes_per_param();
        // EMBED + TIED LM_HEAD INFLATION — the term this estimate used to miss entirely.
        //
        // `est_weights` prices every tensor at the TARGET quant, `token_embd` included. For that
        // one tensor it is wrong by 4-6x, because two allocations do not follow `--quant`:
        //
        //   * the embedding table is uploaded BF16 whatever `--quant` says (a vocab×hidden gather
        //     read by the bf16 `embed`/`embed_tok` kernels), and
        //   * when the Metal island is up the lm_head is transcoded to a SECOND, Q8 copy so the
        //     megakernel can run lm_head inside its own command buffer.
        //
        // On gemma-4-12b (262144 × 3840) that is 2.01 GB bf16 + 1.01 GB Q8 = 3.02 GB, against the
        // 0.53 GB this estimate was charging. MEASURED: the guard predicted 6.3 GB of weights
        // where the resident sidecar is 10.25 GB. That 4 GB blind spot sat on the largest
        // allocation after the KV pool, and it is why a load that "fit" thrashed the page cache
        // instead of failing honestly — 423k page-ins (6.6 GB) during ONE 19-second decode,
        // decode rate halved, and nothing anywhere said why.
        //
        // ⚠️ `ARF_MSL_GEMV.is_some()` IS THE WHOLE ISLAND PREDICATE — do not "harden" it by
        // also requiring `ARF_NO_MSL_GEMV.is_none()`. The island is built below under exactly
        // this test, as is every gate behind it; `ARF_NO_MSL_GEMV` is resolved in
        // `fast_path.rs` (which CLEARS this var) and is read nowhere down here. Adding the
        // negative term would therefore not describe a configuration the loader can reach — it
        // would only make the two predicates disagree when both vars are set, dropping the Q8
        // gigabyte from a budget whose island is up. That is the under-charge direction, i.e. the
        // silent page-thrash this term was added to prevent.
        let est_embed =
            embed_lm_head_overhead(cfg, quant, std::env::var_os("ARF_MSL_GEMV").is_some());
        // KV pool: K and V, per layer, over all slots. Mirrors the sizing at the pool's own
        // construction (`total_slots * layers[i].kv_dim()`, f32 or the 8-bit cache).
        let block_size = 16usize; // KV_BLOCK_SIZE
        let blocks = kv_blocks.unwrap_or_else(|| max_positions.div_ceil(block_size));
        // L192 — count only the layers that ACTUALLY get a KV pool. L139 layer-filtered the
        // allocation (recurrent layers get a 16-byte placeholder, not a pool) but this guard kept
        // charging for all `num_layers`. On Qwen3.8 only 16 of 64 layers are attention, so the
        // estimate was 4x the real allocation and it REFUSED loads that fit comfortably:
        // `--num-blocks 2048` was rejected as "24.0 GB KV" when the true cost is ~6 GB.
        // Same predicate as `kv_layer_needed` (inside `kv_pool_bytes`), so the two cannot drift.
        // Priced by `kv_pool_bytes`, the same function the pool is sized from: under the 8-bit
        // cache the f32 pool is a placeholder and the real cost is int8 + one f32 scale per
        // (slot, kv head). Charging it as f32 refused a 131K-context load as "16.0 GB KV" when
        // the pool it would have built is ~4.3 GB.
        let mut est_kv = kv_pool_bytes(cfg, blocks) as f64;
        // ARF_KV_F16 allocates a SECOND, half-size pool ALONGSIDE the f32 one (L135).
        if std::env::var_os("ARF_KV_F16").is_some() && !crate::kv_q8_for(cfg) {
            est_kv *= 1.5;
        }
        let est = est_weights + est_embed + est_kv;
        let Some(ram) = physical_ram_bytes() else {
            return Ok(());
        }; // can't read RAM → don't block
        let cap = ram as f64 * MEM_CAP_FRACTION;
        // REPORT THE BUDGET EVEN WHEN IT FITS. Fitting under the cap is not the same as being
        // resident: a 36 GB box passed this check at ~22 GB, then paged 6.6 GB back off the SSD
        // during a single 19-second decode because the rest of the machine wanted memory too.
        // Speed collapsed to roughly half, with variance wide enough (8.4-11.2 tok/s across
        // identical runs) to look like noise rather than a resource problem. Two silent multi-GB
        // allocations are what made it unattributable, so print both and let the operator see the
        // KV pool for what it is — usually the largest single number here, and the easiest to
        // lower via --num-blocks.
        let gb = |x: f64| x / 2f64.powi(30);
        // Read once, printed on the budget line even when it fits, so every load shows that the
        // reading ran (and what it was) — not only the loads that trip the warning below.
        let available = available_memory_bytes();
        let available_note = available
            .map(|a| format!(", ~{:.1} GB available now", gb(a as f64)))
            .unwrap_or_default();
        eprintln!(
            "[mem] budget ~{:.1} GB = weights {:.1} + embed/lm_head {:.1} + KV {:.1} ({blocks} \
             blocks) | {:.0} GB RAM, cap {:.1} GB{available_note}",
            gb(est),
            gb(est_weights),
            gb(est_embed),
            gb(est_kv),
            gb(ram as f64),
            gb(cap)
        );
        // THE THRASH ZONE — between "fits" and "refused".
        //
        // The cap below answers "can this load at all", and gemma-4-12b at the DEFAULT
        // --num-blocks 1024 answers yes: 26.6 GB against a 28.8 GB cap. It then ran at roughly
        // half speed, because a box does not hand one process 74% of RAM while a browser and an
        // editor are open — the shortfall came back as page-ins, 6.6 GB of them in a single
        // 19-second decode. Lowering --num-blocks to 256 cut page-ins ~300x (6.6 GB -> 22 MB) and
        // took decode from 8.4-11.2 to 12.4-13.7 tok/s, on identical weights.
        //
        // Warn rather than refuse, and never lower --num-blocks silently: undersizing the pool is
        // its OWN failure, and a worse one. When the pool cannot hold `max_batch_size` concurrent
        // sequences the allocator evicts blocks a RUNNING sequence still needs, and the victim
        // reads KV another request overwrote and emits fluent garbage at full speed. Slow is
        // debuggable; silently wrong is not. So the operator gets the arithmetic and picks — and
        // gets the pairing rule, because halving blocks without halving concurrency is exactly how
        // you trade this bug for that one.
        if est > ram as f64 * 0.60 && est_kv > est_weights {
            eprintln!(
                "[mem] ⚠ KV pool is {:.1} GB of a {:.1} GB budget on a {:.0} GB box. This fits, but \
                 it likely will not stay RESIDENT alongside other apps, and the shortfall returns as \
                 page-ins at roughly half the decode rate. If you are not serving {} concurrent \
                 sequences, lower --num-blocks (each block is {:.1} MB here; --num-blocks 256 => \
                 {:.1} GB). Keep --num-blocks >= --max-batch-size * --max-context/16 when you do: \
                 an undersized pool corrupts concurrent requests silently instead of slowing down.",
                gb(est_kv),
                gb(est),
                gb(ram as f64),
                blocks,
                est_kv / blocks.max(1) as f64 / 2f64.powi(20),
                gb(est_kv / blocks.max(1) as f64 * 256.0),
            );
        }
        // MEMORY AVAILABLE NOW, not just installed (issue #3). Everything above compares against TOTAL
        // RAM, so a 36 GB box with 15 GB held by other apps passes the cap and then thrashes: the
        // shortfall comes back as page-ins at roughly half the decode rate (the 6.6 GB-in-19-s
        // case above), or, under a dirty KV pool, as evicted weight pages that read back ZEROED
        // (L146/L147). WARN, never refuse, on this number: macOS can compress or page out other
        // apps to make room, so "available" UNDERSTATES what the load can actually obtain, and a
        // refusal on it would block loads that run fine.
        if let Some(msg) = available_memory_warning(est, available) {
            eprintln!("{msg}");
        }
        if est > cap {
            // Name the largest --num-blocks that actually fits, so the operator gets a number
            // rather than a wall. This scales with the real box and model, unlike a hardcoded
            // empirical bound.
            // Per block from the same total, so the 8-bit and f16 pricing carry through.
            let per_block = est_kv / blocks.max(1) as f64;
            let fits = if per_block > 0.0 {
                (((cap - est_weights - est_embed).max(0.0)) / per_block) as usize
            } else {
                0
            };
            return Err(arf_core::ArfError::model_load(format!(
                "refusing to load: ~{:.1} GB total ({:.1} GB wired weights at {quant:?} \
                 + {:.1} GB embed/lm_head \
                 + {:.1} GB KV pool for {blocks} blocks) exceeds {:.1} GB (90% of {:.1} GB RAM). \
                 On Apple Silicon this does NOT swap gracefully: the KV pool is dirty anonymous \
                 memory that evicts the file-backed weight pages, which then read back ZEROED and \
                 the engine emits '!!!!' at full speed with no error (L146/L147). \
                 Largest --num-blocks that fits here: {fits}. \
                 Lower --num-blocks (or --max-batch-size), use the checkpoint's native quant \
                 (e.g. QUANT=q4ks for a Q4_K_S GGUF), or set ARF_FORCE_LOAD=1 to override.",
                est / 2f64.powi(30),
                est_weights / 2f64.powi(30),
                est_embed / 2f64.powi(30),
                est_kv / 2f64.powi(30),
                cap / 2f64.powi(30),
                ram as f64 / 2f64.powi(30),
            )));
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (w, quant, cfg, kv_blocks, max_positions);
    Ok(())
}

/// Physical RAM in bytes (`sysctl -n hw.memsize`); `None` off macOS or when it cannot be read.
/// The load guard's denominator, shared with `arf pull`'s pre-download fit check so the two read
/// the same number.
pub fn physical_ram_bytes() -> Option<u64> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
}

/// Memory the system could hand a new allocation right now without paging anything out, in bytes:
/// free + inactive + speculative + purgeable pages, times the page size, as `vm_stat` reports
/// them. `None` off macOS or when `vm_stat` cannot be run or parsed.
///
/// An ESTIMATE, in both directions, and only ever used for a warning:
///   * it UNDERSTATES what a load can obtain — macOS compresses and pages out other apps' memory
///     on demand, and none of that is counted here;
///   * it can slightly OVERSTATE the immediately reclaimable pool — purgeable pages also sit on
///     the active/inactive queues, so part of the purgeable count may already be in `inactive`.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn available_memory_bytes() -> Option<u64> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = std::process::Command::new("vm_stat").output().ok()?;
    parse_vm_stat_available(&String::from_utf8(out.stdout).ok()?)
}

/// Parse `vm_stat` output into available bytes (see [`available_memory_bytes`]). Pure, so the
/// parse is unit-tested without the command. The page size comes from the header line
/// (`(page size of 16384 bytes)`); every one of the four counters must be present, or `None` —
/// a partial sum would read as "almost nothing available" and warn on every load.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_vm_stat_available(vm_stat: &str) -> Option<u64> {
    let page: u64 = vm_stat
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let pages = |label: &str| -> Option<u64> {
        vm_stat.lines().find_map(|l| {
            let rest = l
                .trim()
                .strip_prefix(label)?
                .trim_start()
                .strip_prefix(':')?;
            rest.trim().trim_end_matches('.').parse::<u64>().ok()
        })
    };
    let total = pages("Pages free")?
        + pages("Pages inactive")?
        + pages("Pages speculative")?
        + pages("Pages purgeable")?;
    Some(total * page)
}

/// The `[mem] ⚠` line for a load whose estimated budget exceeds the memory available right now,
/// or `None` when it fits (or availability could not be read — no number, no warning). Pure, so
/// the boundary is unit-tested.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn available_memory_warning(est_bytes: f64, available: Option<u64>) -> Option<String> {
    let available = available?;
    if est_bytes <= available as f64 {
        return None;
    }
    let gb = |x: f64| x / 2f64.powi(30);
    Some(format!(
        "[mem] ⚠ this load needs ~{:.1} GB but only ~{:.1} GB is available right now (free + \
         inactive + speculative + purgeable); other apps hold the rest. macOS will compress or \
         page them out to make room, so it may still run, but expect page-ins and a slower decode \
         (and, under a large KV pool, weight pages that read back zeroed: L146/L147). Close other \
         apps, or pick a smaller model or a lower --num-blocks.",
        gb(est_bytes),
        gb(available as f64),
    ))
}

/// Fraction of physical RAM the load guard will commit to.
///
/// 🔴 0.90, NOT 0.80 — AND THE TWO CHANGED TOGETHER. Read this before lowering it back.
///
/// This threshold is calibrated against whatever the estimate happens to COUNT, so making the
/// estimate more accurate silently tightens the refusal. Adding `est_embed` grew every model's
/// total by up to 3.2 GB, and at the old 0.80 that turned qwen3.8-27B at its DEFAULT
/// `--num-blocks` from a load into a refusal (28.2 -> 30.0 GB against a 28.8 GB cap) — a
/// configuration that is measured and shipped. A guard that refuses a benchmarked config is not
/// a safer guard, it is a broken one, and it would have reached a user as "your model no longer
/// loads".
///
/// 0.90 is derived, not tuned: to be no stricter than before for any model, the cap must satisfy
/// C >= 0.80 + max(est_embed)/RAM. The largest embed term across the supported set is 3.2 GB on a
/// 36 GB box = 0.089, so C = 0.89; 0.90 rounds it.
///
/// If you extend the estimate again, re-derive this the same way rather than assuming the
/// headroom absorbs it. `cap_covers_embed_overhead_for_every_arch` fails if you don't.
///
/// Public because `arf pull` refuses, before downloading, a model whose weights alone exceed this
/// fraction of RAM — and derives its own warning threshold from it, so the two cannot drift.
pub const MEM_CAP_FRACTION: f64 = 0.90;

/// Resident bytes for `token_embd` and the island's `lm_head` copy that `--quant` does NOT
/// govern, over and above what a params×bytes-per-param estimate already charged.
///
/// Two allocations ignore the target quant:
///
///   * the embedding table is uploaded BF16 whatever `--quant` says — it is a vocab×hidden
///     gather read by the bf16 `embed`/`embed_tok` kernels; and
///   * when the Metal island is up, `lm_head` needs a SECOND, Q8 copy so the megakernel can run
///     it inside its own command buffer.
///
/// On gemma-4-12b (262144 × 3840) that is 2.01 GB bf16 + 1.01 GB Q8 against the 0.53 GB a
/// quant-priced estimate charges. MEASURED: the guard predicted 6.3 GB of weights where the
/// resident sidecar is 10.25 GB — a 4 GB blind spot on the largest allocation after the KV pool,
/// and why a load that "fit" thrashed the page cache instead of failing honestly.
///
/// Pure and free of env/sysctl on purpose: this is the term that moved the refusal boundary, so
/// it has to be checkable per architecture without a GPU or a checkpoint.
fn embed_lm_head_overhead(cfg: &ModelConfig, quant: Quant, island_up: bool) -> f64 {
    let embed_params = (cfg.vocab_size as f64) * (cfg.hidden_size as f64);
    let embed_bf16 = embed_params * 2.0;
    let lm_head_q8 = if island_up {
        embed_params + (cfg.vocab_size as f64) * 4.0
    } else {
        0.0
    };
    // Do not double-charge: the caller's params estimate already priced this tensor once.
    (embed_bf16 + lm_head_q8 - embed_params * quant.approx_bytes_per_param()).max(0.0)
}

#[cfg(test)]
mod memory_guard_tests {
    use super::*;
    use arf_core::config::config_for_arch;

    /// Every `--arch` the engine accepts, so a new architecture cannot be added without its
    /// footprint being considered here.
    fn every_arch() -> Vec<(&'static str, ModelConfig)> {
        arf_core::config::ARCHITECTURES
            .iter()
            .map(|e| (e.canonical, config_for_arch(e.canonical).expect("arch")))
            .collect()
    }

    /// 🔴 THE REGRESSION THIS EXISTS TO PREVENT, and it is not hypothetical — it was introduced,
    /// measured, and caught before shipping.
    ///
    /// Teaching the guard about the bf16 embed / Q8 lm_head copies made the estimate MORE
    /// accurate and, at the old 0.80 cap, made it refuse qwen3.8-27B at its DEFAULT
    /// `--num-blocks`: 28.2 GB -> 30.0 GB against a 28.8 GB cap on a 36 GB box. That is a
    /// configuration that is measured and shipped. A guard that refuses a benchmarked
    /// configuration is not safer, it is broken, and it would have reached a user as "your model
    /// stopped loading" with no hint that a memory ESTIMATE caused it.
    ///
    /// So: raising the estimate REQUIRES raising the cap by at least as much. Pin the derivation
    /// rather than the constant, so the next person to extend the estimate is failed by this test
    /// instead of by a bug report.
    #[test]
    fn cap_covers_embed_overhead_for_every_arch() {
        const RAM: f64 = 36.0 * (1u64 << 30) as f64; // the box this was derived on
        const PRIOR_CAP: f64 = 0.80; // what the cap was before the embed term existed

        let worst = every_arch()
            .into_iter()
            .map(|(name, cfg)| {
                // Island up + q4ks is the worst case: the Q8 copy exists and the quant credit
                // subtracted is smallest.
                let bytes = embed_lm_head_overhead(&cfg, Quant::Q4KS, true);
                (name, bytes / RAM)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("at least one arch");

        assert!(
            MEM_CAP_FRACTION >= PRIOR_CAP + worst.1,
            "MEM_CAP_FRACTION {MEM_CAP_FRACTION} is stricter than the guard was before the embed \
             term was added: {} alone needs {:.3} of RAM, so the cap must be >= {:.3}. Loads that \
             worked will now be REFUSED. Raise MEM_CAP_FRACTION rather than lowering the estimate.",
            worst.0,
            worst.1,
            PRIOR_CAP + worst.1,
        );
    }

    /// Real `vm_stat` output (macOS 26, 16 KiB pages) parses to free + inactive + speculative +
    /// purgeable pages × page size; a missing counter is `None`, not a partial (smaller) sum.
    #[test]
    fn vm_stat_available_is_the_four_counters_times_page_size() {
        let out = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                   Pages free:                                   247645.\n\
                   Pages active:                                 593829.\n\
                   Pages inactive:                               624128.\n\
                   Pages speculative:                             21103.\n\
                   Pages throttled:                                   0.\n\
                   Pages wired down:                             519142.\n\
                   Pages purgeable:                               18962.\n\
                   \"Translation faults\":                     5176688193.\n";
        assert_eq!(
            parse_vm_stat_available(out),
            Some((247645 + 624128 + 21103 + 18962) * 16384)
        );
        let no_purgeable = out.replace("Pages purgeable:", "Pages gone:");
        assert_eq!(parse_vm_stat_available(&no_purgeable), None);
        let no_page_size = out.replace("page size of 16384 bytes", "");
        assert_eq!(parse_vm_stat_available(&no_page_size), None);
    }

    /// The available-now warning fires exactly when the budget exceeds what is available, says
    /// both numbers, and stays silent when availability could not be read.
    #[test]
    fn available_memory_warning_boundary() {
        const GIB: u64 = 1 << 30;
        assert!(available_memory_warning((10 * GIB) as f64, Some(10 * GIB)).is_none());
        assert!(available_memory_warning((9 * GIB) as f64, Some(10 * GIB)).is_none());
        assert!(available_memory_warning((30 * GIB) as f64, None).is_none());
        let msg = available_memory_warning((22 * GIB) as f64, Some(15 * GIB))
            .expect("22 GB budget against 15 GB available must warn");
        assert!(msg.starts_with("[mem] ⚠"), "{msg}");
        assert!(
            msg.contains("~22.0 GB") && msg.contains("~15.0 GB"),
            "{msg}"
        );
    }

    /// The overhead must be REAL for every architecture (all of them have an embedding table),
    /// and it must shrink when the island is off, since the Q8 copy is the island's.
    #[test]
    fn overhead_is_charged_and_island_dependent() {
        for (name, cfg) in every_arch() {
            let with = embed_lm_head_overhead(&cfg, Quant::Q4KS, true);
            let without = embed_lm_head_overhead(&cfg, Quant::Q4KS, false);
            assert!(with > 0.0, "{name}: embed table is always resident bf16");
            assert!(
                with > without,
                "{name}: the island's Q8 lm_head copy must be charged only when the island is up"
            );
            // The bf16 table alone, minus the quant credit, is the floor for the island-off case.
            let params = (cfg.vocab_size as f64) * (cfg.hidden_size as f64);
            assert!(
                (without - (params * 2.0 - params * Quant::Q4KS.approx_bytes_per_param())).abs()
                    < 1.0,
                "{name}: island-off overhead should be exactly the bf16 table less the quant credit"
            );
        }
    }
}

fn build_gpu_inner(
    cfg: &ModelConfig,
    w: &Weights,
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
    quant: Quant,
    kv_quant: KvQuant,
    kv_blocks: Option<usize>,
) -> Result<crate::gpu::GpuModel> {
    cfg.validate()?;
    wired_memory_guard(w, quant, cfg, kv_blocks, max_positions)?;

    // WEIGHT CACHE (macOS island path, ARF_WEIGHT_CACHE default-on): start a cache
    // session keyed on the GGUF the load entry point recorded. On a HIT every island weight
    // buffer is wrapped no-copy from the sidecar mmap (load in seconds); on a MISS we
    // transcode straight into the file-backed cache region + wrap THAT no-copy (kills the
    // second copy + the first-decode page-in). Only meaningful when the MSL island is on
    // (only `SharedBuffer::from_pod` participates); a no-op otherwise. `est_bytes` upper-
    // bounds the cached (no-copy) weight set: the island MoeMtl + dense q/k/v/o/router +
    // norms — our q4ks repack ≈ the GGUF's quantized data, so 1.3× the file length is a safe
    // reservation (+ page padding per region). Over-reserving is harmless (the header records
    // the real used length); under-reserving leaves the overflow regions uncached (memcpy).
    #[cfg(target_os = "macos")]
    let _cache_session = {
        let on_msl = std::env::var_os("ARF_MSL_GEMV").is_some();
        if on_msl {
            if let Some(gp) = crate::weight_cache::take_gguf_path() {
                // L211 — 1.3x IS TOO TIGHT. Measured on the shipped Q4_K_M sidecar: 22.603 GB
                // of cache for a 17.107 GB GGUF = 1.321x, already over the 1.3 estimate. It
                // survived only because `reserve` adds ~1.5% + 16 MB on top. Q4_K_S expands
                // MORE (it has fewer Q6_K tensors, so a larger share of the file is Q4_K that
                // gets our unpacked scale/min/dd metadata) and it overflowed for real:
                //     build overflow at model.layers.62.mlp.gate_proj.weight.ddh
                //     (off=20910374912 cap=20911119213); region uncached
                // Every region from layer 62 on went UNCACHED, the sidecar never promoted, and
                // each subsequent run re-transcoded — which is why Q4_K_S first measured 15.9
                // tok/s, SLOWER than Q4_K_M despite being 1.28 GB smaller.
                //
                // Over-reserving is documented as harmless (the header records the real used
                // length, and the file is sparse until written), so size for the worst case.
                let est = std::fs::metadata(&gp)
                    .map(|m| (m.len() as f64 * 1.6) as u64)
                    .unwrap_or(32 * 1024 * 1024 * 1024);
                crate::weight_cache::begin(&gp, est)
            } else {
                false
            }
        } else {
            false
        }
    };
    // Gemma needs its `(1+weight)` norms (folded at upload, below), the extra
    // pre/post feedforward norms, dual RoPE tables and per-layer sliding windows;
    // the decode pipeline branches on `cfg.norm_style` for the four-norm block.
    let is_gemma = cfg.norm_style == NormStyle::Gemma;
    let h = cfg.hidden_size;
    // Per-layer geometry: the q/k/v projection out-dims, the o_proj in-dim, and the
    // KV-pool row stride are recomputed per layer below via `cfg.layer_geometry(i)`
    // — Gemma 4's GLOBAL layers override head_dim/kv_heads (512/4 vs the sliding
    // 256/16). For every other model all layers match the base scalars, so the
    // per-layer values are constant.

    // Native-path counters (eprintln'd after model load for diagnostics).
    // Cell<usize> allows shared capture in the closure without mut borrow conflicts.
    let native_q4k_count = std::cell::Cell::new(0usize);
    let fallback_count = std::cell::Cell::new(0usize);

    // Matmul weights: bf16-packed by default, or per-row int8 when quantized.
    // For Q4K and Q4KS against a GGUF-backed Weights, we first attempt the
    // LOSSLESS native transcode path (get_q4k_blocks + from_ggml_q4k),
    // which avoids the lossy dequant→re-quant round-trip. Only when the tensor is
    // not native Q4_K (e.g. safetensors, or Q5_K/Q6_K tensors in the GGUF) do we
    // fall back to the bf16 self-quant path.
    let up_w = |name: &str, out: usize, in_: usize| -> Result<GpuMatWeight> {
        // For Q4 against a NATIVE Q4_0 GGUF: lossless transcode (no dequant→re-quant
        // double-quant, which adds the sampling-visible logit noise on Q4_0 blobs).
        if matches!(quant, Quant::Q4) {
            if let Some(q) = w.q4_matrix_from_gguf_q4_0(name, out, in_) {
                native_q4k_count.set(native_q4k_count.get() + 1);
                return Ok(GpuMatWeight::Q4 {
                    codes: ctx.storage_init(&format!("{name}.codes"), q.codes()),
                    scales: ctx.storage_init_q4_scales(&format!("{name}.scale"), q.scales()),
                });
            }
        }
        // GROUP-64 AFFINE (2026-09-25): a Q4_1 tensor whose block pairs share d and m is a
        // group-64 weight (the MLX / another engine format, `scripts/g64/mlx_to_q4_1_gguf.py`). It is
        // held in the group-64 kernels' tiled layout, one copy, and the Q4KS fields are
        // placeholders (see `Q4ksMtl::g64`). Cached like every other island weight.
        #[cfg(target_os = "macos")]
        if matches!(quant, Quant::Q4K | Quant::Q4KS) && std::env::var_os("ARF_MSL_GEMV").is_some() {
            use crate::gpu::concurrent_metal::{G64Mtl, Q4ksMtl, SharedBuffer};
            let u = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let hit = crate::weight_cache::is_read_hit()
                .then(|| {
                    Some((
                        SharedBuffer::from_cache(ctx, &format!("{name}.g64w"), u)?,
                        SharedBuffer::from_cache(ctx, &format!("{name}.g64s"), u)?,
                        SharedBuffer::from_cache(ctx, &format!("{name}.g64b"), u)?,
                        SharedBuffer::from_cache(ctx, &format!("{name}.g64ph"), u)?,
                        SharedBuffer::from_cache(ctx, &format!("{name}.dims"), u)?,
                    ))
                })
                .flatten();
            let built = hit.or_else(|| {
                let (blocks, r, c) = w.q4_1_blocks(name)?;
                if (r, c) != (out, in_) {
                    panic!("g64: shape mismatch for {name}: file ({r},{c}), caller ({out},{in_})");
                }
                let (gw, gs, gb) = arf_core::model::weights::g64_from_q4_1(blocks, out, in_)?;
                // 3 BITS (2026-09-26), OPT-IN (ARF_G64_3BIT=1): every code below 8 -> the group-pair
                // 3-bit layout, 25% fewer bytes. ⛔ MEASURED-OUT as a default: the 3-bit kernel runs
                // 0.88-0.92x the 4-bit one per matmul (this kernel is bound by loads in flight, and a
                // 3-bit record cannot be fetched in fewer, wider loads than a 4-bit one — with the
                // extraction made free it only reaches parity), and 3-bit FFN costs +2.0-3.4%
                // perplexity (measured 2026-09-26). A 3-bit file runs in the 4-bit layout by
                // default. The width rides in dims[3].
                let w3 = std::env::var_os("ARF_G64_3BIT")
                    .is_some()
                    .then(|| arf_core::model::weights::g64_pack3(&gw, in_))
                    .flatten();
                let bits = if w3.is_some() { 3u32 } else { 0u32 };
                Some((
                    SharedBuffer::from_pod(
                        ctx,
                        w3.as_deref().unwrap_or(&gw),
                        u,
                        &format!("{name}.g64w"),
                    )?,
                    SharedBuffer::from_pod(ctx, &gs, u, &format!("{name}.g64s"))?,
                    SharedBuffer::from_pod(ctx, &gb, u, &format!("{name}.g64b"))?,
                    // one small placeholder buffer for the Q4KS fields no kernel may read
                    SharedBuffer::from_pod(ctx, &[0u32; 16], u, &format!("{name}.g64ph"))?,
                    SharedBuffer::from_pod(
                        ctx,
                        &[1u32, in_ as u32, out as u32, bits],
                        u,
                        &format!("{name}.dims"),
                    )?,
                ))
            });
            if let Some((gw, gs, gb, ph, dims)) = built {
                let bits = unsafe {
                    *(objc2_metal::MTLBuffer::contents(&*dims.mtl).as_ptr() as *const u32).add(3)
                };
                let bits: u8 = if bits == 3 { 3 } else { 4 };
                if bits == 3 {
                    G64_3BIT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                native_q4k_count.set(native_q4k_count.get() + 1);
                G64_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let mtl = Some(std::sync::Arc::new(Q4ksMtl {
                    codes: ph.mtl.clone(),
                    scales: ph.mtl.clone(),
                    mins: ph.mtl.clone(),
                    dd: ph.mtl.clone(),
                    dims: dims.mtl.clone(),
                    n: out,
                    g64: Some(G64Mtl {
                        w: gw.mtl.clone(),
                        s: gs.mtl.clone(),
                        b: gb.mtl.clone(),
                        k: in_,
                        bits,
                    }),
                }));
                return Ok(GpuMatWeight::Q4KS {
                    codes: ph.wgpu.clone(),
                    scales: ph.wgpu.clone(),
                    mins: ph.wgpu.clone(),
                    dd: ph.wgpu,
                    mtl,
                });
            }
        }
        // WEIGHT CACHE read-hit (macOS island, Q4K/Q4KS): wrap the dense weight's five
        // sub-buffers no-copy from the sidecar and SKIP `q4ks_matrix_from_gguf_q4k` entirely
        // (the transcode result is cached) — the load-in-seconds win for the dense q/k/v/o +
        // router + lm_head GEMVs. Only on a full hit; else fall through to transcode + cache.
        #[cfg(target_os = "macos")]
        if matches!(quant, Quant::Q4K | Quant::Q4KS)
            && std::env::var_os("ARF_MSL_GEMV").is_some()
            && crate::weight_cache::is_read_hit()
        {
            use crate::gpu::concurrent_metal::{Q4ksMtl, SharedBuffer};
            let u = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            // The island's dd is the packed half2 stream, cached under "{name}.ddh"; the wgpu
            // dd stays the interleaved f32 under "{name}.dd". A stale (pre-ddh) sidecar misses
            // the ".ddh" key → we fall through to the transcode path, which rebuilds + caches it.
            if let (Some(c), Some(s), Some(mi), Some(ddb), Some(ddhb), Some(dsb)) = (
                SharedBuffer::from_cache(ctx, &format!("{name}.codes"), u),
                SharedBuffer::from_cache(ctx, &format!("{name}.scale"), u),
                SharedBuffer::from_cache(ctx, &format!("{name}.min"), u),
                SharedBuffer::from_cache(ctx, &format!("{name}.dd"), u),
                SharedBuffer::from_cache(ctx, &format!("{name}.ddh"), u),
                SharedBuffer::from_cache(ctx, &format!("{name}.dims"), u),
            ) {
                native_q4k_count.set(native_q4k_count.get() + 1);
                let mtl = Some(std::sync::Arc::new(Q4ksMtl {
                    codes: c.mtl.clone(),
                    scales: s.mtl.clone(),
                    mins: mi.mtl.clone(),
                    dd: ddhb.mtl.clone(),
                    dims: dsb.mtl.clone(),
                    n: out,
                    g64: None,
                }));
                return Ok(GpuMatWeight::Q4KS {
                    codes: c.wgpu,
                    scales: s.wgpu,
                    mins: mi.wgpu,
                    dd: ddb.wgpu,
                    mtl,
                });
            }
            // Partial hit → fall through to the transcode path below for correctness.
        }
        // For Q4K and Q4KS: try the lossless native path first — no bf16 alloc.
        if matches!(quant, Quant::Q4K | Quant::Q4KS) {
            if let Some(q) = w.q4ks_matrix_from_gguf_q4k(name, out, in_).map(|q| {
                // TEMPORARY EXPERIMENT (2026-09-21): ARF_TMP_PAIR=64|128 re-fits native Q4_K
                // tensors with shared scales per 64/128 weights, to price that in perplexity.
                match std::env::var("ARF_TMP_PAIR")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                {
                    Some(g) if in_.is_multiple_of(256) => {
                        arf_core::tensor::Q4KSMatrix::searched_grouped(&q.to_f32(), out, in_, g)
                    }
                    _ => q,
                }
            }) {
                native_q4k_count.set(native_q4k_count.get() + 1);
                // dd is interleaved [d, dmin] per super-block.
                let mut dd = Vec::with_capacity(q.d().len() * 2);
                for (dv, dmv) in q.d().iter().zip(q.dmin()) {
                    dd.push(*dv);
                    dd.push(*dmv);
                }
                // MSL fast path (macOS + ARF_MSL_GEMV): allocate the weight buffers as
                // SharedBuffers (one Metal alloc, two views) so the hand-MSL gemv_q4ks
                // kernel reads the SAME bytes. Identical packing to the wgpu uploaders,
                // EXCEPT dd: the island reads the packed half2 "ddh" stream (one u32 per
                // super-block pair, bit-lossless f16 repack), a SEPARATE buffer from the
                // wgpu-side interleaved f32 dd.
                #[cfg(target_os = "macos")]
                if std::env::var_os("ARF_MSL_GEMV").is_some() {
                    use crate::gpu::concurrent_metal::{dd_to_ddh, Q4ksMtl, SharedBuffer};
                    let u = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
                    let pack_u8 = |d: &[u8]| -> Vec<u32> {
                        d.chunks_exact(4)
                            .map(|q| {
                                (q[0] as u32)
                                    | ((q[1] as u32) << 8)
                                    | ((q[2] as u32) << 16)
                                    | ((q[3] as u32) << 24)
                            })
                            .collect()
                    };
                    let pack_i8 = |d: &[i8]| -> Vec<u32> {
                        d.chunks_exact(4)
                            .map(|q| {
                                (q[0] as u8 as u32)
                                    | ((q[1] as u8 as u32) << 8)
                                    | ((q[2] as u8 as u32) << 16)
                                    | ((q[3] as u8 as u32) << 24)
                            })
                            .collect()
                    };
                    let sc_p = pack_u8(q.scales());
                    let mn_p = pack_i8(q.mins());
                    let ddh = dd_to_ddh(&dd);
                    if let (Some(c), Some(s), Some(mi), Some(ddb), Some(ddhb)) = (
                        SharedBuffer::from_pod(ctx, q.codes(), u, &format!("{name}.codes")),
                        SharedBuffer::from_pod(ctx, &sc_p, u, &format!("{name}.scale")),
                        SharedBuffer::from_pod(ctx, &mn_p, u, &format!("{name}.min")),
                        SharedBuffer::from_pod(ctx, &dd, u, &format!("{name}.dd")),
                        SharedBuffer::from_pod(ctx, &ddh, u, &format!("{name}.ddh")),
                    ) {
                        // per-weight dims {m:1, k:in_, n:out, 0}
                        let dims_sb = SharedBuffer::from_pod(
                            ctx,
                            &[1u32, in_ as u32, out as u32, 0u32],
                            u,
                            &format!("{name}.dims"),
                        );
                        let mtl = dims_sb.map(|d| {
                            std::sync::Arc::new(Q4ksMtl {
                                codes: c.mtl.clone(),
                                scales: s.mtl.clone(),
                                mins: mi.mtl.clone(),
                                dd: ddhb.mtl.clone(),
                                dims: d.mtl.clone(),
                                n: out,
                                g64: None,
                            })
                        });
                        return Ok(GpuMatWeight::Q4KS {
                            codes: c.wgpu,
                            scales: s.wgpu,
                            mins: mi.wgpu,
                            dd: ddb.wgpu,
                            mtl,
                        });
                    }
                }
                return Ok(GpuMatWeight::Q4KS {
                    codes: ctx.storage_init(&format!("{name}.codes"), q.codes()),
                    scales: ctx.storage_init_u8(&format!("{name}.scale"), q.scales()),
                    mins: ctx.storage_init_q8(&format!("{name}.min"), q.mins()),
                    dd: ctx.storage_init(&format!("{name}.dd"), &dd),
                    #[cfg(target_os = "macos")]
                    mtl: None,
                });
            }
        }
        // Fallback: dequant to bf16 then self-quant (safetensors path, or GGUF
        // tensors that are not native Q4_K such as token_embd/output in Q5_K).
        fallback_count.set(fallback_count.get() + 1);
        // CACHE READ-HIT for the FALLBACK-transcoded tensors (v_proj/down_proj/lm_head, Q6_K in the
        // mixed GGUF). Their Q4KS transcode was written to the cache on first build; on a warm load
        // we MUST wrap it zero-copy here — BEFORE the expensive bf16 dequant + from_f32 re-quant
        // below. Without this early-out, the 73 fallback tensors (the LARGEST ones) re-transcoded on
        // the CPU every single load = ~2 min of the load stall (2026-07-08). Mirrors the native
        // read-hit block above; same 6-key requirement (.ddh present = post-ddh sidecar).
        #[cfg(target_os = "macos")]
        if matches!(quant, Quant::Q4K | Quant::Q4KS)
            && std::env::var_os("ARF_MSL_GEMV").is_some()
            && in_.is_multiple_of(256)
            && crate::weight_cache::is_read_hit()
        {
            use crate::gpu::concurrent_metal::{Q4ksMtl, SharedBuffer};
            let u = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            if let (Some(c), Some(s), Some(mi), Some(ddb), Some(ddhb), Some(dsb)) = (
                SharedBuffer::from_cache(ctx, &format!("{name}.codes"), u),
                SharedBuffer::from_cache(ctx, &format!("{name}.scale"), u),
                SharedBuffer::from_cache(ctx, &format!("{name}.min"), u),
                SharedBuffer::from_cache(ctx, &format!("{name}.dd"), u),
                SharedBuffer::from_cache(ctx, &format!("{name}.ddh"), u),
                SharedBuffer::from_cache(ctx, &format!("{name}.dims"), u),
            ) {
                let mtl = Some(std::sync::Arc::new(Q4ksMtl {
                    codes: c.mtl.clone(),
                    scales: s.mtl.clone(),
                    mins: mi.mtl.clone(),
                    dd: ddhb.mtl.clone(),
                    dims: dsb.mtl.clone(),
                    n: out,
                    g64: None,
                }));
                return Ok(GpuMatWeight::Q4KS {
                    codes: c.wgpu,
                    scales: s.wgpu,
                    mins: mi.wgpu,
                    dd: ddb.wgpu,
                    mtl,
                });
            }
            // Partial hit → fall through to the transcode (rebuilds + re-caches for correctness).
        }
        let m = w.bf16_matrix(name, out, in_)?;
        // MEGAKERNEL: weights that aren't native Q4_K_S in the GGUF (e.g. v_proj/down_proj
        // shipped Q6_K → this fallback) must STILL be island-runnable, or the whole-token
        // megakernel can't run their GEMV. When the island is on + the shape is Q4KS-eligible
        // (cols%256==0), TRANSCODE the bf16 matrix to Q4_K_S and build the same Q4ksMtl as the
        // native path. Slight Q6_K→Q4KS accuracy loss; parity-gated. wgpu still gets a Q4KS
        // view so the portable path is unchanged in shape.
        #[cfg(target_os = "macos")]
        if std::env::var_os("ARF_MSL_GEMV").is_some() && in_.is_multiple_of(256) {
            use crate::gpu::concurrent_metal::{dd_to_ddh, Q4ksMtl, SharedBuffer};
            let q = arf_core::tensor::Q4KSMatrix::from_f32(&m.to_f32(), out, in_);
            let u = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let pack_u8 = |d: &[u8]| -> Vec<u32> {
                d.chunks_exact(4)
                    .map(|q| {
                        (q[0] as u32)
                            | ((q[1] as u32) << 8)
                            | ((q[2] as u32) << 16)
                            | ((q[3] as u32) << 24)
                    })
                    .collect()
            };
            let pack_i8 = |d: &[i8]| -> Vec<u32> {
                d.chunks_exact(4)
                    .map(|q| {
                        (q[0] as u8 as u32)
                            | ((q[1] as u8 as u32) << 8)
                            | ((q[2] as u8 as u32) << 16)
                            | ((q[3] as u8 as u32) << 24)
                    })
                    .collect()
            };
            let mut dd = Vec::with_capacity(q.d().len() * 2);
            for (dv, dmv) in q.d().iter().zip(q.dmin()) {
                dd.push(*dv);
                dd.push(*dmv);
            }
            let sc_p = pack_u8(q.scales());
            let mn_p = pack_i8(q.mins());
            // Island dd = packed half2 "ddh" (separate buffer); wgpu dd stays f32.
            let ddh = dd_to_ddh(&dd);
            if let (Some(c), Some(s), Some(mi), Some(ddb), Some(ddhb)) = (
                SharedBuffer::from_pod(ctx, q.codes(), u, &format!("{name}.codes")),
                SharedBuffer::from_pod(ctx, &sc_p, u, &format!("{name}.scale")),
                SharedBuffer::from_pod(ctx, &mn_p, u, &format!("{name}.min")),
                SharedBuffer::from_pod(ctx, &dd, u, &format!("{name}.dd")),
                SharedBuffer::from_pod(ctx, &ddh, u, &format!("{name}.ddh")),
            ) {
                if let Some(dsb) = SharedBuffer::from_pod(
                    ctx,
                    &[1u32, in_ as u32, out as u32, 0u32],
                    u,
                    &format!("{name}.dims"),
                ) {
                    let mtl = Some(std::sync::Arc::new(Q4ksMtl {
                        codes: c.mtl.clone(),
                        scales: s.mtl.clone(),
                        mins: mi.mtl.clone(),
                        dd: ddhb.mtl.clone(),
                        dims: dsb.mtl.clone(),
                        n: out,
                        g64: None,
                    }));
                    return Ok(GpuMatWeight::Q4KS {
                        codes: c.wgpu,
                        scales: s.wgpu,
                        mins: mi.wgpu,
                        dd: ddb.wgpu,
                        mtl,
                    });
                }
            }
        }
        // Verify the EXACT data being uploaded for L0 gate/up (ARF_DUMP_WUP).
        if std::env::var("ARF_DUMP_WUP").is_ok()
            && (name.contains("layers.0.mlp.gate_proj") || name.contains("layers.0.mlp.up_proj"))
        {
            let f = m.to_f32();
            let absmax = f.iter().fold(0f32, |a, c| a.max(c.abs()));
            eprintln!(
                "[WUP {name}] out={out} in_={in_} len={} absmax={absmax:.4} first6={:?}",
                f.len(),
                &f[..6.min(f.len())]
            );
        }
        Ok(match quant {
            // Upload the bf16 bits directly — no `to_f32()` widen. For the tied
            // lm_head (a vocab×hidden table that hits this fallback under the default
            // --quant none) this drops the same ~5.6 GB transient as the embed above.
            Quant::None => GpuMatWeight::Bf16(ctx.storage_init_bf16_from_u16(name, m.bits())),
            Quant::Int8 => {
                let q = arf_core::tensor::Q8Matrix::from_f32(&m.to_f32(), out, in_);
                GpuMatWeight::Int8 {
                    data: ctx.storage_init_q8(name, q.data()),
                    scale: ctx.storage_init(&format!("{name}.scale"), q.scales()),
                }
            }
            Quant::Q4 => {
                let q = arf_core::tensor::Q4Matrix::from_f32(&m.to_f32(), out, in_);
                GpuMatWeight::Q4 {
                    codes: ctx.storage_init(&format!("{name}.codes"), q.codes()),
                    scales: ctx.storage_init_q4_scales(&format!("{name}.scale"), q.scales()),
                }
            }
            Quant::Q4K => {
                let q = arf_core::tensor::Q4KMatrix::from_f32(&m.to_f32(), out, in_);
                GpuMatWeight::Q4K {
                    codes: ctx.storage_init(&format!("{name}.codes"), q.codes()),
                    scales: ctx.storage_init_q4_scales(&format!("{name}.scale"), q.scales()),
                    mins: ctx.storage_init_q4_scales(&format!("{name}.min"), q.mins()),
                }
            }
            Quant::Q4KS => {
                let q = arf_core::tensor::Q4KSMatrix::from_f32(&m.to_f32(), out, in_);
                // dd is interleaved [d, dmin] per super-block.
                let mut dd = Vec::with_capacity(q.d().len() * 2);
                for (dv, dmv) in q.d().iter().zip(q.dmin()) {
                    dd.push(*dv);
                    dd.push(*dmv);
                }
                GpuMatWeight::Q4KS {
                    codes: ctx.storage_init(&format!("{name}.codes"), q.codes()),
                    scales: ctx.storage_init_u8(&format!("{name}.scale"), q.scales()),
                    mins: ctx.storage_init_q8(&format!("{name}.min"), q.mins()),
                    dd: ctx.storage_init(&format!("{name}.dd"), &dd),
                    #[cfg(target_os = "macos")]
                    mtl: None,
                }
            }
            Quant::Q3K => {
                let q = arf_core::tensor::Q3KMatrix::from_f32(&m.to_f32(), out, in_);
                let mut dd = Vec::with_capacity(q.d().len() * 2);
                for (dv, dmv) in q.d().iter().zip(q.dmin()) {
                    dd.push(*dv);
                    dd.push(*dmv);
                }
                GpuMatWeight::Q3K {
                    ql: ctx.storage_init(&format!("{name}.ql"), q.ql()),
                    qh: ctx.storage_init(&format!("{name}.qh"), q.qh()),
                    scales: ctx.storage_init_u8(&format!("{name}.scale"), q.scales()),
                    mins: ctx.storage_init_q8(&format!("{name}.min"), q.mins()),
                    dd: ctx.storage_init(&format!("{name}.dd"), &dd),
                }
            }
        })
    };
    // Norm vectors stay f32. Gemma folds its `(1+weight)` into the loaded weight so
    // the standard `weight·x` rmsnorm kernel reproduces it exactly (head_dim q/k
    // norms fold the same way below) — but only for raw HF weights. llama.cpp's
    // HF→GGUF converter already bakes the `+1` into every `*norm.weight`, so a GGUF
    // arrives pre-shifted; folding again would store `(2 + w)` and corrupt the norms.
    // Q3K (3-bit) transcode for the megakernel FFN island (ARF_Q3K_FFN). Reconstructs f32
    // from the bf16 weight, Q3K-quantizes, builds a Q3ksMtl (ql/qh/scales/mins/dd MTL views).
    // ARF_Q8_DOWN — a Q8 island view of one projection, built from the file's own precision
    // (bf16 dequant of its Q6_K/Q5_K/Q4_K blocks) exactly the way the parity-safe Q8 lm_head is.
    #[cfg(target_os = "macos")]
    let q8_mtl = |name: &str,
                  out: usize,
                  in_: usize|
     -> Option<crate::gpu::concurrent_metal::Q8Mtl> {
        use crate::gpu::concurrent_metal::{Q8Mtl, SharedBuffer};
        let m = w.bf16_matrix(name, out, in_).ok()?;
        let mut f = m.to_f32();
        // ARF_Q8_DOWN=q4: quantise the SOURCE through the served Q4KS first, so the Q8 view is
        // Q8(Q4KS(f)) — within Q8 rounding of the tensor the model already runs on. Separates
        // "the Q8 arm is wrong" from "Q8(f) differs from the served tensor in a way that matters".
        if std::env::var("ARF_Q8_DOWN").ok().as_deref() == Some("q4") {
            f = arf_core::tensor::Q4KSMatrix::from_f32(&f, out, in_).to_f32();
        }
        // BLOCK-32 int8: a scale per 32 inputs of each row. Per-ROW int8 (`Q8Matrix`) is what the
        // lm_head uses and it DESTROYED ffn_down — one "super weight" per row set the scale for
        // 17,407 others (measured 2026-09-22). Block scales contain the outlier.
        assert!(
            in_.is_multiple_of(32),
            "{name}: block-32 int8 needs cols % 32 == 0"
        );
        let nblk = in_ / 32;
        let mut data = vec![0i8; out * in_];
        let mut scales = vec![0f32; out * nblk];
        for r in 0..out {
            for j in 0..nblk {
                let blk = &f[r * in_ + j * 32..r * in_ + j * 32 + 32];
                let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                scales[r * nblk + j] = sc;
                for (t, v) in blk.iter().enumerate() {
                    data[r * in_ + j * 32 + t] = (v / sc).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
        // SELF-CHECK (a few layers): does the block-Q8 view reproduce its own f32 source?
        if [
            "layers.0.",
            "layers.1.",
            "layers.31.",
            "layers.32.",
            "layers.62.",
            "layers.63.",
        ]
        .iter()
        .any(|t| name.contains(t))
        {
            let mut seed: u64 = 0x9E3779B97F4A7C15;
            let x: Vec<f32> = (0..in_)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
                })
                .collect();
            let mut worst = 0.0f64;
            for r in [0usize, out / 3, out - 1] {
                let y_f: f32 = (0..in_).map(|c| f[r * in_ + c] * x[c]).sum();
                let y_q: f32 = (0..in_)
                    .map(|c| data[r * in_ + c] as f32 * scales[r * nblk + c / 32] * x[c])
                    .sum();
                worst = worst.max(((y_q - y_f).abs() / y_f.abs().max(1e-3)) as f64);
            }
            eprintln!("[q8-down check] {name}: rows={out} cols={in_} block32 | worst rel err over 3 rows = {worst:.3e}");
        }
        let u = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let packed: Vec<u32> = data
            .chunks_exact(4)
            .map(|c| {
                (c[0] as u8 as u32)
                    | ((c[1] as u8 as u32) << 8)
                    | ((c[2] as u8 as u32) << 16)
                    | ((c[3] as u8 as u32) << 24)
            })
            .collect();
        let (data, scale, dims) = (
            SharedBuffer::from_pod(ctx, &packed, u, &format!("{name}.q8"))?,
            SharedBuffer::from_pod(ctx, &scales, u, &format!("{name}.q8scale"))?,
            SharedBuffer::from_pod(
                ctx,
                &[1u32, in_ as u32, out as u32, 0u32],
                u,
                &format!("{name}.q8dims"),
            )?,
        );
        Some(Q8Mtl {
            data: data.mtl,
            scale: scale.mtl,
            dims: dims.mtl,
            n: out,
        })
    };
    #[cfg(target_os = "macos")]
    let q3k_mtl =
        |name: &str, out: usize, in_: usize| -> Option<crate::gpu::concurrent_metal::Q3ksMtl> {
            use crate::gpu::concurrent_metal::{Q3ksMtl, SharedBuffer};
            // ⚠️ Q3K IS NOT REACHABLE FOR A NATIVE-Q4_K GGUF, and L159b's attempted fix was WRONG.
            // `bf16_matrix` has no tensor to return on such a model, so the flag silently no-ops.
            // Routing through `q4k_matrix` instead does NOT help: that helper is itself built on
            // `get_f32_shaped(..)`, i.e. it also wants an f32/bf16 source. Wiring it produced 92.75
            // tok/s of PURE GARBAGE — 4.3x faster than the bandwidth roofline allows, which is the
            // tell that the FFN was reading nothing rather than reading less.
            //
            // A real fix has to dequantize the NATIVE Q4_K BLOCKS (per super-block, from the mmapped
            // GGUF bytes) and requantize to Q3_K. That is a genuine piece of work, not a source swap.
            // Until then this stays bf16-only and simply does nothing on GGUF —; measured out
            // the bandwidth math that makes it worth building (FFN is 82% of the weight stream).
            // L212 — bf16 FIRST (lossless source when the model has one), then the NATIVE-Q4_K
            // route for GGUFs that have no bf16 at all. Before this, the `bf16_matrix` call below
            // was the ONLY source, so on a native-Q4_K GGUF it returned None and ARF_Q3K_FFN was
            // a silent no-op — the flag appeared to "work" (server ran, output correct) while
            // changing nothing, which is how it measured 17.2 tok/s = baseline noise.
            let q = match w.bf16_matrix(name, out, in_) {
                Ok(m) => arf_core::tensor::Q3KMatrix::from_f32(&m.to_f32(), out, in_),
                Err(_) => w.q3k_matrix_from_gguf_q4k(name, out, in_)?,
            };
            let u = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let pack_u8 = |d: &[u8]| -> Vec<u32> {
                d.chunks_exact(4)
                    .map(|q| {
                        (q[0] as u32)
                            | ((q[1] as u32) << 8)
                            | ((q[2] as u32) << 16)
                            | ((q[3] as u32) << 24)
                    })
                    .collect()
            };
            let pack_i8 = |d: &[i8]| -> Vec<u32> {
                d.chunks_exact(4)
                    .map(|q| {
                        (q[0] as u8 as u32)
                            | ((q[1] as u8 as u32) << 8)
                            | ((q[2] as u8 as u32) << 16)
                            | ((q[3] as u8 as u32) << 24)
                    })
                    .collect()
            };
            let mut dd = Vec::with_capacity(q.d().len() * 2);
            for (a, b) in q.d().iter().zip(q.dmin()) {
                dd.push(*a);
                dd.push(*b);
            }
            let sc = pack_u8(q.scales());
            let mn = pack_i8(q.mins());
            let (ql, qh, s, mi, ddb, di) = (
                SharedBuffer::from_pod(ctx, q.ql(), u, &format!("{name}.ql"))?,
                SharedBuffer::from_pod(ctx, q.qh(), u, &format!("{name}.qh"))?,
                SharedBuffer::from_pod(ctx, &sc, u, &format!("{name}.q3sc"))?,
                SharedBuffer::from_pod(ctx, &mn, u, &format!("{name}.q3mn"))?,
                SharedBuffer::from_pod(ctx, &dd, u, &format!("{name}.q3dd"))?,
                SharedBuffer::from_pod(
                    ctx,
                    &[1u32, in_ as u32, out as u32, 0u32],
                    u,
                    &format!("{name}.q3dims"),
                )?,
            );
            Some(Q3ksMtl {
                ql: ql.mtl,
                qh: qh.mtl,
                scales: s.mtl,
                mins: mi.mtl,
                dd: ddb.mtl,
                dims: di.mtl,
                n: out,
            })
        };
    let fold_gemma_norm = is_gemma && !w.is_gguf();
    // S1b norm-weight uploader: returns (wgpu view, optional MTL view). `shared_norm` makes a
    // dual-view SharedBuffer when the island is on, else a plain storage_init + None.
    // L364 — ONE definition on both platforms now that `concurrent_metal::MtlBuf` resolves
    // everywhere (the off-macOS twin is uninhabited, so this is always `None` there). It used to
    // be `()` off-macOS, which made `t.1` a unit at seven call sites that store `Option<MtlBuf>`.
    type NormMtl = Option<crate::gpu::concurrent_metal::MtlBuf>;
    #[cfg(target_os = "macos")]
    let shared_norm =
        |ctx: &crate::gpu::GpuContext, name: &str, v: &[f32]| -> Result<(wgpu::Buffer, NormMtl)> {
            if std::env::var_os("ARF_MSL_GEMV").is_some() {
                if let Some(sb) = crate::gpu::concurrent_metal::SharedBuffer::from_pod(
                    ctx,
                    v,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    name,
                ) {
                    return Ok((sb.wgpu, Some(crate::gpu::concurrent_metal::MtlBuf(sb.mtl))));
                }
            }
            Ok((ctx.storage_init(name, v), None))
        };
    // L364 — off macOS this is the same shape as the island-down branch above: a plain storage
    // buffer and `None`. It returned `()` before `NormMtl` was unified; now the only difference
    // between the two arms is that this one never tries the SharedBuffer path at all.
    #[cfg(not(target_os = "macos"))]
    let shared_norm =
        |ctx: &crate::gpu::GpuContext, name: &str, v: &[f32]| -> Result<(wgpu::Buffer, NormMtl)> {
            Ok((ctx.storage_init(name, v), None))
        };
    // Megakernel S1b: norm weights become dual-view SharedBuffers (macOS + ARF_MSL_GEMV) so
    // the island norms read them; the (1+w) fold is applied to the f32 vec BEFORE from_pod, so
    // the SHARED bytes carry the folded weight (GGUF arrives pre-shifted → no fold → plain w).
    // `_mtl` is None off-macOS / off-flag / on SharedBuffer-alloc failure → wgpu path unaffected.
    // Length-parameterized vector uploader. `up_vec` below is hardcoded to `h` (norm weights);
    // GDN carries 48- and 128-element vectors, so those need the length passed explicitly.
    // No gemma +1 fold here — these are decay/bias/state params, not norm weights.
    let up_vec_len = |name: &str, n: usize| -> Result<(wgpu::Buffer, NormMtl)> {
        let v = w.f32_vec(name, n)?;
        shared_norm(ctx, name, &v)
    };
    // Flat uploader for a RANK-2 tensor consumed as one contiguous buffer — qwen35's depthwise
    // conv1d is [channels, kernel] in the file but the kernel wants it flat, and `f32_vec`
    // rejects anything that is not rank-1.
    let up_mat_flat = |name: &str, rows: usize, cols: usize| -> Result<(wgpu::Buffer, NormMtl)> {
        let m = w.bf16_matrix(name, rows, cols)?;
        shared_norm(ctx, name, &m.to_f32())
    };
    let up_vec = |name: &str| -> Result<(wgpu::Buffer, NormMtl)> {
        let mut v = w.f32_vec(name, h)?;
        if fold_gemma_norm {
            for x in &mut v {
                *x += 1.0;
            }
        }
        shared_norm(ctx, name, &v)
    };
    // Per-head q/k norm ([head_dim]) uploader, with the same Gemma `+1` fold. The
    // norm length is the LAYER's head_dim (Gemma 4 global layers use a different
    // head_dim from the base/sliding ones), so the caller passes it explicitly.
    let up_head_norm = |name: &str, head_dim: usize| -> Result<(wgpu::Buffer, NormMtl)> {
        let mut v = w.f32_vec(name, head_dim)?;
        if fold_gemma_norm {
            for x in &mut v {
                *x += 1.0;
            }
        }
        shared_norm(ctx, name, &v)
    };

    // The embedding TABLE is uploaded bf16 (its native checkpoint dtype): halves its
    // size and keeps a large vocab×hidden table under Metal's ~4 GB single-buffer
    // limit (Gemma 4: 262144×5376 = 5.6 GB f32 → 2.8 GB bf16). The bf16 `embed`/
    // `embed_tok` kernels read it (widening per element); the f32 `embed` kernel
    // still serves the hidden-state gathers. bf16 is exact-to-checkpoint here.
    // Embed table MTL view (megakernel): so the island can run embed first → whole token
    // single-queue (kills the wgpu→island cross-queue handoff, ~6.5ms inter-token GPU idle).
    #[cfg(target_os = "macos")]
    let mut embed_mtl: Option<crate::gpu::concurrent_metal::MtlBuf> = None;
    // 2026-09-23 — THE GGUF'S OWN Q4_K TABLE when it has one. Qwen3.8-27B ships token_embd as Q4_K
    // (0.67 GB); widened to bf16 it was 2.54 GB held resident to gather 1-8 rows a step — memory
    // the KV pool wants. Every gather reads it through a `*_q4k` kernel (ggml's dequant, exact
    // f32). Not for tied models: their lm_head aliases this table and reads it as bf16.
    #[cfg(target_os = "macos")]
    let embed_q4k_blocks: Option<&[u8]> = if std::env::var_os("ARF_NO_EMBED_Q4K").is_none()
        && std::env::var_os("ARF_MSL_GEMV").is_some()
        && !cfg.tie_word_embeddings
        && h.is_multiple_of(256)
    {
        w.q4k_blocks("model.embed_tokens.weight")
            .filter(|(_, r, c)| (*r, *c) == (cfg.vocab_size, h))
            .map(|(b, _, _)| b)
    } else {
        None
    };
    #[cfg(not(target_os = "macos"))]
    let embed_q4k_blocks: Option<&[u8]> = None;
    // Only the macOS branch below fills it; elsewhere it stays `None`.
    #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
    let mut embed_q4k_buf: Option<wgpu::Buffer> = None;
    #[cfg(target_os = "macos")]
    if let Some(blocks) = embed_q4k_blocks {
        if let Some(sb) = crate::gpu::concurrent_metal::SharedBuffer::from_pod(
            ctx,
            blocks,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "embed_q4k",
        ) {
            embed_mtl = Some(crate::gpu::concurrent_metal::MtlBuf(sb.mtl));
            eprintln!(
                "[mem] token embedding kept as the GGUF's Q4_K: {:.2} GB (bf16 would be {:.2} GB; ARF_NO_EMBED_Q4K=1 for bf16)",
                blocks.len() as f64 / 1e9,
                (cfg.vocab_size * h * 2) as f64 / 1e9
            );
            embed_q4k_buf = Some(sb.wgpu);
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = embed_q4k_blocks;
    let embed_q4k = embed_q4k_buf.is_some();
    let embed = if let Some(b) = embed_q4k_buf {
        b
    } else {
        let m = w.bf16_matrix("model.embed_tokens.weight", cfg.vocab_size, h)?;
        // Upload the bf16 bits DIRECTLY (no `to_f32()` widen): on Gemma 4 that drops
        // a ~5.6 GB transient f32 copy of the 262144×5376 table at load. Bit-identical
        // to `storage_init_bf16(&m.to_f32())` (widen-then-truncate is a no-op).
        #[cfg(target_os = "macos")]
        if std::env::var_os("ARF_MSL_GEMV").is_some() {
            // pack the bf16 u16 bits into u32 (two per word, low=even) — the layout
            // storage_init_bf16_from_u16 produces + the embed kernel reads.
            let bits = m.bits();
            let packed: Vec<u32> = bits
                .chunks_exact(2)
                .map(|c| (c[0] as u32) | ((c[1] as u32) << 16))
                .collect();
            if let Some(sb) = crate::gpu::concurrent_metal::SharedBuffer::from_pod(
                ctx,
                &packed,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                "embed",
            ) {
                embed_mtl = Some(crate::gpu::concurrent_metal::MtlBuf(sb.mtl));
                sb.wgpu
            } else {
                ctx.storage_init_bf16_from_u16("model.embed_tokens.weight", m.bits())
            }
        } else {
            ctx.storage_init_bf16_from_u16("model.embed_tokens.weight", m.bits())
        }
        #[cfg(not(target_os = "macos"))]
        ctx.storage_init_bf16_from_u16("model.embed_tokens.weight", m.bits())
    };
    // Commit the embedding to GPU residency NOW, before the ~18 GB of layer weights
    // are allocated. On the wgpu Metal backend, a buffer created via
    // create_buffer_init but never referenced in a submitted command before total
    // resident allocation crosses ~14-16 GiB has its pages reclaimed and reads back
    // ZEROED — and the loss is NOT recoverable by a later touch (verified in
    // isolation: examples/big_buffer_probe). Touching each buffer right after
    // creation (a 4-byte copy in a submitted command) commits it so it survives the
    // later allocations. See GpuContext::make_resident.
    ctx.make_resident(&[&embed]);
    load_trace("after embed upload");

    let mut layers = Vec::with_capacity(cfg.num_layers);
    use crate::gpu::PackedExperts;
    // Pack several `[rows, cols]` expert matrices back-to-back into ONE resident
    // buffer, indexed by `expert*rows` offset. bf16 (parity path), Q4_0, or Q4_K-lite
    // (both fit the 30B). The quantized branches concatenate each expert's packed
    // codes/scales(/mins) one expert at a time — the per-expert f32 transient is
    // freed each iteration, so peak RAM is one expert (~6 MB), not all of them.
    // bf16 experts concatenate full f32 (the parity path; OOMs on the 30B, so the
    // 30B must use Q4/Q4K — guarded below).
    let pack_experts =
        |names: &[String], rows: usize, cols: usize, label: &str| -> Result<PackedExperts> {
            match quant {
                Quant::Q4 => {
                    let mut codes = Vec::new();
                    let mut scales = Vec::new();
                    for n in names {
                        let q = w.q4_matrix(n, rows, cols)?;
                        codes.extend_from_slice(q.codes());
                        scales.extend_from_slice(q.scales());
                    }
                    Ok(PackedExperts::Q4 {
                        codes: ctx.storage_init(&format!("{label}.codes"), &codes),
                        scales: ctx.storage_init_q4_scales(&format!("{label}.scales"), &scales),
                    })
                }
                Quant::Q4K => {
                    let mut codes = Vec::new();
                    let mut scales = Vec::new();
                    let mut mins = Vec::new();
                    for n in names {
                        let q = w.q4k_matrix(n, rows, cols)?;
                        codes.extend_from_slice(q.codes());
                        scales.extend_from_slice(q.scales());
                        mins.extend_from_slice(q.mins());
                    }
                    Ok(PackedExperts::Q4K {
                        codes: ctx.storage_init(&format!("{label}.codes"), &codes),
                        scales: ctx.storage_init_q4_scales(&format!("{label}.scales"), &scales),
                        mins: ctx.storage_init_q4_scales(&format!("{label}.mins"), &mins),
                    })
                }
                Quant::Q4KS => {
                    // Concatenate each expert's Q4_K_S packing back-to-back (the same
                    // per-expert offset contract as Q4/Q4K): codes (u32), scales (u8),
                    // mins (i8), and the interleaved [d, dmin] super-block scales (f32).
                    // Mirrors the dense Q4_K_S arm above, one expert at a time so peak
                    // RAM is one expert's f32 transient, not all of them.
                    //
                    // ⚠️ L109 (2026-08-10) — THIS SET IS A SECOND COPY OF THE MoE WEIGHTS AND IS
                    // NOT DISPATCHED WHEN THE ISLAND IS UP. It is byte-identical to what the
                    // island `MoeMtl` already wraps NO-COPY from the sidecar, but it goes through
                    // `ctx.storage_init` into wgpu-managed ANONYMOUS memory. Measured on
                    // qwen3-coder-30B: **16.3 GB of dirty anonymous pages** (48 layers x 128
                    // experts x 3 matrices x 768 x 2048 @ ~4.5 bits), which is what put the
                    // daemon at a 25 GB footprint and made it die in encrypted swap (L108) while
                    // llama — whose weights stay clean/file-backed — degraded ~3% on the same box.
                    //
                    // A dispatch probe on BOTH wgpu MoE sites under real conc1/conc8 HTTP traffic
                    // counted **0** dispatches with the island on, and **8** with
                    // `ARF_NO_MSL_GEMV=1` (positive control) — the island megakernel serves
                    // prefill AND decode. So when the island owns the MoE path we build a
                    // 16-byte PLACEHOLDER instead of the full set: the enum shape, the binding
                    // contract, and the non-island fallback all stay exactly as they were.
                    // Opt out (restore the full allocation) with ARF_MOE_WGPU_FULL=1.
                    // (Measured 2026-08-10.)
                    #[cfg(target_os = "macos")]
                    if std::env::var_os("ARF_MSL_GEMV").is_some()
                        && std::env::var_os("ARF_MOE_WGPU_FULL").is_none()
                    {
                        return Ok(PackedExperts::Q4KS {
                            codes: ctx.storage_init(&format!("{label}.codes"), &[0u32; 4]),
                            scales: ctx.storage_init_u8(&format!("{label}.scales"), &[0u8; 16]),
                            mins: ctx.storage_init_q8(&format!("{label}.mins"), &[0i8; 16]),
                            dd: ctx.storage_init(&format!("{label}.dd"), &[0f32; 4]),
                        });
                    }
                    let mut codes = Vec::new();
                    let mut scales = Vec::new();
                    let mut mins = Vec::new();
                    let mut dd = Vec::new();
                    for n in names {
                        let q = w.q4ks_matrix(n, rows, cols)?;
                        codes.extend_from_slice(q.codes());
                        scales.extend_from_slice(q.scales());
                        mins.extend_from_slice(q.mins());
                        for (dv, dmv) in q.d().iter().zip(q.dmin()) {
                            dd.push(*dv);
                            dd.push(*dmv);
                        }
                    }
                    Ok(PackedExperts::Q4KS {
                        codes: ctx.storage_init(&format!("{label}.codes"), &codes),
                        scales: ctx.storage_init_u8(&format!("{label}.scales"), &scales),
                        mins: ctx.storage_init_q8(&format!("{label}.mins"), &mins),
                        dd: ctx.storage_init(&format!("{label}.dd"), &dd),
                    })
                }
                _ => {
                    let mut all = Vec::with_capacity(names.len() * rows * cols);
                    for n in names {
                        all.extend_from_slice(&w.bf16_matrix(n, rows, cols)?.to_f32());
                    }
                    Ok(PackedExperts::Bf16(ctx.storage_init_bf16(label, &all)))
                }
            }
        };
    // Shared experts: None when there are none (no empty buffers).
    let pack_shared = |names: &[String],
                       rows: usize,
                       cols: usize,
                       label: &str|
     -> Result<Option<PackedExperts>> {
        if names.is_empty() {
            Ok(None)
        } else {
            Ok(Some(pack_experts(names, rows, cols, label)?))
        }
    };

    // ALL-EXPERTS raw-Metal views (macOS + ARF_MSL_GEMV, Q4KS only) for the megakernel's
    // INDIRECT GEMV. Concatenates each expert's Q4_K_S codes/scales/mins/dd back-to-back (the
    // SAME `expert*cols` offset contract the wgpu PackedExperts::Q4KS uses + the gemv_q4ks_id
    // kernel reads), then packs scales(u8)/mins(i8) → 4-per-u32 (the kernel reads them as
    // `device const uint*`, IDENTICAL to the dense Q4ksMtl packing) and uploads via SharedBuffer.
    // `cols` per expert = `rows` here is the per-expert OUTPUT column count (= moe_inter for
    // gate/up, hidden for down). Returns `None` off the MSL path so the wgpu path is unchanged.
    #[cfg(target_os = "macos")]
    let moe_mtl = |names: &[String],
                   rows: usize,
                   cols: usize,
                   label: &str|
     -> Result<Option<std::sync::Arc<crate::gpu::concurrent_metal::MoeMtl>>> {
        use crate::gpu::concurrent_metal::{MoeMtl, SharedBuffer};
        if std::env::var_os("ARF_MSL_GEMV").is_none() || !matches!(quant, Quant::Q4KS) {
            return Ok(None);
        }
        let u = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        // WEIGHT CACHE read-hit: wrap all four sub-buffers no-copy from the sidecar and SKIP
        // the ~per-expert transcode loop entirely (the real load-in-seconds win). Only when
        // every label is present (a full hit); otherwise fall through to transcode + cache.
        // The island MoeMtl dd is the packed half2 stream, cached under "{label}.mtl.ddh".
        // A stale (pre-ddh) sidecar misses that key → fall through to transcode + re-cache.
        if crate::weight_cache::is_read_hit() {
            if let (Some(c), Some(s), Some(mi_b), Some(ddhb)) = (
                SharedBuffer::from_cache(ctx, &format!("{label}.mtl.codes"), u),
                SharedBuffer::from_cache(ctx, &format!("{label}.mtl.scale"), u),
                SharedBuffer::from_cache(ctx, &format!("{label}.mtl.min"), u),
                SharedBuffer::from_cache(ctx, &format!("{label}.mtl.ddh"), u),
            ) {
                return Ok(Some(std::sync::Arc::new(MoeMtl {
                    codes: c.mtl,
                    scales: s.mtl,
                    mins: mi_b.mtl,
                    dd: ddhb.mtl,
                    n: rows,
                })));
            }
            // Partial hit (shouldn't happen) → fall through and transcode for correctness.
        }
        let mut codes: Vec<u32> = Vec::new();
        let mut scales: Vec<u8> = Vec::new();
        let mut mins: Vec<i8> = Vec::new();
        let mut dd: Vec<f32> = Vec::new();
        for n in names {
            let q = w.q4ks_matrix(n, rows, cols)?;
            codes.extend_from_slice(q.codes());
            scales.extend_from_slice(q.scales());
            mins.extend_from_slice(q.mins());
            for (dv, dmv) in q.d().iter().zip(q.dmin()) {
                dd.push(*dv);
                dd.push(*dmv);
            }
        }
        // Pack u8 scales / i8 mins → 4-per-u32 (the kernel reads them as uint words).
        let pack_u8 = |d: &[u8]| -> Vec<u32> {
            d.chunks_exact(4)
                .map(|q| {
                    (q[0] as u32)
                        | ((q[1] as u32) << 8)
                        | ((q[2] as u32) << 16)
                        | ((q[3] as u32) << 24)
                })
                .collect()
        };
        let pack_i8 = |d: &[i8]| -> Vec<u32> {
            d.chunks_exact(4)
                .map(|q| {
                    (q[0] as u8 as u32)
                        | ((q[1] as u8 as u32) << 8)
                        | ((q[2] as u8 as u32) << 16)
                        | ((q[3] as u8 as u32) << 24)
                })
                .collect()
        };
        let sc_p = pack_u8(&scales);
        let mn_p = pack_i8(&mins);
        // Island dd = packed half2 "ddh" (one u32 per super-block pair, bit-lossless f16
        // repack — the interleaved f32 `dd` above is only the packing intermediate here;
        // the wgpu PackedExperts::Q4KS keeps its own f32 dd via pack_experts).
        let ddh = crate::gpu::concurrent_metal::dd_to_ddh(&dd);
        let (Some(c), Some(s), Some(mi_b), Some(ddhb)) = (
            SharedBuffer::from_pod(ctx, &codes, u, &format!("{label}.mtl.codes")),
            SharedBuffer::from_pod(ctx, &sc_p, u, &format!("{label}.mtl.scale")),
            SharedBuffer::from_pod(ctx, &mn_p, u, &format!("{label}.mtl.min")),
            SharedBuffer::from_pod(ctx, &ddh, u, &format!("{label}.mtl.ddh")),
        ) else {
            return Ok(None);
        };
        Ok(Some(std::sync::Arc::new(MoeMtl {
            codes: c.mtl,
            scales: s.mtl,
            mins: mi_b.mtl,
            dd: ddhb.mtl,
            n: rows,
        })))
    };

    for i in 0..cfg.num_layers {
        if i % 8 == 0 {
            load_trace(&format!("layer {i}/{}", cfg.num_layers));
        }
        let p = format!("model.layers.{i}");
        let mlp = match &cfg.mlp {
            MlpKind::Dense => {
                // Q3K (3-bit) MTL views of gate/up/down for the megakernel island — the
                // bandwidth-floor breaker (ARF_Q3K_FFN). Built from the bf16 weights, in
                // parallel with the Q4KS gate/up/down_proj below (which the wgpu oracle uses).
                #[cfg(target_os = "macos")]
                // Q3K (3-bit) FFN views for the island — the bandwidth lever. The FFN is 82%
                // of muse-glimmer's per-token weight stream (11.66 of 14.15 GB), so 4.5 -> 3.4
                // bits/weight cuts total traffic ~20%. `q3k_mtl` now transcodes from native Q4_K
                // when there is no bf16 (L159 — reading bf16 unconditionally made this flag a
                // SILENT no-op on every native-Q4_K GGUF).
                let q3k = if std::env::var_os("ARF_Q3K_FFN").is_some() && h.is_multiple_of(256) {
                    let g = q3k_mtl(
                        &format!("{p}.mlp.gate_proj.weight"),
                        cfg.intermediate_size,
                        h,
                    );
                    let u = q3k_mtl(&format!("{p}.mlp.up_proj.weight"), cfg.intermediate_size, h);
                    let dn = q3k_mtl(
                        &format!("{p}.mlp.down_proj.weight"),
                        h,
                        cfg.intermediate_size,
                    );
                    match (g, u, dn) {
                        (Some(g), Some(u), Some(dn)) => Some(std::sync::Arc::new((g, u, dn))),
                        _ => None,
                    }
                } else {
                    None
                };
                #[cfg(target_os = "macos")]
                // ARF_Q8_DOWN_LAYERS=n limits the experiment to the first n layers (bisection).
                let q8_layers: usize = std::env::var("ARF_Q8_DOWN_LAYERS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(usize::MAX);
                #[cfg(target_os = "macos")]
                let down_q8 = if std::env::var_os("ARF_Q8_DOWN").is_some() && i < q8_layers {
                    let r = q8_mtl(
                        &format!("{p}.mlp.down_proj.weight"),
                        h,
                        cfg.intermediate_size,
                    )
                    .map(std::sync::Arc::new);
                    eprintln!(
                        "[msl] ARF_Q8_DOWN: layer {i} down_proj at Q8 — {}",
                        if r.is_some() {
                            "built"
                        } else {
                            "NOT built (no source tensor)"
                        }
                    );
                    r
                } else {
                    None
                };
                GpuMlp::Dense(Box::new(crate::gpu::GpuDenseMlp {
                    gate_proj: up_w(
                        &format!("{p}.mlp.gate_proj.weight"),
                        cfg.intermediate_size,
                        h,
                    )?,
                    up_proj: up_w(&format!("{p}.mlp.up_proj.weight"), cfg.intermediate_size, h)?,
                    down_proj: up_w(
                        &format!("{p}.mlp.down_proj.weight"),
                        h,
                        cfg.intermediate_size,
                    )?,
                    #[cfg(target_os = "macos")]
                    q3k,
                    #[cfg(target_os = "macos")]
                    down_q8,
                }))
            }
            MlpKind::Moe {
                num_experts,
                top_k,
                shared_experts,
                moe_intermediate,
                norm_topk,
            } => {
                let mi = *moe_intermediate;
                let names = |proj: &str| -> Vec<String> {
                    (0..*num_experts)
                        .map(|e| format!("{p}.mlp.experts.{e}.{proj}.weight"))
                        .collect()
                };
                let shared_names = |proj: &str| -> Vec<String> {
                    (0..*shared_experts)
                        .map(|_| format!("{p}.mlp.shared_expert.{proj}.weight"))
                        .collect()
                };
                // Empty shared buffers (len 0) are fine — the block skips them when
                // shared_experts == 0.
                // All-experts raw-Metal views (macOS + MSL) for the indirect-GEMV megakernel.
                #[cfg(target_os = "macos")]
                let gate_mtl = moe_mtl(&names("gate_proj"), mi, h, &format!("{p}.moe.gate"))?;
                #[cfg(target_os = "macos")]
                let up_mtl = moe_mtl(&names("up_proj"), mi, h, &format!("{p}.moe.up"))?;
                #[cfg(target_os = "macos")]
                let down_mtl = moe_mtl(&names("down_proj"), h, mi, &format!("{p}.moe.down"))?;
                GpuMlp::Moe(Box::new(crate::gpu::GpuMoe {
                    router: up_w(&format!("{p}.mlp.gate.weight"), *num_experts, h)?,
                    gate: pack_experts(&names("gate_proj"), mi, h, &format!("{p}.moe.gate"))?,
                    up: pack_experts(&names("up_proj"), mi, h, &format!("{p}.moe.up"))?,
                    down: pack_experts(&names("down_proj"), h, mi, &format!("{p}.moe.down"))?,
                    shared_gate: pack_shared(
                        &shared_names("gate_proj"),
                        mi,
                        h,
                        &format!("{p}.moe.sgate"),
                    )?,
                    shared_up: pack_shared(
                        &shared_names("up_proj"),
                        mi,
                        h,
                        &format!("{p}.moe.sup"),
                    )?,
                    shared_down: pack_shared(
                        &shared_names("down_proj"),
                        h,
                        mi,
                        &format!("{p}.moe.sdown"),
                    )?,
                    num_experts: *num_experts,
                    top_k: *top_k,
                    shared_experts: *shared_experts,
                    moe_inter: mi,
                    norm_topk: *norm_topk,
                    #[cfg(target_os = "macos")]
                    gate_mtl,
                    #[cfg(target_os = "macos")]
                    up_mtl,
                    #[cfg(target_os = "macos")]
                    down_mtl,
                }))
            }
        };
        // Gemma's two extra norms (pre/post feedforward); None for Llama/Qwen. Each uploader
        // now returns (wgpu buffer, optional MTL view); split them for the struct + the _mtl.
        // L155 — PLACEMENT, not formula: any four-norm model has these two. The gguf loader
        // normalises each arch's ggml names onto these HF names, so this reads the same for
        // Gemma and Muse Glimmer.
        let (pre_ffn_pair, post_ffn_pair) = if cfg.has_post_norms() {
            (
                Some(up_vec(&format!("{p}.pre_feedforward_layernorm.weight"))?),
                Some(up_vec(&format!("{p}.post_feedforward_layernorm.weight"))?),
            )
        } else {
            (None, None)
        };
        let input_norm_p = up_vec(&format!("{p}.input_layernorm.weight"))?;
        let post_norm_p = up_vec(&format!("{p}.post_attention_layernorm.weight"))?;
        // Per-layer attention geometry: SLIDING layers use the base head_dim/kv_heads
        // (full rotary); Gemma 4 GLOBAL layers (every global_every-th) use the global
        // profile (512 head_dim, 4 kv heads, 0.25 partial rotary). The global predicate
        // matches `window` below. `layer_geometry` returns (head_dim, kv_heads,
        // rotary_dim) per layer — the projection out-dims, KV stride, attention dims
        // and RoPE rotary dim all derive from it. Gemma 3 / Llama / Qwen → base, full.
        let (l_head_dim, l_kv_heads, l_rotary_dim) = cfg.layer_geometry(i);
        // qwen35: attention layers pack q at 2x head_dim, GDN layers use ssm_inner. Every
        // other arch returns nh*head_dim, so this is a no-op for them.
        let l_q_dim = cfg.layer_q_dim(i);
        let l_kv_dim = l_kv_heads * l_head_dim;
        // Gemma interleaves sliding-window local layers with periodic global ones;
        // local layers also rotate with the local RoPE base. Llama/Qwen are global.
        let window = match &cfg.attn {
            AttnKind::HybridLocalGlobal {
                window,
                global_every,
                ..
            } => {
                let is_global = (i + 1) % global_every.max(&1) == 0;
                if is_global {
                    None
                } else {
                    Some(*window)
                }
            }
            // qwen35: attention layers are FULL causal (no sliding window); GDN layers never
            // reach here — they have no attention at all.
            AttnKind::HybridSsmAttn { .. } => None,
            AttnKind::Causal => None,
        };
        // Q/K/V projections. Gemma 4's GLOBAL layers share K as V (the GGUF ships NO
        // `attn_v.weight` for them — `shared_kv_layers`), while sliding layers carry a
        // distinct V. So load V if present, else clone the K buffer (Arc-backed handle,
        // no extra upload). Gemma 4's per-head weightless V-norm (`cfg.value_norm`) is a
        // separate post-projection step. Out-dim is the LAYER's kv_dim.
        // ── GDN LAYER DETECTION (qwen35) ────────────────────────────────────────────────
        // 3 of every 4 layers are gated-delta-net and carry NO attention tensors at all. The
        // switch is the config schedule, cross-checked against the file: if the layer claims
        // GDN, its ssm_* must actually be present, otherwise fail loudly rather than load a
        // half-built layer.
        let is_gdn = match &cfg.attn {
            AttnKind::HybridSsmAttn { attn_every, .. } => (i + 1) % attn_every.max(&1) != 0,
            _ => false,
        };
        if is_gdn && w.shape(&format!("{p}.linear_attn.A_log")).is_none() {
            return Err(ArfError::model_load(format!(
                "layer {i} is scheduled as GDN (attn_every) but has no linear_attn.A_log"
            )));
        }
        // GDN layers: q/k/v/o are the SSM input projection (fused attn_qkv, row-sliced) and
        // out_proj. They are stored in the same GpuLayer slots so the residency set, the
        // weight-cache and the megakernel bookkeeping all keep working unchanged; the forward
        // pass switches on `gdn.is_some()`.
        // On a GDN layer these are the SSM input projection (the fused attn_qkv row-slices),
        // not attention — but they load through the same slots, so the code below is shared.
        let k_proj = up_w(&format!("{p}.self_attn.k_proj.weight"), l_kv_dim, h)?;
        let v_name = format!("{p}.self_attn.v_proj.weight");
        let v_proj = if w.shape(&v_name).is_some() {
            up_w(&v_name, l_kv_dim, h)?
        } else {
            k_proj.clone()
        };
        // Per-head q/k norms (Qwen3/Gemma); now in scope (l_head_dim computed above).
        // GDN layers have NO per-head q/k norms — those tensors exist only on the attention
        // layers (3, 7, 11, ...). Presence in the file is the authority, not the config flag:
        // qwen35 sets qk_norm=true for its attention layers while 3 of 4 layers have neither.
        let q_norm_p = if cfg.qk_norm && w.shape(&format!("{p}.self_attn.q_norm.weight")).is_some()
        {
            Some(up_head_norm(
                &format!("{p}.self_attn.q_norm.weight"),
                l_head_dim,
            )?)
        } else {
            None
        };
        let k_norm_p = if cfg.qk_norm && w.shape(&format!("{p}.self_attn.k_norm.weight")).is_some()
        {
            Some(up_head_norm(
                &format!("{p}.self_attn.k_norm.weight"),
                l_head_dim,
            )?)
        } else {
            None
        };
        // Split each (buffer, mtl) pair: the buffer goes in the struct field, the mtl into a
        // parallel per-layer _mtl (macOS island norms read these).
        // L138 — gated-delta-net per-head statics, loaded BEFORE norm_mtls because their raw-Metal
        // views live there (the megakernel GDN branch binds them directly). up_vec_len returns the
        // (buffer, mtl) pair; the earlier code discarded the view with `.0`. Non-GDN layers skip
        // these entirely, so no other arch pays a load.
        let gdn_vecs = if is_gdn {
            let (rank_n, state_n) = match &cfg.attn {
                AttnKind::HybridSsmAttn {
                    dt_rank, ssm_state, ..
                } => (*dt_rank, *ssm_state),
                _ => unreachable!("is_gdn implies HybridSsmAttn"),
            };
            Some((
                up_vec_len(&format!("{p}.linear_attn.dt_bias"), rank_n)?,
                up_vec_len(&format!("{p}.linear_attn.A_log"), rank_n)?,
                up_vec_len(&format!("{p}.linear_attn.norm.weight"), state_n)?,
            ))
        } else {
            None
        };
        #[cfg(target_os = "macos")]
        let norm_mtls = crate::gpu::NormMtls {
            input_norm: input_norm_p.1,
            post_norm: post_norm_p.1,
            pre_ffn_norm: pre_ffn_pair.as_ref().and_then(|p| p.1.clone()),
            post_ffn_norm: post_ffn_pair.as_ref().and_then(|p| p.1.clone()),
            q_norm: q_norm_p.as_ref().and_then(|p| p.1.clone()),
            k_norm: k_norm_p.as_ref().and_then(|p| p.1.clone()),
            ssm_dt: gdn_vecs.as_ref().and_then(|t| t.0 .1.clone()),
            ssm_a: gdn_vecs.as_ref().and_then(|t| t.1 .1.clone()),
            ssm_norm: gdn_vecs.as_ref().and_then(|t| t.2 .1.clone()),
        };
        // Build the GDN block for a gated-delta-net layer. Uses the SAME up_w/up_vec helpers as
        // everything else, so Q4_K natives, the weight cache and residency all apply unchanged.
        let gdn = if is_gdn {
            let (inner, rank, state, groups_n) = match &cfg.attn {
                AttnKind::HybridSsmAttn {
                    ssm_inner,
                    dt_rank,
                    ssm_state,
                    groups,
                    ..
                } => (*ssm_inner, *dt_rank, *ssm_state, *groups),
                _ => unreachable!("is_gdn implies HybridSsmAttn"),
            };
            let conv_kernel = match &cfg.attn {
                AttnKind::HybridSsmAttn { conv_kernel, .. } => *conv_kernel,
                _ => 4,
            };
            let conv_w = w
                .shape(&format!("{p}.linear_attn.conv1d.weight"))
                .ok_or_else(|| ArfError::model_load(format!("layer {i} missing conv1d")))?;
            let conv_cols = conv_w.iter().product::<usize>() / conv_kernel;
            let conv_dim = inner + 2 * (state * groups_n);
            // Reuse the pairs hoisted above norm_mtls — one load, both consumers.
            let (gdn_dt_pair, gdn_a_pair, gdn_norm_pair) =
                gdn_vecs.as_ref().expect("is_gdn implies gdn_vecs");
            Some(Box::new(GpuGdn {
                // L137 — the FUSED projection, the one the megakernel GDN branch uses. Held whole
                // because the GGUF's [6144|2048|2048] packing is NOT the [2048|2048|6144] order
                // gdn_tile reads (gguf.rs attn_qkv arm has the full note).
                qkv: {
                    // L148 — ARF_GDN_WEIGHT_TRACE: confirm WHAT this load actually resolved to.
                    // The fused entry and its three RowSlice siblings share one desc_idx, so a
                    // wrong resolution here is invisible in the tensor count.
                    if std::env::var_os("ARF_GDN_WEIGHT_TRACE").is_some() && i == 0 {
                        let nm = format!("{p}.linear_attn.qkv_proj.weight");
                        eprintln!(
                            "[gdn-w] {nm}: shape={:?} asked=({conv_dim},{h}) native_q4k={}",
                            w.shape(&nm),
                            w.q4ks_matrix_from_gguf_q4k(&nm, conv_dim, h).is_some(),
                        );
                    }
                    up_w(&format!("{p}.linear_attn.qkv_proj.weight"), conv_dim, h)?
                },
                // The output gate z; on a GDN layer `attn_gate.weight` IS this ([h -> value_dim]).
                qkv_gate: up_w(&format!("{p}.self_attn.gate_proj.weight"), inner, h)?,
                // The fused attn_qkv row-slices land here as q/k/v (see gguf.rs RowSlice).
                in_q: up_w(&format!("{p}.self_attn.q_proj.weight"), inner, h)?,
                in_k: up_w(&format!("{p}.self_attn.k_proj.weight"), l_kv_dim, h)?,
                in_v: up_w(&format!("{p}.self_attn.v_proj.weight"), l_kv_dim, h)?,
                alpha: up_w(&format!("{p}.linear_attn.alpha_proj.weight"), rank, h)?,
                beta: up_w(&format!("{p}.linear_attn.beta_proj.weight"), rank, h)?,
                // Flat f32: conv_cols channels x conv_kernel taps, depthwise.
                conv1d: up_mat_flat(
                    &format!("{p}.linear_attn.conv1d.weight"),
                    conv_cols,
                    conv_kernel,
                )?
                .0,
                // Host mirror for gdn_ensure's one-time upload (see the field's note).
                conv1d_host: std::sync::Arc::new(
                    // Same (rows, cols) up_mat_flat uses, so the flat order matches the buffer
                    // the island scratch is compared against.
                    w.bf16_matrix(
                        &format!("{p}.linear_attn.conv1d.weight"),
                        conv_cols,
                        conv_kernel,
                    )?
                    .to_f32(),
                ),
                out_proj: up_w(&format!("{p}.linear_attn.out_proj.weight"), h, inner)?,
                a_log: gdn_a_pair.0.clone(),
                dt_bias: gdn_dt_pair.0.clone(),
                norm: gdn_norm_pair.0.clone(),
            }))
        } else {
            None
        };
        layers.push(GpuLayer {
            gdn,
            // q/o use the LAYER's q_dim (= num_heads · layer head_dim): the o_proj
            // maps the concatenated heads (q_dim) back to hidden, so its in-dim is
            // also per-layer.
            q_proj: up_w(&format!("{p}.self_attn.q_proj.weight"), l_q_dim, h)?,
            k_proj,
            v_proj,
            // On a GDN layer the output projection IS ssm_out (inner -> hidden), already
            // mapped to linear_attn.out_proj. Same slot, so every downstream size/residency
            // path stays identical.
            o_proj: if is_gdn {
                let inner = match &cfg.attn {
                    AttnKind::HybridSsmAttn { ssm_inner, .. } => *ssm_inner,
                    _ => l_q_dim,
                };
                up_w(&format!("{p}.linear_attn.out_proj.weight"), h, inner)?
            } else {
                // qwen35 attention layers: q packs 2x head_dim (12288 in) but o_proj takes the
                // NORMAL width (nh*head_dim = 6144) — measured, layers.3.o_proj is [5120,6144].
                // So o_proj's in-dim is not l_q_dim here.
                let o_in = match &cfg.attn {
                    AttnKind::HybridSsmAttn { .. } => cfg.num_attention_heads * l_head_dim,
                    _ => l_q_dim,
                };
                up_w(&format!("{p}.self_attn.o_proj.weight"), h, o_in)?
            },
            // Muse Glimmer's attention-output gate. Loaded ONLY when the arch declares it, so
            // every other model gets None and skips the dispatch — presence of the tensor is the
            // feature flag, not an arch string comparison in the hot path.
            attn_gate: if cfg.is_muse_glimmer() {
                Some(up_w(
                    &format!("{p}.self_attn.gate_proj.weight"),
                    l_q_dim,
                    h,
                )?)
            } else {
                None
            },
            mlp,
            input_norm: input_norm_p.0,
            post_norm: post_norm_p.0,
            pre_ffn_norm: pre_ffn_pair.map(|p| p.0),
            post_ffn_norm: post_ffn_pair.map(|p| p.0),
            q_norm: q_norm_p.map(|p| p.0),
            k_norm: k_norm_p.map(|p| p.0),
            #[cfg(target_os = "macos")]
            norm_mtls,
            window,
            head_dim: l_head_dim,
            kv_heads: l_kv_heads,
            rotary_dim: l_rotary_dim,
            // Gemma 4: the learned per-layer output scalar ([1]); absent on other
            // models (then None → no-op). Read as a plain CPU f32 (applied as a
            // uniform multiply at the end of the layer, not a GPU weight buffer).
            // `ARF_NO_LAYER_SCALAR=1` disables it (a debug discriminator).
            layer_scalar: if std::env::var("ARF_NO_LAYER_SCALAR").is_ok() {
                None
            } else {
                w.shape(&format!("{p}.layer_scalar"))
                    .and_then(|_| w.f32_vec(&format!("{p}.layer_scalar"), 1).ok())
                    .map(|v| v[0])
            },
        });
        // Commit this layer's weight buffers to residency before the NEXT layer's
        // allocations can push total resident memory over the ~14-16 GiB Metal
        // working-set threshold and reclaim/zero them (see the embed comment above).
        // Committing per layer keeps the untouched-allocation window to one layer
        // (~0.4 GB), far under the threshold.
        let mut layer_bufs: Vec<&wgpu::Buffer> = Vec::new();
        layers.last().unwrap().buffers(&mut layer_bufs);
        ctx.make_resident(&layer_bufs);
    }

    // L241 — MTP ("nextn") DRAFT HEAD. `blk.64` on qwen35: a full attention+FFN block whose
    // tensors are named exactly like a trunk block, plus four `nextn.*` tensors. It sits AFTER
    // the trunk and outside the forward pass — the model is correct without it — so it is loaded
    // into its own slot rather than appended to `layers`, which would corrupt both the 4-layer
    // attention schedule and the forward pass itself.
    //
    // Loaded ONLY when the arch declares it (`nextn_layers > 0`) AND the tensors are actually
    // present, so every other model pays nothing and a truncated GGUF degrades to "no head"
    // instead of failing the load.
    let mtp = if cfg.nextn_layers > 0 {
        let i = cfg.num_layers; // blk.64 for a 64-layer trunk
        let p = format!("model.layers.{i}");
        // Presence probe: a truncated GGUF degrades to "no head" rather than failing the load.
        if w.f32_vec(&format!("{p}.nextn.enorm.weight"), h).is_ok() {
            load_trace("mtp head (blk.64)");
            // Metal views of the head's norms, captured as each field below loads. Declared
            // without an initial value: the struct literal assigns every one before the
            // matching `*_mtl` field reads it, so a placeholder `None` would only be dead.
            let (en_m, hn_m, an_m, fn_m, qn_m, kn_m, shn_m): (
                Option<Option<crate::gpu::concurrent_metal::MtlBuf>>,
                Option<Option<crate::gpu::concurrent_metal::MtlBuf>>,
                Option<Option<crate::gpu::concurrent_metal::MtlBuf>>,
                Option<Option<crate::gpu::concurrent_metal::MtlBuf>>,
                Option<Option<crate::gpu::concurrent_metal::MtlBuf>>,
                Option<Option<crate::gpu::concurrent_metal::MtlBuf>>,
                Option<Option<crate::gpu::concurrent_metal::MtlBuf>>,
            );
            // blk.64's attention + FFN are named exactly like a trunk block, so they load
            // through the SAME `up_w` calls the trunk loop uses. Loading them here (rather than
            // reusing GpuLayer) keeps the head self-contained and out of every schedule that
            // iterates `layers`.
            let l_head_dim = cfg.head_dim;
            let l_q_dim = cfg.num_attention_heads * l_head_dim * 2; // qwen35 packs q 2x
            let o_in = cfg.num_attention_heads * l_head_dim;
            Some(std::sync::Arc::new(crate::gpu::GpuMtpHead {
                q_proj: up_w(&format!("{p}.self_attn.q_proj.weight"), l_q_dim, h)?,
                k_proj: up_w(
                    &format!("{p}.self_attn.k_proj.weight"),
                    cfg.num_kv_heads * l_head_dim,
                    h,
                )?,
                v_proj: up_w(
                    &format!("{p}.self_attn.v_proj.weight"),
                    cfg.num_kv_heads * l_head_dim,
                    h,
                )?,
                o_proj: up_w(&format!("{p}.self_attn.o_proj.weight"), h, o_in)?,
                // q/k norms are per-HEAD ([head_dim]=256), not per-hidden — same as the trunk.
                q_norm: {
                    let t = up_head_norm(&format!("{p}.self_attn.q_norm.weight"), l_head_dim)?;
                    qn_m = Some(t.1.clone());
                    t.0
                },
                k_norm: {
                    let t = up_head_norm(&format!("{p}.self_attn.k_norm.weight"), l_head_dim)?;
                    kn_m = Some(t.1.clone());
                    t.0
                },
                attn_norm: {
                    let t = up_vec(&format!("{p}.input_layernorm.weight"))?;
                    an_m = Some(t.1.clone());
                    t.0
                },
                ffn_norm: {
                    let t = up_vec(&format!("{p}.post_attention_layernorm.weight"))?;
                    fn_m = Some(t.1.clone());
                    t.0
                },
                attn_norm_mtl: an_m.flatten(),
                ffn_norm_mtl: fn_m.flatten(),
                q_norm_mtl: qn_m.flatten(),
                k_norm_mtl: kn_m.flatten(),
                gate_proj: up_w(
                    &format!("{p}.mlp.gate_proj.weight"),
                    cfg.intermediate_size,
                    h,
                )?,
                up_proj: up_w(&format!("{p}.mlp.up_proj.weight"), cfg.intermediate_size, h)?,
                down_proj: up_w(
                    &format!("{p}.mlp.down_proj.weight"),
                    h,
                    cfg.intermediate_size,
                )?,
                // eh_proj is [2*hidden -> hidden]: it consumes the CONCATENATION of the
                // normed token embedding and the normed trunk hidden.
                eh_proj: up_w(&format!("{p}.nextn.eh_proj.weight"), h, 2 * h)?,
                enorm: {
                    let t = up_vec(&format!("{p}.nextn.enorm.weight"))?;
                    en_m = Some(t.1.clone());
                    t.0
                },
                hnorm: {
                    let t = up_vec(&format!("{p}.nextn.hnorm.weight"))?;
                    hn_m = Some(t.1.clone());
                    t.0
                },
                enorm_mtl: en_m.flatten(),
                hnorm_mtl: hn_m.flatten(),
                shared_head_norm: {
                    let t = up_vec(&format!("{p}.nextn.shared_head_norm.weight"))?;
                    shn_m = Some(t.1.clone());
                    t.0
                },
                shared_head_norm_mtl: shn_m.flatten(),
            }))
        } else {
            None
        }
    } else {
        None
    };

    // Muse Glimmer's pre-trunk embedding norm (see `GpuModel::embed_norm`). It is WEIGHTLESS
    // in llama's graph — `rms_norm(embd)` with no `mul` — so we synthesize an all-ones
    // `[hidden]` weight and reuse the standard rmsnorm kernel. Going through `shared_norm`
    // (not a bare `storage_init`) is what gives the Metal island its dual view, exactly like
    // every real norm; the `fold_gemma_norm` +1 must NOT apply here, so it deliberately does
    // not go through `up_vec`.
    let embed_norm_pair = if cfg.is_muse_glimmer() {
        Some(shared_norm(ctx, "embed_norm.ones", &vec![1.0f32; h])?)
    } else {
        None
    };
    // The island runs its OWN embed, so it needs the Metal view of this weight too — without it
    // the megakernel feeds RAW embeddings to all 52 layers (the same root cause as the wgpu bug).
    #[cfg(target_os = "macos")]
    let embed_norm_mtl: Option<crate::gpu::concurrent_metal::MtlBuf> =
        embed_norm_pair.as_ref().and_then(|p| p.1.clone());
    let embed_norm = embed_norm_pair.map(|p| p.0);

    let final_norm_pair = up_vec("model.norm.weight")?;
    let final_norm = final_norm_pair.0;
    #[cfg(target_os = "macos")]
    let final_norm_mtl = final_norm_pair.1;
    // Tied lm_head shares the embedding weights. The `embed` buffer above is ALWAYS
    // uploaded bf16 (large vocab×hidden table; see its comment), and the tied lm_head
    // reads that SAME `embed_tokens.weight` tensor. Crucially this holds at EVERY
    // `quant`: even under `--quant q4`/Q4KS the embedding tensor is stored Q6_K/Q5_K in
    // the GGUF, so `up_w` finds no native 4-bit path and lands on the bf16 fallback —
    // i.e. it would rebuild a buffer byte-IDENTICAL to `embed` (same tensor, same
    // `[vocab, hidden]` layout, same `storage_init_bf16`; `add_matmul`'s Bf16 arm reads
    // it via `<< 16` exactly as the gather does). So ALIAS the resident buffer (a cheap
    // wgpu handle clone over the same allocation) instead of re-materializing it. This
    // skips a second ~11 GB f32→bf16→f32 load-time transient AND a second 2.8 GB
    // resident copy — the embed double-upload that was the load-time OOM driver on the
    // 36 GB Mac for gemma-4-12B (and it must stay aliased after the GGUF default is
    // promoted to Q4, which would otherwise route lm_head back through the rebuild).
    //
    // The ONLY case where the tied lm_head is NOT bf16-equal to `embed` is a GGUF whose
    // embedding tensor IS natively 4-bit (then `embed` would still be bf16 but a Q4
    // lm_head is smaller/faster). That doesn't occur in the models we load (Llama/Gemma/
    // Qwen all store the tied table in F16/Q6_K/Q8_0), so we alias unconditionally and
    // fall back to a rebuild only when the alias would change numerics — i.e. when a
    // native path exists for the embed tensor.
    let lm_head = if cfg.tie_word_embeddings {
        // True when `up_w("embed_tokens")` would take a NATIVE (non-bf16) path — only
        // then does aliasing the bf16 `embed` change the lm_head numerics.
        let embed_has_native_path = matches!(quant, Quant::Q4)
            && w.q4_matrix_from_gguf_q4_0("model.embed_tokens.weight", cfg.vocab_size, h)
                .is_some()
            || matches!(quant, Quant::Q4K | Quant::Q4KS)
                && w.q4ks_matrix_from_gguf_q4k("model.embed_tokens.weight", cfg.vocab_size, h)
                    .is_some();
        if embed_has_native_path {
            up_w("model.embed_tokens.weight", cfg.vocab_size, h)?
        } else {
            GpuMatWeight::Bf16(embed.clone())
        }
    } else {
        up_w("lm_head.weight", cfg.vocab_size, h)?
    };

    // MEGAKERNEL lm_head: the lm_head GEMV is the biggest single per-token cost (262144×3840),
    // so the island runs it in the same command buffer as the trunk rather than leaving it to
    // the bf16 wgpu tail.
    //
    // 🔴 PREFER THE FILE'S OWN Q4_K BLOCKS. `up_w` above may ALREADY have loaded this exact
    // matrix natively — `GpuMatWeight::Q4KS { mtl: Some(..) }` is a bit-exact repack of the GGUF
    // blocks, at 0.53 GB against Q8's 1.01 GB. When it exists, building the Q8 copy would burn a
    // gigabyte of a memory budget this box does not have, to store a STRICTLY WORSE
    // approximation of a tensor already resident in a better one: Q8 gets there via
    // `Q4_K → bf16 → requantize`, and no requantization recovers information the first
    // quantization removed.
    //
    // Whether that native path exists is a property of the FILE, not the architecture — it is
    // why `--quant q4ks` on a `*_K_S.gguf` is worth more than the ~10% the file-size delta
    // predicts. gemma-4-12b Q4_K_S stores `token_embd.weight` as Q4_K and takes this arm;
    // Q4_K_M stores it Q6_K and falls through to Q8. See `IslandLmHead` for the full derivation
    // and for why the historical "Q4KS flipped ~1 token/32" note does not apply here.
    #[cfg(target_os = "macos")]
    let lm_head_q8_mtl: Option<std::sync::Arc<crate::gpu::concurrent_metal::IslandLmHead>> =
        if std::env::var_os("ARF_MSL_GEMV").is_some() && h.is_multiple_of(4) {
            use crate::gpu::concurrent_metal::{IslandLmHead, Q8Mtl, SharedBuffer};
            // A/B escape: force the bf16→Q8 arm even where native blocks exist. Exists so the two
            // can be measured back-to-back on ONE machine state — comparing against a number
            // recorded on a differently-loaded box is how a 10% regression hides inside
            // run-to-run variance.
            let force_q8 = std::env::var_os("ARF_LM_HEAD_Q8").is_some();
            if let (false, GpuMatWeight::Q4KS { mtl: Some(q), .. }) = (force_q8, &lm_head) {
                // The tied/untied choice was already made when `lm_head` was built above, so
                // reusing it cannot repeat the "wrong matrix for an untied model" bug the Q8 arm
                // warns about.
                // 2026-09-27: said "NATIVE Q4_K" even when the file's lm_head is group-64 (Q4_1
                // pairs) and every lm_head consumer takes the group-64 matmul — the log now names
                // the route that is actually dispatched (rule 7).
                if q.g64.is_some() {
                    eprintln!(
                        "[msl] lm_head on the island as GROUP-64 ({}×{}, the file's own group-64 \
                         weights; every lm_head consumer takes the group-64 matmul)",
                        cfg.vocab_size, h
                    );
                } else {
                    eprintln!(
                        "[msl] lm_head on the island as NATIVE Q4_K ({}×{}, bit-exact to the GGUF, \
                         ~0.53 GB vs ~1.01 GB for the bf16→Q8 transcode)",
                        cfg.vocab_size, h
                    );
                }
                Some(std::sync::Arc::new(IslandLmHead::Q4ks(q.clone())))
            } else {
                // No native blocks for this tensor (Q6_K/Q5_K/F16 embedding table). Reconstruct f32
                // from the bf16 lm_head, then Q8 (per-row int8) — parity-safe at 8 bits, and here it
                // genuinely IS the best available: there is no 4-bit source to repack, so a 4-bit
                // island lm_head would mean requantizing, which is the thing that flipped tokens.
                // data packed 4 int8/u32.
                // CRITICAL: source the SAME matrix the wgpu oracle's `lm_head` uses — the DEDICATED
                // `lm_head.weight` (GGUF `output.weight`) for UNTIED models (Qwen3-30B), and the
                // tied `model.embed_tokens.weight` only when tie_word_embeddings. Using the embed
                // table for an untied model projects the final hidden through the WRONG matrix →
                // garbage logits (per ModelConfig: "NOT tied ... → garbage logits").
                let lm_src = if cfg.tie_word_embeddings {
                    "model.embed_tokens.weight"
                } else {
                    "lm_head.weight"
                };
                let m = w.bf16_matrix(lm_src, cfg.vocab_size, h).ok();
                m.and_then(|m| {
                    let q = arf_core::tensor::Q8Matrix::from_f32(&m.to_f32(), cfg.vocab_size, h);
                    let u = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
                    let packed: Vec<u32> = q
                        .data()
                        .chunks_exact(4)
                        .map(|c| {
                            (c[0] as u8 as u32)
                                | ((c[1] as u8 as u32) << 8)
                                | ((c[2] as u8 as u32) << 16)
                                | ((c[3] as u8 as u32) << 24)
                        })
                        .collect();
                    let (data, scale, dims) = (
                        SharedBuffer::from_pod(ctx, &packed, u, "lmhead.q8")?,
                        SharedBuffer::from_pod(ctx, q.scales(), u, "lmhead.q8scale")?,
                        SharedBuffer::from_pod(
                            ctx,
                            &[1u32, h as u32, cfg.vocab_size as u32, 0u32],
                            u,
                            "lmhead.q8dims",
                        )?,
                    );
                    eprintln!(
                        "[msl] lm_head transcoded bf16→Q8 for the island ({}×{}, parity-safe){}",
                        cfg.vocab_size,
                        h,
                        if force_q8 {
                            " — ARF_LM_HEAD_Q8 set, so the native Q4_K arm was skipped on purpose"
                        } else {
                            ""
                        }
                    );
                    Some(std::sync::Arc::new(IslandLmHead::Q8(Q8Mtl {
                        data: data.mtl,
                        scale: scale.mtl,
                        dims: dims.mtl,
                        n: cfg.vocab_size,
                    })))
                })
            }
        } else {
            None
        };

    // Global RoPE (uses cfg.rope_theta + any scaling). Gemma additionally builds a
    // local base for its sliding-window layers (rope_local_base_freq).
    // qwen35 (HybridSsmAttn) rotates only 64 of each 256-wide head (`layer_geometry`, L155), and
    // every kernel that reads this table indexes it `cos[pos * rotary_dim/2 + i]`. Built as
    // `Rope::new` it had rows of head_dim/2 = 128 at theta^(-2i/256): position p read row p/4,
    // column (p%4)*32 + i — the angle advanced once every FOUR tokens and hopped frequency band
    // within each group of four. Measured 2026-09-24: "Count from 1 to 10" greedy gave
    // "…6, 7, 7, 8…" with the previous number at logprob -3.1 where llama.cpp puts it at -12.5,
    // on the prefill path and the decode path alike. Built over the ROTARY dim, as HF does
    // (`inv_freq[i] = theta^(-2i/rotary_dim)`) and as Gemma 4's partial-rotary globals already are.
    // Teacher-forced, 10,421 tokens, window 128, interleaved: old table 7.0430 (top-1 55.25%),
    // this table 6.0639 (58.18%), on a cache MISS and a cache HIT alike. The fix looked INERT
    // at first — the weight cache was serving the old table back (see `weight_cache::region_for`).
    let rope = match &cfg.attn {
        AttnKind::HybridSsmAttn { .. } => {
            let (hd, _, rot) = cfg.layer_geometry(
                (0..cfg.num_layers)
                    .find(|&i| kv_layer_needed(&cfg.attn, i))
                    .unwrap_or(0),
            );
            Rope::with_theta_rotary(max_positions, cfg.rope_theta, hd, rot)
        }
        _ => Rope::new(cfg, max_positions),
    };
    eprintln!(
        "[rope] table: {} pairs a position over {max_positions} positions (theta {})",
        rope.cos_table().len() / max_positions.max(1),
        cfg.rope_theta
    );
    // Megakernel S1c: rope tables as dual-view SharedBuffers (macOS island reads them in
    // rope_qk). shared_rope returns (wgpu buffer, Option<MtlBuf>); _mtls stored on GpuModel.
    #[cfg(target_os = "macos")]
    let shared_rope = |name: &str,
                       data: &[f32]|
     -> (wgpu::Buffer, Option<crate::gpu::concurrent_metal::MtlBuf>) {
        if std::env::var_os("ARF_MSL_GEMV").is_some() {
            // UNCACHED: a RoPE table is this run's (context length, rotary geometry), not the
            // GGUF's. Through the weight cache every load served the table of the run that
            // built the sidecar (2026-09-24: perplexity 7.04 served vs 6.06 fresh).
            if let Some(sb) = crate::gpu::concurrent_metal::SharedBuffer::from_pod_uncached(
                ctx,
                data,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                name,
            ) {
                return (sb.wgpu, Some(crate::gpu::concurrent_metal::MtlBuf(sb.mtl)));
            }
        }
        (ctx.storage_init(name, data), None)
    };
    #[cfg(not(target_os = "macos"))]
    let shared_rope = |name: &str, data: &[f32]| -> wgpu::Buffer { ctx.storage_init(name, data) };

    #[cfg(target_os = "macos")]
    let (rope_cos, rope_cos_mtl) = shared_rope("rope_cos", rope.cos_table());
    #[cfg(target_os = "macos")]
    let (rope_sin, rope_sin_mtl) = shared_rope("rope_sin", rope.sin_table());
    #[cfg(not(target_os = "macos"))]
    let rope_cos = shared_rope("rope_cos", rope.cos_table());
    #[cfg(not(target_os = "macos"))]
    let rope_sin = shared_rope("rope_sin", rope.sin_table());
    // Gemma's sliding-window layers rotate with a separate (smaller) local base.
    #[cfg(target_os = "macos")]
    let (mut rope_cos_local_mtl, mut rope_sin_local_mtl) = (None, None);
    let (rope_cos_local, rope_sin_local) = match &cfg.attn {
        AttnKind::HybridLocalGlobal { local_theta, .. } => {
            let rl = Rope::with_theta(cfg, max_positions, *local_theta);
            #[cfg(target_os = "macos")]
            {
                let (cb, cm) = shared_rope("rope_cos_local", rl.cos_table());
                let (sb, sm) = shared_rope("rope_sin_local", rl.sin_table());
                rope_cos_local_mtl = cm;
                rope_sin_local_mtl = sm;
                (Some(cb), Some(sb))
            }
            #[cfg(not(target_os = "macos"))]
            {
                (
                    Some(shared_rope("rope_cos_local", rl.cos_table())),
                    Some(shared_rope("rope_sin_local", rl.sin_table())),
                )
            }
        }
        // qwen35: one rope base on the attention layers, none on GDN.
        AttnKind::HybridSsmAttn { .. } => (None, None),
        AttnKind::Causal => (None, None),
    };
    // Gemma 4's GLOBAL layers use a DIFFERENT head_dim (512 vs sliding 256) and
    // PARTIAL rotary (128 of 512 dims), so their RoPE table is built over the global
    // head_dim with rotary_dim frequencies (row stride rotary_dim/2 = 64) at the
    // global θ. Built only when the global geometry actually differs from the base
    // (partial rotary OR a larger head_dim); Gemma 3's global layers (head_dim ==
    // sliding, full rotary) reuse `rope_cos`/`sin` and this stays `None`.
    #[cfg(target_os = "macos")]
    let (mut rope_cos_global_mtl, mut rope_sin_global_mtl) = (None, None);
    let (rope_cos_global, rope_sin_global) = match &cfg.attn {
        AttnKind::HybridLocalGlobal {
            global_theta,
            global_head_dim,
            partial_rotary_factor,
            ..
        } if *global_head_dim != cfg.head_dim || *partial_rotary_factor != 1.0 => {
            let rot = (*global_head_dim as f32 * partial_rotary_factor).round() as usize;
            let rot = rot - (rot % 2); // RoPE rotates pairs → even rotary dim
            let rg = Rope::with_theta_rotary(max_positions, *global_theta, *global_head_dim, rot);
            #[cfg(target_os = "macos")]
            {
                let (cb, cm) = shared_rope("rope_cos_global", rg.cos_table());
                let (sb, sm) = shared_rope("rope_sin_global", rg.sin_table());
                rope_cos_global_mtl = cm;
                rope_sin_global_mtl = sm;
                (Some(cb), Some(sb))
            }
            #[cfg(not(target_os = "macos"))]
            {
                (
                    Some(shared_rope("rope_cos_global", rg.cos_table())),
                    Some(shared_rope("rope_sin_global", rg.sin_table())),
                )
            }
        }
        _ => (None, None),
    };

    // MoE sizing (None for dense): the gate/up scratch must hold the wider of the
    // dense intermediate and an expert's moe_intermediate.
    let moe_dims = match &cfg.mlp {
        MlpKind::Moe {
            num_experts,
            top_k,
            moe_intermediate,
            ..
        } => Some((*num_experts, *top_k, *moe_intermediate)),
        MlpKind::Dense => None,
    };
    let inter_scratch = moe_dims
        .map(|(_, _, mi)| mi.max(cfg.intermediate_size))
        .unwrap_or(cfg.intermediate_size);
    // Per-layer geometry varies (Gemma 4 global layers): the shared q/k/v decode
    // scratch must hold the LARGEST layer's q_dim/kv_dim. `max_attn_dims` gives the
    // max head_dim and max kv_dim over all layers; q_dim scales with head_dim.
    let (max_head_dim, max_kv_dim) = cfg.max_attn_dims();
    let _ = max_head_dim;
    // L154 — `max_q_dim()` also accounts for the hybrid SSM family's 2×-packed attention q
    // (Qwen3.8: 12288 vs the 6144 `num_attention_heads * max_head_dim` gives). Sizing this
    // scratch the old way overflowed it by 2× on every full-attention layer.
    let max_q_dim = cfg.max_q_dim();
    // FFN-block scratch as dual-view SharedBuffers when the MSL fast path is on (macOS +
    // ARF_MSL_GEMV): one allocation, a wgpu view (every existing binding) + a raw MTL
    // view (the island's ffn_dispatch). Else plain storage_zeros + no MTL view.
    #[cfg(target_os = "macos")]
    let msl_on = std::env::var_os("ARF_MSL_GEMV").is_some();
    #[cfg(target_os = "macos")]
    let shared_or_zeros =
        |label: &str, n: usize| -> (wgpu::Buffer, Option<crate::gpu::concurrent_metal::MtlBuf>) {
            use wgpu::BufferUsages as U;
            if msl_on {
                if let Some(sb) = crate::gpu::concurrent_metal::SharedBuffer::alloc(
                    ctx,
                    (n * 4) as u64,
                    U::STORAGE | U::COPY_SRC | U::COPY_DST,
                    label,
                ) {
                    return (sb.wgpu, Some(crate::gpu::concurrent_metal::MtlBuf(sb.mtl)));
                }
            }
            (ctx.storage_zeros(label, n), None)
        };
    #[cfg(target_os = "macos")]
    let (hidden_b, hidden_mtl) = shared_or_zeros("hidden", h);
    #[cfg(target_os = "macos")]
    let (normed_b, normed_mtl) = shared_or_zeros("normed", h);
    #[cfg(target_os = "macos")]
    let (gate_b, gate_mtl) = shared_or_zeros("gate", inter_scratch);
    #[cfg(target_os = "macos")]
    let (up_b, up_mtl) = shared_or_zeros("up", inter_scratch);
    #[cfg(target_os = "macos")]
    let (mlp_down_b, mlp_down_mtl) = shared_or_zeros("mlp_down", h);
    // Attention scratch dual-views (megakernel S1): q sized max_q_dim (8192 on 12B global),
    // k/v max_kv_dim (2048), attn_out h (o_proj maps q_dim→h).
    #[cfg(target_os = "macos")]
    let (q_b, q_mtl) = shared_or_zeros("q", max_q_dim);
    #[cfg(target_os = "macos")]
    let (k_b, k_mtl) = shared_or_zeros("k", max_kv_dim);
    #[cfg(target_os = "macos")]
    let (v_b, v_mtl) = shared_or_zeros("v", max_kv_dim);
    #[cfg(target_os = "macos")]
    let (attn_out_b, attn_out_mtl) = shared_or_zeros("attn_out", h);
    #[cfg(target_os = "macos")]
    let (logits_b, logits_mtl) = shared_or_zeros("logits", cfg.vocab_size);
    // next_token (1 u32) shared so the island argmax writes it + the island embed reads it
    // (single-queue). COPY_DST so the prefill seed + readback still work.
    #[cfg(target_os = "macos")]
    let (next_token_b, next_token_mtl): (
        wgpu::Buffer,
        Option<crate::gpu::concurrent_metal::MtlBuf>,
    ) = {
        use wgpu::BufferUsages as U;
        if msl_on {
            if let Some(sb) = crate::gpu::concurrent_metal::SharedBuffer::alloc(
                ctx,
                4,
                U::STORAGE | U::COPY_SRC | U::COPY_DST,
                "next_token",
            ) {
                (sb.wgpu, Some(crate::gpu::concurrent_metal::MtlBuf(sb.mtl)))
            } else {
                (ctx.storage_zeros_u32("next_token", 1), None)
            }
        } else {
            (ctx.storage_zeros_u32("next_token", 1), None)
        }
    };
    // out_tokens shared so the island store_word writes the emitted id directly (kills the
    // per-token wgpu collect_slot copy → fully single-queue). COPY_SRC for the final drain,
    // COPY_DST for the spec-decode collect_after path.
    #[cfg(target_os = "macos")]
    let (out_tokens_b, out_tokens_mtl): (
        wgpu::Buffer,
        Option<crate::gpu::concurrent_metal::MtlBuf>,
    ) = {
        use wgpu::BufferUsages as U;
        if msl_on {
            if let Some(sb) = crate::gpu::concurrent_metal::SharedBuffer::alloc(
                ctx,
                (max_positions * 4) as u64,
                U::STORAGE | U::COPY_SRC | U::COPY_DST,
                "out_tokens",
            ) {
                (sb.wgpu, Some(crate::gpu::concurrent_metal::MtlBuf(sb.mtl)))
            } else {
                (ctx.storage_io_u32("out_tokens", max_positions), None)
            }
        } else {
            (ctx.storage_io_u32("out_tokens", max_positions), None)
        }
    };
    // SECOND out_tokens bank for the depth-2 ping-pong. Allocated identically to the
    // first (same size + usages) so the two banks are drop-in interchangeable as the batched
    // step's out bank. Only the MTL view is retained (the wgpu buffer is unused — the island
    // reads/writes it directly via Metal); dropping the wgpu half here is fine because the MtlBuf
    // holds the underlying MTLBuffer alive. `None` (alloc failed or MSL off) → depth-2 disabled.
    #[cfg(target_os = "macos")]
    let out_tokens_mtl2: Option<crate::gpu::concurrent_metal::MtlBuf> = {
        use wgpu::BufferUsages as U;
        if msl_on {
            crate::gpu::concurrent_metal::SharedBuffer::alloc(
                ctx,
                (max_positions * 4) as u64,
                U::STORAGE | U::COPY_SRC | U::COPY_DST,
                "out_tokens2",
            )
            .map(|sb| crate::gpu::concurrent_metal::MtlBuf(sb.mtl))
        } else {
            None
        }
    };

    let scratch = GpuScratch {
        #[cfg(target_os = "macos")]
        hidden: hidden_b,
        #[cfg(not(target_os = "macos"))]
        hidden: ctx.storage_zeros("hidden", h),
        #[cfg(target_os = "macos")]
        normed: normed_b,
        #[cfg(not(target_os = "macos"))]
        normed: ctx.storage_zeros("normed", h),
        #[cfg(target_os = "macos")]
        q: q_b,
        #[cfg(not(target_os = "macos"))]
        q: ctx.storage_zeros("q", max_q_dim),
        #[cfg(target_os = "macos")]
        k: k_b,
        #[cfg(not(target_os = "macos"))]
        k: ctx.storage_zeros("k", max_kv_dim),
        #[cfg(target_os = "macos")]
        v: v_b,
        #[cfg(not(target_os = "macos"))]
        v: ctx.storage_zeros("v", max_kv_dim),
        // o_proj maps the concatenated head outputs (q_dim) BACK to the residual
        // stream (hidden_size), so this holds `h` floats — NOT q_dim. They coincide
        // when num_heads*head_dim == hidden (Llama/Qwen) but differ for Gemma-3
        // (q_dim 2048 vs hidden 2560); sizing it q_dim left the top h-q_dim elements
        // unwritten, so post_attn_norm read stale memory.
        #[cfg(target_os = "macos")]
        attn_out: attn_out_b,
        #[cfg(not(target_os = "macos"))]
        attn_out: ctx.storage_zeros("attn_out", h),
        #[cfg(target_os = "macos")]
        gate: gate_b,
        #[cfg(not(target_os = "macos"))]
        gate: ctx.storage_zeros("gate", inter_scratch),
        #[cfg(target_os = "macos")]
        up: up_b,
        #[cfg(not(target_os = "macos"))]
        up: ctx.storage_zeros("up", inter_scratch),
        #[cfg(target_os = "macos")]
        mlp_down: mlp_down_b,
        #[cfg(not(target_os = "macos"))]
        mlp_down: ctx.storage_zeros("mlp_down", h),
        #[cfg(target_os = "macos")]
        logits: logits_b,
        #[cfg(not(target_os = "macos"))]
        logits: ctx.storage_zeros("logits", cfg.vocab_size),
        #[cfg(target_os = "macos")]
        next_token: next_token_b,
        #[cfg(not(target_os = "macos"))]
        next_token: ctx.storage_zeros_u32("next_token", 1),
        #[cfg(target_os = "macos")]
        out_tokens: out_tokens_b,
        #[cfg(not(target_os = "macos"))]
        out_tokens: ctx.storage_io_u32("out_tokens", max_positions),
        router_logits: moe_dims.map(|(ne, _, _)| ctx.storage_zeros("router_logits", ne)),
        moe_ids: moe_dims.map(|(_, k, _)| ctx.storage_zeros_u32("moe_ids", k)),
        moe_wts: moe_dims.map(|(_, k, _)| ctx.storage_zeros("moe_wts", k)),
        // Fused-MoE per-slot gate/up scratch: top_k * moe_inter (slot-major).
        moe_gate_all: moe_dims.map(|(_, k, mi)| ctx.storage_zeros("moe_gate_all", k * mi)),
        moe_up_all: moe_dims.map(|(_, k, mi)| ctx.storage_zeros("moe_up_all", k * mi)),
        shared_ids: match &cfg.mlp {
            MlpKind::Moe { shared_experts, .. } if *shared_experts > 0 => Some(ctx.storage_init(
                "shared_ids",
                &(0..*shared_experts as u32).collect::<Vec<u32>>(),
            )),
            _ => None,
        },
        shared_wts: match &cfg.mlp {
            MlpKind::Moe { shared_experts, .. } if *shared_experts > 0 => {
                Some(ctx.storage_init("shared_wts", &vec![1.0f32; *shared_experts]))
            }
            _ => None,
        },
        #[cfg(target_os = "macos")]
        normed_mtl,
        #[cfg(target_os = "macos")]
        gate_mtl,
        #[cfg(target_os = "macos")]
        up_mtl,
        #[cfg(target_os = "macos")]
        mlp_down_mtl,
        #[cfg(target_os = "macos")]
        hidden_mtl,
        #[cfg(target_os = "macos")]
        q_mtl,
        #[cfg(target_os = "macos")]
        k_mtl,
        #[cfg(target_os = "macos")]
        v_mtl,
        #[cfg(target_os = "macos")]
        attn_out_mtl,
        #[cfg(target_os = "macos")]
        logits_mtl,
        #[cfg(target_os = "macos")]
        next_token_mtl,
        #[cfg(target_os = "macos")]
        out_tokens_mtl,
        #[cfg(target_os = "macos")]
        out_tokens_mtl2,
    };

    // Compile the decode-pipeline kernels once (shared with the qwen35 island generation
    // path, which needs the same matmul_vec_q4ks / matmul GEMV kernels — see build_kernels).
    let kernels = build_kernels(ctx, is_gemma);

    // Per-layer KV pools. Default (None): sized for ONE sequence of
    // `max_positions` (the CLI single-stream path, whose identity slot mapping
    // pos -> block pos/16 requires exactly this coverage). Some(n): the serving
    // path — the scheduler's BlockManager hands out block ids in 0..n and the GPU
    // pool MUST match it exactly (asserted at actor spawn, spec §3 memory safety).
    let block_size = 16;
    let num_blocks = kv_blocks.unwrap_or(max_positions.div_ceil(block_size));
    let total_slots = num_blocks * block_size;
    let (kv, kv_levels) = match kv_quant {
        KvQuant::None => {
            // Each layer's pool is sized to ITS OWN row stride (kv_heads · head_dim
            // floats per cached token). For Gemma 4 the sliding layers (16·256 = 4096)
            // and global layers (4·512 = 2048) differ, so a uniform `cfg`-derived
            // stride would over-/under-size them; the decode/batch dispatch reads the
            // matching stride from `GpuLayer.kv_dim()` (= layer.kv_heads·head_dim).
            // Megakernel S1c: dual-view SharedBuffers for the KV pool (macOS + ARF_MSL_GEMV)
            // so the island's kv_scatter writes + attention reads hit the same memory the wgpu
            // path uses. Else plain storage_zeros + empty _mtl vecs.
            #[cfg(target_os = "macos")]
            let kv_msl = std::env::var_os("ARF_MSL_GEMV").is_some();
            #[cfg(target_os = "macos")]
            let kv_shared =
                |label: &str,
                 n: usize|
                 -> (wgpu::Buffer, Option<crate::gpu::concurrent_metal::MtlBuf>) {
                    use wgpu::BufferUsages as U;
                    if kv_msl {
                        if let Some(sb) = crate::gpu::concurrent_metal::SharedBuffer::alloc(
                            ctx,
                            (n * 4) as u64,
                            U::STORAGE | U::COPY_SRC | U::COPY_DST,
                            label,
                        ) {
                            return (sb.wgpu, Some(crate::gpu::concurrent_metal::MtlBuf(sb.mtl)));
                        }
                    }
                    (ctx.storage_zeros(label, n), None)
                };
            #[cfg(target_os = "macos")]
            let (mut keys_v, mut values_v, mut keys_mtl, mut values_mtl) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            // PARALLEL f16 KV pool (ARF_KV_F16, the 3rd leg: bandwidth). Allocated
            // alongside the f32 pool ONLY when the flag is set (+ the island MSL path is live), sized
            // n*2 bytes/layer = HALF the f32 pool. Metal-private (no wgpu view): the island's f16
            // scatter/attention are the only touchers. kv_quant stays None — f16 is a pool FORMAT,
            // not a KvQuant (orthogonal → none of the kv_quant.is_quantized() bails are tripped).
            #[cfg(target_os = "macos")]
            let kv_q8 = kv_msl && crate::kv_q8_for(cfg);
            // The f16 pool is a SECOND pool beside the f32 one; the 8-bit cache makes it dead
            // weight (its attention is never selected) — half the f32 size, 8.6 GB at 131K.
            #[cfg(target_os = "macos")]
            let kv_f16 = kv_msl && !kv_q8 && std::env::var_os("ARF_KV_F16").is_some();
            // 8-BIT KV (default on the hybrid arch, `kv_q8_for`): the int8 pool REPLACES the f32
            // one (which becomes a placeholder).
            #[cfg(target_os = "macos")]
            let mut q8_pools: Vec<Option<[crate::gpu::concurrent_metal::MtlBuf; 4]>> = Vec::new();
            // Grow-on-use: the 8-bit buffers as placement-sparse allocations (no memory until a
            // slot is written; `sparse_kv.rs`). `ARF_NO_SPARSE_KV=1` = ordinary committed buffers.
            #[cfg(target_os = "macos")]
            let mut q8_sparse = if kv_q8 {
                unsafe { ctx.device.as_hal::<wgpu::hal::api::Metal>() }.and_then(|g| {
                    crate::gpu::metal::sparse_kv::SparseKv::new(g.raw_device(), total_slots)
                })
            } else {
                None
            };
            #[cfg(target_os = "macos")]
            let (mut keys_mtl_f16, mut values_mtl_f16): (
                Vec<Option<crate::gpu::concurrent_metal::MtlBuf>>,
                Vec<Option<crate::gpu::concurrent_metal::MtlBuf>>,
            ) = (Vec::new(), Vec::new());
            // L139 — LAYER-FILTERED KV ALLOCATION (the llama.cpp `layer_filter_cb` lesson,
            // llama-model.cpp:2284-2301). A hybrid arch's RECURRENT layers have no attention and
            // therefore no K/V to cache: their state is the resident conv/ssm pair on the island,
            // which is O(1) per token and allocated elsewhere. Allocating a full KV pool for them
            // wasted 48 of 64 layers = 75% of the pool (1.5 GB at --num-blocks 256, 3.0 GB at 512)
            // on a 36 GB box where OOM has already cost this project three separate tasks.
            //
            // Indices stay LAYER-ALIGNED — every consumer does `kv.keys[li]` — so a filtered-out
            // layer gets a 16-byte PLACEHOLDER rather than being skipped. Binding one would be a
            // bug, but it is a bug that faults loudly on a tiny buffer instead of silently reading
            // a multi-hundred-MB pool of zeros.
            #[cfg(target_os = "macos")]
            let needs_kv = |i: usize| -> bool { kv_layer_needed(&cfg.attn, i) };
            #[cfg(target_os = "macos")]
            let mut kv_skipped = 0usize;
            #[cfg(target_os = "macos")]
            for i in 0..cfg.num_layers {
                let n = if needs_kv(i) {
                    total_slots * layers[i].kv_dim()
                } else {
                    kv_skipped += 1;
                    4 // placeholder: 4 f32 = 16 bytes, the minimum Metal will hand back
                };
                if kv_q8 && needs_kv(i) {
                    let kvh = cfg.num_kv_heads.max(1);
                    let (kd, sd) = (layers[i].kv_dim(), kvh * 4); // bytes per slot: int8 row, scales
                    let mut mk = |per_slot: usize| match q8_sparse.as_mut() {
                        Some(sp) => sp.alloc(per_slot),
                        None => crate::gpu::concurrent_metal::SharedBuffer::alloc_mtl_only(
                            ctx,
                            (total_slots * per_slot) as u64,
                        ),
                    };
                    match (mk(kd), mk(sd), mk(kd), mk(sd)) {
                        (Some(a), Some(b), Some(c), Some(d)) => q8_pools.push(Some([a, b, c, d])),
                        _ => q8_pools.push(None),
                    }
                } else {
                    q8_pools.push(None);
                }
                // the f32 pool is only a placeholder when this layer is served from the int8 one
                let n = if kv_q8 && needs_kv(i) && q8_pools.last().is_some_and(|o| o.is_some()) {
                    4
                } else {
                    n
                };
                let (kb, km) = kv_shared(&format!("kpool{i}"), n);
                let (vb, vm) = kv_shared(&format!("vpool{i}"), n);
                keys_v.push(kb);
                values_v.push(vb);
                keys_mtl.push(km);
                values_mtl.push(vm);
                if kv_f16 {
                    keys_mtl_f16.push(crate::gpu::concurrent_metal::SharedBuffer::alloc_mtl_only(
                        ctx,
                        (n * 2) as u64,
                    ));
                    values_mtl_f16.push(
                        crate::gpu::concurrent_metal::SharedBuffer::alloc_mtl_only(
                            ctx,
                            (n * 2) as u64,
                        ),
                    );
                }
            }
            #[cfg(target_os = "macos")]
            if kv_q8 {
                let n_q8 = q8_pools.iter().filter(|o| o.is_some()).count();
                let kvh = cfg.num_kv_heads.max(1);
                let per_layer = 2 * total_slots * (kvh * cfg.head_dim + kvh * 4);
                eprintln!(
                    "[kv] 8-BIT KV CACHE: {n_q8} attention layers x {total_slots} slots = {:.2} GB (f32 would be {:.2} GB){} — ARF_NO_KV_Q8=1 for f32",
                    (n_q8 * per_layer) as f64 / 1e9,
                    (n_q8 * 2 * total_slots * kvh * cfg.head_dim * 4) as f64 / 1e9,
                    if q8_sparse.is_some() {
                        ", SPARSE: memory mapped as slots are written (ARF_NO_SPARSE_KV=1 commits it all)"
                    } else {
                        ""
                    }
                );
            }
            // The MTP head's pool: one attention layer's geometry, allocated only with a head.
            #[cfg(target_os = "macos")]
            let (mtp_keys_mtl, mtp_values_mtl) =
                match (&mtp, (0..cfg.num_layers).find(|&i| needs_kv(i))) {
                    (Some(_), Some(a)) => {
                        let n = total_slots * layers[a].kv_dim();
                        (kv_shared("kpool_mtp", n).1, kv_shared("vpool_mtp", n).1)
                    }
                    _ => (None, None),
                };
            #[cfg(target_os = "macos")]
            if kv_skipped > 0 {
                let saved = (cfg.num_layers - kv_skipped).max(1);
                eprintln!(
                    "[kv] layer-filtered: {kv_skipped}/{} layers are recurrent (no KV pool); \
                     allocated {saved} attention pools",
                    cfg.num_layers
                );
            } else {
                // L151 — say so POSITIVELY when the filter is off. The old code printed only when
                // it was ON, so `ARF_NO_KV_LAYER_FILTER=1` produced SILENCE — and an absent log
                // line is indistinguishable from "the flag never took effect". That ambiguity is
                // what made the first attempt at this experiment unreadable.
                eprintln!(
                    "[kv] NO layer filter: allocated a full pool for all {} layers{}",
                    cfg.num_layers,
                    if std::env::var_os("ARF_NO_KV_LAYER_FILTER").is_some() {
                        " (ARF_NO_KV_LAYER_FILTER=1)"
                    } else {
                        ""
                    }
                );
            }
            let kv = GpuKvPool {
                #[cfg(target_os = "macos")]
                keys: keys_v,
                // L139 — same layer filter as the macOS arm above: recurrent layers get a
                // 16-byte placeholder, not a full pool. Kept in sync deliberately; a hybrid model
                // must not pay 75% dead KV on any platform.
                #[cfg(not(target_os = "macos"))]
                keys: (0..cfg.num_layers)
                    .map(|i| {
                        let n = if kv_layer_needed(&cfg.attn, i) {
                            total_slots * layers[i].kv_dim()
                        } else {
                            4
                        };
                        ctx.storage_zeros(&format!("kpool{i}"), n)
                    })
                    .collect(),
                #[cfg(target_os = "macos")]
                values: values_v,
                #[cfg(not(target_os = "macos"))]
                values: (0..cfg.num_layers)
                    .map(|i| {
                        let n = if kv_layer_needed(&cfg.attn, i) {
                            total_slots * layers[i].kv_dim()
                        } else {
                            4
                        };
                        ctx.storage_zeros(&format!("vpool{i}"), n)
                    })
                    .collect(),
                #[cfg(target_os = "macos")]
                keys_mtl,
                #[cfg(target_os = "macos")]
                values_mtl,
                #[cfg(target_os = "macos")]
                keys_mtl_f16,
                #[cfg(target_os = "macos")]
                values_mtl_f16,
                #[cfg(target_os = "macos")]
                q8: q8_pools,
                #[cfg(target_os = "macos")]
                q8_sparse: q8_sparse.map(std::sync::Mutex::new),
                #[cfg(target_os = "macos")]
                mtp_keys_mtl,
                #[cfg(target_os = "macos")]
                mtp_values_mtl,
                key_norms: Vec::new(),
                value_norms: Vec::new(),
                block_size,
                num_blocks,
                kv_quant,
            };
            (kv, None)
        }
        KvQuant::Tq { bits, .. } => {
            // The TurboQuant pool is sized with a UNIFORM per-layer geometry
            // (cfg.num_kv_heads / cfg.head_dim). Models with per-layer-varying KV
            // geometry (Gemma 4: global layers are 4·512 vs sliding 16·256) would
            // need a per-layer code-word count + a per-layer Codebook; that is not
            // wired. Reject rather than silently mis-size the packed bitstream.
            let (gm_hd, gm_kv) = cfg.max_attn_dims();
            if gm_hd != cfg.head_dim || gm_kv != cfg.num_kv_heads * cfg.head_dim {
                return Err(ArfError::model_load(
                    "KvQuant::Tq is not supported for models with per-layer-varying KV \
                     geometry (e.g. Gemma 4 global vs sliding layers); use KvQuant::None",
                ));
            }
            // Packed pool: codes are a bits-wide bitstream over all head-vectors;
            // norms are one f32 per head-vector. (head_dim·bits is a multiple of
            // 32 for every supported model, so vectors are u32-word-aligned.)
            let n_vectors = total_slots * cfg.num_kv_heads;
            let code_words = (n_vectors * cfg.head_dim * bits as usize).div_ceil(32);
            let cb = arf_core::tensor::Codebook::new(bits, cfg.head_dim);
            let levels = ctx.storage_init("kv_levels", cb.levels());
            let kv = GpuKvPool {
                keys: (0..cfg.num_layers)
                    .map(|i| ctx.storage_zeros(&format!("kcodes{i}"), code_words))
                    .collect(),
                values: (0..cfg.num_layers)
                    .map(|i| ctx.storage_zeros(&format!("vcodes{i}"), code_words))
                    .collect(),
                key_norms: (0..cfg.num_layers)
                    .map(|i| ctx.storage_zeros(&format!("knorm{i}"), n_vectors))
                    .collect(),
                value_norms: (0..cfg.num_layers)
                    .map(|i| ctx.storage_zeros(&format!("vnorm{i}"), n_vectors))
                    .collect(),
                // Quantized KV pool — the island megakernel only supports KvQuant::None, so
                // no MTL views here (the wgpu attention_tq path serves this).
                #[cfg(target_os = "macos")]
                keys_mtl: Vec::new(),
                #[cfg(target_os = "macos")]
                values_mtl: Vec::new(),
                // no f16 pool for the Tq path (f16 KV is orthogonal to kv_quant; it only
                // applies to the KvQuant::None island coalesced path).
                #[cfg(target_os = "macos")]
                keys_mtl_f16: Vec::new(),
                #[cfg(target_os = "macos")]
                values_mtl_f16: Vec::new(),
                // The island megakernel — and with it the MTP head — does not run on a Tq pool.
                #[cfg(target_os = "macos")]
                q8: Vec::new(),
                #[cfg(target_os = "macos")]
                q8_sparse: None,
                #[cfg(target_os = "macos")]
                mtp_keys_mtl: None,
                #[cfg(target_os = "macos")]
                mtp_values_mtl: None,
                block_size,
                num_blocks,
                kv_quant,
            };
            (kv, Some(levels))
        }
    };

    // 🔴 L147 — COMMIT THE KV POOL TO RESIDENCY. This was MISSING, and it is the root cause of
    // the `--num-blocks 4096` corruption (L146): a daemon at 4096 emitted pure `!!!!` on a single
    // conc1 request while 2048 was clean on the same binary.
    //
    // The mechanism is the one already documented on `make_resident` and on the embed commit
    // above: on the wgpu Metal backend a large buffer that is never referenced in a SUBMITTED
    // command before total resident allocation crosses ~14-16 GiB has its pages reclaimed and
    // reads back ZEROED — and the loss is NOT recoverable by a later touch. Zeroed KV feeds zeroed
    // logits, argmax then picks token 0 every step, and token 0 renders as `!`.
    //
    // The pool is allocated AFTER all ~18 GB of weights and was the only major allocation never
    // passed to make_resident (the calls covered embed, each layer, and lm_head/final_norm). At
    // --num-blocks 2048 the pool is ~6.4 GB and happened to survive; at 4096 it is ~12.9 GB on top
    // of the weights, pushing well past the threshold. That is exactly the observed 2048-clean /
    // 4096-garbage split, and it explains why every parity test passed: they use small geometries
    // that never approach the threshold, and they never vary POOL size.
    {
        let mut kv_bufs: Vec<&wgpu::Buffer> = Vec::new();
        kv_bufs.extend(kv.keys.iter());
        kv_bufs.extend(kv.values.iter());
        kv_bufs.extend(kv.key_norms.iter());
        kv_bufs.extend(kv.value_norms.iter());
        if !kv_bufs.is_empty() {
            ctx.make_resident(&kv_bufs);
        }
    }

    // The four per-token decode arenas, allocated once for the worst-case
    // context and reset each step (instead of recreated — that was 4 device
    // allocations per token on the CPU critical path).
    let decode_arenas =
        std::cell::RefCell::new(crate::gpu::DecodeArenas::new(ctx, cfg, max_positions));

    // Commit the post-loop buffers (lm_head, final_norm) to residency. The embed and
    // every layer's buffers were already committed incrementally as they were created
    // (see GpuContext::make_resident); these are the last allocations, so a final
    // touch ensures the whole resident set is committed before the first decode.
    {
        let mut bufs: Vec<&wgpu::Buffer> = vec![&final_norm];
        lm_head.buffers(&mut bufs);
        ctx.make_resident(&bufs);
    }

    // ── FAST-PATH LEVER DEFAULTS (single source of truth) ───────────────────────────────────────
    // Each verified-bit-exact lever is enabled per this table. `default_on` flips a lever to ON for
    // EVERY fast-path model with ZERO env vars — the goal is that a fast-path model is fast by
    // default, no flag-drift. A lever's `default_on` should be set true ONLY AFTER an A/B script
    // (or the model's bench) measures it a real win on that model class — until then it stays
    // opt-IN (default_on:false) even though it's parity-safe (bit-exact ≠ proven-faster; an
    // unmeasured default could wash/regress). Opt out of any default with ARF_NO_<FLAG>.
    // Current status: all bit-exact + parity-green; speedup pending the supervised A/B (see
    // measured). When the A/B lands, flip default_on and delete this note.
    // Resolve a lever ONCE and PUBLISH it to the env, so the compile-side (here) AND the record-side
    // (concurrent_metal.rs, which reads the flag at dispatch) see ONE consistent decision — no
    // split-brain where the kernel is compiled with a lever but the dispatch doesn't match it.
    #[cfg(target_os = "macos")]
    let lever_on = |flag: &str, default_on: bool| -> bool {
        let no = format!("ARF_NO_{}", flag.trim_start_matches("ARF_"));
        if std::env::var_os(&no).is_some() {
            std::env::remove_var(flag);
            return false;
        }
        let on = std::env::var_os(flag).is_some() || default_on;
        if on {
            std::env::set_var(flag, "1");
        } // publish so the record-side agrees
        on
    };
    // DEFAULT-ON since the 2026-07-06 cool-machine A/B (coder-30B B=1, interleaved ×3):
    // the doty+nsg2+vec4 combo beat base in every pair (~2% — levers overlap on the same
    // registers/memory pipes, so they don't sum), all runs reference-checksum 0xe498… with
    // `[msl] island up` verified per arm (the mixed-FC fix made the combo compilable at all).
    // ARF_NO_* opts each lever out for A/B.
    #[cfg(target_os = "macos")]
    let doty_on = lever_on("ARF_Q4_DOTY", true);
    #[cfg(target_os = "macos")]
    let nsg2_on = lever_on("ARF_GEMV_NSG2", true);
    #[cfg(target_os = "macos")]
    let vec4_on = lever_on("ARF_ATTN_VEC4", true);
    // normpair is read on the RECORD side (concurrent_metal.rs); we resolve it here only for its
    // env-publish side-effect (so the record-side sees the same default/opt-out decision).
    #[cfg(target_os = "macos")]
    let _normpair_on = lever_on("ARF_GEMMA_FUSE_ADDNORM2", false); // ↩ gemma-dense only, +2.1%

    // Native-Metal island: built when the MSL fast path is on (the weights already carry
    // Q4ksMtl views) — compiles the hand-MSL gemv_q4ks kernel once, ready for decode.
    #[cfg(target_os = "macos")]
    let island = if std::env::var_os("ARF_MSL_GEMV").is_some() {
        crate::gpu::concurrent_metal::MetalIsland::new(ctx).and_then(|mut isl| {
            let gemv = include_str!("shaders/metal/matmul_vec_q4ks_msl.metal");
            let ops = include_str!("shaders/metal/decode_ops_msl.metal");
            let rms = include_str!("shaders/metal/rmsnorm_msl.metal");
            let attn = include_str!("shaders/metal/attention_msl.metal");
            // Compile the FULL whole-token megakernel kernel set (FFN path uses the subset).
            // ARF_Q4_DOTY: compile the dense gemv_q4ks with the dot-y nibble fold (fc2=true).
            // Default (unset) → the shift path, byte-identical.
            // gemv_q4ks variant: dot-y fold (fc2) and/or 2-simdgroup occupancy (fc3), per the lever
            // table above. Both can combine (uint fc set covers both); doty alone keeps the bool path.
            let r = if doty_on && nsg2_on {
                // DOTY (fc2) is a BOOL constant, NSG (fc3) a UINT — Metal type-checks each,
                // so the combo needs the mixed-FC compile (uint-only silently failed the
                // function build and dropped the whole island → WGSL fallback + checksum fork).
                isl.compile_msl_mixed_fc(ctx, "gemv_q4ks", gemv, "gemv_q4ks", &[(2usize, true)], &[(3usize, 2u32)])
            } else if doty_on {
                isl.compile_msl_bool_fc(ctx, "gemv_q4ks", gemv, "gemv_q4ks", 2, true)
            } else if nsg2_on {
                isl.compile_msl_uint_fc(ctx, "gemv_q4ks", gemv, "gemv_q4ks", &[(3usize, 2u32)])
            } else {
                isl.compile_msl(ctx, "gemv_q4ks", gemv, "gemv_q4ks")
            }
                .and_then(|()| isl.compile_msl(ctx, "geglu", ops, "geglu"))
                .and_then(|()| isl.compile_msl(ctx, "rmsnorm", rms, "rmsnorm"))
                .and_then(|()| isl.compile_msl(ctx, "rmsnorm_add", ops, "rmsnorm_add"))
                // L361 — Gemma 4 post-FFN pair (rmsnorm_add + layer_scalar) in ONE dispatch.
                .and_then(|()| isl.compile_msl(ctx, "rmsnorm_add_scale", ops, "rmsnorm_add_scale"))
                .and_then(|()| isl.compile_msl(ctx, "qkv_norm", ops, "qkv_norm"))
                .and_then(|()| isl.compile_msl(ctx, "rope_qk", ops, "rope_qk"))
                // INTERLEAVED (ggml NORM) rope for muse-glimmer. The island had only the
                // split-half kernel, so the megakernel scrambled q/k on every layer.
                .and_then(|()| isl.compile_msl(ctx, "rope_qk_interleaved", ops, "rope_qk_interleaved"))
                .and_then(|()| isl.compile_msl(ctx, "kv_scatter", ops, "kv_scatter"))
                .and_then(|()| isl.compile_msl(ctx, "scale_inplace", ops, "scale_inplace"))
                // FUSED gemma dense norm-pair (post-attn rms_add + pre-ffn rms → 1 dispatch);
                // gated by ARF_GEMMA_FUSE_ADDNORM2. Harmless to compile for any model.
                .and_then(|()| isl.compile_msl(ctx, "gemma_normpair", include_str!("shaders/metal/gemma_normpair_msl.metal"), "gemma_normpair"))
                // Muse Glimmer's attention output gate (`attn *= sigmoid(gate)`). Like
                // gemma_normpair above, it is arch-specific but harmless to compile for any
                // model — the island only dispatches it when `MegaLayer::attn_gate` is Some.
                .and_then(|()| isl.compile_msl(ctx, "attn_gate_mul", ops, "attn_gate_mul"))
                // SiLU gate for dense non-Gemma FFNs (llama / muse-glimmer). The island had only
                // the GELU kernel; a SiLU model run through it corrupts every FFN in every layer.
                .and_then(|()| isl.compile_msl(ctx, "swiglu", ops, "swiglu"))
                // ARF_ATTN_VEC4: compile attention_decode with the float4 QK path (fc0=true).
                // Default (unset) → the scalar path via compile_msl below, byte-identical.
                .and_then(|()| if vec4_on {
                    isl.compile_msl_bool_fc(ctx, "attention_decode", attn, "attention_decode", 0, true)
                } else {
                    isl.compile_msl(ctx, "attention_decode", attn, "attention_decode")
                })
                .and_then(|()| isl.compile_msl(ctx, "gemv_q8", ops, "gemv_q8"))
                // the `embed_tok` KEY serves every island gather (decode, verify, the draft):
                // compiled from the Q4_K kernel when the table is the GGUF's own Q4_K
                .and_then(|()| isl.compile_msl(ctx, "embed_tok", ops, if embed_q4k { "embed_tok_q4k" } else { "embed_tok" }))
                .and_then(|()| isl.compile_msl(ctx, "softcap", ops, "softcap"))
                .and_then(|()| isl.compile_msl(ctx, "argmax", ops, "argmax"))
                .and_then(|()| isl.compile_msl(ctx, "store_word", ops, "store_word"))
                .and_then(|()| isl.compile_msl(ctx, "gemv_q3k", include_str!("shaders/metal/matmul_vec_q3k_msl.metal"), "gemv_q3k"))
                .and_then(|()| isl.compile_msl(ctx, "attention_decode_fused", include_str!("shaders/metal/attention_fused_msl.metal"), "attention_decode_fused"))
                // M2 (Qwen3.6-27B gated-delta-net): causal depthwise conv1d + silu + ring advance.
                // One thread per conv channel; carries conv_state across tokens. Parity-gated in
                // ssm_conv1d_probe (no model). Harmless to compile unconditionally (unused unless
                // the hybrid gated-delta-net forward dispatches it at M4).
                .and_then(|()| isl.compile_msl(ctx, "ssm_conv1d", include_str!("shaders/metal/ssm_conv1d_msl.metal"), "ssm_conv1d"));
            // MoE (Qwen3) megakernel kernel set: the indirect-GEMV port + route/swiglu + the
            // Llama/Qwen fused add_norm / plain add_residual (the megakernel's gemma four-norm
            // flow doesn't apply). Only compiled when the model is MoE — dense models skip them.
            let r = if matches!(cfg.mlp, MlpKind::Moe { .. }) {
                r.and_then(|()| isl.compile_msl(ctx, "moe_route", include_str!("shaders/metal/moe_route_msl.metal"), "moe_route"))
                 .and_then(|()| isl.compile_msl(ctx, "moe_swiglu", include_str!("shaders/metal/moe_swiglu_msl.metal"), "moe_swiglu"))
                 .and_then(|()| isl.compile_msl(ctx, "gemv_q4ks_id", include_str!("shaders/metal/gemv_q4ks_id_msl.metal"), "gemv_q4ks_id"))
                 .and_then(|()| isl.compile_msl(ctx, "gemv_q4ks_id_down", include_str!("shaders/metal/gemv_q4ks_id_down_msl.metal"), "gemv_q4ks_id_down"))
            } else { r };
            // L145 — add_norm / add_residual are the LLAMA/QWEN FUSED post-attention norm flow
            // (hidden += attn_out; normed = rmsnorm(hidden)*post_norm). They were compiled inside
            // the MoE-only block above, but the flow is keyed on the NORM STYLE, not on the FFN
            // kind: Qwen3.8 is DENSE and still uses it, so the batched record refused every step
            // with "pso add_norm (fused llama/qwen norm)". Exactly the mismatch the qk_norm comment
            // below documents — a kernel gated on MoE that dense models also need.
            //
            // Gate on "not the gemma four-norm flow" instead. Gemma/muse-glimmer take the
            // pre_ffn/post_ffn norm pair and never dispatch these, so they still skip the compile.
            let r = if !cfg.has_post_norms() {
                r.and_then(|()| isl.compile_msl(ctx, "add_norm", include_str!("shaders/metal/add_norm_msl.metal"), "add_norm"))
                 .and_then(|()| isl.compile_msl(ctx, "add_residual", include_str!("shaders/metal/add_norm_msl.metal"), "add_residual"))
            } else { r };
            // qk_norm (q/k-only per-head RMSNorm) — needed by ANY model with qk_norm && !value_norm
            // (Qwen MoE AND dense gemma3-4b), NOT MoE-specific. The megakernel's `else` norm branch
            // (concurrent_metal.rs, value_norm=false) dispatches this pipeline; gating it on MoE left
            // dense qk-norm models (gemma3) with p_qk=None → greedy-megakernel panic. Gemma-4 uses the
            // fused qkv_norm (value_norm=true) instead, so it doesn't need qk_norm.
            let r = if cfg.qk_norm && !cfg.value_norm {
                r.and_then(|()| isl.compile_msl(ctx, "qk_norm", ops, "qk_norm"))
            } else { r };
            // MILESTONE 4: B-ROW batched megakernel kernel set. DEFAULT-ON (opt-out via
            // ARF_NO_BATCH_MEGA) to MATCH the runtime dispatch gate in batch.rs:425
            // (`if ARF_NO_BATCH_MEGA.is_some() { return None }`). The compile gate was
            // previously opt-in (`ARF_BATCH_MEGA.is_some()`) — a MISMATCH: the runtime tried
            // the megakernel by default but the kernels were never compiled, silently falling
            // back to the slow serial path (the coder-30B conc regression, ~5× single-stream).
            // Now the fast path is genuinely default so `make dev`/`dev-conc` hit the 90/160
            // megakernel-frame speeds. (The m=1 compile set is byte-unchanged; this only ADDS the
            // B-row kernels.) parity-proven in batched_mega_parity.
            // ⚠️ The `MlpKind::Moe` gate below covers this WHOLE block, but the block is not all
            // MoE: everything up to `gemv_q4ks_id_b` is GENERIC — batched attention, the f16 KV
            // cache, coalesced + kv-head attention, the batched norms/rope. Gating those on MoE
            // meant a DENSE model got none of the bandwidth levers: muse-glimmer logged
            // `[m1-kv-f16] engaged=false (scatter_pso=false attn_pso=false)` because the
            // pipelines were never compiled, and f16 KV alone is worth +37% at conc1 (L135).
            // `batch_mega` keeps the MoE condition (its shaders genuinely need MoE config);
            // `attn_set` drops it so dense models compile the generic half too.
            let batch_mega = matches!(cfg.mlp, MlpKind::Moe { .. })
                && std::env::var_os("ARF_NO_BATCH_MEGA").is_none();
            let attn_set = std::env::var_os("ARF_NO_BATCH_MEGA").is_none();
            let r = if r.is_ok() && attn_set {
                let ops_b = include_str!("shaders/metal/decode_ops_batch_msl.metal");
                let moe_b = include_str!("shaders/metal/moe_batch_msl.metal");
                // mm_id tile precision: default HALF (perf, half-mul + f32-acc, ~2-3e-3 vs GEMV →
                // token-coherence-gated); ARF_MOE_MM_F32 prepends the f32-tile define (parity-tight).
                let moe_mm_id_base = include_str!("shaders/metal/moe_mm_id_q4ks.metal");
                // Block-dequant hoist: lift the Q4_K_S scale/min derivation out of the per-nibble
                // staging unroll → once per (thread,sub-block). DEFAULT-ON (ARF_NO_MOE_MM_HOIST=1
                // opts out). MEASURED +24% @conc64 (195→148ms, 328→432 agg, gap 1.23→0.93× — PAST
                // llama parity, 2 cold runs reproduce). The scalar dequant prologue was the serial
                // PRODUCER stalling the simdgroup matmul; hoisting it unblocks the matrix units (an
                // ALU-bound producer-unlock, non-linear). Bit-identical to the un-hoisted staging
                // (kept the s*nib+lo two-step op order); parity max_abs 1.192e-5, greedy-token-safe.
                let mut moe_mm_prefix = String::new();
                if std::env::var_os("ARF_MOE_MM_F32").is_some() { moe_mm_prefix.push_str("#define MOE_MM_F32 1\n"); }
                // L29 — THE OVERLAP FAMILY IS DELETED (all four measured flat or worse, 2026-07-15/20):
                //   DBUF (double-buffered stage/matmul):  GPU-busy ~151ms vs ~150ms baseline = NO win
                //   PREFETCH (software-pipelined loads):  GPU-busy ~162ms vs ~150ms = WORSE
                //   LOWSMEM (small-footprint writeback):  occupancy 12288->6144, FLAT
                //   DEQSHORTCUT (measurement-only probe): proved dequant HIDES latency, not exposes it
                // "the whole OVERLAP family is now ruled out" (2026-07-15).
                // The MOE_MM_{DBUF,PREFETCH,LOWSMEM,DEQSHORTCUT} #ifdef blocks
                // remain in moe_mm_id_q4ks.metal but are now UNREACHABLE: nothing defines them.
                if std::env::var_os("ARF_NO_MOE_MM_HOIST").is_none() {
                    moe_mm_prefix.push_str("#define MOE_MM_HOIST 1\n");
                }
                // L77 MEASUREMENT-ONLY probe (ARF_MOE_MM_GATHERPROBE): force the gate/up
                // activation gather CONTIGUOUS (src_row = tile-local row) instead of following the
                // CSR scatter. Identical byte count / MMA count / weight traffic — only locality
                // changes. Tests whether the L76 gate+up anomaly (3.06x for 1.33x rows) is the
                // scattered `a[src_row*arow + kelem]` read. OUTPUT IS GARBAGE — read GPU-busy only,
                // never logits. Guarded so the default path is untouched.
                if std::env::var_os("ARF_MOE_MM_GATHERPROBE").is_some() {
                    moe_mm_prefix.push_str("#define MOE_MM_GATHERPROBE 1\n");
                }
                // NR1 CROSSOVER (dual-compile): the mm_id row-tile height NR1 is a COMPILE-TIME
                // #define, so ONE define = ONE kernel. To pick NR1 per-batch at runtime we compile
                // BOTH variants and select at dispatch (concurrent_metal.rs batched_megakernel_record):
                //   * NR1=32 (base names, the llama tile) WINS conc64 (~4 rows/expert fills the tile).
                //   * NR1=8  (`_nr8` names) WINS conc16/32 (cuts the ~87% masked padding at low fill).
                // MEASURED (Chrome-dead, qwen3-coder-30B q4ks): conc16 176 vs 168, conc32 270 vs 249
                // (NR1=8 wins both, beats llama 259), conc64 412 vs 399 (NR1=32 keeps the win vs 400).
                // The BASE names are ALWAYS NR1=32; the `_nr8` twins are ALWAYS NR1=8 — same source,
                // only the MOE_MM_NR1 define differs. The CSR builder's CSR_MM_BM (== NR1) MUST match
                // the GEMM NR1 it feeds, so each variant compiles its OWN CSR kernel with the matching
                // define (base CSR = NR1=32, `_nr8` CSR = NR1=8). MOE_MM_HOIST (the +24% default) stays
                // on for BOTH. (Legacy ARF_MOE_NR1_8 is now a RUNTIME force-8 flag, not a compile
                // switch — it just makes the dispatch pick the `_nr8` set for b>=2.)
                let moe_mm_id_owned;
                let moe_mm_id: &str = if moe_mm_prefix.is_empty() { moe_mm_id_base } else {
                    moe_mm_id_owned = format!("{moe_mm_prefix}{moe_mm_id_base}");
                    &moe_mm_id_owned
                };
                let csr_b: &str = moe_b; // base CSR = NR1=32 (shader default), matches the base GEMM.
                // NR1=8 twins: same source, prepend the MOE_MM_NR1=8 define to BOTH the GEMM and its CSR.
                let nr8_define = "#define MOE_MM_NR1 8\n";
                let moe_mm_id_nr8 = format!("{nr8_define}{moe_mm_prefix}{moe_mm_id_base}");
                let csr_b_nr8 = format!("{nr8_define}{moe_b}");
                // Dense q/k/v/o tiled GEMM source, with the optional block-dequant hoist define.
                let dense_mm_base = include_str!("shaders/metal/matmul_mm_q4ks.metal");
                let dense_mm_src: std::borrow::Cow<str> = if std::env::var_os("ARF_DENSE_MM_HOIST").is_some() {
                    std::borrow::Cow::Owned(format!("#define DENSE_MM_HOIST 1\n{dense_mm_base}"))
                } else { std::borrow::Cow::Borrowed(dense_mm_base) };
                // L39 OCCUPANCY FIX — size the attention threadgroup arrays to THIS MODEL's head_dim
                // instead of the worst-case bounds baked into the shader (MAXHD=512 for gemma,
                // KVH_MAXHD=256). MEASURED with ARF_OCCUPANCY=1 on qwen3-coder (hd=128):
                //   attention_decode_coalesced_b_f16  staticSmem 16448 B -> 1 TG resident / 32KiB core
                //   attention_decode_kvhead_b_f16     staticSmem 16384 B -> 2 TG resident
                // One threadgroup per core means a DRAM stall has NOTHING to switch to — the direct
                // mechanism behind L37's "2% of bandwidth AND 6% of compute -> STALLED", with
                // attention at 47% of the step (B=16). At hd=128 the arrays need 4096 B / 8192 B,
                // taking residency to 7 / 4.
                // `max_attn_dims()` (not cfg.head_dim) is the correct source: Gemma 4's GLOBAL layers
                // use hd=512 while sliding layers use 256, and the bound must cover EVERY layer.
                // Rounded up to a 128 multiple (the kernels require hd % 128 == 0) and floored at 128.
                let (attn_max_hd, _) = cfg.max_attn_dims();
                let attn_hd_fit = attn_max_hd.max(128).div_ceil(128) * 128;
                // L39: the fit is UNCONDITIONAL. It measured FLAT on throughput (occupancy rose
                // 1->7 TG/core, no tier moved), so under the env ratchet the opt-out flag is
                // deleted rather than kept — the rule reserves NO_* hatches for defaults that WON.
                // The sizing itself stays because it is simply correct: reserving 16 KB of
                // threadgroup memory for gemma's hd=512 while running qwen3's hd=128 is a bug
                // regardless of whether fixing it shows up in tok/s.
                let attn_b_owned: std::borrow::Cow<'_, str> =
                    {
                        // L40 MEASURED FLAT: keys-per-staged-tile 8/16/32/64 all within the 3.3%
                        // floor at every tier (llama blocks 64 via OP_FLASH_ATTN_EXT_NCPSG; matching
                        // it bought nothing). Flag deleted under the env ratchet; 16 is the shipped
                        // constant in the shader.
                        std::borrow::Cow::Owned(format!(
                            "#define ATTN_MAXHD {attn_hd_fit}u\n#define ATTN_KVH_MAXHD {attn_hd_fit}u\n{}",
                            include_str!("shaders/metal/attention_msl.metal")))
                    };
                let attn_b: &str = &attn_b_owned;
                // L167 — ARF_GEMV_B_NSG=<n>: compile the BATCHED GEMV with n simdgroups per
                // threadgroup (function constant 3). The m=1 `gemv_q4ks` has had this lever since
                // the nsg2 work; the batched kernel — which is the one every daemon token actually
                // goes through — never got it. Default unset = NSG 1 = byte-identical shipped path.
                // The host must dispatch n*32 threads to match (see `tg_gemv_b`).
                // MEASURED (L167, B=1, streamed p50 gap, same binary, one env var apart):
                //     NSG=1  76.6 ms  13.06 tok/s   (the shipped path)
                //     NSG=2  71.0 ms  14.08 tok/s   +7.8%
                //     NSG=4  68.6 ms  14.58 tok/s   +11.6%  <-- DEFAULT
                //     NSG=8  93.9 ms  10.65 tok/s   -18%    (falls off — a real occupancy curve)
                // Text identical at every setting. Set ARF_GEMV_B_NSG=1 to restore the old path.
                r.and_then(|()| match std::env::var("ARF_GEMV_B_NSG").ok().and_then(|v| v.parse::<u32>().ok()).or(Some(4)) {
                    Some(n) if n > 1 => {
                        // L196 — optional ACC_CAP probe (function constant 4) alongside NSG.
                        let mut fcs = vec![(3usize, n)];
                        if let Some(c) = std::env::var("ARF_GEMV_ACC_CAP").ok()
                            .and_then(|v| v.parse::<u32>().ok()) { fcs.push((4usize, c)); }
                        // L220 — COLS_B: output columns per threadgroup (1/2/4). Divides the
                        // activation traffic that L219 proved is 93% of b=3 bytes.
                        // L220 — DEFAULT 4. Measured: marginal row cost 0.96 -> 0.29 and b=1
                        // 18.1-18.5 -> 19.6-20.4 tok/s (interleaved, 4 arms). Set 1 to restore.
                        {
                            let c = std::env::var("ARF_GEMV_B_COLS").ok()
                                .and_then(|v| v.parse::<u32>().ok())
                                .filter(|&c| (1..=8).contains(&c)).unwrap_or(4);
                            fcs.push((5usize, c));
                            // L249 — ACC_ROWS (fc 6): the per-column row stride. Sizing it to the
                            // real max batch instead of MAXB=64 is what makes COLS=8 fit in the
                            // same registers COLS=4 uses today. MUST be >= the largest d.m ever
                            // dispatched, so it is driven by --max-batch-size, never guessed.
                            if let Some(r) = std::env::var("ARF_GEMV_ACC_ROWS").ok()
                                .and_then(|v| v.parse::<u32>().ok()).filter(|&r| r >= 1)
                                { fcs.push((6usize, r)); }
                            // L363s — ARF_GEMV_COLMAJOR=1 compiles the columns-outer/rows-inner
                            // nesting (function constant 7). Default unset = the shipped row-outer
                            // path, byte-identical. See L357/L363r for why this is the candidate.
                            if std::env::var_os("ARF_GEMV_COLMAJOR").is_some() {
                                fcs.push((7usize, 1));
                            }
                        }
                        // L363j — the WIDE twin. ACC_ROWS is sized to --max-batch-size for the
                        // decode record's registers (L249/L343), but the batched record ALSO runs
                        // prefill-as-rows (L167): a chunk of up to 64 rows per tiled dispatch
                        // (L143). Rows >= ACC_ROWS are never written by the kernel, so a 16-row
                        // prompt through a 4-row accumulator came out with rows 4..15 stale —
                        // gemma answered as if it had never seen the prompt (L363i). Same source,
                        // same function constants, minus fc 6: ACC_ROWS falls back to MAXB=64,
                        // the pre-L249 shipped shape. The record picks it only when b > ACC_ROWS,
                        // so every decode step keeps the sized kernel byte-for-byte.
                        let fcs_wide: Vec<(usize, u32)> =
                            fcs.iter().copied().filter(|(i, _)| *i != 6).collect();
                        isl.compile_msl_uint_fc(
                            ctx, "gemv_q4ks_batch",
                            include_str!("shaders/metal/matmul_vec_q4ks_batch_msl.metal"),
                            "gemv_q4ks_batch", &fcs)
                        .and_then(|()| isl.compile_msl_uint_fc(
                            ctx, "gemv_q4ks_batch_wide",
                            include_str!("shaders/metal/matmul_vec_q4ks_batch_msl.metal"),
                            "gemv_q4ks_batch", &fcs_wide))
                    }
                    _ => isl.compile_msl(ctx, "gemv_q4ks_batch", include_str!("shaders/metal/matmul_vec_q4ks_batch_msl.metal"), "gemv_q4ks_batch"),
                })
                // L245 — MTP draft head's input projection (`mtp_eh_concat`). Compiled whenever
                // the island is up; the pipeline is simply unused when the model has no head.
                .and_then(|()| isl.compile_msl(
                    ctx, "mtp_eh_concat",
                    include_str!("shaders/metal/mtp_eh_proj_msl.metal"),
                    "mtp_eh_concat"))
                 // B-ROW attention + kv_scatter (collapse the per-seq dispatch loop into
                 // ONE B-row dispatch each — 3072→48 attention + 3072→48 scatter dispatches/token
                 // @conc64×48 layers). Engaged at runtime by ARF_ATTN_BATCHED; compiling them
                 // unconditionally here is harmless when the flag is off (pipelines unused).
                 .and_then(|()| isl.compile_msl(ctx, "attention_decode_b", attn_b, "attention_decode_b"))
                 // SPLIT-KV attention (ARF_ATTN_SPLITK): two-pass partials + combine, flattening
                 // the KV-growth ramp. splitk==1 is bit-exact identical to _b. Default-OFF; harmless
                 // to compile when the gate flag is unset (pipelines unused).
                 .and_then(|()| isl.compile_msl(ctx, "attention_decode_splitk_b", attn_b, "attention_decode_splitk_b"))
                 .and_then(|()| isl.compile_msl(ctx, "attention_splitk_combine_b", attn_b, "attention_splitk_combine_b"))
                 // COALESCED simdgroup attention (ARF_ATTN_COALESCED): 32-lane coalesced KV
                 // co-load + simd_sum score + barrier-free online softmax — the conc64 memory-latency
                 // WIN. Bindings 0-6 identical to _b; dispatch swaps PSO + grid + tg size (tg32).
                 // Default-OFF; compiling unconditionally is harmless when the flag is unset.
                 .and_then(|()| isl.compile_msl(ctx, "attention_decode_coalesced_b", attn_b, "attention_decode_coalesced_b"))
                 // PREFILL attention, kv-head x row-block (2026-09-23). OPTIONAL: a failed
                 // compile leaves the committed prefill window on the coalesced kernel.
                 .map(|()| {
                     if let Err(e) = isl.compile_msl(ctx, "attention_prefill_kvhead_f32", attn_b, "attention_prefill_kvhead_f32") {
                         eprintln!("[msl] prefill attention kernel did not compile ({e}); prefill windows use the per-row kernel");
                     }
                     for k in ["attention_prefill_fa_f32", "attention_prefill_fa32_f32", "attention_prefill_fa_combine",
                               "kv_scatter_b_q8", "attention_decode_coalesced_b_q8",
                               // 8-bit KV decode, kv-head split-K (opt-in, `ARF_ATTN_Q8_SPLIT`)
                               "attention_decode_kvhead_splitk_b_q8"] {
                         if let Err(e) = isl.compile_msl(ctx, k, attn_b, k) {
                             eprintln!("[msl] {k} did not compile ({e}); prefill windows use the scalar kernel");
                         }
                     }
                 })
                 // f16 KV cache (the 3rd leg: bandwidth). f16 scatter (writes half) +
                 // f16-load coalesced attention (reads half4 -> float4, f32 math). Engaged only when
                 // BOTH ARF_KV_F16 and ARF_ATTN_COALESCED are set (see concurrent_metal.rs);
                 // default-OFF, so compiling unconditionally is harmless off-path.
                 .and_then(|()| isl.compile_msl(ctx, "kv_scatter_b_f16", attn_b, "kv_scatter_b_f16"))
                 .and_then(|()| isl.compile_msl(ctx, "attention_decode_coalesced_b_f16", attn_b, "attention_decode_coalesced_b_f16"))
                 // KV-HEAD-CENTRIC f16 attention (ARF_ATTN_KVHEAD): tg per (kv_head, seq), the
                 // GQA group's 8 q-heads share ONE staged KV read (traffic /8, compounds with f16).
                 .and_then(|()| isl.compile_msl(ctx, "attention_decode_kvhead_b_f16", attn_b, "attention_decode_kvhead_b_f16"))
                 // ... and its SPLIT-K small-batch form (grid (nkv,B,S) + the existing splitk
                 // combine): occupancy below 64 tgs AND the 8x traffic cut — the conc8/16 lever.
                 .and_then(|()| isl.compile_msl(ctx, "attention_decode_kvhead_splitk_b_f16", attn_b, "attention_decode_kvhead_splitk_b_f16"))
                 // f16 KV prefill-mirror: mirror the f32 KV pool -> f16 pool over the valid
                 // range after the WGSL prefill scatter (which writes only the f32 pool). Gated by
                 // ARF_KV_F16 at dispatch (forward_batch_impl); compiling unconditionally is
                 // harmless off-path (pipeline unused when the flag is off).
                 .and_then(|()| isl.compile_msl(ctx, "kv_f32_to_f16_convert", attn_b, "kv_f32_to_f16_convert"))
                 // f16-mode SLOT-LIST pool sync (double-pool residency fix): export the slots a
                 // prefill batch WROTE (f32 staging -> f16 truth) / import the past slots it READS
                 // (f16 -> f32). Replaces the whole-pool mirror in production (which corrupted
                 // decoded-token f16 slots on mid-stream admissions, REQS>CONC). Gated at dispatch.
                 .and_then(|()| isl.compile_msl(ctx, "kv_f32_to_f16_slots", attn_b, "kv_f32_to_f16_slots"))
                 .and_then(|()| isl.compile_msl(ctx, "kv_f16_to_f32_slots", attn_b, "kv_f16_to_f32_slots"))
                 // SHARED-PREFIX VERIFY attention (MTP / spec-decode): prefix staged ONCE, k verify
                 // rows consume it + a k×k causal window. Engaged by try_batched_megakernel_verify
                 // (ARF_NO_VERIFY_MEGA opts out); unconditional compile is harmless off-path.
                 .and_then(|()| isl.compile_msl(ctx, "attention_verify_shared_prefix", attn_b, "attention_verify_shared_prefix"))
                 // L95 — f16-KV twin of the verify kernel. Same body, `half` KV pools (see the
                 // VERIFY_KV_T macro in attention_msl.metal). Compiled unconditionally; only
                 // dispatched when the f16 verify path is armed, so this is off-path by default.
                 .and_then(|()| isl.compile_msl(ctx, "attention_verify_shared_prefix_f16", attn_b, "attention_verify_shared_prefix_f16"))
                 .and_then(|()| isl.compile_msl(ctx, "kv_scatter_b", attn_b, "kv_scatter_b"))
                 .and_then(|()| isl.compile_msl(ctx, "rmsnorm_b", ops_b, "rmsnorm_b"))
                 .and_then(|()| isl.compile_msl(ctx, "qk_norm_b", ops_b, "qk_norm_b"))
                 // L360g — the B-row qkv_norm: Gemma normalizes V too (`value_norm`, weightless).
                 // Without it the batched record dispatched qk_norm_b for every model, leaving
                 // Gemma's V unnormalized on every layer.
                 .and_then(|()| isl.compile_msl(ctx, "qkv_norm_b", ops_b, "qkv_norm_b"))
                 // L363e — B-row layer_scalar; the m=1 scale_inplace only ever covered row 0.
                 .and_then(|()| isl.compile_msl(ctx, "scale_inplace_b", ops_b, "scale_inplace_b"))
                 .and_then(|()| isl.compile_msl(ctx, "rope_qk_b", ops_b, "rope_qk_b"))
                 // INTERLEAVED batched rope (muse-glimmer). L155 added this for m=1 only.
                 .and_then(|()| isl.compile_msl(ctx, "rope_qk_b_interleaved", ops_b, "rope_qk_b_interleaved"))
                // Qwen3.8 IMAGE rows (2026-09-27): interleaved M-RoPE. Hybrid arch only, and
                // NON-FATAL — a failure here only disables image input (the record refuses an
                // image batch without it); it can never take the text path down.
                .map(|()| {
                    if matches!(cfg.attn, AttnKind::HybridSsmAttn { .. }) {
                        if let Err(e) = isl.compile_msl(ctx, "rope_qk_b_mrope", ops_b, "rope_qk_b_mrope") {
                            eprintln!("[island] rope_qk_b_mrope did not compile ({e}); image input disabled");
                        }
                    }
                })
                 .and_then(|()| isl.compile_msl(ctx, "add_residual_b", ops_b, "add_residual_b"))
                 // DENSE-FFN batched kernels (muse-glimmer / any dense model at conc>1). The MoE
                 // twins are laid out by (row, expert-slot, col) and cannot serve a dense FFN.
                 .and_then(|()| isl.compile_msl(ctx, "copy_b", ops_b, "copy_b"))
                 .and_then(|()| isl.compile_msl(ctx, "bank_copy", ops_b, "bank_copy"))
                 .and_then(|()| isl.compile_msl(ctx, "swiglu_b", ops_b, "swiglu_b"))
                 .and_then(|()| isl.compile_msl(ctx, "geglu_b", ops_b, "geglu_b"))
                 .and_then(|()| isl.compile_msl(ctx, "rmsnorm_add_b", ops_b, "rmsnorm_add_b"))
                 .and_then(|()| isl.compile_msl(ctx, "attn_gate_mul_b", ops_b, "attn_gate_mul_b"))
                 // L136 — gated-delta-net readout norm-gate (step 6, build_norm_gated). Only the
                 // hybrid arch dispatches it; compiled only for that arch so nothing else pays.
                 .and_then(|()| if matches!(cfg.attn, AttnKind::HybridSsmAttn { .. }) {
                     isl.compile_msl(ctx, "gdn_norm_gate_b", ops_b, "gdn_norm_gate_b")
                 } else { Ok(()) })
                 // L155 — the JOINT query+gate split for this family's FULL-ATTENTION layers
                 // (`wq` emits query‖gate interleaved per head). Same arch gate as above.
                 .and_then(|()| if matches!(cfg.attn, AttnKind::HybridSsmAttn { .. }) {
                     isl.compile_msl(ctx, "qg_split_b", ops_b, "qg_split_b")
                 } else { Ok(()) })
                 // ── MoE-ONLY from here: these shaders need MoE config (experts/top_k). A dense
                 // model stops at the generic set above, which is exactly what it can use.
                 .and_then(|()| if !batch_mega { Ok(()) } else { isl.compile_msl(ctx, "gemv_q4ks_id_b", moe_b, "gemv_q4ks_id_b") })
                 .and_then(|()| isl.compile_msl(ctx, "gemv_q4ks_id_down_b", moe_b, "gemv_q4ks_id_down_b"))
                 // top_k-parallel down: per-slot partial GEMV + a tiny reduce (ARF_DOWN_PARALLEL).
                 .and_then(|()| isl.compile_msl(ctx, "gemv_q4ks_id_down_partial_b", moe_b, "gemv_q4ks_id_down_partial_b"))
                 .and_then(|()| isl.compile_msl(ctx, "moe_down_reduce_b", moe_b, "moe_down_reduce_b"))
                 .and_then(|()| isl.compile_msl(ctx, "moe_route_b", moe_b, "moe_route_b"))
                 .and_then(|()| isl.compile_msl(ctx, "moe_swiglu_b", moe_b, "moe_swiglu_b"))
                 // OPT-IN tiled Q4_K_S dense GEMM (llama kernel_mul_mm port). Replaces the
                 // per-row dense GEMV (q/k/v/o) with a tiled simdgroup-style matmul that stages
                 // each weight tile in threadgroup memory ONCE and reuses it across all B rows
                 // (up to 32x weight-traffic reuse @conc32). Engaged at runtime only when
                 // ARF_MM_Q4KS is set AND b>=8 (see concurrent_metal.rs p_mm); compiling it
                 // unconditionally here is harmless when the flag is off (pipeline unused).
                 // DENSE_MM_HOIST (ARF_DENSE_MM_HOIST, default-OFF): the SAME block-dequant hoist
                 // that won +24% on the MoE mm_id kernel — the dense q/k/v/o staging loop carries the
                 // identical 32× per-element scale/min re-read. Default-off pending its own cold A/B.
                 .and_then(|()| isl.compile_msl(ctx, "matmul_mm_q4ks", &dense_mm_src, "matmul_mm_q4ks"))
                 // BM=64 build of the SAME source (MM_BM64 define): 2x weight reuse per staged
                 // tile, half the row-tile grid at prefill. Compiled always so ARF_MM_BM64
                 // can A/B it at runtime; costs one pipeline and nothing at dispatch.
                 .and_then(|()| {
                     let src = format!("#define MM_BM64 1\n{dense_mm_src}");
                     isl.compile_msl(ctx, "matmul_mm_q4ks_bm64", &src, "matmul_mm_q4ks")
                 })
                 // HALF INPUT FRAGMENTS build (MM_HALF_FRAG): half af/wf staging + half smem
                 // tiles, float accumulator. The register lever for a register-bound kernel.
                 .and_then(|()| {
                     let src = format!("#define MM_HALF_FRAG 1\n{dense_mm_src}");
                     isl.compile_msl(ctx, "matmul_mm_q4ks_half", &src, "matmul_mm_q4ks")
                 })
                 // v2 dense GEMM (ARF_MM_V2): KC=64 f32-tile variant, A/B vs v1. Harmless if unused.
                 .and_then(|()| isl.compile_msl(ctx, "matmul_mm_q4ks_v2", include_str!("shaders/metal/matmul_mm_q4ks_v2.metal"), "matmul_mm_q4ks_v2"))
                 // SORT-BY-EXPERT tiled MoE (ARF_MOE_SORTED, default-OFF): the CSR scatter (llama
                 // map0) + grouped tiled mm_id for gate/up/down (per-EXPERT weight reuse) + the slot-
                 // reduce. Compiled unconditionally when ARF_BATCH_MEGA is on (harmless if the
                 // runtime flag is off — the GEMV path stays default). The down GEMM writes into the
                 // reused down_partials scratch; moe_reduce_slots_b folds the router weight.
                 .and_then(|()| isl.compile_msl(ctx, "moe_expert_csr_msl", csr_b, "moe_expert_csr_msl"))
                 .and_then(|()| isl.compile_msl(ctx, "moe_reduce_slots_b", moe_b, "moe_reduce_slots_b"))
                 .and_then(|()| isl.compile_msl(ctx, "moe_mm_id_gu_q4ks", moe_mm_id, "moe_mm_id_gu_q4ks"))
                 .and_then(|()| isl.compile_msl(ctx, "moe_mm_id_down_q4ks", moe_mm_id, "moe_mm_id_down_q4ks"))
                 // NR1=8 TWINS (the crossover's low-batch set): same source, MOE_MM_NR1=8. The CSR
                 // builder (`_nr8`) carries the matching CSR_MM_BM=8 so its tile-list stride equals
                 // the `_nr8` GEMM's NR1 — the parity-critical coupling (stride ≠ NR1 = garbage).
                 // Selected at runtime for b<64 (concurrent_metal.rs batched_megakernel_record).
                 .and_then(|()| isl.compile_msl(ctx, "moe_expert_csr_msl_nr8", &csr_b_nr8, "moe_expert_csr_msl"))
                 .and_then(|()| isl.compile_msl(ctx, "moe_mm_id_gu_q4ks_nr8", &moe_mm_id_nr8, "moe_mm_id_gu_q4ks"))
                 .and_then(|()| isl.compile_msl(ctx, "moe_mm_id_down_q4ks_nr8", &moe_mm_id_nr8, "moe_mm_id_down_q4ks"))
                 // batched Q8 lm_head + argmax (ARF_LM_HEAD_BATCHED). The B-row twins of
                 // gemv_q8/argmax — stream the 311 MB vocab matrix ONCE for all B rows (the conc64
                 // lm_head lever). Compiled unconditionally here; engaged at runtime by the flag.
                 .and_then(|()| isl.compile_msl(ctx, "gemv_q8_b", ops, "gemv_q8_b"))
                 .and_then(|()| isl.compile_msl(ctx, "gemv_q8b32_b", ops, "gemv_q8b32_b")) // ARF_Q8_DOWN
                 .and_then(|()| isl.compile_msl(ctx, "argmax_b", ops, "argmax_b"))
                 // TILED Q8 lm_head GEMM: the ALU-optimal twin of gemv_q8_b — feeds the Q8 vocab tile
                 // to the simdgroup matrix units instead of the scalar-FMA acc[64] loop. DEFAULT-ON
                 // (322 agg vs gemv_q8_b's 290 @conc64, parity-clean); ARF_NO_LM_HEAD_GEMM=1 opts out.
                 .and_then(|()| isl.compile_msl(ctx, "matmul_mm_q8", include_str!("shaders/metal/matmul_mm_q8.metal"), "matmul_mm_q8"))
                 .map(|()| { eprintln!("[msl] B-row batched megakernel kernel set compiled (ARF_BATCH_MEGA, +matmul_mm_q4ks tiled GEMM, +sort-by-expert mm_id, +batched lm_head, +tiled Q8 lm_head GEMM)"); })
            } else { r };
            match r {
                Ok(()) => {
                    eprintln!("[msl] native Metal island up — full megakernel kernel set compiled");
                    // MPP Q4 matmul for batched decode (plan 2.3) — DEFAULT ON, and OPTIONAL at
                    // compile time: a failure is "feature off", never an error. An OS older than
                    // macOS 26 has no MPP tensor ops and must load exactly as before.
                    {
                        let mpp = include_str!("shaders/metal/matmul_mpp_q4ks_msl.metal");
                        let ok = [8usize, 16, 32].iter().try_for_each(|r| {
                            ["q4tile_blockmax", "q4tile_scale", "q4tile_narrow", "q4rm_mm"]
                                .iter()
                                .try_for_each(|k| {
                                    let key = format!("{k}_r{r}");
                                    isl.compile_msl(ctx, &key, mpp, &key)
                                })
                        })
                        // the prefill matmul exists at 32 rows only
                        // the prefill matmul (`q4pf2_mm`) at its two shapes
                        .and_then(|()| {
                            ["q4pf2_mm_r32_n64", "q4pf2_mm_r128_n64"]
                                .iter()
                                .try_for_each(|k| isl.compile_msl(ctx, k, mpp, k))
                        })
                        // the fused operand pass (2026-09-27): optional on its own — a failure
                        // leaves the per-tile passes in charge, never fails this chain
                        .inspect(|()| {
                            if let Err(e) = isl.compile_msl(ctx, "q4tile_prep_win_t32", mpp, "q4tile_prep_win_t32") {
                                eprintln!("[msl] fused operand pass not compiled ({e}) — the per-tile passes serve");
                            }
                        })
                        // the 8-bit flash prefill on matmul2d: optional on its own (a failure
                        // leaves the simdgroup kernels in charge), so it never fails this chain
                        .inspect(|()| {
                            let src = include_str!("shaders/metal/attention_q8_mpp_msl.metal");
                            let r = ["pfa_q8_mpp_stage_q", "attention_prefill_fa_q8_mpp_b1", "attention_prefill_fa_q8_mpp_combine"]
                                .iter()
                                .try_for_each(|k| isl.compile_msl(ctx, k, src, k));
                            if let Err(e) = r {
                                eprintln!("[msl] 8-bit MPP flash prefill not compiled ({e}) — the simdgroup kernels serve");
                            } else {
                                // the wide-key forms (default NB = 2; ARF_PFA_NB=1|4): optional on their
                                // own — a failure leaves the one-block form in charge
                                for k in ["attention_prefill_fa_q8_mpp_b2", "attention_prefill_fa_q8_mpp_b4"] {
                                    let _ = isl.compile_msl(ctx, k, src, k);
                                }
                            }
                        })
                        // the split decode matmul + its sum, at the two decode units
                        .and_then(|()| {
                            [8usize, 16].iter().try_for_each(|r| {
                                ["q4rm_split", "q4rm_sum"].iter().try_for_each(|k| {
                                    let key = format!("{k}_r{r}");
                                    isl.compile_msl(ctx, &key, mpp, &key)
                                })
                            })
                        });
                        // The hand-written simdgroup_matrix decode matmul (2026-09-21): 1.31-1.48x
                        // over `q4rm_mm` on every decode shape at the SAME Q4_K_S layout. Behind
                        // ARF_SG_Q4=1 until the end-to-end A/B is recorded. Optional exactly
                        // like the rest of this block: a compile failure is "feature off".
                        let sg_ok = ok.is_ok()
                            && [8usize, 16].iter().all(|r| {
                                ["q4sg_prepare", "q4sg_mm"].iter().all(|k| {
                                    let key = format!("{k}_r{r}");
                                    isl.compile_msl(ctx, &key, mpp, &key).is_ok()
                                })
                            });
                        // v2 (2026-09-23): the scale inside the MMA operand — see `q4sg2_mm`.
                        let sg2_ok = sg_ok
                            && [8usize, 16].iter().all(|r| {
                                ["q4sg2_prepare", "q4sg2_mm"].iter().all(|k| {
                                    let key = format!("{k}_r{r}");
                                    isl.compile_msl(ctx, &key, mpp, &key).is_ok()
                                })
                            });
                        if sg_ok && !sg2_ok {
                            eprintln!("[msl] ⚠ q4sg2 (scale-in-operand) did NOT compile — the v1 simdgroup_matrix kernel is in use");
                        } else if sg2_ok && std::env::var_os("ARF_SG_V1").is_none() {
                            eprintln!("[msl] q4sg2: sub-block scale inside the MMA operand, d once per super-block (ARF_SG_V1=1 = the v1 kernel)");
                        }
                        if sg_ok && std::env::var_os("ARF_NO_SG_Q4").is_none() {
                            eprintln!("[msl] simdgroup_matrix split-K Q4 decode matmul ON at 8 and 16 rows (ARF_NO_SG_Q4 opts out)");
                        } else if !sg_ok {
                            eprintln!("[msl] ⚠ simdgroup_matrix Q4 kernels did NOT compile — the MPP matmul2d path is in use");
                        }
                        // GROUP-64 matmul (2026-09-25, matmul_g64_msl.metal): the only kernel a
                        // group-64 weight has (G64_COUNT), so a failure here makes such a model
                        // refuse at its first matmul (`g64_encode`: "pso g64_prepare").
                        let g64 = include_str!("shaders/metal/matmul_g64_msl.metal");
                        // the group-64 PREFILL v2 (MPP: macOS 26) — optional; default when compiled (ARF_NO_G64_PF2 opts out)
                        let g64pf = include_str!("shaders/metal/matmul_g64_pf_msl.metal");
                        if let Err(e) = ["g64pf2_mm_r32_n64", "g64pf2_mm_r128_n64", "g64pf2_mm3_r32_n64", "g64pf2_mm3_r128_n64"].iter().try_for_each(|k| isl.compile_msl(ctx, k, g64pf, k)) {
                            eprintln!("[msl] group-64 prefill v2 not compiled ({e}) — prefill uses the lane kernel");
                        }
                        // another engine's sg4 prefill form (2026-09-27, matmul_g64_pfs_msl.metal): OPT-IN and
                        // NOT MEASURED, so compiled only under ARF_G64_PFS=1 — the default load pays
                        // nothing for it. A failure leaves prefill on g64pf2_mm. MEASURED-OUT since:
                        // correct, not faster (measured, "another engine's sg4 prefill matmul, ported").
                        if std::env::var("ARF_G64_PFS").is_ok_and(|v| v == "1") {
                            let pfs = include_str!("shaders/metal/matmul_g64_pfs_msl.metal");
                            match ["g64pfs_prepare", "g64pfs_mm"].iter().try_for_each(|k| isl.compile_msl(ctx, k, pfs, k)) {
                                Ok(()) => eprintln!("[msl] ARF_G64_PFS: the sg4 prefill matmul compiled — prefill windows take it ahead of g64pf2_mm"),
                                Err(e) => eprintln!("[msl] ⚠ ARF_G64_PFS set but the sg4 prefill matmul did NOT compile ({e}) — prefill stays on g64pf2_mm"),
                            }
                        }
                        // the other weight-prefetch depths (G64_PF 1 / 3; 2 is the default build) as
                        // separate pipelines, for the ARF_G64_PF A/B — optional
                        for pf in [1usize, 3] {
                            let src = format!("#define G64_PF {pf}\n{g64}");
                            for k in ["g64_mm", "g64_mm_gate_up", "g64_mm3", "g64_mm3_gate_up"] {
                                let _ = isl.compile_msl(ctx, &format!("{k}_pf{pf}"), &src, k);
                            }
                        }
                        match ["g64_prepare", "g64_mm", "g64_mm_gate_up", "g64_mm3", "g64_mm3_gate_up"].iter().try_for_each(|k| isl.compile_msl(ctx, k, g64, k)) {
                            Ok(()) if G64_COUNT.load(std::sync::atomic::Ordering::Relaxed) > 0 => eprintln!(
                                "[msl] group-64 matmul compiled — {} group-64 weights ({} at 3 bits)",
                                G64_COUNT.load(std::sync::atomic::Ordering::Relaxed),
                                G64_3BIT_COUNT.load(std::sync::atomic::Ordering::Relaxed)
                            ),
                            Ok(()) => {}
                            Err(e) => eprintln!("[msl] ⚠ group-64 matmul did NOT compile ({e}) — a group-64 model cannot run"),
                        }
                        match ok {
                            Ok(()) if std::env::var_os("ARF_NO_MPP_Q4").is_some() => eprintln!("[msl] MPP Q4 batched matmul compiled but DISABLED by ARF_NO_MPP_Q4"),
                            Ok(()) => eprintln!("[msl] MPP Q4 batched matmul ON — batches of 4..=16 rows, and prefill windows to 128, take Metal 4 matmul2d (no extra memory)"),
                            Err(_) => eprintln!("[msl] MPP Q4 batched matmul unavailable on this OS (needs macOS 26 / Metal 4); batched decode uses the GEMV"),
                        }
                    }
                    // Fast-path lever summary: reports what THIS process actually resolved
                    // (default-on + any env override), so "why is it fast" is answered by
                    // startup output instead of archaeology in the decisions log (pre-OSS, removed 2026-08-28). Sourced from the
                    // same booleans the compile above used — cannot drift from real behavior.
                    let batch_mega_on = std::env::var_os("ARF_NO_BATCH_MEGA").is_none();
                    let megakernel_on = std::env::var_os("ARF_NO_MEGAKERNEL").is_none();
                    let singleq_on = std::env::var_os("ARF_NO_MEGA_SINGLEQ").is_none();
                    let spec_on = std::env::var_os("ARF_MEGA_SPEC").is_some();
                    let spec_drafter = std::env::var("ARF_MEGA_SPEC_DRAFTER").unwrap_or_else(|_| "ngram".into());
                    eprintln!(
                        "[msl] fast-path levers: doty={} nsg2={} vec4={} megakernel={} singleq={} batch_mega={} spec={}{}",
                        if doty_on { "on" } else { "off" },
                        if nsg2_on { "on" } else { "off" },
                        if vec4_on { "on" } else { "off" },
                        if megakernel_on { "on" } else { "off" },
                        if singleq_on { "on" } else { "off" },
                        if batch_mega_on { "on" } else { "off" },
                        if spec_on { "on" } else { "off" },
                        if spec_on { format!(" (drafter={spec_drafter})") } else { String::new() },
                    );
                    Some(std::sync::Mutex::new(isl))
                }
                Err(e) => {
                    eprintln!("[msl] island compile failed ({e}); falling back to wgpu GEMV");
                    None
                }
            }
        })
    } else {
        None
    };

    // HARD-FAIL ON ISLAND FAILURE (2026-08-06) — macOS is a METAL-ONLY target.
    //
    // Previously a failed island silently degraded to the WGSL path. That is the single worst
    // failure mode this engine has: the daemon comes up, serves correct text, and runs ~10x
    // slower with NOTHING in the log to say why — the same "silently never engaged" class of bug
    // that hid the m=1 megakernel, spec-verify, KSTEP and the 2-flag spec gate (four separate
    // instances found on 2026-08-05/06 alone).
    //
    // With the portable WGSL layer retiring, a silent fallback is not even a degraded engine — it
    // is an unsupported configuration. Fail loudly at load with the reason instead.
    //
    // L350 — a DELIBERATE geometry decline is not a failure. The guard above exists because a
    // SILENT fallback hid four bugs; it does not apply when the engine chose the portable path
    // on purpose, said so on stderr, and is running correctly. Refusing to start would turn a
    // correct-but-slower run into no run at all, which is strictly worse for the one model this
    // affects (`head_dim < 128`, see the island gate). The decline is loud, so the failure mode
    // that guard protects against — no explanation in the log — cannot occur here.
    #[cfg(target_os = "macos")]
    if island.is_none() && std::env::var_os("ARF_MSL_GEMV").is_some() {
        return Err(ArfError::model_load(
            "native Metal island FAILED to initialize on macOS — Arf is Metal-only here and \
             will not silently fall back to the (retiring) WGSL path. Check the [msl] lines above \
             for the failing kernel compile. To run the legacy WGSL engine anyway, unset \
             ARF_MSL_GEMV (expect ~10x lower throughput).",
        ));
    }

    // WEIGHT CACHE: finalize the session (write the sidecar index/header on a build run) and
    // take the keep-alive mmap the no-copy weight buffers alias. Stored on the model so the
    // mapping outlives every MTLBuffer (dropping it early = use-after-free / GPU fault).
    #[cfg(target_os = "macos")]
    let weight_cache_keep = if _cache_session {
        crate::weight_cache::finish()
    } else {
        None
    };
    let result = Ok(GpuModel {
        ctx: ctx.clone(),
        embed_q4k,
        #[cfg(target_os = "macos")]
        weight_cache_keep,
        #[cfg(target_os = "macos")]
        island,
        #[cfg(target_os = "macos")]
        mega: crate::gpu::MegakernelBufs {
            rope_cos: rope_cos_mtl,
            rope_sin: rope_sin_mtl,
            rope_cos_local: rope_cos_local_mtl,
            rope_sin_local: rope_sin_local_mtl,
            rope_cos_global: rope_cos_global_mtl,
            rope_sin_global: rope_sin_global_mtl,
            final_norm: final_norm_mtl,
            embed_norm: embed_norm_mtl,
            lm_head: lm_head_q8_mtl,
            embed: embed_mtl,
        },
        embed,
        layers,
        mtp,
        embed_norm,
        final_norm,
        lm_head,
        rope_cos,
        rope_sin,
        rope_cos_local,
        rope_sin_local,
        rope_cos_global,
        rope_sin_global,
        scratch,
        kernels,
        kv,
        kv_levels,
        cfg: cfg.clone(),
        max_ctx: max_positions,
        decode_arenas,
        // Empty; the first forward_batch sizes it (lazy high-water ratchet).
        batch_arenas: std::cell::RefCell::new(crate::gpu::BatchArenas::new()),
        // Empty; the first steady decode step (token=None) fills it, reused after.
        decode_binds: std::cell::RefCell::new(Vec::new()),
        batch_binds: std::cell::RefCell::new(Vec::new()),
        batch_binds_shape: std::cell::Cell::new(None),
        decode_block_table: std::cell::RefCell::new(None),
        m1_burst_stash: std::cell::RefCell::new(Vec::new()),
        m1_spec: std::cell::RefCell::new(None),
        #[cfg(target_os = "macos")]
        mtp_h_src: std::cell::Cell::new(None),
        dflash_check_n: std::cell::Cell::new(0),
        dflash_check_bad: std::cell::Cell::new(0),
    });
    // RESIDENCY SET (faithful port of llama.cpp ggml-metal-device.m:1422 rset + the
    // ggml-metal-context.m:451 rsets_keep_alive). Build ONE MTLResidencySet over EVERY island
    // weight buffer — dense q/k/v/o (Q4ksMtl), the ~18GB of MoE expert buffers (MoeMtl gate/up/down
    // + router), the KV pool (keys_mtl/values_mtl), and lm_head/embed/final_norm/rope — and
    // requestResidency so they are GPU-resident ONCE at load. This eliminates the per-first-access
    // weight fault-in that was the 38-51s first-decode warmup (the bulk = the MoE expert buffers a
    // prior per-buffer warm-up missed). Runs once at load when the island is present; opt out with
    // ARF_NO_RESIDENCY_SET (debug/A-B). The set is stored on the island and kept alive there.
    #[cfg(target_os = "macos")]
    if let Ok(model) = result.as_ref() {
        if std::env::var_os("ARF_NO_RESIDENCY_SET").is_none() {
            if let Some(isl_mutex) = model.island.as_ref() {
                use objc2::runtime::ProtocolObject;
                use objc2_metal::MTLBuffer;
                let mut raw: Vec<&ProtocolObject<dyn MTLBuffer>> = Vec::new();
                // Helper: a GpuMatWeight's Q4ksMtl view (q/k/v/o/router), if present.
                fn q4ks_of(
                    m: &crate::gpu::GpuMatWeight,
                ) -> Option<std::sync::Arc<crate::gpu::concurrent_metal::Q4ksMtl>> {
                    match m {
                        crate::gpu::GpuMatWeight::Q4KS { mtl: Some(mtl), .. } => Some(mtl.clone()),
                        _ => None,
                    }
                }
                // Collect into owned Arcs first so the &refs we push outlive `raw` (the Arcs are
                // dropped only at the end of this block, after build_residency_set returns).
                let mut q4ks_keep: Vec<std::sync::Arc<crate::gpu::concurrent_metal::Q4ksMtl>> =
                    Vec::new();
                let mut moe_keep: Vec<std::sync::Arc<crate::gpu::concurrent_metal::MoeMtl>> =
                    Vec::new();
                // L214 — 3-bit FFN triples, kept alive so their &MTLBuffer refs outlive `raw`.
                type Q3Triple = (
                    crate::gpu::concurrent_metal::Q3ksMtl,
                    crate::gpu::concurrent_metal::Q3ksMtl,
                    crate::gpu::concurrent_metal::Q3ksMtl,
                );
                let mut q3k_keep: Vec<std::sync::Arc<Q3Triple>> = Vec::new();
                for ly in &model.layers {
                    for m in [&ly.q_proj, &ly.k_proj, &ly.v_proj, &ly.o_proj] {
                        if let Some(w) = q4ks_of(m) {
                            q4ks_keep.push(w);
                        }
                    }
                    match &ly.mlp {
                        crate::gpu::GpuMlp::Moe(moe) => {
                            if let Some(w) = q4ks_of(&moe.router) {
                                q4ks_keep.push(w);
                            }
                            for v in [&moe.gate_mtl, &moe.up_mtl, &moe.down_mtl]
                                .into_iter()
                                .flatten()
                            {
                                moe_keep.push(v.clone());
                            }
                        }
                        crate::gpu::GpuMlp::Dense(d) => {
                            // L214 — when a 3-bit FFN replaced this layer's gate/up/down, wire the
                            // Q3 buffers and NOT the Q4 ones they supersede. Getting this backwards
                            // is what made L213's run thrash: the residency set still committed the
                            // full 19.90 GB of Q4 weights (identical to the Q4-only baseline) while
                            // the Q3 buffers, which are the ones actually dispatched, were never
                            // added at all — so the superseded weights stayed wired and the live
                            // ones demand-paged. Swap hit 8.1 GB and 20 tokens took 4m36s.
                            match d.q3k.as_ref() {
                                Some(t) => {
                                    q3k_keep.push(t.clone());
                                }
                                None => {
                                    for m in [&d.gate_proj, &d.up_proj, &d.down_proj] {
                                        if let Some(w) = q4ks_of(m) {
                                            q4ks_keep.push(w);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(w) = q4ks_of(&model.lm_head) {
                    q4ks_keep.push(w);
                }
                // L243 — the MTP draft head's weights. Without this they are loaded but NOT
                // resident, so the first draft would fault them in mid-decode — the exact
                // first-use stall the residency set exists to remove (L153).
                if let Some(m) = model.mtp.as_ref() {
                    for w in [
                        &m.q_proj,
                        &m.k_proj,
                        &m.v_proj,
                        &m.o_proj,
                        &m.gate_proj,
                        &m.up_proj,
                        &m.down_proj,
                        &m.eh_proj,
                    ] {
                        if let Some(q) = q4ks_of(w) {
                            q4ks_keep.push(q);
                        }
                    }
                }
                // Now push raw &MTLBuffer refs from the kept Arcs (codes/scales/mins/dd each).
                for w in &q4ks_keep {
                    raw.push(&w.codes);
                    raw.push(&w.scales);
                    raw.push(&w.mins);
                    raw.push(&w.dd);
                    // a group-64 weight's real buffers (the four above are one placeholder):
                    // without these the set held 10.5 of 15 GB (2026-09-25)
                    if let Some(g) = &w.g64 {
                        raw.push(&g.w);
                        raw.push(&g.s);
                        raw.push(&g.b);
                    }
                }
                for t in &q3k_keep {
                    for w in [&t.0, &t.1, &t.2] {
                        raw.push(&w.ql);
                        raw.push(&w.qh);
                        raw.push(&w.scales);
                        raw.push(&w.mins);
                        raw.push(&w.dd);
                    }
                }
                for v in &moe_keep {
                    raw.push(&v.codes);
                    raw.push(&v.scales);
                    raw.push(&v.mins);
                    raw.push(&v.dd);
                }
                // Per-layer norm MTL views (input/post/pre_ffn/post_ffn/q/k norm).
                for ly in &model.layers {
                    for n in [
                        &ly.norm_mtls.input_norm,
                        &ly.norm_mtls.post_norm,
                        &ly.norm_mtls.pre_ffn_norm,
                        &ly.norm_mtls.post_ffn_norm,
                        &ly.norm_mtls.q_norm,
                        &ly.norm_mtls.k_norm,
                    ]
                    .into_iter()
                    .flatten()
                    {
                        raw.push(&n.0);
                    }
                }
                // KV pool (keys_mtl/values_mtl per layer).
                //
                // MEASURED-OUT 2026-09-20 — "leave the pool out of the set so it grows with use".
                // Idle phys_footprint at --max-context 32768: 8,548 MB without the pool in the
                // set, 8,541 MB with it. No difference, because Metal makes a buffer resident as
                // a WHOLE ALLOCATION the first time any command references it (the warm-up request
                // does), not page by page: 6,365 MB at 1,024 blocks, 8,541 at 2,048, ~13 GB at
                // 4,096 — exactly the pool's size each time. One big buffer per layer can never
                // be lazy. Adaptive context needs the pool GROWN (realloc + copy as block ids
                // climb), which means interior mutability on every `kv.keys[li]` consumer.
                // DO NOT RE-CHASE the residency set for this.
                for kb in model.kv.keys_mtl.iter().flatten() {
                    raw.push(&kb.0);
                }
                for vb in model.kv.values_mtl.iter().flatten() {
                    raw.push(&vb.0);
                }
                for set in model.kv.q8.iter().flatten() {
                    for b in set {
                        raw.push(&b.0);
                    }
                }
                // Model-level: lm_head, embed, final_norm, rope tables. BOTH lm_head shapes must
                // be registered — the residency set is what keeps these buffers off the eviction
                // path, and a native-Q4_K lm_head is exactly as resident as a Q8 one.
                if let Some(lm) = &model.mega.lm_head {
                    match &**lm {
                        crate::gpu::concurrent_metal::IslandLmHead::Q8(q8) => {
                            raw.push(&q8.data);
                            raw.push(&q8.scale);
                            raw.push(&q8.dims);
                        }
                        crate::gpu::concurrent_metal::IslandLmHead::Q4ks(q) => {
                            raw.push(&q.codes);
                            raw.push(&q.scales);
                            raw.push(&q.mins);
                            raw.push(&q.dd);
                            raw.push(&q.dims);
                            if let Some(g) = &q.g64 {
                                raw.push(&g.w);
                                raw.push(&g.s);
                                raw.push(&g.b);
                            }
                        }
                    }
                }
                for b in [
                    &model.mega.embed,
                    &model.mega.final_norm,
                    &model.mega.rope_cos,
                    &model.mega.rope_sin,
                    &model.mega.rope_cos_local,
                    &model.mega.rope_sin_local,
                    &model.mega.rope_cos_global,
                    &model.mega.rope_sin_global,
                ]
                .into_iter()
                .flatten()
                {
                    raw.push(&b.0);
                }
                match isl_mutex.lock() {
                    Ok(mut isl) => {
                        // ARF_WEIGHT_HASH_DUMP=<file>: one line per resident buffer (index, length,
                        // FNV-1a of its bytes) — diff a weight-cache HIT load against
                        // ARF_NO_WEIGHT_CACHE=1 to name the buffers the two paths disagree on.
                        if let Some(path) = std::env::var_os("ARF_WEIGHT_HASH_DUMP") {
                            let mut out = String::new();
                            for (i, b) in raw.iter().enumerate() {
                                let len = b.length();
                                // private storage (the sparse KV pool) has no CPU view at all
                                let private = objc2_metal::MTLResource::storageMode(*b)
                                    == objc2_metal::MTLStorageMode::Private;
                                let p = if private {
                                    std::ptr::null()
                                } else {
                                    b.contents().as_ptr() as *const u8
                                };
                                // hashes of the first 1,024 / 20,480 / 100,352 bytes and of the
                                // whole buffer: cache regions are page-padded, so two loads of the
                                // same tensor differ in LENGTH and only a prefix compares.
                                let mut hs = [0xcbf29ce484222325u64; 4];
                                let cuts = [1024usize, 20480, 100352, usize::MAX];
                                if !p.is_null() {
                                    let bytes = unsafe { std::slice::from_raw_parts(p, len) };
                                    for (j, &x) in bytes.iter().enumerate() {
                                        for (k, h) in hs.iter_mut().enumerate() {
                                            if j < cuts[k] {
                                                *h = (*h ^ x as u64).wrapping_mul(0x100000001b3);
                                            }
                                        }
                                    }
                                }
                                let label = objc2_metal::MTLResource::label(*b)
                                    .map(|l| l.to_string())
                                    .unwrap_or_default();
                                out.push_str(&format!(
                                    "{i} {len} {:016x} {:016x} {:016x} {:016x} {label}\n",
                                    hs[0], hs[1], hs[2], hs[3]
                                ));
                            }
                            let _ = std::fs::write(&path, out);
                            eprintln!(
                                "[weight-hash] {} buffers -> {}",
                                raw.len(),
                                path.to_string_lossy()
                            );
                        }
                        match isl.build_residency_set(ctx, &raw) {
                            Ok(n) => eprintln!(
                                "[msl] MTLResidencySet committed + requestResidency: {n} weight buffers resident (no first-decode fault-in)"
                            ),
                            Err(e) => eprintln!("[msl] residency set build failed ({e}); falling back to per-encode useResource"),
                        }
                        // Warm every island PSO's first-run JIT at load → no 40-50s first-decode warmup.
                        isl.warm_pipelines(ctx);
                    }
                    Err(_) => eprintln!("[msl] residency set: island mutex poisoned; skipping"),
                }
            }
        }
    }

    // Diagnostic: report how many dense tensors took the lossless native Q4_K path
    // vs the fallback bf16 self-quant path . Helps confirm the native path
    // engaged when loading a GGUF Q4_K checkpoint.
    if matches!(quant, Quant::Q4K | Quant::Q4KS) {
        let native = native_q4k_count.get();
        let fallback = fallback_count.get();
        eprintln!(
            "[arf] dense weight loader: {native} native Q4_K (lossless), {fallback} fallback (bf16 self-quant)"
        );
    }
    result
}

/// Load real weights from safetensors directly into a resident
/// [`GpuModel`].
pub fn load_safetensors_gpu<P: AsRef<Path>>(
    cfg: &ModelConfig,
    paths: &[P],
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
) -> Result<crate::gpu::GpuModel> {
    build_gpu(cfg, &read_weights_lazy(paths)?, max_positions, ctx)
}

/// Like [`load_safetensors_gpu`] but quantizes the resident matmul weights to
/// `quant` (`Quant::Int8` for the int8 decode path).
pub fn load_safetensors_gpu_quant<P: AsRef<Path>>(
    cfg: &ModelConfig,
    paths: &[P],
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
    quant: Quant,
) -> Result<crate::gpu::GpuModel> {
    build_gpu_quant(cfg, &read_weights_lazy(paths)?, max_positions, ctx, quant)
}

/// Load a single-file GGUF directly into a resident [`GpuModel`], quantizing the
/// matmul weights to `quant`. The GGUF is mmap'd and dequantized lazily one tensor
/// at a time (see [`arf_core::model::gguf::LazyGguf`]) — the only way the 30B fits —
/// then each tensor is re-quantized and uploaded. `Quant::Q4K` keeps the GGUF's
/// disk accuracy class; `Quant::Q4` is smaller but lossier. The embedded tokenizer
/// is NOT parsed; supply a `tokenizer.json` or drive decode with raw token ids.
pub fn load_gguf_gpu_quant(
    cfg: &ModelConfig,
    path: &Path,
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
    quant: Quant,
) -> Result<crate::gpu::GpuModel> {
    let w = arf_core::model::gguf::load_gguf(path, cfg)?;
    #[cfg(target_os = "macos")]
    crate::weight_cache::set_gguf_path(path);
    build_gpu_quant(cfg, &w, max_positions, ctx, quant)
}

/// Like [`load_safetensors_gpu_quant`] but also selects the KV-cache quantization
/// (`KvQuant::Tq { bits, .. }` builds the TurboQuant-packed pool).
pub fn load_safetensors_gpu_kv_quant<P: AsRef<Path>>(
    cfg: &ModelConfig,
    paths: &[P],
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
    quant: Quant,
    kv_quant: KvQuant,
) -> Result<crate::gpu::GpuModel> {
    build_gpu_kv_quant(
        cfg,
        &read_weights_lazy(paths)?,
        max_positions,
        ctx,
        quant,
        kv_quant,
    )
}

/// Like [`load_gguf_gpu_quant`] but also selects the KV-cache quantization.
pub fn load_gguf_gpu_kv_quant(
    cfg: &ModelConfig,
    path: &Path,
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
    quant: Quant,
    kv_quant: KvQuant,
) -> Result<crate::gpu::GpuModel> {
    let w = arf_core::model::gguf::load_gguf(path, cfg)?;
    let quant = promote_gguf_quant(&w, quant);
    #[cfg(target_os = "macos")]
    crate::weight_cache::set_gguf_path(path);
    build_gpu_kv_quant(cfg, &w, max_positions, ctx, quant, kv_quant)
}

/// Print this process's current resident size at a load milestone, but only when
/// `ARF_LOAD_TRACE` is set. Used to pinpoint which tensor/layer the load OOMs on
/// (the macOS jetsam kill is silent, so a printed trail is the only post-mortem).
/// Reads RSS via `ps` — heavy, but it runs at most ~7 times per load and only when asked.
fn load_trace(stage: &str) {
    if std::env::var("ARF_LOAD_TRACE").is_err() {
        return;
    }
    let pid = std::process::id();
    let rss_mb = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0);
    eprintln!("[load] {stage}: rss={rss_mb:.0} MB");
}

/// Default a GGUF load to its OWN native block format instead of bf16-expanding it.
///
/// The old `Quant::None` default ran every weight through `dequant → f32 → bf16`, blowing
/// a 6.5 GB Q4_0 GGUF up to ~24 GB resident (+ transient f32 spikes) and OOM-killing a
/// 36 GB Mac at LOAD. A quantized GGUF should load the way ggml/ollama read it — via the
/// lossless native transcode (≈ on-disk footprint). When the user passes an explicit
/// `--quant`, we honor it; we only promote the unset default (`None`).
fn promote_gguf_quant(w: &arf_core::model::weights::Weights, requested: Quant) -> Quant {
    if !matches!(requested, Quant::None) {
        return requested; // explicit flag wins
    }
    match w.native_quant_hint() {
        Some(native) => {
            eprintln!(
                "quant: defaulting GGUF to native {native:?} (lossless block transcode; \
                 pass --quant none to force bf16 expansion)"
            );
            native
        }
        None => requested,
    }
}

/// Like [`load_safetensors_gpu_kv_quant`] but with an explicit KV pool block count
/// for the serving path (must equal the scheduler's `EngineConfig::num_blocks`).
pub fn load_safetensors_gpu_kv_quant_pooled<P: AsRef<Path>>(
    cfg: &ModelConfig,
    paths: &[P],
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
    quant: Quant,
    kv_quant: KvQuant,
    kv_blocks: Option<usize>,
) -> Result<crate::gpu::GpuModel> {
    build_gpu_kv_quant_pooled(
        cfg,
        &read_weights_lazy(paths)?,
        max_positions,
        ctx,
        quant,
        kv_quant,
        kv_blocks,
    )
}

/// Like [`load_gguf_gpu_kv_quant`] but with an explicit KV pool block count for the
/// serving path (must equal the scheduler's `EngineConfig::num_blocks`).
pub fn load_gguf_gpu_kv_quant_pooled(
    cfg: &ModelConfig,
    path: &Path,
    max_positions: usize,
    ctx: &std::sync::Arc<crate::gpu::GpuContext>,
    quant: Quant,
    kv_quant: KvQuant,
    kv_blocks: Option<usize>,
) -> Result<crate::gpu::GpuModel> {
    let w = arf_core::model::gguf::load_gguf(path, cfg)?;
    let quant = promote_gguf_quant(&w, quant);
    #[cfg(target_os = "macos")]
    crate::weight_cache::set_gguf_path(path);
    build_gpu_kv_quant_pooled(cfg, &w, max_positions, ctx, quant, kv_quant, kv_blocks)
}

#[cfg(test)]
mod l147_kv_pool_sizing_tests {
    //! L146/L147 — the KV pool must be counted in the wired-memory guard.
    //!
    //! `wired_memory_guard` priced ONLY the weights while its doc claimed the 80%-of-RAM
    //! threshold "leaves headroom for KV". On qwen3-coder-30B the pool is 6.4 GB at 2048 blocks
    //! and 12.9 GB at 4096 — the second-largest allocation in the process. Over-committing it
    //! evicts the file-backed weight pages, which read back ZEROED, and the engine then emits
    //! `!!!!` at full speed with no error.
    //!
    //! Measured on a 36 GB M4 Max (cap 28.8 GB): blocks 2048 -> 24.8 GB CLEAN,
    //! 3072 -> 28.1 GB CLEAN, 4096 -> 31.3 GB GARBAGE. A threshold crossing.
    //!
    //! These are pure arithmetic — milliseconds, no GPU, no GGUF. Every other gate in this repo
    //! varies BATCH size; none varied POOL size, which is why the bug survived.

    /// KV pool bytes: K and V, per layer, over all slots, f32.
    fn kv_pool_bytes(blocks: usize, layers: usize, kv_heads: usize, head_dim: usize) -> usize {
        const BLOCK: usize = 16;
        2 * layers * (blocks * BLOCK) * (kv_heads * head_dim) * 4
    }

    // qwen3-coder-30B geometry.
    const LAYERS: usize = 48;
    const KVH: usize = 4;
    const HD: usize = 128;

    #[test]
    fn kv_pool_sizes_match_the_measured_footprints() {
        assert_eq!(
            kv_pool_bytes(2048, LAYERS, KVH, HD),
            6_442_450_944,
            "2048 blocks = 6.44 GB"
        );
        assert_eq!(
            kv_pool_bytes(4096, LAYERS, KVH, HD),
            12_884_901_888,
            "4096 blocks = 12.88 GB"
        );
        // Linear in blocks — doubling the pool doubles the bytes.
        assert_eq!(
            kv_pool_bytes(4096, LAYERS, KVH, HD),
            2 * kv_pool_bytes(2048, LAYERS, KVH, HD)
        );
    }

    #[test]
    fn the_guard_must_refuse_4096_and_admit_2048_on_a_36gb_box() {
        // Weights for the q4ks checkpoint, as the guard estimates them.
        let weights = 17.3_f64 * 2f64.powi(30);
        let cap = 36.0_f64 * 2f64.powi(30) * 0.80;
        let total = |blocks: usize| weights + kv_pool_bytes(blocks, LAYERS, KVH, HD) as f64;

        assert!(total(2048) <= cap, "blocks 2048 must FIT (measured clean)");
        assert!(
            total(3072) <= cap,
            "blocks 3072 must FIT (measured clean, 28.08 GB banner)"
        );
        assert!(
            total(4096) > cap,
            "blocks 4096 must be REFUSED (measured '!!!!', 31.30 GB banner)"
        );
    }

    #[test]
    fn f16_pool_is_additive_not_a_replacement() {
        // L135: ARF_KV_F16 allocates a SECOND, half-size pool ALONGSIDE the f32 one, so the
        // guard multiplies by 1.5 rather than swapping the term.
        let f32_only = kv_pool_bytes(2048, LAYERS, KVH, HD) as f64;
        let with_f16 = f32_only * 1.5;
        assert_eq!(
            with_f16 - f32_only,
            f32_only / 2.0,
            "the f16 pool is half the f32 pool"
        );
    }
}
