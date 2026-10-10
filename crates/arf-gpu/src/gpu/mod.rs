//! The GPU backend — the whole forward pass runs on the device.
//!
//! Weights load resident on the GPU once; from there embed, the norms, attention
//! (paged-KV, GQA, batched), the MLP/MoE experts, RoPE, and sampling all run in
//! WGSL kernels — the CPU only schedules. On macOS a native-Metal "island" fuses a
//! whole decode token into a handful of dispatches (the fast path); everywhere else
//! the portable WGSL path runs the same math, kernel for kernel.
//!
//! Cross-platform via wgpu (Vulkan / Metal / DX12). Requires a GPU adapter at
//! runtime; [`GpuContext::new`] returns an error when none is present.

use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::arena::{arena_bytes, GpuArena, Region, ALIGN};
use arf_core::error::{ArfError, Result};

/// Output-tile width of the matmul kernel; must match `TILE` in `matmul.wgsl`.
/// Used CPU-side to compute the dispatch grid.
const TILE: u32 = 16;

/// The tiled matmul compute shader (see `src/shaders/wgsl/matmul.wgsl`).
const WGSL: &str = include_str!("../shaders/wgsl/matmul.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

/// Drafter backend for the m4 self-spec seam (`ARF_MEGA_SPEC_DRAFTER`).
/// Wraps either drafter behind the SAME `observe`/`propose` shape so the
/// batch.rs call site never changes — only which drafter answers it.
/// Default is `Ngram` (unchanged behavior); `suffix` opts into the
/// SuffixDecoding-class drafter (confidence-gated, never net-negative: an
/// empty draft is free — see [`arf_core::model::speculative::SuffixDrafter`]).
pub(crate) enum Drafter {
    Ngram(arf_core::model::speculative::NgramDrafter),
    Suffix(arf_core::model::speculative::SuffixDrafter),
}

impl Drafter {
    pub fn new(order: usize) -> Self {
        let backend = std::env::var("ARF_MEGA_SPEC_DRAFTER").unwrap_or_default();
        if backend.eq_ignore_ascii_case("suffix") {
            Drafter::Suffix(arf_core::model::speculative::SuffixDrafter::new())
        } else {
            Drafter::Ngram(arf_core::model::speculative::NgramDrafter::new(order))
        }
    }

    pub fn observe(&mut self, history: &[u32], tok: u32) {
        match self {
            Drafter::Ngram(d) => d.observe(history, tok),
            Drafter::Suffix(d) => d.observe(history, tok),
        }
    }

    pub fn propose(&self, history: &[u32], k: usize) -> Vec<u32> {
        match self {
            Drafter::Ngram(d) => d.propose(history, k),
            Drafter::Suffix(d) => d.propose(history, k),
        }
    }
}

/// Self-spec state for the B=1 chat bridge (see `GpuModel::m1_spec`).
pub(crate) struct M1Spec {
    /// Position of `history[0]`: `history[i]` = the token fed at position `base + i`.
    /// Non-zero after a mid-stream re-seed (prefix-cache hit or a gap while the seq
    /// ran inside a multi-seq batch) — n-gram keys don't care about absolute positions,
    /// so a window is as draftable as a full history.
    pub base: usize,
    /// Tokens observed at positions `[base, base + history.len())` (prompt + generated).
    pub history: Vec<u32>,
    /// Prompt-lookahead drafter; keys built incrementally from `history`. Backend
    /// selected once at construction by `ARF_MEGA_SPEC_DRAFTER` (default: ngram).
    pub drafter: Drafter,
    /// History indices `[.., observed)` have been keyed into the drafter.
    pub observed: usize,
}

pub struct GpuModel {
    pub ctx: Arc<GpuContext>,
    pub embed: wgpu::Buffer, // [vocab, hidden]
    /// `embed` holds the GGUF's own Q4_K blocks (144 B per 256), not bf16 — every gather must
    /// take the `*_q4k` kernel (2026-09-23; `ARF_NO_EMBED_Q4K=1` keeps bf16).
    pub embed_q4k: bool,
    pub layers: Vec<GpuLayer>,
    /// L241 — the trained MTP draft head (`blk.64`), when the arch ships one. `None` for every
    /// model without `nextn_predict_layers`, so nothing else changes shape.
    pub mtp: Option<std::sync::Arc<GpuMtpHead>>,
    pub final_norm: wgpu::Buffer, // [hidden]
    /// Muse Glimmer's PRE-TRUNK embedding norm: a WEIGHTLESS RMSNorm applied to the raw
    /// embedding output before layer 0 runs. llama.cpp's op order (verified against a
    /// `llama-eval-callback` tensor dump) is
    ///     embd      = get_rows(token_embd, tokens)
    ///     embd_norm = rms_norm(embd)                <- THIS
    ///     norm-0    = rms_norm(embd_norm)           <- then the layer norms again
    /// so the trunk sees a doubly-normalized activation. Skipping it feeds every one of
    /// the 52 layers input at the wrong scale — the machinery stays coherent and the
    /// output distribution is garbage, which is exactly how it presented.
    ///
    /// Stored as an all-ONES `[hidden]` weight so the standard `rmsnorm` kernel
    /// (`y = x/rms(x) * weight`) reproduces `rms_norm(x)` exactly: no new shader, no new
    /// kernel, and the Metal island gets its dual view through the same `shared_norm`
    /// path as every other norm. `None` for every other architecture, and the dispatch
    /// is skipped entirely when it is `None` — zero cost off Muse Glimmer.
    pub embed_norm: Option<wgpu::Buffer>,
    pub lm_head: GpuMatWeight,  // [vocab, hidden]
    pub rope_cos: wgpu::Buffer, // [max_pos, head_dim/2]
    pub rope_sin: wgpu::Buffer, // [max_pos, head_dim/2]
    /// Gemma's local RoPE base (small `rope_local_base_freq`), used by the
    /// sliding-window layers. `None` for single-base models (Llama/Qwen).
    pub rope_cos_local: Option<wgpu::Buffer>,
    pub rope_sin_local: Option<wgpu::Buffer>,
    /// Gemma 4's GLOBAL-layer RoPE table: built for the global `head_dim` (512) with
    /// PARTIAL rotary (`rotary_dim` 128 of 512) over the global θ (1M), so its row
    /// stride is `rotary_dim/2` (64) — different from `rope_cos`/`sin` (which is the
    /// SLIDING-layer/full-rotary table). `None` unless the model has per-layer-varying
    /// geometry (Gemma 4); Gemma 3's global layers share `rope_cos`/`sin` (full rotary).
    /// Global layers select this when present; otherwise they fall back to `rope_cos`.
    pub rope_cos_global: Option<wgpu::Buffer>,
    pub rope_sin_global: Option<wgpu::Buffer>,
    pub scratch: GpuScratch,
    pub kernels: GpuKernels,
    pub kv: GpuKvPool,
    /// Native-Metal island (macOS + ARF_MSL_GEMV): hosts the hand-MSL decode kernels
    /// (Q4_K_S GEMV at 86-97% roofline vs WGSL's ~40% in-frame). `None` everywhere else —
    /// the portable wgpu path is the default + the parity oracle. Built at load when the
    /// weights carry `Q4ksMtl` views. Interior-mutable for compiling pipelines lazily.
    #[cfg(target_os = "macos")]
    pub island: Option<std::sync::Mutex<concurrent_metal::MetalIsland>>,
    /// Megakernel S1c: raw-Metal views of the model-level buffers the whole-token island
    /// path reads (rope tables, final_norm). `None` views off-island. Per-layer norm/weight
    /// views live on GpuLayer; KV-pool views on GpuKvPool.
    #[cfg(target_os = "macos")]
    pub mega: MegakernelBufs,
    /// TurboQuant codebook reconstruction levels (`2^bits` f32), uploaded once at
    /// load; `None` unless the KV pool is `KvQuant::Tq`. Bound by the `*_tq`
    /// kernels.
    pub kv_levels: Option<wgpu::Buffer>,
    pub cfg: arf_core::config::ModelConfig,
    /// Max context (in tokens) the KV pool and decode arenas were sized for at
    /// load (`max_positions`). The streamed-softmax attention kernel itself has
    /// no fixed ctx cap; this is the real decode bound, replacing the old
    /// hardcoded 2048 in the single-stream entry points.
    pub max_ctx: usize,
    pub decode_arenas: std::cell::RefCell<DecodeArenas>,
    /// Reused arena cache for the batched forward (see [`BatchArenas`]).
    pub batch_arenas: std::cell::RefCell<BatchArenas>,
    /// P15: persistent bind groups for `decode_token`'s ctx_len-INVARIANT
    /// dispatches, indexed by a stable per-call-site slot (filled lazily on the
    /// first matching decode step, reused after). Saves ~250 create_bind_group/
    /// token. Only used in the steady decode shape (token=None, want_logits);
    /// prefill/`forward_logits` rebuild (different dispatch shape) and pass a
    /// `None` cache so their bind groups are never confused with decode's slots.
    /// Each cached entry's resources are byte-identical across tokens: resident
    /// scratch/weights + deterministic `uni`/`out`/`idx`(pos) offsets (the
    /// allocator repeats offsets for a fixed alloc sequence; the pos_buf reorder
    /// pins the one idx region a cached site captures). The contents at those
    /// offsets are refreshed by `alloc_write` each token — a bind group references
    /// the buffer, not a snapshot.
    pub decode_binds: std::cell::RefCell<Vec<wgpu::BindGroup>>,
    /// The `forward_batch` analog of [`Self::decode_binds`]: the steady single-sequence
    /// q_len==1 SERVE decode shape rebuilds the whole ~15·num_layers-dispatch graph every
    /// token, and building those bind groups uncached (one `create_bind_group` →
    /// IOGPU resource alloc each) was the ~100× chat slowdown on big models (gemma-12B
    /// 0.28 tok/s — the GPU math is only ~120ms/step; the rest was bind-group churn).
    /// Cached only for the byte-identical-across-tokens sites (resident weights + fixed
    /// arena/uniform offsets — the matmuls and norms); the ctx_len-growing attention /
    /// kv-scatter sites are NOT cached (their arena offsets shift as context grows), and
    /// any non-steady shape (prefill, multi-seq, batch resize) clears + rebuilds. Same
    /// selective-cache discipline as decode.rs.
    pub batch_binds: std::cell::RefCell<Vec<wgpu::BindGroup>>,
    /// The (nseq, q_len, total) shape the `batch_binds` cache was populated for. When the
    /// incoming batch shape differs, the cached bind groups capture stale buffer offsets,
    /// so the cache is cleared and rebuilt. `None` = empty/never populated.
    pub batch_binds_shape: std::cell::Cell<Option<(usize, usize, usize)>>,
    /// Per-sequence block table for `decode_token`'s slot mapping during batched
    /// prefill. `None` = identity (`slots[p]=p`, the single-stream default). `Some(tbl)`
    /// = the sequence owns the disjoint block range `tbl`, so `slots[p] =
    /// tbl[p/bs]*bs + p%bs`. `generate_batch` sets this per sequence to prefill
    /// Gemma/MoE through the correct single-stream decode path (forward_batch is the
    /// Llama/Qwen-shaped path and diverges for Gemma's four-norm prefill).
    pub(crate) decode_block_table: std::cell::RefCell<Option<Vec<u32>>>,
    /// K-TOKEN BURST STASH (the B=1 chat bridge): tokens decoded ahead by
    /// `decode_tokens_k` (one island buffer, one fence per k tokens), as
    /// `(position, expected_feed, token)`. `try_m1_megakernel` serves later scheduler
    /// steps from here with zero GPU work — a serve requires BOTH the position and the
    /// fed token to match (the greedy feedback loop guarantees the match within one
    /// stream; the token check plus the prefill/rewind clears in `forward_batch_impl` /
    /// `m1_spec_observe` stop stale lookahead from crossing a stream boundary, where a
    /// position-only hit would emit an old-timeline token AND skip that position's KV
    /// write). The n-gram self-spec branch (milestone 4) stashes its accepted run here too.
    pub(crate) m1_burst_stash: std::cell::RefCell<Vec<(usize, u32, u32)>>,
    /// N-GRAM SELF-SPEC state (ARF_MEGA_SPEC, the B=1 chat bridge): the single
    /// chat seq's running token history (`history[p]` = the token fed at position p,
    /// prefill AND decode, observed in `forward_batch_impl`) plus the incremental
    /// drafter and its observed-upto watermark. `None` until the first observed batch;
    /// reset whenever the scheduler rewinds past the watermark (edited turn / new chat).
    pub(crate) m1_spec: std::cell::RefCell<Option<M1Spec>>,
    /// Where the last batched record left the trunk's `output_norm`'d rows — the MTP head's
    /// `h_t` input. Published by `try_batched_megakernel`, cleared at every `step`; `None`
    /// declines the draft rather than conditioning it on a stale bank.
    #[cfg(target_os = "macos")]
    pub(crate) mtp_h_src: std::cell::Cell<Option<MtpHiddenSrc>>,
    /// `ARF_DFLASH_GPU_SELECT_CHECK` — windows compared and windows where the GPU selector
    /// disagreed with the CPU one (`verify_window_gpu_draft`).
    pub(crate) dflash_check_n: std::cell::Cell<usize>,
    pub(crate) dflash_check_bad: std::cell::Cell<usize>,
    /// WEIGHT CACHE keep-alive (macOS island path). Owns the file-backed mmap whose pages
    /// the no-copy weight `MTLBuffer`s alias (transcode-cache zero-copy load). MUST live as
    /// long as the model — dropping it while a buffer is live unmaps those pages =
    /// use-after-free / GPU fault. `None` when the cache is off or no buffers were wrapped
    /// no-copy. See [`crate::weight_cache`]. Never read — it exists purely to keep the
    /// mapping alive (the buffers alias its pages); `#[allow(dead_code)]` documents that.
    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    pub(crate) weight_cache_keep: Option<crate::weight_cache::CacheKeep>,
}

impl std::fmt::Debug for GpuModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuModel({} layers on {})",
            self.layers.len(),
            self.ctx.info
        )
    }
}

// The big `impl GpuModel` and the supporting types are split across these
// submodules (decode/batch/generate are `impl GpuModel` continuations;
// context/command/types/dispatch hold the device, dispatch machinery, model
// types, and shared forward-path helpers respectively).
mod batch;

/// Rows in the recurrent (GDN) state bank — `ARF_GDN_ROWS`, default 17 (16 sequences
/// plus the reserved row 0; was 8 until 2026-09-19). Public so a server can
/// size its scheduler to it; see [`crate::recurrent_stream_capacity`].
#[cfg(target_os = "macos")]
pub fn recurrent_bank_rows() -> usize {
    batch::gdn_bank_rows()
}
mod command;
pub mod metal;
// L364b — the island moved to `gpu/metal/island.rs` (and its Linux twin to `metal/shim.rs`).
// These aliases keep every existing `concurrent_metal::` path resolving, on both platforms, so
// the move stayed contiguous instead of rewriting 200+ references.
#[cfg(target_os = "macos")]
pub use metal::island as concurrent_metal;
// L364 — off macOS the island does not exist, but the struct fields that hold its handle types
// still must typecheck (they are `Option<..>` and always `None` there). `metal_shim` supplies
// uninhabited stand-ins under the SAME path, so `concurrent_metal::MtlBuf` resolves on Linux
// without gating ~88 field declarations. See the module header for why they are uninhabited.
#[cfg(not(target_os = "macos"))]
pub use metal::shim as concurrent_metal;
mod context;
mod decode;
mod dispatch;
mod generate;
mod moe_contract;
// L263 — SELFTEST/PARITY MACHINERY, behind a feature.
//
// 4,649 lines of parity oracles, probes and bisect helpers. They are genuinely valuable — they
// are how every fast path in this engine was proven against a reference — but they are not
// something a downstream user should link into a release binary, and previously they were gated
// only on the OS, so they compiled into every build (13% of this crate).
//
// The feature is ON by default so `cargo build`, the examples and the parity tests keep working
// with no ceremony; a lean consumer builds with `--no-default-features`.

pub mod flux;
pub mod profile;
// RESTORED 2026-08-17. Deleted on 2026-08-06 ("drop Qwen3.5-SSM, 4,486 lines") during the
// WGSL-removal sweep. That sweep was aimed at the portable-WGSL layer, but this engine drives
// the METAL ISLAND — 61 island/compile_msl references — so removing it took hybrid-SSM support
// out along with the WGSL it was only partly using.
//
// It already contains everything the qwen3.8 port had to rediscover: load_qwen35,
// read_hparams ground-truthed against a real Qwen3.6-27B GGUF, is_recurrent(il,
// full_attention_interval), linear_geom, and the fact that the final n_layer_nextn block is the
// MTP head rather than a trunk layer .
//
// Do NOT put an attribute on the line above this comment block. A `#[cfg(feature = "profiling")]`
// sat there and applied to THIS module, not to the `pub mod profile;` it was written under — an
// attribute binds to the next ITEM and comments do not interrupt that. `profiling` is not a default
// feature, so the restore above silently un-did itself on every ordinary build, and the four qwen35
// examples that import this module could not compile. That is a CI gate, not a nuisance: the
// workflow runs `cargo clippy --workspace --all-targets` and `cargo test --workspace`, and
// --all-targets reaches examples. The cfg had no valid target either way, since
// `profile::record_step` is called unconditionally from `command.rs`.
pub mod ssm_qwen35;
mod types;
pub mod vision;

pub use batch::BatchPending; // depth-2 pipeline handle: serve_loop_bench holds it in-flight.
#[cfg(target_os = "macos")]
pub(crate) use batch::PREFILL_LOGITS_NEEDED; // the actor's prefill hint (backend.rs)
pub use command::*;
pub use context::*;
pub use types::*;
// dispatch holds only crate-internal forward-path helpers (no public items).
pub(crate) use dispatch::*;
// moe_contract holds the MoE binding-order role tables (the GPU-less contract guard);
// `use` (not re-export) so the sibling decode/batch modules see them via `use super::*`.
#[allow(unused_imports)]
use moe_contract::*;

/// A weight matrix resident on the GPU, with the context needed to use it.
#[derive(Debug, Clone)]
pub struct GpuWeight {
    ctx: Arc<GpuContext>,
    buffer: wgpu::Buffer,
    n: usize,
    k: usize,
}

impl GpuWeight {
    /// `x [m, k]` (flat) → `[m, n]` (flat), computed on the GPU.
    pub fn linear(&self, x: &[f32], m: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), m * self.k);
        self.ctx.matmul_nt(x, m, self.k, &self.buffer, self.n)
    }

    pub fn out_dim(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::WGSL;

    /// The compute shaders arf embeds at build time via `include_str!` (which
    /// requires literal paths, so this list is explicit). Validated on the CPU with
    /// naga — catches WGSL syntax and type errors without a GPU adapter (CI has none).
    ///
    /// This list is NOT the source of truth for coverage: `all_wgsl_parses_and_validates`
    /// globs `src/shaders/*.wgsl` so EVERY shader is validated regardless of whether it
    /// appears here, and `all_listed_shaders_exist_on_disk` asserts no entry here is
    /// stale. So a new kernel is validated the moment its `.wgsl` lands, with no list
    /// edit required.
    const ALL_SHADERS: &[(&str, &str)] = &[
        ("matmul", WGSL),
        (
            "matmul_vec",
            include_str!("../shaders/wgsl/matmul_vec.wgsl"),
        ),
        (
            "matmul_vec_sg",
            include_str!("../shaders/wgsl/matmul_vec_sg.wgsl"),
        ),
        (
            "matmul_vec_batch",
            include_str!("../shaders/wgsl/matmul_vec_batch.wgsl"),
        ),
        (
            "matmul_vec_q8",
            include_str!("../shaders/wgsl/matmul_vec_q8.wgsl"),
        ),
        (
            "matmul_vec_q8_sg",
            include_str!("../shaders/wgsl/matmul_vec_q8_sg.wgsl"),
        ),
        (
            "matmul_vec_q8_batch",
            include_str!("../shaders/wgsl/matmul_vec_q8_batch.wgsl"),
        ),
        (
            "matmul_vec_q4_batch",
            include_str!("../shaders/wgsl/matmul_vec_q4_batch.wgsl"),
        ),
        (
            "matmul_vec_q4k",
            include_str!("../shaders/wgsl/matmul_vec_q4k.wgsl"),
        ),
        (
            "matmul_vec_q4k_mr",
            include_str!("../shaders/wgsl/matmul_vec_q4k_mr.wgsl"),
        ),
        (
            "matmul_vec_q4k_sg",
            include_str!("../shaders/wgsl/matmul_vec_q4k_sg.wgsl"),
        ),
        (
            "matmul_vec_q4k_batch",
            include_str!("../shaders/wgsl/matmul_vec_q4k_batch.wgsl"),
        ),
        (
            "matmul_vec_q4ks",
            include_str!("../shaders/wgsl/matmul_vec_q4ks.wgsl"),
        ),
        (
            "matmul_vec_q4ks_batch",
            include_str!("../shaders/wgsl/matmul_vec_q4ks_batch.wgsl"),
        ),
        (
            "matmul_vec_q3k",
            include_str!("../shaders/wgsl/matmul_vec_q3k.wgsl"),
        ),
        (
            "matmul_vec_q3k_batch",
            include_str!("../shaders/wgsl/matmul_vec_q3k_batch.wgsl"),
        ),
        (
            "matmul_vec_q4",
            include_str!("../shaders/wgsl/matmul_vec_q4.wgsl"),
        ),
        (
            "matmul_vec_q4_mc",
            include_str!("../shaders/wgsl/matmul_vec_q4_mc.wgsl"),
        ),
        ("rmsnorm", include_str!("../shaders/wgsl/rmsnorm.wgsl")),
        (
            "rmsnorm_add",
            include_str!("../shaders/wgsl/rmsnorm_add.wgsl"),
        ),
        ("swiglu", include_str!("../shaders/wgsl/swiglu.wgsl")),
        ("rope", include_str!("../shaders/wgsl/rope.wgsl")),
        ("rope_qk", include_str!("../shaders/wgsl/rope_qk.wgsl")),
        ("qk_norm", include_str!("../shaders/wgsl/qk_norm.wgsl")),
        ("v_norm", include_str!("../shaders/wgsl/v_norm.wgsl")),
        ("qkv_norm", include_str!("../shaders/wgsl/qkv_norm.wgsl")),
        ("moe_route", include_str!("../shaders/wgsl/moe_route.wgsl")),
        (
            "matmul_vec_moe",
            include_str!("../shaders/wgsl/matmul_vec_moe.wgsl"),
        ),
        (
            "matmul_vec_moe_q4",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4.wgsl"),
        ),
        (
            "matmul_vec_moe_q4k",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4k.wgsl"),
        ),
        (
            "matmul_vec_moe_q4ks",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4ks.wgsl"),
        ),
        (
            "matmul_vec_moe_batch_gu",
            include_str!("../shaders/wgsl/matmul_vec_moe_batch_gu.wgsl"),
        ),
        (
            "matmul_vec_moe_q4_batch_gu",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4_batch_gu.wgsl"),
        ),
        (
            "matmul_vec_moe_q4k_batch_gu",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4k_batch_gu.wgsl"),
        ),
        (
            "matmul_vec_moe_q4ks_batch_gu",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4ks_batch_gu.wgsl"),
        ),
        (
            "matmul_vec_moe_down_reduce",
            include_str!("../shaders/wgsl/matmul_vec_moe_down_reduce.wgsl"),
        ),
        (
            "matmul_vec_moe_q4_down_reduce",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4_down_reduce.wgsl"),
        ),
        (
            "matmul_vec_moe_q4k_down_reduce",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4k_down_reduce.wgsl"),
        ),
        (
            "matmul_vec_moe_q4ks_down_reduce",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4ks_down_reduce.wgsl"),
        ),
        (
            "moe_route_b",
            include_str!("../shaders/wgsl/moe_route_b.wgsl"),
        ),
        (
            "matmul_vec_moe_batch_gu_b",
            include_str!("../shaders/wgsl/matmul_vec_moe_batch_gu_b.wgsl"),
        ),
        (
            "matmul_vec_moe_q4ks_batch_gu_b",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4ks_batch_gu_b.wgsl"),
        ),
        (
            "matmul_vec_moe_q4ks_batch_gu_b_v2",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4ks_batch_gu_b_v2.wgsl"),
        ),
        (
            "matmul_vec_moe_down_reduce_b",
            include_str!("../shaders/wgsl/matmul_vec_moe_down_reduce_b.wgsl"),
        ),
        (
            "matmul_vec_moe_q4ks_down_reduce_b",
            include_str!("../shaders/wgsl/matmul_vec_moe_q4ks_down_reduce_b.wgsl"),
        ),
        (
            "kv_scatter",
            include_str!("../shaders/wgsl/kv_scatter.wgsl"),
        ),
        (
            "kv_scatter_tq",
            include_str!("../shaders/wgsl/kv_scatter_tq.wgsl"),
        ),
        ("attention", include_str!("../shaders/wgsl/attention.wgsl")),
        (
            "attention_tq",
            include_str!("../shaders/wgsl/attention_tq.wgsl"),
        ),
        (
            "attention_splitk",
            include_str!("../shaders/wgsl/attention_splitk.wgsl"),
        ),
        (
            "attention_splitk_combine",
            include_str!("../shaders/wgsl/attention_splitk_combine.wgsl"),
        ),
        ("sample", include_str!("../shaders/wgsl/sample.wgsl")),
        (
            "sample_batched",
            include_str!("../shaders/wgsl/sample_batched.wgsl"),
        ),
        (
            "sample_stochastic",
            include_str!("../shaders/wgsl/sample_stochastic.wgsl"),
        ),
        ("embed", include_str!("../shaders/wgsl/embed.wgsl")),
        ("embed_tok", include_str!("../shaders/wgsl/embed_tok.wgsl")),
        ("geglu", include_str!("../shaders/wgsl/geglu.wgsl")),
        (
            "embed_bf16",
            include_str!("../shaders/wgsl/embed_bf16.wgsl"),
        ),
        (
            "embed_tok_bf16",
            include_str!("../shaders/wgsl/embed_tok_bf16.wgsl"),
        ),
        // the GGUF's own Q4_K token table (GpuModel::embed_q4k, 2026-09-23)
        ("embed_q4k", include_str!("../shaders/wgsl/embed_q4k.wgsl")),
        (
            "embed_tok_q4k",
            include_str!("../shaders/wgsl/embed_tok_q4k.wgsl"),
        ),
        ("add", include_str!("../shaders/wgsl/add.wgsl")),
        ("add_norm", include_str!("../shaders/wgsl/add_norm.wgsl")),
        ("softcap", include_str!("../shaders/wgsl/softcap.wgsl")),
        (
            "scale_inplace",
            include_str!("../shaders/wgsl/scale_inplace.wgsl"),
        ),
    ];

    /// The shaders directory, resolved at test time from the crate root. Globbing it
    /// (rather than trusting `ALL_SHADERS`) makes coverage complete by construction: a
    /// new kernel is validated the moment its `.wgsl` lands, and no shader can silently
    /// escape CI — which is how the whole `matmul_coop_*` family had gone unvalidated.
    /// L364b — `src/shaders/wgsl`, not `src/shaders`: the 145-file shader directory was split by
    /// language. These two tests glob this path and compare it against `ALL_SHADERS`, so they
    /// FAILED the moment the move happened — which is exactly what they are for, and why they
    /// are the reason this restructure was tested rather than trusted to a compile check.
    fn shaders_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/wgsl")
    }

    /// Every `.wgsl` in `src/shaders/` parses and passes naga validation on the CPU,
    /// with no GPU adapter (CI has none). Enumerates the directory so coverage cannot
    /// drift behind newly-added kernels.
    #[test]
    fn all_wgsl_parses_and_validates() {
        let dir = shaders_dir();
        let mut count = 0usize;
        for entry in std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read shaders dir {}: {e}", dir.display()))
        {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{name} should parse: {}", e.emit_to_string(&src)));
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name} should pass naga validation: {e}"));
            count += 1;
        }
        // Guard against a glob that silently matched nothing (wrong dir / moved files).
        assert!(
            count >= ALL_SHADERS.len(),
            "globbed only {count} shaders but ALL_SHADERS embeds {} — shaders dir wrong?",
            ALL_SHADERS.len()
        );
    }

    /// Every shader the engine embeds via `include_str!` still exists on disk — catches
    /// a renamed/deleted `.wgsl` that the glob test would otherwise pass right over.
    #[test]
    fn all_listed_shaders_exist_on_disk() {
        let dir = shaders_dir();
        let on_disk: std::collections::HashSet<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".wgsl"))
            .collect();
        for (name, _) in ALL_SHADERS {
            // `matmul` is embedded as the `WGSL` const (matmul.wgsl); the rest are
            // named after their file stem.
            let file = format!("{name}.wgsl");
            assert!(
                on_disk.contains(&file),
                "ALL_SHADERS embeds `{name}` but {file} is not in {}",
                dir.display()
            );
        }
    }
}

/// Rows `0..rows` of the island's `normed[bank]` scratch hold the trunk's `output_norm`'d hidden
/// for positions `past0..past0+rows`. The draft for position `p` conditions on row
/// `p - 1 - past0` — the row that emitted token `p`. Deriving the bank from the draft's own
/// `past_len` instead read the OTHER bank on every plain step (the previous token's hidden) and
/// the wrong row after a verify window.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MtpHiddenSrc {
    pub bank: usize,
    pub past0: usize,
    pub rows: usize,
}
