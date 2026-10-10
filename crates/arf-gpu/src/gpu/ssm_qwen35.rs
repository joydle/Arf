//! Qwen3.6-27B (arch `qwen35`) **HYBRID gated-delta-net linear-attention** port.
//!
//! The beast is a HYBRID transformer: with `full_attention_interval = 4`, 3 of every
//! 4 trunk layers are **LINEAR attention** (gated delta net — the Mamba2 / GatedDeltaNet
//! family) and 1 of every 4 is a standard **full GQA** layer. Both layer kinds share the
//! SAME MoE FFN (routed experts + a gated SHARED expert) and the SAME pre/post norm +
//! residual skeleton. There is also a single `nextn` MTP block (designed in the design,
//! not yet built).
//!
//! ## Ground truth
//! - llama.cpp `src/models/qwen35moe.cpp`:
//!     - `load_arch_hparams` (lines 4-38): the SSM_* / full_attention_interval / is_recr keys.
//!     - `load_arch_tensors` → `load_block_trunk` (lines 57-107): the per-layer tensor set,
//!       branched on `is_recr(il)` (linear) vs full-attn, plus routed + SHARED experts.
//!     - `build_arch_graph::graph` (lines 181-227): the per-layer skeleton
//!       `attn_norm → (linear|full) attn → +residual → attn_post_norm → MoE ffn → +residual`.
//!     - `build_layer_attn_linear` (lines 360-492): THE gated-delta-net forward (this port).
//!     - `build_layer_ffn` (lines 494-548): MoE + gated shared expert.
//! - The hybrid routing predicate (line 28):
//!   `is_recr(il) = (il < n_layer) && ((il + 1) % full_attention_interval != 0)`.
//!
//! ## The gated-delta-net forward (faithful op sequence — `build_layer_attn_linear`)
//! ```text
//! INPUT: cur [n_embd]  (already post attn_norm)
//!   qkv_mixed = wqkv      @ cur          // [key_dim*2 + value_dim]  (q‖k‖v packed)
//!   z         = wqkv_gate @ cur          // [value_dim]              (the output gate)
//!   beta  = sigmoid( ssm_beta  @ cur )                       // [num_v_heads]
//!   alpha = ssm_alpha @ cur                                  // [num_v_heads]
//!   alpha = softplus( alpha + ssm_dt )                       // ssm_dt is a per-head bias
//!   gate  = alpha * ssm_a                                    // ssm_a = -A_log.exp() decay coef
//!   // --- causal depthwise conv1d over the sequence, carrying CONV STATE across tokens ---
//!   conv_in  = qkv_mixed                 // conv_channels = d_inner + 2*ssm_n_group*ssm_d_state
//!   conv_out = silu( ssm_conv1d(conv_in, conv_state) )       // ggml_ssm_conv: causal depthwise
//!   q_conv,k_conv,v_conv = split(conv_out)                   // by head_k/v geometry
//!   q_conv = l2_norm(q_conv); k_conv = l2_norm(k_conv)       // per-head L2 (NOT rms)
//!   (if num_k_heads != num_v_heads: repeat q/k across the v-head groups)
//!   // --- THE DELTA RULE: carries SSM STATE [head_v × head_v × num_v_heads] across tokens ---
//!   readout = recurrent_attn(ssm_state, q_conv, k_conv, v_conv, gate, beta)  // gated delta net
//!   out = rmsnorm(readout, ssm_norm) * silu(z)               // build_norm_gated
//!   out = ssm_out @ out                                      // [n_embd]
//! RETURN out  (+ updated conv_state + ssm_state)
//! ```
//! Geometry (lines 366-371 / 64-70):
//!   head_k_dim = ssm_d_state, num_k_heads = ssm_n_group, num_v_heads = ssm_dt_rank,
//!   head_v_dim = d_inner / num_v_heads, key_dim = head_k_dim*num_k_heads,
//!   value_dim = head_v_dim*num_v_heads, conv_dim = key_dim*2 + value_dim.
//! TWO recurrent states per linear layer (O(1) memory — NO growing KV cache):
//!   - conv_state: [conv_dim, ssm_d_conv-1]   (the depthwise-conv ring)
//!   - ssm_state:  [head_v_dim, head_v_dim, num_v_heads]   (the delta-net matrix state)
//!
//! ## What this REUSES from the existing trunk (the whole point — minimal new surface):
//!   - `super::dispatch::add_matmul` (auto-dispatches `matmul_vec_q4ks`) for EVERY Q4_K_S
//!     GEMV: wqkv, wqkv_gate, ssm_beta, ssm_alpha, ssm_out, and the full-GQA q/k/v/o.
//!   - `k.rmsnorm` (+ `super::dispatch::dims_norm`) for attn_norm / attn_post_norm / ssm_norm.
//!   - `k.swiglu`/`k.add`/`super::dispatch::add_residual` for silu-gate / residual adds.
//!   - `GpuModel::add_moe_block` UNCHANGED for the routed MoE FFN — and it ALREADY carries
//!     `shared_gate/up/down` + `shared_experts` (see `GpuMoe`), so the beast's gated shared
//!     expert is structurally supported (the gate-sigmoid wiring is flagged below).
//!   - The full-GQA layers (1/4) reuse the EXISTING qwen3-coder attention path verbatim — a
//!     trunk [`GpuLayer`] consumed by the same decode attention dispatch.
//!
//! ## What is genuinely NEW (the multi-day port work — enumerated in the build report):
//!   1. `ssm_conv1d`         — causal depthwise conv1d with a carried conv ring state. NEW.
//!   2. gated-delta-net recurrence — the delta-rule state update + per-head readout. NEW.
//!   3. `l2_norm` (per-head) — L2 (not RMS) normalize of q_conv/k_conv. NEW (small).
//!   4. `softplus`           — `log(1+exp(x))`. NEW (small; or fold into a fused kernel).
//!   (sigmoid exists only as a fused step inside attention today — a standalone elementwise
//!    `sigmoid` for `beta` is either a tiny new kernel or folded into the recurrence kernel.)
//!
//! The loader and both forward paths are fully wired — no `todo!()` scaffold remains. GGUF
//! reads go through `raw_mat` / `raw_f32` / `load_dense_ffn`; the kernels listed under "What
//! is genuinely NEW" above are all present in this module. (The original port scaffold marked
//! every unwireable point `todo!("M-…")` / `todo!("KERNEL: …")` and carried module-wide
//! scaffold-only lint allows; those wire-points are filled and the allows are gone.) The few
//! remaining `#[allow(dead_code)]`s are deliberate and justified inline (unread k/v scratch
//! handles kept so each struct owns one buffer per GQA tensor).

use std::sync::Arc;

use arf_core::error::{ArfError, Result};
use arf_core::model::gguf::LazyGguf;
use arf_core::tensor::Q4KSMatrix;

use super::command::CommandPass;
#[cfg(target_os = "macos")]
use super::concurrent_metal::SharedBuffer;
use super::context::GpuContext;
// L364 — `dims_attn` is used only by the macOS island path in this file.
#[cfg(target_os = "macos")]
use super::dispatch::dims_attn;
use super::dispatch::{add_matmul, dims_norm, dims_norm_m};
use super::types::{GpuDenseMlp, GpuKernels, GpuLayer, GpuMatWeight, GpuMlp};
use crate::arena::GpuArena;

// ---------------------------------------------------------------------------
// M6 — RESIDENT-WEIGHT GPU GEMV (the speed lever)
//
// The M4b/M5 generation path ran EVERY large projection through `gguf_gemv` — a host
// CPU loop that re-reads + dequantizes the full weight from the GGUF mmap EVERY token
// (27B mults/token on the CPU + a fresh dequant of ~15 GB/token). That host path is the
// entire ~2000× slowdown vs llama.cpp.
//
// M6 replaces those host GEMVs with the SAME GPU GEMV the proven coder-30B decode uses:
// the resident `GpuMatWeight` (uploaded ONCE by `load_qwen35`) + `add_matmul`
// (auto-dispatches `matmul_vec_q4ks` for Q4_K_S, `matmul` for the bf16-fallback Q5_K/Q6_K
// tensors). The generation loop stays host-driven (the gated-delta-net island ops still
// round-trip), so each GEMV is: upload the activation → add_matmul on a CommandPass →
// submit → read back. The weight is NEVER re-read from disk.
// ---------------------------------------------------------------------------

/// Coarse per-phase timers (µs), gated by ARF_QWEN35_TIME, to attribute the per-token
/// cost across (GEMVs, island gated-delta-net, host compute) — honest bottleneck reporting.
pub mod timers {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static GEMV_US: AtomicU64 = AtomicU64::new(0);
    pub static ISLAND_US: AtomicU64 = AtomicU64::new(0);
    pub fn on() -> bool {
        std::env::var_os("ARF_QWEN35_TIME").is_some()
    }
    pub fn add_gemv(us: u64) {
        GEMV_US.fetch_add(us, Ordering::Relaxed);
    }
    pub fn add_island(us: u64) {
        ISLAND_US.fetch_add(us, Ordering::Relaxed);
    }
    pub fn take() -> (u64, u64) {
        (
            GEMV_US.swap(0, Ordering::Relaxed),
            ISLAND_US.swap(0, Ordering::Relaxed),
        )
    }
}

/// Owns the decode kernel set + scratch buffers for resident-weight GEMVs in the qwen35
/// generation loop. Built ONCE per generation; reused every token. The activation in/out
/// buffers are sized to the largest GEMV (lm_head: in = n_embd, out = vocab) and reused.
pub struct Qwen35GpuGemv {
    ctx: Arc<GpuContext>,
    kernels: GpuKernels,
    /// Uniform arena (the `mm` Dims for each add_matmul). Reset per token.
    uni: GpuArena,
    /// Activation input buffer (STORAGE | COPY_DST), max k floats.
    a_buf: wgpu::Buffer,
    /// Output buffer (STORAGE | COPY_SRC), max n floats.
    c_buf: wgpu::Buffer,
    /// Persistent MAP_READ staging buffer (max n floats) — reused for every readback so the
    /// hot loop never allocates a staging buffer per GEMV group (the per-token submit lever).
    staging: wgpu::Buffer,
    max_k: usize,
    max_n: usize,

    // --- M7: single-CommandPass-per-(sub)layer scratch (the round-trip-elimination lever) ---
    /// Pool of DISTINCT f32 activation buffers. Each per-layer intermediate (normed, qkv_mixed,
    /// z, projections, gated readout, FFN gate/up/silu) is a separate pool buffer so the whole
    /// chain records onto ONE `CommandPass` without a single dispatch ever binding the same
    /// buffer for both read (input) and read_write (output) — wgpu forbids that conflict in one
    /// usage scope. Allocated round-robin within a layer (each layer uses ≤ pool.len() handles);
    /// `act_cursor` resets per layer. STORAGE|COPY_DST|COPY_SRC.
    act_pool: Vec<wgpu::Buffer>,
    act_cursor: usize,
    act_cap: usize,
    /// Persistent MAP_READ staging for the M7 batched-readback (island-feed values + logits).
    rb_staging: wgpu::Buffer,
    /// The residual stream `[n_embd]`, resident across ALL layers of a token. Seeded from the
    /// embedding row each token; advanced in place by every layer's residual adds.
    hidden_buf: wgpu::Buffer,
    n_embd: usize,

    /// **M10 STEP 1** — the gated-delta-net island FEED, as dual-view (wgpu+Metal) SharedBuffers.
    /// Pass A's `qkv_mixed`/`beta_raw`/`alpha` projections write the wgpu views; the island reads
    /// the SAME memory via the Metal views — NO host readback. `ssm_dt`/`ssm_a` are the static
    /// per-layer vectors the island's `gdn_g_decay` consumes. Built lazily on the first linear
    /// layer (sized from geometry). macOS-only (the island is Metal-only).
    /// Retained for the (default-off) island path; M11 (default) uses `gdn_res` instead.
    #[cfg(target_os = "macos")]
    gdn_feed: Option<GdnFeed>,

    /// **M11** — per-linear-layer RESIDENT gated-delta-net state, ALL pure wgpu, so the whole
    /// recurrence records onto the wgpu CommandPass (no Metal island, single queue). Keyed by
    /// trunk layer `il`. `conv_state`/`ssm_state` carry across tokens (zero-init on first touch);
    /// `conv_kernel`/`ssm_dt`/`ssm_a` are the static per-layer vectors uploaded once on first touch.
    gdn_res: std::collections::HashMap<usize, GdnResident>,

    /// **B-PARALLEL (concurrency)** — number of concurrent sequences decoding in lockstep. `1` for
    /// the single-stream path (byte-identical to M11). When `> 1`, every act-pool buffer / hidden
    /// is sized `batch ×` the per-seq length (b-outer: seq b's data at `[b*len..]`), the matmuls
    /// run with `m = batch` rows (the conc64 batched GEMM), and the GDN/GQA chains advance `batch`
    /// independent O(1) states. The `b=0` slice of a batched run reproduces the single-stream M11
    /// path exactly (the parity oracle).
    batch: usize,
}

/// **M11** — resident per-layer gated-delta-net buffers (pure wgpu). The recurrent state advances
/// in place across tokens; the static weights are uploaded once. Sized from geometry.
struct GdnResident {
    /// `[conv_dim*(d_conv-1)]` depthwise-conv ring (READ+WRITE, carried across tokens).
    conv_state: wgpu::Buffer,
    /// `[s*s*nvh]` delta-net matrix state (READ+WRITE, carried across tokens).
    ssm_state: wgpu::Buffer,
    /// `[d_conv*conv_dim]` static conv weight (uploaded once).
    conv_kernel: wgpu::Buffer,
    /// `[nvh]` static ssm_dt bias (uploaded once).
    ssm_dt: wgpu::Buffer,
    /// `[nvh]` static ssm_a scale (uploaded once).
    ssm_a: wgpu::Buffer,
}

/// **M10 STEP 1** — dual-view feed buffers for the gated-delta-net island. Each is a
/// [`SharedBuffer`] so a wgpu dispatch (Pass A's projection) and a Metal dispatch (the island
/// prologue + recurrence) alias the SAME allocation — the qkv/beta/alpha values never round-trip
/// through a host `Vec`. Sized once from `(conv_dim, nvh)`.
#[cfg(target_os = "macos")]
struct GdnFeed {
    /// `qkv_mixed` [conv_dim] — wqkv @ normed; the conv input.
    qkv: SharedBuffer,
    /// `beta_raw` [nvh] — ssm_beta @ normed; sigmoid'd on the island into `beta`.
    beta_raw: SharedBuffer,
    /// `alpha` [nvh] — ssm_alpha @ normed; `softplus(alpha+dt)*ssm_a` on the island into g_decay.
    alpha: SharedBuffer,
    /// `ssm_dt` [nvh] — static per-layer bias for g_decay (uploaded per layer; tiny).
    ssm_dt: SharedBuffer,
    /// `ssm_a` [nvh] — static per-layer scale for g_decay (uploaded per layer; tiny).
    ssm_a: SharedBuffer,
}

/// A handle into the M7 activation pool: which distinct pool buffer + the logical f32 length.
#[derive(Clone, Copy)]
pub struct Act {
    buf_idx: usize,
    len: usize,
}

/// A bindable sub-range resource (offset 0, `len` f32) over a pool buffer.
fn act_binding(buf: &wgpu::Buffer, a: Act) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: buf,
        offset: 0,
        size: wgpu::BufferSize::new((a.len * 4) as u64),
    })
}

impl Qwen35GpuGemv {
    /// Build the GEMV helper (single-stream; `batch = 1`). `max_k` / `max_n` bound the reusable IO
    /// buffers (use the model's largest GEMV: max_k = n_embd, max_n = vocab — the lm_head). `n_embd`
    /// sizes the resident residual-stream buffer. Byte-identical to the M11 path.
    pub fn new(ctx: &Arc<GpuContext>, max_k: usize, max_n: usize, n_embd: usize) -> Self {
        Self::new_batched(ctx, max_k, max_n, n_embd, 1)
    }

    /// **B-PARALLEL** — build the GEMV helper sized for `batch` concurrent sequences. Every act-pool
    /// buffer is `batch × max_n`, the IO buffers `batch × max_{k,n}`, and `hidden` is `batch × n_embd`
    /// (all b-outer). `batch = 1` is byte-identical to `new` (the single-stream regression oracle).
    pub fn new_batched(
        ctx: &Arc<GpuContext>,
        max_k: usize,
        max_n: usize,
        n_embd: usize,
        batch: usize,
    ) -> Self {
        let batch = batch.max(1);
        let max_k = max_k * batch;
        let max_n = max_n * batch;
        let n_embd_buf = n_embd * batch;
        let kernels = crate::weights::build_kernels(ctx, false);
        // Uniforms: a handful of 16-byte Dims per token; 64 KiB is ample, reset each call.
        let uni = GpuArena::new(
            ctx,
            "qwen35_gemv_uni",
            64 * 1024,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )
        .expect("qwen35 gemv uniform arena");
        let a_buf = ctx.create_buffer(
            "qwen35_gemv_a",
            (max_k * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let c_buf = ctx.create_buffer(
            "qwen35_gemv_c",
            (max_n * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = ctx.create_buffer(
            "qwen35_gemv_staging",
            (max_n * 4) as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        // M7: a pool of DISTINCT activation buffers (each sized to the largest scratch =
        // max_n f32, the FFN n_ff / conv_dim / q_out). One per logical intermediate so a single
        // dispatch never reads+writes the SAME buffer (wgpu forbids that usage conflict). The M11
        // single-CommandPass linear layer holds 20 live handles (Pass A + GDN chain intermediates
        // + Pass B + FFN tail); 24 buffers give slack. ~24 × max_n × 4 bytes.
        let act_cap = max_n;
        let act_pool: Vec<wgpu::Buffer> = (0..24)
            .map(|i| {
                ctx.create_buffer(
                    &format!("qwen35_act_{i}"),
                    (act_cap * 4).max(16) as u64,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                )
            })
            .collect();
        let rb_staging = ctx.create_buffer(
            "qwen35_rb_staging",
            (max_n * 4) as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let hidden_buf = ctx.create_buffer(
            "qwen35_hidden",
            (n_embd_buf * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        Qwen35GpuGemv {
            ctx: ctx.clone(),
            kernels,
            uni,
            a_buf,
            c_buf,
            staging,
            max_k,
            max_n,
            act_pool,
            act_cursor: 0,
            act_cap,
            rb_staging,
            hidden_buf,
            n_embd,
            #[cfg(target_os = "macos")]
            gdn_feed: None,
            gdn_res: std::collections::HashMap::new(),
            batch,
        }
    }

    /// **M11** — get-or-create the resident gated-delta-net state for layer `il`. On first touch:
    /// zero-init conv_state/ssm_state (matching a fresh `CpuLinearState`) and upload the static
    /// conv_kernel/ssm_dt/ssm_a once. `hw` = (conv_kernel, ssm_dt, ssm_a, ssm_norm) host vectors.
    fn ensure_gdn_resident(
        &mut self,
        il: usize,
        conv_len: usize,
        ssm_len: usize,
        conv_kernel: &[f32],
        ssm_dt: &[f32],
        ssm_a: &[f32],
    ) {
        if self.gdn_res.contains_key(&il) {
            return;
        }
        let conv_state = self
            .ctx
            .storage_zeros(&format!("qwen35.gdn_conv_state.{il}"), conv_len);
        let ssm_state = self
            .ctx
            .storage_zeros(&format!("qwen35.gdn_ssm_state.{il}"), ssm_len);
        let conv_kernel_b = self.ctx.create_buffer(
            &format!("qwen35.gdn_conv_kernel.{il}"),
            (conv_kernel.len() * 4).max(16) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let ssm_dt_b = self.ctx.create_buffer(
            &format!("qwen35.gdn_ssm_dt.{il}"),
            (ssm_dt.len() * 4).max(16) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let ssm_a_b = self.ctx.create_buffer(
            &format!("qwen35.gdn_ssm_a.{il}"),
            (ssm_a.len() * 4).max(16) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.ctx.write_at(&conv_kernel_b, 0, conv_kernel);
        self.ctx.write_at(&ssm_dt_b, 0, ssm_dt);
        self.ctx.write_at(&ssm_a_b, 0, ssm_a);
        self.gdn_res.insert(
            il,
            GdnResident {
                conv_state,
                ssm_state,
                conv_kernel: conv_kernel_b,
                ssm_dt: ssm_dt_b,
                ssm_a: ssm_a_b,
            },
        );
    }

    /// **M10 STEP 1** — lazily build (once) the dual-view island-feed SharedBuffers, sized for
    /// this model's geometry. Returns `Err` only if the Metal-backed alloc fails (non-Metal
    /// adapter) — the caller then keeps the host-readback path. `nvh` = num_v_heads.
    #[cfg(target_os = "macos")]
    fn ensure_gdn_feed(&mut self, conv_dim: usize, nvh: usize) -> Result<()> {
        if self.gdn_feed.is_some() {
            return Ok(());
        }
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mk = |bytes: usize, label: &str| -> Result<SharedBuffer> {
            SharedBuffer::alloc(&self.ctx, (bytes.max(16)) as u64, usage, label)
                .ok_or_else(|| ArfError::model_load(format!("gdn feed alloc {label}")))
        };
        self.gdn_feed = Some(GdnFeed {
            qkv: mk(conv_dim * 4, "qwen35_gdn_qkv")?,
            beta_raw: mk(nvh * 4, "qwen35_gdn_beta_raw")?,
            alpha: mk(nvh * 4, "qwen35_gdn_alpha")?,
            ssm_dt: mk(nvh * 4, "qwen35_gdn_ssm_dt")?,
            ssm_a: mk(nvh * 4, "qwen35_gdn_ssm_a")?,
        });
        Ok(())
    }

    // -----------------------------------------------------------------------
    // M7 — single-CommandPass-per-(sub)layer API (the round-trip-elimination lever)
    //
    // Instead of one submit+map_async+poll(wait_indefinitely) PER GEMV group (the ~190
    // round-trips/token tax), a (sub)layer's whole chain — rmsnorm → projections →
    // silu/gate/residual → out-proj — records onto ONE `CommandPass` and submits ONCE.
    // Intermediates live in the resident `act` arena (GPU buffers); only the values the
    // gated-delta-net island consumes (per linear layer) and the final logits read back.
    // -----------------------------------------------------------------------

    /// Reset the per-token scratch (the uniform arena + the activation pool cursor). Call ONCE
    /// at the top of every token before recording layers.
    pub fn begin_token(&mut self) {
        self.uni.reset();
        self.act_cursor = 0;
    }

    /// Hand out the next DISTINCT pool buffer as a `len`-f32 activation handle. Round-robins
    /// within a layer; `act_cursor` resets at `begin_layer`. Panics if a layer needs more than
    /// the pool size (raise the pool count in `new`).
    fn act_alloc(&mut self, len: usize) -> Act {
        assert!(
            len <= self.act_cap,
            "act_alloc len {len} > cap {}",
            self.act_cap
        );
        let idx = self.act_cursor;
        assert!(
            idx < self.act_pool.len(),
            "act pool exhausted ({idx}); raise pool size"
        );
        self.act_cursor += 1;
        Act { buf_idx: idx, len }
    }

    /// Reserve + upload a host slice into a fresh pool buffer (the per-token embedding row, or
    /// an island readout coming back to the GPU).
    fn act_upload(&mut self, data: &[f32]) -> Act {
        let a = self.act_alloc(data.len());
        self.ctx.write_at(&self.act_pool[a.buf_idx], 0, data);
        a
    }

    /// A bindable resource for `a` (its pool buffer, first `len` f32).
    fn act_res(&self, a: Act) -> wgpu::BindingResource<'_> {
        act_binding(&self.act_pool[a.buf_idx], a)
    }

    /// Record `out = rmsnorm(x, weight)` over `rows` rows of `hidden` each (per-head rmsnorm
    /// uses rows>1; whole-vector uses rows=1) onto `cp`. `weight` is a resident `[hidden]` buffer.
    fn rec_rmsnorm(
        &mut self,
        cp: &mut CommandPass,
        x: Act,
        weight: &wgpu::Buffer,
        out: Act,
        rows: usize,
        hidden: usize,
        eps: f64,
    ) {
        let nd = self
            .uni
            .alloc_write(&dims_norm_m(rows, hidden, eps))
            .expect("rmsnorm dims");
        let xr = act_binding(&self.act_pool[x.buf_idx], x);
        let outr = act_binding(&self.act_pool[out.buf_idx], out);
        let und = self.uni.resource(&nd);
        cp.add_bound(
            &self.kernels.rmsnorm,
            "qwen35_rmsnorm",
            &[xr, weight.as_entire_binding(), outr, und],
            [rows as u32, 1, 1],
        );
    }

    /// Record `out = silu(gate) * up` (SwiGLU) over `n` elements onto `cp`. For the GDN gate
    /// `rmsnorm(readout)*silu(z)` pass gate=z, up=normed; for the FFN pass gate=ffn_gate, up=ffn_up.
    fn rec_swiglu(&mut self, cp: &mut CommandPass, gate: Act, up: Act, out: Act, n: usize) {
        let d = self
            .uni
            .alloc_write(&[n as u32, 0u32, 0, 0])
            .expect("swiglu dims");
        let gr = act_binding(&self.act_pool[gate.buf_idx], gate);
        let ur = act_binding(&self.act_pool[up.buf_idx], up);
        let outr = act_binding(&self.act_pool[out.buf_idx], out);
        let und = self.uni.resource(&d);
        cp.add_bound(
            &self.kernels.swiglu,
            "qwen35_swiglu",
            &[gr, ur, outr, und],
            [(n as u32).div_ceil(256), 1, 1],
        );
    }

    /// Record `out = W·x` (resident Q4_K_S / bf16 weight) onto `cp`. `x` and `out` are DISTINCT
    /// pool buffers (so the matmul dispatch never reads+writes the same buffer).
    fn rec_matmul(
        &mut self,
        cp: &mut CommandPass,
        x: Act,
        w: &GpuMatWeight,
        out: Act,
        n: usize,
        k: usize,
    ) {
        debug_assert_ne!(
            x.buf_idx, out.buf_idx,
            "matmul in/out must be distinct buffers"
        );
        let xr = act_binding(&self.act_pool[x.buf_idx], x);
        let outr = act_binding(&self.act_pool[out.buf_idx], out);
        add_matmul(cp, &self.kernels, &mut self.uni, xr, w, outr, n, k, None);
    }

    /// **M10 STEP 1** — like [`rec_matmul`] but the `n`-element output is an EXPLICIT external
    /// buffer (e.g. a [`SharedBuffer`]'s wgpu view) instead of a pool `Act`. Used so Pass A's
    /// island-feed projections (qkv_mixed/beta_raw/alpha) write the dual-view feed buffers the
    /// Metal island reads — no host readback. `x` is a resident pool activation.
    fn rec_matmul_out(
        &mut self,
        cp: &mut CommandPass,
        x: Act,
        w: &GpuMatWeight,
        out: &wgpu::Buffer,
        n: usize,
        k: usize,
    ) {
        let xr = act_binding(&self.act_pool[x.buf_idx], x);
        let outr = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: out,
            offset: 0,
            size: wgpu::BufferSize::new((n * 4) as u64),
        });
        add_matmul(cp, &self.kernels, &mut self.uni, xr, w, outr, n, k, None);
    }

    /// **M11** — record the WHOLE gated-delta-net recurrence onto `cp` (single wgpu CommandPass,
    /// no Metal island, no cross-queue wait). Consumes Pass-A's resident `qkv`/`beta_raw`/`alpha`
    /// activations + the layer's resident state (conv_state/ssm_state advance IN PLACE), and writes
    /// the readout `[value_dim]` into `out`. Steps (each a `add_bound` dispatch; wgpu inserts the
    /// per-resource hazard barriers between consecutive dispatches in the merged pass):
    ///   beta=sigmoid(beta_raw) ; g=softplus(alpha+dt)*a ; conv ; l2 ; tile ; recurrence.
    /// `geom` carries the geometry; `il` keys the resident state (must be `ensure_gdn_resident`'d).
    fn rec_gdn_chain(
        &mut self,
        cp: &mut CommandPass,
        il: usize,
        geom: LinearAttnGeom,
        qkv: Act,
        beta_raw: Act,
        alpha: Act,
        out: Act,
        eps: f64,
    ) {
        let conv_dim = geom.conv_dim;
        let key_dim = geom.key_dim;
        let value_dim = geom.value_dim;
        let s = geom.head_k_dim;
        let nkh = geom.num_k_heads;
        let nvh = geom.num_v_heads;
        let d_conv = geom.d_conv;

        // Transient intermediates from the act pool (all <= conv_dim <= max_n).
        let beta = self.act_alloc(nvh);
        let g_decay = self.act_alloc(nvh);
        let conv_out = self.act_alloc(conv_dim);
        let q_rep = self.act_alloc(value_dim);
        let k_rep = self.act_alloc(value_dim);
        let v_rep = self.act_alloc(value_dim);

        // Resident handles cloned out (Arc refcount) so `self.gdn_res` borrow drops before recording.
        let (conv_state, ssm_state, conv_kernel, ssm_dt, ssm_a) = {
            let r = self
                .gdn_res
                .get(&il)
                .expect("gdn resident state (ensure_gdn_resident)");
            (
                r.conv_state.clone(),
                r.ssm_state.clone(),
                r.conv_kernel.clone(),
                r.ssm_dt.clone(),
                r.ssm_a.clone(),
            )
        };

        // (0a) beta = sigmoid(beta_raw): out[nvh], src=beta_raw, dims {nvh,_,_,_}.
        {
            let dd = self
                .uni
                .alloc_write(&[nvh as u32, 0u32, 0, 0])
                .expect("sig dims");
            cp.add_bound(
                &self.kernels.gdn_sigmoid_out,
                "qwen35_gdn_sigmoid",
                &[
                    act_binding(&self.act_pool[beta.buf_idx], beta),
                    act_binding(&self.act_pool[beta_raw.buf_idx], beta_raw),
                    self.uni.resource(&dd),
                ],
                [(nvh as u32).div_ceil(256), 1, 1],
            );
        }
        // (0b) g_decay = softplus(alpha+ssm_dt)*ssm_a: out[nvh], alpha, ssm_dt, ssm_a, dims.
        {
            let dd = self
                .uni
                .alloc_write(&[nvh as u32, 0u32, 0, 0])
                .expect("gdec dims");
            cp.add_bound(
                &self.kernels.gdn_g_decay,
                "qwen35_gdn_g_decay",
                &[
                    act_binding(&self.act_pool[g_decay.buf_idx], g_decay),
                    act_binding(&self.act_pool[alpha.buf_idx], alpha),
                    ssm_dt.as_entire_binding(),
                    ssm_a.as_entire_binding(),
                    self.uni.resource(&dd),
                ],
                [(nvh as u32).div_ceil(256), 1, 1],
            );
        }
        // (1) conv: conv_in=qkv -> conv_out (post-silu), advance conv_state.
        {
            let dd = self
                .uni
                .alloc_write(&[conv_dim as u32, d_conv as u32, 0, 0])
                .expect("conv dims");
            cp.add_bound(
                &self.kernels.gdn_conv1d,
                "qwen35_gdn_conv1d",
                &[
                    act_binding(&self.act_pool[qkv.buf_idx], qkv),
                    conv_kernel.as_entire_binding(),
                    conv_state.as_entire_binding(),
                    act_binding(&self.act_pool[conv_out.buf_idx], conv_out),
                    self.uni.resource(&dd),
                ],
                [(conv_dim as u32).div_ceil(256), 1, 1],
            );
        }
        // (2) l2_norm IN PLACE on conv_out q (head 0..nkh of region [0..key_dim]) and k
        //     ([key_dim..2*key_dim]). The MSL bound q/k via byte offsets; here we run the kernel
        //     ONCE over the q region and ONCE over the k region using offset bindings.
        for region_off in [0usize, key_dim] {
            let dd = self
                .uni
                .alloc_write(&[s as u32, nkh as u32, (eps as f32).to_bits(), 0])
                .expect("l2 dims");
            let xr = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.act_pool[conv_out.buf_idx],
                offset: (region_off * 4) as u64,
                size: wgpu::BufferSize::new((key_dim * 4) as u64),
            });
            cp.add_bound(
                &self.kernels.gdn_l2_norm,
                "qwen35_gdn_l2_norm",
                &[xr, self.uni.resource(&dd)],
                [nkh as u32, 1, 1],
            );
        }
        // (3) tile: split conv_out + repeat q/k across v-head groups -> q_rep/k_rep/v_rep.
        {
            let dd = self
                .uni
                .alloc_write(&[s as u32, nkh as u32, nvh as u32, key_dim as u32])
                .expect("tile dims");
            cp.add_bound(
                &self.kernels.gdn_tile,
                "qwen35_gdn_tile",
                &[
                    act_binding(&self.act_pool[conv_out.buf_idx], conv_out),
                    act_binding(&self.act_pool[q_rep.buf_idx], q_rep),
                    act_binding(&self.act_pool[k_rep.buf_idx], k_rep),
                    act_binding(&self.act_pool[v_rep.buf_idx], v_rep),
                    self.uni.resource(&dd),
                ],
                [(value_dim as u32).div_ceil(256), 1, 1],
            );
        }
        // (4) recurrence: q_rep,k_rep,v_rep,g_decay,beta,ssm_state -> out (readout), advance state.
        //     One workgroup per v-head, S_v threads/wg (== s).
        {
            let dd = self
                .uni
                .alloc_write(&[s as u32, s as u32, nvh as u32, 0])
                .expect("gdn dims");
            cp.add_bound(
                &self.kernels.gdn_recurrence,
                "qwen35_gdn_recurrence",
                &[
                    act_binding(&self.act_pool[q_rep.buf_idx], q_rep),
                    act_binding(&self.act_pool[k_rep.buf_idx], k_rep),
                    act_binding(&self.act_pool[v_rep.buf_idx], v_rep),
                    act_binding(&self.act_pool[g_decay.buf_idx], g_decay),
                    act_binding(&self.act_pool[beta.buf_idx], beta),
                    ssm_state.as_entire_binding(),
                    act_binding(&self.act_pool[out.buf_idx], out),
                    self.uni.resource(&dd),
                ],
                [nvh as u32, 1, 1],
            );
        }
    }

    // =======================================================================
    // B-PARALLEL recording API (the concurrency advantage). Each method is the structural twin of its
    // single-stream sibling, widened to `self.batch` b-outer rows. An `Act` of logical per-seq
    // length `len` is allocated as `act_alloc(self.batch * len)` so its physical span is the whole
    // b-outer block (seq b at `[b*len..]`). For `batch == 1` these reproduce the single-stream
    // dispatches byte-for-byte (the regression oracle).
    // =======================================================================

    /// Allocate a b-outer activation: physical length `self.batch * per_seq`.
    fn act_alloc_b(&mut self, per_seq: usize) -> Act {
        self.act_alloc(self.batch * per_seq)
    }

    /// Seed the resident residual stream with `batch` embedding rows (b-outer; `rows[b]` = seq b's
    /// `[n_embd]` row). Single host upload of the whole `[batch*n_embd]` block.
    fn seed_hidden_batched(&self, flat_rows: &[f32]) {
        debug_assert_eq!(flat_rows.len(), self.batch * self.n_embd);
        self.ctx.write_at(&self.hidden_buf, 0, flat_rows);
    }

    /// Read the resident residual stream back to host as `[batch*n_embd]` (b-outer).
    fn readback_hidden_batched(&mut self) -> Vec<f32> {
        let bytes = (self.batch * self.n_embd * 4) as u64;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.hidden_buf, 0, &self.rb_staging, 0, bytes);
        self.ctx.queue.submit(std::iter::once(enc.finish()));
        let slice = self.rb_staging.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let out: Vec<f32> = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };
        self.rb_staging.unmap();
        out
    }

    /// Record `out_act = rmsnorm(hidden_buf, weight)` over `batch` b-outer rows (whole `[n_embd]`
    /// each). The rmsnorm kernel is per-row; grid = `batch` rows.
    fn rec_rmsnorm_from_hidden_b(
        &mut self,
        cp: &mut CommandPass,
        weight: &wgpu::Buffer,
        out: Act,
        eps: f64,
    ) {
        let nd = self
            .uni
            .alloc_write(&dims_norm_m(self.batch, self.n_embd, eps))
            .expect("rmsnorm_b dims");
        let hr = self.hidden_buf.as_entire_binding();
        let outr = act_binding(&self.act_pool[out.buf_idx], out);
        let und = self.uni.resource(&nd);
        cp.add_bound(
            &self.kernels.rmsnorm,
            "qwen35_rmsnorm_h_b",
            &[hr, weight.as_entire_binding(), outr, und],
            [self.batch as u32, 1, 1],
        );
    }

    /// Record `out = rmsnorm(x, weight)` over `batch * rows_per_seq` b-outer rows of `cols` each
    /// (the per-head GDN readout norm: rows_per_seq = nvh, cols = head_v_dim). grid = total rows.
    fn rec_rmsnorm_b(
        &mut self,
        cp: &mut CommandPass,
        x: Act,
        weight: &wgpu::Buffer,
        out: Act,
        rows_per_seq: usize,
        cols: usize,
        eps: f64,
    ) {
        let total_rows = self.batch * rows_per_seq;
        let nd = self
            .uni
            .alloc_write(&dims_norm_m(total_rows, cols, eps))
            .expect("rmsnorm_b dims");
        let xr = act_binding(&self.act_pool[x.buf_idx], x);
        let outr = act_binding(&self.act_pool[out.buf_idx], out);
        let und = self.uni.resource(&nd);
        cp.add_bound(
            &self.kernels.rmsnorm,
            "qwen35_rmsnorm_b",
            &[xr, weight.as_entire_binding(), outr, und],
            [total_rows as u32, 1, 1],
        );
    }

    /// Record `out = silu(gate) * up` (SwiGLU) over `batch * per_seq` elements (b-outer, fully
    /// elementwise — the weight is shared so per-row geometry is irrelevant).
    fn rec_swiglu_b(&mut self, cp: &mut CommandPass, gate: Act, up: Act, out: Act, per_seq: usize) {
        let total = self.batch * per_seq;
        let d = self
            .uni
            .alloc_write(&[total as u32, 0u32, 0, 0])
            .expect("swiglu_b dims");
        let gr = act_binding(&self.act_pool[gate.buf_idx], gate);
        let ur = act_binding(&self.act_pool[up.buf_idx], up);
        let outr = act_binding(&self.act_pool[out.buf_idx], out);
        let und = self.uni.resource(&d);
        cp.add_bound(
            &self.kernels.swiglu,
            "qwen35_swiglu_b",
            &[gr, ur, outr, und],
            [(total as u32).div_ceil(256), 1, 1],
        );
    }

    /// Record in-place residual add `hidden_buf[i] += b_act[i]` over `[batch*n_embd]`.
    fn rec_residual_into_hidden_b(&mut self, cp: &mut CommandPass, b: Act) {
        let total = self.batch * self.n_embd;
        let d = self
            .uni
            .alloc_write(&[total as u32, 0u32, 0, 0])
            .expect("add_b dims");
        let hr = self.hidden_buf.as_entire_binding();
        let br = act_binding(&self.act_pool[b.buf_idx], b);
        let und = self.uni.resource(&d);
        cp.add_bound(
            &self.kernels.add,
            "qwen35_add_h_b",
            &[hr, br, und],
            [(total as u32).div_ceil(256), 1, 1],
        );
    }

    /// Record a BATCHED matmul `out[batch,n] = x[batch,k] · Wᵀ` (the conc64 batched GEMM/GEMV).
    /// `x` and `out` are b-outer Acts (physical len `batch*k` / `batch*n`). Uses `add_matmul_m`
    /// with `m = batch` — the EXACT mechanism the qwen3-coder conc64 forward_batch path uses.
    fn rec_matmul_b(
        &mut self,
        cp: &mut CommandPass,
        x: Act,
        w: &GpuMatWeight,
        out: Act,
        n: usize,
        k: usize,
    ) {
        debug_assert_ne!(
            x.buf_idx, out.buf_idx,
            "matmul in/out must be distinct buffers"
        );
        let xr = act_binding(&self.act_pool[x.buf_idx], x);
        let outr = act_binding(&self.act_pool[out.buf_idx], out);
        super::dispatch::add_matmul_m(
            cp,
            &self.kernels,
            &mut self.uni,
            self.batch,
            xr,
            w,
            outr,
            n,
            k,
            None,
        );
    }

    /// **B-PARALLEL** — record the whole gated-delta-net recurrence onto `cp` for `batch` lockstep
    /// sequences (the concurrency advantage: each seq carries its own O(1) conv_state/ssm_state, b-outer). The
    /// structural twin of [`rec_gdn_chain`] with the SIX `_b` kernels. The l2_norm is the one place
    /// that can't use a flat `_b` kernel (the q/k sub-regions of a b-outer conv_out have a `conv_dim`
    /// stride, not `key_dim`), so it runs the single-stream `gdn_l2_norm` per (seq, region) via an
    /// offset binding — proven byte-identical (gdn_l2_offset_probe). `qkv`/`beta_raw`/`alpha`/`out`
    /// are b-outer Acts of physical length `batch * {conv_dim, nvh, nvh, value_dim}`.
    fn rec_gdn_chain_b(
        &mut self,
        cp: &mut CommandPass,
        il: usize,
        geom: LinearAttnGeom,
        qkv: Act,
        beta_raw: Act,
        alpha: Act,
        out: Act,
        eps: f64,
    ) {
        let conv_dim = geom.conv_dim;
        let key_dim = geom.key_dim;
        let value_dim = geom.value_dim;
        let s = geom.head_k_dim;
        let nkh = geom.num_k_heads;
        let nvh = geom.num_v_heads;
        let d_conv = geom.d_conv;
        let b = self.batch;

        // b-outer transient intermediates.
        let beta = self.act_alloc_b(nvh);
        let g_decay = self.act_alloc_b(nvh);
        let conv_out = self.act_alloc_b(conv_dim);
        let q_rep = self.act_alloc_b(value_dim);
        let k_rep = self.act_alloc_b(value_dim);
        let v_rep = self.act_alloc_b(value_dim);

        let (conv_state, ssm_state, conv_kernel, ssm_dt, ssm_a) = {
            let r = self
                .gdn_res
                .get(&il)
                .expect("gdn resident state (ensure_gdn_resident)");
            (
                r.conv_state.clone(),
                r.ssm_state.clone(),
                r.conv_kernel.clone(),
                r.ssm_dt.clone(),
                r.ssm_a.clone(),
            )
        };

        // (0a) beta = sigmoid(beta_raw): flat over B*nvh (dims.n = TOTAL).
        {
            let total = b * nvh;
            let dd = self
                .uni
                .alloc_write(&[total as u32, 0u32, 0, 0])
                .expect("sig_b dims");
            cp.add_bound(
                &self.kernels.gdn_sigmoid_out_b,
                "qwen35_gdn_sigmoid_b",
                &[
                    act_binding(&self.act_pool[beta.buf_idx], beta),
                    act_binding(&self.act_pool[beta_raw.buf_idx], beta_raw),
                    self.uni.resource(&dd),
                ],
                [(total as u32).div_ceil(256), 1, 1],
            );
        }
        // (0b) g_decay = softplus(alpha+ssm_dt)*ssm_a: dims {n_per_seq=nvh, batch, _, _}; ssm_dt/a shared.
        {
            let dd = self
                .uni
                .alloc_write(&[nvh as u32, b as u32, 0, 0])
                .expect("gdec_b dims");
            cp.add_bound(
                &self.kernels.gdn_g_decay_b,
                "qwen35_gdn_g_decay_b",
                &[
                    act_binding(&self.act_pool[g_decay.buf_idx], g_decay),
                    act_binding(&self.act_pool[alpha.buf_idx], alpha),
                    ssm_dt.as_entire_binding(),
                    ssm_a.as_entire_binding(),
                    self.uni.resource(&dd),
                ],
                [((b * nvh) as u32).div_ceil(256), 1, 1],
            );
        }
        // (1) conv: b-outer conv_in/state/out. dims {conv_dim, d_conv, batch, _}. grid B*conv_dim.
        {
            let dd = self
                .uni
                .alloc_write(&[conv_dim as u32, d_conv as u32, b as u32, 0])
                .expect("conv_b dims");
            cp.add_bound(
                &self.kernels.gdn_conv1d_b,
                "qwen35_gdn_conv1d_b",
                &[
                    act_binding(&self.act_pool[qkv.buf_idx], qkv),
                    conv_kernel.as_entire_binding(),
                    conv_state.as_entire_binding(),
                    act_binding(&self.act_pool[conv_out.buf_idx], conv_out),
                    self.uni.resource(&dd),
                ],
                [((b * conv_dim) as u32).div_ceil(256), 1, 1],
            );
        }
        // (2) l2_norm per (seq, region) via the SINGLE-STREAM kernel with an offset binding. The
        //     q/k sub-regions of a b-outer conv_out have a `conv_dim` inter-seq stride (not key_dim),
        //     so the flat `_b` kernel can't address them; per-(b,region) offset bindings reproduce
        //     the single-stream l2 exactly (proven byte-identical — gdn_l2_offset_probe).
        for seq in 0..b {
            for region_off in [0usize, key_dim] {
                let dd = self
                    .uni
                    .alloc_write(&[s as u32, nkh as u32, (eps as f32).to_bits(), 0])
                    .expect("l2_b dims");
                let byte_off = ((seq * conv_dim + region_off) * 4) as u64;
                let xr = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.act_pool[conv_out.buf_idx],
                    offset: byte_off,
                    size: wgpu::BufferSize::new((key_dim * 4) as u64),
                });
                cp.add_bound(
                    &self.kernels.gdn_l2_norm,
                    "qwen35_gdn_l2_norm_b",
                    &[xr, self.uni.resource(&dd)],
                    [nkh as u32, 1, 1],
                );
            }
        }
        // (3) tile: split + repeat per seq. dims {s,nkh,nvh,key_dim} + dims2 {conv_dim,batch,_,_}.
        {
            let dd = self
                .uni
                .alloc_write(&[s as u32, nkh as u32, nvh as u32, key_dim as u32])
                .expect("tile_b dims");
            let dd2 = self
                .uni
                .alloc_write(&[conv_dim as u32, b as u32, 0, 0])
                .expect("tile_b dims2");
            cp.add_bound(
                &self.kernels.gdn_tile_b,
                "qwen35_gdn_tile_b",
                &[
                    act_binding(&self.act_pool[conv_out.buf_idx], conv_out),
                    act_binding(&self.act_pool[q_rep.buf_idx], q_rep),
                    act_binding(&self.act_pool[k_rep.buf_idx], k_rep),
                    act_binding(&self.act_pool[v_rep.buf_idx], v_rep),
                    self.uni.resource(&dd),
                    self.uni.resource(&dd2),
                ],
                [((b * value_dim) as u32).div_ceil(256), 1, 1],
            );
        }
        // (4) recurrence: b-outer q/k/v/g/beta/state -> out. dims {S_v,S_k,nvh,batch}. grid B*nvh.
        {
            let dd = self
                .uni
                .alloc_write(&[s as u32, s as u32, nvh as u32, b as u32])
                .expect("gdn_b dims");
            cp.add_bound(
                &self.kernels.gdn_recurrence_b,
                "qwen35_gdn_recurrence_b",
                &[
                    act_binding(&self.act_pool[q_rep.buf_idx], q_rep),
                    act_binding(&self.act_pool[k_rep.buf_idx], k_rep),
                    act_binding(&self.act_pool[v_rep.buf_idx], v_rep),
                    act_binding(&self.act_pool[g_decay.buf_idx], g_decay),
                    act_binding(&self.act_pool[beta.buf_idx], beta),
                    ssm_state.as_entire_binding(),
                    act_binding(&self.act_pool[out.buf_idx], out),
                    self.uni.resource(&dd),
                ],
                [(b * nvh) as u32, 1, 1],
            );
        }
    }

    /// **M10 STEP 1** — submit `cp` and BLOCK until the GPU has finished it (the single sync
    /// point that makes Pass A's writes to the dual-view feed buffers visible to the island
    /// queue). No host copy, no map_async, no staging — just submit + `poll(wait_indefinitely)`.
    /// Replaces [`submit_readback`] on the linear-layer path: the island now reads qkv/beta/alpha
    /// from the shared buffers directly instead of from a host `Vec` ferried by the readback.
    fn submit_and_poll(&self, cp: CommandPass) {
        cp.submit();
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Reset the per-LAYER scratch: the activation pool cursor AND the uniform arena (the
    /// residual `hidden_buf` survives). Each layer's two passes both submit+complete before the
    /// NEXT layer's `begin_layer`, so the uniforms (a few hundred bytes) recycle per layer
    /// instead of accumulating across 64 layers (which overflowed the 64 KiB uniform arena).
    fn begin_layer(&mut self) {
        self.act_cursor = 0;
        self.uni.reset();
    }

    /// Seed the resident residual stream with this token's embedding row (host upload, once/token).
    fn seed_hidden(&self, row: &[f32]) {
        debug_assert_eq!(row.len(), self.n_embd);
        self.ctx.write_at(&self.hidden_buf, 0, row);
    }

    /// Record `out_act = rmsnorm(hidden_buf, weight)` (whole `[n_embd]` row) onto `cp`.
    fn rec_rmsnorm_from_hidden(
        &mut self,
        cp: &mut CommandPass,
        weight: &wgpu::Buffer,
        out: Act,
        eps: f64,
    ) {
        let nd = self
            .uni
            .alloc_write(&dims_norm(self.n_embd, eps))
            .expect("rmsnorm dims");
        let hr = self.hidden_buf.as_entire_binding();
        let outr = act_binding(&self.act_pool[out.buf_idx], out);
        let und = self.uni.resource(&nd);
        cp.add_bound(
            &self.kernels.rmsnorm,
            "qwen35_rmsnorm_h",
            &[hr, weight.as_entire_binding(), outr, und],
            [1, 1, 1],
        );
    }

    /// Record in-place residual add `hidden_buf[i] += b_act[i]` over `[n_embd]` onto `cp`.
    fn rec_residual_into_hidden(&mut self, cp: &mut CommandPass, b: Act) {
        let d = self
            .uni
            .alloc_write(&[self.n_embd as u32, 0u32, 0, 0])
            .expect("add dims");
        let hr = self.hidden_buf.as_entire_binding();
        let br = act_binding(&self.act_pool[b.buf_idx], b);
        let und = self.uni.resource(&d);
        cp.add_bound(
            &self.kernels.add,
            "qwen35_add_h",
            &[hr, br, und],
            [(self.n_embd as u32).div_ceil(256), 1, 1],
        );
    }

    /// Read the resident residual stream back to host (end of the trunk, before lm_head; and for
    /// the prefill `last_hidden` that seeds the first argmax).
    fn readback_hidden(&mut self) -> Vec<f32> {
        let bytes = (self.n_embd * 4) as u64;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.hidden_buf, 0, &self.rb_staging, 0, bytes);
        self.ctx.queue.submit(std::iter::once(enc.finish()));
        let slice = self.rb_staging.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let out: Vec<f32> = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };
        self.rb_staging.unmap();
        out
    }

    /// `y[n] = W[n,k] · x[k]` against a RESIDENT `GpuMatWeight` (Q4_K_S → `matmul_vec_q4ks`,
    /// bf16 → `matmul`), on the GPU. `x` is uploaded; `y` is read back. One submit + poll.
    pub fn gemv(&mut self, x: &[f32], w: &GpuMatWeight, n: usize, k: usize) -> Vec<f32> {
        let mut out = self.gemv_shared(x, k, &[(w, n)]);
        out.pop().unwrap()
    }

    /// Batch SEVERAL GEMVs that share the SAME input `x[k]` into ONE CommandPass + ONE
    /// readback — the per-token submit/poll cost (the M6 bottleneck once the weights are
    /// resident) is paid once instead of per projection. Each `(weight, n)` writes a distinct
    /// 256-byte-aligned region of the shared output buffer; the whole span is read back once
    /// and split. Returns one `Vec<f32>` per `(weight, n)`, in order.
    pub fn gemv_shared(
        &mut self,
        x: &[f32],
        k: usize,
        ws: &[(&GpuMatWeight, usize)],
    ) -> Vec<Vec<f32>> {
        assert_eq!(x.len(), k, "qwen35 gemv: x len {} != k {k}", x.len());
        assert!(k <= self.max_k, "qwen35 gemv: k {k} > max_k {}", self.max_k);
        // Output regions: each n floats, padded to a 256-byte (64-f32) granule so the next
        // region's binding offset is wgpu-storage-aligned (macOS hardcodes 256B offset align).
        let mut offsets = Vec::with_capacity(ws.len());
        let mut cursor = 0usize; // in f32
        for &(_, n) in ws {
            offsets.push(cursor);
            cursor += n.next_multiple_of(64);
        }
        let total = cursor.max(1);
        assert!(
            total <= self.max_n,
            "qwen35 gemv: combined out {total} > max_n {}",
            self.max_n
        );

        let _t = if timers::on() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        self.ctx
            .queue
            .write_buffer(&self.a_buf, 0, bytemuck::cast_slice(x));
        self.uni.reset();
        let mut cp = CommandPass::new(&self.ctx);
        let a_bind = self.a_buf.as_entire_binding();
        for (i, &(w, n)) in ws.iter().enumerate() {
            let byte_off = (offsets[i] * 4) as u64;
            let size = std::num::NonZeroU64::new((n * 4) as u64).unwrap();
            let c_bind = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.c_buf,
                offset: byte_off,
                size: Some(size),
            });
            add_matmul(
                &mut cp,
                &self.kernels,
                &mut self.uni,
                a_bind.clone(),
                w,
                c_bind,
                n,
                k,
                None,
            );
        }
        cp.submit();
        // ONE readback of the whole span into the PERSISTENT staging buffer (no per-call
        // allocation), then split per region.
        let bytes = (total * 4) as u64;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.c_buf, 0, &self.staging, 0, bytes);
        self.ctx.queue.submit(std::iter::once(enc.finish()));
        let slice = self.staging.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let all: Vec<f32> = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };
        self.staging.unmap();
        if let Some(t) = _t {
            timers::add_gemv(t.elapsed().as_micros() as u64);
        }
        ws.iter()
            .enumerate()
            .map(|(i, &(_, n))| all[offsets[i]..offsets[i] + n].to_vec())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// M8 — full-GQA attention ON THE GPU (the 65% host/other leak)
//
// The 16 full-GQA layers ran their per-head q/k RMSNorm, RoPE, KV-cache append (a host
// Vec), and the softmax attention ENTIRELY on the host CPU (×16/token = ~65% of the frame,
// growing with ctx). M8 moves that whole inner chain onto the GPU, reusing the EXACT decode
// kernels (`qk_norm`, `rope_qk`, `kv_scatter`, `attention`) the qwen3-coder path uses — no
// new attention kernel. The only host work left per GQA layer is the trivial fixed-size
// Q|gate de-interleave + sigmoid-gate multiply (O(nh·hd)=6144 floats, ctx-INVARIANT).
//
// KV cache moves from the per-layer host `Vec<f32>` (`extend_from_slice` each token) to a
// per-layer RESIDENT GPU pool [max_ctx × kv_dim]; the new k/v is scattered at slot = pos and
// `attention` reads slots[0..=pos] in place (the FlashInfer-style in-kernel gather).
// ---------------------------------------------------------------------------

/// De-interleave the packed q-projection: `q_full` is head-major `[nh, 2·hd]` where each head
/// stores Q in `[0..hd]` and the GATE in `[hd..2hd]`. This writes the contiguous Q `[nh, hd]`
/// and GATE `[nh, hd]` into two distinct output buffers (one lane per (head, i) pair). Pure
/// strided copy — NOT an attention kernel.
const QGATE_SPLIT_WGSL: &str = r#"
struct Dims { n_head: u32, head_dim: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       q_full: array<f32>;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> gate: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = d.n_head * d.head_dim;
    let idx = gid.x;
    if (idx >= total) { return; }
    let h = idx / d.head_dim;
    let i = idx % d.head_dim;
    let src = h * 2u * d.head_dim + i;
    q[idx] = q_full[src];
    gate[idx] = q_full[src + d.head_dim];
}
"#;

/// Multiply the attention output by `sigmoid(gate)` in place, over `o_in` elements (the
/// beast's per-head output gate, applied AFTER attention, BEFORE o_proj).
const GATE_SIGMOID_WGSL: &str = r#"
struct Dims { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read_write> attn: array<f32>;
@group(0) @binding(1) var<storage, read>       gate: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= d.n) { return; }
    let g = gate[idx];
    attn[idx] = attn[idx] * (1.0 / (1.0 + exp(-g)));
}
"#;

/// One full-GQA layer's RESIDENT GPU attention state: the growing KV pool + the per-head
/// q/k RMSNorm gains (uploaded once). All sized to the model's GQA geometry.
#[cfg(target_os = "macos")]
struct GqaLayerGpu {
    /// `[max_ctx · kv_dim]` f32 — keys scattered at slot = pos each token.
    k_pool: wgpu::Buffer,
    /// `[max_ctx · kv_dim]` f32 — values scattered at slot = pos each token.
    v_pool: wgpu::Buffer,
    /// `[head_dim]` f32 — `attn_q_norm.weight` (resident; was host `hw.gqa.0`).
    q_norm: wgpu::Buffer,
    /// `[head_dim]` f32 — `attn_k_norm.weight` (resident; was host `hw.gqa.1`).
    k_norm: wgpu::Buffer,
}

/// All resident GPU state for the M8 full-GQA attention path, built ONCE in
/// `generate_qwen35`. Shared rope tables + per-token index buffers + the small split/gate
/// kernels; per-trunk-layer KV pools + q/k norms (`None` on linear layers).
#[cfg(target_os = "macos")]
pub struct Qwen35GqaGpu {
    /// Per-trunk-layer attention state (`Some` on full-GQA layers, `None` on linear).
    layers: Vec<Option<GqaLayerGpu>>,
    /// `[max_ctx · rot_half]` f32 cos/sin tables (rot_half = rope_dim/2). Shared by every
    /// GQA layer (single rope base on this model).
    rope_cos: wgpu::Buffer,
    rope_sin: wgpu::Buffer,
    /// INDEX-usage arena for the per-token `positions`/`new_slot`/`slots` u32 buffers.
    idx: GpuArena,
    /// Q|gate de-interleave kernel.
    split: crate::gpu::command::ComputeKernel,
    /// sigmoid-gate multiply kernel.
    gate_sig: crate::gpu::command::ComputeKernel,
    /// Reusable q/k/v/attn scratch sized to the GQA shapes (kept distinct from the act pool so
    /// qk_norm/rope/scatter never alias the matmul's in/out buffers).
    q_buf: wgpu::Buffer, // [nh · hd]
    gate_buf: wgpu::Buffer, // [nh · hd]
    // k/v scratch: allocated for shape symmetry with q_buf; the live GQA path binds k/v via the
    // act pool (see `gpu_gqa_attn`), so these struct handles are unread. Kept so the struct owns
    // one buffer per GQA tensor.
    #[allow(dead_code)]
    k_buf: wgpu::Buffer, // [nkv · hd]
    #[allow(dead_code)]
    v_buf: wgpu::Buffer, // [nkv · hd]
    attn_buf: wgpu::Buffer, // [nh · hd]
    max_ctx: usize,
}

#[cfg(target_os = "macos")]
impl Qwen35GqaGpu {
    /// Build all resident GQA GPU state. `max_ctx` bounds the KV pool / rope tables (prompt +
    /// generation length). Reads the per-layer q/k norm weights from the GGUF (once).
    pub fn new(
        ctx: &Arc<GpuContext>,
        g: &LazyGguf,
        hp: &Qwen35Hparams,
        max_ctx: usize,
    ) -> Result<Self> {
        let hd = hp.n_embd_head_k;
        let nh = hp.n_head;
        let nkv = hp.n_head_kv;
        let kv_dim = nkv * hd;
        let n_trunk = hp.n_layer.saturating_sub(hp.n_layer_nextn);

        // Rope tables: theta = rope_theta, PARTIAL rotary over rope_dim of head_dim. Exactly the
        // host `freq = 1/theta^(2i/rope_dim)` the old loop used; table stride = rope_dim/2.
        let rope =
            arf_core::model::rope::Rope::with_theta_rotary(max_ctx, hp.rope_theta, hd, hp.rope_dim);
        let rope_cos = upload_storage(ctx, "qwen35_gqa_rope_cos", rope.cos_table());
        let rope_sin = upload_storage(ctx, "qwen35_gqa_rope_sin", rope.sin_table());

        let mut layers = Vec::with_capacity(n_trunk);
        for il in 0..n_trunk {
            if is_recurrent(il, hp.full_attention_interval) {
                layers.push(None);
                continue;
            }
            let p = format!("blk.{il}");
            let qn = g
                .get_f32_raw(&format!("{p}.attn_q_norm.weight"))
                .ok_or_else(|| ArfError::model_load(format!("missing {p}.attn_q_norm.weight")))?;
            let kn = g
                .get_f32_raw(&format!("{p}.attn_k_norm.weight"))
                .ok_or_else(|| ArfError::model_load(format!("missing {p}.attn_k_norm.weight")))?;
            let k_pool = ctx.create_buffer(
                &format!("qwen35_gqa_kpool_{il}"),
                (max_ctx * kv_dim * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            let v_pool = ctx.create_buffer(
                &format!("qwen35_gqa_vpool_{il}"),
                (max_ctx * kv_dim * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            layers.push(Some(GqaLayerGpu {
                k_pool,
                v_pool,
                q_norm: upload_storage(ctx, &format!("qwen35_gqa_qn_{il}"), &qn),
                k_norm: upload_storage(ctx, &format!("qwen35_gqa_kn_{il}"), &kn),
            }));
        }

        // INDEX arena for positions/new_slot/slots (≤ 3 small allocations per token; sized for
        // slots = [0..max_ctx] u32 plus the two single-element buffers, padded per 256 B).
        let idx = GpuArena::new(
            ctx,
            "qwen35_gqa_idx",
            ((max_ctx + 2) * 4 * 4).max(4096) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;

        let mk = |label: &str, n: usize| {
            ctx.create_buffer(
                label,
                (n * 4).max(16) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            )
        };
        Ok(Qwen35GqaGpu {
            layers,
            rope_cos,
            rope_sin,
            idx,
            split: crate::gpu::command::ComputeKernel::new(ctx, "qgate_split", QGATE_SPLIT_WGSL),
            gate_sig: crate::gpu::command::ComputeKernel::new(
                ctx,
                "gate_sigmoid",
                GATE_SIGMOID_WGSL,
            ),
            q_buf: mk("qwen35_gqa_q", nh * hd),
            gate_buf: mk("qwen35_gqa_gate", nh * hd),
            k_buf: mk("qwen35_gqa_k", nkv * hd),
            v_buf: mk("qwen35_gqa_v", nkv * hd),
            attn_buf: mk("qwen35_gqa_attn", nh * hd),
            max_ctx,
        })
    }
}

/// Upload an f32 slice into a fresh resident STORAGE buffer.
#[cfg(target_os = "macos")]
fn upload_storage(ctx: &Arc<GpuContext>, label: &str, data: &[f32]) -> wgpu::Buffer {
    let buf = ctx.create_buffer(
        label,
        (data.len() * 4).max(16) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    ctx.write_at(&buf, 0, data);
    buf
}

// ---------------------------------------------------------------------------
// 1) Hyper-parameters + derived geometry
// ---------------------------------------------------------------------------

/// Qwen3.6-27B (`qwen35`) hyper-parameters: the standard transformer hparams plus the
/// SSM / gated-delta-net parameters read from the GGUF `LLM_KV_SSM_*` keys.
///
/// The beast's measured values (block_count = 65, n_layer_nextn = 1):
///   embedding_length 5120, feed_forward_length 17408, head_count 24, head_count_kv 4,
///   key/value_length 256, rms_eps 1e-6, full_attention_interval 4.
/// The SSM params (`ssm_d_conv`, `ssm_d_inner`, `ssm_d_state`, `ssm_dt_rank`, `ssm_n_group`)
/// come from metadata and are read at load (see [`load_qwen35`]).
#[derive(Debug, Clone, Copy)]
pub struct Qwen35Hparams {
    // ---- standard transformer hparams ----
    pub n_embd: usize,        // embedding_length            (5120)
    pub n_ff: usize,          // feed_forward_length         (17408)
    pub n_head: usize,        // head_count                  (24)
    pub n_head_kv: usize,     // head_count_kv               (4)
    pub n_embd_head_k: usize, // key_length                  (256)
    pub n_embd_head_v: usize, // value_length                (256)
    pub n_layer: usize,       // block_count                 (65) — trunk layers
    pub n_layer_nextn: usize, // nextn_predict_layers        (1)
    pub n_vocab: usize,
    pub rms_eps: f64,    // attention.layer_norm_rms_epsilon  (1e-6)
    pub rope_dim: usize, // rope.dimension_count (partial rotary; 64 for the beast)
    pub rope_theta: f64, // rope.freq_base (1e7)

    // ---- MoE FFN ----
    pub n_expert: usize,      // expert_count
    pub n_expert_used: usize, // expert_used_count (top_k)
    pub n_ff_exp: usize,      // expert_feed_forward_length (routed expert width)
    pub n_ff_shexp: usize,    // expert_shared_feed_forward_length (shared expert width)

    // ---- hybrid routing ----
    pub full_attention_interval: u32, // every Nth layer is FULL attention (4)

    // ---- SSM / gated-delta-net params (LLM_KV_SSM_*) ----
    pub ssm_d_conv: usize,  // SSM_CONV_KERNEL   — depthwise conv kernel width
    pub ssm_d_inner: usize, // SSM_INNER_SIZE    — the value channel total (d_inner)
    pub ssm_d_state: usize, // SSM_STATE_SIZE    — per-(k/v)-head dim = head_k_dim
    pub ssm_dt_rank: usize, // SSM_TIME_STEP_RANK — = num_v_heads
    pub ssm_n_group: usize, // SSM_GROUP_COUNT   — = num_k_heads
}

/// The derived gated-delta-net geometry (qwen35moe.cpp:366-371 / 64-70). All shapes the
/// linear-attention tensors and kernels are sized against come from here.
#[derive(Debug, Clone, Copy)]
pub struct LinearAttnGeom {
    pub head_k_dim: usize,  // = ssm_d_state
    pub head_v_dim: usize,  // = ssm_d_inner / num_v_heads
    pub num_k_heads: usize, // = ssm_n_group
    pub num_v_heads: usize, // = ssm_dt_rank
    pub key_dim: usize,     // = head_k_dim * num_k_heads
    pub value_dim: usize,   // = head_v_dim * num_v_heads
    pub conv_dim: usize,    // = key_dim*2 + value_dim  (the depthwise-conv channel count)
    pub d_conv: usize,      // = ssm_d_conv  (conv kernel width; conv_state has d_conv-1 cols)
}

impl Qwen35Hparams {
    /// Derive the gated-delta-net geometry from the SSM params (qwen35moe.cpp:366-371).
    pub fn linear_geom(&self) -> LinearAttnGeom {
        let head_k_dim = self.ssm_d_state;
        let num_k_heads = self.ssm_n_group;
        let num_v_heads = self.ssm_dt_rank;
        // head_v_dim = d_inner / num_v_heads  (line 371). num_v_heads divides d_inner.
        let head_v_dim = self.ssm_d_inner / num_v_heads.max(1);
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        // conv_channels = d_inner + 2*ssm_n_group*ssm_d_state == key_dim*2 + value_dim (line 70/408).
        let conv_dim = key_dim * 2 + value_dim;
        LinearAttnGeom {
            head_k_dim,
            head_v_dim,
            num_k_heads,
            num_v_heads,
            key_dim,
            value_dim,
            conv_dim,
            d_conv: self.ssm_d_conv,
        }
    }
}

// ---------------------------------------------------------------------------
// 2) Hybrid routing predicate
// ---------------------------------------------------------------------------

/// The hybrid routing predicate (qwen35moe.cpp:28): is trunk layer `il` a LINEAR
/// (gated-delta-net / recurrent) layer, or a FULL GQA layer?
///
/// `(il + 1) % full_attention_interval != 0` ⇒ recurrent (linear). With interval 4:
/// layers 0,1,2 = linear, 3 = full, 4,5,6 = linear, 7 = full, … (3-of-4 are linear).
/// The caller is responsible for `il < n_layer` (nextn/MTP blocks are always non-recurrent).
#[inline]
pub fn is_recurrent(il: usize, full_attention_interval: u32) -> bool {
    let interval = full_attention_interval.max(1) as usize;
    !(il + 1).is_multiple_of(interval)
}

// ---------------------------------------------------------------------------
// 3) Per-layer tensor handles
// ---------------------------------------------------------------------------

/// One LINEAR (gated-delta-net) layer's resident tensor handles (qwen35moe.cpp:84-94).
///
/// All large projections are Q4_K_S (consumed by `add_matmul` → `matmul_vec_q4ks`); the
/// small per-head vectors (`ssm_dt`, `ssm_a`, `ssm_norm`) and the conv kernel are f32
/// (uploaded via `GpuContext::storage_init`). The MoE FFN is shared with the full-attn
/// layers and lives in `mlp` (a [`GpuMatWeight`]-backed [`super::types::GpuMlp::Moe`] with
/// the gated SHARED expert).
pub struct LinearAttnLayer {
    // ---- pre/post norm (shared skeleton with full layers) ----
    /// `attn_norm.weight` `n_embd` — RMSNorm gain before the (linear) attention.
    pub attn_norm: wgpu::Buffer,
    /// `post_attention_norm.weight` `n_embd` — RMSNorm gain before the FFN (GGUF spells this
    /// `post_attention_norm`, NOT `attn_post_norm`; ground-truthed against the beast).
    pub attn_post_norm: wgpu::Buffer,

    // ---- gated-delta-net projections (all Q4_K_S) ----
    /// `wqkv` [n_embd → key_dim*2 + value_dim] — packed q‖k‖v mixed projection.
    pub wqkv: GpuMatWeight,
    /// `wqkv_gate` (attn_gate) [n_embd → value_dim] — the output gate `z`.
    pub wqkv_gate: GpuMatWeight,
    /// `ssm_beta` [n_embd → num_v_heads] — pre-sigmoid beta projection.
    pub ssm_beta: GpuMatWeight,
    /// `ssm_alpha` [n_embd → num_v_heads] — pre-(softplus) alpha projection.
    pub ssm_alpha: GpuMatWeight,
    /// `ssm_out` [value_dim → n_embd] — the output projection.
    pub ssm_out: GpuMatWeight,

    // ---- small f32 per-layer vectors / conv kernel ----
    /// `ssm_conv1d.weight` [d_conv, conv_dim] — depthwise causal conv kernel (f32).
    pub ssm_conv1d: wgpu::Buffer,
    /// `ssm_dt.bias` `num_v_heads` — the time-step bias added before softplus (f32).
    pub ssm_dt: wgpu::Buffer,
    /// `ssm_a` (a_noscan) `num_v_heads` — `-A_log.exp()` decay coefficient (f32).
    pub ssm_a: wgpu::Buffer,
    /// `ssm_norm.weight` `head_v_dim` — RMSNorm gain on the per-head readout (f32).
    pub ssm_norm: wgpu::Buffer,

    // ---- FFN — same type the full layer uses ----
    /// The FFN. GROUND-TRUTH (the beast = Qwen3.6-27B DENSE, not the MoE flavor): every trunk
    /// layer carries a plain dense SwiGLU `ffn_gate/ffn_up/ffn_down` — there are ZERO expert /
    /// router tensors in the file. So this is a [`GpuMlp::Dense`], NOT `GpuMlp::Moe` as the
    /// original scaffold assumed. (The `qwen35moe` llama.cpp arch IS MoE; this 27B blob is the
    /// dense sibling — the loader keys off the actual tensor set.)
    pub mlp: GpuMlp,

    // ---- geometry pinned at load ----
    pub geom: LinearAttnGeom,
    pub eps: f64,
}

/// One FULL GQA layer (1/4) — a standard GQA attention + DENSE SwiGLU FFN, held as the trunk
/// [`GpuLayer`] type. (Name kept for scaffold continuity; the FFN is dense, see `mlp` above.)
/// NOTE the geometry differs from qwen3-coder: head_dim 256, n_head 24 (q packs 2× → q_dim
/// = key_length·n_head·2 = 12288), kv_heads 4, plus per-head q/k RMSNorm. The decode attention
/// dispatch (M2) must use THESE per-layer dims, not the coder defaults.
pub struct GqaMoeLayer(pub GpuLayer);

/// A trunk layer of the hybrid stack: either a LINEAR (gated-delta-net) layer or a FULL
/// GQA layer. The vec of these (one per `n_layer`, kind decided by [`is_recurrent`]) is
/// the model's trunk.
pub enum HybridLayer {
    Linear(Box<LinearAttnLayer>),
    Gqa(Box<GqaMoeLayer>),
}

impl HybridLayer {
    pub fn is_linear(&self) -> bool {
        matches!(self, HybridLayer::Linear(_))
    }
}

/// The two carried recurrent states for ONE linear layer at decode time (O(1) memory,
/// no growing KV cache). Allocated once at load, advanced in place every token.
pub struct LinearAttnState {
    /// `conv_state` [conv_dim, d_conv-1] — the depthwise-conv ring buffer (f32). Holds the
    /// previous `d_conv-1` token columns so the causal conv has its left context.
    pub conv_state: wgpu::Buffer,
    /// `ssm_state` [head_v_dim, head_v_dim, num_v_heads] — the gated-delta-net matrix state
    /// (f32). Updated by the delta-rule recurrence each token.
    pub ssm_state: wgpu::Buffer,
}

// ---------------------------------------------------------------------------
// 4) Loader signature
// ---------------------------------------------------------------------------

/// The fully-loaded hybrid trunk (returned by [`load_qwen35`]).
pub struct Qwen35Trunk {
    pub hparams: Qwen35Hparams,
    /// `n_layer` trunk layers, each Linear or Gqa per [`is_recurrent`].
    pub layers: Vec<HybridLayer>,
    /// Per-linear-layer recurrent state (indexed by the SAME `il`; `None` for full layers).
    pub states: Vec<Option<LinearAttnState>>,
    /// `output.weight` (or tied `token_embd.weight`) — the lm_head, resident (M6: the
    /// biggest single per-token GEMV runs on the GPU, not host).
    pub lm_head: GpuMatWeight,
}

/// Read the `qwen35.*` + `qwen35.ssm.*` hyper-parameters from the GGUF metadata. Shared by
/// the GPU loader and the CPU reference forward (so both agree on the geometry).
pub fn read_hparams(g: &LazyGguf, cfg: &arf_core::config::ModelConfig) -> Result<Qwen35Hparams> {
    let mu = |k: &str| -> Result<usize> {
        g.get_metadata_u32(k)
            .map(|v| v as usize)
            .ok_or_else(|| ArfError::model_load(format!("qwen35: missing metadata key {k}")))
    };
    let rms_eps = g
        .get_metadata_f32("qwen35.attention.layer_norm_rms_epsilon")
        .unwrap_or(1e-6) as f64;
    // n_vocab comes from the token_embd / output tensor (the metadata has no vocab key here);
    // shape_raw gives ggml dims [n_embd, n_vocab].
    let n_vocab = g
        .shape_raw("token_embd.weight")
        .and_then(|d| d.get(1).copied())
        .or_else(|| g.shape_raw("output.weight").and_then(|d| d.get(1).copied()))
        .unwrap_or(cfg.vocab_size);

    Ok(Qwen35Hparams {
        n_embd: mu("qwen35.embedding_length")?,
        n_ff: mu("qwen35.feed_forward_length")?,
        n_head: mu("qwen35.attention.head_count")?,
        n_head_kv: mu("qwen35.attention.head_count_kv")?,
        n_embd_head_k: mu("qwen35.attention.key_length")?,
        n_embd_head_v: mu("qwen35.attention.value_length")?,
        n_layer: mu("qwen35.block_count")?,
        n_layer_nextn: g
            .get_metadata_u32("qwen35.nextn_predict_layers")
            .unwrap_or(0) as usize,
        n_vocab,
        rms_eps,
        rope_dim: g
            .get_metadata_u32("qwen35.rope.dimension_count")
            .map(|v| v as usize)
            .unwrap_or(64),
        rope_theta: g.get_metadata_f32("qwen35.rope.freq_base").unwrap_or(1e7) as f64,
        // DENSE model: no expert metadata present. Kept zero so the struct is consistent;
        // the FFN is dense (ffn_gate/up/down) per the actual tensor set.
        n_expert: 0,
        n_expert_used: 0,
        n_ff_exp: 0,
        n_ff_shexp: 0,
        full_attention_interval: g
            .get_metadata_u32("qwen35.full_attention_interval")
            .unwrap_or(4),
        ssm_d_conv: mu("qwen35.ssm.conv_kernel")?,
        ssm_d_inner: mu("qwen35.ssm.inner_size")?,
        ssm_d_state: mu("qwen35.ssm.state_size")?,
        ssm_dt_rank: mu("qwen35.ssm.time_step_rank")?,
        ssm_n_group: mu("qwen35.ssm.group_count")?,
    })
}

/// Load the Qwen3.6-27B (`qwen35`) hybrid trunk from the model GGUF.
///
/// Reads the `qwen35.*` standard hparams + the `LLM_KV_SSM_*` / `full_attention_interval`
/// keys, then builds the per-layer vec: for each `il in 0..n_layer`, a [`LinearAttnLayer`]
/// when [`is_recurrent`] else a [`GqaMoeLayer`] (the existing trunk GQA+MoE path).
///
/// Tensor SHAPES are real and correct per qwen35moe.cpp:57-107. The Q4_K_S reads mirror the
/// trunk loader (`weights.rs` → `q4ks_matrix_from_gguf_q4k` → `GpuMatWeight::Q4KS`); the f32
/// vectors / conv kernel mirror the trunk norm-vector path (`f32_vec` →
/// `GpuContext::storage_init`).
pub fn load_qwen35(
    ctx: &Arc<GpuContext>,
    g: &LazyGguf,
    cfg: &arf_core::config::ModelConfig,
) -> Result<Qwen35Trunk> {
    // ---- M-LOAD-0: read hparams + SSM_* metadata ----
    // ALL keys ground-truthed against the beast (Qwen3.6-27B-Q4_K_S.gguf): the standard
    // `qwen35.*` hparams + the `qwen35.ssm.*` keys + `full_attention_interval` +
    // `nextn_predict_layers`. The RMS eps is f32 (`qwen35.attention.layer_norm_rms_epsilon`).
    let hparams = read_hparams(g, cfg)?;

    let geom = hparams.linear_geom();
    // The trunk is the FIRST n_layer blocks. The final n_layer_nextn block(s) (blk.64) are the
    // MTP/nextn head (see the design) — NOT part of the hybrid trunk.
    let n_trunk = hparams.n_layer.saturating_sub(hparams.n_layer_nextn);
    let mut layers: Vec<HybridLayer> = Vec::with_capacity(n_trunk);
    let mut states: Vec<Option<LinearAttnState>> = Vec::with_capacity(n_trunk);

    for il in 0..n_trunk {
        if is_recurrent(il, hparams.full_attention_interval) {
            // ===== LINEAR (gated-delta-net) layer =====
            let layer = load_linear_layer(ctx, g, &hparams, geom, il)?;
            let state = alloc_linear_state(ctx, geom)?;
            layers.push(HybridLayer::Linear(Box::new(layer)));
            states.push(Some(state));
        } else {
            // ===== FULL GQA layer (1/4) — DENSE GQA attn + dense SwiGLU FFN =====
            let layer = load_full_layer(ctx, g, &hparams, il)?;
            layers.push(HybridLayer::Gqa(Box::new(GqaMoeLayer(layer))));
            states.push(None);
        }
    }

    // lm_head: `output.weight` (Q6_K → bf16) if present, else tied `token_embd.weight`
    // (Q4_K → Q4KS). Resident so the per-token lm_head GEMV runs on the GPU (M6).
    let lm_name = if g.shape_raw("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let lm_head = raw_mat(ctx, g, lm_name, hparams.n_vocab, hparams.n_embd)?;

    // Flush the f32/bf16 weight uploads accumulated above (storage_init does NOT note_upload;
    // big bf16/q8 uploads DO — a single end-of-load flush guarantees nothing is dropped).
    ctx.flush_uploads();

    Ok(Qwen35Trunk {
        hparams,
        layers,
        states,
        lm_head,
    })
}

// ---------------------------------------------------------------------------
// Raw-GGUF weight helpers (mirror the FLUX DiT / trunk loader Q4_K_S path)
// ---------------------------------------------------------------------------

/// Build a [`GpuMatWeight`] from a RAW ggml tensor name. The GGUF lists dims `[in, out]`;
/// the caller passes the logical `out`/`in_` (= matmul `[n, k]`). Native Q4_K → Q4_K_S
/// resident (no f32 materialization); any other ggml type (the Q4_K_S mix stores qkv / down
/// / ssm projections in Q5_K/Q6_K) → dequant-to-bf16. Mirrors weights.rs `up_w`.
fn raw_mat(
    ctx: &Arc<GpuContext>,
    g: &LazyGguf,
    ggml_name: &str,
    out: usize,
    in_: usize,
) -> Result<GpuMatWeight> {
    if let Some((blocks, rows, cols)) = g.get_q4k_blocks_raw(ggml_name) {
        // Native Q4_K: rows = out, cols = in (ggml stores [cols=in, rows=out]).
        debug_assert_eq!(rows, out, "{ggml_name}: q4k rows {rows} != out {out}");
        debug_assert_eq!(cols, in_, "{ggml_name}: q4k cols {cols} != in {in_}");
        let q = Q4KSMatrix::from_ggml_q4k(blocks, rows, cols);
        // TEMPORARY EXPERIMENT (2026-09-21), see weights.rs `ARF_TMP_PAIR`.
        let q = match std::env::var("ARF_TMP_PAIR")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            Some(g) if cols.is_multiple_of(256) => {
                Q4KSMatrix::searched_grouped(&q.to_f32(), rows, cols, g)
            }
            _ => q,
        };
        let mut dd = Vec::with_capacity(q.d().len() * 2);
        for (dv, dmv) in q.d().iter().zip(q.dmin()) {
            dd.push(*dv);
            dd.push(*dmv);
        }
        return Ok(GpuMatWeight::Q4KS {
            codes: ctx.storage_init_u32_tracked(&format!("{ggml_name}.codes"), q.codes()),
            scales: ctx.storage_init_u8(&format!("{ggml_name}.scale"), q.scales()),
            mins: ctx.storage_init_q8(&format!("{ggml_name}.min"), q.mins()),
            dd: ctx.storage_init(&format!("{ggml_name}.dd"), &dd),
            #[cfg(target_os = "macos")]
            mtl: None,
        });
    }
    // Non-Q4_K (Q5_K/Q6_K/etc.): dequant to f32 → upload bf16-packed (lossless to the
    // checkpoint's bf16 view; the matmul kernel unpacks it bit-exactly).
    let f = g
        .get_f32_raw(ggml_name)
        .ok_or_else(|| ArfError::model_load(format!("qwen35: missing tensor {ggml_name}")))?;
    if f.len() != out * in_ {
        return Err(ArfError::model_load(format!(
            "qwen35: {ggml_name} len {} != out*in {}*{}",
            f.len(),
            out,
            in_
        )));
    }
    Ok(GpuMatWeight::Bf16(ctx.storage_init_bf16(ggml_name, &f)))
}

/// Upload an f32 vector / small kernel tensor to a resident STORAGE buffer by RAW name.
fn raw_f32(
    ctx: &Arc<GpuContext>,
    g: &LazyGguf,
    ggml_name: &str,
    expect_len: usize,
) -> Result<wgpu::Buffer> {
    let v = g
        .get_f32_raw(ggml_name)
        .ok_or_else(|| ArfError::model_load(format!("qwen35: missing f32 {ggml_name}")))?;
    if expect_len != 0 && v.len() != expect_len {
        return Err(ArfError::model_load(format!(
            "qwen35: {ggml_name} f32 len {} != expected {expect_len}",
            v.len()
        )));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(ArfError::model_load(format!(
            "qwen35: {ggml_name} has non-finite values"
        )));
    }
    Ok(ctx.storage_init(ggml_name, &v))
}

/// Build one DENSE SwiGLU FFN ([`GpuMlp::Dense`]) from `blk.{il}.ffn_{gate,up,down}.weight`.
fn load_dense_ffn(
    ctx: &Arc<GpuContext>,
    g: &LazyGguf,
    hp: &Qwen35Hparams,
    il: usize,
) -> Result<GpuMlp> {
    let p = format!("blk.{il}");
    let n_embd = hp.n_embd;
    let n_ff = hp.n_ff;
    // ggml dims: ffn_gate/up [n_embd, n_ff] → out=n_ff,in=n_embd; ffn_down [n_ff, n_embd].
    // ARF_Q8_DOWN (2026-09-22 experiment): `ffn_down` ships Q6_K in this GGUF and the island's
    // Q4KS view of it is a second quantisation. Build a Q8 view from the same f32 dequant
    // `raw_mat` uses, to measure what the served target's precision does to draft acceptance.
    #[cfg(target_os = "macos")]
    let down_q8 = if std::env::var_os("ARF_Q8_DOWN").is_some() {
        use crate::gpu::concurrent_metal::{Q8Mtl, SharedBuffer};
        let name = format!("{p}.ffn_down.weight");
        let r = g.get_f32_raw(&name).and_then(|f| {
            let q = arf_core::tensor::Q8Matrix::from_f32(&f, n_embd, n_ff);
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
            let data = SharedBuffer::from_pod(ctx, &packed, u, &format!("{name}.q8"))?;
            let scale = SharedBuffer::from_pod(ctx, q.scales(), u, &format!("{name}.q8scale"))?;
            let dims = SharedBuffer::from_pod(
                ctx,
                &[1u32, n_ff as u32, n_embd as u32, 0u32],
                u,
                &format!("{name}.q8dims"),
            )?;
            Some(std::sync::Arc::new(Q8Mtl {
                data: data.mtl,
                scale: scale.mtl,
                dims: dims.mtl,
                n: n_embd,
            }))
        });
        if il == 0 {
            eprintln!(
                "[msl] ARF_Q8_DOWN: ffn_down served at Q8 on the island — {}",
                if r.is_some() { "built" } else { "NOT built" }
            );
        }
        r
    } else {
        None
    };
    Ok(GpuMlp::Dense(Box::new(GpuDenseMlp {
        gate_proj: raw_mat(ctx, g, &format!("{p}.ffn_gate.weight"), n_ff, n_embd)?,
        up_proj: raw_mat(ctx, g, &format!("{p}.ffn_up.weight"), n_ff, n_embd)?,
        down_proj: raw_mat(ctx, g, &format!("{p}.ffn_down.weight"), n_embd, n_ff)?,
        #[cfg(target_os = "macos")]
        q3k: None,
        #[cfg(target_os = "macos")]
        down_q8,
    })))
}

/// Load one FULL GQA layer (dense attention + dense SwiGLU FFN). The beast's full-attn
/// blocks store SEPARATE `attn_q/attn_k/attn_v/attn_output` + per-head `attn_q_norm/attn_k_norm`
/// (NOT a packed qkv), and `attn_norm` / `post_attention_norm`. Geometry (ground-truthed):
/// q_out = key_length·n_head·2 = 12288, kv_out = key_length·n_head_kv = 1024, o_in = 6144.
fn load_full_layer(
    ctx: &Arc<GpuContext>,
    g: &LazyGguf,
    hp: &Qwen35Hparams,
    il: usize,
) -> Result<GpuLayer> {
    let p = format!("blk.{il}");
    let n_embd = hp.n_embd;
    // q packs 2× the key heads (per qwen35moe.cpp:77 create_tensor_qkv with n_embd_head_k*n_head*2).
    let q_out = hp.n_embd_head_k * hp.n_head * 2;
    let kv_out = hp.n_embd_head_k * hp.n_head_kv;
    let o_in = hp.n_embd_head_k * hp.n_head; // = 6144
                                             // head_dim for the attention kernel: q packs to q_out across n_head heads → head_dim = q_out/n_head.
    let head_dim = q_out / hp.n_head; // = 512

    Ok(GpuLayer {
        // Added since this file was written (Aug 6): muse-glimmer's attention gate and the
        // qwen3.8 GDN block. Neither applies to the path this loader builds — it drives the
        // island's own SSM engine — so both are None here.
        attn_gate: None,
        gdn: None,
        q_proj: raw_mat(ctx, g, &format!("{p}.attn_q.weight"), q_out, n_embd)?,
        k_proj: raw_mat(ctx, g, &format!("{p}.attn_k.weight"), kv_out, n_embd)?,
        v_proj: raw_mat(ctx, g, &format!("{p}.attn_v.weight"), kv_out, n_embd)?,
        o_proj: raw_mat(ctx, g, &format!("{p}.attn_output.weight"), n_embd, o_in)?,
        mlp: load_dense_ffn(ctx, g, hp, il)?,
        input_norm: raw_f32(ctx, g, &format!("{p}.attn_norm.weight"), n_embd)?,
        post_norm: raw_f32(ctx, g, &format!("{p}.post_attention_norm.weight"), n_embd)?,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        // per-head q/k RMSNorm over key_length (= 256).
        q_norm: Some(raw_f32(
            ctx,
            g,
            &format!("{p}.attn_q_norm.weight"),
            hp.n_embd_head_k,
        )?),
        k_norm: Some(raw_f32(
            ctx,
            g,
            &format!("{p}.attn_k_norm.weight"),
            hp.n_embd_head_k,
        )?),
        #[cfg(target_os = "macos")]
        norm_mtls: crate::gpu::types::NormMtls::default(),
        window: None,
        head_dim,
        kv_heads: hp.n_head_kv,
        rotary_dim: hp.n_embd_head_k, // RoPE rotates key_length dims (rope.dimension_count=64 caveat at M2)
        layer_scalar: None,
    })
}

/// Load ONE linear (gated-delta-net) layer's tensors (qwen35moe.cpp:84-94): Q4_K_S matrices
/// via `raw_mat`, f32 vectors / conv kernel via `raw_f32`, the dense FFN via `load_dense_ffn`.
fn load_linear_layer(
    ctx: &Arc<GpuContext>,
    g: &LazyGguf,
    hp: &Qwen35Hparams,
    geom: LinearAttnGeom,
    il: usize,
) -> Result<LinearAttnLayer> {
    let n_embd = hp.n_embd;
    let p = format!("blk.{il}");
    // GGUF tensor keys — ALL ground-truthed against the beast (see qwen35_load_probe). Note
    // the spelling fixes vs the original scaffold: `post_attention_norm` (NOT attn_post_norm)
    // and `attn_output` (NOT attn_o). The ggml dim order is [in, out]; raw_mat takes (out, in).
    let num_v_heads = geom.num_v_heads;
    let value_dim = geom.value_dim;
    let conv_dim = geom.conv_dim; // key_dim*2 + value_dim = 10240
    let qkv_out = geom.key_dim * 2 + geom.value_dim; // = conv_dim (10240)

    // ---- projections (native Q4_K → Q4_K_S; Q5_K/Q6_K → bf16, handled in raw_mat) ----
    let wqkv = raw_mat(ctx, g, &format!("{p}.attn_qkv.weight"), qkv_out, n_embd)?;
    let wqkv_gate = raw_mat(ctx, g, &format!("{p}.attn_gate.weight"), value_dim, n_embd)?;
    let ssm_beta = raw_mat(ctx, g, &format!("{p}.ssm_beta.weight"), num_v_heads, n_embd)?;
    let ssm_alpha = raw_mat(
        ctx,
        g,
        &format!("{p}.ssm_alpha.weight"),
        num_v_heads,
        n_embd,
    )?;
    let ssm_out = raw_mat(ctx, g, &format!("{p}.ssm_out.weight"), n_embd, value_dim)?;

    // ---- f32 vectors / conv kernel ----
    let attn_norm = raw_f32(ctx, g, &format!("{p}.attn_norm.weight"), n_embd)?;
    let attn_post_norm = raw_f32(ctx, g, &format!("{p}.post_attention_norm.weight"), n_embd)?;
    // ssm_conv1d is the depthwise kernel [d_conv, conv_dim] (ggml dims [4, 10240]).
    let ssm_conv1d = raw_f32(
        ctx,
        g,
        &format!("{p}.ssm_conv1d.weight"),
        geom.d_conv * conv_dim,
    )?;
    let ssm_dt = raw_f32(ctx, g, &format!("{p}.ssm_dt.bias"), num_v_heads)?;
    let ssm_a = raw_f32(ctx, g, &format!("{p}.ssm_a"), num_v_heads)?;
    let ssm_norm = raw_f32(ctx, g, &format!("{p}.ssm_norm.weight"), geom.head_v_dim)?;

    // ---- FFN — DENSE SwiGLU (the beast has no expert tensors), SAME loader as full layers ----
    let mlp = load_dense_ffn(ctx, g, hp, il)?;

    Ok(LinearAttnLayer {
        attn_norm,
        attn_post_norm,
        wqkv,
        wqkv_gate,
        ssm_beta,
        ssm_alpha,
        ssm_out,
        ssm_conv1d,
        ssm_dt,
        ssm_a,
        ssm_norm,
        mlp,
        geom,
        eps: hp.rms_eps,
    })
}

/// Allocate the two carried recurrent states for one linear layer, zero-initialised.
/// `conv_state` [conv_dim, d_conv-1], `ssm_state` [head_v_dim, head_v_dim, num_v_heads].
fn alloc_linear_state(ctx: &Arc<GpuContext>, geom: LinearAttnGeom) -> Result<LinearAttnState> {
    let conv_cols = geom.d_conv.saturating_sub(1);
    let conv_len = geom.conv_dim * conv_cols;
    let ssm_len = geom.head_v_dim * geom.head_v_dim * geom.num_v_heads;
    // Zero-init f32 state buffers (O(1) recurrent state — no growing KV cache).
    let conv_state = ctx.storage_zeros("qwen35.conv_state", conv_len);
    let ssm_state = ctx.storage_zeros("qwen35.ssm_state", ssm_len);
    Ok(LinearAttnState {
        conv_state,
        ssm_state,
    })
}

// ---------------------------------------------------------------------------
// 7) M2 — the ssm_conv1d dispatch contract (kernel READY; wired at M4)
// ---------------------------------------------------------------------------

/// The bind/grid contract for the `ssm_conv1d` MSL kernel (M2), documented as a callable so M4
/// can wire it without re-deriving the layout. The kernel itself
/// (`shaders/metal/ssm_conv1d_msl.metal`) is compiled in the island (weights.rs) and parity-gated
/// standalone in `examples/ssm_conv1d_probe.rs` (no model).
///
/// At decode (1 new token) the causal depthwise conv1d is:
/// ```text
/// per channel i1 in 0..conv_dim (one GPU thread each):
///   window = [conv_state[i1][0..d_conv-1] (oldest..newest), conv_in[i1] (new, last)]
///   conv_out[i1] = silu( Σ_{i0<d_conv} window[i0] * conv_kernel[i0 + i1*d_conv] )
///   conv_state[i1] := window[1..d_conv]    // drop oldest, append new input (advance ring)
/// ```
/// BINDINGS (raw-Metal encoder, indices 0..): 0 conv_in`conv_dim` (the new token's qkv_mixed),
/// 1 conv_kernel`d_conv*conv_dim` (the f32 weight, tap-contiguous per channel), 2 conv_state
/// `[conv_dim*(d_conv-1)]` (READ+WRITE ring), 3 conv_out`conv_dim` (post-silu output), 4 dims
/// `{conv_dim, d_conv, _, _}` (u32×4). GRID: one thread per channel — threads/tg 256,
/// threadgroups = ceil(conv_dim/256). conv_out feeds the q/k/v split (§6 below), conv_state is
/// advanced in place for the next token.
///
/// M4 will obtain the four data buffers from the linear-layer scratch set (conv_in = the
/// `qkv_mixed` GEMV output scratch; conv_out = a `conv_dim` scratch; conv_kernel =
/// `layer.ssm_conv1d`; conv_state = `state.conv_state`) and dispatch through
/// `model.island` exactly as [`crate::gpu::concurrent_metal::MetalIsland::ssm_conv1d_selftest`]
/// does (compile is already done at load — just `pso("ssm_conv1d")` + encode).
#[cfg(target_os = "macos")]
#[inline]
pub fn ssm_conv1d_grid(conv_dim: usize) -> (u32, u32) {
    // (threadgroups, threads_per_threadgroup) — one thread per channel.
    (conv_dim.div_ceil(256) as u32, 256)
}

// ---------------------------------------------------------------------------
// 8) M4 — CPU REFERENCE hybrid forward (the sane-logits gate)
// ---------------------------------------------------------------------------
//
// This is the M4 deliverable: a faithful, debuggable reference forward that runs the
// WHOLE Qwen3.6-27B hybrid stack (48 gated-delta-net LINEAR layers + 16 full GQA layers
// + dense SwiGLU FFN + lm_head) on the host, reading the real GGUF weights LAZILY
// (dequantised one tensor at a time from the mmap, then dropped — never the whole 16 GB
// in f32 at once). Every op is a bit-faithful CPU mirror of the SAME math the M2/M3 GPU
// kernels (`ssm_conv1d`, `gated_delta_net`, `gdn_l2_norm/softplus/sigmoid`) implement —
// those kernels are already parity-proven STANDALONE (M2/M3 selftests). The point of this
// reference is to prove the FORWARD WIRING (the splits, the k/v-head repeat, the gate, the
// rope-64 partial rotary, the decay sign) produces SANE logits on the real model — the
// thing the GPU-island dispatch wiring would also have to produce, but with far less
// integration risk than threading the recurrent state through the paged-KV decode machine.
//
// Ground truth (the beast, confirmed from the GGUF metadata + qwen35moe.cpp):
//   n_embd 5120, n_head 24, n_head_kv 4, head_dim 256, rope.dimension_count 64,
//   rope.freq_base 1e7, full_attention_interval 4, dense ffn 17408, vocab 248320.
//   ssm: d_state 128 (=head_k_dim), n_group 16 (=num_k_heads), dt_rank 48 (=num_v_heads),
//   d_inner 6144 → head_v_dim 128 (so S_k==S_v==128), d_conv 4.
//   key_dim 2048, value_dim 6144, conv_dim 10240. k/v-head repeat = 48/16 = 3.

/// A GEMV against a GGUF weight read lazily by raw ggml name. ggml dims are `[in, out]`
/// (cols=in fastest), so the dequantised buffer is row-major `W[o*in + i]`; this computes
/// `y[o] = Σ_i W[o,i] * x[i]`. The dequant buffer is dropped on return (peak = one tensor).
fn gguf_gemv(g: &LazyGguf, name: &str, x: &[f32], out: usize, in_: usize) -> Result<Vec<f32>> {
    let w = g
        .get_f32_raw(name)
        .ok_or_else(|| ArfError::model_load(format!("qwen35 cpu-fwd: missing {name}")))?;
    if w.len() != out * in_ {
        return Err(ArfError::model_load(format!(
            "qwen35 cpu-fwd: {name} len {} != {out}*{in_}",
            w.len()
        )));
    }
    debug_assert_eq!(x.len(), in_, "gemv x len for {name}");
    let mut y = vec![0.0f32; out];
    for (o, yo) in y.iter_mut().enumerate() {
        let row = &w[o * in_..o * in_ + in_];
        let mut acc = 0.0f32;
        for i in 0..in_ {
            acc += row[i] * x[i];
        }
        *yo = acc;
    }
    Ok(y)
}

/// RMSNorm with a learned gain over `dim` (Llama-style): `y = x/sqrt(mean(x²)+eps) * w`.
fn rmsnorm(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let n = x.len();
    let ms = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64;
    let inv = (1.0 / (ms + eps).sqrt()) as f32;
    (0..n).map(|i| x[i] * inv * w[i]).collect()
}

#[inline]
fn silu(x: f32) -> f32 {
    let z = x.clamp(-30.0, 30.0);
    x / (1.0 + (-z).exp())
}
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// The carried recurrent state for ONE linear layer in the CPU reference.
pub struct CpuLinearState {
    /// conv ring, `[conv_dim, d_conv-1]` row-major (oldest..newest per channel).
    pub conv_state: Vec<f32>,
    /// delta-net matrix state, `[num_v_heads][S_v*S_v]` flat S[i + j*S_v].
    pub ssm_state: Vec<f32>,
}

/// One LINEAR (gated-delta-net) layer's CPU forward for a single token. Mirrors the M2/M3
/// kernels exactly. `cur` is the post-attn_norm `[n_embd]` input; returns the pre-residual
/// attention output `[n_embd]` and advances `st` in place.
fn cpu_linear_attn(
    g: &LazyGguf,
    hp: &Qwen35Hparams,
    geom: LinearAttnGeom,
    il: usize,
    cur: &[f32],
    st: &mut CpuLinearState,
) -> Result<Vec<f32>> {
    let p = format!("blk.{il}");
    let n_embd = hp.n_embd;
    let key_dim = geom.key_dim; // 2048
    let value_dim = geom.value_dim; // 6144
    let conv_dim = geom.conv_dim; // 10240
    let s = geom.head_k_dim; // S_k == S_v == 128
    let nkh = geom.num_k_heads; // 16
    let nvh = geom.num_v_heads; // 48
    let d_conv = geom.d_conv; // 4
    let eps = hp.rms_eps;

    // 1) qkv_mixed = wqkv @ cur  [conv_dim]; z = wqkv_gate @ cur  [value_dim].
    let qkv_mixed = gguf_gemv(g, &format!("{p}.attn_qkv.weight"), cur, conv_dim, n_embd)?;
    let z = gguf_gemv(g, &format!("{p}.attn_gate.weight"), cur, value_dim, n_embd)?;
    // L148 — ARF_GDN_ORACLE_DUMP=<il>: print this layer's stage heads from the CPU REFERENCE
    // (the path measured at cosine 1.000000). Compare against the megakernel's ARF_GDN_DUMP
    // rows to attribute a divergence to a specific stage numerically, not by reading text.
    if std::env::var("ARF_GDN_ORACLE_DUMP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        == Some(il)
    {
        let h6 = |v: &[f32]| -> String {
            v.iter()
                .take(6)
                .map(|x| format!("{x:+.5}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        eprintln!("[gdn-oracle l{il}] cur(normed)  {}", h6(cur));
        eprintln!("[gdn-oracle l{il}] qkv(fused)   {}", h6(&qkv_mixed));
        eprintln!("[gdn-oracle l{il}] z(pre-silu)  {}", h6(&z));
    }

    // 2) beta = sigmoid(ssm_beta @ cur)  [nvh]; alpha = ssm_alpha @ cur  [nvh].
    let mut beta = gguf_gemv(g, &format!("{p}.ssm_beta.weight"), cur, nvh, n_embd)?;
    for b in beta.iter_mut() {
        *b = sigmoid(*b);
    }
    let alpha = gguf_gemv(g, &format!("{p}.ssm_alpha.weight"), cur, nvh, n_embd)?;
    // gate = softplus(alpha + ssm_dt) * ssm_a   (per v-head; ssm_a = -exp(A_log) < 0).
    let ssm_dt = g
        .get_f32_raw(&format!("{p}.ssm_dt.bias"))
        .ok_or_else(|| ArfError::model_load(format!("missing {p}.ssm_dt.bias")))?;
    let ssm_a = g
        .get_f32_raw(&format!("{p}.ssm_a"))
        .ok_or_else(|| ArfError::model_load(format!("missing {p}.ssm_a")))?;
    let g_decay: Vec<f32> = (0..nvh)
        .map(|h| softplus(alpha[h] + ssm_dt[h]) * ssm_a[h])
        .collect();

    // 3) conv_out = silu(ssm_conv1d(qkv_mixed, conv_state))  [conv_dim]. Causal depthwise:
    //    window = [state[..d_conv-1] (oldest..newest), new_input]; advance the ring.
    let conv_kernel = g
        .get_f32_raw(&format!("{p}.ssm_conv1d.weight"))
        .ok_or_else(|| ArfError::model_load(format!("missing {p}.ssm_conv1d.weight")))?;
    let cs = d_conv - 1;
    let mut conv_out = vec![0.0f32; conv_dim];
    for i1 in 0..conv_dim {
        let sbase = i1 * cs;
        let kbase = i1 * d_conv;
        let mut window = [0.0f32; 8];
        window[..cs].copy_from_slice(&st.conv_state[sbase..sbase + cs]);
        window[cs] = qkv_mixed[i1];
        let mut acc = 0.0f32;
        for i0 in 0..d_conv {
            acc += window[i0] * conv_kernel[kbase + i0];
        }
        conv_out[i1] = silu(acc);
        // advance the ring: drop oldest, append the new input.
        st.conv_state[sbase..sbase + cs].copy_from_slice(&window[1..cs + 1]);
    }

    // 4) split conv_out → q_conv[key_dim], k_conv[key_dim], v_conv[value_dim]; then per-head
    //    L2-norm of q,k over head_k_dim (NO gain): y = x/max(||x||, eps).  (ggml l2_norm)
    let mut q_conv = conv_out[0..key_dim].to_vec();
    let mut k_conv = conv_out[key_dim..2 * key_dim].to_vec();
    let v_conv = &conv_out[2 * key_dim..2 * key_dim + value_dim];
    let l2 = |buf: &mut [f32], heads: usize| {
        for h in 0..heads {
            let base = h * s;
            let mut sum = 0.0f32;
            for i in 0..s {
                sum += buf[base + i] * buf[base + i];
            }
            let scale = 1.0 / sum.sqrt().max(eps as f32);
            for i in 0..s {
                buf[base + i] *= scale;
            }
        }
    };
    l2(&mut q_conv, nkh);
    l2(&mut k_conv, nkh);

    // 5) gated-delta-net recurrence, per v-head h (k/q head = h/rep, repeated across groups).
    //    State S_h[i + j*S_v]. Mirrors gated_delta_net_msl.metal exactly (q scaled 1/sqrt(S_k)).
    let q_scale = 1.0f32 / (s as f32).sqrt();
    let mut readout = vec![0.0f32; value_dim];
    for h in 0..nvh {
        // ggml_repeat_4d tiles q/k from num_k_heads → num_v_heads, so v-head h reads
        // k/q head `h % num_k_heads` (TILING, not block-grouping `h/rep`).
        let kh = h % nkh;
        let qbase = kh * s;
        let vbase = h * s;
        let sbase = h * s * s;
        let gexp = g_decay[h].exp();
        let beta_h = beta[h];
        // thread j owns column j: decay + sk_j, then S+=k*d, then o_j = Σ S*q.
        // (column-major over j; do all columns.)
        // First pass: decay every entry and compute sk[j] (needs decayed S).
        let mut sk = vec![0.0f32; s];
        for j in 0..s {
            let col0 = sbase + j * s;
            let mut acc = 0.0f32;
            for i in 0..s {
                let dec = st.ssm_state[col0 + i] * gexp;
                st.ssm_state[col0 + i] = dec;
                acc += dec * k_conv[qbase + i];
            }
            sk[j] = acc;
        }
        for j in 0..s {
            let col0 = sbase + j * s;
            let d_j = (v_conv[vbase + j] - sk[j]) * beta_h;
            let mut o_j = 0.0f32;
            for i in 0..s {
                let s_new = st.ssm_state[col0 + i] + k_conv[qbase + i] * d_j;
                st.ssm_state[col0 + i] = s_new;
                o_j += s_new * q_conv[qbase + i] * q_scale;
            }
            readout[vbase + j] = o_j;
        }
    }

    // 6) out = rmsnorm(readout, ssm_norm) * silu(z)  (build_norm_gated, per-head over head_v_dim).
    let ssm_norm = g
        .get_f32_raw(&format!("{p}.ssm_norm.weight"))
        .ok_or_else(|| ArfError::model_load(format!("missing {p}.ssm_norm.weight")))?;
    let mut gated = vec![0.0f32; value_dim];
    for h in 0..nvh {
        let base = h * s;
        let normed = rmsnorm(&readout[base..base + s], &ssm_norm, eps);
        for i in 0..s {
            gated[base + i] = normed[i] * silu(z[base + i]);
        }
    }

    // 7) out = ssm_out @ gated  [n_embd].
    gguf_gemv(g, &format!("{p}.ssm_out.weight"), &gated, n_embd, value_dim)
}

/// One FULL GQA layer's CPU forward for a single token at `pos`, with a per-layer KV cache
/// (`k_cache`/`v_cache` are `[ctx][kv_dim]`, appended in place). Standard qwen3 GQA: per-head
/// q/k RMSNorm over head_dim, partial rope (rotary_dim=64) at freq_base 1e7, causal GQA,
/// then a SIGMOID gate on the packed gate half, then o_proj. `cur` is post-attn_norm.
fn cpu_gqa_attn(
    g: &LazyGguf,
    hp: &Qwen35Hparams,
    il: usize,
    cur: &[f32],
    pos: usize,
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
) -> Result<Vec<f32>> {
    let p = format!("blk.{il}");
    let n_embd = hp.n_embd;
    let hd = hp.n_embd_head_k; // 256
    let nh = hp.n_head; // 24
    let nkv = hp.n_head_kv; // 4
    let group = nh / nkv; // 6
    let q_out = hd * nh * 2; // 12288 (Q|gate packed per head)
    let kv_out = hd * nkv; // 1024
    let o_in = hd * nh; // 6144
    let rot = hp.rope_dim; // 64 (rope.dimension_count)
    let theta = hp.rope_theta; // 1e7
    let eps = hp.rms_eps;

    // q/k/v projections.
    let q_full = gguf_gemv(g, &format!("{p}.attn_q.weight"), cur, q_out, n_embd)?;
    let mut k = gguf_gemv(g, &format!("{p}.attn_k.weight"), cur, kv_out, n_embd)?;
    let v = gguf_gemv(g, &format!("{p}.attn_v.weight"), cur, kv_out, n_embd)?;

    // split each q head's 512 into Q[0..256] and gate[256..512]; q is head-major [nh,512].
    let mut q = vec![0.0f32; nh * hd];
    let mut gate = vec![0.0f32; nh * hd];
    for h in 0..nh {
        let src = h * 2 * hd;
        for i in 0..hd {
            q[h * hd + i] = q_full[src + i];
            gate[h * hd + i] = q_full[src + hd + i];
        }
    }

    // per-head q/k RMSNorm over hd (learned gain).
    let q_norm = g
        .get_f32_raw(&format!("{p}.attn_q_norm.weight"))
        .ok_or_else(|| ArfError::model_load(format!("missing {p}.attn_q_norm.weight")))?;
    let k_norm = g
        .get_f32_raw(&format!("{p}.attn_k_norm.weight"))
        .ok_or_else(|| ArfError::model_load(format!("missing {p}.attn_k_norm.weight")))?;
    for h in 0..nh {
        let base = h * hd;
        let n = rmsnorm(&q[base..base + hd], &q_norm, eps);
        q[base..base + hd].copy_from_slice(&n);
    }
    for h in 0..nkv {
        let base = h * hd;
        let n = rmsnorm(&k[base..base + hd], &k_norm, eps);
        k[base..base + hd].copy_from_slice(&n);
    }

    // partial RoPE (rotary_dim `rot`) at the scalar position `pos`. Rotates the first `rot`
    // dims of each head as `rot/2` (lo,hi) pairs with hi at +rot/2 (NEOX/llama split-half).
    let rope = |vec: &mut [f32], heads: usize| {
        for h in 0..heads {
            let base = h * hd;
            let half = rot / 2;
            for i in 0..half {
                let freq = 1.0 / (theta.powf(2.0 * i as f64 / rot as f64));
                let ang = pos as f64 * freq;
                let (sin, cos) = (ang.sin() as f32, ang.cos() as f32);
                let lo = vec[base + i];
                let hi = vec[base + half + i];
                vec[base + i] = lo * cos - hi * sin;
                vec[base + half + i] = lo * sin + hi * cos;
            }
        }
    };
    rope(&mut q, nh);
    rope(&mut k, nkv);

    // append k/v to the per-layer cache at slot `pos`.
    debug_assert_eq!(k_cache.len(), pos * kv_out);
    k_cache.extend_from_slice(&k);
    v_cache.extend_from_slice(&v);
    let ctx_len = pos + 1;

    // causal GQA: head h attends kv-head h/group over slots 0..=pos.
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut attn = vec![0.0f32; o_in];
    for h in 0..nh {
        let kvh = h / group;
        let qbase = h * hd;
        // scores
        let mut scores = vec![0.0f32; ctx_len];
        let mut mx = f32::NEG_INFINITY;
        for t in 0..ctx_len {
            let kb = t * kv_out + kvh * hd;
            let mut dot = 0.0f32;
            for i in 0..hd {
                dot += q[qbase + i] * k_cache[kb + i];
            }
            let sc = dot * scale;
            scores[t] = sc;
            if sc > mx {
                mx = sc;
            }
        }
        let mut sum = 0.0f32;
        for t in 0..ctx_len {
            scores[t] = (scores[t] - mx).exp();
            sum += scores[t];
        }
        let inv = 1.0 / sum;
        for i in 0..hd {
            let mut acc = 0.0f32;
            for t in 0..ctx_len {
                let vb = t * kv_out + kvh * hd;
                acc += scores[t] * v_cache[vb + i];
            }
            attn[qbase + i] = acc * inv;
        }
    }

    // gate: attn *= sigmoid(gate)  (per the packed gate half, head-major aligned).
    for i in 0..o_in {
        attn[i] *= sigmoid(gate[i]);
    }

    // o_proj.
    gguf_gemv(g, &format!("{p}.attn_output.weight"), &attn, n_embd, o_in)
}

/// Dense SwiGLU FFN CPU forward: `down @ (silu(gate@x) * (up@x))`.
fn cpu_dense_ffn(g: &LazyGguf, hp: &Qwen35Hparams, il: usize, x: &[f32]) -> Result<Vec<f32>> {
    let p = format!("blk.{il}");
    let n_embd = hp.n_embd;
    let n_ff = hp.n_ff;
    let gate = gguf_gemv(g, &format!("{p}.ffn_gate.weight"), x, n_ff, n_embd)?;
    let up = gguf_gemv(g, &format!("{p}.ffn_up.weight"), x, n_ff, n_embd)?;
    let act: Vec<f32> = (0..n_ff).map(|i| silu(gate[i]) * up[i]).collect();
    gguf_gemv(g, &format!("{p}.ffn_down.weight"), &act, n_embd, n_ff)
}

/// The full CPU REFERENCE forward over a prompt token sequence. Returns the logits
/// `[vocab]` for the LAST position. Reads weights lazily from the GGUF mmap; the only
/// large transient buffers are token_embd (gathered once) and `output.weight` (the final
/// lm_head, dequantised once at the end). This is the M4 sane-logits gate.
/// The reference forward pass, shared by [`forward_cpu_reference`] and
/// [`forward_gpu_reference`].
///
/// The two differed by exactly ONE call — which implementation handles a recurrent layer's linear
/// attention — while duplicating 70 identical lines of embedding setup, KV caches, the prefill
/// loop, residuals, dense FFN and lm_head. Any change to that shared logic had to be made twice,
/// correctly, or the two oracles would quietly disagree about what "correct" means. `linear_attn`
/// is that one difference, passed in.
fn forward_reference_with(
    g: &LazyGguf,
    hp: &Qwen35Hparams,
    tokens: &[u32],
    mut linear_attn: impl FnMut(
        &LazyGguf,
        &Qwen35Hparams,
        LinearAttnGeom,
        usize,
        &[f32],
        &mut CpuLinearState,
    ) -> Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    let n_embd = hp.n_embd;
    let geom = hp.linear_geom();
    let n_trunk = hp.n_layer.saturating_sub(hp.n_layer_nextn); // 64
    let eps = hp.rms_eps;

    // ---- embeddings: gather the prompt rows from token_embd, then drop the big buffer ----
    let embd = g
        .get_f32_raw("token_embd.weight")
        .ok_or_else(|| ArfError::model_load("missing token_embd.weight".to_string()))?;
    let mut hidden: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| embd[t as usize * n_embd..(t as usize + 1) * n_embd].to_vec())
        .collect();
    drop(embd);

    // ---- per-linear-layer recurrent state + per-full-layer KV caches ----
    let mut lin_states: Vec<Option<CpuLinearState>> = Vec::with_capacity(n_trunk);
    let mut kv_caches: Vec<Option<(Vec<f32>, Vec<f32>)>> = Vec::with_capacity(n_trunk);
    for il in 0..n_trunk {
        if is_recurrent(il, hp.full_attention_interval) {
            lin_states.push(Some(CpuLinearState {
                conv_state: vec![0.0f32; geom.conv_dim * (geom.d_conv - 1)],
                ssm_state: vec![0.0f32; geom.head_v_dim * geom.head_v_dim * geom.num_v_heads],
            }));
            kv_caches.push(None);
        } else {
            lin_states.push(None);
            kv_caches.push(Some((Vec::new(), Vec::new())));
        }
    }

    // ---- prefill the prompt token-by-token (recurrent state + KV cache carry across pos) ----
    for pos in 0..tokens.len() {
        for il in 0..n_trunk {
            let h = &hidden[pos];
            let p = format!("blk.{il}");
            // attn_norm
            let attn_norm = g
                .get_f32_raw(&format!("{p}.attn_norm.weight"))
                .ok_or_else(|| ArfError::model_load(format!("missing {p}.attn_norm.weight")))?;
            let normed = rmsnorm(h, &attn_norm, eps);

            let attn_out = if is_recurrent(il, hp.full_attention_interval) {
                let st = lin_states[il].as_mut().unwrap();
                linear_attn(g, hp, geom, il, &normed, st)?
            } else {
                let (kc, vc) = kv_caches[il].as_mut().unwrap();
                cpu_gqa_attn(g, hp, il, &normed, pos, kc, vc)?
            };
            // residual
            let mut h2: Vec<f32> = (0..n_embd).map(|i| hidden[pos][i] + attn_out[i]).collect();
            // post_attention_norm → dense FFN → residual
            let post = g
                .get_f32_raw(&format!("{p}.post_attention_norm.weight"))
                .ok_or_else(|| {
                    ArfError::model_load(format!("missing {p}.post_attention_norm.weight"))
                })?;
            let normed2 = rmsnorm(&h2, &post, eps);
            let ffn = cpu_dense_ffn(g, hp, il, &normed2)?;
            for i in 0..n_embd {
                h2[i] += ffn[i];
            }
            hidden[pos] = h2;
        }
    }

    // ---- final norm + lm_head on the LAST position ----
    let last = hidden.last().unwrap();
    let final_norm = g
        .get_f32_raw("output_norm.weight")
        .ok_or_else(|| ArfError::model_load("missing output_norm.weight".to_string()))?;
    let normed = rmsnorm(last, &final_norm, eps);
    // lm_head: output.weight if present else tied token_embd.
    let lm_name = if g.shape_raw("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    gguf_gemv(g, lm_name, &normed, hp.n_vocab, n_embd)
}

/// Pure-CPU reference forward — the correctness oracle the GPU kernels are checked against.
pub fn forward_cpu_reference(g: &LazyGguf, hp: &Qwen35Hparams, tokens: &[u32]) -> Result<Vec<f32>> {
    forward_reference_with(g, hp, tokens, cpu_linear_attn)
}

// ---------------------------------------------------------------------------
// 9) M4b — GPU forward (the beast RUNS ON THE GPU): same hybrid stack as the CPU
//    reference, but the THREE NEW gated-delta-net ops (ssm_conv1d, the gdn prologue
//    l2_norm/softplus/sigmoid, and the gated_delta_net recurrence) dispatch on the
//    MetalIsland. GEMVs / rmsnorm / attention / FFN stay on the (proven) host code, so
//    this isolates exactly the GPU-kernel wiring and diffs against the CPU oracle.
// ---------------------------------------------------------------------------

/// One LINEAR (gated-delta-net) layer's forward for a single token, with the three NEW ops
/// on the GPU island. Identical math to [`cpu_linear_attn`] except: the per-head sigmoid
/// (`beta`) + softplus (`alpha+dt`) prologue, the causal conv1d, the per-head L2-norm of q/k,
/// and the delta-rule recurrence all run on the Metal island. Advances `st` in place.
#[cfg(target_os = "macos")]
fn gpu_linear_attn(
    g: &LazyGguf,
    hp: &Qwen35Hparams,
    geom: LinearAttnGeom,
    il: usize,
    cur: &[f32],
    st: &mut CpuLinearState,
    island: &mut crate::gpu::concurrent_metal::MetalIsland,
    ctx: &GpuContext,
) -> Result<Vec<f32>> {
    let p = format!("blk.{il}");
    let n_embd = hp.n_embd;
    let key_dim = geom.key_dim; // 2048
    let value_dim = geom.value_dim; // 6144
    let conv_dim = geom.conv_dim; // 10240
    let s = geom.head_k_dim; // S_k == S_v == 128
    let nkh = geom.num_k_heads; // 16
    let nvh = geom.num_v_heads; // 48
    let d_conv = geom.d_conv; // 4
    let eps = hp.rms_eps;

    // 1) qkv_mixed = wqkv @ cur [conv_dim]; z = wqkv_gate @ cur [value_dim]  (host GEMV).
    let qkv_mixed = gguf_gemv(g, &format!("{p}.attn_qkv.weight"), cur, conv_dim, n_embd)?;
    let z = gguf_gemv(g, &format!("{p}.attn_gate.weight"), cur, value_dim, n_embd)?;

    // 2) beta = sigmoid(ssm_beta @ cur) [nvh]  (GEMV host, sigmoid on GPU island).
    let beta_raw = gguf_gemv(g, &format!("{p}.ssm_beta.weight"), cur, nvh, n_embd)?;
    let beta = island
        .gdn_prologue_selftest(ctx, "sigmoid", &beta_raw, nvh, 1, 0.0)
        .map_err(|e| ArfError::model_load(format!("gpu gdn_sigmoid: {e}")))?;

    // alpha = ssm_alpha @ cur [nvh] (GEMV host); g_decay = softplus(alpha+dt)*ssm_a.
    let alpha = gguf_gemv(g, &format!("{p}.ssm_alpha.weight"), cur, nvh, n_embd)?;
    let ssm_dt = g
        .get_f32_raw(&format!("{p}.ssm_dt.bias"))
        .ok_or_else(|| ArfError::model_load(format!("missing {p}.ssm_dt.bias")))?;
    let ssm_a = g
        .get_f32_raw(&format!("{p}.ssm_a"))
        .ok_or_else(|| ArfError::model_load(format!("missing {p}.ssm_a")))?;
    // softplus(alpha+dt) on the GPU island; the per-head *ssm_a scale (host, trivial).
    let alpha_dt: Vec<f32> = (0..nvh).map(|h| alpha[h] + ssm_dt[h]).collect();
    let sp = island
        .gdn_prologue_selftest(ctx, "softplus", &alpha_dt, nvh, 1, 0.0)
        .map_err(|e| ArfError::model_load(format!("gpu gdn_softplus: {e}")))?;
    let g_decay: Vec<f32> = (0..nvh).map(|h| sp[h] * ssm_a[h]).collect();

    // 3-5) ssm_conv1d -> split -> l2_norm(q,k) -> gated_delta_net recurrence — ALL on the island.
    let conv_kernel = g
        .get_f32_raw(&format!("{p}.ssm_conv1d.weight"))
        .ok_or_else(|| ArfError::model_load(format!("missing {p}.ssm_conv1d.weight")))?;
    let readout = island
        .gated_delta_net_step(
            ctx,
            &qkv_mixed,
            &conv_kernel,
            &mut st.conv_state,
            &g_decay,
            &beta,
            &mut st.ssm_state,
            key_dim,
            value_dim,
            s,
            nkh,
            nvh,
            d_conv,
            eps as f32,
        )
        .map_err(|e| ArfError::model_load(format!("gpu gated_delta_net_step: {e}")))?;

    // 6) out = rmsnorm(readout, ssm_norm) * silu(z)  (build_norm_gated, per-head; host).
    let ssm_norm = g
        .get_f32_raw(&format!("{p}.ssm_norm.weight"))
        .ok_or_else(|| ArfError::model_load(format!("missing {p}.ssm_norm.weight")))?;
    let mut gated = vec![0.0f32; value_dim];
    for h in 0..nvh {
        let base = h * s;
        let normed = rmsnorm(&readout[base..base + s], &ssm_norm, eps);
        for i in 0..s {
            gated[base + i] = normed[i] * silu(z[base + i]);
        }
    }

    // 7) out = ssm_out @ gated [n_embd]  (host GEMV).
    gguf_gemv(g, &format!("{p}.ssm_out.weight"), &gated, n_embd, value_dim)
}

// ===========================================================================
// M7 — single-CommandPass-per-(sub)layer forward (the round-trip-elimination lever)
//
// Each layer collapses from ~4 wgpu submit+poll round-trips (M6) to TWO submits with ONE
// readback, and the host elementwise (rmsnorm, the silu-gate, the residual adds, the FFN
// swiglu) moves ONTO the GPU passes — they no longer force a GPU→CPU→GPU bounce. The
// residual stream stays resident in `gemv.hidden_buf` across all 64 layers; only the
// gated-delta-net island feed (per linear layer) and the final logits read back.
//
//   Linear layer:
//     Pass A: normed = rmsnorm(hidden, attn_norm); {qkv_mixed,z,beta_raw,alpha} = projs(normed)
//             → submit + readback [qkv_mixed, beta_raw, alpha]  (z STAYS on GPU)
//     island: sigmoid(beta_raw); softplus(alpha+dt)*ssm_a; gated_delta_net_step → readout(host)
//     Pass B: upload readout; gated = swiglu(z, rmsnorm_perhead(readout, ssm_norm));
//             attn_out = ssm_out·gated; hidden += attn_out;
//             post = rmsnorm(hidden, post_norm); ffn; hidden += ffn   → submit (NO readback)
//   GQA layer:
//     Pass A: normed = rmsnorm(hidden, attn_norm); {q,k,v} = projs(normed)
//             → submit + readback [q_full, k, v]
//     host:   per-head norm + rope + KV append + softmax GQA + sigmoid gate → attn(host)
//     Pass B: upload attn; o = o_proj·attn; hidden += o; post-norm; ffn; hidden += ffn → submit
// ===========================================================================

/// Record the shared tail (post_attention_norm → dense SwiGLU FFN → residual) onto `cp`.
/// The residual stream already holds `hidden = hidden + attn_out`. NO readback (output resident).
#[cfg(target_os = "macos")]
fn m7_record_ffn_tail(
    gemv: &mut Qwen35GpuGemv,
    cp: &mut CommandPass,
    mlp: &GpuDenseMlp,
    post_norm: &wgpu::Buffer,
    n_embd: usize,
    n_ff: usize,
    eps: f64,
) {
    // post = rmsnorm(hidden, post_attention_norm)
    let post = gemv.act_alloc(n_embd);
    gemv.rec_rmsnorm_from_hidden(cp, post_norm, post, eps);
    // gate = ffn_gate·post ; up = ffn_up·post
    let g_ = gemv.act_alloc(n_ff);
    let u_ = gemv.act_alloc(n_ff);
    gemv.rec_matmul(cp, post, &mlp.gate_proj, g_, n_ff, n_embd);
    gemv.rec_matmul(cp, post, &mlp.up_proj, u_, n_ff, n_embd);
    // act = silu(gate)*up
    let act = gemv.act_alloc(n_ff);
    gemv.rec_swiglu(cp, g_, u_, act, n_ff);
    // down = ffn_down·act ; hidden += down
    let down = gemv.act_alloc(n_embd);
    gemv.rec_matmul(cp, act, &mlp.down_proj, down, n_embd, n_ff);
    gemv.rec_residual_into_hidden(cp, down);
}

/// M7 LINEAR (gated-delta-net) layer. Reads/advances the resident residual stream
/// (`gemv.hidden_buf`) in place. The island feed + readout are the only host bounce.
#[cfg(target_os = "macos")]
fn m7_linear_layer(
    hp: &Qwen35Hparams,
    geom: LinearAttnGeom,
    il: usize,
    layer: &LinearAttnLayer,
    mlp: &GpuDenseMlp,
    hw: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), // (conv_kernel, ssm_dt, ssm_a, ssm_norm) — cached host f32
    st: &mut CpuLinearState,
    island: &mut crate::gpu::concurrent_metal::MetalIsland,
    gemv: &mut Qwen35GpuGemv,
    ctx: &GpuContext,
) -> Result<()> {
    let n_embd = hp.n_embd;
    let key_dim = geom.key_dim;
    let value_dim = geom.value_dim;
    let conv_dim = geom.conv_dim;
    let s = geom.head_k_dim;
    let nkh = geom.num_k_heads;
    let nvh = geom.num_v_heads;
    let d_conv = geom.d_conv;
    let eps = hp.rms_eps;
    let n_ff = hp.n_ff;

    let (conv_kernel, ssm_dt, ssm_a, _ssm_norm_host) = hw;

    // **M11** (default): the WHOLE linear layer records onto ONE wgpu CommandPass — rmsnorm →
    // projections → gated-delta-net recurrence (WGSL twins of the island kernels) → gate → ssm_out
    // → residual → FFN — submitted ONCE with NO poll, NO Metal island, NO cross-queue wait. This is
    // the EXACT shape `m7_gqa_layer` already proves (1 non-blocking submit, 0 blocking syncs/layer),
    // applied to the linear layers — collapsing the M10 2-blocking-syncs/layer to 0. The recurrent
    // conv_state/ssm_state are resident wgpu buffers carried across tokens. `ARF_QWEN35_ISLAND=1`
    // forces the legacy Metal-island path (M10) for A/B comparison.
    if std::env::var_os("ARF_QWEN35_ISLAND").is_none() {
        let conv_len = st.conv_state.len();
        let ssm_len = st.ssm_state.len();
        gemv.ensure_gdn_resident(il, conv_len, ssm_len, conv_kernel, ssm_dt, ssm_a);

        gemv.begin_layer();
        let mut cp = CommandPass::new(ctx_arc(gemv));
        // Pass A — rmsnorm + the 4 projections (all into resident act buffers).
        let normed = gemv.act_alloc(n_embd);
        gemv.rec_rmsnorm_from_hidden(&mut cp, &layer.attn_norm, normed, eps);
        let qkv = gemv.act_alloc(conv_dim);
        let z = gemv.act_alloc(value_dim);
        let beta_raw = gemv.act_alloc(nvh);
        let alpha = gemv.act_alloc(nvh);
        gemv.rec_matmul(&mut cp, normed, &layer.wqkv, qkv, conv_dim, n_embd);
        gemv.rec_matmul(&mut cp, normed, &layer.wqkv_gate, z, value_dim, n_embd);
        gemv.rec_matmul(&mut cp, normed, &layer.ssm_beta, beta_raw, nvh, n_embd);
        gemv.rec_matmul(&mut cp, normed, &layer.ssm_alpha, alpha, nvh, n_embd);
        // GDN recurrence — beta/g_decay/conv/l2/tile/recurrence, all on `cp`. readout -> `readout`.
        let readout = gemv.act_alloc(value_dim);
        gemv.rec_gdn_chain(&mut cp, il, geom, qkv, beta_raw, alpha, readout, eps);
        // Pass B — gate (rmsnorm·silu) → ssm_out → residual → FFN.
        let gnorm = gemv.act_alloc(value_dim);
        gemv.rec_rmsnorm(&mut cp, readout, &layer.ssm_norm, gnorm, nvh, s, eps);
        let gated = gemv.act_alloc(value_dim);
        gemv.rec_swiglu(&mut cp, z, gnorm, gated, value_dim);
        let attn_out = gemv.act_alloc(n_embd);
        gemv.rec_matmul(&mut cp, gated, &layer.ssm_out, attn_out, n_embd, value_dim);
        gemv.rec_residual_into_hidden(&mut cp, attn_out);
        m7_record_ffn_tail(gemv, &mut cp, mlp, &layer.attn_post_norm, n_embd, n_ff, eps);
        cp.submit(); // ONE submit, NO poll — exactly the GQA-layer shape.
        return Ok(());
    }

    // ----- LEGACY M10 ISLAND PATH (ARF_QWEN35_ISLAND=1) -----
    // Pass A: normed + four projections; the three island-feed projections (qkv/beta_raw/alpha)
    // write DUAL-VIEW SharedBuffers the Metal island reads directly. Submit + ONE poll, then the
    // island computes beta+g_decay on-GPU + conv→l2→tile→recurrence in ONE island command buffer.
    gemv.ensure_gdn_feed(conv_dim, nvh)?;
    let (qkv_w, beta_w, alpha_w, qkv_mtl, beta_mtl, alpha_mtl, dt_w, a_w, dt_mtl, a_mtl) = {
        let feed = gemv.gdn_feed.as_ref().expect("gdn feed built");
        (
            feed.qkv.wgpu.clone(),
            feed.beta_raw.wgpu.clone(),
            feed.alpha.wgpu.clone(),
            feed.qkv.mtl.clone(),
            feed.beta_raw.mtl.clone(),
            feed.alpha.mtl.clone(),
            feed.ssm_dt.wgpu.clone(),
            feed.ssm_a.wgpu.clone(),
            feed.ssm_dt.mtl.clone(),
            feed.ssm_a.mtl.clone(),
        )
    };
    gemv.ctx
        .queue
        .write_buffer(&dt_w, 0, bytemuck::cast_slice(ssm_dt));
    gemv.ctx
        .queue
        .write_buffer(&a_w, 0, bytemuck::cast_slice(ssm_a));

    gemv.begin_layer();
    let mut cp = CommandPass::new(ctx_arc(gemv));
    let normed = gemv.act_alloc(n_embd);
    gemv.rec_rmsnorm_from_hidden(&mut cp, &layer.attn_norm, normed, eps);
    let z = gemv.act_alloc(value_dim);
    gemv.rec_matmul_out(&mut cp, normed, &layer.wqkv, &qkv_w, conv_dim, n_embd);
    gemv.rec_matmul(&mut cp, normed, &layer.wqkv_gate, z, value_dim, n_embd);
    gemv.rec_matmul_out(&mut cp, normed, &layer.ssm_beta, &beta_w, nvh, n_embd);
    gemv.rec_matmul_out(&mut cp, normed, &layer.ssm_alpha, &alpha_w, nvh, n_embd);
    gemv.submit_and_poll(cp);

    let _ti = if timers::on() {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let conv_len = st.conv_state.len();
    let ssm_len = st.ssm_state.len();
    let readout = island
        .gated_delta_net_step_shared(
            ctx,
            il,
            &qkv_mtl,
            &beta_mtl,
            &alpha_mtl,
            &dt_mtl,
            &a_mtl,
            conv_kernel,
            conv_len,
            ssm_len,
            key_dim,
            value_dim,
            s,
            nkh,
            nvh,
            d_conv,
            eps as f32,
        )
        .map_err(|e| ArfError::model_load(format!("gpu gated_delta_net_step_shared: {e}")))?;
    if let Some(t) = _ti {
        timers::add_island(t.elapsed().as_micros() as u64);
    }

    let mut cp = CommandPass::new(ctx_arc(gemv));
    let readout_a = gemv.act_upload(&readout);
    let gnorm = gemv.act_alloc(value_dim);
    gemv.rec_rmsnorm(&mut cp, readout_a, &layer.ssm_norm, gnorm, nvh, s, eps);
    let gated = gemv.act_alloc(value_dim);
    gemv.rec_swiglu(&mut cp, z, gnorm, gated, value_dim);
    let attn_out = gemv.act_alloc(n_embd);
    gemv.rec_matmul(&mut cp, gated, &layer.ssm_out, attn_out, n_embd, value_dim);
    gemv.rec_residual_into_hidden(&mut cp, attn_out);
    m7_record_ffn_tail(gemv, &mut cp, mlp, &layer.attn_post_norm, n_embd, n_ff, eps);
    cp.submit();
    Ok(())
}

/// **M8** FULL GQA layer — attention ENTIRELY ON THE GPU. The per-head q/k RMSNorm, RoPE,
/// KV-cache append, and softmax attention (the ctx-growing 65% host leak) now dispatch on the
/// GPU with the EXACT decode kernels (`qk_norm`, `rope_qk`, `kv_scatter`, `attention`). The KV
/// cache is a per-layer RESIDENT GPU pool (no host Vec). The only host work is the resident-
/// buffer construction (once) — the whole per-token attention chain stays on the GPU, no
/// readback. Bit-faithful to the old host path: same per-head rmsnorm, same partial-rope freqs
/// (rope table built from the identical `theta^(-2i/rope_dim)`), same scaled-softmax, same
/// Q|gate split + sigmoid-gate.
#[cfg(target_os = "macos")]
fn m7_gqa_layer(
    hp: &Qwen35Hparams,
    layer: &GpuLayer,
    mlp: &GpuDenseMlp,
    il: usize,
    pos: usize,
    gemv: &mut Qwen35GpuGemv,
    gqa: &mut Qwen35GqaGpu,
) -> Result<()> {
    let n_embd = hp.n_embd;
    let hd = hp.n_embd_head_k;
    let nh = hp.n_head;
    let nkv = hp.n_head_kv;
    let group = nh / nkv;
    let q_out = hd * nh * 2;
    let kv_out = hd * nkv; // kv row floats (= nkv·hd)
    let o_in = hd * nh;
    let rot = hp.rope_dim;
    let eps = hp.rms_eps;
    let n_ff = hp.n_ff;
    let ctx_len = pos + 1;
    assert!(
        ctx_len <= gqa.max_ctx,
        "qwen35 GQA ctx {ctx_len} > max_ctx {}",
        gqa.max_ctx
    );
    assert!(
        gqa.layers[il].is_some(),
        "m7_gqa_layer on a non-GQA trunk layer {il}"
    );

    // ----- ONE CommandPass: rmsnorm → q/k/v proj → split → qk_norm → rope → kv_scatter →
    //       attention → sigmoid-gate → o_proj → residual → FFN tail. NO readback (output stays
    //       resident in `hidden_buf`). All GPU; the prior host attn loop is gone. -----
    gemv.begin_layer();
    gqa.idx.reset();
    // Per-token index buffers: positions[pos] (rope), new_slot=[pos] (scatter), slots=[0..=pos].
    let positions = gqa.idx.alloc_write(&[pos as u32])?;
    let new_slot = gqa.idx.alloc_write(&[pos as u32])?;
    let slots: Vec<u32> = (0..ctx_len as u32).collect();
    let slots_r = gqa.idx.alloc_write(&slots)?;

    let mut cp = CommandPass::new(ctx_arc(gemv));
    // rmsnorm(hidden) → q_full/k/v projections (resident act buffers; q_full packs Q|gate/head).
    let normed = gemv.act_alloc(n_embd);
    gemv.rec_rmsnorm_from_hidden(&mut cp, &layer.input_norm, normed, eps);
    let q_full_a = gemv.act_alloc(q_out);
    let k_a = gemv.act_alloc(kv_out);
    let v_a = gemv.act_alloc(kv_out);
    gemv.rec_matmul(&mut cp, normed, &layer.q_proj, q_full_a, q_out, n_embd);
    gemv.rec_matmul(&mut cp, normed, &layer.k_proj, k_a, kv_out, n_embd);
    gemv.rec_matmul(&mut cp, normed, &layer.v_proj, v_a, kv_out, n_embd);

    // Borrow the layer's KV pool + norms (immutable handles; the pool buffers are written by
    // kv_scatter, read by attention — both later dispatches of THIS pass).
    let glayer = gqa.layers[il].as_ref().unwrap();
    let kpool = &glayer.k_pool;
    let vpool = &glayer.v_pool;
    let q_norm = &glayer.q_norm;
    let k_norm = &glayer.k_norm;

    // split q_full [nh, 2hd] → q [nh,hd] (gqa.q_buf) + gate [nh,hd] (gqa.gate_buf).
    {
        let d = gemv
            .uni
            .alloc_write(&[nh as u32, hd as u32, 0, 0])
            .expect("split dims");
        cp.add_bound(
            &gqa.split,
            "qgate_split",
            &[
                gemv.act_res(q_full_a),
                gqa.q_buf.as_entire_binding(),
                gqa.gate_buf.as_entire_binding(),
                gemv.uni.resource(&d),
            ],
            [((nh * hd) as u32).div_ceil(256), 1, 1],
        );
    }
    // copy k/v projection rows from the act buffers into the gqa k/v scratch (qk_norm operates
    // in place on gqa.k_buf; kv_scatter reads gqa.k_buf/gqa.v_buf). One add-into-zero via the
    // `add` kernel would alias; instead a direct buffer-to-buffer copy in the encoder is simplest
    // — but the projections write the act buffers in THIS pass, so copy on-GPU with the split's
    // sibling: reuse the split kernel shape is wrong (single src). Use `add` from a zeroed buf?
    // Simplest correct: route k/v through the rmsnorm-less path by binding act buffers directly
    // into qk_norm / kv_scatter. qk_norm needs k as read_write — bind k_a's pool buffer.
    // (q is already in gqa.q_buf from the split.)
    let k_buf = &gemv.act_pool[k_a.buf_idx];
    let v_buf = &gemv.act_pool[v_a.buf_idx];

    // qk_norm: per-head RMSNorm on q (gqa.q_buf, nh rows) and k (k_buf, nkv rows), head_dim hd.
    {
        let d = gemv
            .uni
            .alloc_write(&[nh as u32, nkv as u32, hd as u32, (eps as f32).to_bits()])
            .expect("qk_norm dims");
        cp.add_bound(
            &gemv.kernels.qk_norm,
            "qwen35_gqa_qk_norm",
            &[
                gqa.q_buf.as_entire_binding(),
                k_buf.as_entire_binding(),
                q_norm.as_entire_binding(),
                k_norm.as_entire_binding(),
                gemv.uni.resource(&d),
            ],
            [(nh + nkv) as u32, 1, 1],
        );
    }
    // rope_qk: partial RoPE (rotary_dim = rope_dim) on q (gqa.q_buf) and k (k_buf).
    {
        let d = gemv
            .uni
            .alloc_write(&[1u32, nh as u32, nkv as u32, hd as u32, rot as u32, 0, 0, 0])
            .expect("rope_qk dims");
        let q_total = nh * hd;
        let k_total = nkv * hd;
        cp.add_bound(
            &gemv.kernels.rope_qk,
            "qwen35_gqa_rope_qk",
            &[
                gqa.q_buf.as_entire_binding(),
                k_buf.as_entire_binding(),
                gqa.rope_cos.as_entire_binding(),
                gqa.rope_sin.as_entire_binding(),
                gqa.idx.resource(&positions),
                gemv.uni.resource(&d),
            ],
            [((q_total + k_total) as u32).div_ceil(256), 1, 1],
        );
    }
    // kv_scatter: write this token's k/v into the pool at slot = pos.
    {
        let d = gemv
            .uni
            .alloc_write(&[1u32, kv_out as u32, 0, 0])
            .expect("kv_scatter dims");
        cp.add_bound(
            &gemv.kernels.kv_scatter,
            "qwen35_gqa_kv_scatter",
            &[
                k_buf.as_entire_binding(),
                v_buf.as_entire_binding(),
                gqa.idx.resource(&new_slot),
                kpool.as_entire_binding(),
                vpool.as_entire_binding(),
                gemv.uni.resource(&d),
            ],
            [(kv_out as u32).div_ceil(256), 1, 1],
        );
    }
    // attention: softmax GQA over slots[0..=pos] → gqa.attn_buf. One workgroup per head.
    {
        let scale = 1.0f32 / (hd as f32).sqrt();
        let adims = dims_attn(ctx_len, nh, nkv, hd, pos, scale, group, 0);
        let d = gemv.uni.alloc_write(&adims).expect("attn dims");
        cp.add_bound(
            &gemv.kernels.attention,
            "qwen35_gqa_attention",
            &[
                gqa.q_buf.as_entire_binding(),
                kpool.as_entire_binding(),
                vpool.as_entire_binding(),
                gqa.attn_buf.as_entire_binding(),
                gqa.idx.resource(&slots_r),
                gemv.uni.resource(&d),
            ],
            [nh as u32, 1, 1],
        );
    }
    // sigmoid-gate: attn_buf *= sigmoid(gate_buf), over o_in elements.
    {
        let d = gemv
            .uni
            .alloc_write(&[o_in as u32, 0, 0, 0])
            .expect("gate dims");
        cp.add_bound(
            &gqa.gate_sig,
            "qwen35_gqa_gate_sig",
            &[
                gqa.attn_buf.as_entire_binding(),
                gqa.gate_buf.as_entire_binding(),
                gemv.uni.resource(&d),
            ],
            [(o_in as u32).div_ceil(256), 1, 1],
        );
    }
    // o_proj(attn_buf) → o_a (act buffer); hidden += o_a; post-norm → FFN → residual.
    let o_a = gemv.act_alloc(n_embd);
    {
        // matmul from the EXTERNAL gqa.attn_buf into an act buffer. Use FIELD-level borrows so
        // `&gemv.kernels` + `&mut gemv.uni` + `&gemv.act_pool[..]` coexist (disjoint fields);
        // add_matmul allocates its own `mm` dims uniform.
        let Qwen35GpuGemv {
            kernels,
            uni,
            act_pool,
            ..
        } = gemv;
        let attn_res = gqa.attn_buf.as_entire_binding();
        let out_res = act_binding(&act_pool[o_a.buf_idx], o_a);
        add_matmul(
            &mut cp,
            kernels,
            uni,
            attn_res,
            &layer.o_proj,
            out_res,
            n_embd,
            o_in,
            None,
        );
    }
    gemv.rec_residual_into_hidden(&mut cp, o_a);
    m7_record_ffn_tail(gemv, &mut cp, mlp, &layer.post_norm, n_embd, n_ff, eps);
    cp.submit();
    Ok(())
}

/// Clone the GEMV helper's `ctx` Arc so a `CommandPass` can own it (CommandPass::new needs `&Arc`).
#[cfg(target_os = "macos")]
#[inline]
fn ctx_arc(gemv: &Qwen35GpuGemv) -> &Arc<GpuContext> {
    &gemv.ctx
}

// ===========================================================================
// B-PARALLEL (concurrency) layer drivers + driver loop. Each is the structural twin of its
// single-stream sibling, widened to `gemv.batch` lockstep sequences. The concurrency advantage: the GDN
// O(1) state advances B independent slices in one pass; llama's recurrence stalls at conc8-32.
// ===========================================================================

/// **B-PARALLEL** dense-FFN tail over `batch` b-outer rows. Twin of [`m7_record_ffn_tail`].
#[cfg(target_os = "macos")]
fn m7_record_ffn_tail_b(
    gemv: &mut Qwen35GpuGemv,
    cp: &mut CommandPass,
    mlp: &GpuDenseMlp,
    post_norm: &wgpu::Buffer,
    n_embd: usize,
    n_ff: usize,
    eps: f64,
) {
    let post = gemv.act_alloc_b(n_embd);
    gemv.rec_rmsnorm_from_hidden_b(cp, post_norm, post, eps);
    let g_ = gemv.act_alloc_b(n_ff);
    let u_ = gemv.act_alloc_b(n_ff);
    gemv.rec_matmul_b(cp, post, &mlp.gate_proj, g_, n_ff, n_embd);
    gemv.rec_matmul_b(cp, post, &mlp.up_proj, u_, n_ff, n_embd);
    let act = gemv.act_alloc_b(n_ff);
    gemv.rec_swiglu_b(cp, g_, u_, act, n_ff);
    let down = gemv.act_alloc_b(n_embd);
    gemv.rec_matmul_b(cp, act, &mlp.down_proj, down, n_embd, n_ff);
    gemv.rec_residual_into_hidden_b(cp, down);
}

/// **B-PARALLEL** LINEAR (gated-delta-net) layer. Twin of the M11 default path in
/// [`m7_linear_layer`]: rmsnorm → 4 batched projections → `rec_gdn_chain_b` → gate → ssm_out →
/// residual → FFN, all on ONE CommandPass, ONE submit, NO poll. The recurrent conv_state/ssm_state
/// are resident wgpu buffers sized `batch ×` and advanced per-seq in place.
#[cfg(target_os = "macos")]
fn m7_linear_layer_b(
    hp: &Qwen35Hparams,
    geom: LinearAttnGeom,
    il: usize,
    layer: &LinearAttnLayer,
    mlp: &GpuDenseMlp,
    hw: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
    st: &CpuLinearState,
    gemv: &mut Qwen35GpuGemv,
    _ctx: &GpuContext,
) -> Result<()> {
    let n_embd = hp.n_embd;
    let value_dim = geom.value_dim;
    let conv_dim = geom.conv_dim;
    let s = geom.head_k_dim;
    let nvh = geom.num_v_heads;
    let eps = hp.rms_eps;
    let n_ff = hp.n_ff;
    let b = gemv.batch;

    let (conv_kernel, ssm_dt, ssm_a, _ssm_norm_host) = hw;
    // Resident GDN state sized ×B (b-outer): conv_state/ssm_state span batch copies of the per-seq
    // state; the static conv_kernel/ssm_dt/ssm_a are SHARED (uploaded once at the per-seq length).
    gemv.ensure_gdn_resident(
        il,
        b * st.conv_state.len(),
        b * st.ssm_state.len(),
        conv_kernel,
        ssm_dt,
        ssm_a,
    );

    gemv.begin_layer();
    let mut cp = CommandPass::new(ctx_arc(gemv));
    // Pass A — rmsnorm + the 4 batched projections (b-outer act buffers).
    let normed = gemv.act_alloc_b(n_embd);
    gemv.rec_rmsnorm_from_hidden_b(&mut cp, &layer.attn_norm, normed, eps);
    let qkv = gemv.act_alloc_b(conv_dim);
    let z = gemv.act_alloc_b(value_dim);
    let beta_raw = gemv.act_alloc_b(nvh);
    let alpha = gemv.act_alloc_b(nvh);
    gemv.rec_matmul_b(&mut cp, normed, &layer.wqkv, qkv, conv_dim, n_embd);
    gemv.rec_matmul_b(&mut cp, normed, &layer.wqkv_gate, z, value_dim, n_embd);
    gemv.rec_matmul_b(&mut cp, normed, &layer.ssm_beta, beta_raw, nvh, n_embd);
    gemv.rec_matmul_b(&mut cp, normed, &layer.ssm_alpha, alpha, nvh, n_embd);
    // GDN recurrence (B-parallel) — readout -> `readout` (b-outer).
    let readout = gemv.act_alloc_b(value_dim);
    gemv.rec_gdn_chain_b(&mut cp, il, geom, qkv, beta_raw, alpha, readout, eps);
    // Pass B — gate (rmsnorm·silu) → ssm_out → residual → FFN.
    let gnorm = gemv.act_alloc_b(value_dim);
    gemv.rec_rmsnorm_b(&mut cp, readout, &layer.ssm_norm, gnorm, nvh, s, eps);
    let gated = gemv.act_alloc_b(value_dim);
    gemv.rec_swiglu_b(&mut cp, z, gnorm, gated, value_dim);
    let attn_out = gemv.act_alloc_b(n_embd);
    gemv.rec_matmul_b(&mut cp, gated, &layer.ssm_out, attn_out, n_embd, value_dim);
    gemv.rec_residual_into_hidden_b(&mut cp, attn_out);
    m7_record_ffn_tail_b(gemv, &mut cp, mlp, &layer.attn_post_norm, n_embd, n_ff, eps);
    cp.submit();
    Ok(())
}

/// One full-GQA layer's RESIDENT B-PARALLEL attention state: B independent KV pools (each
/// `[max_ctx · kv_dim]`) + the per-head q/k norms (shared across seqs).
#[cfg(target_os = "macos")]
struct GqaLayerGpuB {
    /// `batch` independent key pools (one per seq). Each seq attends ONLY its own pool.
    k_pools: Vec<wgpu::Buffer>,
    v_pools: Vec<wgpu::Buffer>,
    q_norm: wgpu::Buffer,
    k_norm: wgpu::Buffer,
}

/// **B-PARALLEL** resident GQA GPU state: B KV caches per full-attn layer + shared rope tables +
/// per-seq scratch. Twin of [`Qwen35GqaGpu`] with `batch` independent caches.
#[cfg(target_os = "macos")]
pub struct Qwen35GqaGpuB {
    layers: Vec<Option<GqaLayerGpuB>>,
    rope_cos: wgpu::Buffer,
    rope_sin: wgpu::Buffer,
    idx: GpuArena,
    split: crate::gpu::command::ComputeKernel,
    gate_sig: crate::gpu::command::ComputeKernel,
    /// Per-seq scratch reused for each seq's inner attention chain (q/gate/k/v/attn). Sequenced
    /// per-seq within one pass (wgpu hazard barriers keep the shared scratch safe across seqs).
    q_buf: wgpu::Buffer,
    gate_buf: wgpu::Buffer,
    // k/v scratch: the live path binds k/v via the act pool, so these struct handles are unread;
    // kept so the struct owns one buffer per GQA tensor (shape symmetry with q_buf).
    #[allow(dead_code)]
    k_buf: wgpu::Buffer,
    #[allow(dead_code)]
    v_buf: wgpu::Buffer,
    attn_buf: wgpu::Buffer,
    max_ctx: usize,
    batch: usize,
}

#[cfg(target_os = "macos")]
impl Qwen35GqaGpuB {
    pub fn new(
        ctx: &Arc<GpuContext>,
        g: &LazyGguf,
        hp: &Qwen35Hparams,
        max_ctx: usize,
        batch: usize,
    ) -> Result<Self> {
        let hd = hp.n_embd_head_k;
        let nh = hp.n_head;
        let nkv = hp.n_head_kv;
        let kv_dim = nkv * hd;
        let n_trunk = hp.n_layer.saturating_sub(hp.n_layer_nextn);
        let rope =
            arf_core::model::rope::Rope::with_theta_rotary(max_ctx, hp.rope_theta, hd, hp.rope_dim);
        let rope_cos = upload_storage(ctx, "qwen35b_gqa_rope_cos", rope.cos_table());
        let rope_sin = upload_storage(ctx, "qwen35b_gqa_rope_sin", rope.sin_table());

        let mut layers = Vec::with_capacity(n_trunk);
        for il in 0..n_trunk {
            if is_recurrent(il, hp.full_attention_interval) {
                layers.push(None);
                continue;
            }
            let p = format!("blk.{il}");
            let qn = g
                .get_f32_raw(&format!("{p}.attn_q_norm.weight"))
                .ok_or_else(|| ArfError::model_load(format!("missing {p}.attn_q_norm.weight")))?;
            let kn = g
                .get_f32_raw(&format!("{p}.attn_k_norm.weight"))
                .ok_or_else(|| ArfError::model_load(format!("missing {p}.attn_k_norm.weight")))?;
            let k_pools: Vec<wgpu::Buffer> = (0..batch)
                .map(|seq| {
                    ctx.create_buffer(
                        &format!("qwen35b_gqa_kpool_{il}_{seq}"),
                        (max_ctx * kv_dim * 4) as u64,
                        wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    )
                })
                .collect();
            let v_pools: Vec<wgpu::Buffer> = (0..batch)
                .map(|seq| {
                    ctx.create_buffer(
                        &format!("qwen35b_gqa_vpool_{il}_{seq}"),
                        (max_ctx * kv_dim * 4) as u64,
                        wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    )
                })
                .collect();
            layers.push(Some(GqaLayerGpuB {
                k_pools,
                v_pools,
                q_norm: upload_storage(ctx, &format!("qwen35b_gqa_qn_{il}"), &qn),
                k_norm: upload_storage(ctx, &format!("qwen35b_gqa_kn_{il}"), &kn),
            }));
        }

        let idx = GpuArena::new(
            ctx,
            "qwen35b_gqa_idx",
            ((max_ctx + 2) * 4 * 4).max(4096) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        let mk = |label: &str, n: usize| {
            ctx.create_buffer(
                label,
                (n * 4).max(16) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            )
        };
        Ok(Qwen35GqaGpuB {
            layers,
            rope_cos,
            rope_sin,
            idx,
            split: crate::gpu::command::ComputeKernel::new(ctx, "qgate_split", QGATE_SPLIT_WGSL),
            gate_sig: crate::gpu::command::ComputeKernel::new(
                ctx,
                "gate_sigmoid",
                GATE_SIGMOID_WGSL,
            ),
            q_buf: mk("qwen35b_gqa_q", nh * hd),
            gate_buf: mk("qwen35b_gqa_gate", nh * hd),
            k_buf: mk("qwen35b_gqa_k", nkv * hd),
            v_buf: mk("qwen35b_gqa_v", nkv * hd),
            attn_buf: mk("qwen35b_gqa_attn", nh * hd),
            max_ctx,
            batch,
        })
    }
}

/// **B-PARALLEL** full-GQA layer. The big q/k/v/o projections + FFN run as batched (M=B) GEMMs;
/// the per-seq inner attention chain (split → qk_norm → rope → kv_scatter → attention → gate)
/// runs once per seq against that seq's OWN KV pool, using b-strided offset bindings into the
/// b-outer projection act buffers. All B share the same `pos` (lockstep), so each seq's inner
/// chain is the EXACT single-stream chain → per-seq byte-identical to M11. ONE submit, NO poll.
#[cfg(target_os = "macos")]
fn m7_gqa_layer_b(
    hp: &Qwen35Hparams,
    layer: &GpuLayer,
    mlp: &GpuDenseMlp,
    il: usize,
    pos: usize,
    gemv: &mut Qwen35GpuGemv,
    gqa: &mut Qwen35GqaGpuB,
) -> Result<()> {
    let n_embd = hp.n_embd;
    let hd = hp.n_embd_head_k;
    let nh = hp.n_head;
    let nkv = hp.n_head_kv;
    let group = nh / nkv;
    let q_out = hd * nh * 2;
    let kv_out = hd * nkv;
    let o_in = hd * nh;
    let rot = hp.rope_dim;
    let eps = hp.rms_eps;
    let n_ff = hp.n_ff;
    let b = gemv.batch;
    let ctx_len = pos + 1;
    assert!(
        ctx_len <= gqa.max_ctx,
        "qwen35b GQA ctx {ctx_len} > max_ctx {}",
        gqa.max_ctx
    );
    assert!(
        gqa.layers[il].is_some(),
        "m7_gqa_layer_b on a non-GQA trunk layer {il}"
    );
    assert_eq!(b, gqa.batch, "batch mismatch gemv {b} vs gqa {}", gqa.batch);

    gemv.begin_layer();
    gqa.idx.reset();
    // Per-token index buffers (shared across seqs — all B advance the same pos in lockstep).
    let positions = gqa.idx.alloc_write(&[pos as u32])?;
    let new_slot = gqa.idx.alloc_write(&[pos as u32])?;
    let slots: Vec<u32> = (0..ctx_len as u32).collect();
    let slots_r = gqa.idx.alloc_write(&slots)?;

    let mut cp = CommandPass::new(ctx_arc(gemv));
    // Batched rmsnorm + q/k/v projections (b-outer act buffers).
    let normed = gemv.act_alloc_b(n_embd);
    gemv.rec_rmsnorm_from_hidden_b(&mut cp, &layer.input_norm, normed, eps);
    let q_full_a = gemv.act_alloc_b(q_out);
    let k_a = gemv.act_alloc_b(kv_out);
    let v_a = gemv.act_alloc_b(kv_out);
    gemv.rec_matmul_b(&mut cp, normed, &layer.q_proj, q_full_a, q_out, n_embd);
    gemv.rec_matmul_b(&mut cp, normed, &layer.k_proj, k_a, kv_out, n_embd);
    gemv.rec_matmul_b(&mut cp, normed, &layer.v_proj, v_a, kv_out, n_embd);
    // o_proj output, b-outer.
    let o_a = gemv.act_alloc_b(n_embd);

    let scale = 1.0f32 / (hd as f32).sqrt();
    // Per-seq inner attention chain. Each seq's chain is recorded in sequence; the shared GQA
    // scratch (q_buf/k_buf/...) is reused — wgpu inserts hazard barriers between dispatches so
    // seq's o_proj reads q_buf before the next seq's split overwrites it.
    for seq in 0..b {
        // b-outer slices of the projection act buffers (byte offsets must be 256-aligned; q_out
        // = 12288, kv_out = 1024 → ·4 always 256-aligned for the beast geometry).
        let q_full_slice = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &gemv.act_pool[q_full_a.buf_idx],
            offset: (seq * q_out * 4) as u64,
            size: wgpu::BufferSize::new((q_out * 4) as u64),
        });
        // split q_full[seq] [nh,2hd] → gqa.q_buf [nh,hd] + gqa.gate_buf [nh,hd].
        {
            let d = gemv
                .uni
                .alloc_write(&[nh as u32, hd as u32, 0, 0])
                .expect("split_b dims");
            cp.add_bound(
                &gqa.split,
                "qgate_split_b",
                &[
                    q_full_slice,
                    gqa.q_buf.as_entire_binding(),
                    gqa.gate_buf.as_entire_binding(),
                    gemv.uni.resource(&d),
                ],
                [((nh * hd) as u32).div_ceil(256), 1, 1],
            );
        }
        // k/v for this seq live in the act buffers at seq's b-outer slice. Copy into gqa.k_buf/
        // v_buf (qk_norm operates in place on k_buf; kv_scatter reads k_buf/v_buf). Encoder copy
        // is cleanest (no aliasing); but we are mid-pass — instead bind the act slices directly:
        // qk_norm needs k as read_write → bind k_a's seq slice. We DO need k in gqa.k_buf for the
        // later rope/scatter (which read gqa.k_buf), so copy k/v act-slice → gqa scratch with the
        // split kernel's sibling is wrong. Simplest correct in-pass: run qk_norm/rope/scatter on
        // the act-buffer seq slices directly (k_buf := k_a[seq], v_buf := v_a[seq]).
        let kslice = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &gemv.act_pool[k_a.buf_idx],
            offset: (seq * kv_out * 4) as u64,
            size: wgpu::BufferSize::new((kv_out * 4) as u64),
        });
        let vslice = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &gemv.act_pool[v_a.buf_idx],
            offset: (seq * kv_out * 4) as u64,
            size: wgpu::BufferSize::new((kv_out * 4) as u64),
        });
        let glayer = gqa.layers[il].as_ref().unwrap();
        let kpool = &glayer.k_pools[seq];
        let vpool = &glayer.v_pools[seq];
        let q_norm = &glayer.q_norm;
        let k_norm = &glayer.k_norm;
        // qk_norm: per-head RMSNorm on q (gqa.q_buf) and k (k_a[seq] slice, in place).
        {
            let d = gemv
                .uni
                .alloc_write(&[nh as u32, nkv as u32, hd as u32, (eps as f32).to_bits()])
                .expect("qk_norm_b dims");
            cp.add_bound(
                &gemv.kernels.qk_norm,
                "qwen35b_gqa_qk_norm",
                &[
                    gqa.q_buf.as_entire_binding(),
                    kslice.clone(),
                    q_norm.as_entire_binding(),
                    k_norm.as_entire_binding(),
                    gemv.uni.resource(&d),
                ],
                [(nh + nkv) as u32, 1, 1],
            );
        }
        // rope_qk: partial RoPE on q (gqa.q_buf) and k (k_a[seq] slice).
        {
            let d = gemv
                .uni
                .alloc_write(&[1u32, nh as u32, nkv as u32, hd as u32, rot as u32, 0, 0, 0])
                .expect("rope_qk_b dims");
            let q_total = nh * hd;
            let k_total = nkv * hd;
            cp.add_bound(
                &gemv.kernels.rope_qk,
                "qwen35b_gqa_rope_qk",
                &[
                    gqa.q_buf.as_entire_binding(),
                    kslice.clone(),
                    gqa.rope_cos.as_entire_binding(),
                    gqa.rope_sin.as_entire_binding(),
                    gqa.idx.resource(&positions),
                    gemv.uni.resource(&d),
                ],
                [((q_total + k_total) as u32).div_ceil(256), 1, 1],
            );
        }
        // kv_scatter: write this seq's k/v into its OWN pool at slot = pos.
        {
            let d = gemv
                .uni
                .alloc_write(&[1u32, kv_out as u32, 0, 0])
                .expect("kv_scatter_b dims");
            cp.add_bound(
                &gemv.kernels.kv_scatter,
                "qwen35b_gqa_kv_scatter",
                &[
                    kslice.clone(),
                    vslice.clone(),
                    gqa.idx.resource(&new_slot),
                    kpool.as_entire_binding(),
                    vpool.as_entire_binding(),
                    gemv.uni.resource(&d),
                ],
                [(kv_out as u32).div_ceil(256), 1, 1],
            );
        }
        // attention: softmax GQA over THIS seq's pool slots[0..=pos] → gqa.attn_buf.
        {
            let adims = dims_attn(ctx_len, nh, nkv, hd, pos, scale, group, 0);
            let d = gemv.uni.alloc_write(&adims).expect("attn_b dims");
            cp.add_bound(
                &gemv.kernels.attention,
                "qwen35b_gqa_attention",
                &[
                    gqa.q_buf.as_entire_binding(),
                    kpool.as_entire_binding(),
                    vpool.as_entire_binding(),
                    gqa.attn_buf.as_entire_binding(),
                    gqa.idx.resource(&slots_r),
                    gemv.uni.resource(&d),
                ],
                [nh as u32, 1, 1],
            );
        }
        // sigmoid-gate: attn_buf *= sigmoid(gate_buf).
        {
            let d = gemv
                .uni
                .alloc_write(&[o_in as u32, 0, 0, 0])
                .expect("gate_b dims");
            cp.add_bound(
                &gqa.gate_sig,
                "qwen35b_gqa_gate_sig",
                &[
                    gqa.attn_buf.as_entire_binding(),
                    gqa.gate_buf.as_entire_binding(),
                    gemv.uni.resource(&d),
                ],
                [(o_in as u32).div_ceil(256), 1, 1],
            );
        }
        // o_proj(attn_buf) → o_a[seq] slice (single-row matmul, m=1, this seq only).
        {
            let Qwen35GpuGemv {
                kernels,
                uni,
                act_pool,
                ..
            } = gemv;
            let attn_res = gqa.attn_buf.as_entire_binding();
            let out_res = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &act_pool[o_a.buf_idx],
                offset: (seq * n_embd * 4) as u64,
                size: wgpu::BufferSize::new((n_embd * 4) as u64),
            });
            add_matmul(
                &mut cp,
                kernels,
                uni,
                attn_res,
                &layer.o_proj,
                out_res,
                n_embd,
                o_in,
                None,
            );
        }
    }
    // Batched residual + FFN tail (b-outer).
    gemv.rec_residual_into_hidden_b(&mut cp, o_a);
    m7_record_ffn_tail_b(gemv, &mut cp, mlp, &layer.post_norm, n_embd, n_ff, eps);
    cp.submit();
    Ok(())
}

/// M4b GATE — the GPU forward over a prompt. Structurally identical to
/// [`forward_cpu_reference`], but the linear (gated-delta-net) layers route their three NEW
/// ops through the Metal island (see `gpu_linear_attn`). Returns the LAST-position logits.
#[cfg(target_os = "macos")]
/// GPU reference forward — identical to [`forward_cpu_reference`] except that recurrent layers run
/// their linear attention on the island. Holding everything else to the CPU path is what lets a
/// divergence be localised to the GPU linear-attn kernel specifically.
pub fn forward_gpu_reference(
    g: &LazyGguf,
    hp: &Qwen35Hparams,
    tokens: &[u32],
    island: &mut crate::gpu::concurrent_metal::MetalIsland,
    ctx: &GpuContext,
) -> Result<Vec<f32>> {
    forward_reference_with(g, hp, tokens, |g, hp, geom, il, normed, st| {
        gpu_linear_attn(g, hp, geom, il, normed, st, island, ctx)
    })
}

// ---------------------------------------------------------------------------
// 10) M5 — REAL TEXT GENERATION (autoregressive, INCREMENTAL O(1) state)
//
//    The single forward proves "correct logits". M5 turns it into "writes text":
//    prefill the prompt, then generate tokens one at a time, greedily, carrying the
//    SAME recurrent state across steps. The 48 LINEAR (gated-delta-net) layers
//    advance in O(1) per token (conv_state + ssm_state updated in place, NO growing
//    KV), while the 16 FULL-GQA layers keep the standard growing KV cache (the only
//    memory that grows with sequence length).
//
//    Design = option (B), incremental. The per-layer state (CpuLinearState +
//    GQA k/v caches) ALREADY persists across positions within one forward call —
//    we just lift it into a struct [`Qwen35GenState`] that lives across forward
//    *calls*, and feed one token per step into [`Qwen35GenState::step_gpu`]. The
//    absolute position drives the GQA RoPE + the cache-append invariant, so the
//    step must track `pos` and feed it through.
// ---------------------------------------------------------------------------

/// The full carried generation state: per-trunk-layer recurrent state (linear layers,
/// O(1)) + KV caches (full-GQA layers, growing) + the current absolute position. Built
/// once for a prompt, advanced one token at a time by `Qwen35GenState::step_gpu`.
pub struct Qwen35GenState {
    /// Per-linear-layer recurrent state; `None` on full-GQA layers (indexed by trunk `il`).
    pub lin_states: Vec<Option<CpuLinearState>>,
    /// Per-full-GQA-layer `(k_cache, v_cache)`; `None` on linear layers (indexed by `il`).
    pub kv_caches: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    /// Next absolute position to process (== number of tokens consumed so far).
    pub pos: usize,
}

/// Per-layer SMALL f32 vectors the gated-delta-net island + GQA norms consume on the HOST,
/// gathered ONCE at generation start so the hot per-token loop never re-reads/dequantises them
/// from the GGUF mmap (the M5/M6 `get_f32_raw` per layer per token was a hidden host tax —
/// conv_kernel alone is 40960 f32 = 164 KB re-read per linear layer per token). Indexed by trunk
/// layer `il`; the variant present depends on whether `il` is recurrent (linear) or full GQA.
pub struct Qwen35HostWeights {
    /// Linear layers: (ssm_conv1d.weight, ssm_dt.bias, ssm_a, ssm_norm.weight). `None` on GQA.
    pub lin: Vec<Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>>,
    /// GQA layers: (attn_q_norm.weight, attn_k_norm.weight). `None` on linear.
    pub gqa: Vec<Option<(Vec<f32>, Vec<f32>)>>,
}

impl Qwen35HostWeights {
    /// Gather all per-layer host f32 vectors once. Errors if any expected tensor is missing.
    pub fn gather(g: &LazyGguf, hp: &Qwen35Hparams) -> Result<Self> {
        let n_trunk = hp.n_layer.saturating_sub(hp.n_layer_nextn);
        let mut lin = Vec::with_capacity(n_trunk);
        let mut gqa = Vec::with_capacity(n_trunk);
        let rd = |name: &str| -> Result<Vec<f32>> {
            g.get_f32_raw(name)
                .ok_or_else(|| ArfError::model_load(format!("qwen35 host-cache: missing {name}")))
        };
        for il in 0..n_trunk {
            let p = format!("blk.{il}");
            if is_recurrent(il, hp.full_attention_interval) {
                lin.push(Some((
                    rd(&format!("{p}.ssm_conv1d.weight"))?,
                    rd(&format!("{p}.ssm_dt.bias"))?,
                    rd(&format!("{p}.ssm_a"))?,
                    rd(&format!("{p}.ssm_norm.weight"))?,
                )));
                gqa.push(None);
            } else {
                lin.push(None);
                gqa.push(Some((
                    rd(&format!("{p}.attn_q_norm.weight"))?,
                    rd(&format!("{p}.attn_k_norm.weight"))?,
                )));
            }
        }
        Ok(Qwen35HostWeights { lin, gqa })
    }
}

impl Qwen35GenState {
    /// Allocate fresh (zeroed) generation state for the hybrid trunk.
    pub fn new(hp: &Qwen35Hparams) -> Self {
        let geom = hp.linear_geom();
        let n_trunk = hp.n_layer.saturating_sub(hp.n_layer_nextn);
        let mut lin_states = Vec::with_capacity(n_trunk);
        let mut kv_caches = Vec::with_capacity(n_trunk);
        for il in 0..n_trunk {
            if is_recurrent(il, hp.full_attention_interval) {
                lin_states.push(Some(CpuLinearState {
                    conv_state: vec![0.0f32; geom.conv_dim * (geom.d_conv - 1)],
                    ssm_state: vec![0.0f32; geom.head_v_dim * geom.head_v_dim * geom.num_v_heads],
                }));
                kv_caches.push(None);
            } else {
                lin_states.push(None);
                kv_caches.push(Some((Vec::new(), Vec::new())));
            }
        }
        Qwen35GenState {
            lin_states,
            kv_caches,
            pos: 0,
        }
    }

    /// **M6** — advance ONE token through the full trunk using RESIDENT weights + GPU GEMVs.
    /// Every large projection (linear-layer wqkv/gate/beta/alpha/out, full-GQA q/k/v/o, dense
    /// FFN gate/up/down) runs on the GPU against the trunk's resident `GpuMatWeight` handles —
    /// NO per-token GGUF re-read (the M4b/M5 `gguf_gemv` host path, the ~2000× cost, is gone).
    /// The three gated-delta-net island ops + the tiny per-head f32 vectors / norms stay host.
    /// Returns the post-trunk hidden state `[n_embd]`; advances `self.pos`.
    #[cfg(target_os = "macos")]
    fn step_gpu(
        &mut self,
        trunk: &Qwen35Trunk,
        hw: &Qwen35HostWeights,
        token: u32,
        embd: &[f32],
        island: &mut crate::gpu::concurrent_metal::MetalIsland,
        gemv: &mut Qwen35GpuGemv,
        gqa_gpu: &mut Qwen35GqaGpu,
        ctx: &GpuContext,
    ) -> Result<Vec<f32>> {
        let hp = &trunk.hparams;
        let n_embd = hp.n_embd;
        let geom = hp.linear_geom();
        let n_trunk = hp.n_layer.saturating_sub(hp.n_layer_nextn);
        let pos = self.pos;
        let _t_step = if timers::on() {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // M7: seed the RESIDENT residual stream with this token's embedding row, then advance
        // it IN PLACE through every layer (no host residual bounce). Reset the per-token uniform
        // arena once up front; the per-layer activation arena is reset per layer (`begin_layer`).
        gemv.begin_token();
        let row = &embd[token as usize * n_embd..(token as usize + 1) * n_embd];
        gemv.seed_hidden(row);

        for il in 0..n_trunk {
            match &trunk.layers[il] {
                HybridLayer::Linear(lin) => {
                    let GpuMlp::Dense(mlp) = &lin.mlp else {
                        return Err(ArfError::model_load(
                            "qwen35: linear layer FFN is not Dense".to_string(),
                        ));
                    };
                    let st = self.lin_states[il].as_mut().unwrap();
                    let lhw = hw.lin[il].as_ref().expect("linear layer host weights");
                    m7_linear_layer(hp, geom, il, lin, mlp, lhw, st, island, gemv, ctx)?;
                }
                HybridLayer::Gqa(gqa) => {
                    let GpuMlp::Dense(mlp) = &gqa.0.mlp else {
                        return Err(ArfError::model_load(
                            "qwen35: gqa layer FFN is not Dense".to_string(),
                        ));
                    };
                    // M8: full-GQA attention is now ENTIRELY on the GPU (resident KV pool +
                    // qk_norm/rope_qk/kv_scatter/attention kernels). No host KV Vec, no host loop.
                    m7_gqa_layer(hp, &gqa.0, mlp, il, pos, gemv, gqa_gpu)?;
                }
            }
        }

        // Read the resident residual stream back ONCE (feeds the final-norm + lm_head, and the
        // prefill `last_hidden` that seeds the first argmax). This is the only end-of-trunk sync.
        let hidden = gemv.readback_hidden();

        if timers::on() {
            let t_tok = std::time::Instant::now()
                .duration_since(_t_step.unwrap())
                .as_micros() as u64;
            let (g_us, i_us) = timers::take();
            eprintln!(
                "[time] tok: total {:.1}ms | gemv {:.1}ms | island-gdn {:.1}ms | host/other {:.1}ms",
                t_tok as f64 / 1000.0,
                g_us as f64 / 1000.0,
                i_us as f64 / 1000.0,
                (t_tok.saturating_sub(g_us).saturating_sub(i_us)) as f64 / 1000.0,
            );
        }
        self.pos += 1;
        Ok(hidden)
    }

    /// **B-PARALLEL** — advance ONE token through the trunk for `gemv.batch` lockstep sequences.
    /// `tokens[b]` is seq b's next token id; all B advance the same `pos`. The GDN O(1) state +
    /// the B KV pools live RESIDENT on the GPU (in `gemv` / `gqa_gpu`), advanced in place. Returns
    /// the post-trunk hidden state `[batch*n_embd]` (b-outer). Twin of [`step_gpu`].
    #[cfg(target_os = "macos")]
    fn step_gpu_batched(
        &mut self,
        trunk: &Qwen35Trunk,
        hw: &Qwen35HostWeights,
        tokens: &[u32],
        embd: &[f32],
        gemv: &mut Qwen35GpuGemv,
        gqa_gpu: &mut Qwen35GqaGpuB,
        ctx: &GpuContext,
    ) -> Result<Vec<f32>> {
        let hp = &trunk.hparams;
        let n_embd = hp.n_embd;
        let geom = hp.linear_geom();
        let n_trunk = hp.n_layer.saturating_sub(hp.n_layer_nextn);
        let pos = self.pos;
        let b = gemv.batch;
        assert_eq!(
            tokens.len(),
            b,
            "step_gpu_batched: {} tokens != batch {b}",
            tokens.len()
        );

        gemv.begin_token();
        // Seed the b-outer residual stream with the B embedding rows.
        let mut flat = vec![0.0f32; b * n_embd];
        for (seq, &t) in tokens.iter().enumerate() {
            let row = &embd[t as usize * n_embd..(t as usize + 1) * n_embd];
            flat[seq * n_embd..(seq + 1) * n_embd].copy_from_slice(row);
        }
        gemv.seed_hidden_batched(&flat);

        let dbg_time = std::env::var_os("ARF_QWEN35B_TIME").is_some();
        let mut t_lin = 0u128;
        let mut t_gqa = 0u128;
        for il in 0..n_trunk {
            match &trunk.layers[il] {
                HybridLayer::Linear(lin) => {
                    let GpuMlp::Dense(mlp) = &lin.mlp else {
                        return Err(ArfError::model_load(
                            "qwen35: linear layer FFN is not Dense".to_string(),
                        ));
                    };
                    // The CpuLinearState is used only for the per-seq state SIZE; the B-parallel
                    // state itself lives resident in `gemv.gdn_res` (sized ×B on first touch).
                    let st = self.lin_states[il].as_ref().unwrap();
                    let lhw = hw.lin[il].as_ref().expect("linear layer host weights");
                    let t0 = if dbg_time {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    m7_linear_layer_b(hp, geom, il, lin, mlp, lhw, st, gemv, ctx)?;
                    if dbg_time {
                        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
                        t_lin += t0.unwrap().elapsed().as_micros();
                    }
                }
                HybridLayer::Gqa(gqa) => {
                    let GpuMlp::Dense(mlp) = &gqa.0.mlp else {
                        return Err(ArfError::model_load(
                            "qwen35: gqa layer FFN is not Dense".to_string(),
                        ));
                    };
                    let t0 = if dbg_time {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    m7_gqa_layer_b(hp, &gqa.0, mlp, il, pos, gemv, gqa_gpu)?;
                    if dbg_time {
                        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
                        t_gqa += t0.unwrap().elapsed().as_micros();
                    }
                }
            }
        }
        if dbg_time {
            eprintln!(
                "[qwen35b time] linear(48L) {:.1}ms | gqa(16L) {:.1}ms",
                t_lin as f64 / 1000.0,
                t_gqa as f64 / 1000.0
            );
        }

        let hidden = gemv.readback_hidden_batched();
        self.pos += 1;
        Ok(hidden)
    }
}

/// **M6** — final-norm + lm_head GEMV against the RESIDENT lm_head weight (the biggest single
/// per-token GEMV, 5120×248320). `output_norm` is a tiny F32 vector → cheap host memcpy.
#[cfg(target_os = "macos")]
fn lm_head_logits_resident(
    g: &LazyGguf,
    trunk: &Qwen35Trunk,
    hidden: &[f32],
    gemv: &mut Qwen35GpuGemv,
) -> Result<Vec<f32>> {
    let hp = &trunk.hparams;
    let final_norm = g
        .get_f32_raw("output_norm.weight")
        .ok_or_else(|| ArfError::model_load("missing output_norm.weight".to_string()))?;
    let normed = rmsnorm(hidden, &final_norm, hp.rms_eps);
    Ok(gemv.gemv(&normed, &trunk.lm_head, hp.n_vocab, hp.n_embd))
}

/// **B-PARALLEL** lm_head: final-norm (per-seq, host — tiny) + ONE batched GEMM `[B,vocab] =
/// [B,n_embd]·lm_headᵀ` against the resident weight, then B host argmaxes. `hidden_flat` is the
/// b-outer post-trunk hidden `[batch*n_embd]`. Returns the B next-token ids. Uses the gemv's
/// `a_buf`/`c_buf` (sized ×B) + `add_matmul_m(m=batch)` — the conc64 batched GEMM.
#[cfg(target_os = "macos")]
fn lm_head_argmax_batched(
    g: &LazyGguf,
    trunk: &Qwen35Trunk,
    hidden_flat: &[f32],
    gemv: &mut Qwen35GpuGemv,
) -> Result<Vec<u32>> {
    let hp = &trunk.hparams;
    let n_embd = hp.n_embd;
    let vocab = hp.n_vocab;
    let b = gemv.batch;
    debug_assert_eq!(hidden_flat.len(), b * n_embd);
    let final_norm = g
        .get_f32_raw("output_norm.weight")
        .ok_or_else(|| ArfError::model_load("missing output_norm.weight".to_string()))?;
    // Per-seq final rmsnorm (host; output_norm is a tiny F32 vector) into a b-outer input.
    let mut normed_flat = vec![0.0f32; b * n_embd];
    for seq in 0..b {
        let row = &hidden_flat[seq * n_embd..(seq + 1) * n_embd];
        let nr = rmsnorm(row, &final_norm, hp.rms_eps);
        normed_flat[seq * n_embd..(seq + 1) * n_embd].copy_from_slice(&nr);
    }
    // Batched GEMM into c_buf [B*vocab], then read back + argmax per seq.
    gemv.ctx
        .queue
        .write_buffer(&gemv.a_buf, 0, bytemuck::cast_slice(&normed_flat));
    gemv.uni.reset();
    let mut cp = CommandPass::new(&gemv.ctx);
    let a_bind = gemv.a_buf.as_entire_binding();
    let c_bind = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: &gemv.c_buf,
        offset: 0,
        size: wgpu::BufferSize::new((b * vocab * 4) as u64),
    });
    super::dispatch::add_matmul_m(
        &mut cp,
        &gemv.kernels,
        &mut gemv.uni,
        b,
        a_bind,
        &trunk.lm_head,
        c_bind,
        vocab,
        n_embd,
        None,
    );
    cp.submit();
    let logits = gemv.ctx.read_f32(&gemv.c_buf, b * vocab);
    let mut ids = Vec::with_capacity(b);
    for seq in 0..b {
        ids.push(argmax(&logits[seq * vocab..(seq + 1) * vocab]));
    }
    Ok(ids)
}

/// argmax of a logit vector (greedy decode).
#[inline]
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// **M5 GATE — real autoregressive text generation on the GPU.**
///
/// Prefill `prompt_tokens` (advancing the recurrent state + GQA KV caches), then generate
/// `n_new` tokens greedily (argmax the last-position logits each step), carrying the SAME
/// state incrementally — O(1) per token through the 48 linear layers, growing KV only on
/// the 16 full-GQA layers. Stops early on the GGUF EOS token if seen.
///
/// Returns the GENERATED token ids (NOT including the prompt). `on_token`, if provided, is
/// called with each newly generated id as it is produced (for streaming / progress).
#[cfg(target_os = "macos")]
pub fn generate_qwen35(
    g: &LazyGguf,
    hp: &Qwen35Hparams,
    prompt_tokens: &[u32],
    n_new: usize,
    island: &mut crate::gpu::concurrent_metal::MetalIsland,
    ctx: &Arc<GpuContext>,
    mut on_token: Option<&mut dyn FnMut(u32)>,
) -> Result<Vec<u32>> {
    assert!(!prompt_tokens.is_empty(), "generate_qwen35: empty prompt");
    let n_embd = hp.n_embd;
    // M10 STEP 2: start every generation from zeroed resident GDN state (a reused island must
    // not carry the prior conversation's conv_state/ssm_state).
    island.reset_gdn_resident_state();

    // ---- M6: load ALL weights RESIDENT on the GPU ONCE (the GpuMatWeight handles). Every
    //      per-token GEMV then runs on the GPU against these — the host gguf_gemv (the ~2000×
    //      cost) is gone. ----
    let t_load = std::time::Instant::now();
    let trunk = load_qwen35(ctx, g, &arf_core::config::ModelConfig::qwen35_27b())?;
    eprintln!(
        "[M6] resident weights loaded in {:.1}s (trunk + lm_head on GPU)",
        t_load.elapsed().as_secs_f64()
    );
    // GEMV helper: IO buffers sized to the LARGEST GEMV in either dim. Inputs range over
    // {n_embd, value_dim, n_ff (FFN down), o_in (GQA o_proj)}; outputs over {vocab (lm_head),
    // n_ff (FFN gate/up), q_out (GQA q), conv_dim (wqkv), n_embd}.
    let geom = hp.linear_geom();
    let q_out = hp.n_embd_head_k * hp.n_head * 2;
    let max_k = n_embd
        .max(geom.value_dim)
        .max(hp.n_ff)
        .max(hp.n_embd_head_k * hp.n_head);
    let max_n = hp
        .n_vocab
        .max(hp.n_ff)
        .max(q_out)
        .max(geom.conv_dim)
        .max(n_embd);
    let mut gemv = Qwen35GpuGemv::new(ctx, max_k, max_n, n_embd);

    // Gather the embedding table ONCE (the only big transient besides the lm_head). Held
    // for the whole generation so every step is a cheap row slice.
    let embd = g
        .get_f32_raw("token_embd.weight")
        .ok_or_else(|| ArfError::model_load("missing token_embd.weight".to_string()))?;

    let eos = g.get_metadata_u32("tokenizer.ggml.eos_token_id");

    let mut state = Qwen35GenState::new(hp);
    // M7: gather all per-layer SMALL host f32 vectors ONCE (conv_kernel / ssm_dt / ssm_a /
    // ssm_norm per linear layer) so the hot loop never re-reads them. (The GQA q/k norms now
    // live RESIDENT on the GPU in `gqa_gpu`; `hw.gqa` is left populated but unused by M8.)
    let hw = Qwen35HostWeights::gather(g, hp)?;
    // M8: resident GPU state for the full-GQA attention (KV pool + rope tables + q/k norms).
    // Sized to the worst-case context = prompt + n_new (the KV pool grows by one slot per token).
    let max_ctx = prompt_tokens.len() + n_new + 1;
    let mut gqa_gpu = Qwen35GqaGpu::new(ctx, g, hp, max_ctx)?;
    let mut generated: Vec<u32> = Vec::with_capacity(n_new);

    // ---- PREFILL: run every prompt token through the trunk, carrying state. The hidden
    //      state of the LAST prompt token feeds the first generation argmax. ----
    let mut last_hidden: Vec<f32> = vec![0.0; n_embd];
    for &tok in prompt_tokens {
        last_hidden = state.step_gpu(
            &trunk,
            &hw,
            tok,
            &embd,
            island,
            &mut gemv,
            &mut gqa_gpu,
            ctx,
        )?;
    }

    // ---- GENERATE: argmax → emit → feed back, advancing the SAME state O(1) per step. ----
    for _ in 0..n_new {
        let logits = lm_head_logits_resident(g, &trunk, &last_hidden, &mut gemv)?;
        let next = argmax(&logits);
        generated.push(next);
        if let Some(cb) = on_token.as_deref_mut() {
            cb(next);
        }
        if Some(next) == eos {
            break;
        }
        // feed the just-emitted token as the next input (advances state to pos+1).
        last_hidden = state.step_gpu(
            &trunk,
            &hw,
            next,
            &embd,
            island,
            &mut gemv,
            &mut gqa_gpu,
            ctx,
        )?;
    }

    Ok(generated)
}

/// **B-PARALLEL (concurrency) GENERATION.** Decode `prompts.len() == B` sequences in lockstep — one
/// trunk pass per step processes all B. The gated-delta-net O(1) state advances B independent
/// slices in a single pass (the advantage: it grows with B but NOT context length, so we scale where
/// llama's recurrence stalls at conc8-32). NO Metal island (the batched linear path is pure wgpu).
///
/// All B advance the same `pos` (lockstep greedy). For the parity gate, pass B copies of the same
/// prompt: every returned id stream MUST be identical AND equal the single-stream [`generate_qwen35`]
/// ids. Per-seq EOS: once a seq hits EOS its stream is frozen (further steps still run it through
/// the trunk to keep the batch in lockstep, but its emitted ids stop). Returns one id stream per seq.
///
/// `trunk_in`/`embd_in` let a caller (the conc-sweep) load the model ONCE and reuse it across batch
/// sizes; pass `None` to load fresh.
#[cfg(target_os = "macos")]
pub fn generate_qwen35_batched(
    g: &LazyGguf,
    hp: &Qwen35Hparams,
    prompts: &[Vec<u32>],
    n_new: usize,
    ctx: &Arc<GpuContext>,
    mut on_step: Option<&mut dyn FnMut(usize, &[u32])>,
) -> Result<Vec<Vec<u32>>> {
    assert!(!prompts.is_empty(), "generate_qwen35_batched: empty batch");
    assert!(
        prompts.iter().all(|p| !p.is_empty()),
        "generate_qwen35_batched: empty prompt"
    );
    let b = prompts.len();
    let n_embd = hp.n_embd;

    let t_load = std::time::Instant::now();
    let trunk = load_qwen35(ctx, g, &arf_core::config::ModelConfig::qwen35_27b())?;
    eprintln!(
        "[B-par] resident weights loaded in {:.1}s (B={b})",
        t_load.elapsed().as_secs_f64()
    );
    let geom = hp.linear_geom();
    let q_out = hp.n_embd_head_k * hp.n_head * 2;
    let max_k = n_embd
        .max(geom.value_dim)
        .max(hp.n_ff)
        .max(hp.n_embd_head_k * hp.n_head);
    let max_n = hp
        .n_vocab
        .max(hp.n_ff)
        .max(q_out)
        .max(geom.conv_dim)
        .max(n_embd);
    let mut gemv = Qwen35GpuGemv::new_batched(ctx, max_k, max_n, n_embd, b);

    let embd = g
        .get_f32_raw("token_embd.weight")
        .ok_or_else(|| ArfError::model_load("missing token_embd.weight".to_string()))?;
    let eos = g.get_metadata_u32("tokenizer.ggml.eos_token_id");

    let mut state = Qwen35GenState::new(hp);
    let hw = Qwen35HostWeights::gather(g, hp)?;
    // Max context = longest prompt + n_new (all seqs share pos; shorter prompts are left-padded by
    // the lockstep rule — for the parity gate all prompts are equal length so this is exact).
    let max_prompt = prompts.iter().map(|p| p.len()).max().unwrap();
    let max_ctx = max_prompt + n_new + 1;
    let mut gqa_gpu = Qwen35GqaGpuB::new(ctx, g, hp, max_ctx, b)?;

    // For the parity/lockstep contract, all prompts must be the same length (one token per seq per
    // step). Enforce it (the conc-sweep uses identical prompts; ragged continuous-batch is future).
    assert!(
        prompts.iter().all(|p| p.len() == max_prompt),
        "generate_qwen35_batched: all prompts must be equal length (lockstep); got {:?}",
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
    );

    let mut outs: Vec<Vec<u32>> = vec![Vec::with_capacity(n_new); b];
    let mut done = vec![false; b];

    // ---- PREFILL: feed prompt token `pos` of EVERY seq in lockstep. ----
    let mut last_hidden = vec![0.0f32; b * n_embd];
    for pos in 0..max_prompt {
        let toks: Vec<u32> = prompts.iter().map(|p| p[pos]).collect();
        last_hidden =
            state.step_gpu_batched(&trunk, &hw, &toks, &embd, &mut gemv, &mut gqa_gpu, ctx)?;
    }

    // ---- GENERATE: B argmaxes → B next tokens → feed back, lockstep. ----
    let dbg_tok = std::env::var_os("ARF_QWEN35B_TOK").is_some();
    for step_i in 0..n_new {
        let t0 = if dbg_tok {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let next = lm_head_argmax_batched(g, &trunk, &last_hidden, &mut gemv)?;
        for seq in 0..b {
            if !done[seq] {
                outs[seq].push(next[seq]);
                if Some(next[seq]) == eos {
                    done[seq] = true;
                }
            }
        }
        if let Some(cb) = on_step.as_deref_mut() {
            cb(state.pos, &next);
        }
        if done.iter().all(|&d| d) {
            break;
        }
        last_hidden =
            state.step_gpu_batched(&trunk, &hw, &next, &embd, &mut gemv, &mut gqa_gpu, ctx)?;
        if let Some(t) = t0 {
            eprintln!(
                "[qwen35b tok {step_i}] {:.1}ms",
                t.elapsed().as_micros() as f64 / 1000.0
            );
        }
    }

    Ok(outs)
}
