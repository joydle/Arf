//! Batched / prefill forward path for [`GpuModel`]: `forward_batch*`,
//! `forward_batch_impl`, and the batched + grouped MoE expert kernels.
//! Split out of the original monolithic gpu.rs; behavior unchanged.

use super::*;

/// Handle for a submitted-but-not-yet-read batched decode step (depth-2 pipelining).
///
/// Returned by [`GpuModel::submit_batched_step`] (record + async commit) and consumed by
/// [`GpuModel::read_pending`] (fence + read). Splitting these lets the decode loop submit
/// step N+1 before reading step N. `pos0` is the fence position; `b` is the batch
/// size (rows to read out of the out bank). `k` is the K-step chain length: the record
/// chained `k` decode steps into one buffer writing out_bank[0..k*b], so `read_pending` reads
/// `k*b` tokens (step-major: rows [ks*b .. ks*b+b] are step ks's B sampled ids). k=1 = one step.
pub struct BatchPending {
    pub pos0: usize,
    pub b: usize,
    pub k: usize,
}

// ── L11 SPEC PROFILER (ARF_SPEC_PROF=1) ──────────────────────────────────────────────
// The spec lever is gated on ONE number: what a verify window costs RELATIVE to a decode
// step on the SAME daemon (measured). Both are timed at the same seam —
// wall-clock around the island record — and reported as a ratio at process exit, so the
// BEFORE/AFTER comparison is apples-to-apples regardless of machine heat.
// L364 — island/speculation-only: dead on Linux by construction. Gated rather than blanket
// #[allow(dead_code)], which would also hide genuinely dead code on macOS.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpecProfKind {
    Verify,
    Decode,
    Draft,
    /// L331 — the verify's record BUILD (CPU encode + commit), split out of `Verify`.
    VerifySubmit,
    /// L331 — the verify's GPU EXECUTION (spin until the command buffer signals done).
    VerifyRead,
    /// L331 — `gdn_restore_checkpoint`: 96 bank_copy dispatches on a BLOCKING submit, run
    /// after the record on every partial accept. Never previously attributed to anything.
    Restore,
}

#[cfg(target_os = "macos")]
pub(crate) fn spec_prof() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("ARF_SPEC_PROF").is_some())
}

#[cfg(target_os = "macos")]
/// L11 — ARF_SPEC_FUSED_VERIFY (default OFF until measured). Arms the two changes that
/// let spec-verify actually RUN on the island in the daemon's default config:
///   1. verify windows may import f16→f32 for their prefix instead of bailing on the
///      f16-KV guard (the guard's requirement is satisfied by the import, not skipped);
///   2. paged (non-identity) slot tables are allowed — the B-row record already honours them.
/// OFF = byte-identical to pre-L11 behaviour (verify bails; spec is inert).
/// L235 — the recurrent bank width to size for, once, at the daemon's ceiling. Never the
/// current step's `b`: a grow mid-stream zeroes the state (see the call site).
pub(crate) fn gdn_bank_rows() -> usize {
    static R: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        std::env::var("ARF_GDN_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(GDN_BANK_ROWS_DEFAULT)
    })
}

/// Default recurrent-bank rows: 16 concurrently decoding sequences + the reserved row 0 (L206).
///
/// Was 8 (= 7 sequences) until 2026-09-19. 7 made sense when nothing above 8 rows was fast; with
/// the MPP path a 16-row step costs 259 ms against 189 ms for 8, so 16 streams is where a hybrid
/// model's throughput is (29.0 -> 61.7 tok/s aggregate on the 27B). Cost: ~157 MB a row on the
/// 27B, +1.4 GB over the old default. `ARF_GDN_ROWS` still overrides it.
#[cfg(target_os = "macos")]
pub(crate) const GDN_BANK_ROWS_DEFAULT: usize = 17;

/// Tokens of one prompt fed through a committed verify window at a time (hybrid prefill). Equal to
/// `Q4TILE_WIDE_MAX_ROWS`: four 32-row MPP tiles side by side. MEASURED 2026-09-19: 16-row windows
/// gave ~61 prompt tok/s on the 27B; the matmuls of a 128-row window cost ~5.7 ms a token.
#[cfg(target_os = "macos")]
const PREFILL_WINDOW: usize = crate::gpu::concurrent_metal::Q4TILE_WIDE_MAX_ROWS;

/// Rows per windowed-prefill window: `PREFILL_WINDOW` (256); `ARF_PREFILL_WINDOW` (1..=256)
/// overrides it.
///
/// RESOLVED 2026-09-23 — a window wider than 64 rows read the prompt WRONGLY. Found from a
/// long-context recall miss (the LAST fact of a 29,811-token prompt; llama.cpp on the same GGUF,
/// another engine, and our token-by-token prefill all found it) and bisected on window width; the default
/// was capped at 96 while the cause was open. Cause: the GDN projections called `gemv_b` with no
/// L143 row-chunk dims, and the 48-wide `ssm_beta`/`ssm_alpha` gates (never whole MPP tiles) went
/// to `gemv_q4ks_batch`, which DROPS rows >= 64 — stale gates for rows 64.. of every window. Fixed
/// in the record (`gdc_gdn_*`). Teacher-forced perplexity over 10,421 tokens (`ppl_windows`):
/// before 7.04 / 8.81 / 9.82 / 10.14 at windows 16 / 96 / 120 / 128; after 7.042 at 16, 64, 96, 128.
#[cfg(target_os = "macos")]
const PREFILL_WINDOW_SAFE: usize = PREFILL_WINDOW;

/// The actor's hint for the step in flight (`BatchedBackend::set_prefill_logits_needed`): `false` =
/// this prefill chunk does not finish its prompt and its predicted token is never sampled, so its
/// last window takes no lm_head and no separate last-row pass. MEASURED 2026-09-26: at the default
/// 512-token chunks a 7,433-token prompt paid 14 of those 1-row passes — 0.95 s of TTFT (42.13 /
/// 42.15 s vs 41.21 / 41.18 with one 8,192-token chunk). ARF_PREFILL_ALL_LOGITS=1 ignores it (A/B).
#[cfg(target_os = "macos")]
pub(crate) static PREFILL_LOGITS_NEEDED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Clears `verify_window`'s island flags on EVERY exit (2026-09-26). The explicit clears run only
/// on the path that reaches them; a declined submit (`?`) or a missing out bank returned with
/// `gdn_serial_rows` still set, and a serial flag leaked into the next plain multi-stream step
/// chains one stream's recurrence into another's. The explicit clears stay: they also run
/// before the restore and the ring commit, which must not see the verify's flags.
#[cfg(target_os = "macos")]
struct VerifyFlags<'a>(&'a std::sync::Mutex<crate::gpu::concurrent_metal::MetalIsland>);

#[cfg(target_os = "macos")]
impl Drop for VerifyFlags<'_> {
    fn drop(&mut self) {
        if let Ok(g) = self.0.lock() {
            g.set_verify_prefix(None);
            g.set_skip_lm_head(false);
            g.set_gdn_serial_commit_all(false);
            g.set_gdn_serial_rows(false);
            g.set_verify_multi(false);
            g.set_verify_segs(Vec::new());
        }
    }
}
#[cfg(target_os = "macos")]
fn prefill_window_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ARF_PREFILL_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(|n: usize| n.clamp(1, PREFILL_WINDOW))
            .unwrap_or(PREFILL_WINDOW_SAFE)
    })
}

/// Batched decode-row logits on a hybrid model (2026-09-27, `hybrid_decode_rows_logits`): every
/// sequence of an all-decode non-greedy step shares ONE multi-segment record. DEFAULT ON — built,
/// NOT MEASURED when written; measured live the same night (measured 2026-09-27, "Batched decode-row
/// logits on the hybrid model"): 6 concurrent sampled requests 18.8 / 18.4 -> 50.1 / 56.6 tok/s
/// aggregate, level with another engine. `ARF_NO_BATCHED_HYBRID_LOGITS=1` = one record per
/// sequence, as before (A/B).
#[cfg(target_os = "macos")]
fn batched_hybrid_logits_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("ARF_NO_BATCHED_HYBRID_LOGITS").is_none())
}

#[cfg(target_os = "macos")]
/// `ARF_SPEC_DEBUG` — the actor's speculative-step trace; the draft head reports its
/// decline reasons under the same switch.
pub(crate) fn spec_debug() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("ARF_SPEC_DEBUG").is_some())
}

#[cfg(target_os = "macos")]
pub(crate) fn spec_fused_verify() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("ARF_SPEC_FUSED_VERIFY").is_some())
}

#[cfg(target_os = "macos")]
/// When the last verify window returned (ARF_SPEC_PROF only) — see the `[cycle]` line.
static SPEC_LAST_VERIFY_END: std::sync::Mutex<Option<std::time::Instant>> =
    std::sync::Mutex::new(None);

#[cfg(target_os = "macos")]
/// (count, total_ms, total_rows) for verify and decode, plus a one-shot exit dump.
static SPEC_PROF_ACC: std::sync::Mutex<Option<SpecProfAcc>> = std::sync::Mutex::new(None);

#[cfg(target_os = "macos")]
#[derive(Default)]
pub(crate) struct SpecProfAcc {
    v_n: usize,
    v_ms: f64,
    v_rows: usize,
    d_n: usize,
    d_ms: f64,
    dr_n: usize,
    dr_ms: f64,
    vs_n: usize,
    vs_ms: f64,
    vr_n: usize,
    vr_ms: f64,
    rs_n: usize,
    rs_ms: f64,
}

#[cfg(target_os = "macos")]
pub(crate) fn spec_prof_record(kind: SpecProfKind, rows: usize, ms: f64) {
    let mut g = match SPEC_PROF_ACC.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let a = g.get_or_insert_with(SpecProfAcc::default);
    match kind {
        SpecProfKind::Verify => {
            a.v_n += 1;
            a.v_ms += ms;
            a.v_rows += rows;
        }
        SpecProfKind::Decode => {
            a.d_n += 1;
            a.d_ms += ms;
        }
        SpecProfKind::Draft => {
            a.dr_n += 1;
            a.dr_ms += ms;
        }
        SpecProfKind::VerifySubmit => {
            a.vs_n += 1;
            a.vs_ms += ms;
        }
        SpecProfKind::VerifyRead => {
            a.vr_n += 1;
            a.vr_ms += ms;
        }
        SpecProfKind::Restore => {
            a.rs_n += 1;
            a.rs_ms += ms;
        }
    }
    // Periodic dump: a long-lived daemon never "exits", so a periodic line is the only
    // way to read the ratio. Fire on EITHER counter (a verify that never runs is itself
    // the finding — the old `v_n%32 && d_n>0` form printed nothing in exactly that case).
    let tick = match kind {
        SpecProfKind::Verify => a.v_n % 16 == 0,
        SpecProfKind::Decode => a.d_n % 64 == 0,
        SpecProfKind::Draft => false,
        SpecProfKind::VerifySubmit | SpecProfKind::VerifyRead | SpecProfKind::Restore => false,
    };
    if tick {
        let vm = if a.v_n > 0 {
            a.v_ms / a.v_n as f64
        } else {
            f64::NAN
        };
        let dm = if a.d_n > 0 {
            a.d_ms / a.d_n as f64
        } else {
            f64::NAN
        };
        let drm = if a.dr_n > 0 {
            a.dr_ms / a.dr_n as f64
        } else {
            f64::NAN
        };
        eprintln!("[spec-prof] verify n={} mean={:.2}ms (mean_k={:.1}) | draft n={} mean={:.2}ms | decode n={} mean={:.2}ms | RATIO={:.2}x",
            a.v_n, vm, if a.v_n > 0 { a.v_rows as f64 / a.v_n as f64 } else { 0.0 },
            a.dr_n, drm, a.d_n, dm, vm / dm);
        // L331 — the phase split. `submit` is CPU record-build, `read` is the GPU-execution
        // spin, `restore` is the post-record checkpoint rollback (partial accepts only, so its
        // n is ~20% of verify's at p=0.80 and its mean must be weighted by that to compare).
        let mean = |n: usize, t: f64| if n > 0 { t / n as f64 } else { f64::NAN };
        eprintln!(
            "[spec-prof]   phases: submit n={} mean={:.2}ms | read(GPU) n={} mean={:.2}ms | \
             restore n={} mean={:.2}ms (amortised {:.2}ms/window)",
            a.vs_n,
            mean(a.vs_n, a.vs_ms),
            a.vr_n,
            mean(a.vr_n, a.vr_ms),
            a.rs_n,
            mean(a.rs_n, a.rs_ms),
            if a.v_n > 0 {
                a.rs_ms / a.v_n as f64
            } else {
                f64::NAN
            },
        );
    }
}

impl GpuModel {
    /// Wait for every command buffer this model submitted — the island's pipelined ring and the
    /// wgpu queue — before its buffers are released (`BatchedBackend::quiesce`).
    pub fn quiesce(&self) {
        #[cfg(target_os = "macos")]
        if let Some(isl) = self.island.as_ref() {
            if let Ok(isl) = isl.lock() {
                isl.drain_pipeline();
            }
        }
        let _ = self.ctx.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        });
    }

    /// The island owns this model's MoE experts and the wgpu set is the L109 placeholder
    /// (`weights.rs`, `pack_experts`, Q4KS arm): every step that is not plain greedy reaches the
    /// wgpu MoE dispatch and its guard. Read from the buffers themselves, not the environment.
    pub fn island_only_moe(&self) -> bool {
        self.has_island()
            && self.layers.iter().any(|l| match &l.mlp {
                GpuMlp::Moe(m) => {
                    matches!(&m.gate, PackedExperts::Q4KS { codes, .. } if codes.size() <= 16)
                }
                _ => false,
            })
    }

    /// Whether the native-Metal island is attached. Always `false` off macOS, where it does not
    /// exist; lets portable code ask without naming the macOS-only `island` field.
    pub fn has_island(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.island.is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// How many layers to record before flushing the command pass (submit-split granularity).
    /// Each flush is a `queue.submit` with fixed Metal driver cost, so fewer/fatter submits cut
    /// the per-token submission overhead that dominates single-stream decode (llama.cpp uses 1-2
    /// command buffers/token; FLUSH_LAYERS=4 was 12 submits on a 48-layer model). Default 8 (≈6
    /// submits/token); override with ARF_FLUSH_LAYERS. Min 1.
    fn flush_layers_hint(&self) -> usize {
        // PROFILING: flush every layer. Each profiled flush band reserves 2 timestamp
        // slots/dispatch into the 4096-slot query_set (its HARD per-QuerySet cap); a
        // wide band (8 layers × a conc16 MoE batch) overruns it and the band's tail
        // layers get NO timestamp — biasing the per-kernel breakdown. One layer per
        // band (~17 batched dispatches × 16 row-chunks worst case) stays well under cap,
        // so EVERY layer's matmul/MoE/attn is captured. `record_step` accumulates all
        // bands globally, so the total is identical — just split into cap-safe pieces.
        #[cfg(feature = "profiling")]
        if self.ctx.profiling_on() {
            return 1;
        }
        std::env::var("ARF_FLUSH_LAYERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(8)
    }

    /// BATCHED MoE block over all `b` rows in one set of dispatches (bf16 / Q4_K_S).
    ///
    /// Routes & runs the whole batch: router GEMV `normed[b,h]·routerᵀ →
    /// logits[b,num_experts]` → batched top-k route (b workgroups) → batched gate/up
    /// (output `[b, top_k, mi]`, `(row,slot,col)`-major) → swiglu over `b·top_k·mi`
    /// → batched down (weight-sums each row's top_k experts → `mlp_down[b,h]`, written
    /// to `dst` = the trunk's `tmp_h`). The shared expert(s), if any, are run as a
    /// second batched pass (top_k = shared_experts, constant per-row ids [0..sh) +
    /// weight 1.0, accumulate into the same `dst`). Per `(row, output)` the math is
    /// byte-identical to the single-token kernels — so batched == single-stream
    /// bit-exact. `a_gu` owns logits/gate_all/up_all, `a_sc` owns ids/wts/silu/shared
    /// ids/wts (the read+read_write split — see `BatchArenas::moe_silu`); `normed` is the
    /// trunk's rmsnorm'd activation `[b, h]` (row r at `r·h`); `dst` is `tmp_h`.
    fn add_moe_block_batched(
        &self,
        cp: &mut CommandPass,
        k: &GpuKernels,
        uni: &mut GpuArena,
        a_gu: &GpuArena,
        a_sc: &GpuArena,
        ms: &MoeBatchScratch,
        moe: &GpuMoe,
        b: usize,
        h: usize,
        normed: wgpu::BindingResource<'_>,
        dst: wgpu::BindingResource<'_>,
    ) {
        let (ne, tk) = (ms.ne, ms.tk);

        // 1) router logits[b, num_experts] = normed[b,h] · routerᵀ (batched GEMV).
        add_matmul_m(
            cp,
            k,
            uni,
            b,
            normed.clone(),
            &moe.router,
            a_gu.resource(&ms.logits),
            ne,
            h,
            None,
        );

        // 2) batched top-k route: one workgroup per row (b workgroups). Each row's
        // softmax+top-k over logits(A)[r*ne..] → ids(B)[r*tk..] + wts(B)[r*tk..].
        let routed = uni
            .alloc_write(&[ne as u32, tk as u32, moe.norm_topk as u32, 0])
            .expect("moe route dims");
        cp.add_bound(
            &k.moe_route_b,
            "moe_route_b",
            &[
                a_gu.resource(&ms.logits),
                a_sc.resource(&ms.ids),
                a_sc.resource(&ms.wts),
                uni.resource(&routed),
            ],
            [b as u32, 1, 1],
        );

        // High-occupancy grid: the kernels grid-stride, so launching up to the cap
        // splits the work finer (more workgroups busy). Cap < wgpu's 65535/dim.
        let cap = |total: usize| [(total.min(32768) as u32).max(1), 1u32, 1u32];

        // 3) routed experts: batched gate/up → swiglu → down (write into dst).
        self.batched_moe_experts(
            cp,
            k,
            uni,
            a_gu,
            a_sc,
            ms,
            b,
            tk,
            h,
            &moe.gate,
            &moe.up,
            &moe.down,
            a_sc.resource(&ms.ids),
            a_sc.resource(&ms.wts),
            normed.clone(),
            dst.clone(),
            1, // down mode 1 = write
            cap,
        );

        // 4) shared expert(s): a second batched pass with top_k = shared_experts,
        // the per-row constant ids [0..sh) + weight 1.0, accumulating into dst
        // (mode 2). Reuses gate_all/up_all/silu (sh ≤ top_k, so they fit).
        if moe.shared_experts > 0 {
            let sgate = moe.shared_gate.as_ref().expect("shared_gate");
            let sup = moe.shared_up.as_ref().expect("shared_up");
            let sdown = moe.shared_down.as_ref().expect("shared_down");
            self.batched_moe_experts(
                cp,
                k,
                uni,
                a_gu,
                a_sc,
                ms,
                b,
                moe.shared_experts,
                h,
                sgate,
                sup,
                sdown,
                a_sc.resource(&ms.sh_ids),
                a_sc.resource(&ms.sh_wts),
                normed,
                dst,
                2, // down mode 2 = accumulate (routed experts already wrote dst)
                cap,
            );
        }
    }

    /// One batched SwiGLU expert pass: gate/up `[b, n_slots, mi]` → swiglu →
    /// weight-summed down `[b, h]`. `n_slots` is the per-row routed-expert count
    /// (top_k for the routed pass, shared_experts for the shared pass); `ids`/`wts`
    /// are the matching `[b·n_slots]` routing buffers. `down_mode` (1 write / 2
    /// accumulate) selects how the down writes `dst`.
    fn batched_moe_experts(
        &self,
        cp: &mut CommandPass,
        k: &GpuKernels,
        uni: &mut GpuArena,
        a_gu: &GpuArena,
        a_sc: &GpuArena,
        ms: &MoeBatchScratch,
        b: usize,
        n_slots: usize,
        h: usize,
        gate: &PackedExperts,
        up: &PackedExperts,
        down: &PackedExperts,
        ids: wgpu::BindingResource<'_>,
        wts: wgpu::BindingResource<'_>,
        normed: wgpu::BindingResource<'_>,
        dst: wgpu::BindingResource<'_>,
        down_mode: u32,
        cap: impl Fn(usize) -> [u32; 3],
    ) {
        let mi = ms.mi;
        // gate/up output [b, n_slots, mi] (row,slot,col)-major; grid covers
        // b·n_slots·mi indices. gate/up don't apply the router weight (the down does),
        // so these kernels have NO wts binding. dims: m=b, k=h, n=mi, top_k=n_slots.
        let gu_total = b * n_slots * mi;
        fn batched_gu_b(
            cp: &mut CommandPass,
            k: &GpuKernels,
            uni: &mut GpuArena,
            experts: &PackedExperts,
            a: wgpu::BindingResource<'_>,
            c: wgpu::BindingResource<'_>,
            ids: wgpu::BindingResource<'_>,
            b: usize,
            kdim: usize,
            n: usize,
            n_slots: usize,
            label: &str,
            groups: [u32; 3],
        ) {
            // L109 SAFETY GUARD — the wgpu expert set is a PLACEHOLDER when the island owns MoE
            // (see weights.rs `pack_experts` Q4KS arm). Reaching a dispatch here in that state
            // would silently compute garbage, so fail loudly instead. Measured: this path takes
            // 0 dispatches with the island up, 8 with ARF_NO_MSL_GEMV=1.
            #[cfg(target_os = "macos")]
            if std::env::var_os("ARF_MSL_GEMV").is_some()
                && std::env::var_os("ARF_MOE_WGPU_FULL").is_none()
            {
                panic!(
                    "wgpu MoE dispatch ({label}) reached while the island owns the MoE path — \
                     the wgpu expert set is an L109 placeholder and would compute garbage. The \
                     serving loop keeps such a model on greedy steps (`greedy_only`); reaching \
                     this is a bug. ARF_MOE_WGPU_FULL=1 restores the full expert set, a second \
                     ~16 GB copy of the experts, which swaps on a 36 GB Mac."
                );
            }
            let dims = uni
                .alloc_write(&[b as u32, kdim as u32, n as u32, 0, 0, n_slots as u32, 0, 0])
                .expect("moe gu_b dims");
            match experts {
                PackedExperts::Bf16(buf) => cp.add_bound(
                    &k.matmul_vec_moe_batch_gu_b,
                    label,
                    &[a, buf.as_entire_binding(), c, uni.resource(&dims), ids],
                    groups,
                ),
                PackedExperts::Q4KS {
                    codes,
                    scales,
                    mins,
                    dd,
                } => {
                    let bind = [
                        a,
                        codes.as_entire_binding(),
                        c,
                        uni.resource(&dims),
                        scales.as_entire_binding(),
                        ids,
                        mins.as_entire_binding(),
                        dd.as_entire_binding(),
                    ];
                    // The nr0/NSG-layout v2 kernel is the DEFAULT: each workgroup owns
                    // INDICES_PER_WG = GROUPS*NR0 output columns (GROUPS=2, NR0=4 → 8),
                    // so divide the index cap by 8 (MUST match the v2 kernel's const).
                    // MEASURED on the q4ks batched serve path (the real conc path):
                    // gate/up GEMV 1.37x faster, +6.7% conc16 (104.7→111.7) / +10.7%
                    // conc32 (111.3→123.2), greedy-token-identical to v1. ARF_MOE_GU_V2=0
                    // forces the v1 ROWS_PER_WG=2 kernel (bit-exact-vs-oracle A/B / fallback).
                    if std::env::var("ARF_MOE_GU_V2").ok().as_deref() != Some("0") {
                        const INDICES_PER_WG_V2: u32 = 8; // GROUPS(2) * NR0(4)
                        let g = [
                            groups[0].div_ceil(INDICES_PER_WG_V2).max(1),
                            groups[1],
                            groups[2],
                        ];
                        cp.add_bound(&k.matmul_vec_moe_q4ks_batch_gu_b_v2, label, &bind, g)
                    } else {
                        // Multi-row (ROWS_PER_WG=2 over columns within a (row,slot)):
                        // the kernel grid-strides over GROUPS of 2, so halve the count.
                        const ROWS_PER_WG: u32 = 2;
                        let g = [groups[0].div_ceil(ROWS_PER_WG).max(1), groups[1], groups[2]];
                        cp.add_bound(&k.matmul_vec_moe_q4ks_batch_gu_b, label, &bind, g)
                    }
                }
                // Q4/Q4K never reach the batched path (caller gates bf16|Q4KS).
                _ => unreachable!("batched MoE gate/up only supports bf16 + Q4_K_S"),
            }
        }
        batched_gu_b(
            cp,
            k,
            uni,
            gate,
            normed.clone(),
            a_gu.resource(&ms.gate_all),
            ids.clone(),
            b,
            h,
            mi,
            n_slots,
            "moe_gate_all_b",
            cap(gu_total),
        );
        batched_gu_b(
            cp,
            k,
            uni,
            up,
            normed,
            a_gu.resource(&ms.up_all),
            ids.clone(),
            b,
            h,
            mi,
            n_slots,
            "moe_up_all_b",
            cap(gu_total),
        );
        // swiglu over all b·n_slots·mi elements (elementwise, same kernel). Reads
        // gate_all,up_all(A), writes silu(B) — distinct buffers, no usage conflict.
        let swd = uni
            .alloc_write(&[gu_total as u32, 0, 0, 0])
            .expect("moe swd");
        cp.add_bound(
            &k.swiglu,
            "moe_swiglu_all_b",
            &[
                a_gu.resource(&ms.gate_all),
                a_gu.resource(&ms.up_all),
                a_sc.resource(&ms.silu),
                uni.resource(&swd),
            ],
            [(gu_total as u32).div_ceil(256), 1, 1],
        );
        // batched down: per (row, h_j), weight-sum over this row's n_slots experts →
        // dst[row*h + h_j]. dims: m=b, k=mi, n=h, mode=down_mode, top_k=n_slots.
        let dd = uni
            .alloc_write(&[
                b as u32,
                mi as u32,
                h as u32,
                0,
                down_mode,
                n_slots as u32,
                0,
                0,
            ])
            .expect("moe down_b dims");
        match down {
            PackedExperts::Bf16(buf) => cp.add_bound(
                &k.matmul_vec_moe_down_reduce_b,
                "moe_down_reduce_b",
                &[
                    a_sc.resource(&ms.silu),
                    buf.as_entire_binding(),
                    dst,
                    uni.resource(&dd),
                    ids,
                    wts,
                ],
                cap(b * h),
            ),
            PackedExperts::Q4KS {
                codes,
                scales,
                mins,
                dd: ddw,
            } => cp.add_bound(
                &k.matmul_vec_moe_q4ks_down_reduce_b,
                "moe_down_reduce_b",
                &[
                    a_sc.resource(&ms.silu),
                    codes.as_entire_binding(),
                    dst,
                    uni.resource(&dd),
                    scales.as_entire_binding(),
                    ids,
                    wts,
                    mins.as_entire_binding(),
                    ddw.as_entire_binding(),
                ],
                cap(b * h),
            ),
            _ => unreachable!("batched MoE down only supports bf16 + Q4_K_S"),
        }
    }

    /// Run a ragged batch of sequences through the trunk in one batched pass and
    /// return each sequence's last-token logits `[vocab]`. Mirrors
    /// `Llama::forward` + `logits_last`, the correctness oracle.
    ///
    /// `input_ids` is the flattened batch (`total = Σ q_len`); `batch.seqs` gives
    /// each sequence's `q_start`/`q_len`/`past_len`/`slots`. The batch-ready
    /// kernels (embed, rmsnorm, matmul, rope, swiglu, residual) run over all
    /// `total` rows at once; attention is run per sequence with the single-stream
    /// kernel (`batched_attention` replaces this loop where it is enabled).
    /// This step's K/V is written into the resident pool, so a later step reads it.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(level = "info", name = "gpu_forward_batch", skip_all, fields(total = input_ids.len(), seqs = batch.seqs.len()))
    )]
    pub fn forward_batch(
        &self,
        input_ids: &[u32],
        batch: &arf_core::model::batch::ForwardBatch,
    ) -> Vec<Vec<f32>> {
        self.forward_batch_impl(input_ids, batch, false, false)
    }

    /// GREEDY serve path: forward + ON-GPU batched argmax, returning each sequence's sampled token
    /// id directly (NO full-vocab readback). The dominant single-stream speedup — `step()` routes
    /// here when every sequence is plain greedy. Returns one length-1 vec per row holding the id
    /// bit-cast to f32 (the `step()` wrapper unwraps it); empty downstream logits are intentional.
    pub fn forward_batch_greedy(
        &self,
        input_ids: &[u32],
        batch: &arf_core::model::batch::ForwardBatch,
    ) -> Vec<u32> {
        self.forward_batch_impl(input_ids, batch, false, true)
            .into_iter()
            .map(|row| row.first().map(|f| f.to_bits()).unwrap_or(0))
            .collect()
    }

    /// Like [`Self::forward_batch`] but returns the logits for EVERY input position
    /// (`total` rows), not just each sequence's last. Used by GPU speculative decode
    /// to verify a draft window in one pass (it needs the greedy argmax after each
    /// drafted token). `batch` must hold a single sequence (q_start 0).
    pub fn forward_batch_all(
        &self,
        input_ids: &[u32],
        batch: &arf_core::model::batch::ForwardBatch,
    ) -> Vec<Vec<f32>> {
        self.forward_batch_impl(input_ids, batch, true, false)
    }

    /// B=1 SINGLE-CHAT fast path: route a lone greedy decode step through the m=1 megakernel
    /// (`decode_token`) — the whole token in ONE island command buffer with the single-stream-tuned
    /// kernels (fused scatter+attention, m=1 MoE GEMV, Q8 lm_head GEMV). This is the ~90 tok/s
    /// path; the B-row machinery at B=1 measures ~58 (empty-tile MoE GEMM, per-token B-row
    /// uniforms). The daemon's paged slots are injected via `decode_block_table` — the same hook
    /// `generate_batch` uses — recovered from the seq's slot list (block-affine by construction:
    /// `slots[p] = tbl[p/bs]*bs + p%bs`, so `tbl[j] = slots[j*bs]/bs`). Requires ARF_MEGAKERNEL
    /// (the serve daemon defaults it on). Falls back to the B-row path when ineligible.
    /// DEFAULT-ON: same-window A/B (2026-07-05, coder-30B, 94 warm tokens) = bridge p50/p90/p95/p99
    /// 67/49/47/47 vs B-row 52/46/46/27 — wins every percentile, and the 96-token stream is
    /// bit-identical to the B-row reference (checksum-equal). ARF_NO_M1_CHAT=1 opts out (A/B).
    /// Known headroom: the decode.rs flush+poll before the island fires every CPU-fed token
    /// (the ~21ms cluster) — island-side embed removes it (follow-up).
    #[cfg(target_os = "macos")]
    // The two `next_token_mtl` / `out_tokens_mtl` unwraps here are each guarded by an `is_some()`
    // ~150 lines above them, not on the adjacent line. Clippy's rewrite is to bind the value in the
    // guard, but that holds a borrow of `self.scratch` across the whole burst block, which also
    // touches `self` mutably — so taking the suggestion means restructuring a hot path to satisfy a
    // lint. The invariant is real and the scrutinee is never reassigned in between; allowed here
    // rather than at the call sites so the reason is stated once.
    #[allow(clippy::unnecessary_unwrap)]
    fn try_m1_megakernel(
        &self,
        input_ids: &[u32],
        batch: &arf_core::model::batch::ForwardBatch,
    ) -> Option<Vec<u32>> {
        // 8-bit KV: the m=1 megakernel's attention/scatter read the f32 pool, which is a
        // placeholder then — single-row decode goes through the batched record instead.
        if self.kv.q8.iter().any(Option::is_some) {
            return None;
        }
        if std::env::var_os("ARF_NO_M1_CHAT").is_some() {
            return None;
        }
        // off while group-64 weights are loaded — see `megakernel_on`
        if !crate::weights::megakernel_on() {
            return None;
        }
        if batch.seqs.len() != 1 || input_ids.len() != 1 {
            return None;
        }
        let s = &batch.seqs[0];
        if s.q_len != 1 {
            return None;
        }
        self.island.as_ref()?;
        if self.kv.kv_quant.is_quantized() {
            return None;
        }
        // L140 — HYBRID (gated-delta-net) models are REFUSED here. This path has no recurrent
        // branch: decode.rs builds its MegaLayer set with `gdn: None` hardcoded (decode.rs:328),
        // so it would run ATTENTION on all 64 trunk layers — including the 48 recurrent ones whose
        // `q_proj` is really the SSM's v-block. That is exactly the observed failure: one plausible
        // token, then `!!!!!` forever. Falling through to try_batched_megakernel (which DOES have
        // the L136 GDN branch) is correct, not a slow path.
        //
        // Structural, not cosmetic: this mirrors what llama.cpp does with layer filters — a hybrid
        // arch must never reach code that assumes every layer has attention.
        if self.layers.iter().any(|l| l.gdn.is_some()) {
            return None;
        }
        let bs = self.kv.block_size;
        let n = s.past_len + 1; // context including this token
        if s.slots.len() < n {
            return None;
        }
        // Recover the block table; bail (misaligned slot) rather than assume.
        let nblk = n.div_ceil(bs);
        let mut tbl = Vec::with_capacity(nblk);
        for j in 0..nblk {
            let s0 = s.slots[j * bs];
            if !(s0 as usize).is_multiple_of(bs) {
                return None;
            }
            tbl.push(s0 / bs as u32);
        }
        // L7b (2026-08-04) — PAGED tables SUPPORTED (the conc1 fix). The old IDENTITY-ONLY bail
        // here was why the daemon's m=1 megakernel never ran: the LIFO BlockAllocator hands the
        // first request the warmup's recycled blocks in REVERSE order (tbl = [2,1,0,3,…]), so
        // after the pristine warmup NO real request is identity and every conc1 decode token fell
        // to the B-row path at ~14.6 ms (measured 2026-08-04: arms A/A'/B/C all 55.7-59.3 — every
        // m=1 env lever was a no-op because the path itself never engaged; certification 55.9 vs
        // the 70-76 prediction). The island record already routes ALL pool access through its
        // slots ring (attention reads) + the L7b write-slot uniform (scatter), so paged support
        // is: hand it the mapped table for THIS call (`set_m1_slot_map`, one-shot, consumed by
        // the record's uniform patch). Identity passes None — byte-identical to the old fill.
        // The KTOK burst never crosses a block boundary (`burst = ktok.min(bs - pos%bs)` below),
        // so `map[0..nblk*bs)` covers every position the record touches (attention reads 0..=pos
        // +k-1, scatter writes pos..pos+k-1). ARF_M1_IDENTITY_ONLY=1 restores the old bail
        // (A/B lever).
        let identity = tbl.iter().enumerate().all(|(j, &t)| t as usize == j);
        if !identity && std::env::var_os("ARF_M1_IDENTITY_ONLY").is_some() {
            return None;
        }
        let slot_map: Option<Vec<u32>> = if identity {
            None
        } else {
            let mut m = Vec::with_capacity(nblk * bs);
            for &t in &tbl {
                let base = t * bs as u32;
                m.extend((0..bs as u32).map(|o| base + o));
            }
            Some(m)
        };
        // SINGLE-QUEUE upgrade (ARF_MEGA_SINGLEQ): feed the token through the island's
        // next_token buffer and run the token=None feedback path — the island embeds it as op 0,
        // NO wgpu work is queued this token, so decode.rs skips the flush+poll whose periodic
        // deferred-upload hit is the ~21ms p90 spike cluster. The CPU write is safe here: the
        // previous bridge step fully fenced its island work before returning (blocking wait, or
        // the L1 wait_m1_step ring fence below) — the island queue is idle when we write.
        // The 4-byte-read fast branch REQUIRES an island fence: without one the m=1 command
        // buffers pile up unwaited, which measured as BOTH nondeterministic reads (next_token
        // raced in-flight argmax writes: 0x9993/0x7cf1 stream forks) AND 5-31s p99 stalls
        // (queue-backpressure drains). Two fences qualify:
        //   - ARF_MEGA_BLOCKING (the old per-token waitUntilCompleted inside the record;
        //     reference-exact checksum 0xe498…, p99 ~14-25ms), or
        //   - L1 RING FENCE (default): the record runs the NOWAIT + async-completion ring
        //     (set_m1_pipelined) and this bridge spins on THIS step's slot_done via wait_m1_step
        //     before any CPU read — at most one buffer in flight, every read behind a completed
        //     Release/Acquire handshake. This was the "unwaited buffers" fix the old comment
        //     deferred; it unblocks the daemon's forced ARF_MEGA_BLOCKING=1 (main.rs) for the
        //     m=1 stream only. Predicted (2026-08-04): daemon conc1 53.8 →
        //     70-76 tok/s (the in-tree pipelined-vs-blocking measurement is 57→75). NOT yet
        //     measured — perf measurement gated on the post-reboot clean race.
        // ARF_M1_BLOCKING=1 disarms the ring fence (A/B lever): with the daemon's BLOCKING
        // env that is exactly the pre-L1 behavior; standalone (no BLOCKING) it routes to the
        // self-fencing logits-read fallback below, as before.
        fn m1_ring_fenced() -> bool {
            static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *V.get_or_init(|| std::env::var_os("ARF_M1_BLOCKING").is_none())
        }
        let singleq_tok = std::env::var_os("ARF_MEGA_SINGLEQ").is_some()
            && (std::env::var_os("ARF_MEGA_BLOCKING").is_some() || m1_ring_fenced());
        // ONE-TIME diagnostic (ARF_TRACE_BURST): which single-stream path does token 0 take?
        if std::env::var_os("ARF_TRACE_BURST").is_some() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static ONCE: AtomicBool = AtomicBool::new(false);
            if !ONCE.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[trace-burst] singleq_tok={} next_token_mtl={} → {}",
                    singleq_tok,
                    self.scratch.next_token_mtl.is_some(),
                    if singleq_tok && self.scratch.next_token_mtl.is_some() {
                        "BURST path"
                    } else {
                        "logits-read path (slow)"
                    }
                );
            }
        }
        if singleq_tok && self.scratch.next_token_mtl.is_some() {
            use objc2_metal::MTLBuffer as _;
            // MILESTONE 3 — K-TOKEN BURST: serve from the stash when a previous burst already
            // decoded this position (zero GPU work; the greedy feedback loop makes the stash
            // exact — the scheduler feeds back precisely the token we returned).
            {
                let mut stash = self.m1_burst_stash.borrow_mut();
                // Serve only on position AND fed-token match: a position-only hit could
                // cross a stream boundary (stale lookahead from an ended/edited chat whose
                // prompt length collides), emitting an old-timeline token with no KV write.
                if let Some(i) = stash
                    .iter()
                    .position(|&(p, feed, _)| p == s.past_len && feed == input_ids[0])
                {
                    let (_, _, tok) = stash.remove(i);
                    stash.retain(|&(p, _, _)| p > s.past_len);
                    return Some(vec![tok]);
                }
                stash.clear(); // no entry for this step → any leftovers are stale
            }
            // MILESTONE 4 — N-GRAM SELF-SPEC (ARF_MEGA_SPEC): draft a chain from the
            // running history and verify `[t, d0..dj]` in ONE shared-prefix megakernel pass.
            // The accept rule (draft[i] == preds[i], stop at first miss) makes the emitted
            // stream greedy-exact WITHIN the verify kernel stack. CAVEAT (why this stays
            // opt-in): preds come from the B-row kernels, whose logits sit ~4e-5 from the
            // m=1 island's (batched_mega_parity) — a near-tie argmax can pick differently
            // than the burst path would, forking the stream. Battery-gated, not structural.
            // One model pass emits 1+accepted tokens (the burst below pays one pass per
            // token). KV for accepted rows is exact; rejected-tail rows hold wrong-token KV
            // at positions the stream hasn't reached, which the next pass overwrites before
            // any attention read (same phantom-KV argument as the burst).
            // L7b: spec-verify stays IDENTITY-ONLY — its verify window builds identity slots
            // for the B-row path (below), which does not consume the m=1 slot map. Opt-in +
            // rare; paged spec support is its own (unscoped) follow-up.
            // L11 diagnostic (ARF_SPEC_PROF): `identity` is the gate that kept spec dead in
            // the daemon — it is TRUE only for the pristine warmup request and FALSE for every
            // real one (LIFO block recycling). Logged sparsely so the finding stays reproducible.
            if spec_prof() {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static GN: AtomicUsize = AtomicUsize::new(0);
                let gn = GN.fetch_add(1, Ordering::Relaxed);
                if gn < 8 || gn.is_multiple_of(256) {
                    eprintln!(
                        "[spec-prof] gate#{gn}: identity={} past_len={}",
                        identity, s.past_len
                    );
                }
            }
            // L11 (2026-08-04) — PAGED spec-verify. The old gate was `identity &&`, which meant
            // spec NEVER ran in the daemon: the LIFO BlockAllocator hands every post-warmup
            // request recycled blocks in reverse order, so `identity=false` from the first real
            // request onward (measured: gate#0..6 identity=true = the warmup only; gate#7+ all
            // false). Verify does NOT need identity — unlike the m=1 record (whose uniform slot
            // ring L7b had to teach paging), `try_batched_megakernel_verify` feeds the B-ROW
            // record, which gathers `seq_slots[r]` per row into slots_all
            // (concurrent_metal.rs:1424/1540) and therefore honours ARBITRARY paged tables
            // already. So the window just has to carry the seq's REAL slots instead of a
            // synthetic 0..n identity range (see the call below). ARF_SPEC_IDENTITY_ONLY=1
            // restores the old identity-only bail (A/B lever).
            // Paged windows only when the L11 flag is armed (OFF ⇒ the old identity-only gate,
            // byte-identical). ARF_SPEC_IDENTITY_ONLY=1 forces identity even with L11 on.
            let spec_paged_ok =
                spec_fused_verify() && std::env::var_os("ARF_SPEC_IDENTITY_ONLY").is_none();
            if (identity || spec_paged_ok) && std::env::var_os("ARF_MEGA_SPEC").is_some() {
                // Same block-boundary collision guard as the burst: the window's KV writes
                // must stay inside the seq's current (scheduler-reserved) KV block.
                let room = bs - (s.past_len % bs);
                let kmax = std::env::var("ARF_MEGA_SPEC_K")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(7usize)
                    .min(room.saturating_sub(1))
                    .min(15); // verify KMAX=16 rows incl. the committed token
                let draft = {
                    let g = self.m1_spec.borrow();
                    match g.as_ref() {
                        // history must reach past_len inclusive (the observer just pushed t).
                        Some(sp) if kmax >= 1 && sp.base + sp.history.len() == s.past_len + 1 => {
                            sp.drafter.propose(&sp.history, kmax)
                        }
                        _ => Vec::new(),
                    }
                };
                // L11 diagnostic: separates "history misaligned" from "drafter proposed nothing"
                // — the two ways a drafted window silently fails to materialise.
                if spec_prof() {
                    use std::sync::atomic::{AtomicUsize, Ordering};
                    static N: AtomicUsize = AtomicUsize::new(0);
                    let n = N.fetch_add(1, Ordering::Relaxed);
                    if n < 8 || n.is_multiple_of(256) {
                        let g = self.m1_spec.borrow();
                        let (base, hlen) = g
                            .as_ref()
                            .map(|sp| (sp.base, sp.history.len()))
                            .unwrap_or((0, 0));
                        eprintln!("[spec-prof] draft#{n}: kmax={kmax} room={room} draft_len={} aligned={}",
                            draft.len(), base + hlen == s.past_len + 1);
                    }
                }
                if !draft.is_empty() {
                    let mut window = Vec::with_capacity(1 + draft.len());
                    window.push(input_ids[0]);
                    window.extend_from_slice(&draft);
                    // L11 — REAL (paged) slot table for the window: [0, past_len+k).
                    // `s.slots` is the scheduler's own table for this seq, so positions
                    // [0, past_len] come straight from it. The k-1 lookahead positions past
                    // `past_len` may sit beyond what the scheduler materialised, but the
                    // `room` guard above caps the window INSIDE the seq's current KV block,
                    // so every one of them is `tbl[last_block]*bs + offset` — derived from
                    // the same block the committed token writes into, i.e. reserved for this
                    // seq. Build them from the block table rather than assuming identity.
                    let slots: Vec<u32> = {
                        let need = s.past_len + window.len();
                        let mut v: Vec<u32> = Vec::with_capacity(need);
                        v.extend_from_slice(&s.slots[..(s.past_len + 1).min(s.slots.len())]);
                        // Extend inside the current block (guaranteed by `room`).
                        while v.len() < need {
                            let prev = *v.last()?;
                            // Same block, next offset — `room` guarantees no boundary cross.
                            if (prev as usize + 1).is_multiple_of(bs) && v.len() + 1 < need {
                                return None;
                            }
                            v.push(prev + 1);
                        }
                        v
                    };
                    debug_assert_eq!(slots.len(), s.past_len + window.len());
                    if let Some(preds) =
                        self.try_batched_megakernel_verify(&window, s.past_len, &slots, s.stream_id)
                    {
                        let mut acc = 0usize;
                        while acc < draft.len() && draft[acc] == preds[acc] {
                            acc += 1;
                        }
                        let mut stash = self.m1_burst_stash.borrow_mut();
                        // Entry for position p+i: the scheduler will feed preds[i-1] there.
                        for i in 1..=acc {
                            stash.push((s.past_len + i, preds[i - 1], preds[i]));
                        }
                        return Some(vec![preds[0]]);
                    }
                    // Verify couldn't run (island busy/ineligible) → fall through to the burst.
                }
            }
            // L1 — arm the NOWAIT ring for THIS step's record (scoped: the wait_m1_step fence
            // after the record disarms it). Under the daemon's forced ARF_MEGA_BLOCKING=1
            // this override is what actually flips the m=1 record to the pipelined path.
            // L7b — hand the record THIS call's paged slot map (one-shot: the record's uniform
            // patch consumes it, so it can never leak onto an unrelated record). Set AFTER the
            // stash/spec early-returns (no GPU work → nothing to map) and BEFORE the burst /
            // 4-byte-read decode calls below. None (identity) also clears defensively.
            if let Some(isl) = self.island.as_ref() {
                let g = isl.lock().unwrap();
                if m1_ring_fenced() {
                    g.set_m1_pipelined(true);
                }
                g.set_m1_slot_map(slot_map);
            }
            let nt = self.scratch.next_token_mtl.as_ref().unwrap();
            unsafe {
                *(nt.0.contents().as_ptr() as *mut u32) = input_ids[0];
            }
            if self.scratch.out_tokens_mtl.is_some() {
                // Burst k tokens in ONE island command buffer (decode_tokens_k: on-GPU next_token
                // chaining, tokens stored to out_tokens[1..=k]) — ONE fence per k tokens instead
                // of per token. COLLISION GUARD: burst only within the seq's CURRENT KV block,
                // so we never write a slot in a block the scheduler hasn't reserved for this
                // seq; at a block boundary burst=1. (L7b: also what keeps the paged slot_map's
                // nblk*bs entries covering every burst position.)
                // DEFAULT k=1 (2026-08-05, L30) — THE BURST IS NOW A PESSIMIZATION. Measured ABBA
                // on the DAEMON (real prompts, REPS=3, rep 1 discarded, fresh daemon per arm):
                //   k=1: 82.4/81.8 and 82.4/82.1  (mean 82.2)   ← DEFAULT
                //   k=8: 64.8/64.8 and 64.9/64.9  (mean 64.9)
                //   +26.9%, same-config spread 0.2%/0.15% — the effect is ~130x the noise.
                //   Monotonic in k: k=16 58.3 · k=8 64.4 · k=4 68.4 · k=1 81.4. Text
                //   byte-identical across all arms (greedy unchanged — pure overhead, no
                //   quality dimension).
                //
                // WHY IT INVERTED (the premise expired, the measurement did not): decode.rs:387
                // states the burst's purpose — "kills the per-buffer ~20ms inter-buffer GPU idle
                // gap". That gap was REAL when this was written and the old in-engine numbers
                // below were honest. Since then L16 measured 93-99% GPU-BUSY with 6-24 MICROsecond
                // inter-encoder gaps, and a later A/B re-confirmed it (+4.6% only from un-blocking the
                // batched queue). The 20ms gap is GONE — fixed by other work — so the burst now
                // pays on-GPU serialization (each sub-token's attention waits on the previous
                // sub-token's KV write inside one command buffer) to amortize a fence that L1's
                // ring (wait_m1_step) already made cheap. Cost with no remaining benefit.
                //
                // SUPERSEDED in-engine numbers (2026-07-06, coder-30B, 96-tok greedy, checksum
                // 0xe498…), kept because they were correct under the old premise: k=1 12.1/13.0ms,
                // k=4 12.4/11.75, k=8 11.4/11.5 (~87 tok/s, +9%); fence ≈1.26ms ÷k, kernel ≈11.3ms.
                // Set ARF_MEGA_KTOK=8 to restore the burst (A/B lever).
                //
                // L4a — the env parse was a per-burst getenv+String+parse on the hot path; the
                // flag is process-constant, so read it once.
                static KTOK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                let ktok = *KTOK.get_or_init(|| {
                    std::env::var("ARF_MEGA_KTOK")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1usize)
                        .max(1)
                });
                let burst = ktok.min(bs - (s.past_len % bs)).max(1);
                // L11: time the burst record; charge it per TOKEN produced so the decode
                // mean is a per-token step cost directly comparable to a verify window.
                let dt0 = spec_prof().then(std::time::Instant::now);
                let did = self.decode_tokens_k(s.past_len, 0, burst);
                let _ = self.ctx.device.poll(wgpu::PollType::Poll); // wgpu housekeeping (see below)
                                                                    // FENCE the island before reading out_tokens: decode_tokens_k commits the k-token
                                                                    // burst on the island's Metal queue; reading out_tokens.contents() before it
                                                                    // completes = the same wgpu<->Metal race as the fallback path ("Buffer is not
                                                                    // mapped" / stale reads). L1: wait_m1_step = single-slot slot_done spin on THIS
                                                                    // burst's command buffer (the whole k-token burst is ONE buffer at ring slot
                                                                    // past_len % MEGA_DEPTH) — replaces the old FULL drain_pipeline waitUntilCompleted
                                                                    // sweep, and also disarms the per-call ring override. On the ARF_M1_BLOCKING
                                                                    // path the record already waited internally and this returns immediately.
                if let Some(isl) = self.island.as_ref() {
                    isl.lock().unwrap().wait_m1_step(s.past_len);
                }
                // L11: fence closed => the burst's GPU work is complete. Charge per token.
                if let Some(t0) = dt0 {
                    let ms = t0.elapsed().as_secs_f64() * 1e3 / (did.max(1) as f64);
                    for _ in 0..did.max(1) {
                        spec_prof_record(SpecProfKind::Decode, 1, ms);
                    }
                }
                let ot = self.scratch.out_tokens_mtl.as_ref().unwrap();
                let toks: Vec<u32> = (1..=did)
                    .map(|i| unsafe { *(ot.0.contents().as_ptr() as *const u32).add(i) })
                    .collect();
                let mut stash = self.m1_burst_stash.borrow_mut();
                // Entry for position p+j: the scheduler will feed toks[j-1] there.
                for j in 1..toks.len() {
                    stash.push((s.past_len + j, toks[j - 1], toks[j]));
                }
                return Some(vec![toks[0]]);
            }
            // MILESTONE 2 (single-token flow, when out_tokens isn't available): the island's own
            // argmax (island_sample) wrote the sampled token into next_token — read the 4 bytes
            // directly instead of the 600KB logits readback + CPU argmax. Coherent because a
            // fence completes before the read: the BLOCKING record's internal wait, or the L1
            // wait_m1_step slot_done spin below (which also disarms the ring override).
            //
            // Maintenance poll: the removed logits readback was the ONLY per-token device.poll();
            // without any polls wgpu's staging/cleanup debt accumulated into multi-second forced
            // drains (the p99 stalls). A non-blocking poll each token (~µs) keeps it incremental.
            self.decode_token(None, s.past_len, true, None, None);
            let _ = self.ctx.device.poll(wgpu::PollType::Poll);
            if let Some(isl) = self.island.as_ref() {
                isl.lock().unwrap().wait_m1_step(s.past_len);
            }
            let tok = unsafe { *(nt.0.contents().as_ptr() as *const u32) };
            return Some(vec![tok]);
        }
        let _tr = std::env::var_os("ARF_TRACE_BURST").is_some();
        if _tr {
            eprintln!(
                "[trace] fallback: decode_token start (past_len={})",
                s.past_len
            );
        }
        // L7b — same one-shot slot-map handoff for the logits-read fallback's record (this path
        // is only reached when the singleq branch above did NOT run, so the map is still ours).
        if let Some(isl) = self.island.as_ref() {
            isl.lock().unwrap().set_m1_slot_map(slot_map);
        }
        self.decode_token(Some(input_ids[0]), s.past_len, true, None, None);
        // FENCE the island's Metal queue before wgpu reads its logits output. decode_token's
        // megakernel writes `logits` on the island's SEPARATE MTLCommandQueue; without this drain,
        // the wgpu read_f32 copy races the island write → map_async never completes → "Buffer is
        // not mapped" panic / hang (the B=1 single-stream bug, 2026-07-08). drain_pipeline does a
        // Metal-native waitUntilCompleted on the island ring — the correct cross-queue fence.
        if let Some(isl) = self.island.as_ref() {
            isl.lock().unwrap().drain_pipeline();
        }
        if _tr {
            eprintln!("[trace] fallback: decode_token DONE, read_f32 start");
        }
        // Greedy argmax on the read-back logits (the non-SINGLEQ fallback).
        let logits = self.ctx.read_f32(&self.scratch.logits, self.cfg.vocab_size);
        if _tr {
            eprintln!("[trace] fallback: read_f32 DONE");
        }
        let tok = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)?;
        Some(vec![tok])
    }

    /// MILESTONE 4 — n-gram self-spec observer (ARF_MEGA_SPEC). Maintains the single
    /// chat seq's token window (`history[i]` = token fed at position `base + i`) and keys
    /// the drafter incrementally. A rewind truncates (and clears the lookahead stash — the
    /// old timeline's tokens must not serve on the new one); a position GAP (prefix-cache
    /// hit, or steps that ran inside a multi-seq batch) RE-SEEDS the window at the current
    /// batch instead of disabling drafting for the rest of the sequence — n-gram keys are
    /// position-independent, so a mid-stream window drafts fine. Correctness never depends
    /// on this state, only draft quality.
    #[cfg(target_os = "macos")]
    fn m1_spec_observe(&self, input_ids: &[u32], s: &arf_core::model::batch::SeqAttn) {
        use crate::gpu::Drafter;
        let order: usize = std::env::var("ARF_MEGA_SPEC_ORDER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3)
            .max(2);
        let mut g = self.m1_spec.borrow_mut();
        let sp = g.get_or_insert_with(|| crate::gpu::M1Spec {
            base: 0,
            history: Vec::new(),
            drafter: Drafter::new(order),
            observed: 0,
        });
        let end = sp.base + sp.history.len();
        if s.past_len < end {
            // Rewind (edited turn): drop the abandoned timeline — window, stale drafter
            // keys past the watermark, and any stashed lookahead computed under it.
            self.m1_burst_stash.borrow_mut().clear();
            if s.past_len < sp.base {
                sp.base = s.past_len;
                sp.history.clear();
                sp.drafter = Drafter::new(order);
                sp.observed = 0;
            } else {
                sp.history.truncate(s.past_len - sp.base);
                if sp.observed > sp.history.len() {
                    sp.drafter = Drafter::new(order);
                    sp.observed = 0;
                }
            }
        }
        if s.past_len != sp.base + sp.history.len() {
            // Gap: tokens in between were never seen — re-seed the window here.
            sp.base = s.past_len;
            sp.history.clear();
            sp.drafter = Drafter::new(order);
            sp.observed = 0;
        }
        sp.history
            .extend_from_slice(&input_ids[s.q_start..s.q_start + s.q_len]);
        for i in sp.observed..sp.history.len() {
            let (pre, rest) = sp.history.split_at(i);
            sp.drafter.observe(pre, rest[0]);
        }
        sp.observed = sp.history.len();
    }

    /// MILESTONE 4c — try the native batched megakernel for a pure-decode greedy step (all q_len=1,
    /// qwen3moe Q4KS with the island up). Returns `Some(B token ids)` if it ran, else `None` to
    /// fall back to the serial wgpu batched path. DEFAULT-ON when the island is up (ARF_MSL_GEMV):
    /// this is the SHIPPED fast path carrying every conc win (B-row attention, mm_id MoE GEMM, tiled
    /// dense GEMM) — measured cold conc16/32/64 = 160/198/259 agg (vs the old grouped-GEMV serial
    /// path). The eligibility checks below (island present, pure greedy decode, MoE, KV not quantized,
    /// B>=thresh) are the real safety — it bails cleanly to the serial WGSL path whenever it can't run.
    /// ARF_NO_BATCH_MEGA=1 forces the serial path (A/B / debugging). The m=1 megakernel + serial
    /// batched path are untouched.
    #[cfg(target_os = "macos")]
    /// L144 — SERIAL GDN PREFILL. A hybrid (gated-delta-net) prompt cannot use the B-row
    /// prefill expansion: the conv ring and the delta-net matrix state are ONE serial stream per
    /// layer, so token t+1's state depends on token t's, and q_len parallel rows would all read
    /// the same pre-prompt state with the last writer winning (L143).
    ///
    /// So feed the prompt through the batched megakernel ONE TOKEN AT A TIME at b=1. Each step is
    /// the already-parity-proved decode shape: the 48 recurrent layers advance their resident
    /// conv/ssm state in place, the 16 attention layers append their own KV row. Only the LAST
    /// token's sampled id is returned (a prefill emits one continuation token).
    ///
    /// Cost is q_len island steps instead of one — correct, and prompts are 8-100 tokens for chat.
    /// llama.cpp's chunked-parallel delta rule (build_delta_net_chunking, CS=64) is the faster
    /// shape and stays future work; it is a different algorithm, not a reordering, so it does not
    /// belong in the commit that first makes this arch CORRECT.
    #[cfg(target_os = "macos")]
    fn try_gdn_serial_prefill(
        &self,
        input_ids: &[u32],
        batch: &arf_core::model::batch::ForwardBatch,
    ) -> Option<Vec<u32>> {
        use arf_core::cache::paged::WriteRun;
        if !self.layers.iter().any(|l| l.gdn.is_some()) {
            return None;
        }
        if batch.seqs.len() != 1 {
            return None;
        }
        let s0 = &batch.seqs[0];
        if s0.q_len <= 1 {
            return None;
        }
        if input_ids.len() != s0.q_len {
            return None;
        }
        // Every step needs token i's own physical slot, so the full context table must be present.
        if s0.slots.len() < s0.past_len + s0.q_len {
            return None;
        }
        // Image spans would need per-step remapping of their local offsets; refuse instead of
        // mis-slicing (text-only is the whole hybrid use case today).
        if !s0.image_spans.is_empty() {
            return None;
        }

        // WINDOWED PREFILL (2026-09-19). One token per step made a lone prompt cost ~48 ms a
        // token: MEASURED time to first token 7.1 s at 114 tokens, 25.9 s at 475, **117.9 s at
        // 1,825** (15-18 prompt tok/s) against another engine's 0.95 / 3.0 / 9.8 s (170-190 tok/s) on the
        // same Mac. Only the recurrent STATE UPDATE is serial; the matmuls are 85% of a step and
        // are not. The verify window already runs k rows of one sequence through the batched
        // record with the recurrence chained row by row (L227) and progressive causal attention,
        // and at 4..=16 rows its matmuls take the MPP path — so a prompt is fed through it 16
        // tokens at a time with `commit_all` (no rollback: prompt rows are all history).
        // Falls back to the token-serial loop below ONLY if the FIRST window declines, before
        // any state has moved. Opt out (A/B): ARF_NO_PREFILL_WINDOW=1.
        if std::env::var_os("ARF_NO_PREFILL_WINDOW").is_none() && s0.stream_id.is_some() {
            let mut last_tok: Option<u32> = None;
            let mut declined_first = false;
            let pw = prefill_window_rows();
            let n_win = input_ids.len().div_ceil(pw);
            let logits_needed = PREFILL_LOGITS_NEEDED.load(std::sync::atomic::Ordering::Relaxed)
                || std::env::var_os("ARF_PREFILL_ALL_LOGITS").is_some();
            for (w, win) in input_ids.chunks(pw).enumerate() {
                let prefix = s0.past_len + w * pw;
                let last = w + 1 == n_win && logits_needed;
                match self.prefill_window(win, prefix, &s0.slots, s0.stream_id, last) {
                    Ok(pred) => last_tok = pred,
                    Err(false) if w == 0 => {
                        declined_first = true;
                        break;
                    }
                    // Mid-prompt: the state is part-advanced and nothing can recover it.
                    Err(_) => return None,
                }
            }
            if !declined_first {
                return last_tok.map(|t| vec![t]);
            }
        }
        let mut last: Option<Vec<u32>> = None;
        for (i, &tok) in input_ids.iter().enumerate() {
            let pos = s0.past_len + i;
            // Step i sees exactly the prefix [0, pos] and writes ONE K/V row at pos's own physical
            // slot — the same view and the same write the equivalent decode step would have, so the
            // KV rows and the recurrent state advance identically to a token-by-token generation.
            let mut seq = s0.clone();
            seq.q_start = 0;
            seq.q_len = 1;
            seq.past_len = pos;
            seq.slots.truncate(pos + 1);
            seq.write_runs = vec![WriteRun {
                src_off: 0,
                phys_start: s0.slots[pos] as usize,
                count: 1,
            }];
            let mut step = batch.clone();
            step.seqs = vec![seq];
            last = self.try_batched_megakernel(std::slice::from_ref(&tok), &step);
            // A refusal mid-prompt would leave the recurrent state HALF-ADVANCED, and the caller
            // cannot recover by falling back (the generic path would then read a dirtied state).
            // Fail the whole prefill so the refusal is visible rather than silently wrong.
            last.as_ref()?;
        }
        last
    }

    /// L364 — macOS-only: this drives the native-Metal island record. Its only caller is the
    /// `cfg(target_os = "macos")` block in `forward_batch_impl`, so gating the whole fn keeps
    /// Linux from compiling a body whose `self.island` field does not exist there.
    #[cfg(target_os = "macos")]
    pub(super) fn try_batched_megakernel(
        &self,
        input_ids: &[u32],
        batch: &arf_core::model::batch::ForwardBatch,
    ) -> Option<Vec<u32>> {
        use crate::gpu::concurrent_metal::TokenSrc;
        // DEPTH-1 (shipped default): submit the step, then immediately fence + read. This is a pure
        // wrapper over the submit/read split — byte-identical to the old single-shot path.
        // The default out bank is the scratch out_tokens_mtl buffer; `.0.as_ref()` yields the raw
        // `&ProtocolObject<dyn MTLBuffer>` the record + read expect.
        let out_bank = self.scratch.out_tokens_mtl.as_ref()?.0.as_ref();
        let pending = self.submit_batched_step(TokenSrc::Cpu(input_ids), batch, out_bank)?;
        let ids = self.read_pending(&pending, out_bank);
        self.dflash_commit_plain(input_ids, batch);
        Some(ids)
    }

    /// C1 (2026-09-26): a plain batched record's taps go to their streams' draft context rings.
    /// Before, only verify and prefill windows committed, so a stream that shared the batch
    /// (B >= 2 decode, or decode rows riding a newcomer's prefill) fell behind its ring and never
    /// drafted again — the concurrent harness's solo tail ran plain. The record was waited on
    /// (`read_pending`) and nothing was submitted since, so its taps are this step's rows; the
    /// commit is ordered before the next record on the island's queue. Rows of one stream at
    /// consecutive positions (a prefill pack) are one run. `ARF_NO_STREAM_RINGS=1` skips it.
    #[cfg(target_os = "macos")]
    fn dflash_commit_plain(&self, input_ids: &[u32], batch: &arf_core::model::batch::ForwardBatch) {
        use crate::gpu::metal::dflash2_ctx::{stream_rings_off, CommitRun};
        if stream_rings_off() {
            return;
        }
        let Some(isl) = self.island.as_ref() else {
            return;
        };
        let g = isl.lock().unwrap();
        if !g.dflash_attached() {
            return;
        }
        let mut runs: Vec<CommitRun> = Vec::with_capacity(batch.seqs.len());
        for s in &batch.seqs {
            match runs.last_mut() {
                Some(r)
                    if r.stream == s.stream_id
                        && r.row0 + r.rows == s.q_start
                        && r.start + r.rows == s.past_len =>
                {
                    r.rows += s.q_len
                }
                _ => runs.push(CommitRun {
                    stream: s.stream_id,
                    row0: s.q_start,
                    rows: s.q_len,
                    start: s.past_len,
                }),
            }
        }
        if let Err(e) = g.dflash_commit_runs(&runs) {
            eprintln!("[dflash] {e}");
        }
        // rule 7: say once that this path is dispatched, and with how many streams
        static SHOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if runs.len() > 1 && !SHOWN.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[dflash] plain batched record committed to per-stream rings: {} runs {:?}",
                runs.len(),
                runs.iter()
                    .map(|r| (r.stream, r.rows, r.start))
                    .collect::<Vec<_>>()
            );
        }
        for r in &runs {
            if let Some(t) = input_ids.get(r.row0..r.row0 + r.rows) {
                g.dflash_note_tokens(r.stream, t);
            }
        }
    }

    /// DEPTH-2 SUBMIT HALF — record the batched megakernel step and async-commit it, WITHOUT
    /// waiting or reading. Everything the old `try_batched_megakernel` did up to and including the
    /// `batched_megakernel_record` commit lives here; it returns a [`BatchPending`] handle carrying
    /// the fence position + batch size so a later [`read_pending`](Self::read_pending) can drain it.
    ///
    /// `token_src` is this step's input token source (`TokenSrc::Cpu` today; `TokenSrc::GpuBank` is
    /// the depth-2 ping-pong path). `out_bank` is the u32 buffer the island writes sampled ids into.
    ///
    /// Same eligibility bails as the old path (returns `None` → caller falls back to serial WGSL).
    /// LOCK DISCIPLINE: this method takes the island lock across the whole build + record, then drops
    /// it (via `drop(isl)`) before returning. The record commits async with a completion handler that
    /// flips `slot_done` regardless of the lock, so `read_pending` can re-lock to fence independently.
    #[cfg(target_os = "macos")]
    pub fn submit_batched_step(
        &self,
        token_src: crate::gpu::concurrent_metal::TokenSrc<'_>,
        batch: &arf_core::model::batch::ForwardBatch,
        out_bank: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    ) -> Option<BatchPending> {
        // depth-1 / depth-2 callers use the single-step path (k=1, byte-identical).
        self.submit_batched_step_k(token_src, batch, out_bank, 1)
    }

    /// K-STEP variant: chain `k` decode steps into ONE record (no CPU seam between them).
    /// `batch.seqs[r].slots` MUST cover `past_len[r]+k` positions (the burst's k new contiguous
    /// slots); `past_len[r]+k` must stay within the seq's current KV block (the caller caps k to the
    /// block boundary). Returns a [`BatchPending`] with `k` set so `read_pending` reads `k*b` tokens.
    #[cfg(target_os = "macos")]
    pub fn submit_batched_step_k(
        &self,
        token_src: crate::gpu::concurrent_metal::TokenSrc<'_>,
        batch: &arf_core::model::batch::ForwardBatch,
        out_bank: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        k: usize,
    ) -> Option<BatchPending> {
        use crate::gpu::concurrent_metal::MegaLayer;
        use crate::gpu::types::{GpuMatWeight::Q4KS, GpuMlp};
        let k = k.max(1);
        if std::env::var_os("ARF_NO_BATCH_MEGA").is_some() {
            return None;
        }
        let dbg = std::env::var_os("ARF_BATCH_MEGA_DEBUG").is_some();
        macro_rules! bail {
            ($w:expr) => {{
                if dbg {
                    eprintln!("[batch-mega] fall back: {}", $w);
                }
                return None;
            }};
        }
        let island = self.island.as_ref()?;
        let mc = &self.cfg;
        // Eligibility: pure decode (every seq q_len==1), greedy, MoE model, KV not quantized.
        // `b` is the ROW count the record works in. For decode that is nseq (one row per
        // sequence); for a single-sequence prefill chunk it is the token count.
        let b = match &token_src {
            crate::gpu::concurrent_metal::TokenSrc::Cpu(ids)
                if batch.seqs.len() == 1 && ids.len() > 1 =>
            {
                ids.len()
            }
            _ => batch.seqs.len(),
        };
        // For a CPU token source the ids slice must be exactly B (byte-identical to the old
        // `input_ids.len() != b` guard). A GpuBank source has no CPU slice to size-check — the bank
        // is externally sized to B rows — so only the `b == 0` guard applies there.
        // PREFILL AS B-ROWS (L167). The old gate demanded ids.len() == nseq, i.e. pure decode,
        // so a 256-token prefill chunk of ONE sequence never reached the island — it fell to the
        // WGSL path. But the record does not care about sequences: it takes `b` rows each with
        // its own past_len[r] and slot table, and builds per-row attn_dims with ctx = pl+1, which
        // IS causal prefill. So a chunk of N tokens at consecutive positions is presentable as N
        // rows with past_len[r] = base + r.
        //
        // This is the route to the island GEMM (matmul_mm_q4ks, 32x64 tile) instead of the WGSL
        // coop kernel (8x8). L165 measured the cost of being locked out: Qwen (MoE, route open)
        // 28.5 tok/s prefill vs Muse (dense, route closed) 4.8 — same binary, 6x.
        let cpu_len_mismatch =
            matches!(token_src, crate::gpu::concurrent_metal::TokenSrc::Cpu(ids) if ids.len() != b);
        let prefill_rows = cpu_len_mismatch
            && batch.seqs.len() == 1
            && matches!(token_src, crate::gpu::concurrent_metal::TokenSrc::Cpu(_));
        // L363 — ARF_BATCH_MEGA_NO_PREFILL=1: keep single-sequence PREFILL chunks off this record
        // (diagnostic). Under UNGUARD, gemma-4's prompt ran here as b = token count (B=256/17/5
        // submits alongside the B=2 decode), so every "batched is garbage" result so far was a mix
        // of prefill-mode and decode-mode. This separates them: with it set, prefill takes the wgpu
        // path and only genuine decode steps reach the record.
        // The first version keyed this on `prefill_rows`, which is FALSE for exactly these chunks:
        // for a one-sequence prompt `b` is SET to ids.len(), so the length mismatch it tests cannot
        // hold, and B=256/17/5 chunks sailed through (L363c). "Prefill" here is structural — more
        // rows than sequences, or any sequence with q_len != 1.
        // Three shapes are prefill: one sequence with q_len > 1 (before the L167 expansion), more
        // rows than sequences, and — the one that slipped past the second version — N rows that all
        // carry the SAME `stream_id` (an expanded prefill presented as N single-token rows). A
        // genuine decode step has one distinct stream per row.
        let distinct_streams = {
            let mut v: Vec<u64> = batch.seqs.iter().filter_map(|s| s.stream_id).collect();
            v.sort_unstable();
            v.dedup();
            v.len()
        };
        // Fourth clause (L363c): N rows whose past_len run base, base+1, …, base+N-1 — the L167
        // expansion's signature. Dense models carry no stream_id, so the shared-id clause is
        // silent for them; consecutive positions across independent decode rows is a coincidence
        // this DIAGNOSTIC lever is allowed to misjudge.
        let consecutive = b > 1
            && batch
                .seqs
                .windows(2)
                .all(|w| w[1].past_len == w[0].past_len + 1);
        let is_prefill_shape = b != batch.seqs.len()
            || batch.seqs.iter().any(|s| s.q_len != 1)
            || (b > 1 && distinct_streams > 0 && distinct_streams < b)
            || consecutive;
        if dbg {
            let pl: Vec<usize> = batch.seqs.iter().take(6).map(|s| s.past_len).collect();
            let ql: Vec<usize> = batch.seqs.iter().take(6).map(|s| s.q_len).collect();
            eprintln!(
                "[bmega-shape] b={b} nseq={} q_len[..6]={ql:?} past_len[..6]={pl:?} streams={distinct_streams} consecutive={consecutive} -> prefill_shape={is_prefill_shape}",
                batch.seqs.len()
            );
        }
        if is_prefill_shape && std::env::var_os("ARF_BATCH_MEGA_NO_PREFILL").is_some() {
            bail!(format!(
                "prefill chunk excluded by ARF_BATCH_MEGA_NO_PREFILL (b={b}, nseq={}) (L363)",
                batch.seqs.len()
            ))
        }
        if b == 0 || (cpu_len_mismatch && !prefill_rows) {
            bail!("not pure decode (total != nseq)")
        }
        if std::env::var_os("ARF_BMEGA_SHAPE").is_some() {
            let qlens: Vec<usize> = batch.seqs.iter().map(|s| s.q_len).collect();
            let ids_n = match &token_src {
                crate::gpu::concurrent_metal::TokenSrc::Cpu(ids) => ids.len() as i64,
                _ => -1,
            };
            eprintln!("[bmega-shape] b={b} nseq={} ids_len={ids_n} prefill_rows={prefill_rows} q_lens={qlens:?}",
                batch.seqs.len());
        }
        // L143 — HYBRID PREFILL cannot use the B-row expansion. `prefill_rows` turns one
        // q_len-token prompt into q_len PARALLEL B-rows, which is exactly right for attention (each
        // row has its own KV slot) and exactly WRONG for a gated-delta-net layer: the conv ring and
        // the delta-net matrix state are ONE SERIAL stream per layer, so token t+1's state depends
        // on token t's. Running them as parallel rows would have every row read the same
        // pre-prompt state and the last writer win — silent, plausible garbage.
        //
        // llama.cpp solves this with a chunked-parallel delta rule (build_delta_net_chunking,
        // src/models/delta-net-base.cpp:16, CS=64), a different algorithm we do not have yet.
        // Until then a hybrid prefill is REFUSED here and served by the generic path.
        if prefill_rows && self.layers.iter().any(|l| l.gdn.is_some()) {
            bail!("hybrid prefill: B-row expansion would break the serial recurrent state")
        }
        // L207 — THE SAME REFUSAL, FOR THE MULTI-SEQUENCE CASE THE GUARD ABOVE MISSES.
        //
        // `prefill_rows` requires `batch.seqs.len() == 1`, so it only recognises a prefill that
        // arrives ALONE. When a second request is already decoding, the scheduler packs the new
        // request's prefill rows AND the live sequence into one batch, and every row is presented
        // as its own `SeqAttn` — measured with ARF_BMEGA_SHAPE on two concurrent chats:
        //     b=22 nseq=22 ids_len=22 prefill_rows=false
        // 22 "sequences" for 2 requests: 21 prefill rows of one prompt plus 1 decode row. Because
        // nseq != 1, the L143 guard did not fire, and those 21 rows reached `gdn_ensure`, which
        // refused them ("21 recurrent streams requested but only 8 are banked").
        //
        // That refusal is CORRECT but it lands too late: it bails the whole record, so the request
        // falls back to the generic path — which is itself corrupt (L206: ARF_NO_BATCH_MEGA=1
        // is wrong even at b=1). THAT is the b>=2 corruption. Not batched-row addressing, not GDN
        // bank mapping, not eviction: a hybrid prefill sneaking into the island by the one door
        // the guard did not cover, then poisoning the fallback.
        //
        // Refuse on the SHAPE rather than on `prefill_rows`: any row with q_len > 1, or more rows
        // than sequences the recurrent bank can hold, is a prefill in disguise on this arch.
        if self.layers.iter().any(|l| l.gdn.is_some()) {
            // The rows arrive ALREADY EXPANDED — measured q_lens = [1; 21] — so `q_len > 1`
            // does NOT identify them. What does: the recurrent bank has a hard capacity, and a
            // batch asking for more streams than exist is by definition not `b` real chats.
            // ONE source of truth for the bank width: this re-read the env with its OWN default of
            // 8, which would have silently disagreed with `gdn_bank_rows()` the day one changed.
            let banked: usize = gdn_bank_rows();
            // A COMMITTED SERIAL WINDOW is the one legitimate way past the bank: k causal
            // positions of ONE sequence on ONE bank row, recurrence chained in-kernel, armed only
            // by `verify_window(.., commit_all)`. It is a prefill, but not "in disguise" — it is
            // the shape L227 made correct. Everything else still stops here.
            let committed_window = b <= PREFILL_WINDOW
                && self
                    .island
                    .as_ref()
                    .is_some_and(|i| i.lock().unwrap().gdn_commit_window_armed());
            if b > banked && !committed_window {
                bail!(format!("hybrid: {b} rows exceeds {banked} banked recurrent streams — a prefill expanded into rows alongside a live sequence"))
            }
        }
        // 🔴 L142 — HARD BOUND ON B. `gemv_q4ks_batch` holds one accumulator per row in per-lane
        // registers (`float acc[MAXB]`, MAXB=64) and does `const uint B = min(d.m, MAXB)`, so rows
        // [64, d.m) are **SILENTLY NEVER WRITTEN** — no error, full speed, stale output. The
        // shader says so itself: "Host must keep conc <= MAXB or chunk." The host did not.
        //
        // This is the root cause of the L126 cross-seq prefill corruption. The prefill pack path
        // defaults chunk_rows=256 (batch.rs:1296) and passed b=pack.len() straight through. At
        // n<2048 the dispatch takes this clamped GEMV, which for qwen3-coder is k_proj (n=512),
        // v_proj (512) and the router (128) — so rows >=64 got STALE K/V and STALE routing, were
        // scattered into real KV slots, and exported to f16 as permanent truth for that sequence.
        // Scratch is never cleared, so with per-seq packs a row inherited the SAME sequence's
        // older K/V (wrong but on-topic — the text gate passed); with a straddling pack it
        // inherited ANOTHER sequence's, which is the fluent garbage L126 saw in ~1 of 16 agents.
        //
        // Every parity test stopped at exactly B=64 (batched_mega_parity.rs:539/545/570/187),
        // where min(d.m,64) is a no-op — we tested the boundary and never crossed it.
        //
        // Bail rather than clamp: callers fall back to a correct path. Removing this guard
        // requires raising MAXB via a compile-time #define AND re-measuring (acc[MAXB] is a
        // per-lane register array; at 256 it will spill and likely erase the GEMV's advantage).
        // L142/L143 — `gemv_q4ks_batch` and `gemv_q8_b` hold one accumulator per row in per-lane
        // registers and clamp `B = min(d.m, MAXB=64)`, so a SINGLE dispatch with b>64 silently
        // never writes the overflow rows. That was the root cause of the L126 cross-seq prefill
        // corruption. It is now handled INSIDE the dispatch (`gemv_b` and the Q8 lm_head arm in
        // concurrent_metal.rs row-chunk via buffer offsets), so b>64 is safe here and no bail is
        // needed. Verified by batched_mega_parity at b=65: drift 8.39e0 FAIL -> 0.00e0 PASS.
        // If you add a NEW caller of either kernel, it must chunk the same way.
        // A single-sequence PREFILL chunk (q_len > 1) is now expanded into q_len B-rows above,
        // so it is a legal shape for the record. Multi-sequence prefill is still refused: those
        // rows would need per-row slot tables from DIFFERENT sequences, which the expansion
        // above does not build.
        if !prefill_rows && !batch.seqs.iter().all(|s| s.q_len == 1) {
            bail!("some seq q_len!=1 (prefill)")
        }
        // Default threshold 1: B=1 (a single chat) through the megakernel measures p50 56 tok/s vs
        // ~11 on the serial fallback (coder-30B q4ks, 2026-07-05) — the daemon's single-chat path
        // was the ONE real workload getting no fast kernel. Parity at B=1 is covered explicitly by
        // batched_mega_parity (max_abs 4.2e-5 PASS). ARF_BATCH_MEGA_MINB=2 restores the old gate.
        let thresh = std::env::var("ARF_BATCH_MEGA_MINB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1usize);
        if b < thresh {
            bail!("B below threshold")
        }
        if self.kv.kv_quant.is_quantized() {
            bail!("KV quantized")
        }
        // L360 — GEMMA-4 ON THE BATCHED PATH IS NOT CORRECT YET. It emitted 'Count' and then
        // empty deltas forever over HTTP — silently, for as long as it has been in this repo.
        //
        // The cause is NOT the dual attention geometry, which is what the first five fixes
        // chased. An NLAYER bisect settled it: gemma is garbage at NLAYER=1, a single SLIDING
        // layer, while qwen at NLAYER=3 is coherent — so truncation alone is fine and the global
        // layers were never implicated. The first CONFIRMED cause is `layer_scalar`, an in-place
        // per-layer output scale the m=1 record applies (its step 17) and this record did not
        // (zero `scale_dims` uses in 4,000 lines). Proven by disabling it on the SERIAL path,
        // which reproduces the same class of garbage: `ARF_NO_LAYER_SCALAR=1` ->
        // 'icon আService гру<0x0A> sé<unused10>EV…'.
        //
        // HISTORY (L360–L363f, kept as the record of a wrong first diagnosis — see L363g below):
        // `layer_scalar` is now wired in and VERIFIED FIRING (a runtime probe logged its
        // dispatches). Output is still wrong, so at least one more op differs between the two
        // records — the remaining search is a mechanical diff of their op lists, not a guess.
        //
        // Until that lands this model takes the serial path, which produces correct text.
        // Rule 4: correctness outranks speed. The guard keys on the dual geometry because that is
        // the cheap structural signal identifying a gemma-4-class model here — the geometry
        // itself is NOT the bug, and the five fixes that assumed it was are kept because each
        // corrected a real inconsistency, not because any of them was the cause.
        // L363 — ARF_BATCH_MEGA_UNGUARD=1 bypassed that guard for DIAGNOSIS: it let the batched
        // record run on gemma-4 so its per-layer capture (ARF_GDN_CAPTURE) could be diffed
        // against the wgpu reference. That diff is what found both faults.
        // L363j — NO model-class guard here any more. HISTORY, kept because each step was a wrong
        // diagnosis that a future reader will be tempted to repeat:
        //   L360   gemma-4 (dual attention geometry) produced fluent garbage through this record ->
        //          the whole class was routed to the serial path.
        //   L360f  layer_scalar was missing from the record (real, fixed) — output still wrong.
        //   L363e/f  B1: the layer_scalar fix reached row 0 only (fixed with scale_inplace_b).
        //   L363g  B2: TWO "is this a global layer?" tests were dead code after the L360b
        //          shadowing (`ly.hd != hd` with `hd` already `ly.hd`) — the alt GEMV descriptors
        //          and the alt attention/scatter dims were never bound. Fixed: one predicate.
        //   L363h  "bug A closed" on repeated-'1' text — RETRACTED in L363i: a 17-row chat prompt
        //          still failed while a 5-row prompt passed.
        //   L363i/j  bug A was never gemma's: `gemv_q4ks_batch` is compiled with ACC_ROWS =
        //          --max-batch-size (L249) and never writes rows >= ACC_ROWS, while prefill-as-rows
        //          (L167) pushes up to 64 rows per dispatch. Proven by the flag alone: the same
        //          prompt is garbage at --max-batch-size 4 and correct at 32. The record now selects
        //          a wide twin whenever b > ACC_ROWS (concurrent_metal.rs, L363j), for every model.
        // ARF_BATCH_MEGA_UNGUARD is retained as a no-op so the diagnosis scripts keep running.
        let _ = std::env::var_os("ARF_BATCH_MEGA_UNGUARD");
        // L156 — DENSE models are supported now (the record grew a dense-FFN branch). This gate
        // used to bail, dropping every dense model onto the SERIAL path at conc>1: muse-glimmer
        // measured 0.41x vs llama at conc4 while qwen got 2.5x, purely because of this line.
        // Dense reports (0, 0, intermediate_size); the record keys off `ly.moe.is_none()` and the
        // MoE-shaped buffers below are simply sized for one "expert".
        let (num_experts, top_k, moe_inter) = match &mc.mlp {
            arf_core::config::MlpKind::Moe {
                num_experts,
                top_k,
                moe_intermediate,
                ..
            } => (*num_experts, *top_k, *moe_intermediate),
            // DENSE is SUPPORTED (L166). The last defect was the pre-trunk embedding norm
            // writing a scratch buffer instead of `hidden` — `hidden` IS the residual stream, so
            // it kept the RAW embedding and every one of the 52 layers compounded the error.
            // Verified "2 + 2 = 4" and two_model_gate.sh PASS/PASS.
            //
            // This is also the prefill lever: try_batched_megakernel is what routes work at the
            // island GEMM (matmul_mm_q4ks, 32x64 tile) instead of the WGSL coop kernel (8x8), and
            // dense models were locked out of it. Measured before this landed: Qwen (MoE, route
            // open) 28.5 tok/s prefill vs Muse (dense, route closed) 4.8 — 6x, same binary.
            _ => (0usize, 1usize, mc.intermediate_size),
        };
        let final_norm = self.mega.final_norm.as_ref()?;
        // Both lm_head formats dispatch on this record now: the Q8 arm via gemv_q8_b / the tiled
        // Q8 GEMM, the native Q4_K arm via gemv_q4ks_batch — the same kernel the trunk's batched
        // GEMVs already used, which had simply never been wired to the lm_head.
        //
        // This used to DECLINE the whole record for a native Q4_K lm_head, which was safe (a
        // mis-dispatch gives wrong logits) but expensive: gemma-4 reads its lm_head from the
        // GGUF's own Q4_K blocks, so it fell to the portable path for concurrent decode and lost
        // most of its aggregate throughput to it.
        let lm_head = self.mega.lm_head.as_deref()?;
        let (h, nh) = (mc.hidden_size, mc.num_attention_heads);
        // L156 — take the attention geometry from a REAL ATTENTION LAYER, not layer 0.
        //
        // The record writes ONE global `attn_dims` (num_heads/kv_heads/head_dim/group/row stride)
        // and binds it on every layer. On a HYBRID model layer 0 is a GDN layer, whose `kv_heads`
        // is not an attention head count at all: `layer_geometry` reports it as
        // `(groups*ssm_state)/head_dim` = 2048/256 = 8 purely so the loader's `l_kv_dim` comes out
        // right for the fused SSM qkv. The real attention layers have kv_heads = 4.
        //
        // Reading layer 0 therefore fed the SDPA:
        //     group      = nh/nkv = 24/8 = 3   (correct: 24/4 = 6)
        //     row_floats = nkv*hd = 2048       (correct: 4*256 = 1024)
        // and `attention_decode_b` does `kv_head = head/group`, so from head 12 onward it indexes
        // kv_head >= 4 — past the end of every KV pool row. Measured exactly: layer 3's `attn`
        // goes non-finite at float 4608 = head 18 of 24, and every downstream value (attn_out,
        // hidden, normed) is NaN from there.
        //
        // For every NON-hybrid model all layers share one geometry, so this picks layer 0 and is
        // byte-identical to the old behaviour.
        //
        // 🔴 L360 — THAT SENTENCE IS FALSE, and it cost gemma-4-12b every token it ever served.
        // Gemma 4 is not hybrid in the SSM sense, so it lands here — and its GLOBAL layers use a
        // DIFFERENT head_dim and kv_head count from its sliding ones (512/1 vs 256/8 on the 12B).
        // `attn_ref` picks the first non-GDN layer, which is layer 0, which is SLIDING. The record
        // then binds that one geometry for all 48 layers via its `nh, nkv, hd` PARAMETERS —
        // `batched_megakernel_record` takes them once, not per layer — so every global layer read
        // the wrong head_dim and emitted fluent garbage rather than crashing.
        //
        // NOTE the plumbing BELOW this point is all correct and was not the bug: `MegaLayer`
        // carries per-layer nh/nkv/hd, `MegaDimDesc` is built per layer, RoPE picks global/local
        // by `ly.window`, and `max_attn_dims()` sizes scratch over BOTH profiles. The single
        // model-wide binding is here, in the record's own signature. Do not go looking for it in
        // the layer descriptions — it is not there.
        let attn_ref = self
            .layers
            .iter()
            .find(|l| l.gdn.is_none())
            .unwrap_or(&self.layers[0]);
        let (nkv0, hd0) = (attn_ref.kv_heads, attn_ref.head_dim);
        let rotary0 = attn_ref.rotary_dim;
        // L360e — this model's SECOND attention geometry, if it has one. Gemma 4's global layers
        // use 512/1 with partial rotary (128 of 512) where its sliding layers use 256/8 with full
        // rotary; `attn_ref` is layer 0, which is SLIDING, so the record described the wrong
        // geometry to every global layer. `None` for the other eight architectures.
        let alt_geo4 = self
            .layers
            .iter()
            .find(|l| {
                l.gdn.is_none()
                    && (l.head_dim != attn_ref.head_dim || l.kv_heads != attn_ref.kv_heads)
            })
            .map(|l| (nh, l.kv_heads, l.head_dim, l.rotary_dim));
        let n_layers = self.layers.len();

        let _t_lock = std::time::Instant::now();
        let mut isl = island.lock().unwrap();
        let _lock_ms = _t_lock.elapsed().as_secs_f64() * 1000.0;
        let _t_rec = std::time::Instant::now();
        // Ensure per-layer dims (geometry fixed) for the m=1-style attn_dims the record reads scale from.
        // L246 — build dims for n_layers + the MTP head, so blk.64 is addressable as layer
        // index `n_layers`. `mega_dims_ready` is checked against the SAME count that is built,
        // or the record would rebuild them every token.
        let dims_n = n_layers + if self.mtp.is_some() { 1 } else { 0 };
        if !isl.mega_dims_ready(dims_n) {
            let mut descs: Vec<_> = self
                .layers
                .iter()
                .map(|ly| crate::gpu::concurrent_metal::MegaDimDesc {
                    nh,
                    nkv: ly.kv_heads,
                    hd: ly.head_dim,
                    rotary_dim: ly.rotary_dim,
                    eps: mc.rms_norm_eps as f32,
                    scale: mc
                        .query_pre_attn_scalar
                        .unwrap_or(1.0 / (ly.head_dim as f32).sqrt()),
                    window: ly.window.unwrap_or(0),
                    layer_scalar: ly.layer_scalar,
                })
                .collect();
            if self.mtp.is_some() {
                // blk.64 has the FULL-ATTENTION geometry (kv_heads 4, head_dim 256), not a GDN
                // layer's — it is a standard attention block appended after the trunk.
                let a = self
                    .layers
                    .iter()
                    .find(|l| l.gdn.is_none())
                    .unwrap_or(&self.layers[0]);
                descs.push(crate::gpu::concurrent_metal::MegaDimDesc {
                    nh,
                    nkv: a.kv_heads,
                    hd: a.head_dim,
                    rotary_dim: a.rotary_dim,
                    eps: mc.rms_norm_eps as f32,
                    scale: mc
                        .query_pre_attn_scalar
                        .unwrap_or(1.0 / (a.head_dim as f32).sqrt()),
                    window: a.window.unwrap_or(0),
                    layer_scalar: a.layer_scalar,
                });
            }
            // ctx_len for the dims build = max past_len+1 across seqs.
            let ctx_len = batch.seqs.iter().map(|s| s.past_len + 1).max().unwrap_or(1);
            if !isl.build_mega_dims(&self.ctx, &descs, ctx_len, h) {
                bail!("build_mega_dims")
            }
        }
        // Ensure persistent B-row scratch + dims (sized once).
        let (_max_hd, max_kv_dim) = mc.max_attn_dims();
        // L154 — `max_q_dim()`, NOT `nh * max_hd`. On the hybrid SSM family an attention layer's q
        // is packed 2× (Qwen3.8: 12288, vs the 6144 `nh * max_hd` yields), so the old expression
        // sized this scratch at HALF what the first attention layer writes into it — a silent 2×
        // overflow past the end of `q`, which is exactly where the layer-3 NaN was bisected to.
        let max_q_dim = mc.max_q_dim();
        // ctx_len = current context high-water (max past_len+1 across seqs). Threaded so the pooled
        // per-token B-row uniforms (slots_all et al.) are sized once and reused every step (no
        // per-token alloc / no fragmentation drift).
        // K-STEP: the chain writes/reads KV up to slot index past_len+k-1, so the pooled slot-table
        // stride (bmega_slots_cap) must cover max(past_len+k) — otherwise step k-1's slots_all row
        // would overflow the pool. (k=1 ⇒ past_len+1, the original value.)
        // ... AND every slot table the record will copy: it sizes its rows by `slots.len()`, which
        // is what its `max_slots <= slot_stride` assert checks. 🔴 MEASURED 2026-09-20: a prompt of
        // 2,228 tokens PANICKED the model actor — "max_slots 2229 exceeds pooled slot stride 2228" —
        // on the token-serial path and the windowed one alike, and the server answered 503 to
        // everything afterwards. Below 2,048 the pool's floor hid it.
        let mega_ctx_len = batch
            .seqs
            .iter()
            .map(|s| (s.past_len + k).max(s.slots.len()))
            .max()
            .unwrap_or(k);
        if isl
            .ensure_mega_bufs_batched(
                &self.ctx,
                b,
                h,
                max_q_dim,
                max_kv_dim,
                mc.vocab_size,
                mega_ctx_len,
            )
            .is_err()
        {
            bail!("ensure_mega_bufs_batched")
        }
        if isl
            .ensure_moe_bufs_batched(&self.ctx, b, top_k, moe_inter, num_experts, h)
            .is_err()
        {
            bail!("ensure_moe_bufs_batched")
        }
        // Sort-by-expert CSR scratch (ARF_MOE_SORTED) — ~3 KiB; cheap & idempotent (only re-allocs
        // on B growth). The record's sorted branch reads it; the GEMV default ignores it.
        if isl
            .ensure_csr_scratch_batched(&self.ctx, b, top_k, num_experts)
            .is_err()
        {
            bail!("ensure_csr_scratch_batched")
        }
        let norm_topk =
            matches!(&mc.mlp, arf_core::config::MlpKind::Moe { norm_topk, .. } if *norm_topk);
        if isl
            .ensure_mega_dims_batched(
                &self.ctx,
                b,
                h,
                moe_inter,
                top_k,
                num_experts,
                norm_topk,
                nh,
                nkv0,
                hd0,
                rotary0,
                mc.vocab_size,
                mc.rms_norm_eps as f32,
                alt_geo4,
            )
            .is_err()
        {
            bail!("ensure_mega_dims_batched")
        }
        // Also build the m=1 MoE dims so MegaMoe's gu/down/route/swiglu_dims fields (m=1 accessors)
        // are non-None — the batched record ignores them (it uses the B-row bdims), but MegaMoe
        // requires the borrows. Cheap + idempotent.
        if isl
            .ensure_moe_dims(
                &self.ctx,
                h,
                moe_inter,
                top_k,
                num_experts,
                norm_topk,
                nh,
                nkv0,
                hd0,
                mc.rms_norm_eps as f32,
            )
            .is_err()
        {
            bail!("ensure_moe_dims")
        }
        // L138 — gated-delta-net per-layer setup. MUST run BEFORE the record: gdn_record_inline
        // takes &self (it dispatches onto the caller's encoder), while allocating the resident
        // conv/ssm state and the per-layer scratch needs &mut self. Idempotent — the first call for
        // a layer allocates + uploads the static conv weight and the five dims uniforms, every
        // later call is a HashMap hit. Skipped entirely for models with no GDN layer.
        for (li, ly) in self.layers.iter().enumerate() {
            let Some(gd) = ly.gdn.as_ref() else { continue };
            let (inner, state, groups_n, rank, ck) = match &mc.attn {
                arf_core::config::AttnKind::HybridSsmAttn {
                    ssm_inner,
                    ssm_state,
                    groups,
                    dt_rank,
                    conv_kernel,
                    ..
                } => (*ssm_inner, *ssm_state, *groups, *dt_rank, *conv_kernel),
                _ => bail!("layer {li} has a GDN block but the arch is not HybridSsmAttn"),
            };
            let key_dim = state * groups_n;
            let conv_dim = key_dim * 2 + inner;
            // The host mirror the loader kept, NOT a GPU readback: gdn_ensure only consumes it on
            // the layer's first touch, but this loop runs every decode step.
            let conv_w = &gd.conv1d_host;
            if let Err(e) = isl.gdn_ensure(
                &self.ctx,
                li,
                conv_w,
                conv_dim * (ck - 1),
                state * state * rank,
                key_dim,
                inner,
                state,
                groups_n,
                rank,
                ck,
                mc.rms_norm_eps as f32,
                // L160 — bank ONE recurrent stream PER ROW. The state is serial (token t+1
                // depends on t), so a shared stream forced b==1 and blocked MTP verify, hybrid
                // prefill, and any --max-batch-size > 1 on this arch.
                // L235 — BANK FOR THE MAX BATCH, NOT THIS STEP'S `b`. Passing the step's own b
                // means a later, larger b reallocates the recurrent state mid-stream, and a
                // resize ZEROES it: the conv ring and delta-net matrix ARE the sequence's memory,
                // so that is instant amnesia (L160 observed `!!!!`; here it was a single dash).
                // Speculative verify makes this reachable in NORMAL operation — steady decode is
                // b=1, a k=2 verify is b=3 — which is why spec corrupted the engine on its FIRST
                // call and never recovered, while a model with NO GDN layers (qwen3-coder-30b,
                // 14 verifies) stayed perfectly clean.
                //
                // The bank is a fixed ARF_GDN_ROWS-wide allocation anyway, so asking for the
                // ceiling up front costs nothing extra and removes the resize entirely.
                b.max(gdn_bank_rows()),
            ) {
                // L207 — SURFACE THE REAL ERROR. This was `bail!("gdn_ensure layer {li}")`, a
                // string LITERAL: `{li}` never interpolated and the Err payload was dropped
                // entirely. So the one message that explains why a hybrid model silently loses
                // b>=2 batching read "gdn_ensure layer {li}" and named neither the layer nor
                // the reason.
                bail!(format!("gdn_ensure layer {li}: {e}"));
            }
        }
        // L162 — resolve this batch's sequence ids to STABLE BANK ROWS before recording. The row a
        // sequence gets is pinned for its lifetime, evicted when it leaves, and zeroed before it is
        // handed to a new sequence. Without this the banked state is keyed by the POSITIONAL row
        // index, so row 0 could be sequence A this step and B the next — each inheriting the
        // other's conv ring and delta-net matrix (L161: 4 concurrent sequences garbled, the same 4
        // serially perfect).
        let gdn_stream_rows: Vec<usize> = if self.layers.iter().any(|l| l.gdn.is_some()) {
            let ids: Vec<Option<u64>> = batch.seqs.iter().map(|s| s.stream_id).collect();
            // ONE source of truth for the bank width: this re-read the env with its OWN default of
            // 8, which would have silently disagreed with `gdn_bank_rows()` the day one changed.
            let banked: usize = gdn_bank_rows();
            match isl.gdn_map_streams(&ids, banked) {
                Ok(v) => {
                    if std::env::var_os("ARF_GDN_STREAM_TRACE").is_some() {
                        eprintln!("[gdn-stream] b={} ids={:?} -> banks {:?}", b, ids, v);
                    }
                    v
                }
                Err(e) => {
                    // 🔴 ALWAYS VISIBLE (2026-09-19). This `bail!` prints only under a debug
                    // switch, and its `None` sends the caller to the serial WGSL path — which keeps
                    // its OWN recurrent state, not this bank's. For a hybrid model mid-generation
                    // that is not a slow fallback, it is WRONG TEXT: measured on Qwen3.8-27B, 7
                    // concurrent streams 7/7 sane, 8 concurrent 0/8 sane, and with
                    // ARF_GDN_ROWS=16 the same 8 are 8/8 sane. The bank holds ROWS-1 sequences
                    // (row 0 is reserved, L206). arf-serve now clamps the scheduler to that; this
                    // line is for every other caller, and for the day the clamp is wrong.
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        eprintln!(
                            "🔴 [gdn] recurrent bank refused a batch of {b} ({e}). Falling back to \
                             the serial path, whose recurrent state is NOT this bank's — output for \
                             these sequences is unreliable. Lower --max-batch-size or raise \
                             ARF_GDN_ROWS."
                        );
                    });
                    bail!(format!("gdn stream map: {e}"))
                }
            }
        } else {
            Vec::new()
        };

        // GROW-ON-USE KV (sparse_kv.rs): back every slot this step WRITES with memory before
        // anything is recorded. Every slot a step READS was written by an earlier step, so the
        // new rows (`slots[past_len..]`) are the only ones that can be past the mapped range.
        // A failure is FATAL on purpose: writes to an unmapped page are dropped and reads return
        // 0, so carrying on (or falling back) would emit fluent wrong text, not an error.
        #[cfg(target_os = "macos")]
        if let Some(sp) = self.kv.q8_sparse.as_ref() {
            let need = batch
                .seqs
                .iter()
                .flat_map(|s| s.slots.get(s.past_len..).unwrap_or(&[]))
                .max()
                .map_or(0, |&m| m as usize + 1);
            let mut sp = sp.lock().unwrap();
            let need = if super::metal::sparse_kv::SparseKv::premap() {
                need.max(super::metal::sparse_kv::PREMAP_KV_SLOTS.min(sp.max_slots()))
            } else {
                need
            };
            if sp.would_grow(need) {
                isl.idle_for_mapping();
            }
            if let Err(e) = sp.ensure(need, isl.residency_set_obj()) {
                panic!("[kv] {e} — cannot back slot {need} of the KV cache with memory");
            }
        }
        // Build the real per-layer MegaLayer slice (mirrors decode.rs run_megakernel).
        let extract = |w: &crate::gpu::types::GpuMatWeight| -> Option<*const crate::gpu::concurrent_metal::Q4ksMtl> {
            if let Q4KS { mtl: Some(m), .. } = w { Some(std::sync::Arc::as_ptr(m)) } else { None }
        };
        let mut megas: Vec<MegaLayer> = Vec::with_capacity(n_layers);
        for (li, ly) in self.layers.iter().enumerate() {
            let (Some(q), Some(k), Some(v), Some(o)) = (
                extract(&ly.q_proj),
                extract(&ly.k_proj),
                extract(&ly.v_proj),
                extract(&ly.o_proj),
            ) else {
                bail!("q/k/v/o .mtl")
            };
            // Muse Glimmer's attention gate. If the layer HAS a gate but it did not land on the
            // native path, refuse the island rather than silently dropping the gate — that would
            // be a correctness bug (fluent output that ignores the prompt), not a slow path.
            let attn_gate_mtl = match ly.attn_gate.as_ref() {
                None => None,
                Some(g) => match extract(g) {
                    Some(p) => Some(unsafe { &*p }),
                    None => bail!("attn_gate .mtl (island would silently drop the gate)"),
                },
            };
            let moe = match &ly.mlp {
                GpuMlp::Moe(m) => {
                    let Some(router) = extract(&m.router) else {
                        bail!("router .mtl")
                    };
                    let (Some(gm), Some(um), Some(dm)) =
                        (m.gate_mtl.as_ref(), m.up_mtl.as_ref(), m.down_mtl.as_ref())
                    else {
                        bail!("moe gate/up/down .mtl")
                    };
                    Some(crate::gpu::concurrent_metal::MegaMoe {
                        router: unsafe { &*router },
                        gate: gm.as_ref(),
                        up: um.as_ref(),
                        down: dm.as_ref(),
                        gu_dims: isl.moe_gu_dims_buf(),
                        down_dims: isl.moe_down_dims_buf(),
                        route_dims: isl.moe_route_dims_buf(),
                        swiglu_dims: isl.moe_swiglu_dims_buf(),
                        num_experts: m.num_experts,
                        top_k: m.top_k,
                        inter: m.moe_inter,
                    })
                }
                // L156 — DENSE is supported now. This was the THIRD dense refusal on this path
                // (after the record's own and the host `not MoE` gate); each one alone was enough
                // to drop the model onto the serial path, which is why fixing the first two
                // changed nothing measurable.
                GpuMlp::Dense(_) => None,
            };
            // Dense gate/up/down for the record's dense-FFN branch. Mirrors decode.rs:268.
            let (dgate, dup, ddown) = match &ly.mlp {
                GpuMlp::Dense(d) => {
                    let (Some(g), Some(u), Some(dn)) = (
                        extract(&d.gate_proj),
                        extract(&d.up_proj),
                        extract(&d.down_proj),
                    ) else {
                        bail!("dense gate/up/down .mtl")
                    };
                    (
                        Some(unsafe { &*g }),
                        Some(unsafe { &*u }),
                        Some(unsafe { &*dn }),
                    )
                }
                GpuMlp::Moe(_) => (None, None, None),
            };
            // L212 — carry the 3-bit FFN triple into the BATCHED record. `megakernel_record`
            // (the m=1 path) has read `gate_q3k` since the Q3K work landed, but this path — the
            // one every daemon token actually goes through — hardcoded None, so ARF_Q3K_FFN
            // built and cached the q3 buffers and then never dispatched them. That is why it
            // measured +4% (baseline noise) instead of the ~20% the bandwidth math predicts.
            let q3 = match &ly.mlp {
                GpuMlp::Dense(d) => d.q3k.as_ref().map(std::sync::Arc::as_ptr),
                GpuMlp::Moe(_) => None,
            };
            let (gq3, uq3, dq3) = match q3 {
                Some(t) => {
                    let t = unsafe { &*t };
                    (Some(&t.0), Some(&t.1), Some(&t.2))
                }
                None => (None, None, None),
            };
            // ARF_Q8_DOWN: the Q8 down_proj view (Arc on the layer; outlives the record).
            let dq8 = match &ly.mlp {
                GpuMlp::Dense(d) => d
                    .down_q8
                    .as_ref()
                    .map(|a| unsafe { &*std::sync::Arc::as_ptr(a) }),
                GpuMlp::Moe(_) => None,
            };
            let nm = &ly.norm_mtls;
            // L145 — q_norm/k_norm are REQUIRED only on attention layers. A gated-delta-net layer
            // has NEITHER tensor in the file (weights.rs only loads them when present, and 3 of
            // every 4 qwen35 layers have no attention at all), so demanding all four here made the
            // batched record refuse EVERY step of a hybrid model with "norm_mtls" — 17 refusals on
            // a single 5-token request, including plain q_len=1 decode. That is why the GDN branch
            // never ran and why the output was garbage no matter what the branch computed.
            //
            // On a GDN layer the per-head q/k norms are not part of the math at all (the recurrence
            // uses gdn_l2_norm on the conv output instead), so we substitute input_norm as an
            // unused-but-bindable placeholder. The branch never dispatches qk_norm for these.
            let (Some(input_norm), Some(post_norm)) =
                (nm.input_norm.as_ref(), nm.post_norm.as_ref())
            else {
                bail!("norm_mtls: input/post")
            };
            let (qn, kn) = if ly.gdn.is_some() {
                (input_norm, input_norm)
            } else {
                let (Some(q), Some(k)) = (nm.q_norm.as_ref(), nm.k_norm.as_ref()) else {
                    bail!("norm_mtls: q/k on an attention layer")
                };
                (q, k)
            };
            let (Some(kpool), Some(vpool)) = (
                self.kv.keys_mtl.get(li).and_then(|o| o.as_ref()),
                self.kv.values_mtl.get(li).and_then(|o| o.as_ref()),
            ) else {
                bail!("kv pool mtl")
            };
            // the PARALLEL f16 KV pool (ARF_KV_F16). Present only when the flag was set
            // at build (weights.rs allocated it); then the batched coalesced attention uses the f16
            // scatter + f16-load kernel against it. Absent → None → the f32 path (byte-identical).
            let kpool_f16 = self
                .kv
                .keys_mtl_f16
                .get(li)
                .and_then(|o| o.as_ref())
                .map(|b| b.0.as_ref());
            let vpool_f16 = self
                .kv
                .values_mtl_f16
                .get(li)
                .and_then(|o| o.as_ref())
                .map(|b| b.0.as_ref());
            let (cos, sin) = if ly.window.is_none() {
                (
                    self.mega
                        .rope_cos_global
                        .as_ref()
                        .or(self.mega.rope_cos.as_ref()),
                    self.mega
                        .rope_sin_global
                        .as_ref()
                        .or(self.mega.rope_sin.as_ref()),
                )
            } else {
                (
                    self.mega
                        .rope_cos_local
                        .as_ref()
                        .or(self.mega.rope_cos.as_ref()),
                    self.mega
                        .rope_sin_local
                        .as_ref()
                        .or(self.mega.rope_sin.as_ref()),
                )
            };
            let (Some(cos), Some(sin)) = (cos, sin) else {
                bail!("rope table mtl")
            };
            let dims = isl.mega_layer_dims(li, 0);
            // L138 — assemble MegaGdn for a recurrent layer. Geometry comes from the config
            // (asserted again by gdn_ensure), the five weights from the fused-safe loader fields.
            // ARF_GDN_MAX_LAYER=N — BISECT KNOB (debug only, unset = all layers). Only the
            // first N recurrent layers take the megakernel GDN branch; the rest refuse the island
            // so the whole step falls back to the proven path. Lets a wrong stage be located by
            // sweeping N instead of by inspection.
            let gdn_max = std::env::var("ARF_GDN_MAX_LAYER")
                .ok()
                .and_then(|v| v.parse::<usize>().ok());
            // ARF_GDN_ONLY_LAYER=N — run the branch on EXACTLY layer N (refuse every other GDN
            // layer). Isolates one layer without the confound that ARF_GDN_MAX_LAYER=1 carries:
            // Qwen3.8's attn_qkv is Q6_K on 24 layers and Q4_K on the other 24, and layer 0 is
            // Q6_K — so "first N layers" always tests the lossily-transcoded case first.
            let gdn_only = std::env::var("ARF_GDN_ONLY_LAYER")
                .ok()
                .and_then(|v| v.parse::<usize>().ok());
            let gdn_mega = match ly.gdn.as_ref() {
                Some(_)
                    if gdn_max.is_some_and(|m| li >= m)
                        || gdn_only.is_some_and(|only| li != only) =>
                {
                    bail!("GDN bisect knob: refusing island at gdn layer {li}")
                }
                None => None,
                Some(gd) => {
                    let (inner, state, groups_n, rank) = match &mc.attn {
                        arf_core::config::AttnKind::HybridSsmAttn {
                            ssm_inner,
                            ssm_state,
                            groups,
                            dt_rank,
                            ..
                        } => (*ssm_inner, *ssm_state, *groups, *dt_rank),
                        _ => bail!("layer {li} has a GDN block but the arch is not HybridSsmAttn"),
                    };
                    let (Some(qkv), Some(qkv_gate), Some(beta), Some(alpha), Some(out)) = (
                        extract(&gd.qkv),
                        extract(&gd.qkv_gate),
                        extract(&gd.beta),
                        extract(&gd.alpha),
                        extract(&gd.out_proj),
                    ) else {
                        bail!("layer {li}: GDN projections missing .mtl (needs ARF_MSL_GEMV + native Q4_K)")
                    };
                    let (Some(dt), Some(a), Some(sn)) =
                        (nm.ssm_dt.as_ref(), nm.ssm_a.as_ref(), nm.ssm_norm.as_ref())
                    else {
                        bail!("layer {li}: GDN statics (ssm_dt/ssm_a/ssm_norm) missing .mtl view")
                    };
                    Some(crate::gpu::concurrent_metal::MegaGdn {
                        qkv: unsafe { &*qkv },
                        qkv_gate: unsafe { &*qkv_gate },
                        ssm_beta: unsafe { &*beta },
                        ssm_alpha: unsafe { &*alpha },
                        ssm_out: unsafe { &*out },
                        ssm_dt: dt.0.as_ref(),
                        ssm_a: a.0.as_ref(),
                        ssm_norm: sn.0.as_ref(),
                        key_dim: state * groups_n,
                        value_dim: inner,
                        s: state,
                        nkh: groups_n,
                        nvh: rank,
                        d_conv: match &mc.attn {
                            arf_core::config::AttnKind::HybridSsmAttn { conv_kernel, .. } => {
                                *conv_kernel
                            }
                            _ => 4,
                        },
                    })
                }
            };
            megas.push(MegaLayer {
                q: unsafe { &*q },
                k: unsafe { &*k },
                v: unsafe { &*v },
                o: unsafe { &*o },
                gate: dgate,
                up: dup,
                down: ddown,
                gate_q3k: gq3,
                up_q3k: uq3,
                down_q3k: dq3,
                down_q8: dq8,
                moe,
                // L138: recurrent (gated-delta-net) layers. Built ONLY when every handle the
                // branch binds is present; a partial set REFUSES the island (bail above) rather
                // than silently dropping the recurrence, which would be a correctness bug of the
                // "fluent but wrong" kind, not a slow path.
                gdn: gdn_mega,
                // Four-norm pair (gemma / muse-glimmer). Hardcoded None here until L156 — the
                // batched path had no four-norm branch to feed, so nothing noticed. decode.rs
                // has always passed these (decode.rs:299).
                input_norm,
                post_norm,
                pre_ffn_norm: nm.pre_ffn_norm.as_deref(),
                post_ffn_norm: nm.post_ffn_norm.as_deref(),
                attn_gate: attn_gate_mtl,
                rope_interleaved: mc.rope_interleaved(),
                gate_silu: matches!(mc.gate_act, arf_core::config::GateAct::Silu),
                q_norm: qn,
                k_norm: kn,
                value_norm: mc.value_norm,
                kpool,
                vpool,
                kv_q8: self
                    .kv
                    .q8
                    .get(li)
                    .and_then(|o| o.as_ref())
                    .map(|q| [&*q[0].0, &*q[1].0, &*q[2].0, &*q[3].0]),
                kpool_f16,
                vpool_f16,
                rope_cos: cos,
                rope_sin: sin,
                qkv_dims: dims.0,
                rope_dims: dims.1,
                scatter_dims: dims.2,
                attn_dims: dims.3,
                scale_dims: dims.4,
                nh,
                nkv: ly.kv_heads,
                hd: ly.head_dim,
                // L155 — a FULL-ATTENTION layer of the hybrid SSM family has a JOINT query+gate
                // `q` weight (2× the plain width, interleaved per head). `layer_q_dim` is the
                // authority the loader already uses; agreeing with it here keeps the record's
                // dispatch and the loaded weight the same shape by construction.
                q_joint_gate: {
                    let jg = ly.gdn.is_none()
                        && matches!(mc.attn, arf_core::config::AttnKind::HybridSsmAttn { .. })
                        && mc.layer_q_dim(li) == 2 * nh * ly.head_dim;
                    // Positive proof the branch is armed, per layer. The L149 lesson: never infer
                    // that a code path ran — an absent log line is not evidence it executed.
                    if jg && std::env::var_os("ARF_QJOINT_TRACE").is_some() {
                        eprintln!(
                            "[qjoint] layer {li}: JOINT q+gate armed (q_dim {} -> {})",
                            mc.layer_q_dim(li),
                            nh * ly.head_dim
                        );
                    }
                    jg
                },
            });
        }
        // Per-seq KV metadata (slot tables + past_len).
        // One entry PER ROW. Decode: one row per sequence. Prefill: N rows of the same
        // sequence, sharing its slot table, with past_len[r] = s.past_len + r so row r attends
        // exactly [0, past_len+r] — the causal frontier for that token.
        let (seq_slots, past_len): (Vec<&[u32]>, Vec<usize>) = if prefill_rows {
            let s0 = &batch.seqs[0];
            (
                (0..b).map(|_| s0.slots.as_slice()).collect(),
                (0..b).map(|r| s0.past_len + r).collect(),
            )
        } else {
            (
                batch.seqs.iter().map(|s| s.slots.as_slice()).collect(),
                batch.seqs.iter().map(|s| s.past_len).collect(),
            )
        };
        // Embed table (bf16) for the B-row embed — each seq's input token → bmega.hidden[r].
        // Without this the record processes a stale/zero hidden → catastrophic decode divergence.
        let Some(embed_table) = self.mega.embed.as_ref() else {
            bail!("mega.embed table")
        };
        let embed_scale = mc.embedding_scale.unwrap_or(1.0);
        // Record + async-commit into the caller-supplied `out_bank` with the caller-supplied
        // `token_src` (depth-1 passes the scratch out_tokens_mtl + TokenSrc::Cpu; the depth-2 ping-pong
        // passes an alternate bank + TokenSrc::GpuBank). The record writes out_bank[r] = sampled id.
        // ARM the batched pipelining override. The daemon force-sets
        // ARF_MEGA_BLOCKING=1 for SINGLEQ correctness, which otherwise makes THIS record drain
        // the GPU queue on every HTTP decode step (m=1 got a scoped escape in L7b; the batched path
        // never did). Armed here, DISARMED at the matching `wait_batched_step` fence in
        // read_pending — the arm/fence pair is strictly scoped to one submit->read cycle, so no
        // other consumer of ARF_MEGA_BLOCKING is affected. Opt out with
        // ARF_NO_BATCHED_PIPELINED=1 (NO_-style: a `=0` on a positive flag would still read as
        // set, the trap L8 documented).
        // SAFETY: this is only sound because read_pending ALWAYS fences on slot_done before reading
        // out_bank. Arming without that fence would let the CPU overwrite a uniform the GPU is
        // still reading.
        if std::env::var_os("ARF_NO_BATCHED_PIPELINED").is_none() {
            isl.set_batched_pipelined(true);
        }
        // L162 — publish this step's sequence->bank mapping for the record to consume.
        isl.set_gdn_stream_rows(gdn_stream_rows);
        let r = isl.batched_megakernel_record(
            &self.ctx,
            &megas,
            final_norm,
            lm_head,
            out_bank,
            embed_table,
            token_src,
            embed_scale,
            self.mega.embed_norm.as_deref(),
            b,
            h,
            mc.vocab_size,
            &seq_slots,
            &past_len,
            nh,
            nkv0,
            hd0,
            moe_inter,
            top_k,
            k,
        );
        if let Err(e) = r {
            bail!(format!("record: {e}"))
        }
        self.publish_mtp_hidden(&seq_slots, &past_len, b, k);
        // L146 — ARF_GDN_DUMP=<il>: after this step completes, print the first few floats of
        // each GDN stage buffer for layer <il>. Names the wrong STAGE instead of guessing. The
        // read is valid because read_pending fences on completion before the caller returns.
        if let Some(il) = std::env::var("ARF_GDN_DUMP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            let mut rows: Vec<(String, Vec<f32>)> = isl
                .gdn_dump(il, 6)
                .unwrap_or_default()
                .into_iter()
                .map(|(n, v)| (n.to_string(), v))
                .collect();
            // The megakernel binds its OWN batched buffers, not the island scratch — dump both so
            // a wrong stage can be attributed on the path that actually runs.
            //
            // THE BANK MUST MATCH THE STEP. `BatchedScratch`/`BatchedMoeScratch` are 2-banked
            // rings indexed by `sbank = past_len[0] % DEPTH2_BANKS` (concurrent_metal.rs:1725).
            // Hardcoding bank 0 here made the dump read the IDLE bank on every odd step, printing
            // that bank's stale warm-up contents (NaN) alongside the live bank's correct values —
            // an alternating live/NaN pattern that looks exactly like a real every-other-token
            // corruption and is purely an instrument bug. Derive the same bank the record used.
            let sbank =
                past_len.first().copied().unwrap_or(0) % crate::gpu::concurrent_metal::DEPTH2_BANKS;
            if let Some(mega) = isl.gdn_dump_mega(sbank, 6) {
                rows.extend(mega.into_iter().map(|(n, v)| (n.to_string(), v)));
            }
            // The one remaining input to gdn_norm_gate_b that is NOT in the scratch dump: the
            // per-layer `ssm_norm` gain. Read it off the MegaGdn the record just bound.
            if let Some(g) = megas.get(il).and_then(|m| m.gdn.as_ref()) {
                rows.push((
                    "ssm_norm(gain)".to_string(),
                    isl.gdn_dump_norm_weight(g.ssm_norm, 6),
                ));
            }
            for (name, vals) in rows {
                let f: Vec<String> = vals.iter().map(|v| format!("{v:+.5}")).collect();
                eprintln!("[gdn-dump l{il}] {name:<31} {}", f.join(" "));
            }
        }
        // L151 — the LAYER-ACCURATE rows, printed INDEPENDENTLY of ARF_GDN_DUMP. Everything in
        // the block above reads shared scratch after the step and therefore describes the LAST
        // layer; these were copied aside at each requested layer while it was live. A bisect wants
        // only these, and should not have to name an unrelated layer via ARF_GDN_DUMP to see
        // them. `ARF_GDN_CAPTURE=0,1,2,3` covers four layers in ONE model load.
        // L182 — only dump on genuinely BATCHED steps. A b=1 step has one real row; the reader
        // still emits row 1, which reads the previous step's slab and shows up as an "identical
        // pair". Mixing those into a population count makes a healthy engine look 40% aliased —
        // that is what produced the 200/482 figure in L181.
        if b > 1 {
            if let Some(cap) = isl.gdn_dump_capture(b, mc.hidden_size, 6, nh * hd0, nkv0 * hd0) {
                for (name, vals) in cap {
                    let f: Vec<String> = vals.iter().map(|v| format!("{v:+.5}")).collect();
                    eprintln!("[gdn-cap] {name:<26} {}", f.join(" "));
                }
            }
        }
        let _rec_ms = _t_rec.elapsed().as_secs_f64() * 1000.0;
        drop(megas);
        // The record committed the last chunk with a completion handler (no premature
        // waitUntilCompleted), so the GPU queue never drains. Drop the island lock NOW — the
        // completion handler flips slot_done regardless of the lock, so a later `read_pending` can
        // re-lock and fence this step independently (enabling submit-before-read at depth-2). The
        // fence position is the first seq's past_len (all B seqs advance in lockstep this step).
        let pos0 = batch.seqs.first().map(|s| s.past_len).unwrap_or(0);
        drop(isl);
        if dbg || std::env::var_os("ARF_BATCH_MEGA_TIME").is_some() {
            eprintln!(
                "[batch-mega] submitted B={b} | lock {:.1}ms record(+submit) {:.1}ms",
                _lock_ms, _rec_ms
            );
        }
        Some(BatchPending { pos0, b, k })
    }

    /// DEPTH-2 READ HALF — fence a previously-submitted batched step and read its B sampled token
    /// ids out of `out_bank`. Re-locks the island to spin on `slot_done` (`wait_batched_step`, a
    /// light CPU spin — NOT a GPU drain), then reads `out_bank.contents()[0..b]` directly (same
    /// coherent-shared-memory pattern as the m=1 burst path). `out_bank` MUST be the same buffer
    /// passed to the matching `submit_batched_step`.
    ///
    /// BLOCKING FALLBACK (ARF_MEGA_BLOCKING): the record already waitUntilCompleted'd internally;
    /// wait_batched_step is a no-op; the direct read is still coherent.
    #[cfg(target_os = "macos")]
    pub fn read_pending(
        &self,
        pending: &BatchPending,
        out_bank: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    ) -> Vec<u32> {
        if let Some(island) = self.island.as_ref() {
            let g = island.lock().unwrap();
            g.wait_batched_step(pending.pos0);
            // DISARM immediately after the fence — the override must never outlive the
            // submit->read cycle it was armed for.
            g.set_batched_pipelined(false);
        }
        use objc2_metal::MTLBuffer as _;
        // K-STEP: read k*b tokens step-major — out_bank[ks*b + r] is step ks's row-r sampled id.
        // k=1 ⇒ 0..b, byte-identical to depth-1. The caller commits them step-by-step in order.
        (0..pending.k * pending.b)
            .map(|i| unsafe { *(out_bank.contents().as_ptr() as *const u32).add(i) })
            .collect()
    }

    /// DEPTH-2 BANK ACCESSOR. Returns out_tokens bank `i∈{0,1}` as the raw
    /// `&ProtocolObject<dyn MTLBuffer>` that `submit_batched_step`/`read_pending` expect. Bank 0 is
    /// the shipped `out_tokens_mtl`; bank 1 is the depth-2 sibling `out_tokens_mtl2`. Returns `None`
    /// if the requested bank isn't allocated (MSL off / alloc failed) — the caller then stays
    /// depth-1. Any `i` other than 0/1 is `None`.
    #[cfg(target_os = "macos")]
    pub fn out_bank(
        &self,
        i: usize,
    ) -> Option<&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>> {
        match i {
            0 => self.scratch.out_tokens_mtl.as_ref().map(|b| b.0.as_ref()),
            1 => self.scratch.out_tokens_mtl2.as_ref().map(|b| b.0.as_ref()),
            _ => None,
        }
    }

    /// DEPTH-2 ELIGIBILITY GUARD. Whether THIS step can run the submit-before-read depth-2
    /// pipeline: macOS island up, BOTH out_tokens banks allocated, KV not quantized, MoE model, and
    /// a pure greedy decode step (every seq q_len==1). This mirrors the internal bails in
    /// `submit_batched_step` so the decode loop can pre-check before deciding to prime/overlap; a
    /// step that passes this AND `all_plain_greedy` (checked by the caller) is safe to pipeline. On
    /// a non-eligible step (prefill, quantized KV, banks missing) the loop falls back to the plain
    /// depth-1 `backend.step`. NB: this does NOT re-check `ARF_NO_BATCH_MEGA` — `submit_batched_step`
    /// returns `None` for that and the loop drains cleanly.
    #[cfg(target_os = "macos")]
    pub fn depth2_eligible(&self, batch: &arf_core::model::batch::ForwardBatch) -> bool {
        self.island.is_some()
            && self.out_bank(0).is_some()
            && self.out_bank(1).is_some()
            && !self.kv.kv_quant.is_quantized()
            && matches!(self.cfg.mlp, arf_core::config::MlpKind::Moe { .. })
            && !batch.seqs.is_empty()
            && batch.seqs.iter().all(|s| s.q_len == 1)
    }

    /// Non-macOS stub: depth-2 pipelining is a Metal-island feature, never eligible elsewhere.
    #[cfg(not(target_os = "macos"))]
    pub fn depth2_eligible(&self, _batch: &arf_core::model::batch::ForwardBatch) -> bool {
        false
    }

    /// MIXED-BATCH PREFILL FAST PATH — THE conc>1 TTFT wall.
    ///
    /// MEASURED 2026-08-03: conc8 with 200-word prompts had TTFT 27-32 s while conc1 was 0.30 s,
    /// FLAT across conc 8/16/32/64, budget-independent (512 vs 8192 identical). Cause: any batch
    /// containing a q_len>1 seq fell through BOTH megakernel entries to the serial WGSL path —
    /// prefill AND the innocent decode rows riding in the same step.
    ///
    /// Split the batch instead: each prefill seq walks the verify-megakernel windows via
    /// `try_batched_megakernel_verify` (~1 ms/token, parity-green — the mechanism the deleted
    /// single-seq reference walker `try_prefill_fast` used at conc1), decode seqs run as
    /// a pure-decode sub-batch through the batched megakernel. Outputs reassemble in seq order
    /// (one next-token per seq — cross-seq attention does not exist, so split order is safe).
    /// Any bail returns None => the untouched serial path, so this can only help.
    #[cfg(target_os = "macos")]
    fn try_prefill_fast_mixed(
        &self,
        input_ids: &[u32],
        batch: &arf_core::model::batch::ForwardBatch,
    ) -> Option<Vec<u32>> {
        use arf_core::model::batch::ForwardBatch;
        const KMAX: usize = 32; // (L1: was 16 — window width sets prefill weight-stream count)
                                // L22 REGION TIMER (ARF_PREFILL_REGIONS=1, default OFF) — LOCALIZE the fixed per-batch
                                // prefill cost L20 measured (TTFT = 0.321s FIXED + 0.201ms/token). L20 proved the cost is
                                // GPU-busy and invariant to prompt length; the only way to attribute it without per-kernel
                                // GPU counters (CLOSED on this box, L16/L17, four paths) is to bracket CPU wall time around
                                // each structural region of THIS function and watch which one carries the fixed term.
                                // Regions: decode sub-batch, f16 import leg (L5b), each pack call (with its row count),
                                // f16 export leg (L5b), volatile re-arm (L8).
        let rprof = std::env::var_os("ARF_PREFILL_REGIONS").is_some();
        let t_entry = std::time::Instant::now();
        let mut ms_dec = 0.0f64;
        let mut ms_import = 0.0f64;
        let mut ms_packs = 0.0f64;
        #[allow(unused_assignments)]
        let mut ms_export = 0.0f64;
        let mut pack_log: Vec<(usize, f64)> = Vec::new();
        // Per-seq flat-token offsets.
        let mut offs = Vec::with_capacity(batch.seqs.len());
        let mut off = 0usize;
        for s in &batch.seqs {
            offs.push(off);
            off += s.q_len;
        }
        if off != input_ids.len() || batch.positions.len() != off {
            return None;
        }
        let mut out: Vec<Option<u32>> = vec![None; batch.seqs.len()];
        // Decode rows first (they are latency-critical token streams; the design).
        let dec: Vec<usize> = (0..batch.seqs.len())
            .filter(|&i| batch.seqs[i].q_len == 1)
            .collect();
        if !dec.is_empty() {
            let dbatch = ForwardBatch {
                positions: dec.iter().map(|&i| batch.positions[offs[i]]).collect(),
                seqs: dec.iter().map(|&i| batch.seqs[i].clone()).collect(),
                image_embeds: None,
                mrope_positions: None,
            };
            let dids: Vec<u32> = dec.iter().map(|&i| input_ids[offs[i]]).collect();
            let _td = std::time::Instant::now();
            let toks = self.try_batched_megakernel(&dids, &dbatch)?;
            if rprof {
                ms_dec = _td.elapsed().as_secs_f64() * 1e3;
            }
            if toks.len() != dec.len() {
                return None;
            }
            for (j, &i) in dec.iter().enumerate() {
                out[i] = Some(toks[j]);
            }
        }
        // Prefill seqs: DECODE-ROW CHUNKS through the batched megakernel (L3, 2026-08-03).
        //
        // WHY NOT verify windows: k=32 windows give ~2 rows per routed expert — the mm_id tile
        // under-fills and every row streams its experts' weights nearly alone (MEASURED: forced
        // sorted at k=32 is WORSE, 111.4 vs 128.6 agg @conc8). Weight traffic scales with ROWS
        // until the tile fills (~8+ rows/expert), so the chunk must be BIG.
        //
        // WHY THIS IS SAFE: the batched decode record already handles B rows with PER-ROW
        // past_len and PER-ROW slot tables. A prefill chunk of C tokens IS C decode rows with
        // progressive past_len: row r scatters its K/V to slots[prefix+r] and its attention ctx
        // is [0, prefix+r] — later rows' K/V live OUTSIDE that range, so causality holds
        // structurally even though all C rows scatter before attention reads. Same trick the
        // verify path uses, minus its KMAX threadgroup bound. At C=128: ~8 rows/expert -> the
        // sorted mm_id crossover (b>=16) engages WITH filled tiles = GEMM-class prefill.
        //
        // ARF_PREFILL_CHUNK_ROWS overrides C. Default was 128; L107 (2026-08-08) re-raced
        // under the g2 protocol at conc8 GEN=128 and measured C=256 = **+11.3%** vs C=128
        // (179.8 ± 1.7 vs 161.6 ± 0.9, spread ≤1%) — the fixed per-pack weight-stream cost
        // (L22) dominates small tails, so fewer larger packs win. Decline -> verify windows ->
        // the serial path, so this can only help.
        // ⚠️ **L107's "+11.3% at C=256" IS VOID** (L142). It was measured while `gemv_q4ks_batch`
        // was SILENTLY DROPPING every row >= 64 — the config was faster because it was doing less
        // work, and the k/v/router it produced for those rows was stale. L126's "C=512/1024 are
        // clean" is void for the same reason: those avoid CROSS-seq inheritance (packs stop
        // straddling) but still dropped rows >= 64 within a sequence. Neither may be quoted.
        //
        // L143 made b > 64 SAFE by row-chunking the GEMV dispatch (ceil(b/64) calls at buffer
        // offsets, `gemv_b` in concurrent_metal.rs), verified by parity at b=65 going from
        // drift 8.39e0 FAIL to drift 0.00e0 PASS. So the 256 default is restored — but its
        // throughput benefit is now UNMEASURED on a correct engine and must be re-established
        // before anyone quotes a number for it.
        let mut chunk_rows: usize = std::env::var("ARF_PREFILL_CHUNK_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        // L211 — HYBRID (gated-delta-net) MODELS MUST PREFILL ONE TOKEN PER CALL.
        //
        // A pack of N rows is presented to the island as N `SeqAttn`s (each q_len=1, past_len
        // advancing), which is right for attention — every row has its own KV slot — and WRONG
        // for a recurrent layer, whose conv ring and delta-net matrix are ONE SERIAL stream:
        // token t+1's state depends on t's, so N parallel rows all read the same pre-pack state
        // and the last writer wins.
        //
        // L143 already refuses that expansion, but ONLY when the prefill arrives alone
        // (`prefill_rows` requires batch.seqs.len()==1). When a second request is already
        // decoding, the scheduler packs this prefill NEXT TO the live sequence, nseq != 1, the
        // guard misses it, and 21 rows reach gdn_ensure — which refuses (8 banks), bailing the
        // WHOLE record to the generic path, which is itself corrupt (L206). Measured:
        //     [bmega-shape] b=21 nseq=21 ids_len=21 prefill_rows=false q_lens=[1; 21]
        // That is the entire b>=2 corruption chain (L207).
        //
        // Cutting chunk_rows to 1 on this arch means a hybrid prefill never expands in the first
        // place, so the batch stays `nseq` real sequences and the island keeps serving it. Costs
        // prefill throughput (one weight stream per token) — the same trade L143 already accepted
        // by refusing outright, except the record now stays on the fast path for DECODE.
        let hybrid_serial = self.layers.iter().any(|l| l.gdn.is_some());
        if hybrid_serial {
            chunk_rows = 1;
        }
        // L4: CROSS-SEQ PACKED CHUNKS. Per-seq chunking leaves TAIL calls (260 tokens ->
        // 128+128+4: the 4-row call streams the FULL expert weights for 4 rows of work). At
        // conc16 that is 16 wasted weight streams per step wave; at conc64, 64. Pool every
        // prefill row across seqs into ONE row-stream and cut it into FULL chunk_rows calls
        // (rows of different seqs are independent; rows of the same seq stay progressive —
        // the same causality argument as before, per row via past_len/slot truncation).
        // MEASURED basis: conc16 TTFT 1.20s at 48 tail-y calls; predict ~0.85s at 33 full calls.
        struct Row {
            seq: usize,
            r: usize,
        }
        let mut rows: Vec<Row> = Vec::new();
        for (i, seq) in batch.seqs.iter().enumerate() {
            if seq.q_len <= 1 {
                continue;
            }
            if seq.slots.len() < seq.past_len + seq.q_len {
                return None;
            }
            for r in 0..seq.q_len {
                rows.push(Row { seq: i, r });
            }
        }
        // 🔴 L126 (2026-08-11) — CROSS-SEQ PACKING IS **DEFAULT-OFF**: IT CORRUPTS OUTPUT.
        //
        // L4 pooled rows from DIFFERENT sequences into one pack, on the premise that "rows of
        // different seqs are independent". That premise is FALSE in practice: on a cold wave of
        // concurrent long prompts, one sequence comes back as fluent GARBAGE ("Bug Bug Bug ...").
        //
        // ISOLATED to this one variable, both directions, on a real 16-agent coding workload
        // (2026-08-11):
        //     C=256, cross-seq packing ON  -> 1/4 agents GARBAGE
        //     C=256, packs cut at seq bnds -> 0/4  CLEAN
        //     C=512 / C=1024               -> 0/4  (few enough chunks to rarely straddle seqs)
        //     prefix cache ON vs OFF       -> IRRELEVANT (1/4 either way)
        // The victim is always a sequence whose rows shared a pack with another sequence's.
        //
        // ⚠️ It raises NO error: the corrupted stream generates at full speed, so every
        // throughput metric looks healthy. Only a text gate catches it (this is the third such
        // bug today).
        //
        // COST OF THE FIX: per-seq chunking leaves small tail calls, which is exactly the
        // inefficiency L4 set out to remove (and L107 later measured C=256 as +11.3%). We are
        // giving back some of that throughput. **That trade is not optional** — the win was
        // bought with a correctness bug on the cold concurrent long-prompt wave, i.e. precisely
        // the agentic workload this engine targets.
        //
        // Opt back IN (A/B only, measured harmful): ARF_PREFILL_PACK_CROSS_SEQ=1.
        let no_pack = std::env::var_os("ARF_PREFILL_PACK_CROSS_SEQ").is_none();
        // HYBRID PREFILL IN LOCKSTEP (2026-09-19). chunk_rows=1 (L211) plus per-seq packs (L126)
        // meant N concurrent prompts prefilled as sum(q_len) ONE-ROW passes. MEASURED on the 27B,
        // 16 requests: `nseq=16 ptoks=273 npacks=273 TOTAL 13128 ms`, ~48 ms a pass — 13 s of a
        // 48.8 s burst, and the reason a 59.8 tok/s decode delivered 41.7 end to end against
        // another engine's 47.3 (measured).
        //
        // The recurrence is serial WITHIN a sequence and independent ACROSS sequences, so pack t
        // takes row t of EVERY prefilling sequence: one row per sequence, each with its own
        // stream id, bank row, slots and past_len. That is not L126's shape (many rows of one
        // sequence sharing a pack with another's — still refused below for every other arch) and
        // it is not L211's (N rows of ONE sequence). It is exactly the shape of a b=nseq DECODE
        // step, which is the path `conc_prompt_fidelity.py` gates at 16 streams.
        // A lone prompt is unchanged: its packs are still one row each.
        // Opt out (A/B): ARF_NO_PREFILL_LOCKSTEP=1.
        let lockstep =
            hybrid_serial && no_pack && std::env::var_os("ARF_NO_PREFILL_LOCKSTEP").is_none();
        // WINDOWS FIRST (2026-09-19, later the same night). Lockstep gives P rows a pass, P = the
        // number of prompts still prefilling; a 16-row window of ONE prompt (`verify_window`,
        // `commit_all`) gives 16 whatever P is. A pass costs ~48 / 77 / 107 / 145 ms at 1-4 rows,
        // ~190 at 5-8 and ~250-268 at 9-16, so windows are >= lockstep at every P and equal at 16
        // — and the case that matters for chat is P = 1: ONE new prompt arriving while others
        // decode used to crawl at one row a pass AND stall every decoder for the whole prompt.
        // Lockstep stays as the fallback for a window that declines before any state has moved
        // (no stream id, verify switched off); once a window has run, a decline is fatal, as in
        // `try_gdn_serial_prefill`.
        let mut windowed = false;
        if lockstep && std::env::var_os("ARF_NO_PREFILL_WINDOW").is_none() {
            let mut advanced = false;
            let mut declined = false;
            'seqs: for (i, seq) in batch.seqs.iter().enumerate() {
                if seq.q_len <= 1 {
                    continue;
                }
                let toks = &input_ids[offs[i]..offs[i] + seq.q_len];
                let n_win = toks.len().div_ceil(prefill_window_rows());
                for (w, win) in toks.chunks(prefill_window_rows()).enumerate() {
                    let _tp = std::time::Instant::now();
                    let prefix = seq.past_len + w * prefill_window_rows();
                    let last = w + 1 == n_win;
                    match self.prefill_window(win, prefix, &seq.slots, seq.stream_id, last) {
                        Ok(pred) => {
                            advanced = true;
                            out[i] = pred;
                        }
                        Err(false) if !advanced => {
                            declined = true;
                            break 'seqs;
                        }
                        Err(_) => return None,
                    }
                    if rprof {
                        let ms = _tp.elapsed().as_secs_f64() * 1e3;
                        ms_packs += ms;
                        pack_log.push((win.len(), ms));
                    }
                }
            }
            windowed = !declined;
        }
        if windowed {
            rows.clear(); // every prompt row is done: no packs, and nothing for the f16 export
        }
        if lockstep {
            // Stable: within one `r` the sequences keep their batch order.
            rows.sort_by_key(|w| w.r);
        }
        let mut packs: Vec<&[Row]> = Vec::new();
        if lockstep {
            let mut lo = 0usize;
            while lo < rows.len() {
                let r = rows[lo].r;
                let mut hi = lo;
                while hi < rows.len() && rows[hi].r == r {
                    hi += 1;
                }
                packs.push(&rows[lo..hi]);
                lo = hi;
            }
        } else if no_pack {
            let mut lo = 0usize;
            while lo < rows.len() {
                let si = rows[lo].seq;
                let mut hi = lo;
                while hi < rows.len() && rows[hi].seq == si {
                    hi += 1;
                }
                for c in rows[lo..hi].chunks(chunk_rows) {
                    packs.push(c);
                }
                lo = hi;
            }
        } else {
            for c in rows.chunks(chunk_rows) {
                packs.push(c);
            }
        }
        // L5b (2026-08-03) — f16-KV PER-CALL SPLIT. Global ARF_KV_F16 measured MIXED through
        // the daemon: conc16 +7.6% agg (halved DECODE attention reads) but TTFT +65% / conc64
        // -5.4% — the f16 scatter+attention TAX these prefill pack calls. So under f16 mode the
        // packs run FORCED-f32 (set_force_kv_f32 around each call) and every pack row's written
        // slot is exported f32→f16 ONCE after the loop (kv_f32_to_f16_slots, slot-list),
        // keeping the f16 truth store fresh for the decode steps that keep the +7.6%. Past slots
        // are imported f16→f32 FIRST: decode scatters ONLY f16, so a continued seq's past decode
        // tokens exist nowhere in f32 (the serial path's import-leg invariant). The one-shot
        // export (vs per-pack) is safe because nothing reads f16 between packs: the decode
        // sub-batch already ran above and the forced-f32 packs read only the f32 pool.
        // f16 mode off → f16_tuples empty → every branch below is a no-op (byte-identical).
        //
        // L107 (2026-08-08) — the L5b comment said export was "microseconds". MEASURED under
        // ARF_PREFILL_REGIONS: a 256-token cold prefill paid **export 395ms** (packs 503ms,
        // TOTAL 898ms). The f32 force + import/export seam is NOT free. `ARF_PREFILL_NATIVE_F16=1`
        // skips the whole L5b split: packs ride the same f16 path as decode (no import, no
        // force_kv_f32, no export). Correctness holds because the pack megakernel scatters f16
        // when ARF_KV_F16 is on (same as decode). Default OFF until A/B proves net win —
        // L5b's pack-time tax may still dominate on some shapes.
        let native_f16_prefill = std::env::var_os("ARF_PREFILL_NATIVE_F16").is_some();
        let (f16_tuples, f16_pools, f16_all_eligible) = if !native_f16_prefill
            && std::env::var_os("ARF_KV_F16").is_some()
            && !rows.is_empty()
            && !self.kv.kv_quant.is_quantized()
            && self.island.is_some()
            && self.kv.keys_mtl_f16.iter().any(|o| o.is_some())
        {
            self.kv_f16_sync_layers()
        } else {
            (Vec::new(), Vec::new(), false)
        };
        let _timp = std::time::Instant::now();
        if !f16_tuples.is_empty() {
            let isl = self.island.as_ref()?.lock().unwrap();
            // L8 RE-ARM — must precede ANY f32 access: the pack calls below scatter into AND
            // attend the f32 pool even when there are no past slots to import. If the OS
            // RECLAIMED the volatile pages (prior state Empty), that is RECOVERABLE right here
            // by construction: the import below restores EVERY past slot any pack row reads
            // (slots[..past_len] per prefill seq, from the f16 truth store), each pack row's
            // CURRENT slot is written by its own scatter before any later row attends it
            // (the progressive-causality argument above), and the decode sub-batch already ran
            // f16-only. So reclaim needs no extra work beyond the import that runs anyway.
            if isl.kv_f32_volatile() {
                let _reclaimed = isl.set_kv_f32_purgeable(&f16_pools, false);
            }
            let past: Vec<u32> = batch
                .seqs
                .iter()
                .filter(|s| s.q_len > 1)
                .flat_map(|s| s.slots[..s.past_len.min(s.slots.len())].iter().copied())
                .collect();
            if !past.is_empty() {
                if let Err(e) = isl.convert_kv_slots(&self.ctx, &f16_tuples, &past, false) {
                    // Bail to the serial path — its own import leg re-establishes the invariant
                    // (and re-arms the pool again, harmlessly).
                    eprintln!("[kv-f16] L5b past import failed ({e}) — serial fallback");
                    return None;
                }
            }
        }
        if rprof {
            ms_import = _timp.elapsed().as_secs_f64() * 1e3;
        }
        for pack in packs {
            let _tp = std::time::Instant::now();
            let cb = arf_core::model::batch::ForwardBatch {
                positions: pack
                    .iter()
                    .map(|w| (batch.seqs[w.seq].past_len + w.r) as u32)
                    .collect(),
                seqs: pack
                    .iter()
                    .map(|w| {
                        let sq = &batch.seqs[w.seq];
                        arf_core::model::batch::SeqAttn {
                            // L217 — CARRY THE REAL STREAM ID THROUGH PREFILL.
                            //
                            // This was `None`, which `gdn_map_streams` maps to bank 0 — the scratch
                            // bank reserved for id-less rows (L206). So a hybrid model's prefill
                            // advanced its conv ring and delta-net matrix in bank 0, and then decode
                            // switched to the sequence's REAL bank, which had never seen the prompt.
                            // The recurrent state was silently discarded at the prefill/decode seam.
                            //
                            // With one sequence that is merely wrong-but-consistent (bank 0 happens
                            // to hold this sequence's state, so it mostly works). With TWO concurrent
                            // prefills BOTH land in bank 0 and clobber each other, which is the b>=2
                            // corruption. Measured, isolated to timing overlap alone:
                            //     both prefill together (delay 0.0s) -> 0/2 correct, IDENTICAL garbage
                            //     staggered 3.0s / 20.0s             -> 2/2 correct
                            // and the trace showed exactly why:
                            //     22x [gdn-stream] b=1 ids=[None] -> banks [0]
                            //
                            // For attention layers this changed nothing either way — the KV slot table
                            // carries prefill's history. It is only the RECURRENT state that lives in
                            // the bank, which is why this arch is the only one that ever noticed.
                            stream_id: sq.stream_id,
                            q_start: 0,
                            q_len: 1,
                            past_len: sq.past_len + w.r,
                            slots: sq.slots[..sq.past_len + w.r + 1].to_vec(),
                            write_runs: Vec::new(),
                            image_spans: Vec::new(),
                        }
                    })
                    .collect(),
                image_embeds: None,
                mrope_positions: None,
            };
            let wids: Vec<u32> = pack.iter().map(|w| input_ids[offs[w.seq] + w.r]).collect();
            // L5b: force the f32 pools/kernels for THIS pack call (no-op when f16 mode is off).
            // Cleared immediately after — the flag must never leak into a decode step's record.
            if !f16_tuples.is_empty() {
                if let Some(isl) = self.island.as_ref() {
                    isl.lock().unwrap().set_force_kv_f32(true);
                }
            }
            // EMBEDDED ROWS (2026-09-27, Qwen3-Omni audio; the batch has no M-RoPE table and no
            // bidirectional spans — the caller checks): this pack's rows that carry an embedding
            // take it instead of their token's, through the island's one-shot `RowOverride` (the
            // Qwen3.8 image path's mechanism, with no M-RoPE: text/audio positions are (p,p,p)).
            // The generic loop cannot serve these rows on Metal — its MoE experts are an L109
            // placeholder (MEASURED: the Omni Thinker's first spoken question hit that guard).
            let pack_embeds: Vec<(usize, Vec<f32>)> = match batch.image_embeds.as_ref() {
                Some(ie) if !ie.rows.is_empty() => pack
                    .iter()
                    .enumerate()
                    .filter_map(|(i, w)| {
                        let flat = offs[w.seq] + w.r;
                        let j = ie.rows.iter().position(|&r| r == flat)?;
                        Some((i, ie.embeds[j * ie.hidden..(j + 1) * ie.hidden].to_vec()))
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let armed = !pack_embeds.is_empty();
            if armed {
                self.island.as_ref()?.lock().unwrap().set_row_override(Some(
                    crate::gpu::concurrent_metal::RowOverride {
                        mrope: None,
                        sections: [0; 4],
                        embeds: pack_embeds,
                    },
                ));
            }
            let pack_out = self.try_batched_megakernel(&wids, &cb);
            if armed {
                self.island.as_ref()?.lock().unwrap().set_row_override(None);
                // The verify-window walk below embeds by token id: never let it serve these rows.
                pack_out.as_ref()?;
            }
            if !f16_tuples.is_empty() {
                if let Some(isl) = self.island.as_ref() {
                    isl.lock().unwrap().set_force_kv_f32(false);
                }
            }
            let preds = match pack_out {
                Some(p) => p,
                None => {
                    // Pack declined — walk THIS pack's rows through verify windows per seq-run.
                    // WHY THIS WORKS (from the deleted single-seq walker try_prefill_fast):
                    // try_batched_megakernel_verify pushes k query rows of ONE sequence through
                    // the batched megakernel as k B-rows (row r: q_len=1, past_len=prefix+r),
                    // and attention_verify_shared_prefix PHASE 2 gives row i a strictly causal
                    // view of rows 0..i. That is prefill semantics exactly.
                    let mut lp: Vec<u32> = Vec::with_capacity(pack.len());
                    let mut j = 0usize;
                    while j < pack.len() {
                        let si = pack[j].seq;
                        let run_start = j;
                        while j < pack.len() && pack[j].seq == si {
                            j += 1;
                        }
                        let sq = &batch.seqs[si];
                        let mut p2 = sq.past_len + pack[run_start].r;
                        let run_toks: Vec<u32> = (run_start..j)
                            .map(|t| input_ids[offs[si] + pack[t].r])
                            .collect();
                        for vw in run_toks.chunks(KMAX) {
                            let preds = self.try_batched_megakernel_verify(
                                vw,
                                p2,
                                &sq.slots,
                                sq.stream_id,
                            )?;
                            for &pr in preds.iter() {
                                lp.push(pr);
                            }
                            p2 += vw.len();
                        }
                    }
                    lp
                }
            };
            if preds.len() != pack.len() {
                return None;
            }
            // A seq's LAST row in this pack carries its most recent next-token candidate; the
            // final pack containing its last prompt row wins (overwrites are in row order).
            for (j, w) in pack.iter().enumerate() {
                if w.r + 1 == batch.seqs[w.seq].q_len {
                    out[w.seq] = Some(preds[j]);
                }
            }
            if rprof {
                let ms = _tp.elapsed().as_secs_f64() * 1e3;
                ms_packs += ms;
                pack_log.push((pack.len(), ms));
            }
        }
        let _tex = std::time::Instant::now();
        // L5b export: the forced-f32 packs scattered ONLY f32 — convert exactly the prompt rows'
        // written slots into the f16 truth store before any decode step reads them. Safe here:
        // try_batched_megakernel is synchronous (read_pending fences the step), so the f32 pool
        // writes are durable, and convert_kv_slots waitUntilCompleted's before returning.
        if !f16_tuples.is_empty() {
            let written: Vec<u32> = rows
                .iter()
                .map(|w| {
                    let sq = &batch.seqs[w.seq];
                    sq.slots[sq.past_len + w.r]
                })
                .collect();
            let isl = self.island.as_ref()?.lock().unwrap();
            if let Err(e) = isl.convert_kv_slots(&self.ctx, &f16_tuples, &written, true) {
                // Bail to the serial path: it re-runs the whole batch (f32 scatter + its own
                // f32→f16 mirror export), repairing both pools.
                eprintln!("[kv-f16] L5b pack export failed ({e}) — serial fallback");
                return None;
            }
            // L8 (2026-08-04) — f32 KV RESIDENCY IS PREFILL-ONLY. The export above just made the
            // f16 pool the complete truth for everything this step touched, so the f32 pool's
            // ~3.2GB (2048 blocks) is dead weight until the next prefill: mark it Volatile and
            // let the OS discard (not swap) it under pressure. WHY THIS IS SAFE NOW (vs the
            // measured-harmful 2026-07-28 experiment): the batched decode record reads ONLY f16
            // under this gate (use_kv_f16 conditions == all_eligible + coalesced env + the PSOs
            // checked in kv_f16_decode_ready), the m=1 single-stream paths re-arm through
            // kv_f32_rearm_for_single BEFORE any f32 access (decode_token_impl choke point) and
            // keep the f16 truth store fresh via the record's f16 mirror scatter, and every
            // prefill entry (fast-mixed above, serial import leg) re-arms + imports. Default ON;
            // opt out ARF_NO_KV_F32_VOLATILE=1 (NO_-style — downstream gates are is_some(),
            // a =0 value on a positive flag would still enable).
            let volatile_ok = std::env::var_os("ARF_NO_KV_F32_VOLATILE").is_none()
                && f16_all_eligible
                && std::env::var_os("ARF_ATTN_COALESCED").is_some()
                && isl.kv_f16_decode_ready();
            if volatile_ok {
                isl.set_kv_f32_purgeable(&f16_pools, true);
            }
        }
        if rprof {
            ms_export = _tex.elapsed().as_secs_f64() * 1e3;
            let total = t_entry.elapsed().as_secs_f64() * 1e3;
            let nseq = batch.seqs.len();
            let ptoks: usize = batch
                .seqs
                .iter()
                .filter(|s| s.q_len > 1)
                .map(|s| s.q_len)
                .sum();
            let np = pack_log.len();
            let plist: String = pack_log
                .iter()
                .map(|(n, ms)| format!("{n}:{ms:.1}"))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "[prefill-regions] nseq={nseq} ptoks={ptoks} npacks={np} TOTAL {total:.1}ms \
                       | dec {ms_dec:.1} | import {ms_import:.1} | packs {ms_packs:.1} \
                       | export {ms_export:.1} | other {:.1} | packs[rows:ms] {plist}",
                total - ms_dec - ms_import - ms_packs - ms_export
            );
        }
        out.into_iter().collect()
    }

    /// L244 — MTP DRAFT: predict token t+1 with the trained `blk.64` head.
    ///
    /// The whole point of this head is HIT-RATE. `SuffixDrafter` proposes on 12% of steps
    /// because it needs a >=4-token suffix seen before (L238); this proposes on EVERY step by
    /// construction, which is what turns a measured 1.16x per-window win into an actual speedup.
    ///
    /// The formulation is fixed by the tensor shapes: `nextn.eh_proj` is [2*hidden -> hidden],
    /// so it consumes the CONCATENATION of two hidden-width vectors — the normed embedding of
    /// the token just emitted, and the normed trunk hidden that produced it:
    ///
    /// ```text
    /// x       = eh_proj([ enorm(embed(tok_t)) || hnorm(h_t) ])
    /// h_draft = blk64_attention_and_ffn(x)
    /// logits  = lm_head(shared_head_norm(h_draft))
    /// ```
    ///
    /// Returns `None` whenever anything is missing or unsupported, so the caller simply does not
    /// draft this step — never a wrong token. A draft is only ever a PROPOSAL: it is handed to
    /// `try_batched_megakernel_verify`, which is what decides correctness, so a bad draft costs
    /// one verify window and nothing else.
    #[cfg(target_os = "macos")]
    pub fn mtp_draft(&self, tok: u32, past_len: usize, slots: &[u32]) -> Option<u32> {
        self.mtp_draft_chain(tok, past_len, slots, 1)
            .first()
            .copied()
    }

    /// L337 — CHAINED MTP: draft up to `kk` tokens by feeding the head its own output. Step 0
    /// conditions on the trunk's published hidden exactly as before; step s>0 conditions on
    /// (embed(draft_{s-1}) || hnorm(head's own step-(s-1) residual)) at position past_len+s —
    /// the recurrence llama.cpp's draft-mtp implements by passing `llama_get_embeddings_nextn`
    /// back into the next decode (common/speculative.cpp:784,822). The chain stops at the first
    /// declined step and returns what it has; a partial chain is still a valid (shorter) window.
    /// `slots` must cover positions [0, past_len + kk).
    /// L364 — the Linux twin. The MTP draft head is an island path (it reads `Q4ksMtl` weight
    /// views and the depth-2 bank ring, neither of which exists off-Metal), so off-macOS this
    /// returns no draft. The caller treats an empty chain as "no speculation this step", which
    /// is the same thing it does on Mac when the head declines — not an error path.
    #[cfg(not(target_os = "macos"))]
    pub fn mtp_draft_chain(
        &self,
        _tok: u32,
        _past_len: usize,
        _slots: &[u32],
        _kk: usize,
    ) -> Vec<u32> {
        Vec::new()
    }

    /// DFlash 2 — the block draft: up to 7 tokens after `tok` (the pending token of `stream`, at
    /// position `past_len`) from ONE pass of the attached draft model. Empty when there is no
    /// draft, or its context ring is not an unbroken history of this stream up to `past_len`
    /// (see `dflash2_ctx.rs`) — the caller then falls back to the MTP head.
    #[cfg(target_os = "macos")]
    pub fn dflash_draft(&self, tok: u32, past_len: usize, stream: u64) -> Vec<u32> {
        let (Some(isl), Some(lm), Some(embed)) = (
            self.island.as_ref(),
            self.mega.lm_head.as_deref(),
            self.mega.embed.as_ref(),
        ) else {
            return Vec::new();
        };
        let t0 = spec_prof().then(std::time::Instant::now);
        let scale = self.cfg.embedding_scale.unwrap_or(1.0);
        let r = isl.lock().unwrap().dflash_draft_block(
            &embed.0,
            scale,
            lm,
            Some(stream),
            tok,
            past_len,
        );
        if let Some(t0) = t0 {
            spec_prof_record(SpecProfKind::Draft, 8, t0.elapsed().as_secs_f64() * 1e3);
        }
        match r {
            Ok(Some(d)) => d,
            Ok(None) => Vec::new(),
            Err(e) => {
                eprintln!("[dflash] draft failed: {e}");
                Vec::new()
            }
        }
    }

    /// The block draft with a SAMPLED selector (speculative sampling, step 2): the drafts and
    /// each drafted position's distribution q. Empty drafts = no draft this step.
    #[cfg(target_os = "macos")]
    pub fn dflash_draft_sampled(
        &self,
        tok: u32,
        past_len: usize,
        stream: u64,
        temperature: f32,
        seed: u64,
    ) -> Option<(Vec<u32>, Vec<Vec<(u32, f32)>>)> {
        let (Some(isl), Some(lm), Some(embed)) = (
            self.island.as_ref(),
            self.mega.lm_head.as_deref(),
            self.mega.embed.as_ref(),
        ) else {
            return None;
        };
        let scale = self.cfg.embedding_scale.unwrap_or(1.0);
        let r = isl.lock().unwrap().dflash_draft_block_sampled(
            &embed.0,
            scale,
            lm,
            Some(stream),
            tok,
            past_len,
            temperature,
            seed,
        );
        match r {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[dflash] sampled draft failed: {e}");
                None
            }
        }
    }

    /// GPU-SELECTED block draft (2026-09-22): the block runs, the candidate selector runs ON THE
    /// GPU, and the chosen window `[anchor, t1..t7]` is left in a GPU token buffer — the CPU never
    /// waits for it. The verify record that follows on the SAME queue is ordered behind it and
    /// reads the window through `TokenSrc::GpuBank` (`verify_window_gpu_draft`). MEASURED before
    /// this existed: the draft's `waitUntilCompleted` + CPU selector + verify encode left the GPU
    /// idle ~42 ms of every ~172 ms cycle (measured). `Some(proposals)` = launched.
    #[cfg(target_os = "macos")]
    pub fn dflash_draft_gpu(&self, tok: u32, past_len: usize, stream: u64) -> Option<usize> {
        let (Some(isl), Some(lm), Some(embed)) = (
            self.island.as_ref(),
            self.mega.lm_head.as_deref(),
            self.mega.embed.as_ref(),
        ) else {
            return None;
        };
        let scale = self.cfg.embedding_scale.unwrap_or(1.0);
        match isl.lock().unwrap().dflash_draft_block_gpu(
            &embed.0,
            scale,
            lm,
            Some(stream),
            tok,
            past_len,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[dflash] gpu draft failed: {e}");
                None
            }
        }
    }

    /// [`Self::dflash_draft_gpu`] for a SAMPLED request with the sampled draft on (2026-09-26):
    /// the GPU chain DRAWS each position (`dflash_chain_sampled`) and its q is read back by the
    /// verify. `None` = not launched (no ring, or the kernel is not compiled) — the caller takes
    /// the CPU sampled selector (`dflash_draft_sampled`).
    #[cfg(target_os = "macos")]
    pub fn dflash_draft_gpu_sampled(
        &self,
        tok: u32,
        past_len: usize,
        stream: u64,
        temperature: f32,
        seed: u64,
    ) -> Option<usize> {
        let (Some(isl), Some(lm), Some(embed)) = (
            self.island.as_ref(),
            self.mega.lm_head.as_deref(),
            self.mega.embed.as_ref(),
        ) else {
            return None;
        };
        let scale = self.cfg.embedding_scale.unwrap_or(1.0);
        match isl.lock().unwrap().dflash_draft_block_gpu_sampled(
            &embed.0,
            scale,
            lm,
            Some(stream),
            tok,
            past_len,
            temperature,
            seed,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[dflash] gpu sampled draft failed: {e}");
                None
            }
        }
    }

    /// [`Self::dflash_draft_gpu_sampled`], WAITED: the drafts and their q on the host
    /// (`MetalIsland::dflash_draft_block_gpu_sampled_waited`) — the multi-stream cycle's sampled
    /// draft (2026-09-27), whose record takes CPU token windows and whose streams share the one
    /// set of block buffers. `None` = no draft (no ring, the kernel is not compiled, or it
    /// failed) — the caller takes the CPU sampled selector (`dflash_draft_sampled`).
    /// `ARF_DFLASH_GPU_SELECT_CHECK=1` re-runs the CPU sampled selector on the same block right
    /// after the wait and counts mismatches, as the lone path's verify does.
    #[cfg(target_os = "macos")]
    pub fn dflash_draft_gpu_sampled_waited(
        &self,
        tok: u32,
        past_len: usize,
        stream: u64,
        temperature: f32,
        seed: u64,
    ) -> Option<(Vec<u32>, Vec<Vec<(u32, f32)>>)> {
        let (Some(isl), Some(lm), Some(embed)) = (
            self.island.as_ref(),
            self.mega.lm_head.as_deref(),
            self.mega.embed.as_ref(),
        ) else {
            return None;
        };
        let scale = self.cfg.embedding_scale.unwrap_or(1.0);
        let t0 = spec_prof().then(std::time::Instant::now);
        let g = isl.lock().unwrap();
        let r = match g.dflash_draft_block_gpu_sampled_waited(
            &embed.0,
            scale,
            lm,
            Some(stream),
            tok,
            past_len,
            temperature,
            seed,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[dflash] waited gpu sampled draft failed: {e}");
                None
            }
        };
        if let Some(t0) = t0 {
            spec_prof_record(SpecProfKind::Draft, 8, t0.elapsed().as_secs_f64() * 1e3);
        }
        if r.is_some() && std::env::var_os("ARF_DFLASH_GPU_SELECT_CHECK").is_some() {
            if let Some((cpu, gpu, dq)) = g.dflash_gpu_select_check() {
                self.dflash_check_report("multi-stream sampled draft: ", &cpu, &gpu, dq, true);
            }
        }
        r
    }

    /// The verify half of `dflash_draft_gpu`: a `k`-row window whose tokens are the draft's GPU
    /// buffer. Returns the window's REAL tokens (read back after the verify completed — the draft
    /// preceded it on the same queue, so they are final) together with the predictions.
    #[cfg(target_os = "macos")]
    pub fn verify_window_gpu_draft(
        &self,
        k: usize,
        prefix_len: usize,
        prefix_slots: &[u32],
        stream_id: u64,
        sampler: Option<&arf_core::sampling::RowSampler>,
    ) -> Option<(Vec<u32>, Vec<u32>)> {
        let buf = self
            .island
            .as_ref()?
            .lock()
            .unwrap()
            .dflash_token_window()?;
        // Only the LENGTH of this is read before the verify; the record embeds from `buf`.
        let placeholder = vec![0u32; k];
        let preds = self.verify_window(
            &placeholder,
            prefix_len,
            prefix_slots,
            Some(stream_id),
            false,
            true,
            Some(&buf.0),
            sampler,
        )?;
        use objc2_metal::MTLBuffer as _;
        let window =
            unsafe { std::slice::from_raw_parts(buf.0.contents().as_ptr() as *const u32, k) }
                .to_vec();
        // ARF_DFLASH_GPU_SELECT_CHECK=1: re-run the CPU selector on the same block and compare.
        // Counts are printed so a silent arm cannot pass as a green one (rule 8).
        if std::env::var_os("ARF_DFLASH_GPU_SELECT_CHECK").is_some() {
            // A SAMPLED GPU draft (2026-09-26) is checked against the CPU SAMPLED selector under
            // the same key; `dq` = the largest |q_cpu - q_gpu| (inf = candidate ids differ).
            if let Some((cpu, gpu, dq)) = self
                .island
                .as_ref()?
                .lock()
                .unwrap()
                .dflash_gpu_select_check()
            {
                self.dflash_check_report(
                    "",
                    &cpu,
                    &gpu,
                    dq,
                    sampler.is_some_and(|s| !s.params.is_greedy()),
                );
            }
        }
        Some((window, preds))
    }

    /// One `ARF_DFLASH_GPU_SELECT_CHECK` comparison of the GPU selector against the CPU one
    /// (`MetalIsland::dflash_gpu_select_check`), counted and printed: every mismatch, every
    /// q difference above 1e-5, and every tenth window. `what` prefixes the count ("" for the
    /// lone stream's windows, as the line always read); the counts are the process's, lone and
    /// multi-stream drafts together.
    #[cfg(target_os = "macos")]
    fn dflash_check_report(&self, what: &str, cpu: &[u32], gpu: &[u32], dq: f32, sampled: bool) {
        let same = cpu == gpu;
        let n = self.dflash_check_n.get() + 1;
        let bad = self.dflash_check_bad.get() + usize::from(!same);
        self.dflash_check_n.set(n);
        self.dflash_check_bad.set(bad);
        if !same || dq > 1e-5 || n % 10 == 1 {
            eprintln!(
                "[dflash-check] {what}{} windows, {bad} MISMATCH{}{}",
                n,
                if sampled {
                    format!(" (sampled request; this window's max |q_cpu - q_gpu| {dq:.2e})")
                } else {
                    String::new()
                },
                if same {
                    String::new()
                } else {
                    format!("  cpu={cpu:?} gpu={gpu:?}")
                }
            );
        }
    }

    /// EXPERIMENTS ONLY: see `MetalIsland::dflash_propose_from_hidden`.
    #[cfg(target_os = "macos")]
    pub fn dflash_propose_from_hidden(&self, hidden: &[f32], anchor: u32) -> Option<Vec<u32>> {
        let (isl, lm) = (self.island.as_ref()?, self.mega.lm_head.as_deref()?);
        isl.lock()
            .unwrap()
            .dflash_propose_from_hidden(lm, hidden, anchor)
            .ok()
    }

    /// HYBRID PREFIX CACHE — save `stream`'s recurrent state under `key` (see
    /// `metal/state_snapshots.rs`). `(saved, evicted key)`.
    #[cfg(target_os = "macos")]
    pub fn state_save(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        let Some(isl) = self.island.as_ref() else {
            return (false, None);
        };
        match isl.lock().unwrap().state_snapshot_save(stream, key) {
            Ok(evicted) => (true, evicted),
            Err(e) => {
                eprintln!("[state-snapshot] save failed: {e}");
                (false, None)
            }
        }
    }

    /// HYBRID PREFIX CACHE — [`state_save`](Self::state_save) into the ANCHOR pool (the end of a
    /// request's shared system + tools prefix; `metal/state_snapshots.rs`, 2026-09-26). Called
    /// once per anchored prompt, never per token.
    #[cfg(target_os = "macos")]
    pub fn state_save_anchor(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        let Some(isl) = self.island.as_ref() else {
            return (false, None);
        };
        match isl.lock().unwrap().state_snapshot_save_anchor(stream, key) {
            Ok(evicted) => (true, evicted),
            Err(e) => {
                eprintln!("[state-snapshot] anchor save failed: {e}");
                (false, None)
            }
        }
    }

    /// HYBRID PREFIX CACHE — [`state_save`](Self::state_save) into the JUNCTION pool (where a
    /// prompt left cached history; `metal/state_snapshots.rs`, 2026-09-26). Called at most once
    /// per prompt, never per token.
    #[cfg(target_os = "macos")]
    pub fn state_save_junction(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        let Some(isl) = self.island.as_ref() else {
            return (false, None);
        };
        match isl
            .lock()
            .unwrap()
            .state_snapshot_save_junction(stream, key)
        {
            Ok(evicted) => (true, evicted),
            Err(e) => {
                eprintln!("[state-snapshot] junction save failed: {e}");
                (false, None)
            }
        }
    }

    /// A rolling checkpoint (`SeqPlan::snapshot_checkpoint`), filed in the checkpoint pool.
    #[cfg(target_os = "macos")]
    pub fn state_save_checkpoint(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        let Some(isl) = self.island.as_ref() else {
            return (false, None);
        };
        match isl
            .lock()
            .unwrap()
            .state_snapshot_save_checkpoint(stream, key)
        {
            Ok(evicted) => (true, evicted),
            Err(e) => {
                eprintln!("[state-snapshot] checkpoint save failed: {e}");
                (false, None)
            }
        }
    }

    /// HYBRID PREFIX CACHE — restore the snapshot under `key` into `stream` before its first step.
    #[cfg(target_os = "macos")]
    pub fn state_restore(&self, stream: u64, key: u64) -> bool {
        let Some(isl) = self.island.as_ref() else {
            return false;
        };
        match isl
            .lock()
            .unwrap()
            .state_snapshot_restore(stream, key, gdn_bank_rows())
        {
            Ok(hit) => hit,
            Err(e) => {
                eprintln!("[state-snapshot] restore failed: {e}");
                false
            }
        }
    }

    /// The KV a cached prefix lives in, with its bytes a slot: every attention layer's 8-bit
    /// `[k8, ks, v8, vs]`, then the MTP head's f32 K and V (same slot ids). Empty off the 8-bit
    /// pool — the on-disk prefix cache covers the default KV format only.
    #[cfg(target_os = "macos")]
    fn kv_slot_buffers(
        &self,
    ) -> Vec<(
        &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        usize,
    )> {
        use objc2_metal::MTLBuffer as _;
        let total = self.kv.num_blocks * self.kv.block_size;
        if total == 0 || self.kv.q8.iter().all(|q| q.is_none()) {
            return Vec::new();
        }
        let mut out: Vec<_> = self
            .kv
            .q8
            .iter()
            .flatten()
            .flat_map(|q| q.iter().map(|b| (&b.0, b.0.length() / total)))
            .collect();
        for b in [&self.kv.mtp_keys_mtl, &self.kv.mtp_values_mtl]
            .into_iter()
            .flatten()
        {
            out.push((&b.0, b.0.length() / total));
        }
        out
    }

    /// ON-DISK PREFIX CACHE (issue #17, 2026-10-06): the state of a cached prefix as one blob — the
    /// snapshot under `key` and the KV of `slots` (the prefix's positions, in order). `Err` when
    /// there is no such snapshot or no 8-bit KV pool.
    #[cfg(target_os = "macos")]
    pub fn prefix_export(&self, key: u64, slots: &[u32]) -> std::result::Result<Vec<u8>, String> {
        let isl = self
            .island
            .as_ref()
            .ok_or("no Metal island")?
            .lock()
            .unwrap();
        let bufs = self.kv_slot_buffers();
        if bufs.is_empty() {
            return Err("the prefix cache needs the 8-bit KV pool".into());
        }
        isl.dflash_fence(); // every step that wrote these slots has finished
        let snap = isl
            .state_snapshot_export(key)
            .ok_or("no snapshot under that key")?;
        let per_slot: usize = bufs.iter().map(|b| b.1).sum();
        let kv_len = per_slot * slots.len();
        let mut out = Vec::with_capacity(32 + snap.len() + kv_len);
        out.extend_from_slice(b"ARFPX001");
        for v in [per_slot as u64, slots.len() as u64, snap.len() as u64] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&snap);
        drop(snap);
        let at = out.len();
        out.resize(at + kv_len, 0); // the KV is read straight into the blob
        isl.slot_rows_io(&bufs, slots, Some(&mut out[at..]), None)?;
        Ok(out)
    }

    /// The inverse of [`prefix_export`](Self::prefix_export): write the KV into `slots` (backing
    /// them with memory first) and file the snapshot under `key` in the anchor pool. The caller
    /// registers the blocks and the key with the scheduler only after this returns `Ok` — a key
    /// the scheduler can match whose KV is not there would answer with fluent wrong text. Returns
    /// the snapshot key evicted to make room.
    #[cfg(target_os = "macos")]
    pub fn prefix_import(
        &self,
        key: u64,
        slots: &[u32],
        blob: &[u8],
    ) -> std::result::Result<Option<u64>, String> {
        let isl = self
            .island
            .as_ref()
            .ok_or("no Metal island")?
            .lock()
            .unwrap();
        let bufs = self.kv_slot_buffers();
        let per_slot: usize = bufs.iter().map(|b| b.1).sum();
        let word = |i: usize| -> std::result::Result<usize, String> {
            Ok(u64::from_le_bytes(
                blob.get(8 + 8 * i..16 + 8 * i)
                    .ok_or("prefix blob: truncated header")?
                    .try_into()
                    .unwrap(),
            ) as usize)
        };
        if blob.get(..8) != Some(b"ARFPX001".as_slice()) {
            return Err("prefix blob: not a version-1 prefix".into());
        }
        let (saved_per_slot, n, snap_len) = (word(0)?, word(1)?, word(2)?);
        if saved_per_slot != per_slot || n != slots.len() {
            return Err(format!(
                "prefix blob: {n} slots of {saved_per_slot} bytes, this server has {} slots of {per_slot}",
                slots.len()
            ));
        }
        let snap = blob
            .get(32..32 + snap_len)
            .ok_or("prefix blob: truncated snapshot")?;
        let kv = blob
            .get(32 + snap_len..)
            .ok_or("prefix blob: truncated KV")?;
        if kv.len() != per_slot * n {
            return Err("prefix blob: KV size does not match".into());
        }
        if let Some(sp) = self.kv.q8_sparse.as_ref() {
            let need = slots.iter().max().map_or(0, |&m| m as usize + 1);
            let mut sp = sp.lock().unwrap();
            if sp.would_grow(need) {
                isl.idle_for_mapping();
            }
            sp.ensure(need, isl.residency_set_obj())?;
        }
        isl.dflash_fence();
        isl.slot_rows_io(&bufs, slots, None, Some(kv))?;
        isl.state_snapshot_import(key, snap)
    }

    /// Raw last-row logits per sequence for a hybrid model, through committed island windows
    /// (see the call site in `forward_batch_impl`). `None` = something declined before any state
    /// moved for the FIRST sequence; the caller then uses the generic path.
    ///
    /// BATCHED DECODE ROWS (2026-09-27): when EVERY sequence is a single row (a decode step of
    /// sampled / logprobs / penalised streams — `step`'s non-greedy fallback), the loop below ran
    /// ONE full record per sequence: MEASURED, two concurrent sampled streams took 166 ms a plain
    /// step against 75 ms for a greedy pair, which shares one record (measured 2026-09-27,
    /// "CONCURRENT SAMPLED requests"). Now they go through `hybrid_decode_rows_logits`: one
    /// multi-segment record, each sequence a one-row segment. Prefill rows (`q_len > 1`) keep the
    /// loop exactly as it was. `ARF_NO_BATCHED_HYBRID_LOGITS=1` = the loop for decode rows too.
    #[cfg(target_os = "macos")]
    fn hybrid_logits_via_windows(
        &self,
        input_ids: &[u32],
        batch: &arf_core::model::batch::ForwardBatch,
    ) -> Option<Vec<Vec<f32>>> {
        if batched_hybrid_logits_on()
            && batch.seqs.len() >= 2
            && batch
                .seqs
                .iter()
                .all(|s| s.q_len == 1 && s.stream_id.is_some() && s.image_spans.is_empty())
        {
            return self.hybrid_decode_rows_logits(input_ids, batch);
        }
        let vocab = self.cfg.vocab_size;
        let pw = prefill_window_rows();
        let mut out = Vec::with_capacity(batch.seqs.len());
        let mut off = 0usize;
        for (si, seq) in batch.seqs.iter().enumerate() {
            let toks = &input_ids[off..off + seq.q_len];
            off += seq.q_len;
            let n_win = toks.len().div_ceil(pw);
            let mut logits = None;
            for (w, win) in toks.chunks(pw).enumerate() {
                let prefix = seq.past_len + w * pw;
                let last = w + 1 == n_win;
                let preds = self.verify_window(
                    win,
                    prefix,
                    &seq.slots,
                    seq.stream_id,
                    true,
                    last,
                    None,
                    None,
                );
                if preds.is_none() {
                    // Before any state moved for the first sequence the caller can still fall
                    // back; after that the state is part-advanced and nothing can recover it.
                    if si == 0 && w == 0 {
                        return None;
                    }
                    panic!("hybrid logits window declined mid-sequence (seq {si}, window {w})");
                }
                if last {
                    let all = self
                        .island
                        .as_ref()?
                        .lock()
                        .unwrap()
                        .bmega_logits_snapshot(prefix, win.len(), vocab)?;
                    logits = Some(all[(win.len() - 1) * vocab..].to_vec());
                }
            }
            out.push(logits?);
        }
        Some(out)
    }

    /// BATCHED DECODE ROWS (2026-09-27) — see `hybrid_logits_via_windows`. Every sequence of
    /// `batch` is one decode row with a stream id. Chunks of at most `SPEC_CKPT_ROWS` (7)
    /// sequences each run as ONE multi-segment record (`verify_segments_rec` with one-row
    /// segments), and each sequence's logits row comes out of that record's logits snapshot
    /// (`segment_last_rows`: segment g = record row g). Why 7 and not the record's 8 rows: the
    /// per-segment fused recurrence (`gdn_verify_fused`, the kernel the multi-stream records run
    /// live) takes at most `SPEC_CKPT_ROWS` segments; an 8th would move the whole record onto the
    /// legacy per-row kernels. 8+ sequences split evenly (8 -> 4 + 4; `decode_row_chunks`).
    ///
    /// WHAT THE ROWS RELY ON — that a one-row segment of this record computes what the one-row
    /// window of the loop computed, up to last-bit kernel differences:
    ///  * the RECURRENT state: each row runs on ITS stream's bank row (the record's row -> bank
    ///    map from the stream ids, `set_gdn_stream_rows`; `verify_segments` declines two segments
    ///    of one stream), advanced by exactly one position; a one-row segment always "accepts"
    ///    its row (`accepted_run` of a 1-row window is 1 row), so nothing is restored;
    ///  * the KV: each row scatters to ITS slot `slots[past_len]` and attends over its own slot
    ///    table `[0, past_len + 1)`, per-row attention as in any multi-segment record;
    ///  * the draft's context ring: each segment's row is committed to its stream's ring
    ///    (`dflash_commit_runs`, a run of 1), as the loop's committed window did
    ///    (`dflash_commit_context`) — so a stream that later speculates alone keeps its draft;
    ///  * the logits: the record's `[rows, vocab]` lm_head output sits in the bank of its first
    ///    row, exactly where the sampled multi-segment verify draws from.
    /// Differences that remain: the fused per-segment kernel saves each segment's initial state to
    /// its checkpoint slot (a restore a one-row segment never needs), and the argmax predictions
    /// are not noted as the draft's committed tokens (the loop's committed window did not either).
    ///
    /// A chunk whose record DECLINES (before running) falls back to one window per sequence, the
    /// loop's own call; `None` = nothing ran (the caller takes the generic path, as before).
    #[cfg(target_os = "macos")]
    fn hybrid_decode_rows_logits(
        &self,
        input_ids: &[u32],
        batch: &arf_core::model::batch::ForwardBatch,
    ) -> Option<Vec<Vec<f32>>> {
        use crate::gpu::concurrent_metal::SPEC_CKPT_ROWS;
        let vocab = self.cfg.vocab_size;
        let seqs = &batch.seqs;
        let n = seqs.len();
        let tok = |i: usize| &input_ids[seqs[i].q_start..seqs[i].q_start + 1];
        let record = |r: std::ops::Range<usize>| {
            let k = r.len();
            let segs: Vec<arf_core::backend::SegReq<'_>> = r
                .map(|i| arf_core::backend::SegReq {
                    stream: seqs[i].stream_id.expect("checked by the caller"),
                    window: tok(i),
                    prefix_len: seqs[i].past_len,
                    slots: &seqs[i].slots,
                })
                .collect();
            let (_, rows) = self.verify_segments_rec(&segs, &[], true)?;
            // once per record width (rule 7: shown to have run, at each width)
            static SHOWN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let bit = 1u32 << k.min(31);
            if SHOWN.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0 {
                eprintln!(
                    "[step] hybrid logits for {k} decode rows in ONE record (was {k} records)"
                );
            }
            Some(rows)
        };
        // the loop's own call for one decode row: a committed one-row window, then its logits
        let single = |i: usize| {
            let s = &seqs[i];
            self.verify_window(
                tok(i),
                s.past_len,
                &s.slots,
                s.stream_id,
                true,
                true,
                None,
                None,
            )?;
            // the window RAN: from here a missing row cannot be a decline
            let row = self
                .island
                .as_ref()
                .and_then(|isl| {
                    isl.lock()
                        .unwrap()
                        .bmega_logits_snapshot(s.past_len, 1, vocab)
                })
                .expect("hybrid logits: a decode row's logits are not readable after its window");
            Some(row)
        };
        let out = arf_core::backend::batched_decode_logits(n, SPEC_CKPT_ROWS, record, single);
        if n > SPEC_CKPT_ROWS && out.is_some() {
            static SHOWN: std::sync::Once = std::sync::Once::new();
            SHOWN.call_once(|| {
                eprintln!(
                    "[step] hybrid logits: {n} decode rows in {} records of <= {SPEC_CKPT_ROWS}",
                    n.div_ceil(SPEC_CKPT_ROWS)
                )
            });
        }
        out
    }

    /// EVAL ONLY — teacher-forced log-probabilities through the path that SERVES this model: one
    /// committed window of `window.len()` prompt rows at `prefix_len` (the windowed prefill's own
    /// record, lm_head on), then `ln p(targets[r] | tokens up to and including window[r])` for
    /// each row, from the record's logits. `arf perplexity` goes through `forward_batch_all`,
    /// which is WRONG for the hybrid 27B (2.7e6 on plain English, 2026-09-21); this is the
    /// instrument for any change to the weights' numbers. Returns (log-prob, argmax == target).
    #[cfg(target_os = "macos")]
    pub fn window_logprobs(
        &self,
        window: &[u32],
        prefix_len: usize,
        slots: &[u32],
        stream_id: u64,
        targets: &[u32],
    ) -> Option<Vec<(f32, bool)>> {
        let preds = self.verify_window(
            window,
            prefix_len,
            slots,
            Some(stream_id),
            true,
            true,
            None,
            None,
        )?;
        let vocab = self.cfg.vocab_size;
        let logits = self
            .island
            .as_ref()?
            .lock()
            .unwrap()
            .bmega_logits_snapshot(prefix_len, window.len(), vocab)?;
        Some(
            targets
                .iter()
                .enumerate()
                .map(|(r, &t)| {
                    let row = &logits[r * vocab..][..vocab];
                    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let lse = m + row.iter().map(|x| (*x as f64 - m).exp()).sum::<f64>().ln();
                    ((row[t as usize] as f64 - lse) as f32, preds[r] == t)
                })
                .collect(),
        )
    }

    /// GATES ONLY: the target's raw embedding rows for `tokens`, `[n, hidden]`, read from the
    /// island's bf16 table on the host (the block reference's input).
    #[cfg(target_os = "macos")]
    pub fn dflash_embed_rows(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        use objc2_metal::MTLBuffer as _;
        let embed = self.mega.embed.as_ref()?;
        let h = self.cfg.hidden_size;
        let scale = self.cfg.embedding_scale.unwrap_or(1.0);
        // The table is the GGUF's own Q4_K when `embed_q4k` (2026-09-23): read it as bf16 and this
        // gate would read 2.54 GB out of a 0.72 GB buffer. Decode the rows exactly as
        // `embed_tok_q4k` does (ggml dequantize_row_q4_K).
        if self.embed_q4k {
            let blocks = unsafe {
                std::slice::from_raw_parts(
                    embed.0.contents().as_ptr() as *const u8,
                    embed.0.length(),
                )
            };
            let per_row = h / 256 * 144;
            return Some(
                tokens
                    .iter()
                    .flat_map(|t| {
                        let row = &blocks[*t as usize * per_row..][..per_row];
                        (0..h).map(move |i| {
                            let blk = &row[(i / 256) * 144..][..144];
                            let (j, l) = ((i % 256) / 32, i % 32);
                            let d = crate::gpu::concurrent_metal::f16_bits_to_f32(
                                u16::from_le_bytes([blk[0], blk[1]]),
                            );
                            let dm = crate::gpu::concurrent_metal::f16_bits_to_f32(
                                u16::from_le_bytes([blk[2], blk[3]]),
                            );
                            let sc = &blk[4..16];
                            let (sv, mv) = if j < 4 {
                                ((sc[j] & 63) as f32, (sc[j + 4] & 63) as f32)
                            } else {
                                (
                                    ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)) as f32,
                                    ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) as f32,
                                )
                            };
                            let qb = blk[16 + (j / 2) * 32 + l];
                            let q = if j % 2 == 1 { qb >> 4 } else { qb & 0xF } as f32;
                            (d * sv * q - dm * mv) * scale
                        })
                    })
                    .collect(),
            );
        }
        let bits = unsafe {
            std::slice::from_raw_parts(
                embed.0.contents().as_ptr() as *const u16,
                self.cfg.vocab_size * h,
            )
        };
        Some(
            tokens
                .iter()
                .flat_map(|t| {
                    bits[*t as usize * h..][..h]
                        .iter()
                        .map(move |b| f32::from_bits((*b as u32) << 16) * scale)
                })
                .collect(),
        )
    }

    /// TIMING HARNESSES ONLY: one block forward with `DFLASH_CUT_*` stages skipped; returns the
    /// wall milliseconds, or `None` when the draft would not have run.
    #[cfg(target_os = "macos")]
    pub fn dflash_draft_timed(
        &self,
        tok: u32,
        past_len: usize,
        stream: u64,
        cut: u32,
    ) -> Option<f64> {
        let (isl, lm, embed) = (
            self.island.as_ref()?,
            self.mega.lm_head.as_deref()?,
            self.mega.embed.as_ref()?,
        );
        let scale = self.cfg.embedding_scale.unwrap_or(1.0);
        let g = isl.lock().unwrap();
        g.dflash_fence();
        let t = std::time::Instant::now();
        let r = g.dflash_draft_block_cut(&embed.0, scale, lm, Some(stream), tok, past_len, cut);
        matches!(r, Ok(Some(_))).then(|| t.elapsed().as_secs_f64() * 1e3)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn dflash_draft(&self, _tok: u32, _past_len: usize, _stream: u64) -> Vec<u32> {
        Vec::new()
    }

    /// `stream`'s draft context ring is valid and at `past_len` (see `MetalIsland::dflash_ready`).
    #[cfg(target_os = "macos")]
    pub fn dflash_ready(&self, stream: u64, past_len: usize) -> bool {
        self.island
            .as_ref()
            .is_some_and(|isl| isl.lock().unwrap().dflash_ready(Some(stream), past_len))
    }

    /// A finished sequence's recurrent bank row and draft context ring go back to their pools
    /// (see `MetalIsland::gdn_release_stream`, `dflash_release_stream`).
    #[cfg(target_os = "macos")]
    pub fn release_stream(&self, stream: u64) {
        if let Some(isl) = self.island.as_ref() {
            let g = isl.lock().unwrap();
            g.gdn_release_stream(stream);
            g.dflash_release_stream(stream);
        }
    }

    /// Attach a DFlash 2 draft checkpoint (a directory with `config.json` + `model.safetensors`).
    /// From here on every batched record taps the target and feeds its streams' context rings.
    /// Call before serving; nothing may be in flight.
    #[cfg(target_os = "macos")]
    pub fn dflash_attach(&self, dir: &std::path::Path) -> std::result::Result<(), String> {
        let isl = self.island.as_ref().ok_or("dflash: no Metal island")?;
        let draft = crate::gpu::metal::dflash2::Dflash2Draft::load(&self.ctx, dir)?;
        if draft.cfg.hidden != self.cfg.hidden_size || draft.cfg.vocab != self.cfg.vocab_size {
            return Err(format!(
                "dflash: the draft is for hidden {} / vocab {}, this model is {} / {}",
                draft.cfg.hidden, draft.cfg.vocab, self.cfg.hidden_size, self.cfg.vocab_size
            ));
        }
        isl.lock().unwrap().dflash_attach(&self.ctx, draft, false)
    }

    /// Can a block draft run on this machine (see `MetalIsland::dflash_supported`)?
    #[cfg(target_os = "macos")]
    pub fn dflash_supported(&self) -> bool {
        self.island
            .as_ref()
            .is_some_and(|i| i.lock().unwrap().dflash_supported())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn dflash_supported(&self) -> bool {
        false
    }

    /// Off macOS there is no Metal island, so there is nothing to attach a draft to.
    #[cfg(not(target_os = "macos"))]
    pub fn dflash_attach(&self, _dir: &std::path::Path) -> std::result::Result<(), String> {
        Err("dflash: the DFlash 2 draft runs on the Metal island (macOS only)".into())
    }

    #[cfg(target_os = "macos")]
    pub fn mtp_draft_chain(&self, tok: u32, past_len: usize, slots: &[u32], kk: usize) -> Vec<u32> {
        let t0 = spec_prof().then(std::time::Instant::now);
        // Snapshot BOTH hidden banks: chained steps alternate parity, and the residual carry in
        // either bank belongs to the live decode. Restored on every exit path below.
        let h = self.cfg.hidden_size;
        let snaps: Vec<Option<Vec<f32>>> = match self.island.as_ref() {
            Some(i) => {
                let g = i.lock().unwrap();
                (0..crate::gpu::concurrent_metal::DEPTH2_BANKS)
                    .map(|b| g.bmega_hidden_snapshot(b, 1, h))
                    .collect()
            }
            None => return Vec::new(),
        };
        let mut out = Vec::with_capacity(kk);
        let mut cur = tok;
        for s in 0..kk {
            let r = self.mtp_draft_reasoned(cur, past_len + s, slots, s > 0);
            if let Some(t0) = t0.filter(|_| s == 0) {
                spec_prof_record(SpecProfKind::Draft, 1, t0.elapsed().as_secs_f64() * 1e3);
            }
            match r {
                Ok(t) => {
                    out.push(t);
                    cur = t;
                }
                Err(why) => {
                    if spec_debug() {
                        eprintln!("[mtp] chain step {s} declined at past_len {past_len}: {why}");
                    }
                    break;
                }
            }
        }
        if let Some(g) = self.island.as_ref().map(|i| i.lock().unwrap()) {
            for (b, snap) in snaps.iter().enumerate() {
                if let Some(sn) = snap {
                    g.bmega_hidden_restore(b, sn);
                }
            }
        }
        out
    }

    /// `mtp_draft` with the decline reason kept: every early exit names what was missing, so
    /// a silent "no draft" is one env var away from a diagnosis instead of a probe.
    #[cfg(target_os = "macos")]
    fn mtp_draft_reasoned(
        &self,
        tok: u32,
        past_len: usize,
        slots: &[u32],
        // L337 — a CHAINED step: condition on the head's own previous-step residual (which the
        // previous step's record left in bm.hidden[(past_len-1) % 2]) instead of the trunk's
        // published row. The trunk publication is neither needed nor consulted.
        chained: bool,
    ) -> std::result::Result<u32, &'static str> {
        use crate::gpu::types::GpuMatWeight::Q4KS;
        use arf_core::model::batch::{ForwardBatch, SeqAttn};
        let head = self.mtp.as_ref().ok_or("no MTP head loaded")?;
        let mc = &self.cfg;
        let h = mc.hidden_size;
        let mc_h = h;

        // Every projection must be on the native Q4_K_S island path, or decline rather than
        // build a half-native record.
        let q4 = |w: &crate::gpu::types::GpuMatWeight| -> Option<*const crate::gpu::concurrent_metal::Q4ksMtl> {
            if let Q4KS { mtl: Some(m), .. } = w { Some(std::sync::Arc::as_ptr(m)) } else { None }
        };
        let (q, k, v, o) = (
            q4(&head.q_proj).ok_or("q_proj is not on the native Q4_K_S island path")?,
            q4(&head.k_proj).ok_or("k_proj is not on the native Q4_K_S island path")?,
            q4(&head.v_proj).ok_or("v_proj is not on the native Q4_K_S island path")?,
            q4(&head.o_proj).ok_or("o_proj is not on the native Q4_K_S island path")?,
        );
        let (gp, up, dn) = (
            q4(&head.gate_proj).ok_or("gate_proj is not on the native Q4_K_S island path")?,
            q4(&head.up_proj).ok_or("up_proj is not on the native Q4_K_S island path")?,
            q4(&head.down_proj).ok_or("down_proj is not on the native Q4_K_S island path")?,
        );

        // h_t: the trunk's `output_norm`'d hidden from the row that EMITTED `tok`. That is what
        // llama.cpp hands the head (`t_h_nextn` is taken AFTER `output_norm`, qwen35.cpp) — L261
        // read our record correctly and the reference wrongly: the raw residual is not it. The
        // record that produced the row published where it left it (`MtpHiddenSrc`); deriving
        // the bank from `past_len` here read the other bank on every plain step.
        let h_src = if chained {
            // The previous chain step's record ran at position past_len-1 and left blk.64's
            // output residual (pre shared_head_norm — the "prenorm" tensor upstream feeds
            // back) as row 0 of its hidden bank.
            (
                (past_len - 1) % crate::gpu::concurrent_metal::DEPTH2_BANKS,
                0,
            )
        } else {
            let src = self
                .mtp_h_src
                .get()
                .ok_or("no published hidden: the last step was not a one-sequence island record")?;
            let row = past_len
                .checked_sub(src.past0 + 1)
                .ok_or("draft position precedes the published rows")?;
            if row >= src.rows {
                return Err("draft position is past the published rows");
            }
            (src.bank, row)
        };

        // The draft is ONE row at position `past_len` — the position the emitted token occupies,
        // which is what the head conditions on.
        let batch = ForwardBatch {
            positions: vec![past_len as u32],
            seqs: vec![SeqAttn {
                stream_id: None, // the head has NO recurrent state; blk.64 is pure attention
                q_start: 0,
                q_len: 1,
                past_len,
                slots: slots.to_vec(),
                write_runs: Vec::new(),
                image_spans: Vec::new(),
            }],
            image_embeds: None,
            mrope_positions: None,
        };
        // The head scatters K/V for its row at `slots[past_len]` — into ITS OWN pool
        // (`kv.mtp_keys_mtl`), never a trunk layer's. That row is the head's history for
        // position `past_len` and is kept on purpose: llama.cpp re-runs the MTP layer over every
        // committed token to build exactly this. (L252 snapshot/restored the trunk's slot because
        // the record was bound to layer 0's pool — a 16-byte placeholder on this hybrid.)
        slots
            .get(past_len)
            .ok_or("no slot for the draft position")?;
        // L257 — and the RESIDUAL. The draft stages its input into bmega.hidden[bank] and the
        // record overwrites it with blk.64's output; that buffer is the trunk's cross-layer carry
        // for the live decode. Protect it exactly like the KV slot.
        // L337 — the snapshot/restore that used to live HERE moved to `mtp_draft_chain`:
        // a chained step s+1 reads the residual step s left in bm.hidden, so restoring after
        // every step would feed the chain its own pre-draft garbage. The chain snapshots BOTH
        // banks once and restores both on every exit path; a lone k=1 draft goes through the
        // same wrapper, so no caller can reach this function outside that protection.
        let _ = mc_h;
        self.mtp_draft_inner(
            tok,
            past_len,
            &batch,
            h_src,
            chained,
            head,
            (q, k, v, o, gp, up, dn),
        )
    }

    /// L253 — the draft's actual record. Split out so `mtp_draft` can wrap it in the KV
    /// snapshot/restore on every exit path without a `finally` construct.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn mtp_draft_inner(
        &self,
        tok: u32,
        past_len: usize,
        batch: &arf_core::model::batch::ForwardBatch,
        h_src: (usize, usize),
        // L337 — chained steps read the head's own output residual (bm.hidden) instead of the
        // trunk's published output_norm'd row (bm.normed). Step 0 passes false.
        h_from_hidden: bool,
        head: &std::sync::Arc<crate::gpu::GpuMtpHead>,
        w: (
            *const crate::gpu::concurrent_metal::Q4ksMtl,
            *const crate::gpu::concurrent_metal::Q4ksMtl,
            *const crate::gpu::concurrent_metal::Q4ksMtl,
            *const crate::gpu::concurrent_metal::Q4ksMtl,
            *const crate::gpu::concurrent_metal::Q4ksMtl,
            *const crate::gpu::concurrent_metal::Q4ksMtl,
            *const crate::gpu::concurrent_metal::Q4ksMtl,
        ),
    ) -> std::result::Result<u32, &'static str> {
        use crate::gpu::concurrent_metal::{MegaLayer, TokenSrc};
        let mc = &self.cfg;
        let h = mc.hidden_size;
        let (q, k, v, o, gp, up, dn) = w;

        let isl = self.island.as_ref().ok_or("no island")?.lock().unwrap();
        // Either format: the record dispatches both, and the MTP head only forwards this on to
        // it. The Q8-only restriction that used to live here was a property of the record, not of
        // speculation, so it lifted with the record.
        let lm_head = self
            .mega
            .lm_head
            .as_deref()
            .ok_or("lm_head has no island view")?;
        // The head norms with ITS norm before the shared lm_head, not the trunk's
        // `model.norm.weight` — the two differ per channel and argmax does not survive it.
        let final_norm = head
            .shared_head_norm_mtl
            .as_ref()
            .ok_or("shared_head_norm has no island view")?;

        // The head's own norms, as MTL views (L252).
        let attn_norm = head
            .attn_norm_mtl
            .as_ref()
            .ok_or("attn_norm has no island view")?;
        let ffn_norm = head
            .ffn_norm_mtl
            .as_ref()
            .ok_or("ffn_norm has no island view")?;

        // Geometry: blk.64 is a FULL-ATTENTION block, so take it from a real attention layer —
        // never layer 0, which is recurrent on this hybrid (the L156 trap).
        let a = self
            .layers
            .iter()
            .find(|l| l.gdn.is_none())
            .ok_or("no attention layer to take the head geometry from")?;
        let (nh, nkv, hd) = (mc.num_attention_heads, a.kv_heads, a.head_dim);
        let dims = isl.mega_layer_dims(mc.num_layers, 0); // index 64, built in L246
        let (kpool, vpool) = (
            self.kv.mtp_keys_mtl.as_ref().ok_or("no MTP KV pool")?,
            self.kv.mtp_values_mtl.as_ref().ok_or("no MTP KV pool")?,
        );
        let (cos, sin) = (
            self.mega.rope_cos.as_ref().ok_or("no island rope tables")?,
            self.mega.rope_sin.as_ref().ok_or("no island rope tables")?,
        );

        let megas = [MegaLayer {
            q: unsafe { &*q },
            k: unsafe { &*k },
            v: unsafe { &*v },
            o: unsafe { &*o },
            gate: Some(unsafe { &*gp }),
            up: Some(unsafe { &*up }),
            down: Some(unsafe { &*dn }),
            gate_q3k: None,
            up_q3k: None,
            down_q3k: None,
            down_q8: None,
            moe: None,
            gdn: None, // the head is pure attention — no recurrent branch
            input_norm: &attn_norm.0,
            post_norm: &ffn_norm.0,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            attn_gate: None,
            rope_interleaved: mc.rope_interleaved(),
            gate_silu: matches!(mc.gate_act, arf_core::config::GateAct::Silu),
            // qk_norm is REQUIRED on this arch (cfg.qk_norm = true) and blk.64 ships both,
            // so a missing view is a load bug, not an optional feature — decline rather than
            // substitute anything.
            q_norm: &head
                .q_norm_mtl
                .as_ref()
                .ok_or("q_norm has no island view")?
                .0,
            k_norm: &head
                .k_norm_mtl
                .as_ref()
                .ok_or("k_norm has no island view")?
                .0,
            value_norm: mc.value_norm,
            kpool: &kpool.0,
            vpool: &vpool.0,
            kv_q8: None,
            kpool_f16: None,
            vpool_f16: None,
            rope_cos: cos,
            rope_sin: sin,
            qkv_dims: dims.0,
            rope_dims: dims.1,
            scatter_dims: dims.2,
            attn_dims: dims.3,
            scale_dims: dims.4,
            nh,
            nkv,
            hd,
            // blk.64 has the SAME joint q+gate packing as the trunk's attention layers
            // (q is [5120, 12288] = 2 x nh x hd), so this must be armed or the record
            // would read the gate half as query.
            q_joint_gate: true,
        }];
        // ---- the head's INPUT: x = eh_proj([enorm(embed(tok)) || hnorm(h_t)]) ----
        // `mtp_project` (L245) writes it straight into the record's hidden slot, and
        // TokenSrc::PreEmbedded (L244) stops the record from overwriting it with an embed.
        let embed_table = self
            .mega
            .embed
            .as_ref()
            .ok_or("embed table has no island view")?;
        let eh = match &head.eh_proj {
            crate::gpu::types::GpuMatWeight::Q4KS { mtl: Some(m), .. } => m.clone(),
            _ => return Err("eh_proj is not on the native Q4_K_S island path"),
        };
        isl.mtp_stage_input(
            &self.ctx,
            embed_table,
            tok,
            h_src,
            &head.enorm_mtl.as_ref().ok_or("enorm has no island view")?.0,
            &head.hnorm_mtl.as_ref().ok_or("hnorm has no island view")?.0,
            &eh,
            h,
            mc.embedding_scale.unwrap_or(1.0),
            mc.rms_norm_eps as f32,
            past_len % 2,
            h_from_hidden,
        )
        .map_err(|e| {
            if spec_debug() {
                eprintln!("[mtp] mtp_stage_input: {e}");
            }
            "staging the head's input failed"
        })?;

        // ---- run blk.64 as a ONE-LAYER record over that hidden ----
        let out_bank = self
            .scratch
            .out_tokens_mtl
            .as_ref()
            .ok_or("no island out-token bank")?
            .0
            .as_ref();
        let seq_slots: Vec<&[u32]> = vec![&batch.seqs[0].slots];
        let past = [past_len];
        let r = isl.batched_megakernel_record(
            &self.ctx,
            &megas,
            final_norm,
            lm_head,
            out_bank,
            embed_table,
            TokenSrc::PreEmbedded,
            mc.embedding_scale.unwrap_or(1.0),
            None,
            1,
            h,
            mc.vocab_size,
            &seq_slots,
            &past,
            megas[0].nh,
            megas[0].nkv,
            megas[0].hd,
            mc.intermediate_size,
            1,
            1,
        );
        r.map_err(|_| "the head's record failed")?;

        // FENCE, then read the MTL buffer the record ACTUALLY wrote. Two bugs were stacked
        // here: reading `self.scratch.out_tokens` (the WGPU twin, which this Metal record never
        // touches — it returned a stale 0 every time, which is exactly the 'MTP draft: 0' the
        // log showed), and reading before the GPU had run the record at all.
        isl.wait_batched_step(past_len);
        let tok_out = {
            use objc2_metal::MTLBuffer;
            unsafe { *(out_bank.contents().as_ptr() as *const u32) }
        };
        drop(isl);
        Ok(tok_out)
    }

    /// Publish where the record just submitted left the trunk's `output_norm`'d rows, for
    /// `mtp_draft` to condition on. Only a run of consecutive positions of ONE sequence
    /// qualifies — a decode step at b=1, a verify window, a prefill chunk — and not a k-step
    /// chain, whose last row is not at `past_len[0]`.
    #[cfg(target_os = "macos")]
    fn publish_mtp_hidden(&self, seq_slots: &[&[u32]], past_len: &[usize], b: usize, k: usize) {
        let one_seq_run = k == 1
            && seq_slots.windows(2).all(|w| w[0] == w[1])
            && past_len.windows(2).all(|w| w[1] == w[0] + 1);
        self.mtp_h_src
            .set(one_seq_run.then(|| crate::gpu::MtpHiddenSrc {
                bank: past_len[0] % crate::gpu::concurrent_metal::DEPTH2_BANKS,
                past0: past_len[0],
                rows: b,
            }));
    }

    #[cfg(not(target_os = "macos"))]
    fn publish_mtp_hidden(&self, _: &[&[u32]], _: &[usize], _: usize, _: usize) {}

    /// SHARED-PREFIX VERIFY (MTP / spec-decode lever). Runs the k-row verify window
    /// `[committed, draft_0..draft_{k-1}]` — ONE sequence, k query rows sharing the prefix
    /// [0, prefix_len) — through the batched megakernel with the shared-prefix attention kernel
    /// (prefix K/V staged ONCE, consumed by all k rows) instead of the per-row attention that
    /// re-streams the prefix k times. Returns the k greedy argmax tokens (out_tokens[0..k]) so the
    /// caller consumes them directly (no CPU argmax-over-logits). SYNCHRONOUS (the record waits).
    ///
    /// `prefix_slots` = the seq's slot table for logical positions [0, prefix_len+k): the cached
    /// prefix slots followed by the k NEW-token pool slots (progressive). `window` = the k input
    /// tokens. Returns `None` (caller falls back to the WGSL forward_batch_all oracle) when the
    /// island is down, the model isn't MoE, KV is quantized, k>32 (KMAX), or ARF_NO_VERIFY_MEGA.
    /// L364 — macOS-only, for the same reason as `try_batched_megakernel`: 17 references to
    /// `self.island` and the Metal scratch banks. On Linux the caller falls back to the WGSL
    /// `forward_batch_all` oracle, which is exactly what this returns `None` for on Mac.
    #[cfg(target_os = "macos")]
    pub fn try_batched_megakernel_verify(
        &self,
        window: &[u32],
        prefix_len: usize,
        prefix_slots: &[u32],
        stream_id: Option<u64>,
    ) -> Option<Vec<u32>> {
        self.verify_window(
            window,
            prefix_len,
            prefix_slots,
            stream_id,
            false,
            true,
            None,
            None,
        )
    }

    /// [`Self::try_batched_megakernel_verify`] for a SAMPLED request (speculative sampling).
    #[cfg(target_os = "macos")]
    pub fn verify_sampled(
        &self,
        window: &[u32],
        prefix_len: usize,
        prefix_slots: &[u32],
        stream_id: u64,
        sampler: &arf_core::sampling::RowSampler,
    ) -> Option<Vec<u32>> {
        self.verify_window(
            window,
            prefix_len,
            prefix_slots,
            Some(stream_id),
            false,
            true,
            None,
            Some(sampler),
        )
    }

    /// C2 (2026-09-26) — ONE verify record for SEVERAL streams: each segment is one stream's
    /// window (its pending token + drafts) at its own positions, its rows contiguous in the
    /// record. Every per-row stage (matmuls, KV scatter, lm_head, argmax, taps) is the shipped
    /// batched record; the recurrence runs serially with each row on its own bank row
    /// (`MetalIsland::verify_multi` selects the kernels that do that). Afterwards, per segment:
    /// the accept rule, the draft-ring commit of its accepted rows, and the rollback of a partial
    /// accept to its last accepted row's checkpoint (slot = the row's index in the record).
    ///
    /// `None` = declined BEFORE anything was submitted (the caller takes the plain step). Once
    /// the record has run, state has advanced for every stream: a failure after that point is a
    /// panic, never a silent fallback onto advanced state.
    ///
    /// SAMPLED SEGMENTS (2026-09-27): `samplers` is empty (every segment greedy — the record as it
    /// was) or one per segment. A segment with a sampler has its predictions DRAWN from its own
    /// rows of the record's logits (`arf_core::backend::segment_predictions`: rows
    /// `[start, start + w)` of the `[total, vocab]` lm_head output, which sits in the logits bank
    /// of the record's first row, `segs[0].prefix_len`), right after the record and BEFORE the
    /// accept count, the ring commit and the per-segment restore below read them — so all three
    /// follow the draws, as `verify_window` does for one sampled stream. A greedy segment keeps
    /// the argmax, byte for byte. A failure to draw after the record is a panic, never the argmax.
    #[cfg(target_os = "macos")]
    pub fn verify_segments(
        &self,
        segs: &[arf_core::backend::SegReq<'_>],
        samplers: &[Option<&arf_core::sampling::RowSampler>],
    ) -> Option<Vec<Vec<u32>>> {
        self.verify_segments_rec(segs, samplers, false)
            .map(|(p, _)| p)
    }

    /// The body of [`Self::verify_segments`]. `logits_rows` (2026-09-27, the batched decode
    /// logits of `hybrid_logits_via_windows`): also return each segment's LAST row of the record's
    /// logits (`arf_core::backend::segment_last_rows`), for a caller that samples from them
    /// itself — its predictions are then not the sequence's tokens, so they are not noted as the
    /// draft's committed tokens (a committed window's are not either, `verify_window`). Otherwise
    /// the second value is empty.
    #[cfg(target_os = "macos")]
    #[allow(clippy::type_complexity)]
    fn verify_segments_rec(
        &self,
        segs: &[arf_core::backend::SegReq<'_>],
        samplers: &[Option<&arf_core::sampling::RowSampler>],
        logits_rows: bool,
    ) -> Option<(Vec<Vec<u32>>, Vec<Vec<f32>>)> {
        use crate::gpu::concurrent_metal::{GDN_VERIFY_ROWS, SPEC_CKPT_ROWS};
        use arf_core::model::batch::{ForwardBatch, SeqAttn};
        let total: usize = segs.iter().map(|g| g.window.len()).sum();
        let isl = self.island.as_ref()?;
        // 2026-10-05: up to GDN_VERIFY_ROWS rows (4 streams as 7 + 7 + 1 + 1) where the record
        // takes the fused kernel per segment and the replay restore; 8 under the controls that
        // take per-row checkpoints or the legacy kernels. Each stream's window <= 8 either way.
        let wide = std::env::var_os("ARF_NO_GDN_FUSED").is_none()
            && std::env::var_os("ARF_GDN_CKPT_ALL").is_none()
            && std::env::var_os("ARF_SEG_LEGACY").is_none();
        let max_rows = if wide {
            GDN_VERIFY_ROWS
        } else {
            SPEC_CKPT_ROWS + 1
        };
        if segs.len() < 2
            || total > max_rows
            || segs.iter().any(|g| g.window.len() > SPEC_CKPT_ROWS + 1)
            || segs.iter().any(|g| g.window.is_empty())
            // the decode-row logits records are ONE committed row per segment: a longer window
            // would be judged by the accept rule and rolled back, which a given row must not be
            || (logits_rows && segs.iter().any(|g| g.window.len() != 1))
            || !self.layers.iter().any(|l| l.gdn.is_some())
            || self.kv.kv_quant.is_quantized()
            || std::env::var_os("ARF_SPEC_SHARED_PREFIX").is_some()
            || std::env::var_os("ARF_KV_F16").is_some()
        {
            return None;
        }
        // Everything a draw could trip on is checked HERE, before the record: one sampler per
        // segment, and a sampled draft's q (a sampled segment's since 2026-09-27 — the actor's
        // `multi_sampled_draft`) covering every drafted row, or `sample_rows` would assert after
        // the state moved.
        if !samplers.is_empty()
            && (samplers.len() != segs.len()
                || segs.iter().zip(samplers).any(|(g, s)| {
                    s.is_some_and(|s| !s.draft_q.is_empty() && s.draft_q.len() + 1 < g.window.len())
                }))
        {
            return None;
        }
        // two segments of one stream would race one recurrent bank row
        for (i, a) in segs.iter().enumerate() {
            if segs[..i].iter().any(|b| b.stream == a.stream)
                || a.slots.len() < a.prefix_len + a.window.len()
            {
                return None;
            }
        }
        isl.lock()
            .unwrap()
            .gdn_prepare_verify(&self.ctx, total)
            .ok()?;
        let mut seqs: Vec<SeqAttn> = Vec::with_capacity(total);
        let mut positions: Vec<u32> = Vec::with_capacity(total);
        let mut tokens: Vec<u32> = Vec::with_capacity(total);
        let mut starts = Vec::with_capacity(segs.len());
        for g in segs {
            starts.push(seqs.len());
            let w = g.window.len();
            let slots: Vec<u32> = g.slots[..g.prefix_len + w].to_vec();
            for r in 0..w {
                seqs.push(SeqAttn {
                    stream_id: Some(g.stream),
                    q_start: seqs.len(),
                    q_len: 1,
                    past_len: g.prefix_len + r,
                    slots: slots.clone(),
                    write_runs: Vec::new(),
                    image_spans: Vec::new(),
                });
                positions.push((g.prefix_len + r) as u32);
            }
            tokens.extend_from_slice(g.window);
        }
        let batch = ForwardBatch {
            positions,
            seqs,
            image_embeds: None,
            mrope_positions: None,
        };
        {
            let g = isl.lock().unwrap();
            g.set_gdn_serial_rows(true);
            g.set_gdn_serial_commit_all(false);
            g.set_verify_multi(true);
            g.set_verify_segs(
                starts
                    .iter()
                    .zip(segs)
                    .map(|(&rs, g)| (rs, g.window.len()))
                    .collect(),
            );
            g.set_skip_lm_head(false);
        }
        let _flags = VerifyFlags(isl);
        let out_bank = self.scratch.out_tokens_mtl.as_ref()?.0.as_ref();
        // ARF_MULTI_PROF=1 (2026-10-05): the record (encode + submit + wait) and what follows it
        // per stream (draft-context commit, recurrent restore), separately — the actor's `verify=`
        // is their sum, measured at 2x a one-stream 8-row verify.
        let prof = std::env::var_os("ARF_MULTI_PROF").is_some();
        let t_rec = std::time::Instant::now();
        let pending = self.submit_batched_step(
            crate::gpu::concurrent_metal::TokenSrc::Cpu(&tokens),
            &batch,
            out_bank,
        )?;
        let t_submitted = t_rec.elapsed();
        // SUBMITTED: every stream's state has moved. No decline from here on.
        let preds = self.read_pending(&pending, out_bank);
        let t_record = t_rec.elapsed();
        // with ARF_BMEGA_GPUTIME too: the record's hardware GPU time and its matmul shapes, as
        // `verify_window` prints them for the one-stream record
        if prof {
            let g = isl.lock().unwrap();
            if let Some((gsum, gspan, nch)) = g.take_step_gpu_time() {
                let census: Vec<String> = g
                    .take_mm_census()
                    .iter()
                    .map(|((n, kk), c)| format!("{kk}->{n}x{c}"))
                    .collect();
                eprintln!(
                    "[gputime] multi segs={} rows={total} gpu_sum={gsum:.1} gpu_span={gspan:.1} \
                     chunks={nch} | matmuls: {}",
                    segs.len(),
                    census.join(" ")
                );
            }
        }
        assert_eq!(
            preds.len(),
            total,
            "multi-segment verify: {} preds for {total} rows",
            preds.len()
        );
        debug_assert_eq!(starts, arf_core::backend::segment_starts(segs));
        // SAMPLED segments: each one's rows drawn by its own sampler from the record's logits,
        // BEFORE anything below (accept count, ring commit, restore) uses them. The logits bank is
        // the record's first row's (`sbank = past_len[0] % DEPTH2_BANKS`, as `verify_window`
        // reads it with `prefix_len`); every row's full-vocabulary logits stay there until the
        // next record. Greedy segments are the argmax slices, exactly as before.
        let vocab = self.cfg.vocab_size;
        let sampled = samplers.iter().any(Option::is_some);
        let logits: Vec<f32> = if sampled || logits_rows {
            let l = isl
                .lock()
                .unwrap()
                .bmega_logits_snapshot(segs[0].prefix_len, total, vocab)
                .expect("multi-segment verify: the record's logits are not readable");
            // ARF_SPEC_SAMPLE_CHECK=1 (rule 8): the rows drawn from (or handed back) must be the
            // rows the GPU argmaxed — a wrong bank or row offset makes nearly every row disagree.
            if std::env::var_os("ARF_SPEC_SAMPLE_CHECK").is_some() {
                let bad = (0..total)
                    .filter(|&r| {
                        arf_core::sampling::argmax(&l[r * vocab..(r + 1) * vocab]) != preds[r]
                    })
                    .count();
                eprintln!(
                    "[spec-sample-check] multi-segment{}: {total} rows, {bad} whose logits argmax differs from the GPU's",
                    if logits_rows { " (decode-row logits)" } else { "" }
                );
            }
            l
        } else {
            Vec::new()
        };
        let seg_preds: Vec<Vec<u32>> = if sampled {
            let t0 = spec_prof().then(std::time::Instant::now);
            let drawn =
                arf_core::backend::segment_predictions(segs, samplers, &preds, &logits, vocab)
                    .expect("multi-segment sampled verify: drawing a segment's rows failed");
            // rule 7: a sampled draft's q reached a segment's draws (2026-09-27)
            let with_q = samplers
                .iter()
                .flatten()
                .filter(|s| !s.draft_q.is_empty() && !s.params.is_greedy())
                .count();
            if with_q > 0 {
                static SHOWN_Q: std::sync::Once = std::sync::Once::new();
                SHOWN_Q.call_once(|| {
                    eprintln!(
                        "[spec-multi] {with_q} of {} segments carry their sampled draft's q: \
                         min(1, p/q) acceptance",
                        segs.len()
                    )
                });
            }
            if let Some(t0) = t0 {
                eprintln!(
                    "[spec-sample] multi-segment: {} sampled segments, {total} rows, drawn in {:.2} ms",
                    samplers.iter().filter(|s| s.is_some()).count(),
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            // once per segment count (rule 7: the draws ran in the backend, from which rows)
            static SHOWN_S: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let bit = 1u32 << segs.len().min(31);
            if SHOWN_S.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0 {
                eprintln!(
                    "[spec-multi] {} segments, sampled rows drawn from the record's logits (bank {}): {:?}",
                    segs.len(),
                    segs[0].prefix_len % crate::gpu::concurrent_metal::DEPTH2_BANKS,
                    segs.iter()
                        .zip(&starts)
                        .zip(samplers)
                        .map(|((g, &rs), s)| (
                            g.stream,
                            rs..rs + g.window.len(),
                            if s.is_some() { "sampled" } else { "greedy" }
                        ))
                        .collect::<Vec<_>>()
                );
            }
            drawn
        } else {
            starts
                .iter()
                .zip(segs)
                .map(|(&rs, g)| preds[rs..rs + g.window.len()].to_vec())
                .collect()
        };
        let g = isl.lock().unwrap();
        let banks = g.gdn_step_rows_snapshot();
        g.set_verify_multi(false);
        g.set_gdn_serial_rows(false);
        let mut runs = Vec::with_capacity(segs.len());
        let mut restores = Vec::new();
        let mut out = Vec::with_capacity(segs.len());
        for ((g_i, seg), p) in segs.iter().enumerate().zip(seg_preds) {
            let (rs, w) = (starts[g_i], seg.window.len());
            let acc = arf_core::model::speculative::accepted_run(seg.window, &p).len();
            let rows = acc.min(w);
            let bank = banks.get(rs).copied().unwrap_or(rs);
            assert!(
                (rs..rs + w).all(|r| banks.get(r).copied().unwrap_or(r) == bank),
                "multi-segment verify: segment {g_i} spans more than one bank row ({banks:?})"
            );
            runs.push(crate::gpu::metal::dflash2_ctx::CommitRun {
                stream: Some(seg.stream),
                row0: rs,
                rows,
                start: seg.prefix_len,
            });
            if acc < w {
                restores.push((g_i, rs, acc - 1, bank as u32));
            }
            g.dflash_note_tokens(Some(seg.stream), &seg.window[..rows]);
            if !logits_rows {
                g.dflash_note_tokens(Some(seg.stream), &p[..rows]);
            }
            out.push(p);
        }
        let t_after = std::time::Instant::now();
        if let Err(e) = g.dflash_commit_runs(&runs) {
            eprintln!("[dflash] {e}");
        }
        let t_commit = t_after.elapsed();
        if let Err(e) = g.gdn_restore_segments(&restores) {
            panic!("multi-segment verify: recurrent restore failed after the record ran: {e}");
        }
        if prof {
            eprintln!(
                "[multi-prof-rec] segs={} rows={total} encode+submit={:.2} ms record={:.2} ms \
                 commit={:.2} ms restore={:.2} ms ({} restores)",
                segs.len(),
                t_submitted.as_secs_f64() * 1e3,
                t_record.as_secs_f64() * 1e3,
                t_commit.as_secs_f64() * 1e3,
                (t_after.elapsed() - t_commit).as_secs_f64() * 1e3,
                restores.len()
            );
        }
        // once per segment count (rule 7: each shape is shown to have run). Not for the decode-row
        // logits records, which log their own line (`hybrid_decode_rows_logits`): a `[spec-multi]`
        // line is the evidence that multi-stream SPECULATION ran.
        static SHOWN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let bit = 1u32 << segs.len().min(31);
        if !logits_rows
            && (SHOWN.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0 || spec_prof())
        {
            eprintln!(
                "[spec-multi] {} segments {:?} in one {total}-row record, restores {restores:?}",
                segs.len(),
                segs.iter()
                    .zip(&starts)
                    .map(|(g, rs)| (g.stream, *rs, g.window.len()))
                    .collect::<Vec<_>>()
            );
        }
        let rows = if logits_rows {
            arf_core::backend::segment_last_rows(segs, &logits, vocab)
                .expect("multi-segment verify: the record's logits do not cover its rows")
        } else {
            Vec::new()
        };
        Some((out, rows))
    }

    /// One PREFILL window of a prompt. `Ok(pred)` = it ran (`pred` is the token predicted after the
    /// window's last row — meaningful only when `last`); `Err(advanced)` = it declined, and
    /// `advanced` says whether any recurrent/KV state had ALREADY moved (then the prompt cannot be
    /// handed to another path).
    ///
    /// THE LAST WINDOW IS SPLIT when it is big: only its final row's prediction is read, but the
    /// lm_head ran for every row of it — on the batched GEMV for 17..=128 rows, ~2.4 ms a row on
    /// the 27B, i.e. ~270 ms of a 114-token prompt's 1.30 s to first token. So rows [0, k-1) run
    /// with the lm_head skipped and row k-1 runs alone — a one-row step, the original proven
    /// prefill shape (L144) — for one extra ~48 ms pass. Worth it past ~24 rows; at <= 16 rows the
    /// lm_head is on the MPP path already and the window is left whole.
    #[cfg(target_os = "macos")]
    fn prefill_window(
        &self,
        win: &[u32],
        prefix: usize,
        slots: &[u32],
        stream_id: Option<u64>,
        last: bool,
    ) -> std::result::Result<Option<u32>, bool> {
        const SPLIT_MIN_ROWS: usize = 24;
        let k = win.len();
        let run = |w: &[u32], at: usize, need: bool| {
            self.verify_window(w, at, slots, stream_id, true, need, None, None)
                .filter(|p| p.len() == w.len())
        };
        if last && k > SPLIT_MIN_ROWS && std::env::var_os("ARF_NO_PREFILL_LAST_ROW").is_none() {
            run(&win[..k - 1], prefix, false).ok_or(false)?;
            let p = run(&win[k - 1..], prefix + k - 1, true).ok_or(true)?;
            return Ok(p.last().copied());
        }
        run(win, prefix, last)
            .map(|p| p.last().copied())
            .ok_or(false)
    }

    /// The k-row window of ONE sequence. `commit_all = false` is speculative verify: afterwards
    /// the recurrent state is rolled back to the last ACCEPTED row. `commit_all = true` is
    /// PREFILL: the rows are the prompt itself, every one of them is history, and the state must
    /// stay where row k-1 left it — the accept rule (prediction == next window token) is
    /// meaningless for given tokens and would roll a prompt back almost every window.
    #[cfg(target_os = "macos")]
    fn verify_window(
        &self,
        window: &[u32],
        prefix_len: usize,
        prefix_slots: &[u32],
        stream_id: Option<u64>,
        commit_all: bool,
        // Only meaningful with `commit_all`: this window's predictions will be READ (it is the
        // prompt's last). Every other prefill window skips the lm_head — see `skip_lm_head`.
        preds_needed: bool,
        // `Some(buf)` = the window's tokens live in `buf` on the GPU (a GPU-selected draft) and
        // the record embeds from it via `TokenSrc::GpuBank`; `window` then supplies only `k`.
        gpu_tokens: Option<&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        // `Some` = a SAMPLED request's verify (speculative sampling, 2026-09-26): each row's
        // prediction is the token the sampler draws from that row's logits, not the argmax, and
        // everything after the record (accept count, ring commit, rollback) follows the draws.
        sampler: Option<&arf_core::sampling::RowSampler>,
    ) -> Option<Vec<u32>> {
        use arf_core::model::batch::{ForwardBatch, SeqAttn};
        let vdbg = std::env::var_os("ARF_PREFILL_FAST_DEBUG").is_some();
        macro_rules! vbail {
            ($w:expr) => {{
                if vdbg {
                    eprintln!("[verify-entry] decline: {}", $w);
                }
                return None;
            }};
        }
        if std::env::var_os("ARF_NO_VERIFY_MEGA").is_some() {
            vbail!("ARF_NO_VERIFY_MEGA")
        }
        let dbg = std::env::var_os("ARF_BATCH_MEGA_DEBUG").is_some();
        let k = window.len();
        // 32 is the shared-prefix verify kernel's tg-array bound (KMAX). A COMMITTED window never
        // arms that kernel's rollback machinery and may be as wide as the prefill MPP path serves.
        let kmax = if commit_all && std::env::var_os("ARF_SPEC_SHARED_PREFIX").is_none() {
            PREFILL_WINDOW
        } else {
            32
        };
        if k == 0 || k > kmax {
            vbail!("k out of range")
        } // KMAX = 32 (the verify kernel's tg-array bound; L1: was 16)
          // Verify targets the MoE island (qwen3-coder-30B). Gate the same as try_batched_megakernel:
          // island up, MoE, KV not quantized — else fall back to the WGSL oracle.
        if self.island.is_none() {
            vbail!("no island")
        }
        // L227 — the MoE-only gate is GONE. It dated from when verify targeted qwen3-coder-30B;
        // the record itself is arch-agnostic (it swaps only the attention dispatch), and the
        // recurrent hazard that actually blocked a dense hybrid is now handled by
        // set_gdn_serial_rows above.
        let _ = &self.cfg.mlp;
        if self.kv.kv_quant.is_quantized() {
            vbail!("kv quantized")
        }
        // f16-KV GUARD (2026-07-28 recon finding): the verify record forces the f32 pools
        // (use_kv_f16 requires !use_verify), but in f16 mode decode scatters ONLY f16 — the f32
        // pool is STALE for decoded tokens, so an island verify would silently accept against
        // wrong context. Bail → callers fall back to the WGSL oracle (forward_batch_impl), whose
        // f16 import leg restores the batch's past slots correctly.
        //
        // L11 — the guard is SATISFIED, NOT WEAKENED, when ARF_SPEC_FUSED_VERIFY is armed:
        // instead of bailing we re-arm the f32 pool and IMPORT this window's prefix slots
        // f16→f32 first (`convert_kv_slots(.., false)` = the same f16→f32 leg L5b runs before
        // its prefill packs, batch.rs ~:1167). After the import the f32 pool is authoritative
        // for exactly the slots this window reads, which is precisely what the guard demands.
        // Any import failure falls back to the original bail (the WGSL oracle), so a broken
        // convert can never let verify read stale context.
        let f16_live = std::env::var_os("ARF_KV_F16").is_some()
            && self.kv.keys_mtl_f16.iter().any(|o| o.is_some());
        // L309 — with the f16 pool live the record reads AND scatters f16 (the L95 twin
        // kernel); the f32 import that used to precede an f16 verify (L93's tax) is gone. A
        // build without the twin cannot verify against the pool decode uses, so it declines.
        if f16_live
            && !self
                .island
                .as_ref()
                .map(|i| {
                    i.lock()
                        .unwrap()
                        .has_pipeline("attention_verify_shared_prefix_f16")
                })
                .unwrap_or(false)
        {
            vbail!("f16 KV is live but the f16 verify kernel is not compiled")
        }
        // The full progressive slot table must cover [0, prefix_len+k).
        if prefix_slots.len() < prefix_len + k {
            vbail!("slot table too short")
        }
        let slots: Vec<u32> = prefix_slots[..prefix_len + k].to_vec();
        // Build the k-row window as ONE seq's k query rows (each row r: q_len=1, past_len=prefix_len+r,
        // its scatter target = slots[past_len[r]] = the r-th new-token slot). Every row shares the SAME
        // full slot table so row k-1 (whose base the shared-prefix kernel reads) sees the whole prefix
        // + all k drafted slots. The record's per-row scatter still writes each row to slots[past_len].
        let seqs: Vec<SeqAttn> = (0..k)
            .map(|r| SeqAttn {
                // L233 — CARRY THE REAL SEQUENCE ID. The k rows ARE k causal positions of one
                // sequence, so they must run on THAT sequence's recurrent bank, seeded with its
                // actual state — and L227's serial row chaining is what makes advancing it k times
                // correct. Passing None sent every verify to the id-less scratch bank (row 0), where
                // it predicted from state belonging to no sequence: L232 measured
                // window=[20956,1,6681] -> preds=[198,6918,13], pure noise, and committing one of
                // those poisoned the real stream.
                stream_id,
                q_start: 0,
                q_len: 1,
                past_len: prefix_len + r,
                slots: slots.clone(),
                write_runs: Vec::new(), // unused by the batched megakernel (it uses slots[past_len])
                image_spans: Vec::new(),
            })
            .collect();
        let batch = ForwardBatch {
            positions: (prefix_len as u32..(prefix_len + k) as u32).collect(),
            seqs,
            image_embeds: None,
            mrope_positions: None,
        };
        // Arm shared-prefix verify mode for the NEXT record, then run the standard batched megakernel
        // over the k rows. The record swaps ONLY the attention dispatch to attention_verify_shared_
        // prefix (all other ops — embed, norms, GEMV, MoE, lm_head, argmax — are the shipped B-row
        // path). Clear the flag after (a normal decode must never see verify mode).
        // L309 — CHECKPOINTS, NOT A HOST SNAPSHOT. L232 copied the whole recurrent row to the
        // host before verify and put it back after, which restored the state to BEFORE
        // `window[0]` — a position the scheduler had already consumed — and cost 2 x 144 MB of
        // memcpy per window. The record now leaves each row's post-update state in a checkpoint
        // slot on the GPU, and after the accept decision the bank is restored from the LAST
        // ACCEPTED row's slot (`gdn_restore_checkpoint`). The k scattered K/V rows are likewise
        // kept: accepted rows ARE the sequence's history, and a rejected row sits beyond the
        // causal frontier until the next real token overwrites it.
        let hybrid = self.layers.iter().any(|l| l.gdn.is_some());
        if hybrid {
            let (Some(isl), Some(_)) = (self.island.as_ref(), stream_id) else {
                vbail!("a hybrid verify needs the island and a stream id")
            };
            // A prefill window commits every row: no rollback, so no checkpoint slots — which
            // is what lets it be 16 rows wide when only SPEC_CKPT_ROWS (3) slots exist.
            if !commit_all {
                if let Err(e) = isl.lock().unwrap().gdn_prepare_verify(&self.ctx, k) {
                    vbail!(format!("recurrent checkpoints: {e}"))
                }
            }
        }
        if let Some(isl) = self.island.as_ref() {
            let g = isl.lock().unwrap();
            // L309 — the verify rows run the ORDINARY batched attention: each row is its own
            // SeqAttn over the full slot table with past_len = prefix_len + r, which is exactly
            // causal verify. `attention_verify_shared_prefix` (the read-the-prefix-once
            // optimisation, L220) is NOT armed: with it, both rows lose the prompt — the first
            // token that must be copied from the prompt comes out wrong, on the f32 and f16
            // pools alike (2026-08-30, md5 f9f95ec0e7eb vs greedy ede36e9a116a); without it the
            // speculative output is byte-identical to greedy. Opt back in only to debug it.
            if std::env::var_os("ARF_SPEC_SHARED_PREFIX").is_some() {
                // It has no 8-bit form: arming it under the 8-bit cache made the record return
                // Err MID-ENCODE and hung the request (identity check, 2026-09-23, 900 s timeout).
                // Not armed there — the ordinary batched attention verifies, as by default.
                if self.kv.q8.iter().any(Option::is_some) {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        eprintln!(
                            "[spec] ARF_SPEC_SHARED_PREFIX ignored: the shared-prefix verify kernel \
                             has no 8-bit KV form (ARF_NO_KV_Q8=1 to use it)"
                        )
                    });
                } else {
                    g.set_verify_prefix(Some((prefix_len, k)));
                }
            }
            // L227 — the k verify rows are k causal POSITIONS OF ONE sequence sharing one
            // recurrent bank, so the delta-rule recurrence must be chained row by row. Without
            // this every row reads the same pre-window state and the last writer wins: L222
            // measured that as `parity MISMATCH` with 0-8% acceptance on this hybrid.
            if self.layers.iter().any(|l| l.gdn.is_some()) {
                g.set_gdn_serial_rows(true);
                g.set_gdn_serial_commit_all(commit_all);
            }
            g.set_skip_lm_head(commit_all && !preds_needed);
        }
        // Declared after the arming block released its lock: it drops (and locks) only when no
        // island guard of this function is alive.
        let _flags = self.island.as_ref().map(VerifyFlags);
        // L11 INSTRUMENTATION (ARF_SPEC_PROF=1): time THIS verify window's island record.
        // Paired with the decode-step timer in try_m1_megakernel, this yields the
        // verify/decode ratio that gates the spec un-park (measured).
        let vt0 = spec_prof().then(std::time::Instant::now);
        // THE CYCLE GAP (2026-09-24): wall time from the previous verify's end to this one's start —
        // the draft, the actor's accept/sample/stream, and anything else outside the verify timer.
        // Verify + draft GPU explained ~90 of a ~100 ms cycle; this is the line that finds the rest.
        if let Some(t) = vt0 {
            if let Some(prev) = *SPEC_LAST_VERIFY_END.lock().unwrap() {
                eprintln!(
                    "[cycle] k={k} gap={:.2} ms",
                    t.duration_since(prev).as_secs_f64() * 1e3
                );
            }
        }
        // L331 — PHASE SPLIT. The single Verify timer could not say whether 119 ms is record
        // BUILD (CPU encoding ~80 dispatches x 64 layers) or GPU EXECUTION. They need opposite
        // fixes, so time them apart: `submit_batched_step` returns once the command buffer is
        // committed (CPU work), and `read_pending` spins until the GPU signals done (GPU work).
        let out = if spec_prof() {
            let out_bank = match self.scratch.out_tokens_mtl.as_ref() {
                Some(b) => b.0.as_ref(),
                None => return None,
            };
            let ts = std::time::Instant::now();
            let src = match gpu_tokens {
                Some(b) => crate::gpu::concurrent_metal::TokenSrc::GpuBank(b),
                None => crate::gpu::concurrent_metal::TokenSrc::Cpu(window),
            };
            let pending = self.submit_batched_step(src, &batch, out_bank)?;
            let submit_ms = ts.elapsed().as_secs_f64() * 1e3;
            let tr = std::time::Instant::now();
            let v = self.read_pending(&pending, out_bank);
            let read_ms = tr.elapsed().as_secs_f64() * 1e3;
            spec_prof_record(SpecProfKind::VerifySubmit, k, submit_ms);
            spec_prof_record(SpecProfKind::VerifyRead, k, read_ms);
            // ARF_BMEGA_GPUTIME: hardware GPU time of THIS verify's chunks against its wall, with
            // the fast path unserialized. gpu_sum ≈ wall → GPU-bound; gpu_sum ≪ wall → the GPU
            // idled and the CPU/submission side is the critical path.
            if let Some((gsum, gspan, nch)) = self
                .island
                .as_ref()
                .and_then(|i| i.lock().unwrap().take_step_gpu_time())
            {
                eprintln!(
                    "[gputime] verify k={k} wall={:.1} submit={submit_ms:.1} read={read_ms:.1} | gpu_sum={gsum:.1} gpu_span={gspan:.1} chunks={nch} | gpu/wall={:.0}%",
                    submit_ms + read_ms,
                    gsum / (submit_ms + read_ms) * 100.0
                );
                let census = self
                    .island
                    .as_ref()
                    .map(|i| i.lock().unwrap().take_mm_census())
                    .unwrap_or_default();
                if !census.is_empty() {
                    let line: Vec<String> = census
                        .iter()
                        .map(|((n, kk), c)| format!("{kk}->{n}x{c}"))
                        .collect();
                    eprintln!("[gputime]   matmuls this record: {}", line.join(" "));
                }
            }
            Some(v)
        } else {
            // The body of `try_batched_megakernel`, with the token source selectable.
            let out_bank = self.scratch.out_tokens_mtl.as_ref()?.0.as_ref();
            let src = match gpu_tokens {
                Some(b) => crate::gpu::concurrent_metal::TokenSrc::GpuBank(b),
                None => crate::gpu::concurrent_metal::TokenSrc::Cpu(window),
            };
            let pending = self.submit_batched_step(src, &batch, out_bank)?;
            Some(self.read_pending(&pending, out_bank))
        };
        if let Some(t0) = vt0 {
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            spec_prof_record(SpecProfKind::Verify, k, ms);
        }
        if std::env::var_os("ARF_Q8_DOWN_CHECK").is_some() {
            if let Some(i) = self.island.as_ref() {
                i.lock().unwrap().q8_down_check_report();
            }
        }
        // A GPU-selected draft: the window's REAL tokens are final now (the verify that embedded
        // them has completed), and everything below — the accept decision, the recurrent
        // checkpoint restore, the ring commit — must see them, not the caller's placeholder.
        // 🔴 The first version left the placeholder in scope here: `accepted_run` then counted
        // against zeros, the bank was restored to the wrong row, and b=1 text DIVERGED while the
        // 8-way batch (where the draft barely runs) stayed identical. Caught by gate 1.
        let real_window: Vec<u32>;
        let window: &[u32] = match gpu_tokens {
            Some(b) if out.is_some() => {
                use objc2_metal::MTLBuffer as _;
                real_window =
                    unsafe { std::slice::from_raw_parts(b.contents().as_ptr() as *const u32, k) }
                        .to_vec();
                &real_window
            }
            _ => window,
        };
        // SPECULATIVE SAMPLING: replace the argmax with each row's draw, from the record's own
        // logits (every verify row's full-vocabulary logits are in the shared bank until the next
        // record), AFTER the real window is known and BEFORE the accept count is used below. The
        // record has run: a failure here cannot decline (the state moved), and falling back to
        // the argmax would serve greedy text to a sampled request — the bug fixed 2026-09-26.
        // A SAMPLED GPU draft (2026-09-26, `dflash_chain_sampled`): its q is final now — the
        // kernel that wrote the window wrote q beside it — and the acceptor must use the q the
        // draft really drew from. The caller's sampler carries none (the draft never left the
        // GPU), so the verify fills it here. After a greedy chain `dflash_gpu_q` is `None` and
        // the point-mass rule stands, exactly as before.
        let gpu_q_sampler: Option<arf_core::sampling::RowSampler>;
        let sampler = match (sampler, gpu_tokens) {
            (Some(s), Some(_))
                if out.is_some() && s.draft_q.is_empty() && !s.params.is_greedy() =>
            {
                gpu_q_sampler = self
                    .island
                    .as_ref()
                    .and_then(|i| i.lock().unwrap().dflash_gpu_q())
                    .map(|q| {
                        static SHOWN_GQ: std::sync::Once = std::sync::Once::new();
                        SHOWN_GQ.call_once(|| {
                            eprintln!(
                                "[spec] sampled draft on the GPU selector: q over {} candidates (dflash_chain_sampled)",
                                q.first().map_or(0, |q| q.len())
                            )
                        });
                        arf_core::sampling::RowSampler {
                            params: s.params.clone(),
                            history: s.history.clone(),
                            draft_q: q,
                        }
                    });
                gpu_q_sampler.as_ref().or(Some(s))
            }
            (s, _) => s,
        };
        let out = match (sampler, out) {
            (Some(s), Some(greedy)) => {
                let vocab = self.cfg.vocab_size;
                let logits = self
                    .island
                    .as_ref()
                    .and_then(|i| {
                        i.lock()
                            .unwrap()
                            .bmega_logits_snapshot(prefix_len, k, vocab)
                    })
                    .expect("sampled verify: the record's logits are not readable");
                // ARF_SPEC_SAMPLE_CHECK=1 (rule 8): the rows sampled from must be the rows the GPU
                // argmaxed — each snapshot row's argmax against the record's own prediction.
                if std::env::var_os("ARF_SPEC_SAMPLE_CHECK").is_some() {
                    let bad = (0..k)
                        .filter(|&r| {
                            arf_core::sampling::argmax(&logits[r * vocab..(r + 1) * vocab])
                                != greedy[r]
                        })
                        .count();
                    eprintln!("[spec-sample-check] {k} rows, {bad} whose logits argmax differs from the GPU's");
                }
                let t0 = spec_prof().then(std::time::Instant::now);
                let drawn = s
                    .sample_rows(&logits, vocab, prefix_len, window)
                    .expect("sampled verify: sampling a verify row failed");
                static SHOWN: std::sync::Once = std::sync::Once::new();
                SHOWN.call_once(|| {
                    eprintln!(
                        "[spec] sampled verify: {k} rows drawn by the plain sampler (temperature {}, top_k {:?}, top_p {:?}, penalty {})",
                        s.params.temperature, s.params.top_k, s.params.top_p, s.params.repetition_penalty
                    )
                });
                if !s.draft_q.is_empty() {
                    static SHOWN_Q: std::sync::Once = std::sync::Once::new();
                    SHOWN_Q.call_once(|| {
                        eprintln!(
                            "[spec] sampled draft: q over {} candidates, min(1, p/q) acceptance",
                            s.draft_q.first().map_or(0, |q| q.len())
                        )
                    });
                }
                if let Some(t0) = t0 {
                    eprintln!(
                        "[spec-sample] {k} rows drawn in {:.2} ms",
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
                Some(drawn)
            }
            (_, o) => o,
        };
        if let Some(isl) = self.island.as_ref() {
            let g = isl.lock().unwrap();
            g.set_verify_prefix(None);
            g.set_skip_lm_head(false);
            g.set_gdn_serial_commit_all(false);
            g.set_gdn_serial_rows(false); // must never leak into a normal decode step
                                          // Roll the recurrent state back to exactly where the sequence actually is. Runs
                                          // after `try_batched_megakernel_step` returned, i.e. after the record completed,
                                          // so the GPU's own writes cannot land on top of the restore.
                                          // L309 — restore the bank to the state after the LAST ACCEPTED row. The accept rule
                                          // is shared with the actor (`speculative::accepted_run`), so both sides name the
                                          // same row. A full accept needs nothing: the bank already holds row k-1's state.
                                          // DFlash 2 — the rows that ARE history go into the draft's context ring, before the
                                          // next record overwrites the taps: every row of a prefill window, and rows 0..=a of a
                                          // verify window (their inputs were true tokens; a rejected row's hidden is of a token
                                          // the sequence never had). EVERY window feeds it — the draft's own, a prompt-lookup
                                          // one, an MTP one — or the ring would have a hole. No-op without a draft; a failed
                                          // commit invalidates the ring (the draft then stays silent) and never fails the step.
            if let Some(preds) = out.as_ref() {
                let rows = if commit_all {
                    k
                } else {
                    // `accepted_run().len()` is `accepted + 1`: it counts the BONUS token, whose
                    // hidden state no row of this window computed. Row `a` was run with
                    // `window[a]` — on a PARTIAL accept that is the first REJECTED draft token,
                    // so the ring's newest entry is the hidden of a token the sequence does not
                    // contain. MEASURED (158 windows, plain text): after a full accept the next
                    // window accepts nothing 5.3% of the time, after a partial accept 33.3% —
                    // 6x worse, mean 4.74 vs 2.73. That asymmetry is real and unexplained.
                    //
                    // ⛔ BUT COMMITTING ONE FEWER ROW IS NOT THE FIX, MEASURED 2026-09-21:
                    // it leaves the ring one short of `past_len`, and `dflash_draft_block`'s
                    // `committed == past_len` guard then SKIPS the draft almost every cycle —
                    // 8 block windows instead of ~80, 31.6 -> 23.2 tok/s. Any real fix has to
                    // give position `P+a` the hidden of the BONUS token, which means computing
                    // it (an extra row in the next record), not dropping the row.
                    //
                    // 🔴 CORRECTED 2026-09-27 (a line-by-line audit against another engine and SGLang): the
                    // premise above is wrong. With `a` drafts accepted, `accepted_run().len()` is
                    // a + 1, so rows 0..=a are committed: row 0 is the anchor and rows 1..=a are
                    // `window[1..=a]`, the ACCEPTED drafts — all tokens the sequence contains. No
                    // rejected token's hidden enters the ring; the code was right. another engine commits the
                    // same rows (`retained = accepted + 1`) and SGLang too. The zero-accept asymmetry
                    // (22% vs 12% in the 2026-09-27 log) is most plausibly hard stretches of text
                    // clustering, not a ring defect.
                    arf_core::model::speculative::accepted_run(window, preds).len()
                };
                if let Err(e) = g.dflash_commit_context(stream_id, rows.min(k), prefix_len) {
                    eprintln!("[dflash] {e}");
                }
                // the committed rows' tokens and the target's own tokens for them (the accepted
                // drafts and the bonus) widen the draft's candidate range (`dflash_vocab_eff`)
                let r = rows.min(k);
                g.dflash_note_tokens(stream_id, &window[..r.min(window.len())]);
                // A PREFILL window's predictions are not the model's tokens: rows that skipped the
                // lm_head (every non-final window, and the split last window's first k-1 rows) hold
                // STALE ids from earlier records — another request's, possibly far up the
                // vocabulary. Only a verify window's predictions (the accepted drafts and the
                // bonus) are real (2026-09-26).
                if !commit_all {
                    g.dflash_note_tokens(stream_id, &preds[..r.min(preds.len())]);
                }
            }
            if let (true, false, Some(preds)) = (hybrid, commit_all, out.as_ref()) {
                let a = arf_core::model::speculative::accepted_run(window, preds).len() - 1;
                if a + 1 < k {
                    // L331 — the restore is 96 bank_copy dispatches on a BLOCKING submit, and it
                    // runs AFTER the record on every partial accept (~20% of windows at p=0.80).
                    // It was never inside the Verify timer, so its cost has never been attributed.
                    let rt0 = spec_prof().then(std::time::Instant::now);
                    let rres = g.gdn_restore_checkpoint(a);
                    if let Some(t0) = rt0 {
                        spec_prof_record(
                            SpecProfKind::Restore,
                            k,
                            t0.elapsed().as_secs_f64() * 1e3,
                        );
                    }
                    if let Err(e) = rres {
                        eprintln!(
                            "[verify] recurrent restore FAILED ({e}); the state is ahead of the \
                             sequence — declining, output from here is suspect"
                        );
                        return None;
                    }
                }
            }
        }
        if dbg {
            eprintln!(
                "[verify-mega] k={k} prefix_len={prefix_len} -> {:?}",
                out.as_ref().map(|v| v.len())
            );
        }
        if spec_prof() {
            *SPEC_LAST_VERIFY_END.lock().unwrap() = Some(std::time::Instant::now());
        }
        out
    }

    /// Qwen3.8 IMAGE batches — `ForwardBatch::mrope_positions` is `Some` (2026-09-27; NOT yet
    /// run on a GPU when written: see the commit).
    ///
    /// The island record serves this arch; two things differ from a text batch and both reach it
    /// through a one-shot [`RowOverride`](crate::gpu::concurrent_metal::RowOverride):
    /// - every row's rope angle comes from its (t, h, w) triple (`rope_qk_b_mrope`) instead of
    ///   its KV index — image rows use the grid, and every row after an image sits `delta` below
    ///   its KV index;
    /// - an `<|image_pad|>` row's hidden state is the vision encoder's output, not the pad
    ///   token's embedding.
    ///
    /// Shapes: a pure-decode batch is ONE record with every row's triple (a text row's triple is
    /// `[p, p, p]`, which rotates as before). Anything else runs one sequence at a time: a text
    /// sequence with no layout goes back through `forward_batch_impl` unchanged (its sub-batch
    /// has `mrope_positions: None`), an image sequence's prefill chunk goes through committed
    /// windows exactly as `try_gdn_serial_prefill` feeds a text prompt, each window with its rows'
    /// override. Greedy only (the HTTP layer forces it for image requests).
    #[cfg(target_os = "macos")]
    fn forward_mrope_batch(
        &self,
        input_ids: &[u32],
        batch: &arf_core::model::batch::ForwardBatch,
        all_positions: bool,
        greedy: bool,
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        use crate::gpu::concurrent_metal::RowOverride;
        use arf_core::model::batch::{ForwardBatch, SeqAttn};
        let mrope = batch.mrope_positions.as_ref().ok_or("no mrope table")?;
        if !greedy || all_positions {
            return Err("image requests are greedy-only (no logits path yet)".into());
        }
        let isl = self.island.as_ref().ok_or("no Metal island")?;
        if !self.layers.iter().any(|l| l.gdn.is_some()) {
            return Err("M-RoPE is wired only for the qwen35 hybrid arch".into());
        }
        if !isl.lock().unwrap().has_mrope_rope() {
            return Err("rope_qk_b_mrope was not compiled into the island".into());
        }
        if mrope.len() != input_ids.len() {
            return Err("mrope table length != token count".into());
        }
        let s = arf_core::model::mrope::QWEN35_MROPE_SECTIONS;
        let sections = [s[0] as u32, s[1] as u32, s[2] as u32, s[3] as u32];
        let img = batch.image_embeds.as_ref();
        // flat row -> its embedding
        let embed_of = |row: usize| -> Option<Vec<f32>> {
            let ie = img?;
            let i = ie.rows.iter().position(|&r| r == row)?;
            Some(ie.embeds[i * ie.hidden..(i + 1) * ie.hidden].to_vec())
        };
        let arm = |o: Option<RowOverride>| isl.lock().unwrap().set_row_override(o);
        let ids_to_rows =
            |ids: Vec<u32>| ids.into_iter().map(|t| vec![f32::from_bits(t)]).collect();

        // Pure decode, no image rows: one record, every row with its own triple.
        if batch.seqs.iter().all(|s| s.q_len == 1)
            && input_ids.len() == batch.seqs.len()
            && img.is_none_or(|ie| ie.rows.is_empty())
        {
            arm(Some(RowOverride {
                mrope: Some(mrope.clone()),
                sections,
                embeds: Vec::new(),
            }));
            let r = self.try_batched_megakernel(input_ids, batch);
            arm(None);
            return r
                .map(ids_to_rows)
                .ok_or_else(|| "the batched record declined an image-sequence decode step".into());
        }

        // Otherwise: one sequence at a time.
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(batch.seqs.len());
        for s0 in &batch.seqs {
            let rows = s0.q_start..s0.q_start + s0.q_len;
            let ids = &input_ids[rows.clone()];
            let m = &mrope[rows.clone()];
            let embeds: Vec<(usize, Vec<f32>)> = rows
                .clone()
                .filter_map(|r| embed_of(r).map(|e| (r - s0.q_start, e)))
                .collect();
            let plain = embeds.is_empty()
                && m.iter()
                    .zip(&batch.positions[rows.clone()])
                    .all(|(t, &p)| *t == [p, p, p]);
            let seq = SeqAttn {
                q_start: 0,
                image_spans: Vec::new(),
                ..s0.clone()
            };
            if plain {
                // A text sequence: the unchanged path, exactly as if it were alone.
                let sub = ForwardBatch {
                    positions: batch.positions[rows.clone()].to_vec(),
                    seqs: vec![seq],
                    image_embeds: None,
                    mrope_positions: None,
                };
                out.extend(self.forward_batch_impl(ids, &sub, false, true));
                continue;
            }
            if s0.q_len == 1 {
                let sub = ForwardBatch {
                    positions: batch.positions[rows.clone()].to_vec(),
                    seqs: vec![seq],
                    image_embeds: None,
                    mrope_positions: None,
                };
                arm(Some(RowOverride {
                    mrope: Some(m.to_vec()),
                    sections,
                    embeds,
                }));
                let r = self.try_batched_megakernel(ids, &sub);
                arm(None);
                out.extend(ids_to_rows(
                    r.ok_or("the batched record declined an image row")?,
                ));
                continue;
            }
            // An image sequence's prefill chunk: committed windows, as try_gdn_serial_prefill.
            if s0.stream_id.is_none() {
                return Err("an image prefill needs the sequence's stream id".into());
            }
            if s0.slots.len() < s0.past_len + s0.q_len {
                return Err("slot table too short for the image prefill chunk".into());
            }
            let w_rows = prefill_window_rows();
            let n_win = ids.len().div_ceil(w_rows);
            let mut last_tok = None;
            for (w, win) in ids.chunks(w_rows).enumerate() {
                let row0 = w * w_rows;
                let last = w + 1 == n_win;
                let win_embeds: Vec<(usize, Vec<f32>)> = embeds
                    .iter()
                    .filter(|(r, _)| *r >= row0 && *r < row0 + win.len())
                    .map(|(r, e)| (r - row0, e.clone()))
                    .collect();
                arm(Some(RowOverride {
                    mrope: Some(m[row0..row0 + win.len()].to_vec()),
                    sections,
                    embeds: win_embeds,
                }));
                let preds = self.verify_window(
                    win,
                    s0.past_len + row0,
                    &s0.slots,
                    s0.stream_id,
                    true,
                    last,
                    None,
                    None,
                );
                arm(None);
                match preds {
                    Some(p) if p.len() == win.len() => last_tok = p.last().copied(),
                    // Mid-prompt the recurrent state is part-advanced: nothing can recover it.
                    _ => {
                        return Err(format!(
                            "image prefill window {w}/{n_win} declined (ARF_PREFILL_FAST_DEBUG=1 \
                             says why)"
                        ))
                    }
                }
            }
            out.push(vec![f32::from_bits(
                last_tok.ok_or("image prefill produced no token")?,
            )]);
        }
        Ok(out)
    }

    fn forward_batch_impl(
        &self,
        input_ids: &[u32],
        batch: &arf_core::model::batch::ForwardBatch,
        all_positions: bool,
        greedy: bool,
    ) -> Vec<Vec<f32>> {
        // STREAM-BOUNDARY stash clear: any prefill (q_len>1) marks a new chat, an edited
        // turn, or a continuation re-feed. Lookahead stashed under the old timeline must
        // not survive it — a position-collision serve on the new stream would emit a stale
        // token with zero GPU work and leave that position's KV row unwritten (review
        // finding, 2026-07-06; the hazard predates the spec branch — the m3 burst stashes
        // the same way).
        if batch.seqs.iter().any(|s| s.q_len > 1) {
            self.m1_burst_stash.borrow_mut().clear();
        }
        // Qwen3.8 IMAGE batch (2026-09-27): some sequence carries an M-RoPE layout. Its own path,
        // entered ONLY when `mrope_positions` is Some — every text batch skips this block and runs
        // exactly the code below. A batch this path cannot serve is a loud failure, never a quiet
        // fall-through: the generic loop below has no image splice for this arch and no M-RoPE,
        // so it would answer as if the image were not there.
        if batch.mrope_positions.is_some() {
            #[cfg(target_os = "macos")]
            let r = self.forward_mrope_batch(input_ids, batch, all_positions, greedy);
            #[cfg(not(target_os = "macos"))]
            let r: std::result::Result<Vec<Vec<f32>>, String> =
                Err("M-RoPE (Qwen3.8 image input) is wired only on the Metal island".into());
            match r {
                Ok(out) => return out,
                Err(e) => panic!("Qwen3.8 image batch cannot be served: {e}"),
            }
        }
        // 🔴 HYBRID + LOGITS WANTED (2026-09-23). Every island fast path below is gated on
        // `greedy`; a request that needs the raw logits — logprobs, a grammar/JSON constraint —
        // arrives non-greedy and fell through to the generic loop, which on a hybrid model is
        // GDN-FREE: 48 of 64 layers never run. MEASURED: the same prompt answered correctly
        // without logprobs and as `旁полуupt .IsNullOr…` with `logprobs: true`, on every
        // configuration back to the morning's (pre-existing, not a regression). Here the tokens
        // go through the island's own committed windows (the path that serves this model), and
        // the last row's logits come back from the record's logits buffer.
        #[cfg(target_os = "macos")]
        if !greedy
            && !all_positions
            && batch.image_embeds.is_none()
            && self.island.is_some()
            && self.layers.iter().any(|l| l.gdn.is_some())
            && std::env::var_os("ARF_NO_HYBRID_LOGITS_WINDOW").is_none()
        {
            if let Some(out) = self.hybrid_logits_via_windows(input_ids, batch) {
                return out;
            }
        }
        // N-GRAM SELF-SPEC observer (ARF_MEGA_SPEC): record every single-seq token —
        // prefill AND decode — into the bridge's history. The drafter's chains live mostly
        // in the prompt (code repeats itself), and prefill is the only place we see it.
        #[cfg(target_os = "macos")]
        if batch.seqs.len() == 1 && std::env::var_os("ARF_MEGA_SPEC").is_some() {
            self.m1_spec_observe(input_ids, &batch.seqs[0]);
        }
        // NATIVE MEGAKERNEL FAST PATH — for greedy pure-decode this is THE path (no slow fallback).
        // NOTE (2026-07-08): the silent fallback to the slow wgpu forward_batch_impl was the
        // bug that raced the island readback, returned zeros/garbage, timed out 10s/step, and made
        // conc look "slow". llama has ONE path; so do we now. For an ELIGIBLE step (macOS + island up
        // + MoE + greedy + pure-decode + KV not quantized) the megakernel MUST run — if it can't, we
        // PANIC with the reason instead of silently limping. Prefill / non-greedy / dense still use
        // the generic path below (correct + needed), but greedy MoE DECODE never falls through slow.
        // L144 — HYBRID PREFILL on the island, token-serially. Must come BEFORE the
        // `!all_positions` gate below: a prefill sets all_positions, so it would otherwise fall to
        // the generic loop (batch.rs GDN-free layer loop), which runs ATTENTION on the 48 recurrent
        // layers and produces plausible garbage (L143). Returns one continuation token, which is
        // what a prefill step yields.
        #[cfg(target_os = "macos")]
        if greedy && batch.seqs.len() == 1 && self.layers.iter().any(|l| l.gdn.is_some()) {
            match self.try_gdn_serial_prefill(input_ids, batch) {
                Some(ids) => {
                    if std::env::var_os("ARF_GDN_TRACE").is_some() {
                        eprintln!("[gdn] serial prefill OK: {} tokens", input_ids.len());
                    }
                    return ids.into_iter().map(|t| vec![f32::from_bits(t)]).collect();
                }
                // q_len==1 is DECODE, which belongs to try_batched_megakernel downstream — not a
                // refusal worth reporting. Only a real multi-token prefill falling through is bad.
                None if std::env::var_os("ARF_GDN_TRACE").is_some() && batch.seqs[0].q_len > 1 => {
                    eprintln!(
                        "[gdn] serial prefill REFUSED (q_len={}, ids={}) -> generic path, which has \
                         NO GDN branch: output will be wrong",
                        batch.seqs[0].q_len, input_ids.len()
                    );
                }
                None => {}
            }
        }
        #[cfg(target_os = "macos")]
        if greedy && !all_positions {
            // B=1 single chat → the m=1 megakernel; else the B-row batched megakernel.
            if let Some(ids) = self.try_m1_megakernel(input_ids, batch) {
                return ids.into_iter().map(|t| vec![f32::from_bits(t)]).collect();
            }
            if let Some(ids) = self.try_batched_megakernel(input_ids, batch) {
                // L179 — ARF_TOKEN_TRACE: the b sampled ids this step returns, next to the b
                // input ids it consumed. If the trunk diverges per row (L178: 100% at every
                // stage) but these ids are equal, the defect is between the GPU argmax and here.
                if std::env::var_os("ARF_TOKEN_TRACE").is_some() && ids.len() > 1 {
                    eprintln!("[tok] in={input_ids:?} out={ids:?}");
                }
                return ids.into_iter().map(|t| vec![f32::from_bits(t)]).collect();
            }
            if std::env::var_os("ARF_GDN_TRACE").is_some()
                && self.layers.iter().any(|l| l.gdn.is_some())
            {
                eprintln!(
                    "[gdn] BATCHED megakernel refused too (q_len={}) -> generic GDN-free path",
                    batch.seqs[0].q_len
                );
            }
            // ── G4 PATH INTEGRITY (ARF_STRICT_PATH=1) ─────────────────────────────────────
            // The comment above has long promised that greedy MoE decode "PANICs with the
            // reason instead of silently limping". It never did — both `try_` calls above fall
            // through to the generic (slow) path with no signal, which is exactly how a 5-10x
            // regression hides in a benchmark that still produces correct text.
            //
            // Opt-in (default OFF) so no shipping run changes behaviour, but any bench or serve
            // process can demand the fast path and get a LOUD failure instead of a quiet one:
            //   ARF_STRICT_PATH=1  -> panic here with the batch shape
            // This is the G4 gate: "never silent WGSL/slow fallback". Pure decode only — prefill,
            // non-greedy and dense legitimately use the generic path and are NOT covered.
            // Scope: PURE DECODE ONLY. q_len>1 rows are prefill chunks, which legitimately use the
            // generic path (and the prefill fast path below) — firing on those makes the gate
            // useless noise. Verified on 2026-08-08: an unscoped version tripped at conc8 with
            // q_lens=[3;8], i.e. prefill, not the regression the gate is for.
            let pure_decode = batch.seqs.iter().all(|s| s.q_len == 1);
            if pure_decode && std::env::var_os("ARF_STRICT_PATH").is_some() {
                let qlens: Vec<usize> = batch.seqs.iter().map(|s| s.q_len).collect();
                panic!(
                    "ARF_STRICT_PATH: greedy pure-decode FELL OFF the Metal megakernel to the \
generic/slow path. b={} q_lens={:?} kv_quantized={} — this is the silent 5-10x regression the \
strict gate exists to catch. Re-run with ARF_BATCH_MEGA_DEBUG=1 for the bail reason.",
                    batch.seqs.len(),
                    qlens,
                    self.kv.kv_quant.is_quantized()
                );
            }
            // ── PREFILL FAST PATH ────────────────────────────────────────────────
            // MEASURED 2026-08-03: the first turn of a real chat runs at 14 tok/s while every
            // later turn runs at 50-51, because the batched megakernel bails on `q_len != 1`
            // and prefill falls back to the SERIAL WGSL path (~40x slower).
            //
            // But the fast megakernel ALREADY processes multi-token windows: the spec-decode
            // verify path (try_batched_megakernel_verify) pushes k query rows of ONE sequence
            // through it as k B-rows (row r: q_len=1, past_len=prefix_len+r) and its attention
            // kernel gives row i a STRICTLY CAUSAL view of rows 0..i (PHASE 2 of
            // attention_verify_shared_prefix). That is exactly prefill semantics — token i
            // attends the prefix plus all earlier prompt tokens — and it is parity-green in
            // production today. So prefill needs WIRING, not a new kernel.
            //
            // Scope (deliberately narrow, first cut): ONE sequence, its own prompt chunk, no
            // images. The window is capped at KMAX=16 rows per call, so a long prompt is walked
            // in chunks. Any bail → the untouched serial path, so this can only help.
            // ARF_NO_PREFILL_FAST=1 disables it.
            // Embedded rows (Qwen3-Omni audio) are spliced per pack inside; M-RoPE and
            // bidirectional image spans still keep their own paths.
            let embeds_ok = batch.image_embeds.is_none()
                || (batch.mrope_positions.is_none()
                    && batch.seqs.iter().all(|s| s.image_spans.is_empty()));
            if batch.seqs.len() == 1
                && batch.seqs[0].q_len > 1
                && embeds_ok
                && std::env::var_os("ARF_NO_PREFILL_FAST").is_none()
            {
                // (L3) the mixed function subsumes the single-seq case — decode-row chunks
                // give conc1 prefill the same filled-tile GEMM economics as conc>1.
                match self.try_prefill_fast_mixed(input_ids, batch) {
                    Some(ids) => {
                        if std::env::var_os("ARF_PREFILL_FAST_DEBUG").is_some() {
                            eprintln!(
                                "[prefill-fast] ENGAGED q_len={} past={}",
                                batch.seqs[0].q_len, batch.seqs[0].past_len
                            );
                        }
                        return ids.into_iter().map(|t| vec![f32::from_bits(t)]).collect();
                    }
                    None => {
                        if std::env::var_os("ARF_PREFILL_FAST_DEBUG").is_some() {
                            eprintln!(
                                "[prefill-fast] DECLINED q_len={} past={} slots={} ids={}",
                                batch.seqs[0].q_len,
                                batch.seqs[0].past_len,
                                batch.seqs[0].slots.len(),
                                input_ids.len()
                            );
                        }
                    }
                }
            }
            // MIXED batch (multi-seq, some prefill): split fast paths — see try_prefill_fast_mixed.
            if batch.seqs.len() > 1
                && batch.seqs.iter().any(|s| s.q_len > 1)
                && embeds_ok
                && std::env::var_os("ARF_NO_PREFILL_FAST").is_none()
            {
                match self.try_prefill_fast_mixed(input_ids, batch) {
                    Some(ids) => {
                        if std::env::var_os("ARF_PREFILL_FAST_DEBUG").is_some() {
                            eprintln!(
                                "[prefill-fast] MIXED ENGAGED B={} prefill_seqs={}",
                                batch.seqs.len(),
                                batch.seqs.iter().filter(|s| s.q_len > 1).count()
                            );
                        }
                        return ids.into_iter().map(|t| vec![f32::from_bits(t)]).collect();
                    }
                    None => {
                        if std::env::var_os("ARF_PREFILL_FAST_DEBUG").is_some() {
                            eprintln!("[prefill-fast] MIXED DECLINED B={}", batch.seqs.len());
                        }
                    }
                }
            }
            // Eligible greedy pure-decode that DIDN'T take a megakernel = a real gap. Is this a pure
            // decode step (every seq q_len==1, total==nseq)? If so the fast path SHOULD have run.
            let pure_decode =
                input_ids.len() == batch.seqs.len() && batch.seqs.iter().all(|s| s.q_len == 1);
            if pure_decode
                && self.island.is_some()
                && !self.kv.kv_quant.is_quantized()
                && matches!(self.cfg.mlp, arf_core::config::MlpKind::Moe { .. })
            {
                panic!(
                    "FAST-PATH GAP: greedy MoE pure-decode (B={}) fell through the megakernel. \
                        Run with ARF_BATCH_MEGA_DEBUG=1 to see the bail reason — FIX the gap, do \
                        NOT restore a slow fallback (that was the 2026-07-08 slowness bug).",
                    batch.seqs.len()
                );
            }
        }
        // WHOLE-CALL timer: the daemon's 40s stall was never localized because every
        // probe timed a REGION (submit 94ms, readback 1.3ms, f16 mirror 0ms) and the sum was <100ms
        // of a 42s request. Time the ENTIRE call so the missing seconds have nowhere to hide.
        let _t_fb = std::time::Instant::now();
        let fb_prof = std::env::var_os("ARF_PREFILL_PROF").is_some();
        let _fb_guard = crate::gpu::generate::scopeguard_fb(
            fb_prof,
            _t_fb,
            batch.seqs.len(),
            batch.seqs.first().map(|s| s.q_len).unwrap_or(0),
        );
        // f16-mode POOL SYNC (import leg): the wgpu path below READS the f32 staging pool (prefill
        // attention gathers past KV by slot table — chunked prompts, prefix-cache hits, and decode
        // rows riding a mixed admission batch). In f16 mode the f16 pool is the SOURCE OF TRUTH
        // (decode scatters ONLY f16, and the f32 pool may be volatile/discarded between prefills),
        // so re-arm the f32 pool and import THIS batch's past slots f16→f32 before any encoding.
        #[cfg(target_os = "macos")]
        self.import_kv_f16_for_batch(batch);
        let mc = &self.cfg;
        let (h, hd) = (mc.hidden_size, mc.head_dim);
        let (nh, nkv) = (mc.num_attention_heads, mc.num_kv_heads);
        // Base/sliding geometry; the attention dispatches use PER-LAYER values
        // (q_dim_l/kv_dim_l/hd_l/nkv_l/group_l, computed in the loop) — gemma-4 global
        // layers differ. q_dim's only in-loop use was replaced by q_dim_l, so it's gone.
        let kv_dim = nkv * hd;
        let group = nh / nkv;
        let inter = mc.intermediate_size;
        // Gemma overrides the pre-softmax query scale (`query_pre_attn_scalar`);
        // everyone else uses 1/√head_dim. Matches the single-stream decode path.
        // Base/sliding scale; per-layer `scale_l` (computed in the loop) is what's
        // actually used at each attention dispatch (Gemma-4 global layers differ).
        let _ = mc.query_pre_attn_scalar;
        // Gemma's four-norm block + per-layer local/global RoPE must be reproduced
        // here too, or the serving path corrupts gemma logits (the single-stream
        // decode path handles it; forward_batch used to be Llama/Qwen-shaped only).
        // PLACEMENT, not formula: gates the four-norm block dispatches below.
        let is_gemma = mc.has_post_norms();
        // Gemma-4 GLOBAL layers have DIFFERENT geometry (head_dim 512, kv_heads 1,
        // full rotary) than the SLIDING base above. The q/k/v/attn buffers must fit
        // the LARGEST layer, and each layer's attention dispatch must use ITS OWN
        // geometry (computed per-layer in the loop) — exactly as decode.rs does, else
        // the global layers run attention with the wrong head_dim/kv_heads → garbage
        // (the batched-gemma4 corruption bug). For every non-gemma4 model these
        // max/base values are identical, so this is a no-op there.
        let (_max_hd, max_kv_dim) = mc.max_attn_dims();
        // L154 — see the note at the other `max_q_dim` site: the hybrid SSM family packs an
        // attention layer's q 2×, so `nh * max_hd` undersizes this by half.
        let max_q_dim = mc.max_q_dim();
        // GQA-grouped attention reads each K/V slot once and serves all `group`
        // query heads — a `group`× cut in the decode-bound KV traffic. Gated to the
        // kernel's shared-memory budget (group ≤ 8 partitions a 16 KB score tile;
        // head_dim ≤ 256 gives one output dim per lane). MHA (group==1) and larger
        // dims keep the per-(head,row) `attention_batched`.
        //
        // Context-pressure gate: at SHORT context, batched decode is weight-
        // bandwidth-bound (the matmul weight stream dwarfs the KV read), so GQA's KV
        // saving is <5% while its lower workgroup count (nkv vs nh) costs occupancy —
        // net neutral/slight loss. GQA wins once the KV read is a real share of step
        // bytes, i.e. when the batch's total cached positions are large. Measured on
        // Llama-1B Q4 (M3 Pro): at sum_ctx≈15k (8×~1900) GQA cut the p50 decode gap
        // 89→69 ms (−22%); at sum_ctx≈2k (short prompts) it was a slight loss. Gate
        // at 8192 cached positions (conservative — never hurts the short path).
        // `ARF_GQA=1`/`=0` force on/off for measurement.
        let gqa_ok = group > 1 && group <= 8 && hd <= 256;
        let sum_ctx: usize = batch.seqs.iter().map(|s| s.past_len + s.q_len).sum();
        let use_gqa = gqa_ok
            && match std::env::var("ARF_GQA").ok().as_deref() {
                Some("1") => true,
                Some("0") => false,
                _ => sum_ctx >= 8192,
            };
        let k = &self.kernels;
        let total = input_ids.len();
        let wg = |n: usize| [(n as u32).div_ceil(256), 1u32, 1u32];
        let f32b = std::mem::size_of::<f32>() as u64;

        // Each activation that is read by one dispatch and written by another lives
        // in its OWN buffer (a buffer can't be read + read_write in one dispatch):
        // hidden (read by norm, written by residual), normed (written by norm/
        // swiglu, read by matmul), tmp (written by o_proj/down, read by residual).
        // `aux` holds q/k/v/gate/up/attn (each written then read by a *different*
        // buffer's dispatch, so they may share).
        let su = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        // Trunk arenas hold ONE region each; size them with arena_bytes (pow2) so
        // the offset-allocator's free run actually seats the request — a bare
        // "size + ALIGN" gets floored below the request for large `total` (the same
        // bin-allocator lesson `normed`/`logits` already apply). Without this, a
        // long prefill (now hit by generate()'s batched prefill, not just
        // forward_batch callers) overflowed `hidden` at total≈54.
        let rowh = arena_bytes(&[(total * h) as u64 * f32b]);
        // Per layer the trunk allocates a fixed set of uniform blocks, plus the
        // per-seq attention loop allocates a few PER SEQUENCE — so the count
        // scales with nseq, not just layers. Plus the gather_last (one per seq)
        // and the final norm/lm_head. Sized generously; uniforms are tiny.
        let nseq_u = batch.seqs.len() as u64;
        // Tail includes one embed uniform PER INPUT ROW (`total` — for prefill
        // that's the whole prompt, not nseq); a decode-only count (total==nseq)
        // misses it and a long prefill would overflow the uniform arena. When
        // `all_positions` (spec-decode), the tail emits per-ROW uniforms: gather_last
        // (one per row = `total`) AND the final lm_head runs over `total` rows, which
        // the batched matmul chunks into ceil(total/MAXM) dispatches (one cdims each).
        // Reserve `2*total` for the all-positions tail vs `nseq` for last-only.
        let tail_u = if all_positions {
            2 * total as u64
        } else {
            nseq_u
        };
        // int8/Q4 batched matmuls CHUNK m into MAXM-row pieces, each consuming one
        // `cdims` uniform. There are 7 such matmuls per layer (q/k/v/o/gate/up/down)
        // plus the lm_head, so a long prefill (m = total > MAXM) needs
        // ceil(total/MAXM) extra uniforms each — not covered by the fixed per-layer
        // 16. (bf16 m>1 uses the tiled GEMM, which allocates one dims block, so this
        // over-estimates harmlessly there.) Without it, m>16 prefill overflowed.
        let chunks = (total as u64).div_ceil(MATMUL_VEC_BATCH_MAXM as u64);
        let matmul_chunk_u = chunks * (7 * mc.num_layers as u64 + 1);
        // MoE batched path (one block over all B rows, NOT a per-row loop): each
        // layer allocates a small FIXED set of uniforms — the router GEMV (chunked
        // ceil(B/MAXM) for the Q4* path) + route(1) + gate(1) + up(1) + swiglu(1) +
        // down(1), plus a shared-expert tail (gate/up/swiglu/down = 4) when shared>0.
        // Independent of B (no per-row loop), so this is per-layer constant + the
        // router chunks. Dense models add nothing (the per-layer 16 covers their MLP).
        let moe_u = match &mc.mlp {
            arf_core::config::MlpKind::Moe { shared_experts, .. } => {
                let per_layer = chunks + 5 + if *shared_experts > 0 { 4 } else { 0 };
                mc.num_layers as u64 * per_layer
            }
            arf_core::config::MlpKind::Dense => 0,
        };
        // MoE SAFETY MARGIN: some MoE paths (e.g. the per-routed-row copy-out, batch.rs:1674)
        // allocate uniforms PER routed row (total × top_k), which the per-layer-constant moe_u
        // under-counts → "GPU arena out of space" at conc>1 on qwen3moe-30b. These uniforms are
        // tiny (16-32 B); over-reserving is cheap. Reserve total × top_k × layers extra for MoE.
        let moe_rows_u = match &mc.mlp {
            arf_core::config::MlpKind::Moe { top_k, .. } => {
                (total as u64) * (*top_k as u64) * mc.num_layers as u64 + 1024
            }
            arf_core::config::MlpKind::Dense => 0,
        };
        let uni_count = (16 + 8 * nseq_u) * mc.num_layers as u64
            + total as u64
            + tail_u
            + matmul_chunk_u
            + moe_u
            + moe_rows_u
            + 8;

        // Borrow the reused arena cache and grow-or-reuse each trunk arena for
        // this call's `total`. After the largest call these are all pure resets —
        // no per-step device allocations. Each transient keeps its OWN buffer
        // (whole-buffer bindings, no read+read_write aliasing), exactly as before.
        let mut arenas = self.batch_arenas.borrow_mut();
        let BatchArenas {
            hidden: s_hidden,
            normed: s_normed,
            tmp: s_tmp,
            attn: s_attn,
            q: s_q,
            k: s_k,
            v: s_v,
            gate: s_gate,
            up: s_up,
            pos: s_pos,
            uni: s_uni,
            logits: s_logits,
            moe_silu: s_moe_silu,
            moe_gu: s_moe_gu,
            seq: s_seq,
            attn_meta: s_attn_meta,
        } = &mut *arenas;
        let ctx = &self.ctx;
        // Sized for the LARGEST layer (Gemma-4 global head_dim 512) so any layer's
        // per-layer q/k/v write fits; base == max for non-gemma4 models.
        let q_bytes = arena_bytes(&[(total * max_q_dim) as u64 * f32b]);
        let kv_bytes = arena_bytes(&[(total * max_kv_dim) as u64 * f32b]);
        let inter_bytes = arena_bytes(&[(total * inter) as u64 * f32b]);
        let a_hidden = BatchArenas::ensure(s_hidden, ctx, "fb-hidden", rowh, su);
        // Holds `normed` (total×inter) plus `normed_last` (nseq×h).
        // `a_normed` holds TWO simultaneously-live regions: `normed` and
        // `normed_last` (nseq×h, never freed before normed). `normed` itself is
        // sized total×inter, not total×h — it doubles as the swiglu output (read
        // by the down projection), and a region's declared size is its binding
        // length, so an h-sized region would truncate swiglu and zero down's tail.
        // arena_bytes (pow2) seats both regions; a bare sum strands the second in
        // the bin allocator (the decode-kv lesson — bites at large batch on real
        // dims, not the tiny test config).
        //
        // L28 FIX: a_normed also holds `normed_last` (out_rows×h), allocated AFTER
        // `normed` and live at the same time (final_norm reads `last` and writes
        // normed_last). The old list omitted it, so in ALL-LOGITS mode — where
        // out_rows == total, i.e. `forward_batch_all`, the path `perplexity` and
        // `dump-logits` use — the arena was short by total×h bytes and panicked
        // "GPU arena out of space" for any corpus past ~200 tokens on the 30B
        // (h=2048 > inter=768). Decode/prefill were unaffected because there
        // out_rows is 1 per sequence, which the pow2 slack absorbed.
        let a_normed = BatchArenas::ensure(
            s_normed,
            ctx,
            "fb-normed",
            arena_bytes(&[
                (total * inter) as u64 * f32b,
                (total * h) as u64 * f32b,
                // out_rows: `total` when all_positions, else nseq (one per sequence).
                (if all_positions {
                    total
                } else {
                    batch.seqs.len()
                }) as u64
                    * h as u64
                    * f32b,
            ]),
            su,
        );
        let a_tmp = BatchArenas::ensure(s_tmp, ctx, "fb-tmp", rowh, su);
        let a_attn = BatchArenas::ensure(s_attn, ctx, "fb-attn", q_bytes, su);
        let a_q = BatchArenas::ensure(s_q, ctx, "fb-q", q_bytes, su);
        let a_k = BatchArenas::ensure(s_k, ctx, "fb-k", kv_bytes, su);
        let a_v = BatchArenas::ensure(s_v, ctx, "fb-v", kv_bytes, su);
        // a_gate holds the per-layer MLP gate scratch (total×inter) AND, after the
        // layer loop frees `gate`, the gathered last-rows `last` (out_rows×h ≤
        // total×h). For MoE/Gemma dims where h > inter (e.g. 30B h=2560, inter=768),
        // total×h exceeds inter_bytes, so size for the max — else `last` overflows
        // the arena at batch>1 (the tiny test config has h==inter so it never bit).
        let gate_bytes = arena_bytes(&[(total * inter) as u64 * f32b, (total * h) as u64 * f32b]);
        let a_gate = BatchArenas::ensure(s_gate, ctx, "fb-gate", gate_bytes, su);
        let a_up = BatchArenas::ensure(s_up, ctx, "fb-up", inter_bytes, su);
        let uni = BatchArenas::ensure(
            s_uni,
            ctx,
            "fb-uni",
            (uni_count + 8) * ALIGN,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let hidden = a_hidden.alloc((total * h) as u64 * f32b).expect("hidden");
        // Sized for its widest use: the rmsnorm output is total×h, the dense swiglu
        // output is total×inter — `normed` doubles as both (the dense MLP swiglu
        // writes here), so it must hold the MAX. For dense Llama/Qwen inter>h, but a
        // MoE model can have inter<h (e.g. the 30B: inter=768, h=2048), and the
        // batched MoE gate/up reads this row-major [total,h] activation — an
        // inter-sized region would clamp the binding and drop rows past (inter/h)·total
        // (a silent wrong-output bug at batch>1, masked whenever inter≥h). The
        // a_normed arena is already sized for both regions; this matches the length.
        let normed = a_normed
            .alloc((total * inter.max(h)) as u64 * f32b)
            .expect("normed");
        let tmp_h = a_tmp.alloc((total * h) as u64 * f32b).expect("tmp_h");
        // q/k/v/attn regions sized for the LARGEST layer (gemma-4 global) so each
        // per-layer dispatch's write fits; base == max for non-gemma4 models.
        let qb = a_q.alloc((total * max_q_dim) as u64 * f32b).expect("q");
        let kb = a_k.alloc((total * max_kv_dim) as u64 * f32b).expect("k");
        let vb = a_v.alloc((total * max_kv_dim) as u64 * f32b).expect("v");
        let gate = a_gate.alloc((total * inter) as u64 * f32b).expect("gate");
        let up = a_up.alloc((total * inter) as u64 * f32b).expect("up");
        let attn = a_attn
            .alloc((total * max_q_dim) as u64 * f32b)
            .expect("attn");
        // `last` (gathered last rows) reuses a_gate's space — it's free post-MLP.

        // MoE-only: TWO dedicated scratch arenas for the BATCHED MoE path (see the
        // `GpuMlp::Moe` arm below), holding ALL the per-batch MoE intermediates as
        // ×B regions so the whole batch routes & runs in one block (not a per-row
        // loop). The regions are split across arena A (`moe_gu`) and B (`moe_silu`)
        // so NO single dispatch binds the same physical buffer as both read and
        // read_write (wgpu rejects that even at disjoint offsets):
        //   A (moe_gu):   logits [B·ne], gate_all + up_all [B·top_k·mi each]
        //   B (moe_silu): ids/wts [B·top_k], silu [B·top_k·mi], shared ids/wts [B·sh]
        // route reads logits(A)/writes ids,wts(B); gate/up read ids(B)/write
        // gate_all,up_all(A); swiglu reads gate_all,up_all(A)/writes silu(B); down
        // reads silu,ids,wts(B)/writes the trunk's tmp_h. Cached in `BatchArenas`
        // (allocate once, grow with `total`). u32b sizes the u32 ids regions.
        let u32b = std::mem::size_of::<u32>() as u64;
        let moe_scratch = if let arf_core::config::MlpKind::Moe {
            num_experts,
            top_k,
            shared_experts,
            moe_intermediate,
            ..
        } = &mc.mlp
        {
            let (ne, tk, sh, mi) = (*num_experts, *top_k, *shared_experts, *moe_intermediate);
            // The routed gate/up/swiglu need B·top_k·mi; the Q4/Q4K per-row fallback
            // in add_moe_block reuses `silu` as a single-row scratch wanting
            // max(top_k·mi, inter), so keep silu ≥ inter — but B·top_k·mi dominates.
            let tkmi = (total * tk * mi).max(mc.intermediate_size);
            let logits_b = (total * ne) as u64 * f32b;
            let ids_b = (total * tk) as u64 * u32b;
            let wts_b = (total * tk) as u64 * f32b;
            let gu_b = tkmi as u64 * f32b;
            let sh_ids_b = (total * sh).max(1) as u64 * u32b;
            let sh_wts_b = (total * sh).max(1) as u64 * f32b;
            // Arena A: logits + gate_all + up_all.
            // Same FRAGMENTATION HEADROOM as arena B below (measured 2026-08-03: with
            // max_prefill_tokens=8192 this arena panicked "requested 4032 units, largest free
            // run is 3840 of 8192" — total capacity fine, contiguity not). One extra region of
            // slack forces the next power-of-two step.
            let need_a = arena_bytes(&[logits_b, gu_b, gu_b, gu_b]);
            let a_gu = BatchArenas::ensure(s_moe_gu, ctx, "fb-moe-gu", need_a, su);
            let logits = a_gu.alloc(logits_b).expect("moe logits");
            let gate_all = a_gu.alloc(gu_b).expect("moe gate_all");
            let up_all = a_gu.alloc(gu_b).expect("moe up_all");
            // Arena B: ids + wts + silu + shared ids/wts.
            // FRAGMENTATION HEADROOM (measured 2026-08-03): at conc32/64 this panicked with
            // "GPU arena out of space: requested 15936 units, largest free run is 15360 of 16384"
            // — the arena was big enough in TOTAL but not in one CONTIGUOUS run, because
            // arena_bytes() rounds the SUM to a power of two and packing is not perfect. That
            // killed the model-actor thread and took the whole daemon's concurrency with it.
            // Add one region's worth of slack so the next power-of-two step is taken.
            let need_b = arena_bytes(&[ids_b, wts_b, gu_b, sh_ids_b, sh_wts_b, gu_b]);
            let a_sc = BatchArenas::ensure(s_moe_silu, ctx, "fb-moe-scratch", need_b, su);
            let ids = a_sc.alloc(ids_b).expect("moe ids");
            let wts = a_sc.alloc(wts_b).expect("moe wts");
            let silu = a_sc.alloc(gu_b).expect("moe silu");
            // Constant shared ids/wts: per row the slot list is [0,1,..,sh-1] and the
            // weights are all 1.0 (always-on, no router weight). Written once here,
            // read every layer. Only meaningful when sh > 0.
            let sh_ids = a_sc.alloc(sh_ids_b).expect("moe shared ids");
            let sh_wts = a_sc.alloc(sh_wts_b).expect("moe shared wts");
            if sh > 0 {
                let ids_init: Vec<u32> = (0..total).flat_map(|_| 0..sh as u32).collect();
                let wts_init = vec![1.0f32; total * sh];
                a_sc.write(&sh_ids, &ids_init);
                a_sc.write(&sh_wts, &wts_init);
            }
            Some(MoeBatchScratch {
                ne,
                tk,
                mi,
                logits,
                ids,
                wts,
                gate_all,
                up_all,
                silu,
                sh_ids,
                sh_wts,
            })
        } else {
            None
        };
        // Reborrow the MoE arenas as shared (`&GpuArena`) for the MoE arm (read-only
        // via `.resource`). `None` for dense models.
        #[allow(clippy::type_complexity)]
        let (a_moe_gu, a_moe_sc): (Option<&GpuArena>, Option<&GpuArena>) = if moe_scratch.is_some()
        {
            (
                Some(s_moe_gu.as_ref().expect("moe gu arena present")),
                Some(s_moe_silu.as_ref().expect("moe scratch arena present")),
            )
        } else {
            (None, None)
        };

        // Positions live in their own tiny arena (read by rope; never written in a
        // dispatch that also reads it).
        let pidx = BatchArenas::ensure(s_pos, ctx, "fb-pos", total as u64 * 4 + 2 * ALIGN, su);
        let pos_buf = pidx.alloc_write(&batch.positions).expect("pos");

        let nseq = batch.seqs.len();
        // The non-quantized KV pool takes the BATCHED attention path (one scatter +
        // one attention dispatch over the whole ragged batch). The quantized pool
        // still loops per sequence — only it needs the per-seq scratch arenas below.
        let batched_attn = !self.kv.kv_quant.is_quantized() && total > 0;

        // Per-sequence attention scratch (QUANTIZED-KV path only): one DISTINCT
        // (idx, kv) buffer pair per sequence, reused across calls. Distinct buffers
        // (not one shared) so seq i never aliases seq j within the single command
        // pass. Grow the Vec to nseq and each pair to the high-water ctx_len.
        if !batched_attn {
            let max_ctx_len = batch
                .seqs
                .iter()
                .map(|s| s.past_len + s.q_len)
                .max()
                .unwrap_or(0);
            let seq_kv_bytes = arena_bytes(&[
                (max_ctx_len * kv_dim) as u64 * f32b,
                (max_ctx_len * kv_dim) as u64 * f32b,
            ]);
            // The per-seq idx arena holds TWO live regions: `slots` (ctx_len u32) and
            // `new_slots` (q_len u32). For a full prefill q_len == ctx_len, so it needs
            // ~2*max_ctx_len u32 — NOT `max_ctx_len + 2` (which only fit the decode case
            // q_len==1 and overflowed long prefills, capping generate()'s batched
            // prefill ~256). Size it pow2 for BOTH regions like seq_kv_bytes, so the
            // offset-allocator seats them.
            let max_q_len = batch.seqs.iter().map(|s| s.q_len).max().unwrap_or(0);
            let seq_idx_bytes = arena_bytes(&[
                (max_ctx_len as u64) * 4, // slots
                (max_q_len as u64) * 4,   // new_slots
            ]);
            while s_seq.len() < nseq {
                let idx =
                    GpuArena::new(ctx, "fb-seq-idx", seq_idx_bytes, su).expect("seq idx arena");
                let kv = GpuArena::new(ctx, "fb-seq-kv", seq_kv_bytes, su).expect("seq kv arena");
                s_seq.push((idx, kv));
            }
            // Grow existing pairs if a longer context arrived than they were sized for.
            for (idx, kv) in s_seq.iter_mut().take(nseq) {
                if idx.capacity() < seq_idx_bytes {
                    *idx =
                        GpuArena::new(ctx, "fb-seq-idx", seq_idx_bytes, su).expect("seq idx arena");
                }
                if kv.capacity() < seq_kv_bytes {
                    *kv = GpuArena::new(ctx, "fb-seq-kv", seq_kv_bytes, su).expect("seq kv arena");
                }
            }
        }

        // Batched-attention metadata (non-quantized KV pool). One dispatch each for
        // scatter and attention over the WHOLE ragged batch replaces the per-seq
        // loop below — the lever capping concurrent-decode throughput. Build it once
        // (constant across layers) and reuse the buffers for every layer:
        //   global_slots — every sequence's slot table, concatenated; row `r` reads
        //                  slots[slot_base[seq] + key] (the per-seq gather offset).
        //   row_meta      — 2 u32 per GLOBAL query row: (slot_base, last=past+row).
        //   write_slots   — per GLOBAL row, the pool slot its new K/V scatters to.
        // The query rows are laid out by ascending `q_start`, exactly matching the
        // q/k/v buffers, so iterating seqs in order yields the right global rows.
        // The quantized-KV path keeps the per-seq loop (its scatter/attention read
        // codebook levels; not yet batched).
        let attn_meta = if batched_attn {
            let mut global_slots: Vec<u32> = Vec::new();
            // 3 u32 per query row: (slot_base, last, bidir_last). `bidir_last` is the absolute
            // context key index of the END of the image span this row belongs to (Gemma-3
            // bidirectional image attention), or 0 for a normal causal/text row. The shader
            // streams keys [lo, max(last, bidir_last)] and, when bidir_last>0, forces lo=0 so
            // the local-attention window can't clip the bidirectional image block.
            let mut row_meta: Vec<u32> = Vec::with_capacity(total * 3);
            let mut write_slots: Vec<u32> = Vec::with_capacity(total);
            for s in &batch.seqs {
                let ctx_len = s.past_len + s.q_len;
                let slot_base = global_slots.len() as u32;
                global_slots.extend_from_slice(&s.slots[..ctx_len]);
                for r in 0..s.q_len {
                    // If this query row is inside an image span, extend its frontier to the
                    // span's last key so it attends the whole block bidirectionally.
                    let bidir_last = s
                        .image_spans
                        .iter()
                        .find(|(st, ln)| r >= *st && r < *st + *ln)
                        .map(|(st, ln)| (s.past_len + st + ln - 1) as u32)
                        .unwrap_or(0);
                    row_meta.push(slot_base);
                    row_meta.push((s.past_len + r) as u32);
                    row_meta.push(bidir_last);
                    write_slots.push(s.slots[s.past_len + r]);
                }
            }
            let meta_bytes = arena_bytes(&[
                global_slots.len() as u64 * 4,
                row_meta.len() as u64 * 4,
                write_slots.len() as u64 * 4,
            ]);
            let am = BatchArenas::ensure(s_attn_meta, ctx, "fb-attn-meta", meta_bytes, su);
            let gsb = am.alloc_write(&global_slots).expect("global_slots");
            let rmb = am.alloc_write(&row_meta).expect("row_meta");
            let wsb = am.alloc_write(&write_slots).expect("write_slots");
            Some((am, gsb, rmb, wsb))
        } else {
            None
        };

        let mut cp = CommandPass::new(&self.ctx);

        // embed: one dispatch per row gathers its token into hidden[row]. Slot 3
        // is the embedding scale (Gemma √hidden); forward_batch is Llama/Qwen-shaped
        // so this is 1.0 today, but read it from cfg to stay correct if that changes.
        let embed_scale_bits = self.cfg.embedding_scale.unwrap_or(1.0).to_bits();
        // Vision: precomputed image soft-tokens override the token-embed at their rows.
        // Upload them once as an f32 "table" and remember which flat rows they fill, so the
        // per-row loop copies embeds[i]→hidden[row] (scale 1.0 — already in embed space).
        let img = batch.image_embeds.as_ref();
        let (img_buf, img_row_to_src) = match img {
            Some(ie) if !ie.rows.is_empty() => {
                let buf = self.ctx.storage_init("image_embeds", &ie.embeds);
                let map: std::collections::HashMap<usize, usize> = ie
                    .rows
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, r)| (r, i))
                    .collect();
                (Some(buf), map)
            }
            _ => (None, std::collections::HashMap::new()),
        };
        for (row, &token) in input_ids.iter().enumerate() {
            if let (Some(buf), Some(&src)) = (&img_buf, img_row_to_src.get(&row)) {
                // image soft-token: copy embeds[src] → hidden[row] (scale 1.0, no √hidden).
                let ed = uni
                    .alloc_write(&[src as u32, h as u32, row as u32, 1.0f32.to_bits()])
                    .expect("ed_img");
                cp.add_bound(
                    &k.embed,
                    "embed_img",
                    &[
                        buf.as_entire_binding(),
                        a_hidden.resource(&hidden),
                        uni.resource(&ed),
                    ],
                    wg(h),
                );
                continue;
            }
            let ed = uni
                .alloc_write(&[token, h as u32, row as u32, embed_scale_bits])
                .expect("ed");
            cp.add_bound(
                // bf16 table → bf16 gather (self.embed is uploaded bf16). The other
                // `k.embed` uses below gather f32 SCRATCH (hidden/mlp_down), not the
                // token table, so they stay on the f32 kernel.
                if self.embed_q4k {
                    &k.embed_q4k
                } else {
                    &k.embed_bf16
                },
                "embed",
                &[
                    self.embed.as_entire_binding(),
                    a_hidden.resource(&hidden),
                    uni.resource(&ed),
                ],
                wg(h),
            );
        }

        // Muse Glimmer: a WEIGHTLESS pre-trunk RMSNorm on the embedding output, before layer 0.
        // llama's graph is `embd_norm = rms_norm(embd)` then the layer's own `rms_norm` on top
        // (verified against a llama-eval-callback dump); omitting it hands all 52 layers input
        // at the wrong scale. `embed_norm` is an all-ones weight so the standard kernel IS a
        // weightless norm. `normed` is the scratch — it is dead until layer 0's `in_norm`
        // overwrites it a few dispatches later, so this borrows it rather than allocating.
        if let Some(embed_norm) = self.embed_norm.as_ref() {
            let end = uni
                .alloc_write(&dims_norm_m(total, h, mc.rms_norm_eps))
                .expect("end");
            cp.add_bound(
                &k.rmsnorm,
                "embed_norm",
                &[
                    a_hidden.resource(&hidden),
                    embed_norm.as_entire_binding(),
                    a_normed.resource(&normed),
                    uni.resource(&end),
                ],
                [total as u32, 1, 1],
            );
            // Copy back so the trunk reads the normalized activation from `hidden` (which is
            // also the residual stream). `k.embed` with scale 1.0 is a plain row copy — the
            // same identity-gather `gather_last` uses.
            for row in 0..total {
                let cd = uni
                    .alloc_write(&[row as u32, h as u32, row as u32, 1.0f32.to_bits()])
                    .expect("embed_norm_cd");
                cp.add_bound(
                    &k.embed,
                    "embed_norm_copy",
                    &[
                        a_normed.resource(&normed),
                        a_hidden.resource(&hidden),
                        uni.resource(&cd),
                    ],
                    wg(h),
                );
            }
        }

        // Bind-group cache for the STEADY single-sequence decode shape (the serve chat hot
        // path): one seq, q_len==1 → total==1, so the matmul/norm dispatch sequence is
        // byte-identical every token and each cacheable site lands at a stable slot. Building
        // those bind groups uncached every token was the ~100× chat slowdown. Any other shape
        // (prefill q_len>1, multi-seq batch, batch resize) is NOT cached: pass cursor=None so
        // those sites build fresh, and clear any stale cache so the next steady run repopulates.
        // Validity mirrors decode.rs: only sites binding resident weights + fixed arena/uniform
        // offsets are cached (threaded below via add_maybe_cached); the ctx_len-growing
        // attention / kv-scatter sites always build fresh.
        let steady = batch.seqs.len() == 1 && total == 1;
        let shape = (
            batch.seqs.len(),
            batch.seqs.first().map(|s| s.q_len).unwrap_or(0),
            total,
        );
        if !steady || self.batch_binds_shape.get() != Some(shape) {
            self.batch_binds.borrow_mut().clear();
            self.batch_binds_shape
                .set(if steady { Some(shape) } else { None });
        }
        let mut binds_guard = self.batch_binds.borrow_mut();
        let mut cursor = if steady {
            Some(crate::gpu::command::BindCursor::new(&mut binds_guard))
        } else {
            None
        };
        let _ = &mut cursor; // used by add_maybe_cached at the cacheable sites below

        // 8-BIT KV (`kv_q8_for`): the f32 pool this generic loop binds is a 16-byte PLACEHOLDER —
        // the cache lives in `kv.q8`, which only the island record reads. Reaching here means an
        // island path bailed; attending the placeholder is fluent garbage or a hung queue
        // (2026-09-23: an Err raised mid-record by the shared-prefix verify hung a request for
        // 900 s). Say why, loudly, instead.
        #[cfg(target_os = "macos")]
        if self.kv.q8.iter().any(Option::is_some) {
            panic!(
                "[kv] 8-bit KV cache is live but this step fell through to the generic f32 layer \
                 loop, which cannot read it (b={}, q_lens={:?}). Rerun with \
                 ARF_BATCH_MEGA_DEBUG=1 for the bail reason; ARF_NO_KV_Q8=1 serves f32.",
                batch.seqs.len(),
                batch.seqs.iter().map(|s| s.q_len).collect::<Vec<_>>()
            );
        }
        for (li, layer) in self.layers.iter().enumerate() {
            // PER-LAYER attention geometry (Gemma-4 global layers differ from the
            // sliding base) — mirrors decode.rs. For every other model these equal the
            // base cfg scalars, so non-gemma4 paths are byte-identical. Using the base
            // `hd`/`nkv`/`q_dim`/`kv_dim`/`scale`/`group` for ALL layers was the
            // batched-gemma4 corruption bug (global layers ran with wrong geometry).
            let hd_l = layer.head_dim;
            let nkv_l = layer.kv_heads;
            let q_dim_l = layer.q_dim(nh);
            let kv_dim_l = layer.kv_dim();
            let group_l = nh / nkv_l;
            let scale_l = mc
                .query_pre_attn_scalar
                .unwrap_or(1.0 / (hd_l as f32).sqrt());
            // normed = rmsnorm(hidden) over all rows.
            let nd = uni
                .alloc_write(&dims_norm_m(total, h, mc.rms_norm_eps))
                .expect("nd");
            cp.add_bound(
                &k.rmsnorm,
                "in_norm",
                &[
                    a_hidden.resource(&hidden),
                    layer.input_norm.as_entire_binding(),
                    a_normed.resource(&normed),
                    uni.resource(&nd),
                ],
                [total as u32, 1, 1],
            );

            // ARF_DUMP_BATCH=<layer>: dump this layer's INPUT (hidden) and post-input-norm
            // (normed) so they can be diffed against llama's tensors. decode.rs has the same
            // hooks, but serve requests go through THIS batched path, so those never fired —
            // which is why six muse-glimmer hypotheses were tested blind.
            if std::env::var("ARF_DUMP_BATCH").ok().as_deref() == Some(&li.to_string()) {
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                // L363l — ALL rows of this layer's input, not row 0's first 64 floats: the
                // per-row reference for a prefill-as-rows step (row r of the batched record's
                // capture is position r here). One line per row, the first 8 floats.
                let hv = self
                    .ctx
                    .read_f32_at(a_hidden.buffer(), hidden.offset(), total * h);
                let nv = self
                    .ctx
                    .read_f32_at(a_normed.buffer(), normed.offset(), h.min(64));
                eprintln!(
                    "[dump L{li}] rows={total} hidden[..8] = {:?}",
                    &hv[..8.min(hv.len())]
                );
                eprintln!("[dump L{li}] normed[..8] = {:?}", &nv[..8.min(nv.len())]);
                std::fs::write(
                    format!("/tmp/ours_batch_hidden_L{li}.txt"),
                    hv.chunks(h)
                        .map(|row| {
                            row.iter()
                                .take(8)
                                .map(|v| format!("{v:.6}"))
                                .collect::<Vec<_>>()
                                .join(" ")
                                + "\n"
                        })
                        .collect::<String>(),
                )
                .ok();
                std::fs::write(
                    format!("/tmp/ours_batch_normed_L{li}.txt"),
                    nv.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                )
                .ok();
            }

            // q/k/v = normed · Wᵀ  (m = total)
            add_matmul_m(
                &mut cp,
                k,
                &mut *uni,
                total,
                a_normed.resource(&normed),
                &layer.q_proj,
                a_q.resource(&qb),
                q_dim_l,
                h,
                cursor.as_mut(),
            );
            add_matmul_m(
                &mut cp,
                k,
                &mut *uni,
                total,
                a_normed.resource(&normed),
                &layer.k_proj,
                a_k.resource(&kb),
                kv_dim_l,
                h,
                cursor.as_mut(),
            );
            add_matmul_m(
                &mut cp,
                k,
                &mut *uni,
                total,
                a_normed.resource(&normed),
                &layer.v_proj,
                a_v.resource(&vb),
                kv_dim_l,
                h,
                cursor.as_mut(),
            );

            // Per-head q/k RMSNorm before RoPE (Qwen3/Gemma); Llama skips it.
            // q_rows = total*nh, k_rows = total*nkv (one row per (token, head)).
            if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
                let q_rows = total * nh;
                let k_rows = total * nkv_l;
                let qkd = uni
                    .alloc_write(&[
                        q_rows as u32,
                        k_rows as u32,
                        hd_l as u32,
                        (mc.rms_norm_eps as f32).to_bits(),
                    ])
                    .expect("qkd");
                cp.add_bound(
                    &k.qk_norm,
                    "qk_norm",
                    &[
                        a_q.resource(&qb),
                        a_k.resource(&kb),
                        qn.as_entire_binding(),
                        kn.as_entire_binding(),
                        uni.resource(&qkd),
                    ],
                    [(q_rows + k_rows) as u32, 1, 1],
                );
            }

            // Gemma-4 weightless per-head V-norm (RMS_NORM of Vcur, no gain, no RoPE),
            // after v_proj, before attention — mirrors decode.rs. MISSING from the
            // batched path was part of the gemma-4 corruption. Normalizes all
            // total·nkv rows of head_dim in the V buffer.
            if mc.value_norm {
                let vn_rows = total * nkv_l;
                let vnd = uni
                    .alloc_write(&[
                        vn_rows as u32,
                        hd_l as u32,
                        (mc.rms_norm_eps as f32).to_bits(),
                        0,
                    ])
                    .expect("vnd_b");
                cp.add_bound(
                    &k.v_norm,
                    "v_norm",
                    &[a_v.resource(&vb), uni.resource(&vnd)],
                    [vn_rows as u32, 1, 1],
                );
            }

            // Fused RoPE q,k in place, per-row positions, one dispatch. Table per
            // layer (mirrors decode.rs): sliding/local layers rotate with the local
            // base; Gemma-4 global layers use the dedicated partial-rotary global
            // table; everything else uses the default table. `rotary_dim` gates
            // partial rotation (= head_dim for full rotary on every non-gemma4-global
            // layer, so this is a no-op there). The Dims uniform is 8 u32 = 32 B to
            // match `rope_qk.wgsl` {tokens,q_heads,k_heads,head_dim,rotary_dim,_×3}.
            let (rope_cos, rope_sin) = match (
                layer.window,
                &self.rope_cos_local,
                &self.rope_sin_local,
                &self.rope_cos_global,
                &self.rope_sin_global,
            ) {
                (Some(_), Some(cl), Some(sl), _, _) => (cl, sl),
                (None, _, _, Some(cg), Some(sg)) => (cg, sg),
                _ => (&self.rope_cos, &self.rope_sin),
            };
            let rd = uni
                .alloc_write(&[
                    total as u32,
                    nh as u32,
                    nkv_l as u32,
                    hd_l as u32,
                    layer.rotary_dim as u32,
                    0,
                    0,
                    0,
                ])
                .expect("rd");
            // ROPE PAIRING IS PER-ARCH (same rule as decode.rs). muse-glimmer's GGUF stores
            // q/k INTERLEAVED (ggml NORM); everything else we ship is split-half (NEOX). This
            // is the BATCHED path — the one the server actually uses — so the decode.rs fix
            // alone never reached a real request.
            let (rope_pso_b, rope_label_b) = if mc.rope_interleaved() {
                (&k.rope_qk_interleaved, "rope_qk_interleaved")
            } else {
                (&k.rope_qk, "rope_qk")
            };
            cp.add_bound(
                rope_pso_b,
                rope_label_b,
                &[
                    a_q.resource(&qb),
                    a_k.resource(&kb),
                    rope_cos.as_entire_binding(),
                    rope_sin.as_entire_binding(),
                    pidx.resource(&pos_buf),
                    uni.resource(&rd),
                ],
                wg(total * q_dim_l + total * kv_dim_l),
            );

            // KV scatter + attention. Non-quantized pool: ONE batched dispatch each
            // over the whole ragged batch (FlashInfer-style in-kernel gather), keyed
            // by the per-row metadata built above — replaces the per-sequence loop
            // that capped concurrent-decode throughput. Quantized pool falls through
            // to the per-seq loop (its kernels read codebook levels; not yet batched).
            if let Some((am, gsb, rmb, wsb)) = &attn_meta {
                // Scatter every row's new K/V into the pool in one dispatch:
                // write_slots[r] is row r's destination slot. Reuses the single-stream
                // kv_scatter with src_row 0 over all `total` rows.
                let sd = uni
                    .alloc_write(&[total as u32, kv_dim_l as u32, 0u32, 0u32])
                    .expect("sd_b");
                cp.add_bound(
                    &k.kv_scatter,
                    "kv_scatter_batched",
                    &[
                        a_k.resource(&kb),
                        a_v.resource(&vb),
                        am.resource(wsb),
                        self.kv.keys[li].as_entire_binding(),
                        self.kv.values[li].as_entire_binding(),
                        uni.resource(&sd),
                    ],
                    wg(total * kv_dim_l),
                );
                // Attention over every (head, global row) in one [nh, total, 1] grid.
                // Gemma local layers mask to the last `window` keys; 0 = global/causal.
                let ad = uni
                    .alloc_write(&[
                        nh as u32,
                        nkv_l as u32,
                        hd_l as u32,
                        scale_l.to_bits(),
                        group_l as u32,
                        layer.window.unwrap_or(0) as u32,
                        0u32,
                        0u32,
                    ])
                    .expect("ad_b");
                // GQA-grouped when several query heads share a KV head: one
                // workgroup per (kv_head, row) reads each K/V slot once and fans it
                // out across the group (the decode-bandwidth lever). Falls back to
                // the per-(head,row) kernel for MHA (group==1) or out-of-budget dims.
                let binds = [
                    a_q.resource(&qb),
                    self.kv.keys[li].as_entire_binding(),
                    self.kv.values[li].as_entire_binding(),
                    a_attn.resource(&attn),
                    am.resource(gsb),
                    am.resource(rmb),
                    uni.resource(&ad),
                ];
                if use_gqa {
                    cp.add_bound(
                        &k.attention_gqa,
                        "attention_gqa",
                        &binds,
                        [nkv as u32, total as u32, 1],
                    );
                } else {
                    cp.add_bound(
                        &k.attention_batched,
                        "attention_batched",
                        &binds,
                        [nh as u32, total as u32, 1],
                    );
                }
            } else {
                for (si, s) in batch.seqs.iter().enumerate() {
                    let ctx_len = s.past_len + s.q_len;
                    // This sequence's own idx arena from the cache — distinct per seq,
                    // reset and reused. (The per-seq kv arena `_skv` is no longer used:
                    // attention reads the paged pool directly by slot, like the decode
                    // path. Left allocated to keep the cache's tuple shape unchanged.)
                    let (sidx, _skv) = &mut s_seq[si];
                    sidx.reset();
                    let slots_buf = sidx.alloc_write(&s.slots).expect("slots");
                    let new_slots: Vec<u32> = s.slots[s.past_len..].to_vec();
                    let new_slot_buf = sidx.alloc_write(&new_slots).expect("new_slots");

                    // scatter new k/v into the pool; attention reads the pool by slot
                    // (read-after-write across distinct dispatches — wgpu-legal).
                    let mut adims = dims_attn_q(
                        s.q_start,
                        s.q_len,
                        ctx_len,
                        nh,
                        nkv_l,
                        hd_l,
                        s.past_len,
                        scale_l,
                        group_l,
                        layer.window.unwrap_or(0),
                    );
                    if self.kv.kv_quant.is_quantized() {
                        let bits = self.kv.kv_quant.bits() as u32;
                        let sd = uni
                            .alloc_write(&[
                                s.q_len as u32,
                                nkv_l as u32,
                                hd_l as u32,
                                s.q_start as u32,
                                bits,
                                0,
                                0,
                                0,
                            ])
                            .expect("sd_tq");
                        cp.add_bound(
                            &k.kv_scatter_tq,
                            "kv_scatter_tq",
                            &[
                                a_k.resource(&kb),
                                a_v.resource(&vb),
                                sidx.resource(&new_slot_buf),
                                self.kv.keys[li].as_entire_binding(),
                                self.kv.values[li].as_entire_binding(),
                                self.kv.key_norms[li].as_entire_binding(),
                                self.kv.value_norms[li].as_entire_binding(),
                                self.kv_levels.as_ref().unwrap().as_entire_binding(),
                                uni.resource(&sd),
                            ],
                            [((s.q_len * nkv_l) as u32).div_ceil(64), 1, 1],
                        );
                        adims[10] = bits;
                        let ad = uni.alloc_write(&adims).expect("ad_tq");
                        cp.add_bound(
                            &k.attention_tq,
                            "attention_tq",
                            &[
                                a_q.resource(&qb),
                                self.kv.keys[li].as_entire_binding(),
                                self.kv.values[li].as_entire_binding(),
                                a_attn.resource(&attn),
                                sidx.resource(&slots_buf),
                                uni.resource(&ad),
                                self.kv.key_norms[li].as_entire_binding(),
                                self.kv.value_norms[li].as_entire_binding(),
                                self.kv_levels.as_ref().unwrap().as_entire_binding(),
                            ],
                            [nh as u32, s.q_len as u32, 1],
                        );
                    } else {
                        let sd = uni
                            .alloc_write(&[s.q_len as u32, kv_dim_l as u32, s.q_start as u32, 0])
                            .expect("sd");
                        cp.add_bound(
                            &k.kv_scatter,
                            "kv_scatter",
                            &[
                                a_k.resource(&kb),
                                a_v.resource(&vb),
                                sidx.resource(&new_slot_buf),
                                self.kv.keys[li].as_entire_binding(),
                                self.kv.values[li].as_entire_binding(),
                                uni.resource(&sd),
                            ],
                            wg(s.q_len * kv_dim_l),
                        );
                        let ad = uni.alloc_write(&adims).expect("ad");
                        cp.add_bound(
                            &k.attention,
                            "attention",
                            &[
                                a_q.resource(&qb),
                                self.kv.keys[li].as_entire_binding(),
                                self.kv.values[li].as_entire_binding(),
                                a_attn.resource(&attn),
                                sidx.resource(&slots_buf),
                                uni.resource(&ad),
                            ],
                            [nh as u32, s.q_len as u32, 1],
                        );
                    }
                }
            }

            // ARF_DUMP_BATCH=<layer>, stage 2: q/k AFTER rope+qk_norm, and the attention
            // output BEFORE o_proj. With the stage-1 hidden/normed dump this brackets the whole
            // attention sublayer, so a diff against llama localises the divergence to one op.
            if std::env::var("ARF_DUMP_BATCH").ok().as_deref() == Some(&li.to_string()) {
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let qv = self.ctx.read_f32_at(a_q.buffer(), qb.offset(), 8);
                let kv = self.ctx.read_f32_at(a_k.buffer(), kb.offset(), 8);
                let av = self.ctx.read_f32_at(a_attn.buffer(), attn.offset(), 8);
                eprintln!("[dump L{li}] q_postrope[..8] = {qv:?}");
                eprintln!("[dump L{li}] k_postrope[..8] = {kv:?}");
                eprintln!("[dump L{li}] attn_out[..8]   = {av:?}");
                // L363l — the same three, EVERY row (first 8 floats each), one line per row, so a
                // prefill-as-rows step can be diffed op by op against its row in the batched record.
                let rows8 = |v: &[f32], pitch: usize| -> String {
                    v.chunks(pitch)
                        .map(|row| {
                            row.iter()
                                .take(8)
                                .map(|x| format!("{x:.6}"))
                                .collect::<Vec<_>>()
                                .join(" ")
                                + "\n"
                        })
                        .collect()
                };
                let qa = self
                    .ctx
                    .read_f32_at(a_q.buffer(), qb.offset(), total * q_dim_l);
                let ka = self
                    .ctx
                    .read_f32_at(a_k.buffer(), kb.offset(), total * kv_dim_l);
                let aa = self
                    .ctx
                    .read_f32_at(a_attn.buffer(), attn.offset(), total * q_dim_l);
                std::fs::write(format!("/tmp/ours_batch_q_L{li}.txt"), rows8(&qa, q_dim_l)).ok();
                std::fs::write(format!("/tmp/ours_batch_k_L{li}.txt"), rows8(&ka, kv_dim_l)).ok();
                std::fs::write(
                    format!("/tmp/ours_batch_attn_L{li}.txt"),
                    rows8(&aa, q_dim_l),
                )
                .ok();
            }

            // MUSE GLIMMER ATTENTION GATE — `attn *= sigmoid(attn_gate @ normed)`, between SDPA
            // and o_proj (llama.cpp src/models/muse-glimmer.cpp:107,137-139). The twin of the
            // decode.rs site; THIS is the one that serves, so without it every server reply and
            // every batch-prefilled prompt ran the model with its gate missing — fluent output
            // that drifts off the question, which is exactly how it presented.
            //
            // The gate projects the PRE-attention hidden state (`normed`, what q/k/v saw), not
            // the attention output. `None` for every other arch, so this is a no-op there.
            //
            // Reuses the `gate` region: it is sized `total * inter` (19968) for the FFN, which
            // dwarfs the `total * q_dim` (4096) needed here, and the FFN does not write it until
            // later in this same layer.
            if let Some(gate_w) = layer.attn_gate.as_ref() {
                add_matmul_m(
                    &mut cp,
                    k,
                    &mut *uni,
                    total,
                    a_normed.resource(&normed),
                    gate_w,
                    a_gate.resource(&gate),
                    q_dim_l,
                    h,
                    cursor.as_mut(),
                );
                let gd = uni
                    .alloc_write(&[(total * q_dim_l) as u32, 0, 0, 0])
                    .expect("attn_gate dims");
                cp.add_bound(
                    &k.attn_gate_mul,
                    "attn_gate_mul",
                    &[
                        a_attn.resource(&attn),
                        a_gate.resource(&gate),
                        uni.resource(&gd),
                    ],
                    wg(total * q_dim_l),
                );
            }

            // o_proj(attn) + residual into hidden. Contraction dim is the PER-LAYER
            // q_dim (Gemma-4 global layers = 16·512 = 8192, not the base 4096) — using
            // the base q_dim read only the first 4096 of the global attention output
            // (the twin of the attention_batched DPL bug: DPL writes all 512 dims/head,
            // this reads all 8192).
            add_matmul_m(
                &mut cp,
                k,
                &mut *uni,
                total,
                a_attn.resource(&attn),
                &layer.o_proj,
                a_tmp.resource(&tmp_h),
                h,
                q_dim_l,
                cursor.as_mut(),
            );
            // Post-attention norm. Gemma's four-norm block normalizes the attention
            // output (post_attention_layernorm) BEFORE the residual add, then takes a
            // separate pre-FFN norm — unlike Llama/Qwen's fused residual-add + pre-MLP
            // norm. Without this branch the serving path corrupts gemma logits.
            let pnd = uni
                .alloc_write(&dims_norm_m(total, h, mc.rms_norm_eps))
                .expect("pnd");
            if is_gemma {
                //   hidden += rmsnorm(attn_out=tmp_h, post_attention_layernorm)
                cp.add_bound(
                    &k.rmsnorm_add,
                    "post_attn_norm",
                    &[
                        a_tmp.resource(&tmp_h),
                        layer.post_norm.as_entire_binding(),
                        a_hidden.resource(&hidden),
                        uni.resource(&pnd),
                    ],
                    [total as u32, 1, 1],
                );
                //   normed = rmsnorm(hidden, pre_feedforward_layernorm)
                let pre = layer.pre_ffn_norm.as_ref().expect("gemma pre_ffn_norm");
                cp.add_bound(
                    &k.rmsnorm,
                    "pre_ffn_norm",
                    &[
                        a_hidden.resource(&hidden),
                        pre.as_entire_binding(),
                        a_normed.resource(&normed),
                        uni.resource(&pnd),
                    ],
                    [total as u32, 1, 1],
                );
            } else {
                cp.add_bound(
                    &k.add_norm,
                    "post_norm",
                    &[
                        a_hidden.resource(&hidden),
                        a_tmp.resource(&tmp_h),
                        layer.post_norm.as_entire_binding(),
                        a_normed.resource(&normed),
                        uni.resource(&pnd),
                    ],
                    [total as u32, 1, 1],
                );
            }
            match &layer.mlp {
                GpuMlp::Dense(dense) => {
                    let GpuDenseMlp {
                        gate_proj,
                        up_proj,
                        down_proj,
                        ..
                    } = dense.as_ref();
                    add_matmul_m(
                        &mut cp,
                        k,
                        &mut *uni,
                        total,
                        a_normed.resource(&normed),
                        gate_proj,
                        a_gate.resource(&gate),
                        inter,
                        h,
                        cursor.as_mut(),
                    );
                    add_matmul_m(
                        &mut cp,
                        k,
                        &mut *uni,
                        total,
                        a_normed.resource(&normed),
                        up_proj,
                        a_up.resource(&up),
                        inter,
                        h,
                        cursor.as_mut(),
                    );
                    let swd = uni
                        .alloc_write(&[(total * inter) as u32, 0, 0, 0])
                        .expect("swd");
                    // swiglu in-place on an input is forbidden (read+write); reuse the
                    // `normed` region (act arena) as the output — free here (its matmul
                    // inputs are done). Activation by gate_act (GeGLU/gelu-tanh for
                    // Gemma, SwiGLU/SiLU for Llama/Qwen) — mirrors decode.rs. Hardcoding
                    // k.swiglu ran Gemma's FFN with the WRONG activation, the
                    // forward_batch-vs-decode_token gemma divergence.
                    let act = match self.cfg.gate_act {
                        arf_core::config::GateAct::GeluTanh => &k.geglu,
                        arf_core::config::GateAct::Silu => &k.swiglu,
                    };
                    cp.add_bound(
                        act,
                        "swiglu",
                        &[
                            a_gate.resource(&gate),
                            a_up.resource(&up),
                            a_normed.resource(&normed),
                            uni.resource(&swd),
                        ],
                        wg(total * inter),
                    );
                    add_matmul_m(
                        &mut cp,
                        k,
                        &mut *uni,
                        total,
                        a_normed.resource(&normed),
                        down_proj,
                        a_tmp.resource(&tmp_h),
                        h,
                        inter,
                        cursor.as_mut(),
                    );
                }
                GpuMlp::Moe(moe) => {
                    // BATCHED MoE over all `total` rows in ONE block (no per-row
                    // loop): the dense trunk above already batched, and the per-row
                    // MoE loop left the expert weights un-amortized. This runs the
                    // whole batch's routing + experts in one set of B-dimension
                    // dispatches so the GPU stays occupancy-filled (B× more output
                    // elements per dispatch). Per (row,output) the math is BYTE-
                    // IDENTICAL to the single-token kernels (same per-expert
                    // dequant+dot, same ascending-slot reduce, same per-row routing),
                    // so batched == single-stream bit-exact.
                    //
                    // bf16 + Q4_K_S have B-dimension kernels (the batch-tested + 30B
                    // paths); Q4/Q4_K fall back to the proven per-row `add_moe_block`
                    // loop below (correct, just un-batched — those formats have no
                    // batched-generate parity test).
                    let ms = moe_scratch
                        .as_ref()
                        .expect("moe scratch present for MoE model");
                    let a_gu = a_moe_gu.expect("moe gu arena present for MoE model");
                    let a_sc = a_moe_sc.expect("moe scratch arena present for MoE model");
                    let batched_ok = matches!(
                        &moe.gate,
                        PackedExperts::Bf16(_) | PackedExperts::Q4KS { .. }
                    );
                    if batched_ok {
                        self.add_moe_block_batched(
                            &mut cp,
                            k,
                            &mut *uni,
                            a_gu,
                            a_sc,
                            ms,
                            moe,
                            total,
                            h,
                            a_normed.resource(&normed),
                            a_tmp.resource(&tmp_h),
                        );
                    } else {
                        // Per-row fallback (Q4/Q4K): copy normed[r] → scratch.normed,
                        // run the single-token block, copy scratch.mlp_down → tmp_h[r].
                        let silu = &ms.silu; // reused as the single-row swiglu scratch
                        for r in 0..total {
                            let cin = uni
                                .alloc_write(&[r as u32, h as u32, 0u32, 1.0f32.to_bits()])
                                .expect("moe row copy-in dims");
                            cp.add_bound(
                                &k.embed,
                                "moe_row_in",
                                &[
                                    a_normed.resource(&normed),
                                    self.scratch.normed.as_entire_binding(),
                                    uni.resource(&cin),
                                ],
                                wg(h),
                            );
                            self.add_moe_block(&mut cp, k, &mut *uni, a_sc, silu, moe, h, wg, None);
                            let cout = uni
                                .alloc_write(&[0u32, h as u32, r as u32, 1.0f32.to_bits()])
                                .expect("moe row copy-out dims");
                            cp.add_bound(
                                &k.embed,
                                "moe_row_out",
                                &[
                                    self.scratch.mlp_down.as_entire_binding(),
                                    a_tmp.resource(&tmp_h),
                                    uni.resource(&cout),
                                ],
                                wg(h),
                            );
                        }
                    }
                }
            }
            if is_gemma {
                // Gemma post-FFN norm on the MLP output (tmp_h) before the residual:
                //   hidden += rmsnorm(mlp_down, post_feedforward_layernorm)
                let post = layer.post_ffn_norm.as_ref().expect("gemma post_ffn_norm");
                let fnd = uni
                    .alloc_write(&dims_norm_m(total, h, mc.rms_norm_eps))
                    .expect("fnd");
                cp.add_bound(
                    &k.rmsnorm_add,
                    "post_ffn_norm",
                    &[
                        a_tmp.resource(&tmp_h),
                        post.as_entire_binding(),
                        a_hidden.resource(&hidden),
                        uni.resource(&fnd),
                    ],
                    [total as u32, 1, 1],
                );
            } else {
                add_residual_r(
                    &mut cp,
                    k,
                    &mut *uni,
                    a_hidden.resource(&hidden),
                    a_tmp.resource(&tmp_h),
                    total * h,
                );
            }

            // Gemma-4 learned per-layer output scale (hidden *= layer_scalar), at the
            // very end of the layer — mirrors decode.rs. Was MISSING from the batched
            // path (one of the gemma-4 corruption bugs). Scales all total·h elements.
            if let Some(scale) = layer.layer_scalar {
                let lsd = uni
                    .alloc_write(&[(total * h) as u32, scale.to_bits(), 0, 0])
                    .expect("layer_scalar_b");
                cp.add_bound(
                    &k.scale_inplace,
                    "layer_scalar",
                    &[a_hidden.resource(&hidden), uni.resource(&lsd)],
                    wg(total * h),
                );
            }

            // Submit-SPLIT: flush the command buffer every FLUSH_LAYERS layers instead of
            // packing all ~15·num_layers dispatches into ONE monolithic Metal submit. A single
            // giant submit blocks indefinitely in IOGPU submitCommandBuffers (the machine-burning
            // chat hang — reproduced on gemma-4-12B even at 2 prompt tokens: 48 layers × ~15 =
            // ~720 dispatches in one buffer). llama.cpp pipelines per-layer for the same reason.
            // SAFE here: the flush lands at the LAYER BOUNDARY (every dispatch of this layer is
            // queued, all writes done); every buffer is a resident arena/weight owned by the model
            // (per-layer uniforms live at DISTINCT offsets — `uni` is sized ×num_layers), so the
            // committed GPU work is visible to the next band and nothing is overwritten in flight.
            // Output is bit-identical to the single-submit path (pure submission re-grouping).
            // TUNABLE (ARF_FLUSH_LAYERS): each flush is a queue.submit with fixed Metal driver
            // cost (~hundreds of µs), so FLUSH_LAYERS=4 means 12 submits/token on a 48-layer model
            // — vs llama.cpp's n_cb=1-2 (whole token in 1-2 buffers). For DECODE (~15 dispatches
            // /layer) a larger value (fewer, fatter submits) cuts the per-token submission overhead
            // that dominates single-stream; too large risks the IOGPU monolithic-submit hang on
            // long prefill. Default 8 (6 submits/token) balances both; sweep via env to tune.
            let flush_layers = self.flush_layers_hint();
            if (li + 1) % flush_layers == 0 {
                cp.flush();
            }
        }

        // Gather each seq's last row, final-norm + lm_head over B rows. (`nseq`
        // was bound above for the per-seq arena setup.)
        // Rows to gather + emit logits for: each seq's last position (the throughput
        // path), or EVERY input position (spec-decode verify, single seq).
        let last_rows: Vec<u32> = if all_positions {
            (0..total as u32).collect()
        } else {
            batch
                .seqs
                .iter()
                .map(|s| (s.q_start + s.q_len - 1) as u32)
                .collect()
        };
        let out_rows = last_rows.len();
        // `gate` is per-layer MLP scratch, dead after the layer loop. Free it so
        // `last` (the gathered rows, out_rows×h) can reuse a_gate's space — for a
        // multi-sequence batch out_rows×h can exceed what +ALIGN slack alone allows.
        a_gate.free(gate);
        let last = a_gate.alloc((out_rows * h) as u64 * f32b).expect("last");
        // Gather each seq's last hidden row into `last` (reuse the embed/row-copy
        // kernel: table = hidden states in `act`, out = `last` in `aux`).
        for (i, &r) in last_rows.iter().enumerate() {
            // Plain row copy (not an embed): scale 1.0 so the gather is identity.
            let cd = uni
                .alloc_write(&[r, h as u32, i as u32, 1.0f32.to_bits()])
                .expect("cd");
            cp.add_bound(
                &k.embed,
                "gather_last",
                &[
                    a_hidden.resource(&hidden),
                    a_gate.resource(&last),
                    uni.resource(&cd),
                ],
                wg(h),
            );
        }
        // normed_last in a_normed (not aux) — final_norm reads `last`(aux) and
        // writes normed_last, so they must be in different buffers.
        let normed_last = a_normed
            .alloc((out_rows * h) as u64 * f32b)
            .expect("normed_last");
        let fnd = uni
            .alloc_write(&dims_norm_m(out_rows, h, mc.rms_norm_eps))
            .expect("fnd");
        cp.add_bound(
            &k.rmsnorm,
            "final_norm",
            &[
                a_gate.resource(&last),
                self.final_norm.as_entire_binding(),
                a_normed.resource(&normed_last),
                uni.resource(&fnd),
            ],
            [out_rows as u32, 1, 1],
        );
        // logits go in their own arena (written by lm_head, then read back).
        // arena_bytes (pow2) so the single nseq×vocab region actually seats — the
        // bin allocator floors a tight "+slack" arena's free run below the request.
        let lg = BatchArenas::ensure(
            s_logits,
            ctx,
            "fb-logits",
            arena_bytes(&[(out_rows * mc.vocab_size) as u64 * f32b]),
            su,
        );
        let logits = lg
            .alloc((out_rows * mc.vocab_size) as u64 * f32b)
            .expect("logits");
        add_matmul_m(
            &mut cp,
            k,
            &mut *uni,
            out_rows,
            a_normed.resource(&normed_last),
            &self.lm_head,
            lg.resource(&logits),
            mc.vocab_size,
            h,
            None,
        );
        // Gemma-4 final-logit soft-cap over ALL emitted rows' logits in place,
        // BEFORE the readback feeds sampling/argmax. `None` (Gemma 3 / Llama /
        // Qwen) skips the dispatch (no-op).
        if let Some(cap) = mc.final_logit_softcap {
            let n = out_rows * mc.vocab_size;
            let scd = uni
                .alloc_write(&[n as u32, cap.to_bits(), 0, 0])
                .expect("softcap dims");
            cp.add_bound(
                &k.softcap,
                "softcap",
                &[lg.resource(&logits), uni.resource(&scd)],
                [(n as u32).div_ceil(256), 1, 1],
            );
        }
        // GREEDY ON-GPU SAMPLING: instead of reading the full [out_rows × vocab] logits back to the
        // CPU (a ~600KB/token blocking transfer that stalls the GPU between tokens — the dominant
        // single-stream gap vs llama.cpp), dispatch the batched argmax (one workgroup/row) into a
        // small B-wide token buffer and read back only out_rows u32s. Bit-identical to CPU greedy.
        if greedy {
            // tok output in a DEDICATED standalone STORAGE buffer — NOT in `lg` (wgpu rejects the
            // same physical buffer bound as read `logits` + read_write `tok` even at disjoint
            // offsets; same rule that split moe_gu/moe_silu). uni is UNIFORM-only, also unusable.
            let tok_buf = self.ctx.storage_zeros_u32("greedy_tokens", out_rows.max(1));
            let sd = uni
                .alloc_write(&[mc.vocab_size as u32, out_rows as u32, 0, 0])
                .expect("sample_batched dims");
            cp.add_bound(
                &self.kernels.sample_batched,
                "sample_batched",
                &[
                    lg.resource(&logits),
                    tok_buf.as_entire_binding(),
                    uni.resource(&sd),
                ],
                [out_rows as u32, 1, 1],
            );
            let _t_g0 = std::time::Instant::now();
            let g_prof = std::env::var_os("ARF_PREFILL_PROF").is_some();
            cp.submit();
            let _t_g1 = std::time::Instant::now();
            let toks = self.ctx.read_u32(&tok_buf, out_rows);
            let _t_g2 = std::time::Instant::now();
            // mirror prefill's f32 KV → the f16 pool (gated ARF_KV_F16). read_u32 above
            // fenced the wgpu scatter, so the f32 pool is durable. No-op when the flag is off.
            #[cfg(target_os = "macos")]
            self.mirror_prefill_kv_f16(batch);
            if g_prof {
                let t3 = std::time::Instant::now();
                eprintln!("[prefill-prof] GREEDY rows={out_rows} submit={:.1}ms read_u32={:.1}ms f16mirror={:.1}ms",
                    (_t_g1-_t_g0).as_secs_f64()*1000.0,
                    (_t_g2-_t_g1).as_secs_f64()*1000.0,
                    (t3-_t_g2).as_secs_f64()*1000.0);
            }
            // Return each token as a length-1 "logits" vec holding the id as f32 — the greedy
            // step() wrapper unwraps it back to u32. (Non-greedy callers never pass greedy=true.)
            return toks.into_iter().map(|t| vec![f32::from_bits(t)]).collect();
        }

        let _t_pf = std::time::Instant::now();
        let pf_prof = std::env::var_os("ARF_PREFILL_PROF").is_some();
        cp.submit();
        let _t_sub = std::time::Instant::now();

        let flat = self
            .ctx
            .read_f32_at(lg.buffer(), logits.offset(), out_rows * mc.vocab_size);
        let _t_read = std::time::Instant::now();
        if pf_prof {
            eprintln!(
                "[prefill-prof] rows={} vocab={} submit={:.1}ms readback={:.1}ms",
                out_rows,
                mc.vocab_size,
                (_t_sub - _t_pf).as_secs_f64() * 1000.0,
                (_t_read - _t_sub).as_secs_f64() * 1000.0
            );
        }
        // mirror prefill's f32 KV → the f16 pool (gated ARF_KV_F16). read_f32_at above
        // fenced the wgpu scatter, so the f32 pool is durable. No-op when the flag is off.
        #[cfg(target_os = "macos")]
        self.mirror_prefill_kv_f16(batch);
        flat.chunks(mc.vocab_size).map(|c| c.to_vec()).collect()
    }

    /// f16-mode POOL SYNC — shared gathering. Returns (a) per-layer (f32 K, f32 V, f16 K, f16 V,
    /// kv_dim) tuples for every layer with both pools present, (b) the raw (f32 K, f32 V) pairs
    /// for purgeable toggling, and (c) whether EVERY layer is f16-decode-eligible (both pools AND
    /// hd%128==0 — the decode record's `use_kv_f16` conditions). (c) is the SAFETY condition for
    /// ever making the f32 pool volatile: if any layer decodes from f32, its pages must stay.
    #[cfg(target_os = "macos")]
    #[allow(clippy::type_complexity)]
    fn kv_f16_sync_layers(
        &self,
    ) -> (
        Vec<(
            &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
            &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
            &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
            &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
            usize,
        )>,
        Vec<(
            &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
            &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        )>,
        bool,
    ) {
        let mut tuples = Vec::with_capacity(self.layers.len());
        let mut pools = Vec::with_capacity(self.layers.len());
        let mut all_eligible = !self.layers.is_empty();
        for (li, layer) in self.layers.iter().enumerate() {
            match (
                self.kv.keys_mtl.get(li).and_then(|o| o.as_ref()),
                self.kv.values_mtl.get(li).and_then(|o| o.as_ref()),
                self.kv.keys_mtl_f16.get(li).and_then(|o| o.as_ref()),
                self.kv.values_mtl_f16.get(li).and_then(|o| o.as_ref()),
            ) {
                (Some(kf32), Some(vf32), Some(kf16), Some(vf16)) => {
                    tuples.push((&*kf32.0, &*vf32.0, &*kf16.0, &*vf16.0, layer.kv_dim()));
                    pools.push((&*kf32.0, &*vf32.0));
                    if layer.head_dim % 128 != 0 {
                        all_eligible = false;
                    }
                }
                _ => {
                    all_eligible = false;
                }
            }
        }
        (tuples, pools, all_eligible)
    }

    /// f16-mode POOL SYNC (export leg) — replaces the whole-pool prefill-mirror. In f16
    /// mode the f16 pool is the SOURCE OF TRUTH: decode scatters ONLY f16, so a whole-pool
    /// f32→f16 mirror would overwrite decoded-token f16 slots with STALE f32 on every mid-stream
    /// admission (the REQS>CONC corruption this replaces). Instead, EXPORT exactly the slots THIS
    /// batch wrote (`slots[past_len .. past_len+q_len]` per seq) f32→f16. The incremental export
    /// is ITSELF the fix for the double-pool residency regression
    /// (measured 2026-07-27): the f32 pages go cold after one
    /// touch and the OS compresses them once. Optionally (ARF_KV_F16_PURGE=1, measured-off by
    /// default — see below) also mark the f32 pool volatile.
    ///
    /// GATING: no-op unless ARF_KV_F16 is set, the island + f16 pool exist, and the batched
    /// (non-quantized) scatter ran. Default (flag off) → the f16 pool is `None`, so this returns
    /// immediately and the f32 path is byte-UNCHANGED.
    ///
    /// ORDERING: the caller MUST have fenced the wgpu prefill scatter (a readback or an explicit
    /// `poll(Wait)`) BEFORE this, so the f32 pool writes are durable before the island reads them
    /// on its separate Metal queue. The island method itself `waitUntilCompleted`s, so the f16
    /// pool is fully written before the next (decode) call reads it — and before volatile.
    #[cfg(target_os = "macos")]
    fn mirror_prefill_kv_f16(&self, batch: &arf_core::model::batch::ForwardBatch) {
        // ⚠️ ARF_NO_KV_F16 MUST be honoured here too. arf-serve SETS ARF_KV_F16 itself
        // at startup (its concurrency-conditional default), so testing only for that var's
        // PRESENCE made the opt-out unreachable: `ARF_NO_KV_F16=1` disabled the f16 attention
        // path while leaving this mirror running.
        //
        // That is expensive. Measured on muse-glimmer prefill (ARF_PREFILL_PROF), the mirror
        // is 95% OF TOTAL PREFILL TIME — 3525 ms of mirror against 182 ms of actual submit across
        // five chunks. It re-copies every slot the chunk wrote, f32 -> f16, on every chunk.
        if std::env::var_os("ARF_KV_F16").is_none() || std::env::var_os("ARF_NO_KV_F16").is_some() {
            return;
        }
        // Only export when a real KV write happened on the batched non-quantized path.
        if self.kv.kv_quant.is_quantized() {
            return;
        }
        if self.kv.keys_mtl_f16.is_empty() || self.kv.values_mtl_f16.is_empty() {
            return;
        }
        let Some(isl) = self.island.as_ref() else {
            return;
        };
        // The slots THIS batch wrote (the WGSL scatter's targets) — never the whole pool.
        let written: Vec<u32> = batch
            .seqs
            .iter()
            .flat_map(|s| {
                let lo = s.past_len.min(s.slots.len());
                let hi = (s.past_len + s.q_len).min(s.slots.len());
                s.slots[lo..hi].iter().copied()
            })
            .collect();
        if written.is_empty() {
            return;
        }
        let (tuples, pools, all_eligible) = self.kv_f16_sync_layers();
        if tuples.is_empty() {
            return;
        }
        // Fence the wgpu prefill scatter so the f32 pool is durable before the island reads it.
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let isl = isl.lock().unwrap();
        if let Err(e) = isl.convert_kv_slots(&self.ctx, &tuples, &written, true) {
            // Non-fatal: an older compile set without the slot-convert pipelines leaves the f16
            // pool as the decode scatter filled it (prefill slots zero). Warn; the f32 default
            // path is unaffected (this whole method is behind ARF_KV_F16). Do NOT go volatile.
            eprintln!("[kv-f16] prefill export skipped: {e}");
            return;
        }
        // Volatile is OPT-IN (ARF_KV_F16_PURGE=1) and MEASURED-OFF by default: on the 39GB M4
        // (conc64 PREFIX=512, 2026-07-28) marking the ~7GB f32 pool volatile made the OS reclaim
        // pages DURING the decode window — steady step wall ballooned 159→261ms while GPU-busy
        // improved, canonical 452→245. The incremental export alone already removes the pressure
        // (the f32 pages go cold once and compress once — it was the whole-pool re-touch that
        // churned). If opted in: only when decode reads f16 on EVERY layer (else a layer still
        // decodes from the f32 pool and a discarded page would corrupt it); ARF_ATTN_COALESCED
        // is a hard precondition of the f16 decode path (use_kv_f16 in the record).
        if all_eligible
            && std::env::var_os("ARF_ATTN_COALESCED").is_some()
            && std::env::var_os("ARF_KV_F16_PURGE").is_some()
        {
            isl.set_kv_f32_purgeable(&pools, true);
        }
    }

    /// f16-mode POOL SYNC (import leg) — called at the top of the wgpu `forward_batch_impl` path.
    /// Re-arms the f32 staging pool (non-volatile) and imports THIS batch's PAST slots
    /// (`slots[..past_len]` per seq) f16→f32 so the WGSL prefill attention reads exactly what the
    /// f16 truth store holds (f16-rounded — the same values the island decode reads). Needed even
    /// without purging: decode writes ONLY f16, so a mixed admission batch's decode rows have past
    /// KV that EXISTS NOWHERE in f32. The convert `waitUntilCompleted`s before the wgpu encoder
    /// runs, and the wgpu scatter overwrites the batch's CURRENT slots after import — no overlap
    /// (import touches only past slots).
    #[cfg(target_os = "macos")]
    fn import_kv_f16_for_batch(&self, batch: &arf_core::model::batch::ForwardBatch) {
        if std::env::var_os("ARF_KV_F16").is_none() {
            return;
        }
        if self.kv.kv_quant.is_quantized() {
            return;
        }
        if self.kv.keys_mtl_f16.is_empty() || self.kv.values_mtl_f16.is_empty() {
            return;
        }
        let Some(isl) = self.island.as_ref() else {
            return;
        };
        let (tuples, pools, _all_eligible) = self.kv_f16_sync_layers();
        if tuples.is_empty() {
            return;
        }
        let isl = isl.lock().unwrap();
        // Re-arm FIRST — the scatter/attention below touch the pool regardless of past slots.
        // A reclaimed pool (L8: prior state Empty) is RECOVERABLE by the import that follows:
        // it restores EVERY past slot of EVERY seq in this batch from the f16 truth store, and
        // the WGSL scatter writes this batch's current slots fresh — together that is every f32
        // row the serial path reads. Slots of seqs NOT in this batch stay garbage in f32, which
        // is fine: any future window imports its own past before reading (the same invariant).
        let _reclaimed = isl.set_kv_f32_purgeable(&pools, false);
        let past: Vec<u32> = batch
            .seqs
            .iter()
            .flat_map(|s| s.slots[..s.past_len.min(s.slots.len())].iter().copied())
            .collect();
        if past.is_empty() {
            return;
        }
        if let Err(e) = isl.convert_kv_slots(&self.ctx, &tuples, &past, false) {
            eprintln!("[kv-f16] past import skipped: {e}");
        }
    }

    /// L8 — SINGLE-STREAM f32 RE-ARM GATE, the choke point for every m=1 / WGSL single-stream
    /// decode step (called at the top of `decode_token_impl`, which all of `try_m1_megakernel`'s
    /// burst/singleq/fallback flows, the CLI generate loops, and the WGSL per-token fallback run
    /// through). Under the daemon's default env the m=1 megakernel uses the FUSED f32
    /// scatter+attention (`attention_decode_fused` preempts the m=1 f16 leg), so — contrary to
    /// the batched path — single-stream decode READS the f32 pool: it must never run against a
    /// volatile pool. No-op unless the pool is actually volatile (one Cell read under the island
    /// lock), i.e. exactly once per prefill→single-stream transition.
    ///
    /// RECLAIM RECOVERY: if the OS discarded the pages, restore the WHOLE past ([0, past_len) —
    /// single-stream is identity-slot by construction, guarded in `try_m1_megakernel` and true
    /// for the CLI loops) from the f16 truth store in one slot-list convert. The f16 store is
    /// complete for every past token: prefill rows via the L5b/serial exports, m=1 decode rows
    /// via the record's f16 mirror scatter (L8). A failed recovery convert PANICS: decoding
    /// against a discarded pool would silently emit garbage, and correctness beats liveness here.
    #[cfg(target_os = "macos")]
    pub(super) fn kv_f32_rearm_for_single(&self, past_len: usize) {
        let Some(isl) = self.island.as_ref() else {
            return;
        };
        let isl = isl.lock().unwrap();
        if !isl.kv_f32_volatile() {
            return;
        }
        let (tuples, pools, _all_eligible) = self.kv_f16_sync_layers();
        if pools.is_empty() {
            return;
        } // unreachable: volatile is only ever set with pools live
        let reclaimed = isl.set_kv_f32_purgeable(&pools, false);
        if reclaimed && past_len > 0 {
            let slots: Vec<u32> = (0..past_len as u32).collect();
            if let Err(e) = isl.convert_kv_slots(&self.ctx, &tuples, &slots, false) {
                panic!(
                    "[kv-f32-volatile] reclaim recovery failed ({e}): refusing to decode \
                        from a discarded f32 pool (ARF_NO_KV_F32_VOLATILE=1 to disable)"
                );
            }
        }
    }
}
