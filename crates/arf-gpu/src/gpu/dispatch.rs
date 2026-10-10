//! Dispatch helpers shared by the decode/batch forward paths: matmul +
//! residual recorders and the kernel `Dims`-uniform builders.
//! Split out of gpu/mod.rs; behavior unchanged.

use super::*;

/// Max contraction dim (K) for the cooperative-matrix GEMM path. Above this, m≥8
/// matmuls route to the bit-exact sub-block GEMV (`matmul_vec_*_batch`).
///
/// This was briefly 256 on 2026-06-18 — a blanket gate added to stop
/// batched Gemma-4 garbage, on the belief that coop "diverges past 32 K-tiles".
/// Both premises were wrong:
///   - The "backend bug" is FOLKLORE — `examples/coop_nt_probe` swept K=64..3840:
///     the f32 coop NT GEMM matches CPU to 6.4e-7 rel-err at K=3840 (√K reassociation
///     noise, not a magnitude error). The real Gemma garbage was the GeGLU bug (fixed
///     in batch.rs), not coop.
///   - Coop's tiled accumulation DOES reassociate the dot vs the GEMV oracle, and
///     `final_logit_softcap=30` can amplify a ~1e-6 delta into a token flip in theory.
///     But empirically it does NOT: `examples/coop_gemma_match` ran 57 diverse greedy
///     trajectories across all coop tile paths (b8 base/rb2, b16 cb4, b32 cb8) and got
///     57/57 token-identical to the GEMV oracle — 0 flips.
/// Worse, the 256 gate also disabled coop for LLAMA/QWEN (hidden 2048+ > 256), which
/// have no softcap and were never at risk — costing llama-1B ~2.4x batched throughput
/// (b32 193.8 → 466.1 tok/s with coop restored) and gemma-12B ~1.5x (b32 18.7 → 27.3).
///
/// 16384 was itself a fix for this exact failure on gemma: its down_proj (K = 15360) missed the
/// old 8192 cap, so that one matmul per layer fell to the 16-row-chunked GEMV and ate ~97% of
/// prefill — the long-prompt TTFT bug. coop_nt_probe confirms accuracy holds there (2.5e-6
/// rel-err, √K noise). ARF_COOP_MAX_K overrides (set 256 to force the GEMV for bit-exact
/// parity tests or to A/B the coop win).
///
/// ⚠️ RAISED 16384 -> 32768 (L162) because the SAME BUG came straight back. The 16384 default
/// only covered the FFN widths that existed then; Muse Glimmer's down_proj is K =
/// intermediate_size = 19968, which clears 16384 as well.
///
/// MEASURED on muse-glimmer: `ARF_MM_TRACE` shows every projection taking the COOP path at
/// m=512 EXCEPT down_proj, which is absent from the trace because it lands on the untraced GEMV
/// arm. That is ~27% of all model FLOPs (down is 1/3 of the FFN, and the FFN is 82% of the
/// weights) running per-row inside an otherwise-tiled prefill. Raising the cap measured
/// 4.3 -> 5.2 tok/s prefill (+21%) with output quality unchanged.
///
/// 32768 is picked to cover the foreseeable range rather than the current model, so the next
/// wider-FFN arch does not silently re-open it. Accuracy is not the constraint here — the
/// coop_nt_probe note below already establishes the error grows only as sqrt(K) (2.5e-6 rel-err
/// at K=15360), and this is a bandwidth/tiling gate, not a numerics one.
fn coop_max_k() -> usize {
    std::env::var("ARF_COOP_MAX_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32768)
}

/// Min batch rows `m` for the cooperative-matrix GEMM (else the sub-block GEMV). The
/// coop kernels tile m by ceil(m/16), so they're correct for any m≥1 (just half-empty
/// tiles below 16); the question is the speed crossover vs the weight-amortizing GEMV.
/// Default 6. MEASURED 2026-06-18 (median-of-2 llama-1B sweep): m=6 beats m=8 across the
/// board — conc=4 170.5 vs 163.7, conc=6 184.7 vs 168.5, conc=8 276 vs 272 — and is
/// token-IDENTICAL to the GEMV oracle on gemma + llama (verified, 6/6 prompts). m=4 is
/// too far (a 4-row tile wastes 12/16 of the mma → conc=4 regresses); 6 is the crossover.
/// ARF_COOP_MIN_M overrides for tuning.
fn coop_min_m() -> usize {
    std::env::var("ARF_COOP_MIN_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6)
}

/// Batch threshold for the Q4_K_S tiled GEMM (matmul_coop_q4ks) to replace the per-row
/// GEMV on the dense q/k/v/o/lm_head projections. = llama.cpp's `ne11_mm_min = 8`
/// (ggml-metal-ops.cpp:2059): at m >= 8 the tiled kernel's weight-reuse-across-rows wins;
/// below it the GEMV's batch-CSE is already enough and the 8x8 tile's lower occupancy loses.
/// MEASURED +12% end-to-end at conc16/32 (70.3->78.4, 76.5->85.6).
const MM_TILE_MIN_M: usize = 8;

/// Fuse the per-head q/k norm + weightless v-norm into ONE decode dispatch (Gemma 4),
/// cutting one dispatch/layer. DEFAULT-ON: measured +1.4% single-stream (25.0→25.4
/// tok/s), bit-exact vs the split path. ARF_QKV_NORM=0 falls back to qk_norm+v_norm.
pub(crate) fn qkv_norm() -> bool {
    std::env::var("ARF_QKV_NORM").ok().as_deref() != Some("0")
}

/// How many layers per command-buffer flush in single-stream decode (submit-split).
/// 0 = never flush (one submit/token) — the DEFAULT and the FASTEST. A controlled
/// 3-way sweep on gemma-4-12B (96-tok) measured flush=0 at 24.3 tok/s vs flush=1/4 at
/// ~16.9: the per-step bind-group cache (which a flush invalidates — it reopens the
/// encoder and clears `binds`) is worth far more than any digestible-submit gain at
/// single-stream decode scale (~720 dispatches/token is fine in one buffer; the
/// command-buffer-size penalty only bites at the long-context/batched scale where the
/// batched path already submit-splits). An earlier mid-step probe suggested a flush win
/// but that was variance + a partially-live cache; the clean sweep refutes it.
/// ARF_DECODE_FLUSH>0 re-enables flushing for A/B on other models/shapes.
pub(crate) fn decode_flush_layers() -> usize {
    std::env::var("ARF_DECODE_FLUSH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// The m==1 Q4_0 decode GEMV runs through the multi-column kernel (4 output columns
/// per workgroup) by DEFAULT — the occupancy lever for the medium-n FFN matmuls
/// (n=15360, 50% of the gemma-4-12B frame): measured +8.6% end-to-end tok/s
/// (22.75→24.72) and the FFN shape 46%→55% roofline, bit-exact vs the 1-col kernel.
/// Unlike the multi-row variants that washed on the 1B's tiny n=4096 q_proj, the win
/// survives on the 12B's large FFN. Set ARF_Q4_MC=0 to fall back to the 1-col path.
fn q4_mc() -> bool {
    std::env::var("ARF_Q4_MC").ok().as_deref() != Some("0")
}

/// `Dims { tokens:1, hidden, eps, _pad }` for the rmsnorm kernel.
pub(crate) fn dims_norm(hidden: usize, eps: f64) -> [u32; 4] {
    dims_norm_m(1, hidden, eps)
}

/// `Dims { tokens, hidden, eps, _pad }` for the rmsnorm kernel over `tokens` rows.
pub(crate) fn dims_norm_m(tokens: usize, hidden: usize, eps: f64) -> [u32; 4] {
    [tokens as u32, hidden as u32, (eps as f32).to_bits(), 0]
}

/// Single-sequence attention dims (q_start 0, q_len 1) — the decode_token case.
pub(crate) fn dims_attn(
    ctx: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    pos: usize,
    scale: f32,
    group: usize,
    window: usize,
) -> [u32; 12] {
    dims_attn_q(0, 1, ctx, nh, nkv, hd, pos, scale, group, window)
}

/// Attention dims: `q_len` query rows starting at `q_start` in the batched
/// q/out buffers; query row `r` attends keys `0..=past_len + r`. 12 u32s so the
/// uniform is a 16-byte multiple (q_start + 3 pad).
pub(crate) fn dims_attn_q(
    q_start: usize,
    q_len: usize,
    ctx: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    past_len: usize,
    scale: f32,
    group: usize,
    window: usize,
) -> [u32; 12] {
    [
        q_len as u32,
        ctx as u32,
        nh as u32,
        nkv as u32,
        hd as u32,
        past_len as u32,
        scale.to_bits(),
        group as u32,
        q_start as u32,
        window as u32,
        0,
        0,
    ]
}

/// Record a resident matmul `out[1,n] = a[1,k] · weightᵀ` (`weight [n,k]`). The
/// `mm` dims uniform is sub-allocated from `uni`; `a` and `out` are passed as
/// bind resources so they may themselves be arena sub-ranges.
// One operand per binding plus the command pass, kernels, and arena — they do
// not collapse into a meaningful struct.
pub(crate) fn add_matmul(
    cp: &mut CommandPass,
    k: &GpuKernels,
    uni: &mut GpuArena,
    a: wgpu::BindingResource<'_>,
    weight: &GpuMatWeight,
    out: wgpu::BindingResource<'_>,
    n: usize,
    kdim: usize,
    cache: Option<&mut BindCursor<'_>>,
) {
    add_matmul_m(cp, k, uni, 1, a, weight, out, n, kdim, cache);
}

/// Record a resident matmul `out[m,n] = a[m,k] · weightᵀ` (`weight [n,k]`) for a
/// batch of `m` rows, picking the kernel from the weight's precision and `m`:
/// - bf16, m == 1: the GEMV (full-occupancy, weight bandwidth).
/// - bf16, m  > 1: the tiled 16×16 GEMM, grid `ceil(m/16) × ceil(n/16)`.
/// - int8, m == 1: the int8 GEMV (a quarter of f32 bytes; CPU int8 is the oracle).
///
/// int8 with m > 1 is not yet implemented (batched int8 GEMM is future work);
/// the batched path uses bf16 weights, so this branch is unreachable in practice.
pub(crate) fn add_matmul_m(
    cp: &mut CommandPass,
    k: &GpuKernels,
    uni: &mut GpuArena,
    m: usize,
    a: wgpu::BindingResource<'_>,
    weight: &GpuMatWeight,
    out: wgpu::BindingResource<'_>,
    n: usize,
    kdim: usize,
    cache: Option<&mut BindCursor<'_>>,
) {
    let dims = uni
        .alloc_write(&[m as u32, kdim as u32, n as u32, 0])
        .expect("decode uniform arena overflow");
    // One workgroup per output column for the GEMVs, capped at 32768 (< wgpu's
    // 65535 per-dim limit); the kernels grid-stride to cover n > cap (e.g. the
    // lm_head's vocab of 128256).
    let gemv_groups = (n as u32).min(32768);
    // Profiling label carries the weight bytes moved (n×k × bytes-per-weight) so
    // the timestamp resolver reports effective GB/s per matmul — directly
    // comparable to the bandwidth probe's ~288 GB/s ceiling. `matmul/<bytes>`;
    // non-matmul kernels keep their plain name (no GB/s line). Only built on the
    // profiling path so the hot decode loop pays nothing for it otherwise.
    #[cfg(feature = "profiling")]
    let mm_label = format!("matmul/{}", weight.bytes_num(n, kdim));
    #[cfg(feature = "profiling")]
    let mm: &str = &mm_label;
    #[cfg(not(feature = "profiling"))]
    let mm: &str = "matmul";
    match (weight, m) {
        // Decision: a dedicated GEMV over padding the tiled GEMM — the
        // m==1 GEMM wastes 15/16 of every workgroup on zero rows (profiled
        // down_proj at ~5% of roofline). int8 narrows the weight read further.
        (GpuMatWeight::Int8 { data, scale }, 1) => {
            let kern = k.matmul_vec_q8_sg.as_ref().unwrap_or(&k.matmul_vec_q8);
            let res = [
                a,
                data.as_entire_binding(),
                out,
                uni.resource(&dims),
                scale.as_entire_binding(),
            ];
            match cache {
                Some(cur) => cp.add_bound_cached(cur, kern, mm, &res, [gemv_groups, 1, 1]),
                None => cp.add_bound(kern, mm, &res, [gemv_groups, 1, 1]),
            }
        }
        // Q4_0 (block-32 4-bit) decode GEMV — same binding shape as int8 (the
        // scale buffer holds packed bf16 block scales instead of per-row f32).
        (GpuMatWeight::Q4 { codes, scales }, 1) => {
            let res = [
                a,
                codes.as_entire_binding(),
                out,
                uni.resource(&dims),
                scales.as_entire_binding(),
            ];
            // ARF_Q4_MC=1: multi-COLUMN GEMV (4 cols/workgroup) — occupancy lever
            // for the medium-n FFN matmuls (n=15360, 50% of the gemma-4-12B frame at
            // 46% roofline). ceil(n/4) workgroups (still capped < 65535). Default-off
            // until the A/B win is confirmed; the 1-col kernel stays the default.
            if q4_mc() {
                let mc_groups = (n as u32).div_ceil(4).min(32768);
                match cache {
                    Some(cur) => {
                        cp.add_bound_cached(cur, &k.matmul_vec_q4_mc, mm, &res, [mc_groups, 1, 1])
                    }
                    None => cp.add_bound(&k.matmul_vec_q4_mc, mm, &res, [mc_groups, 1, 1]),
                }
            } else {
                match cache {
                    Some(cur) => {
                        cp.add_bound_cached(cur, &k.matmul_vec_q4, mm, &res, [gemv_groups, 1, 1])
                    }
                    None => cp.add_bound(&k.matmul_vec_q4, mm, &res, [gemv_groups, 1, 1]),
                }
            }
        }
        // Cooperative-matrix GEMM for Q4 (m ≥ 8): unpack + dequantize each weight
        // element into f32 staging, then run the dot on the matrix units — the Q4
        // analog of the bf16/int8 coop paths, the realistic serving quant. Block
        // scale folds into the staged element (per-block, not per-row), so it stays
        // the exact matmul_nt_q4 oracle modulo f32 reassociation. m < 8 keeps the
        // block-amortizing GEMV below.
        (GpuMatWeight::Q4 { codes, scales }, _)
            if m >= coop_min_m() && kdim <= coop_max_k() && k.matmul_coop_q4.is_some() =>
        {
            // m ≥ 16 → CB=4 (16×32 tile, 8 mma/barrier; gy=ceil(n/32)), else RB=2,
            // else RB=1. ARF_CB=0 forces RB=2 (per-device A/B + safety valve).
            let force_rb2 = std::env::var("ARF_CB").ok().as_deref() == Some("0");
            let cb4 = if force_rb2 {
                None
            } else if std::env::var("ARF_DK").ok().as_deref() == Some("0") {
                k.matmul_coop_q4_rb2_cb4.as_ref()
            } else {
                k.matmul_coop_q4_rb2_cb4_dk2
                    .as_ref()
                    .or(k.matmul_coop_q4_rb2_cb4.as_ref())
            };
            let (coop, gx, gy) = match (m >= 16, cb4, k.matmul_coop_q4_rb2.as_ref()) {
                (true, Some(cb4), _) => (cb4, (m as u32).div_ceil(16), (n as u32).div_ceil(32)),
                (true, None, Some(rb2)) => (rb2, (m as u32).div_ceil(16), (n as u32).div_ceil(8)),
                _ => (
                    k.matmul_coop_q4.as_ref().unwrap(),
                    (m as u32).div_ceil(8),
                    (n as u32).div_ceil(8),
                ),
            };
            cp.add_bound(
                coop,
                mm,
                &[
                    a,
                    codes.as_entire_binding(),
                    out,
                    uni.resource(&dims),
                    scales.as_entire_binding(),
                ],
                [gx, gy, 1],
            );
        }
        // Batched Q4 (m > 1): Q4 bytes + read-each-block-once amortization. Chunked
        // by MAXM like int8. Unblocks Q4 batched throughput and the spec-decode
        // verify (stream the weights once for a k+1-row window). Same binding shape
        // as the m==1 Q4 GEMV (codes + scales).
        (GpuMatWeight::Q4 { codes, scales }, _) => {
            let mut row_off = 0usize;
            while row_off < m {
                let chunk = (m - row_off).min(MATMUL_VEC_BATCH_MAXM);
                let cdims = uni
                    .alloc_write(&[chunk as u32, kdim as u32, n as u32, row_off as u32])
                    .expect("q4 batch dims");
                cp.add_bound(
                    &k.matmul_vec_q4_batch,
                    mm,
                    &[
                        a.clone(),
                        codes.as_entire_binding(),
                        out.clone(),
                        uni.resource(&cdims),
                        scales.as_entire_binding(),
                    ],
                    [gemv_groups, 1, 1],
                );
                row_off += chunk;
            }
        }
        // Q4_K-lite decode GEMV — same shape as Q4 plus a `mins` binding.
        (
            GpuMatWeight::Q4K {
                codes,
                scales,
                mins,
            },
            1,
        ) => {
            let res = [
                a,
                codes.as_entire_binding(),
                out,
                uni.resource(&dims),
                scales.as_entire_binding(),
                mins.as_entire_binding(),
            ];
            // Kernel priority: (1) ARF_MR multi-row lever (default-off, kept for
            // A/B); (2) the ggml-style simdgroup GEMV — microbench-measured +15% on
            // q_proj-class shapes (n=4096), BUT end-to-end single-stream is a WASH
            // (~198 vs ~201 tok/s llama-1B median-of-3): the isolated microbench win
            // doesn't survive the real decode frame, where the q/k/v/o GEMVs are a
            // dependent chain in one already-pipeline-filled submit (same lesson as
            // PGO + the refuted unroll/multi-row/split-K levers — see
            // arf-gemv-occupancy-attack). DEFAULT-OFF; ARF_GGML_GEMV=1 opts in
            // (kept wired + parity-correct for other shapes/devices). NSG·NR0=8
            // cols/workgroup → ceil(n/8) groups.
            let ggml = k
                .matmul_vec_q4k_ggml
                .as_ref()
                .filter(|_| std::env::var("ARF_GGML_GEMV").ok().as_deref() == Some("1"));
            if let Some(mr) = k.matmul_vec_q4k_mr.as_ref() {
                let mr_groups = gemv_groups.div_ceil(4);
                match cache {
                    Some(cur) => cp.add_bound_cached(cur, mr, mm, &res, [mr_groups, 1, 1]),
                    None => cp.add_bound(mr, mm, &res, [mr_groups, 1, 1]),
                }
            } else if let Some(g) = ggml {
                // NSG·NR0 = 2·4 = 8 output columns per workgroup.
                let g_groups = gemv_groups.div_ceil(8);
                match cache {
                    Some(cur) => cp.add_bound_cached(cur, g, mm, &res, [g_groups, 1, 1]),
                    None => cp.add_bound(g, mm, &res, [g_groups, 1, 1]),
                }
            } else {
                let kern = k.matmul_vec_q4k_sg.as_ref().unwrap_or(&k.matmul_vec_q4k);
                match cache {
                    Some(cur) => cp.add_bound_cached(cur, kern, mm, &res, [gemv_groups, 1, 1]),
                    None => cp.add_bound(kern, mm, &res, [gemv_groups, 1, 1]),
                }
            }
        }
        // Cooperative-matrix GEMM for Q4_K-lite (m ≥ 8): unpack the unsigned nibble +
        // fold the per-sub-block scale·nib+min into f32 staging, then mma. The Q4_K
        // analog of the Q4 coop path. m < 8 keeps the sub-block-amortizing GEMV below.
        (
            GpuMatWeight::Q4K {
                codes,
                scales,
                mins,
            },
            _,
        ) if m >= coop_min_m()
            && kdim <= coop_max_k()
            && k.matmul_coop_q4k.is_some()
            && std::env::var_os("ARF_NO_COOP").is_none() =>
        {
            // m ≥ 16 prefers the RB=2×CB=2 16×16-tile kernel (4 mma/barrier — lifts
            // the matrix-unit occupancy that capped batched decode at ~15% of
            // roofline); falls back to RB=2 (two stacked tiles) then RB=1. The CB=2
            // kernel covers 16 output columns per workgroup, so its y-grid is
            // ceil(n/16) vs ceil(n/8) for the others.
            // Tile width is selectable for A/B + per-device tuning: ARF_CB=4
            // (16×32, default), =2 (16×16), =0 forces the RB=2 (16×8) path. Wider
            // tiles do more mma per barrier (the occupancy lever) at the cost of
            // register pressure, so the sweet spot is device-dependent.
            let cb = std::env::var("ARF_CB").ok();
            // Back-compat: ARF_NO_CB2 still forces the RB=2 path.
            let force_rb2 = cb.as_deref() == Some("0") || std::env::var_os("ARF_NO_CB2").is_some();
            // ARF_CB=8 opts into the widest (16×64) tile; default is CB=4.
            let cb8 = if cb.as_deref() == Some("8") {
                k.matmul_coop_q4k_rb2_cb8.as_ref()
            } else {
                None
            };
            // The CB=4 tile defaults to its deeper-K twin (DK=2: 16 mma / barrier,
            // same 8 accumulators — measured +3.5% over plain CB=4 at B=32, no
            // spill). ARF_DK=0 forces the plain CB=4 path for A/B.
            let cb4 = if force_rb2 || cb.as_deref() == Some("2") || cb.as_deref() == Some("8") {
                None
            } else if std::env::var("ARF_DK").ok().as_deref() == Some("0") {
                k.matmul_coop_q4k_rb2_cb4.as_ref()
            } else {
                k.matmul_coop_q4k_rb2_cb4_dk2
                    .as_ref()
                    .or(k.matmul_coop_q4k_rb2_cb4.as_ref())
            };
            let cb2 = if force_rb2 {
                None
            } else {
                k.matmul_coop_q4k_rb2_cb2.as_ref()
            };
            // The wide CB tiles assume m ≥ 16 (one 16-row mma tile). ARF_WIDE_M
            // lowers that threshold for A/B tuning the mid-range concurrency dip
            // (decode steps with 8–15 rows otherwise take the narrower RB=1 tile):
            // the kernels tile m by ceil(m/16) so they are correct for m ≥ 8, just
            // half-empty in the last tile — the experiment measures whether a
            // half-full wide tile beats a full narrow one at m∈[8,16). Default 16.
            let wide_m: usize = std::env::var("ARF_WIDE_M")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(16);
            let (coop, gx, gy) = match (m >= wide_m, cb8, cb4, cb2, k.matmul_coop_q4k_rb2.as_ref())
            {
                (true, Some(cb8), _, _, _) => {
                    (cb8, (m as u32).div_ceil(16), (n as u32).div_ceil(64))
                }
                (true, _, Some(cb4), _, _) => {
                    (cb4, (m as u32).div_ceil(16), (n as u32).div_ceil(32))
                }
                (true, _, None, Some(cb2), _) => {
                    (cb2, (m as u32).div_ceil(16), (n as u32).div_ceil(16))
                }
                (true, _, None, None, Some(rb2)) => {
                    (rb2, (m as u32).div_ceil(16), (n as u32).div_ceil(8))
                }
                _ => (
                    k.matmul_coop_q4k.as_ref().unwrap(),
                    (m as u32).div_ceil(8),
                    (n as u32).div_ceil(8),
                ),
            };
            cp.add_bound(
                coop,
                mm,
                &[
                    a,
                    codes.as_entire_binding(),
                    out,
                    uni.resource(&dims),
                    scales.as_entire_binding(),
                    mins.as_entire_binding(),
                ],
                [gx, gy, 1],
            );
        }
        // Batched Q4_K-lite (m > 1) — Q4_K prefill. Chunked by MAXM like Q4.
        (
            GpuMatWeight::Q4K {
                codes,
                scales,
                mins,
            },
            _,
        ) => {
            let mut row_off = 0usize;
            while row_off < m {
                let chunk = (m - row_off).min(MATMUL_VEC_BATCH_MAXM);
                let cdims = uni
                    .alloc_write(&[chunk as u32, kdim as u32, n as u32, row_off as u32])
                    .expect("q4k batch dims");
                cp.add_bound(
                    &k.matmul_vec_q4k_batch,
                    mm,
                    &[
                        a.clone(),
                        codes.as_entire_binding(),
                        out.clone(),
                        uni.resource(&cdims),
                        scales.as_entire_binding(),
                        mins.as_entire_binding(),
                    ],
                    [gemv_groups, 1, 1],
                );
                row_off += chunk;
            }
        }
        // Q4_K_S decode GEMV — Q4_K plus u8/i8 scales + a `dd` (f32 d,dmin) binding.
        (
            GpuMatWeight::Q4KS {
                codes,
                scales,
                mins,
                dd,
                ..
            },
            1,
        ) => {
            let res = [
                a,
                codes.as_entire_binding(),
                out,
                uni.resource(&dims),
                scales.as_entire_binding(),
                mins.as_entire_binding(),
                dd.as_entire_binding(),
            ];
            match cache {
                Some(cur) => {
                    cp.add_bound_cached(cur, &k.matmul_vec_q4ks, mm, &res, [gemv_groups, 1, 1])
                }
                None => cp.add_bound(&k.matmul_vec_q4ks, mm, &res, [gemv_groups, 1, 1]),
            }
        }
        // Batched Q4_K_S COOP tiled GEMM (m ≥ coop_min_m): the dense q/k/v/o/lm_head
        // concurrency lever. The per-row GEMV below re-reads each weight column per row
        // (bandwidth-bound at high B); this tiled kernel stages the weight tile in
        // threadgroup mem ONCE and reuses it across all m rows via coopMultiplyAdd —
        // the b32 advantage, applied to the model's actual quant (was built in
        // matmul_coop_q4ks.wgsl but never wired). Grid = one 8×8 output tile per workgroup.
        // Falls through to the GEMV when coop is unavailable / m small / k too large /
        // ARF_NO_COOP. Coop folds the two-level dequant into a staged f32 weight, so
        // it matches the GEMV's `s·Σ(nib·a)+lo·Σa` modulo f32 reassociation (parity vs
        // matmul_nt_q4ks oracle; greedy-token-identical, not bit-identical logits).
        (
            GpuMatWeight::Q4KS {
                codes,
                scales,
                mins,
                dd,
                ..
            },
            _,
        ) if m >= MM_TILE_MIN_M
            && kdim <= coop_max_k()
            && k.matmul_coop_q4ks.is_some()
            && std::env::var_os("ARF_NO_COOP").is_none() =>
        {
            // CONCURRENCY DEFAULT (m >= 8, llama.cpp's ne11_mm_min threshold): the tiled Q4KS
            // GEMM beats the per-row GEMV at batch — it stages each weight tile in threadgroup
            // mem ONCE and reuses across all m token rows (vs the GEMV re-reading per row).
            // MEASURED end-to-end (qwen3-coder-30B serve, today's binary): conc16 70.3->78.4
            // (+12%), conc32 76.5->85.6 (+12%). Gated at m>=8 (NOT the lower coop_min_m): at
            // m<8 the GEMV's batch-CSE already amortizes and the 8x8 tile's lower occupancy
            // loses, so single-stream/low-conc stays on the GEMV (untouched at parity). Opt out
            // with ARF_NO_COOP. Parity: greedy-token-identical (coop folds the two-level
            // dequant, matches the GEMV modulo f32 reassociation). This wgpu coop is the
            // shippable serial-path win.
            let cdims = uni
                .alloc_write(&[m as u32, kdim as u32, n as u32, 0u32])
                .expect("q4ks coop dims");
            cp.add_bound(
                k.matmul_coop_q4ks.as_ref().unwrap(),
                mm,
                &[
                    a,
                    codes.as_entire_binding(),
                    out,
                    uni.resource(&cdims),
                    scales.as_entire_binding(),
                    mins.as_entire_binding(),
                    dd.as_entire_binding(),
                ],
                // Grid: each workgroup owns 8 rows x 32 cols (NT=4 column fragments), so the
                // n dimension divides by 32 not 8 — 4x fewer workgroups, same output.
                [(m as u32).div_ceil(8), (n as u32).div_ceil(32), 1],
            );
        }
        // Batched Q4_K_S (m > 1) GEMV — small-m / no-coop fallback + prefill. Chunked by MAXM.
        (
            GpuMatWeight::Q4KS {
                codes,
                scales,
                mins,
                dd,
                ..
            },
            _,
        ) => {
            let mut row_off = 0usize;
            while row_off < m {
                let chunk = (m - row_off).min(MATMUL_VEC_BATCH_MAXM);
                let cdims = uni
                    .alloc_write(&[chunk as u32, kdim as u32, n as u32, row_off as u32])
                    .expect("q4ks batch dims");
                cp.add_bound(
                    &k.matmul_vec_q4ks_batch,
                    mm,
                    &[
                        a.clone(),
                        codes.as_entire_binding(),
                        out.clone(),
                        uni.resource(&cdims),
                        scales.as_entire_binding(),
                        mins.as_entire_binding(),
                        dd.as_entire_binding(),
                    ],
                    [gemv_groups, 1, 1],
                );
                row_off += chunk;
            }
        }
        // Q3_K decode GEMV — 3-bit split-plane (ql/qh) + Q4_K_S two-level scales.
        (
            GpuMatWeight::Q3K {
                ql,
                qh,
                scales,
                mins,
                dd,
            },
            1,
        ) => {
            let res = [
                a,
                ql.as_entire_binding(),
                qh.as_entire_binding(),
                out,
                uni.resource(&dims),
                scales.as_entire_binding(),
                mins.as_entire_binding(),
                dd.as_entire_binding(),
            ];
            match cache {
                Some(cur) => {
                    cp.add_bound_cached(cur, &k.matmul_vec_q3k, mm, &res, [gemv_groups, 1, 1])
                }
                None => cp.add_bound(&k.matmul_vec_q3k, mm, &res, [gemv_groups, 1, 1]),
            }
        }
        // Batched Q3_K (m > 1) — Q3_K prefill. Chunked by MAXM.
        (
            GpuMatWeight::Q3K {
                ql,
                qh,
                scales,
                mins,
                dd,
            },
            _,
        ) => {
            let mut row_off = 0usize;
            while row_off < m {
                let chunk = (m - row_off).min(MATMUL_VEC_BATCH_MAXM);
                let cdims = uni
                    .alloc_write(&[chunk as u32, kdim as u32, n as u32, row_off as u32])
                    .expect("q3k batch dims");
                cp.add_bound(
                    &k.matmul_vec_q3k_batch,
                    mm,
                    &[
                        a.clone(),
                        ql.as_entire_binding(),
                        qh.as_entire_binding(),
                        out.clone(),
                        uni.resource(&cdims),
                        scales.as_entire_binding(),
                        mins.as_entire_binding(),
                        dd.as_entire_binding(),
                    ],
                    [gemv_groups, 1, 1],
                );
                row_off += chunk;
            }
        }
        // Cooperative-matrix int8 GEMM (m ≥ 8) when the device has it: the matrix
        // units lift the batched-GEMV compute ceiling while keeping int8 bandwidth.
        // One dispatch for any m (coop tiles internally, no MAXM chunking). m < 8
        // keeps the weight-amortizing GEMV below.
        (GpuMatWeight::Int8 { data, scale }, _)
            if m >= coop_min_m() && kdim <= coop_max_k() && k.matmul_coop_q8.is_some() =>
        {
            // m ≥ 16 → CB=4 (16×32 tile, 8 mma/barrier; gy=ceil(n/32)), else RB=2,
            // else RB=1. ARF_CB=0 forces RB=2 (per-device A/B + safety valve).
            let force_rb2 = std::env::var("ARF_CB").ok().as_deref() == Some("0");
            let cb4 = if force_rb2 {
                None
            } else if std::env::var("ARF_DK").ok().as_deref() == Some("0") {
                k.matmul_coop_q8_rb2_cb4.as_ref()
            } else {
                k.matmul_coop_q8_rb2_cb4_dk2
                    .as_ref()
                    .or(k.matmul_coop_q8_rb2_cb4.as_ref())
            };
            let (coop, gx, gy) = match (m >= 16, cb4, k.matmul_coop_q8_rb2.as_ref()) {
                (true, Some(cb4), _) => (cb4, (m as u32).div_ceil(16), (n as u32).div_ceil(32)),
                (true, None, Some(rb2)) => (rb2, (m as u32).div_ceil(16), (n as u32).div_ceil(8)),
                _ => (
                    k.matmul_coop_q8.as_ref().unwrap(),
                    (m as u32).div_ceil(8),
                    (n as u32).div_ceil(8),
                ),
            };
            cp.add_bound(
                coop,
                mm,
                &[
                    a,
                    data.as_entire_binding(),
                    out,
                    uni.resource(&dims),
                    scale.as_entire_binding(),
                ],
                [gx, gy, 1],
            );
        }
        // Batched int8: int8 bandwidth + the read-each-weight-row-once batching
        // win. The kernel handles <= MAXM rows per dispatch; for m > MAXM (int8
        // prefill) we chunk, advancing row_off by MAXM each dispatch (one weight
        // re-stream per chunk — prefill is one-shot, not the throughput path).
        (GpuMatWeight::Int8 { data, scale }, _) => {
            let mut row_off = 0usize;
            while row_off < m {
                let chunk = (m - row_off).min(MATMUL_VEC_BATCH_MAXM);
                let cdims = uni
                    .alloc_write(&[chunk as u32, kdim as u32, n as u32, row_off as u32])
                    .expect("int8 batch dims");
                cp.add_bound(
                    &k.matmul_vec_q8_batch,
                    mm,
                    &[
                        a.clone(),
                        data.as_entire_binding(),
                        out.clone(),
                        uni.resource(&cdims),
                        scale.as_entire_binding(),
                    ],
                    [gemv_groups, 1, 1],
                );
                row_off += chunk;
            }
        }
        // m==1 bf16: subgroup-reduction GEMV when the device has it (subgroupAdd
        // instead of the shared-mem tree), else the tree-reduction GEMV. Same bindings.
        (GpuMatWeight::Bf16(w), 1) => {
            let kern = k.matmul_vec_sg.as_ref().unwrap_or(&k.matmul_vec);
            let res = [a, w.as_entire_binding(), out, uni.resource(&dims)];
            match cache {
                Some(cur) => cp.add_bound_cached(cur, kern, mm, &res, [gemv_groups, 1, 1]),
                None => cp.add_bound(kern, mm, &res, [gemv_groups, 1, 1]),
            }
        }
        // Cooperative-matrix GEMM (Apple simdgroup_matrix / tensor cores) for the
        // batched-decode (m=8..16) AND prefill (m>16) bf16 path when the device has
        // it — the inner product runs on the matrix units, lifting the scalar GEMV's
        // compute-bound ceiling (P8). Gated m≥8 (one 8×8 tile row); m<8 keeps the
        // weight-amortizing GEMV below, which is better for tiny batches. One
        // workgroup per 8×8 output tile; edges (m,n,k % 8) guarded in-shader.
        (GpuMatWeight::Bf16(w), _)
            if m >= coop_min_m() && kdim <= coop_max_k() && k.matmul_coop.is_some() =>
        {
            // m ≥ 16 → CB=4 (16×32 tile, 8 mma/barrier; gy=ceil(n/32)) raises the
            // plateau further than RB=2; else RB=2; m ∈ [8,16) keeps RB=1 (a second
            // tile would be wholly wasted). ARF_CB=0 forces RB=2 (A/B + safety valve).
            let force_rb2 = std::env::var("ARF_CB").ok().as_deref() == Some("0");
            let cb4 = if force_rb2 {
                None
            } else if std::env::var("ARF_DK").ok().as_deref() == Some("0") {
                k.matmul_coop_rb2_cb4.as_ref()
            } else {
                k.matmul_coop_rb2_cb4_dk2
                    .as_ref()
                    .or(k.matmul_coop_rb2_cb4.as_ref())
            };
            let (coop, gx, gy) = match (m >= 16, cb4, k.matmul_coop_rb2.as_ref()) {
                (true, Some(cb4), _) => (cb4, (m as u32).div_ceil(16), (n as u32).div_ceil(32)),
                (true, None, Some(rb2)) => (rb2, (m as u32).div_ceil(16), (n as u32).div_ceil(8)),
                _ => (
                    k.matmul_coop.as_ref().unwrap(),
                    (m as u32).div_ceil(8),
                    (n as u32).div_ceil(8),
                ),
            };
            cp.add_bound(
                coop,
                mm,
                &[a, w.as_entire_binding(), out, uni.resource(&dims)],
                [gx, gy, 1],
            );
        }
        // Small batch (continuous decode): the batched GEMV streams each weight
        // row once and applies it to all m rows — the amortization the tiled GEMM
        // (98% of batched GPU time) doesn't get. Gated m <= MATMUL_VEC_MAXM; the
        // shader's per-lane accumulator array is that fixed size. Prefill (m =
        // prompt_len, routinely > the cap) falls through to the tiled GEMM.
        (GpuMatWeight::Bf16(w), _) if m <= MATMUL_VEC_BATCH_MAXM => cp.add_bound(
            &k.matmul_vec_batch,
            mm,
            &[a, w.as_entire_binding(), out, uni.resource(&dims)],
            [gemv_groups, 1, 1],
        ),
        (GpuMatWeight::Bf16(w), _) => cp.add_bound(
            &k.matmul,
            mm,
            &[a, w.as_entire_binding(), out, uni.resource(&dims)],
            [(m as u32).div_ceil(16), (n as u32).div_ceil(16), 1],
        ),
    }
}

/// Max batch rows the batched GEMV ([`matmul_vec_batch.wgsl`]) handles in one
/// dispatch — MUST equal the shader's `MAXM` (its per-lane accumulator array and
/// `partial` workgroup array are sized to it: 64 lanes × 16 = 4 KiB). Set to 16,
/// not 32: a 32-slot per-lane array spilled registers and collapsed throughput
/// past m~20 (measured). Larger m (prefill, or batch > 16) uses the tiled GEMM.
pub(crate) const MATMUL_VEC_BATCH_MAXM: usize = 16;

/// Record an in-place residual add `a[i] += b[i]` of length `n`. The `ad` dims
/// uniform is sub-allocated from `uni`.
pub(crate) fn add_residual(
    cp: &mut CommandPass,
    k: &GpuKernels,
    uni: &mut GpuArena,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    n: usize,
) {
    let dims = uni
        .alloc_write(&[n as u32, 0, 0, 0])
        .expect("decode uniform arena overflow");
    cp.add_bound(
        &k.add,
        "add",
        &[
            a.as_entire_binding(),
            b.as_entire_binding(),
            uni.resource(&dims),
        ],
        [(n as u32).div_ceil(256), 1, 1],
    );
}

/// In-place residual add `a[i] += b[i]` over arena sub-ranges (`a` read+written,
/// `b` read — they must be in different arena buffers). The `n`-length batch
/// counterpart of [`add_residual`].
pub(crate) fn add_residual_r(
    cp: &mut CommandPass,
    k: &GpuKernels,
    uni: &mut GpuArena,
    a: wgpu::BindingResource<'_>,
    b: wgpu::BindingResource<'_>,
    n: usize,
) {
    let dims = uni
        .alloc_write(&[n as u32, 0, 0, 0])
        .expect("decode uniform arena overflow");
    cp.add_bound(
        &k.add,
        "add",
        &[a, b, uni.resource(&dims)],
        [(n as u32).div_ceil(256), 1, 1],
    );
}
