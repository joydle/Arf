// MSL gated-delta-net AUTOREGRESSIVE (1-token decode) recurrence for Qwen3.6-27B (M3).
// Bit-exact port of llama.cpp src/models/delta-net-base.cpp
// `build_delta_net_autoregressive` (lines 319-370), the n_tokens==1 path.
//
// THE DELTA RULE (per v-head h, GDA = scalar gate per head). State S_h is [S_v x S_v] f32,
// carried across tokens. Inputs per head (1 token): q[S_k], k[S_k], v[S_v], g (scalar),
// beta (scalar). num_v_heads heads total; q/k already L2-normed AND repeated across the
// H_v/H_k v-head groups by the host (so this kernel sees one q/k per v-head). q is ALSO
// already scaled by 1/sqrt(S_k) by the host? NO — see note: we scale q here to keep the
// host trivial. Geometry: S_v == S_k == 128 for the beast.
//
// ===== EXACT ggml ops -> per-element math (confirmed from delta-net-base.cpp) =====
// Layout: S[i][j] is flat S[i + j*S_v]  (i = ne[0] row/innermost, j = ne[1] column).
//   q = q * (1/sqrt(S_k))                          (ggml_scale, line 321)
//   gexp = exp(g_h)                                (line 339, GDA scalar per head)
//   S[i][j] *= gexp                                (line 340, g broadcasts over whole matrix)
//   sk[m] = sum_i S[i][m] * k[i]                   (lines 344-345: s*k then sum_rows over ne0)
//   d[m]  = (v[m] - sk[m]) * beta_h                (lines 349-350: v - transpose(sk), * b)
//   S[i][j] += k[i] * d[j]                         (lines 358-361: outer(k,d_t), k on ne0, d on ne1)
//   o[j]  = sum_i S[i][j] * q[i]                   (lines 365-366: s*q then sum_rows over ne0)
// Output o[j] per head -> [head_v_dim, num_v_heads] = value_dim. State S advanced in place.
//
// NOTE the sum_rows axis: BOTH sk and o reduce the ROW (i/ne0) axis and produce a per-COLUMN
// (j) result. The outer product puts k on the row axis and d on the column axis. So the natural
// parallelization is ONE THREAD PER COLUMN j: thread j owns column j of S (the 128 values
// S[i][j], i=0..S_v-1). It needs k[i],q[i] for all i (read from threadgroup-shared copies) and
// the scalar d[j] which depends on sk[j] = its own column's dot with k -> NO cross-thread state
// update is needed between the read (sk) and the write (S+=k*d) because d[j] uses ONLY column j.
//
// DESIGN: one THREADGROUP per v-head (num_v_heads groups). S_v threads per group (128), thread
// j owns column j. Steps:
//   (a) load k,q,v into threadgroup memory (each thread loads element j of k,q,v).
//   (b) barrier.
//   (c) thread j: decay its column S[i][j]*=gexp; compute sk_j = sum_i S[i][j]*k[i];
//       d_j = (v[j]-sk_j)*beta; then S[i][j] += k[i]*d_j  (write-back its column).
//   (d) o[j] = sum_i S[i][j]*q[i]  (post-update S, same column) -> write o[head_off + j].
// No second barrier needed for o because each thread reads/writes ONLY its own column j and q is
// in shared mem. (Step c's S+= touches only column j; step d reads only column j.)
//
// Bindings (raw-Metal encoder, indices 0..):
//   0 q   [S_k * num_v_heads] f32   (post-L2, post-repeat; NOT pre-scaled — we scale here)
//   1 k   [S_k * num_v_heads] f32   (post-L2, post-repeat)
//   2 v   [S_v * num_v_heads] f32
//   3 g   [num_v_heads] f32         (the GDA scalar gate per head, PRE-exp)
//   4 beta[num_v_heads] f32         (post-sigmoid)
//   5 S   [S_v * S_v * num_v_heads] f32  (READ+WRITE state, advanced in place)
//   6 o   [S_v * num_v_heads] f32   (readout output)
//   7 dims {S_v, S_k, num_v_heads, _} u32x4
// GRID: threadgroups = num_v_heads, threads/tg = S_v (== S_k == 128 for the beast).

#include <metal_stdlib>
using namespace metal;

// L174 — `_p` becomes `n_seqs`. 0 or 1 == one stream == pre-L174 behaviour.
// L227 — `row0` is the batch row this dispatch starts at. The recurrence is SERIAL in the
// token axis: when the k rows are k causal POSITIONS OF ONE SEQUENCE (speculative verify), they
// share one state bank, and dispatching them together has every row read the same pre-step state
// with the last writer winning. The host then issues k dispatches of height 1, row0 = 0..k, with
// a barrier between, so position r observes position r-1's state. row0=0 with a full-height grid
// is the shipped behaviour, byte-identical.
// L313 — `serial_rows` moves the L227 serial loop INSIDE the kernel: 0 keeps the shipped
// per-dispatch behaviour byte-for-byte; N > 0 makes ONE dispatch (height 1) walk rows 0..N of one
// sequence, chaining the state through registers/device memory with threadgroup barriers only.
// The host loop it replaces issued N dispatches with encoder-wide memoryBarriers between — ~100
// barriers per verify window across 48 layers, and they billed like it. `ckpt` (with buffer 9)
// makes the thread that OWNS each state column also write it to checkpoint slot r after row r —
// the copy dispatches disappear entirely.
struct GdnDims { uint s_v; uint s_k; uint n_vheads; uint n_seqs; uint row0; uint serial_rows; uint ckpt; uint _pad; };

#define GDN_MAX_S 256u   // S_v == S_k == 128 for the beast; threadgroup arrays cap.

kernel void gated_delta_net(
        device const float *q     [[buffer(0)]],
        device const float *k     [[buffer(1)]],
        device const float *v     [[buffer(2)]],
        device const float *g     [[buffer(3)]],
        device const float *beta  [[buffer(4)]],
        device       float *S     [[buffer(5)]],
        device       float *o     [[buffer(6)]],
        constant GdnDims   &d     [[buffer(7)]],
        // L175 — s_copy: batch row -> PINNED STATE ROW. llama.cpp's `s_copy` (llama-graph.cpp:3384
        // `ggml_get_rows(states, state_copy)`) does the same job: a sequence's recurrent state must
        // follow the SEQUENCE, not its position in this step's batch, because a row's position
        // shifts as sequences are admitted and evicted. Everything else here is indexed by `seq`
        // (the batch row); ONLY the S state uses s_copy[seq].
        device const uint  *s_copy [[buffer(8)]],
        // L313 — checkpoint slots [SPEC_CKPT_ROWS][n_vheads*S_v*S_v]; written only when
        // d.ckpt != 0 && d.serial_rows > 0. The host binds the state buffer here as a dummy on
        // the ordinary path, which never reads or writes it.
        device       float *ckpt   [[buffer(9)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]]) {
    // L174 — SEQUENCE AXIS, ggml attribute set (uint3 / ushort3 — mixing uint2 with a scalar
    // uint silently breaks the kernel; that cost a red gate on gdn_l2_norm).
    // Threadgroup-per-(head, seq): tgpig.x is the v-head, tgpig.y the sequence.
    const uint j    = uint(tpitg.x);
    const uint head = tgpig.x;
    const uint S_v = d.s_v;
    const uint S_k = d.s_k;          // == S_v for delta-net (GGML_ASSERT S_k == S_v)
    if (head >= d.n_vheads) return;
    const uint iters = d.serial_rows > 0u ? d.serial_rows : 1u;
    for (uint r = 0u; r < iters; ++r) {
    const uint seq  = d.serial_rows > 0u ? r : (tgpig.y + d.row0);

    // Per-(seq, head) bases. Each sequence owns a contiguous slab; within it the per-head layout
    // is exactly the single-stream one, so seq==0 reproduces it byte-for-byte.
    const uint nvh    = d.n_vheads;
    const uint qkbase = (seq * nvh + head) * S_k;       // q,k row base
    const uint vbase  = (seq * nvh + head) * S_v;       // v / o row base
    const uint bank   = s_copy[seq];                    // pinned state row for this sequence
    const uint sbase  = (bank * nvh + head) * S_v * S_v; // this head's [S_v x S_v] state base

    // Threadgroup copies of q,k (every thread reads ALL of k,q over the i axis).
    threadgroup float ksh[GDN_MAX_S];
    threadgroup float qsh[GDN_MAX_S];

    const float scale = 1.0f / sqrt((float)S_k);

    // (a) cooperative load: thread j fills slot j of k,q (q pre-scaled here). In the serial
    // loop, iteration r's load must not race iteration r-1's reads of the same arrays.
    if (r > 0u) { threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (j < S_k) {
        ksh[j] = k[qkbase + j];
        qsh[j] = q[qkbase + j] * scale;     // ggml_scale(q, 1/sqrt(S_k)) folded in
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (j >= S_v) continue; // guard threads past S_v (none for the beast: S_v==threads).
                            // `continue`, not `return`: a returned thread would desert the
                            // serial loop's barriers.

    // g and beta are [n_seqs, n_vheads] — indexing by `head` alone would feed every sequence
    // row 0's decay and gate.
    const uint hscal   = seq * nvh + head;
    const float gexp   = exp(g[hscal]);
    const float beta_h = beta[hscal];

    // (c) thread j owns COLUMN j: S[i][j] at flat index sbase + i + j*S_v.
    const uint col0 = sbase + j * S_v;       // S[0][j]

    // decay + sk_j = sum_i (S[i][j]*gexp) * k[i]   (decay folded into the read).
    float sk_j = 0.0f;
    for (uint i = 0u; i < S_v; ++i) {
        float s_dec = S[col0 + i] * gexp;    // S[i][j] after decay
        S[col0 + i] = s_dec;                 // write decayed state back
        sk_j += s_dec * ksh[i];
    }
    // d_j = (v[j] - sk_j) * beta_h
    const float d_j = (v[vbase + j] - sk_j) * beta_h;

    // (c cont.) S[i][j] += k[i] * d_j ; (d) o[j] = sum_i S[i][j] * q[i]  (post-update).
    float o_j = 0.0f;
    for (uint i = 0u; i < S_v; ++i) {
        float s_new = S[col0 + i] + ksh[i] * d_j;
        S[col0 + i] = s_new;
        o_j += s_new * qsh[i];
        // L313 — checkpoint slot r: the post-row state, written by the column's owner.
        if (d.ckpt != 0u && d.serial_rows > 0u && r + 1u < d.serial_rows) {
            ckpt[(r * d.n_vheads + head) * S_v * S_v + j * S_v + i] = s_new;
        }
    }
    o[vbase + j] = o_j;
    } // serial-rows loop (single iteration on the ordinary path)
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// L333 — REGISTER-RESIDENT VARIANT (ARF_GDN_SIMD=1). Ported from llama.cpp's
// ggml/src/ggml-metal/kernels/gated_delta_net.metal (upstream 8887a48f0), which is 250 lines
// and contains NO threadgroup_barrier at all. Three differences from the kernel above, all
// measured-relevant on the speculative verify path (L331: the verify is 99% GPU execution at
// 2.54x a 1-row step, and 48 of this model's 64 layers are GDN):
//
//   1. STATE IN REGISTERS. `ls[NSG]` is loaded ONCE before the row loop and carried in
//      registers across every row, written back once at the end. The kernel above reads AND
//      writes S[col0+i] TWICE per row — the value written by the decay loop is re-read by the
//      update loop, a device-memory round trip for a value the thread already holds.
//   2. simd_sum INSTEAD OF BARRIERS. s_k and o are reduced with simd_sum(), which needs no
//      threadgroup memory and no barrier. The kernel above pays two threadgroup_barriers per
//      row; at 48 GDN layers x 2 rows that is ~192 per verify window.
//   3. THE ROW LOOP IS THE NATURAL SHAPE. State persists in registers between iterations, so a
//      k+1-row verify costs close to one row instead of ~2x.
//
// LAYOUT — why the registers fit, and why my first analysis said they would not. The kernel
// above gives ONE THREAD A WHOLE COLUMN, so caching it means S_v=128 floats = 512 B/thread,
// which spills. Here the column is SPLIT ACROSS THE SIMD GROUP:
//     row  i20 = tgpig.x*NSG + ty     (which row of S_v this thread-row owns)
//     col  is  = tx*NSG + j           (column index, distributed over the 32 lanes)
// At NSG=4 a 32-lane group covers 128 columns holding only 4 floats (16 B) per thread. The
// spill argument disappears once the layout changes with it.
//
// STATE TRANSPOSE. This variant reads and writes S as M[i20][is] = S[is][i20] — row i20
// contiguous — because each thread-row owns a ROW of the state and wants its lanes' columns
// adjacent. The kernel above is column-major per thread. The host must therefore NOT mix the
// two on one bank without a transpose; ARF_GDN_SIMD selects one for the whole run.
//
// NOT YET THE DEFAULT. Gated behind ARF_GDN_SIMD so the shipped path stays byte-identical
// until an interleaved A/B says otherwise (rule 7: host and kernel revert together).

#define GDN_SIMD_LANES 32u

template<ushort NSG>
void gdn_simd_impl(
        device const float *q, device const float *k, device const float *v,
        device const float *g, device const float *beta,
        device       float *S, device float *o,
        constant GdnDims &d, device const uint *s_copy, device float *ckpt, device float *rin,
        uint3 tgpig, ushort3 tpitg) {
    const uint S_v  = d.s_v;
    const uint nvh  = d.n_vheads;
    const uint head = tgpig.x;
    if (head >= nvh) return;

    const uint tx  = uint(tpitg.x);              // SIMD lane: which column block
    const uint ty  = uint(tpitg.y);              // which state row within this threadgroup
    const uint i20 = tgpig.y * NSG + ty;         // this thread-row's row of S_v
    if (i20 >= S_v) return;

    const uint rows = d.serial_rows > 0u ? d.serial_rows : 1u;
    const float scale = 1.0f / sqrt((float)d.s_k);

    // Bank for row 0. Every row of a serial window shares ONE sequence, hence one bank —
    // that is exactly why the recurrence is serial (L227).
    const uint bank0  = s_copy[d.serial_rows > 0u ? 0u : (tgpig.z + d.row0)];
    const uint sbase  = (bank0 * nvh + head) * S_v * S_v + i20 * S_v;

    // (1) LOAD THE STATE ROW INTO REGISTERS — once, for the whole row loop.
    // ckpt modes (2026-09-24): 1 = a checkpoint after every row but the last (L313); 2 = SAVE THE
    // INITIAL STATE to slot 0 only, then advance in place (a partial accept REPLAYS from it: one
    // 3 MB write a layer instead of seven — the per-row slots were 2.36 ms of an 8-row verify);
    // 3 = REPLAY: load the state from slot 0 instead of the bank and write the bank at the end.
    device float *cp0 = ckpt + head * S_v * S_v + i20 * S_v;
    float ls[NSG];
    #pragma unroll
    for (ushort jj = 0; jj < NSG; ++jj) {
        const uint is = tx * NSG + jj;
        ls[jj] = (is < S_v) ? (d.ckpt == 3u ? cp0[is] : S[sbase + is]) : 0.0f;
    }
    if (d.ckpt == 2u) {
        #pragma unroll
        for (ushort jj = 0; jj < NSG; ++jj) {
            const uint is = tx * NSG + jj;
            if (is < S_v) { cp0[is] = ls[jj]; }
        }
        // the replay's inputs: q/k by the head's row-0 simdgroup, v by every row's, g/beta once.
        // RIN = GDN_VERIFY_ROWS (island.rs): the layout's row capacity, 16 since 2026-10-05 (a
        // multi-stream record's rows are indexed record-wide).
        const uint W = nvh * S_v, RIN = 16u;
        device float *rq = rin, *rk = rin + RIN * W, *rv = rin + 2u * RIN * W;
        device float *rg = rin + 3u * RIN * W, *rbt = rg + RIN * nvh;
        for (uint r = 0u; r < rows && r < RIN; ++r) {
            const uint qkb = (r * nvh + head) * d.s_k, vb = (r * nvh + head) * S_v;
            if (i20 == 0u) {
                #pragma unroll
                for (ushort jj = 0; jj < NSG; ++jj) {
                    const uint is = tx * NSG + jj;
                    if (is < S_v) { rq[qkb + is] = q[qkb + is]; rk[qkb + is] = k[qkb + is]; }
                }
                if (tx == 0u) { rg[r * nvh + head] = g[r * nvh + head]; rbt[r * nvh + head] = beta[r * nvh + head]; }
            }
            if (tx == 0u) { rv[vb + i20] = v[vb + i20]; }
        }
    }

    for (uint r = 0u; r < rows; ++r) {
        const uint seq   = d.serial_rows > 0u ? r : (tgpig.z + d.row0);
        const uint qkb   = (seq * nvh + head) * d.s_k;
        const uint vb    = (seq * nvh + head) * S_v;
        const uint hscal = seq * nvh + head;
        const float gexp = exp(g[hscal]);
        const float bh   = beta[hscal];

        // decay in registers + partial sum over this lane's columns
        float sk = 0.0f;
        #pragma unroll
        for (ushort jj = 0; jj < NSG; ++jj) {
            const uint is = tx * NSG + jj;
            if (is < S_v) { ls[jj] *= gexp; sk += ls[jj] * k[qkb + is]; }
        }
        sk = simd_sum(sk);                       // (2) no barrier, no threadgroup memory

        const float dj = (v[vb + i20] - sk) * bh;

        float y = 0.0f;
        #pragma unroll
        for (ushort jj = 0; jj < NSG; ++jj) {
            const uint is = tx * NSG + jj;
            if (is < S_v) { ls[jj] += k[qkb + is] * dj; y += ls[jj] * q[qkb + is]; }
        }
        y = simd_sum(y) * scale;                 // q pre-scaling folded here instead

        if (tx == 0u) { o[vb + i20] = y; }

        // (3) checkpoint straight from registers — no separate copy dispatch, and the LAST row
        // is deliberately not written (a full accept needs no restore). Same rule as L313.
        if (d.ckpt == 1u && d.serial_rows > 0u && r + 1u < d.serial_rows) {
            device float *cp = ckpt + (r * nvh + head) * S_v * S_v + i20 * S_v;
            #pragma unroll
            for (ushort jj = 0; jj < NSG; ++jj) {
                const uint is = tx * NSG + jj;
                if (is < S_v) { cp[is] = ls[jj]; }
            }
        }
    }

    // write the state back ONCE, after every row
    #pragma unroll
    for (ushort jj = 0; jj < NSG; ++jj) {
        const uint is = tx * NSG + jj;
        if (is < S_v) { S[sbase + is] = ls[jj]; }
    }
}

kernel void gated_delta_net_simd(
        device const float *q     [[buffer(0)]],
        device const float *k     [[buffer(1)]],
        device const float *v     [[buffer(2)]],
        device const float *g     [[buffer(3)]],
        device const float *beta  [[buffer(4)]],
        device       float *S     [[buffer(5)]],
        device       float *o     [[buffer(6)]],
        constant GdnDims   &d     [[buffer(7)]],
        device const uint  *s_copy [[buffer(8)]],
        device       float *ckpt   [[buffer(9)]],
        // mode 2: THIS LAYER's copy of the window's inputs for a replay, [q | k | v] each
        // [8][nvh*s_v] then [g | beta] each [8][nvh] — the record's q/k/v/g/beta are temporaries
        // SHARED by every layer (a replay reading them got the last layer's; caught by the text).
        device       float *rin    [[buffer(10)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]]) {
    // NSG = S_v / 32 lanes. 128 -> 4, 64 -> 2, 32 -> 1; anything else falls back to the
    // barrier kernel on the host side rather than being silently wrong here.
    gdn_simd_impl<4>(q, k, v, g, beta, S, o, d, s_copy, ckpt, rin, tgpig, tpitg);
}

// =================================================================================================
// THE FUSED GDN VERIFY KERNEL (2026-09-24) — another engine's `verify_gdn_fused_q4` idea, our state format.
// An 8-row serial verify window took SEVEN dependent stages between the conv and the out
// projection — sigmoid(beta), the decay gate, two l2 norms, the q/k tile, the scan, and the gated
// RMS norm (8 per-row dispatches) — ~100 us a layer measured one by one
// (a standalone probe + gdn_verify_probe.swift) against another engine's ~38 us
// for all of it. Here ONE threadgroup per value head (8 simdgroups):
//   A. simdgroup t = token t: its head's q and k from conv_out (key head = head % nkh, the tile
//      rule), l2-normalized (gdn_l2_norm's formula; the sum is a simd tree, not a serial loop, so
//      the bits differ in the last place), v, beta = sigmoid, gate = exp(softplus(alpha+dt)*a),
//      all to threadgroup memory — and the replay copy (`rin`) the restore reads.
//   B. the scan: simdgroup s owns state rows 16s..16s+15, a lane 4 columns of each (64 registers),
//      the initial state saved to checkpoint slot 0 (ckpt mode 2, see gdn_simd_impl).
//   C. simdgroup t = token t: the gated RMS norm (gdn_norm_gate_b's formula) straight into the out
//      projection's input.
// The conv stays its own dispatch: three value heads share each key head's ring, so no single
// threadgroup may own its update. Serial verify windows only (rows <= 8, s = 128).
struct GdnFusedDims { uint s; uint nkh; uint nvh; uint key_dim; uint conv_dim; uint rows; float norm_eps; float l2_eps; };

kernel void gdn_verify_fused(
        device const float *conv_out [[buffer(0)]],   // [rows][conv_dim] (post conv+silu)
        device const float *beta_raw [[buffer(1)]],   // [rows][nvh]
        device const float *alpha    [[buffer(2)]],   // [rows][nvh]
        device const float *ssm_dt   [[buffer(3)]],   // [nvh]
        device const float *ssm_a    [[buffer(4)]],   // [nvh]
        device       float *S        [[buffer(5)]],   // bank: [bank][nvh][s][s]
        device const float *z        [[buffer(6)]],   // [rows][value_dim]
        device       float *gated    [[buffer(7)]],   // [rows][value_dim]
        device const float *nweight  [[buffer(8)]],   // [s]
        constant GdnFusedDims &d     [[buffer(9)]],
        device const uint  *s_copy   [[buffer(10)]],
        device       float *ckpt     [[buffer(11)]],  // slot 0: [nvh][s][s]
        device       float *rin      [[buffer(12)]],  // replay inputs (gdn_simd_impl's layout)
        uint  head [[threadgroup_position_in_grid]],
        uint  lane [[thread_index_in_simdgroup]],
        uint  sg   [[simdgroup_index_in_threadgroup]]) {
    // R8: this dispatch's rows (one stream's window, <= 8). RIN: the replay layout's row capacity
    // (GDN_VERIFY_ROWS) — `rin` is bound at the segment's record row, so its stride is the record's.
    constexpr uint SV = 128u, R8 = 8u, RPS = 16u, RIN = 16u;
    const uint nvh = d.nvh, rows = d.rows, kh = head % d.nkh, vd = nvh * SV;
    threadgroup float4 tq[R8][32], tk[R8][32];
    threadgroup float tv[R8][SV], to[R8][SV], tg[R8], tb[R8];

    const uint W = vd;
    device float *rq = rin, *rk = rin + RIN * W, *rv = rin + 2u * RIN * W;
    device float *rg = rin + 3u * RIN * W, *rbt = rg + RIN * nvh;

    // ---- A ----
    if (sg < rows) {
        const uint t = sg;
        device const float *cr = conv_out + ulong(t) * d.conv_dim;
        float4 q4 = *reinterpret_cast<device const float4 *>(cr + kh * SV + lane * 4u);
        float4 k4 = *reinterpret_cast<device const float4 *>(cr + d.key_dim + kh * SV + lane * 4u);
        const float4 v4 = *reinterpret_cast<device const float4 *>(cr + 2u * d.key_dim + head * SV + lane * 4u);
        const float qs = simd_sum(dot(q4, q4)), ks = simd_sum(dot(k4, k4));
        q4 *= 1.0f / fmax(sqrt(qs), d.l2_eps);
        k4 *= 1.0f / fmax(sqrt(ks), d.l2_eps);
        tq[t][lane] = q4;
        tk[t][lane] = k4;
        *reinterpret_cast<threadgroup float4 *>(&tv[t][lane * 4u]) = v4;
        const uint o = (t * nvh + head) * SV + lane * 4u;
        *reinterpret_cast<device float4 *>(rq + o) = q4;
        *reinterpret_cast<device float4 *>(rk + o) = k4;
        *reinterpret_cast<device float4 *>(rv + o) = v4;
        if (lane == 0u) {
            const uint i = t * nvh + head;
            const float bt = 1.0f / (1.0f + exp(-beta_raw[i]));     // gdn_sigmoid_out
            const float x = alpha[i] + ssm_dt[head];
            const float sp = (x > 20.0f) ? x : log(1.0f + exp(x));  // gdn_g_decay
            const float gv = sp * ssm_a[head];
            tb[t] = bt;
            tg[t] = exp(gv);
            rg[i] = gv;
            rbt[i] = bt;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- B ----
    const float scale = 1.0f / sqrt(float(SV));
    const uint bank0 = s_copy[0];
    device float *sb = S + (ulong(bank0) * nvh + head) * SV * SV;
    device float *cb = ckpt + ulong(head) * SV * SV;
    float4 ls[RPS];
    for (uint j = 0; j < RPS; ++j) {
        const uint row = sg * RPS + j;
        ls[j] = *reinterpret_cast<device const float4 *>(sb + row * SV + lane * 4u);
        *reinterpret_cast<device float4 *>(cb + row * SV + lane * 4u) = ls[j];
    }
    for (uint r = 0; r < rows; ++r) {
        const float4 k4 = tk[r][lane], q4 = tq[r][lane];
        const float gx = tg[r], bt = tb[r];
        for (uint j = 0; j < RPS; ++j) {
            const uint row = sg * RPS + j;
            ls[j] *= gx;
            const float sk = simd_sum(dot(ls[j], k4));
            const float dj = (tv[r][row] - sk) * bt;
            ls[j] += k4 * dj;
            const float y = simd_sum(dot(ls[j], q4)) * scale;
            if (lane == 0u) to[r][row] = y;
        }
    }
    for (uint j = 0; j < RPS; ++j) {
        const uint row = sg * RPS + j;
        *reinterpret_cast<device float4 *>(sb + row * SV + lane * 4u) = ls[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- C ----
    if (sg < rows) {
        const uint t = sg;
        const float4 o4 = *reinterpret_cast<threadgroup const float4 *>(&to[t][lane * 4u]);
        const float tot = simd_sum(dot(o4, o4));
        const float inv = 1.0f / sqrt(tot / float(SV) + d.norm_eps);
        const float4 w4 = *reinterpret_cast<device const float4 *>(nweight + lane * 4u);
        const ulong zo = ulong(t) * vd + head * SV + lane * 4u;
        const float4 z4 = *reinterpret_cast<device const float4 *>(z + zo);
        const float4 sil = z4 / (1.0f + exp(-z4));
        *reinterpret_cast<device float4 *>(gated + zo) = o4 * inv * w4 * sil;
    }
}

// =================================================================================================
// THE PREFILL SCAN (2026-09-27) — another engine's prefill `gdn_scan` layout, our state format. Serial
// windows of MORE than 8 rows only (a prefill window: every row is committed, nothing is ever
// rolled back, so no checkpoint or replay input is written); decode and the <= 8-row verify keep
// gated_delta_net_simd / gdn_verify_fused.
// WHY: measured in context (ARF_DUP_GDN=scan, per-layer GPU time of 256-row windows, 2 rounds) the
// SIMD kernel's scan is ~1.6-1.8 ms of a GDN layer a window — ~0.8 s of a 2.4K-token prefill.
// It spends 32 lanes on each 128-wide state row (4 columns a lane): 196K threads a layer, several
// waves, EACH walking every token of the window with q/k/v/g/beta read from device memory.
// Here a lane owns PF_COLS = 16 columns of a row (8 lanes a row, 4 rows a simdgroup, 16 rows a
// threadgroup): a quarter of the threads, and each 16-token block of q, k (the head's whole
// 128-vectors), v (the threadgroup's 16 rows), exp(g) and beta is staged in threadgroup memory by
// the whole threadgroup first, so the per-token chain reads only threadgroup memory. The math is
// the SIMD kernel's; the dot products sum 16 columns per lane then 3 shuffle steps instead of 4 then
// 5, so the last bits differ (not bit-identical — gated on text, logprobs and perplexity).
// DISPATCH: grid (n_vheads, S_v / PF_ROWS), threadgroup (32 * PF_SG, 1, 1).
constant constexpr uint PF_COLS = 16u;
constant constexpr uint PF_LPR = 128u / PF_COLS;   // 8 lanes a state row
constant constexpr uint PF_RPS = 32u / PF_LPR;     // 4 state rows a simdgroup
constant constexpr uint PF_SG = 4u;
constant constexpr uint PF_ROWS = PF_RPS * PF_SG;  // 16 state rows a threadgroup
constant constexpr uint PF_BLK = 16u;              // tokens staged a block

kernel void gated_delta_net_pf(
        device const float *q      [[buffer(0)]],
        device const float *k      [[buffer(1)]],
        device const float *v      [[buffer(2)]],
        device const float *g      [[buffer(3)]],
        device const float *beta   [[buffer(4)]],
        device       float *S      [[buffer(5)]],
        device       float *o      [[buffer(6)]],
        constant GdnDims   &d      [[buffer(7)]],
        device const uint  *s_copy [[buffer(8)]],
        uint2  tgp  [[threadgroup_position_in_grid]],
        uint   tid  [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr uint SV = 128u;
    const uint nvh = d.n_vheads, head = tgp.x;
    if (head >= nvh || d.s_v != SV || d.s_k != SV) return;   // uniform per threadgroup
    const uint rrow = uint(sg) * PF_RPS + uint(lane) / PF_LPR;  // row within the threadgroup
    const uint row = tgp.y * PF_ROWS + rrow;                   // state row (i20)
    const uint c4 = (uint(lane) % PF_LPR) * (PF_COLS / 4u);    // first float4 of this lane's columns
    const uint rows = d.serial_rows;
    const float scale = 1.0f / sqrt((float)d.s_k);
    const ulong sbase = (ulong(s_copy[0]) * nvh + head) * SV * SV + ulong(row) * SV;
    device float4 *S4 = reinterpret_cast<device float4 *>(S + sbase);

    float4 ls[PF_COLS / 4u];
    for (uint j = 0u; j < PF_COLS / 4u; ++j) ls[j] = S4[c4 + j];

    threadgroup float4 tq[PF_BLK][SV / 4u], tk[PF_BLK][SV / 4u];
    threadgroup float tv[PF_BLK][PF_ROWS], tge[PF_BLK], tb[PF_BLK];
    for (uint t0 = 0u; t0 < rows; t0 += PF_BLK) {
        const uint nb = min(PF_BLK, rows - t0);
        for (uint i = tid; i < nb * (SV / 4u); i += 32u * PF_SG) {
            const uint t = i / (SV / 4u), f = i % (SV / 4u);
            const ulong b = (ulong(t0 + t) * nvh + head) * SV;
            tq[t][f] = reinterpret_cast<device const float4 *>(q + b)[f];
            tk[t][f] = reinterpret_cast<device const float4 *>(k + b)[f];
        }
        for (uint i = tid; i < nb * PF_ROWS; i += 32u * PF_SG) {
            const uint t = i / PF_ROWS, r = i % PF_ROWS;
            tv[t][r] = v[(ulong(t0 + t) * nvh + head) * SV + tgp.y * PF_ROWS + r];
        }
        if (tid < nb) {
            const uint hs = (t0 + tid) * nvh + head;
            tge[tid] = exp(g[hs]);
            tb[tid] = beta[hs];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0u; t < nb; ++t) {
            const float gexp = tge[t], bh = tb[t];
            float sk = 0.0f;
            for (uint j = 0u; j < PF_COLS / 4u; ++j) {
                const float4 kk = tk[t][c4 + j];
                ls[j] *= gexp;
                sk += ls[j].x * kk.x; sk += ls[j].y * kk.y; sk += ls[j].z * kk.z; sk += ls[j].w * kk.w;
            }
            sk += simd_shuffle_xor(sk, 1); sk += simd_shuffle_xor(sk, 2); sk += simd_shuffle_xor(sk, 4);
            const float dj = (tv[t][rrow] - sk) * bh;
            float y = 0.0f;
            for (uint j = 0u; j < PF_COLS / 4u; ++j) {
                const float4 kk = tk[t][c4 + j], qq = tq[t][c4 + j];
                ls[j] += kk * dj;
                y += ls[j].x * qq.x; y += ls[j].y * qq.y; y += ls[j].z * qq.z; y += ls[j].w * qq.w;
            }
            y += simd_shuffle_xor(y, 1); y += simd_shuffle_xor(y, 2); y += simd_shuffle_xor(y, 4);
            if (uint(lane) % PF_LPR == 0u) o[(ulong(t0 + t) * nvh + head) * SV + row] = y * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0u; j < PF_COLS / 4u; ++j) S4[c4 + j] = ls[j];
}
