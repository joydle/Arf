// B-ROW (batched) MoE decode kernels for the native Metal island batched megakernel — the
// B-row generalizations of the m=1 island MoE kernels (gemv_q4ks_id_msl / _down_msl /
// moe_route_msl / moe_swiglu_msl). Each is a FAITHFUL port of its m=1 sibling with the B-row
// (row,slot,col) / (row,h_j) layouts of the parity-proven WGSL kernels
// (matmul_vec_moe_q4ks_batch_gu_b.wgsl + matmul_vec_moe_q4ks_down_reduce_b.wgsl). At B=1
// (row=0) every kernel reduces EXACTLY to the m=1 island kernel → bit-identical.
//
// Layouts (ROW-MAJOR over B sequences):
//   normed/logits : [B, hidden] / [B, num_experts]
//   ids / wts     : [B, top_k]              (row r's routed experts/weights at r*top_k)
//   gate/up/silu  : [B, top_k, inter]       (row,slot,col)-major: (r*top_k+slot)*inter+col
//   mlp_down      : [B, hidden]             row r's output at r*hidden
//   activation a  : [B, k]                  row r's segment at r*k
// Per-expert weight packing is UNCHANGED (concatenated over num_experts; column = expert*n+col).

#include <metal_stdlib>
using namespace metal;

// ARF_MOE_GU_NSG2 (function_constant 4): simdgroups per threadgroup for the gate/up id-GEMV.
// Default 1 (32 threads, shipped path, byte-identical). NSG=2 (64 threads) splits the sub-block
// weight stream across TWO simdgroups → ~2× in-flight memory requests to hide the Q4_K weight-stream
// latency AT LOW FILL (conc16). This is the conc16 occupancy lever — llama runs its mul_mv_id with
// nsg=2 (64 lanes) at conc16 and gets 230 vs our starved 174; matching its occupancy is the fix.
// Each simdgroup sums a DISJOINT stride of sub-blocks; partials combine via threadgroup memory.
// Below-tie f32 reorder (different sub-block partition) — parity-gated, NOT bit-exact. gate/up ONLY
// (the down GEMV is reduction-bound and REGRESSED -25% under N_SG=2; not touched here).
constant uint MOE_GU_NSG_FC [[function_constant(4)]];
constant uint MOE_GU_NSG = is_function_constant_defined(MOE_GU_NSG_FC) ? MOE_GU_NSG_FC : 1u;

// ===================== gate/up indirect GEMV (B-row) =====================
// {m=B, k(=hidden), n(=inter), top_k}. grid = (ceil(n/COLS), B, top_k); Z = routed slot,
// Y = batch row. Output c[(row*top_k+slot)*n + col]. Activation a[row*k..]. ids[row*top_k+slot].
struct IdDims { uint m; uint k; uint n; uint top_k; };

kernel void gemv_q4ks_id_b(
        device const float4 *a       [[buffer(0)]],  // [B, k] activation
        device const uint4  *codes   [[buffer(1)]],  // all experts' codes
        device       float  *c       [[buffer(2)]],  // [B, top_k, n] (row,slot,col)-major
        constant     IdDims &d       [[buffer(3)]],
        device const uint   *scales  [[buffer(4)]],
        device const uint   *ids      [[buffer(5)]], // [B, top_k]
        device const uint   *mins    [[buffer(6)]],
        device const half2  *dd      [[buffer(7)]],
        uint3  tgpig                 [[threadgroup_position_in_grid]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]]) {
    const uint subs = d.k / 32u;
    // COLS = output columns per simdgroup. Each thread loads its activation sub-block
    // (av[8] + a_sum) ONCE per 32-stride and reuses it across all COLS columns — more
    // arithmetic per activation byte (the nr0/NSG amortization). The weight stream
    // (one codes[] vec4 per column per sub-block) is what bounds us, so widening COLS
    // does not add activation traffic. Host grid x-divisor MUST match (concurrent_metal.rs
    // gu_grid div_ceil(COLS) at every gemv_q4ks_id_b dispatch).
    const uint COLS = 8u;
    const uint row  = tgpig.y;                  // batch row (grid Y)
    const uint slot = tgpig.z;                  // routed-expert slot (grid Z)
    if (row >= d.m || slot >= d.top_k) return;
    const uint col0 = tgpig.x * COLS;
    if (col0 >= d.n) return;

    const uint expert = ids[row * d.top_k + slot];   // this row's slot-th routed expert
    const uint exp_col_base = expert * d.n;
    const uint arow = d.k / 4u;                       // float4s per activation row
    const uint a_off = row * arow;                    // this row's activation segment

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Late-scale-fold nibble handling (llama kernel_mul_mv_q4_K_f32_impl structure, adapted to
    // our uint4 packing). Instead of shift+mask+convert per nibble (32 shifts/uint4), we keep each
    // nibble MASKED-IN-PLACE (float(w & 0xF000) etc = nib*16^lane, exact in fp32) and fold the
    // per-lane 16^-lane correction (1, 1/16, 1/256, 1/4096) into the ACTIVATION once per sub-block.
    // Correction is amortized across all COLS (avc computed once, reused per column). Only the hi
    // 16-bit half needs a single w>>16 shift/uint (4 shifts/uint4 vs 32). Numerically a reassociation
    // of the SAME products → parity-close (min-fold `lo*a_sum` path unchanged).
    const float4 CORR = float4(1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f);
    // NSG=1: simdgroup 0 strides all sub-blocks (b += 32). NSG=2: each simdgroup takes a disjoint
    // stride (start sgitg*32 + tiisg, step NSG*32) → 64 lanes issue ~2× in-flight memory requests.
    for (uint b = sgitg * 32u + tiisg; b < subs; b += MOE_GU_NSG * 32u) {
        const uint abase = a_off + 8u * b;
        float4 av[8];
        for (short i = 0; i < 8; ++i) av[i] = a[abase + i];
        float a_sum = 0.0f;
        for (short i = 0; i < 8; ++i) a_sum += av[i].x + av[i].y + av[i].z + av[i].w;
        float4 avc[8];
        for (short i = 0; i < 8; ++i) avc[i] = av[i] * CORR;   // per-lane late-scale fold, amortized over COLS

        const uint sblk = b / 8u;
        const uint byte_b = b & 3u;

        for (uint cc = 0u; cc < COLS; ++cc) {
            const uint col = col0 + cc;
            if (col >= d.n) break;
            const uint wcol = exp_col_base + col;
            const uint idx = wcol * subs + b;
            const uint4 qv = codes[idx];

            const uint sbase = wcol * subs;
            const uint ddbase = wcol * (subs / 8u);
            const float2 dp = float2(dd[ddbase + sblk]);
            const float dv  = dp.x;
            const float dmv = dp.y;
            const uint sword = scales[(sbase + b) >> 2];
            const uint mword = mins[(sbase + b) >> 2];
            const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
            const int  mraw = int(mword << (24u - 8u * byte_b)) >> 24;
            const float s  = dv * su8;
            const float lo = dmv * float(mraw);

            float code_sum = 0.0f;
            uint hx = qv.x >> 16, hy = qv.y >> 16, hz = qv.z >> 16, hw = qv.w >> 16;
            code_sum += dot(avc[0], float4(float(qv.x & 0xFu), float(qv.x & 0xF0u), float(qv.x & 0xF00u), float(qv.x & 0xF000u)))
                      + dot(avc[1], float4(float(hx & 0xFu), float(hx & 0xF0u), float(hx & 0xF00u), float(hx & 0xF000u)));
            code_sum += dot(avc[2], float4(float(qv.y & 0xFu), float(qv.y & 0xF0u), float(qv.y & 0xF00u), float(qv.y & 0xF000u)))
                      + dot(avc[3], float4(float(hy & 0xFu), float(hy & 0xF0u), float(hy & 0xF00u), float(hy & 0xF000u)));
            code_sum += dot(avc[4], float4(float(qv.z & 0xFu), float(qv.z & 0xF0u), float(qv.z & 0xF00u), float(qv.z & 0xF000u)))
                      + dot(avc[5], float4(float(hz & 0xFu), float(hz & 0xF0u), float(hz & 0xF00u), float(hz & 0xF000u)));
            code_sum += dot(avc[6], float4(float(qv.w & 0xFu), float(qv.w & 0xF0u), float(qv.w & 0xF00u), float(qv.w & 0xF000u)))
                      + dot(avc[7], float4(float(hw & 0xFu), float(hw & 0xF0u), float(hw & 0xF00u), float(hw & 0xF000u)));

            acc[cc] += s * code_sum + lo * a_sum;
        }
    }

    const uint obase = (row * d.top_k + slot) * d.n;  // (row,slot,col)-major output
    if (MOE_GU_NSG == 1u) {
        for (uint cc = 0u; cc < COLS; ++cc) {
            const uint col = col0 + cc;
            const float total = simd_sum(acc[cc]);
            if (tiisg == 0u && col < d.n) c[obase + col] = total;
        }
    } else {
        // NSG simdgroups each hold a partial over their disjoint sub-block stride: simd_sum within
        // each → sgpart[sgitg][cc], then simdgroup 0 sums the NSG partials and writes.
        threadgroup float sgpart[8][8];   // up to 8 simdgroups × COLS(=8)
        for (uint cc = 0u; cc < COLS; ++cc) {
            const float part = simd_sum(acc[cc]);
            if (tiisg == 0u) sgpart[sgitg][cc] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0u && tiisg == 0u) {
            for (uint cc = 0u; cc < COLS; ++cc) {
                const uint col = col0 + cc;
                if (col >= d.n) continue;
                float total = 0.0f;
                for (uint g = 0u; g < MOE_GU_NSG; ++g) total += sgpart[g][cc];
                c[obase + col] = total;
            }
        }
    }
}

// ===================== down indirect GEMV + router-weight reduce (B-row) =====================
// {m=B, k(=mi), n(=h), top_k}. grid = (ceil(h/DCOLS), B, 1); Y = batch row. Output
// mlp_down[row*h + h_j]. silu segment a[row*top_k*mi + slot*mi ..]; ids/wts[row*top_k+slot].
//
// DCOLS = output h_j columns per simdgroup (nr0/NSG amortization). For a given (slot,
// sub-block) the silu activation av[8] + a_sum depend ONLY on the slot, NOT on h_j —
// so loading them ONCE and reusing across DCOLS weight columns (which differ only by
// h_j → wcol = expert*n + (h_j0+dc)) removes (DCOLS-1)/DCOLS of the silu reload traffic.
// Host grid x-divisor MUST match (concurrent_metal.rs down dispatch width: h.div_ceil(DCOLS)).
kernel void gemv_q4ks_id_down_b(
        device const float4 *a       [[buffer(0)]],  // [B, top_k, mi] silu_all
        device const uint4  *codes   [[buffer(1)]],
        device       float  *c       [[buffer(2)]],  // [B, h] mlp_down
        constant     IdDims &d       [[buffer(3)]],
        device const uint   *scales  [[buffer(4)]],
        device const uint   *ids      [[buffer(5)]], // [B, top_k]
        device const float  *wts     [[buffer(6)]],  // [B, top_k]
        device const uint   *mins    [[buffer(7)]],
        device const half2  *dd      [[buffer(8)]],
        uint3  tgpig                 [[threadgroup_position_in_grid]],
        ushort tiisg                 [[thread_index_in_simdgroup]]) {
    const uint subs = d.k / 32u;
    const uint slot_vecs = d.k / 4u;            // float4s per slot's silu segment (mi/4)
    const uint row_vecs = d.top_k * slot_vecs;  // float4s per row's silu (top_k*mi/4)
    const uint DCOLS = 8u;
    const uint h_j0 = tgpig.x * DCOLS;          // first output hidden index of this group
    const uint row = tgpig.y;                   // batch row
    if (h_j0 >= d.n || row >= d.m) return;

    const uint ibase = row * d.top_k;           // this row's ids/wts segment
    const uint rbase = row * row_vecs;          // this row's silu segment (float4 units)
    const uint supers = subs / 8u;              // super-blocks per weight row

    // Late-scale-fold nibble handling (see gemv_q4ks_id_b): keep nibbles masked-in-place and fold
    // the per-lane 16^-lane correction into the (once-loaded, DCOLS-reused) silu activation.
    const float4 CORR = float4(1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint slot = 0u; slot < d.top_k; ++slot) {
        const uint expert = ids[ibase + slot];
        const float rw = wts[ibase + slot];
        const uint exp_base = expert * d.n;
        const uint sbase = rbase + slot * slot_vecs;
        float slot_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint b = tiisg; b < subs; b += 32u) {
            const uint sblk = b / 8u;
            const uint byte_b = b & 3u;

            // Load THIS slot's silu sub-block + a_sum ONCE; reuse across DCOLS h_j cols.
            const uint abase = sbase + 8u * b;
            float4 av[8];
            for (short i = 0; i < 8; ++i) av[i] = a[abase + i];
            float a_sum = 0.0f;
            for (short i = 0; i < 8; ++i) a_sum += av[i].x + av[i].y + av[i].z + av[i].w;
            float4 avc[8];
            for (short i = 0; i < 8; ++i) avc[i] = av[i] * CORR;

            for (uint dc = 0u; dc < DCOLS; ++dc) {
                const uint h_j = h_j0 + dc;
                if (h_j >= d.n) break;
                const uint wcol = exp_base + h_j;
                const uint base = wcol * subs;
                const uint ddbase = wcol * supers;
                const float2 dp = float2(dd[ddbase + sblk]);
                const float dv  = dp.x;
                const float dmv = dp.y;
                const uint sword = scales[(base + b) >> 2];
                const uint mword = mins[(base + b) >> 2];
                const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
                const int  mraw = int(mword << (24u - 8u * byte_b)) >> 24;
                const float s  = dv * su8;
                const float lo = dmv * float(mraw);

                const uint4 qv = codes[base + b];
                float code_sum = 0.0f;
                uint hx = qv.x >> 16, hy = qv.y >> 16, hz = qv.z >> 16, hw = qv.w >> 16;
                code_sum += dot(avc[0], float4(float(qv.x & 0xFu), float(qv.x & 0xF0u), float(qv.x & 0xF00u), float(qv.x & 0xF000u)))
                          + dot(avc[1], float4(float(hx & 0xFu), float(hx & 0xF0u), float(hx & 0xF00u), float(hx & 0xF000u)));
                code_sum += dot(avc[2], float4(float(qv.y & 0xFu), float(qv.y & 0xF0u), float(qv.y & 0xF00u), float(qv.y & 0xF000u)))
                          + dot(avc[3], float4(float(hy & 0xFu), float(hy & 0xF0u), float(hy & 0xF00u), float(hy & 0xF000u)));
                code_sum += dot(avc[4], float4(float(qv.z & 0xFu), float(qv.z & 0xF0u), float(qv.z & 0xF00u), float(qv.z & 0xF000u)))
                          + dot(avc[5], float4(float(hz & 0xFu), float(hz & 0xF0u), float(hz & 0xF00u), float(hz & 0xF000u)));
                code_sum += dot(avc[6], float4(float(qv.w & 0xFu), float(qv.w & 0xF0u), float(qv.w & 0xF00u), float(qv.w & 0xF000u)))
                          + dot(avc[7], float4(float(hw & 0xFu), float(hw & 0xF0u), float(hw & 0xF00u), float(hw & 0xF000u)));

                slot_acc[dc] += s * code_sum + lo * a_sum;
            }
        }
        for (uint dc = 0u; dc < DCOLS; ++dc) acc[dc] += rw * slot_acc[dc];
    }

    const uint obase = row * d.n;
    for (uint dc = 0u; dc < DCOLS; ++dc) {
        const uint h_j = h_j0 + dc;
        const float total = simd_sum(acc[dc]);
        if (tiisg == 0u && h_j < d.n) c[obase + h_j] = total;
    }
}

// ===================== down indirect GEMV — PER-SLOT PARTIAL (B-row, top_k parallel) =====================
// Companion to gemv_q4ks_id_down_b: instead of LOOPING top_k slots serially inside one simdgroup,
// this kernel parallelizes the slots across grid-Z. grid = (ceil(h/DCOLS), B, top_k); each simdgroup
// computes ONE slot's weighted contribution (rw * dot) to DCOLS h-columns and writes it to a
// per-slot scratch partials[(row*top_k+slot)*h + h_j]. A tiny second reduce kernel (moe_down_reduce_b)
// then sums the top_k partials → mlp_down[row*h + h_j], IN SLOT ORDER, matching the serial loop's
// accumulation order → bit-close parity. {m=B, k(=mi), n(=h), top_k} (same IdDims as down_b).
kernel void gemv_q4ks_id_down_partial_b(
        device const float4 *a       [[buffer(0)]],  // [B, top_k, mi] silu_all
        device const uint4  *codes   [[buffer(1)]],
        device       float  *partials[[buffer(2)]],  // [B, top_k, h] per-slot weighted partials
        constant     IdDims &d       [[buffer(3)]],
        device const uint   *scales  [[buffer(4)]],
        device const uint   *ids      [[buffer(5)]], // [B, top_k]
        device const float  *wts     [[buffer(6)]],  // [B, top_k]
        device const uint   *mins    [[buffer(7)]],
        device const half2  *dd      [[buffer(8)]],
        uint3  tgpig                 [[threadgroup_position_in_grid]],
        ushort tiisg                 [[thread_index_in_simdgroup]]) {
    const uint subs = d.k / 32u;
    const uint slot_vecs = d.k / 4u;            // float4s per slot's silu segment (mi/4)
    const uint row_vecs = d.top_k * slot_vecs;  // float4s per row's silu (top_k*mi/4)
    const uint DCOLS = 8u;
    const uint h_j0 = tgpig.x * DCOLS;          // first output hidden index of this group
    const uint row  = tgpig.y;                  // batch row
    const uint slot = tgpig.z;                  // routed-expert slot (grid Z) — THE parallelism
    if (h_j0 >= d.n || row >= d.m || slot >= d.top_k) return;

    const uint ibase = row * d.top_k;           // this row's ids/wts segment
    const uint rbase = row * row_vecs;          // this row's silu segment (float4 units)
    const uint supers = subs / 8u;              // super-blocks per weight row

    const uint expert = ids[ibase + slot];
    const float rw = wts[ibase + slot];
    const uint exp_base = expert * d.n;
    const uint sbase = rbase + slot * slot_vecs;

    // Late-scale-fold nibble handling (see gemv_q4ks_id_b).
    const float4 CORR = float4(1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f);
    float slot_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint b = tiisg; b < subs; b += 32u) {
        const uint sblk = b / 8u;
        const uint byte_b = b & 3u;

        // Load THIS slot's silu sub-block + a_sum ONCE; reuse across DCOLS h_j cols.
        const uint abase = sbase + 8u * b;
        float4 av[8];
        for (short i = 0; i < 8; ++i) av[i] = a[abase + i];
        float a_sum = 0.0f;
        for (short i = 0; i < 8; ++i) a_sum += av[i].x + av[i].y + av[i].z + av[i].w;
        float4 avc[8];
        for (short i = 0; i < 8; ++i) avc[i] = av[i] * CORR;

        for (uint dc = 0u; dc < DCOLS; ++dc) {
            const uint h_j = h_j0 + dc;
            if (h_j >= d.n) break;
            const uint wcol = exp_base + h_j;
            const uint base = wcol * subs;
            const uint ddbase = wcol * supers;
            const float2 dp = float2(dd[ddbase + sblk]);
            const float dv  = dp.x;
            const float dmv = dp.y;
            const uint sword = scales[(base + b) >> 2];
            const uint mword = mins[(base + b) >> 2];
            const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
            const int  mraw = int(mword << (24u - 8u * byte_b)) >> 24;
            const float s  = dv * su8;
            const float lo = dmv * float(mraw);

            const uint4 qv = codes[base + b];
            float code_sum = 0.0f;
            uint hx = qv.x >> 16, hy = qv.y >> 16, hz = qv.z >> 16, hw = qv.w >> 16;
            code_sum += dot(avc[0], float4(float(qv.x & 0xFu), float(qv.x & 0xF0u), float(qv.x & 0xF00u), float(qv.x & 0xF000u)))
                      + dot(avc[1], float4(float(hx & 0xFu), float(hx & 0xF0u), float(hx & 0xF00u), float(hx & 0xF000u)));
            code_sum += dot(avc[2], float4(float(qv.y & 0xFu), float(qv.y & 0xF0u), float(qv.y & 0xF00u), float(qv.y & 0xF000u)))
                      + dot(avc[3], float4(float(hy & 0xFu), float(hy & 0xF0u), float(hy & 0xF00u), float(hy & 0xF000u)));
            code_sum += dot(avc[4], float4(float(qv.z & 0xFu), float(qv.z & 0xF0u), float(qv.z & 0xF00u), float(qv.z & 0xF000u)))
                      + dot(avc[5], float4(float(hz & 0xFu), float(hz & 0xF0u), float(hz & 0xF00u), float(hz & 0xF000u)));
            code_sum += dot(avc[6], float4(float(qv.w & 0xFu), float(qv.w & 0xF0u), float(qv.w & 0xF00u), float(qv.w & 0xF000u)))
                      + dot(avc[7], float4(float(hw & 0xFu), float(hw & 0xF0u), float(hw & 0xF00u), float(hw & 0xF000u)));

            slot_acc[dc] += s * code_sum + lo * a_sum;
        }
    }

    // Per-slot simd-reduced weighted partial → partials[(row*top_k+slot)*h + h_j].
    const uint obase = (row * d.top_k + slot) * d.n;
    for (uint dc = 0u; dc < DCOLS; ++dc) {
        const uint h_j = h_j0 + dc;
        const float total = simd_sum(slot_acc[dc]);
        if (tiisg == 0u && h_j < d.n) partials[obase + h_j] = rw * total;
    }
}

// ===================== down partial reduce (B-row) =====================
// Sums the top_k per-slot partials → mlp_down[row*h + h_j], IN SLOT ORDER (0..top_k) so the
// accumulation matches the serial gemv_q4ks_id_down_b's `acc += rw*slot_acc` order → bit-close.
// grid covers B*h (one thread per output element). dims: {m=B, k=_, n=h, top_k}.
kernel void moe_down_reduce_b(
        device const float *partials [[buffer(0)]],  // [B, top_k, h]
        device       float *out      [[buffer(1)]],  // [B, h]
        constant     IdDims &d       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = d.m * d.n;                  // B * h
    if (gid >= total) return;
    const uint row = gid / d.n;
    const uint h_j = gid - row * d.n;
    const uint base = row * d.top_k * d.n + h_j;   // partials[(row*top_k+0)*h + h_j]
    float acc = 0.0f;
    for (uint slot = 0u; slot < d.top_k; ++slot) acc += partials[base + slot * d.n];
    out[gid] = acc;
}

// ===================== route top-k softmax (B-row) =====================
// One threadgroup PER ROW (grid.x = B); each does the m=1 route over logits[row*num_experts..]
// → ids[row*top_k..] + wts[row*top_k..]. Byte-identical selection to moe_route_msl per row.
struct RouteDims { uint num_experts; uint top_k; uint norm_topk; uint _pad; };

constant uint RWG = 128u;
constant float R_NEG_INF = -3.4e38f;

kernel void moe_route_b(
        device const float *logits [[buffer(0)]],  // [B, num_experts]
        device       uint  *ids    [[buffer(1)]],  // [B, top_k]
        device       float *wts    [[buffer(2)]],  // [B, top_k]
        constant RouteDims &d      [[buffer(3)]],
        uint  row                  [[threadgroup_position_in_grid]],
        uint  lane                 [[thread_position_in_threadgroup]]) {
    const uint n = d.num_experts;
    const uint k = min(d.top_k, n);
    const uint lbase = row * n;                 // this row's logits segment
    const uint obase = row * d.top_k;           // this row's ids/wts segment

    threadgroup float red[128];
    threadgroup uint  redi[128];
    threadgroup float sh_max;
    threadgroup float sh_sum;
    threadgroup bool  taken[256];

    float m = R_NEG_INF;
    for (uint i = lane; i < n; i += RWG) m = max(m, logits[lbase + i]);
    red[lane] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = RWG/2u; stride > 0u; stride /= 2u) {
        if (lane < stride) red[lane] = max(red[lane], red[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) sh_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maxv = sh_max;

    float s = 0.0f;
    for (uint i = lane; i < n; i += RWG) s += exp(logits[lbase + i] - maxv);
    red[lane] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = RWG/2u; stride > 0u; stride /= 2u) {
        if (lane < stride) red[lane] += red[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) sh_sum = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sum = sh_sum;

    for (uint i = lane; i < n; i += RWG) taken[i] = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = 0u; slot < k; ++slot) {
        float bv = R_NEG_INF; uint bi = 0xFFFFFFFFu;
        for (uint i = lane; i < n; i += RWG) {
            if (!taken[i]) { float v = logits[lbase + i]; if (v > bv) { bv = v; bi = i; } }
        }
        red[lane] = bv; redi[lane] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = RWG/2u; stride > 0u; stride /= 2u) {
            if (lane < stride) {
                float ov = red[lane + stride]; uint oi = redi[lane + stride];
                float cv = red[lane]; uint ci = redi[lane];
                if (ov > cv || (ov == cv && oi < ci)) { red[lane] = ov; redi[lane] = oi; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0u) {
            uint best = redi[0];
            taken[best] = true;
            ids[obase + slot] = best;
            wts[obase + slot] = exp(logits[lbase + best] - maxv) / sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u && d.norm_topk != 0u) {
        float wsum = 0.0f;
        for (uint slot = 0u; slot < k; ++slot) wsum += wts[obase + slot];
        if (wsum > 0.0f) for (uint slot = 0u; slot < k; ++slot) wts[obase + slot] /= wsum;
    }
}

// ===================== swiglu (B-row) =====================
// Elementwise silu(gate)*up over B*top_k*inter. Identical to moe_swiglu; the n in dims carries
// the FULL B*top_k*inter count. Bit-exact to the m=1 swiglu per element.
struct SwDims { uint n; uint _p0; uint _p1; uint _p2; };

kernel void moe_swiglu_b(
        device const float *gate [[buffer(0)]],  // [B*top_k*inter]
        device const float *up   [[buffer(1)]],
        device       float *out  [[buffer(2)]],
        constant SwDims    &d    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    const float g = gate[gid];
    out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

// ===================== expert->rows CSR scatter (llama map0) =====================
// MSL port of oldgrouped/moe_expert_csr.wgsl — THE scatter. Builds, from the batched route
// ids[B*top_k], the expert-grouped CSR consumed by the grouped tiled mm_id kernels:
//   offsets[ne+1]       : exclusive prefix-sum of per-expert counts (INDEXED BY EXPERT ID):
//                         lo = offsets[expert], hi = offsets[expert+1].
//   row_slot[B*top_k]   : CSR values; entry packs i = row*top_k+slot in ASCENDING source-index
//                         order within each expert (the determinism invariant the slot-reduce
//                         relies on). Recover row = i/top_k, slot = i%top_k.
//   active_experts[ne]  : compacted list of experts with >=1 row (first n_active valid).
//   csr_meta[0]         : n_active.
//   csr_meta[1]         : n_tiles (the EXACT total row-tiles = Σ_active ceil(count_e / MM_BM)).
//   tile_list[n_tiles]  : the PRECISE work list (llama's mul_mat_id "tokens per expert" → exact
//                         tiles). Each entry packs (expert_id << 16) | tile_idx, where tile_idx is
//                         the 0-based row-tile within that expert (tile_row0 = tile_idx*MM_BM).
//                         The grouped GEMM dispatches grid.x = WORST_CASE_TILES and reads its
//                         (expert, tile_row0) from tile_list[grid.x] instead of over-dispatching one
//                         row-tile lane PER active expert (the 8192→~192 threadgroup cut at conc64).
// Single-threadgroup, ONE lane (B*top_k <= 512 → the sequential build is ~us). dims: word0 =
// B*top_k (n entries), word1 = ne (num_experts). Invariant: ids[i] < ne <= 256 (enforced by
// moe_route_b) — counts[256] has no OOB guard, like the WGSL source.
struct CsrDims { uint n; uint ne; uint _p0; uint _p1; };

// CSR row-tile height = the llama NR1 tile. MUST EQUAL NR1 in moe_mm_id_q4ks.metal, because the
// GEMM decodes tile_idx back as `r1 = tile_idx*NR1`: if the CSR emits ceil(count/32) tiles but the
// GEMM reads tile_idx*8, rows are mis-addressed (wrong coverage). The host drives BOTH from the same
// `MOE_MM_NR1` #define (ARF_MOE_NR1_8 prepends it to this file AND moe_mm_id_q4ks.metal).
#ifndef MOE_MM_NR1
#define MOE_MM_NR1 32
#endif
constant constexpr uint CSR_MM_BM = (uint)MOE_MM_NR1;  // == NR1

kernel void moe_expert_csr_msl(
        device const uint *ids            [[buffer(0)]],  // [B*top_k]
        device       uint *offsets        [[buffer(1)]],  // [ne+1]
        device       uint *row_slot       [[buffer(2)]],  // [B*top_k]
        device       uint *active_experts [[buffer(3)]],  // [ne]
        device       uint *csr_meta       [[buffer(4)]],  // [0]=n_active, [1]=n_tiles
        constant     CsrDims &d           [[buffer(5)]],
        device       uint *tile_list      [[buffer(6)]]) {  // [max_tiles] packed (expert<<16)|tile_idx
    const uint n = d.n;
    const uint ne = d.ne;
    threadgroup uint counts[256];
    // 1) zero counts.
    for (uint e = 0u; e < ne; ++e) { counts[e] = 0u; }
    // 2) histogram.
    for (uint i = 0u; i < n; ++i) { const uint eid = ids[i]; counts[eid] = counts[eid] + 1u; }
    // 3) exclusive prefix-sum -> offsets[0..=ne]; compact active list; AND build the exact tile-list.
    uint run = 0u;
    uint na = 0u;
    uint nt = 0u;
    for (uint e = 0u; e < ne; ++e) {
        offsets[e] = run;
        const uint ce = counts[e];
        if (ce > 0u) {
            active_experts[na] = e; na = na + 1u;
            // emit ceil(ce / MM_BM) tiles for expert e, packing (e<<16)|tile_idx.
            const uint et = (ce + CSR_MM_BM - 1u) / CSR_MM_BM;
            for (uint t = 0u; t < et; ++t) { tile_list[nt] = (e << 16) | t; nt = nt + 1u; }
        }
        run = run + ce;
    }
    offsets[ne] = run;     // == n
    csr_meta[0] = na;
    csr_meta[1] = nt;      // n_tiles — the exact dispatch count the GEMM early-outs against
    // 4) scatter in ASCENDING source index i -> stable per-expert order. Reuse counts[] as a
    //    per-expert running cursor (reset to 0).
    for (uint e = 0u; e < ne; ++e) { counts[e] = 0u; }
    for (uint i = 0u; i < n; ++i) {
        const uint e = ids[i];
        const uint pos = offsets[e] + counts[e];
        row_slot[pos] = i;     // i = row*top_k + slot
        counts[e] = counts[e] + 1u;
    }
}

// ===================== slot-reduce + router-weight fold (B-row) =====================
// Sums the top_k per-(row,slot) down_slots contributions IN SLOT ORDER (0..top_k) and folds the
// router weight wts[row*top_k+slot]:
//   mlp_down[row*h + h_j] = Σ_slot wts[row*top_k+slot] * down_slots[(row*top_k+slot)*h + h_j].
// Slot-order accumulation == the serial gemv_q4ks_id_down_b's `acc += rw*slot_acc` order → bit-
// close parity. Companion to the grouped down GEMM (which PLACES per-(row,slot) un-weighted
// contributions into down_slots, router-weight-agnostic). One thread per (row, h_j).
// dims: {m=B, k=_, n=h, top_k} (IdDims). down_slots layout [B, top_k, h].
kernel void moe_reduce_slots_b(
        device const float *down_slots [[buffer(0)]],  // [B, top_k, h]
        device const float *wts        [[buffer(1)]],  // [B, top_k]
        device       float *out        [[buffer(2)]],  // [B, h]
        constant     IdDims &d         [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = d.m * d.n;                   // B * h
    if (gid >= total) return;
    const uint row = gid / d.n;
    const uint h_j = gid - row * d.n;
    const uint base = row * d.top_k * d.n + h_j;    // down_slots[(row*top_k+0)*h + h_j]
    const uint wbase = row * d.top_k;
    float acc = 0.0f;
    for (uint slot = 0u; slot < d.top_k; ++slot) {
        acc += wts[wbase + slot] * down_slots[base + slot * d.n];
    }
    out[gid] = acc;
}
