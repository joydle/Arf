// B-ROW (batched) Q4_K_S decode GEMV for the native Metal island megakernel — the B-row
// generalization of gemv_q4ks (matmul_vec_q4ks_msl.metal). Computes
//   c[r,col] = Σ_subblocks ( (d·u8scale)·Σ(nibble·a[r]) + (dmin·i8min)·Σ(a[r]) )
// for ALL B rows, amortizing the per-sub-block weight UNPACK across the B activation rows
// (the batch CSE — identical to matmul_vec_q4ks_batch.wgsl). One simdgroup per output column;
// the 32 lanes stride the sub-blocks; each lane unpacks a sub-block ONCE then dots it against
// every row's matching activation slice. simd_sum reduces each row's partial (no threadgroup
// tree). Bit-faithful to the m==1 gemv_q4ks math (same dequant, same accumulation order over
// sub-blocks) — at B=1 it is byte-identical to gemv_q4ks.
//
// Activation layout: a is ROW-MAJOR [B, k] (row r at r*k). Output c is ROW-MAJOR [B, n]
// (row r at r*n). Weights (codes/scales/mins/dd) are the SAME bytes the m==1 kernel reads.
//
// Bindings: 0 a[B,k](f32x4), 1 codes(uint4/sub-block), 2 c[B,n], 3 dims{m=B,k,n,_},
// 4 scales(u8×4/word), 5 mins(i8×4/word), 6 dd(half2 {d,dmin}/super-block).

#include <metal_stdlib>
using namespace metal;

struct Dims { uint m; uint k; uint n; uint _pad; };

// Max batch rows held in per-lane registers (acc[MAXB]). MUST be >= the largest `d.m` this kernel
// is ever dispatched with, or rows [MAXB, d.m) are SILENTLY DROPPED (never written) — the conc64
// correctness bug (GH #21): batch-mega dispatches k/v/router GEMV with d.m = b = 64, but MAXB was 32
// → rows 32-63 garbage. Raised to 64 to cover conc64. Host must keep conc <= MAXB or chunk. The
// sub-block unpack (the dominant register cost) is shared across rows, so more accumulators is cheap.
#ifndef MAXB_Q4KS_BATCH
#define MAXB_Q4KS_BATCH 64u
#endif
constant constexpr uint MAXB = MAXB_Q4KS_BATCH;

// L196 — PROBE (function_constant 4): cap the accumulator array below MAXB. `float acc[64]` is
// declared regardless of the actual `b`, so if the compiler allocates all 64 slots that is 256 B
// of registers per lane — enough to cut occupancy and explain part of the b-penalty (L195:
// activation traffic predicts 2.78x at b=3, measured 2.85x, so most is explained, but register
// pressure would show up as an ADDITIONAL fixed cost even at b=1).
//
// ⚠️ CORRECTNESS: rows >= ACC_CAP are NEVER WRITTEN. Only safe for an A/B where b <= ACC_CAP.
// Unset (default) = 64 = the shipped path, byte-identical.
constant uint ACC_CAP_FC [[function_constant(4)]];
constant uint ACC_CAP = is_function_constant_defined(ACC_CAP_FC) ? ACC_CAP_FC : MAXB;

// L167 — ARF_GEMV_B_NSG (function_constant 3): simdgroups per threadgroup, mirroring the lever
// the m=1 `gemv_q4ks` already has (matmul_vec_q4ks_msl.metal:43). Default 1 = the shipped path,
// byte-identical. NSG=2 (64 threads) splits this column's sub-block weight stream across TWO
// simdgroups on DISJOINT strides, so the threadgroup issues ~2x the memory requests in flight.
//
// WHY IT MATTERS HERE AND NOT AT conc16: the earlier N_SG measurement (the decisions log, pre-OSS,
// removed 2026-08-28: "wash, +1% at
// conc16") was taken where the GEMV is already weight-stream saturated across many rows. At B=1 —
// a single chat, which is the whole daemon path for a hybrid model — one 32-lane simdgroup streams
// an entire 17408-wide `down` row alone, and the note records `down` ALONE at -25% there. The dense
// FFN (gate/up/down) is 70% of this model's per-token byte traffic, so that is where the time is.
// L220 — COLS_B (function_constant 5): output columns per threadgroup.
//
// PROVEN BY BYTES (L219/L220). Per sub-block this kernel reads 28 B of weight (codes+scale+min+dd,
// shared across rows) and 128 B of ACTIVATIONS PER ROW. Across n=17408 columns that is 78 MB of
// weights against 1070 MB of activations at b=3 — activations are 93% of all traffic. The
// activation slab is only 60 KB; it is re-read ONCE PER COLUMN, i.e. 17408x.
//
// That re-read IS the b-penalty: activations scale with b, weights do not, so the marginal row
// costs 0.96 of a full step (L218) instead of ~0. With COLS_B>1 one threadgroup computes COLS_B
// columns from ONE activation load, dividing the dominant term by COLS_B.
//
// ⚠️ L197 implemented this with `float4 av[MAXB*8]` — 512 dynamically-indexed slots that SPILLED
// and cost 34% even at COLS=1 (found by L200's bisect). This version keeps the activations in
// NAMED registers (a0..a7, statically indexed) and puts the column loop INSIDE the row loop, so
// each row's 8 float4 are loaded once and dotted against every column. No array, no spill.
constant uint COLS_B_FC [[function_constant(5)]];
constant uint COLS_B = is_function_constant_defined(COLS_B_FC) ? COLS_B_FC : 1u;

// L249 — ACC_ROWS: the per-column ROW STRIDE of `acc`, and hence its real register cost
// (COLS_B * ACC_ROWS floats per lane). Defaults to MAXB so the shipped shape is unchanged.
//
// WHY IT UNLOCKS COLS=8. `acc` is declared MAXB(64)-wide because the PREFILL path can push
// b up to 64. A decode or verify record never exceeds --max-batch-size (4 here), so 60 of
// those 64 slots per column are dead registers. At COLS=4 that is already 1 KB/lane; COLS=8
// would be 2 KB and spill — which is why L197's staging array cost 34%.
//
// With ACC_ROWS sized to the real batch, COLS=8 costs 8*8 = 64 floats = 256 B/lane — the SAME
// as COLS=4 does today. The register wall was never COLS; it was the unused stride.
//
// ⚠️ CORRECTNESS: rows >= ACC_ROWS are NEVER WRITTEN. The host MUST compile this >= the largest
// `d.m` it will dispatch, or it silently drops rows (the L142 failure mode).
// L363s — COLMAJOR (function constant 7): swap the inner nesting to columns-outer / rows-inner,
// so each column's 32-nibble unpack (24 float4 constructions) happens ONCE per column instead of
// once per column PER ROW. Trades 24 ALU ops per row for 8 float4 activation reloads per column.
// Register-neutral — NOT L197's staging array, which spilled 512 dynamically-indexed slots.
// At b=1 the two orders do identical work; the win, if any, is at b>=2. Default OFF: unset, the
// shipped row-outer path is byte-identical.
// uint, not bool: the host passes every function constant through `compile_msl_uint_fc`, which
// sets them as MTLDataTypeUInt. A bool declaration here fails the compile with "Constant
// COLMAJOR_FC (7) is of type MTLDataTypeBool but value found has type MTLDataTypeUInt" — and the
// island then REFUSES to serve rather than falling back (which is the guard working).
constant uint COLMAJOR_FC [[function_constant(7)]];
constant bool COLMAJOR = is_function_constant_defined(COLMAJOR_FC) ? (COLMAJOR_FC != 0u) : false;

constant uint ACC_ROWS_FC [[function_constant(6)]];
constant uint ACC_ROWS = is_function_constant_defined(ACC_ROWS_FC) ? ACC_ROWS_FC : MAXB;

constant uint NSGB_FC [[function_constant(3)]];
constant uint NSGB = is_function_constant_defined(NSGB_FC) ? NSGB_FC : 1u;

kernel void gemv_q4ks_batch(
        device const float4 *a       [[buffer(0)]],   // [B, k] row-major (k/4 vec4 per row)
        device const uint4  *codes   [[buffer(1)]],
        device       float  *c       [[buffer(2)]],   // [B, n] row-major
        constant     Dims   &d       [[buffer(3)]],
        device const uint   *scales  [[buffer(4)]],
        device const uint   *mins    [[buffer(5)]],
        device const half2  *dd      [[buffer(6)]],
        uint  tgpig                  [[threadgroup_position_in_grid]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]]) {
    const uint subs = d.k / 32u;
    const uint arow = d.k / 4u;            // float4 stride between rows in `a`
    const uint col0 = tgpig * COLS_B;      // this threadgroup owns COLS_B consecutive columns
    if (col0 >= d.n) return;
    const uint B = min(d.m, MAXB);

    // per-row accumulator (this lane's partial over its strided sub-blocks)
    // [column][row] accumulators. COLS_B is a compile-time constant so the outer index unrolls;
    // ACC_ROWS caps the row stride; COLS_B <= 8 and the loops below are bounded by BR.
    float acc[MAXB * 8u];   // declared for the worst case; ACC_ROWS bounds what is INDEXED
    // ⚠️ BR MUST be clamped by ACC_ROWS as well: acc is indexed [cc*ACC_ROWS + r], so a row
    // beyond the stride would write into the NEXT column's slots — silent cross-column
    // corruption, not an out-of-bounds fault.
    const uint BR = min(min(B, ACC_CAP), ACC_ROWS);
    for (uint cc = 0u; cc < COLS_B; ++cc)
        for (uint r = 0u; r < BR; ++r) acc[cc * ACC_ROWS + r] = 0.0f;

    // NSGB=1: this simdgroup strides every sub-block (b += 32) — the shipped path.
    // NSGB=2: each simdgroup takes a disjoint stride so the 64 lanes double the requests in flight.
    if (COLMAJOR) {
        // L363s — COLUMNS OUTER, ROWS INNER. Each (column, sub-block) is unpacked once into named
        // registers; every row then dots against it. Same arithmetic, same acc[] layout, same
        // results to the bit ordering of the fma chain — only the nesting differs.
        for (uint b = sgitg * 32u + tiisg; b < subs; b += NSGB * 32u) {
            const uint sblk   = b / 8u;
            const uint byte_b = b & 3u;
            const uint avbase = 8u * b;

            for (uint cc = 0u; cc < COLS_B; ++cc) {
                const uint col = col0 + cc;
                if (col >= d.n) break;
                const uint cbase = col * subs;
                const uint sbase = col * subs;
                const uint ddbase = col * (subs / 8u);

                const uint4 w = codes[cbase + b];
                const float4 nlo0 = float4(float(w.x & 0xFu), float((w.x>>4)&0xFu), float((w.x>>8)&0xFu), float((w.x>>12)&0xFu));
                const float4 nhi0 = float4(float((w.x>>16)&0xFu), float((w.x>>20)&0xFu), float((w.x>>24)&0xFu), float((w.x>>28)&0xFu));
                const float4 nlo1 = float4(float(w.y & 0xFu), float((w.y>>4)&0xFu), float((w.y>>8)&0xFu), float((w.y>>12)&0xFu));
                const float4 nhi1 = float4(float((w.y>>16)&0xFu), float((w.y>>20)&0xFu), float((w.y>>24)&0xFu), float((w.y>>28)&0xFu));
                const float4 nlo2 = float4(float(w.z & 0xFu), float((w.z>>4)&0xFu), float((w.z>>8)&0xFu), float((w.z>>12)&0xFu));
                const float4 nhi2 = float4(float((w.z>>16)&0xFu), float((w.z>>20)&0xFu), float((w.z>>24)&0xFu), float((w.z>>28)&0xFu));
                const float4 nlo3 = float4(float(w.w & 0xFu), float((w.w>>4)&0xFu), float((w.w>>8)&0xFu), float((w.w>>12)&0xFu));
                const float4 nhi3 = float4(float((w.w>>16)&0xFu), float((w.w>>20)&0xFu), float((w.w>>24)&0xFu), float((w.w>>28)&0xFu));

                const float2 dp = float2(dd[ddbase + sblk]);
                const uint sword = scales[(sbase + b) >> 2];
                const uint mword = mins[(sbase + b) >> 2];
                const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
                const int  mraw = int(mword << (24u - 8u * byte_b)) >> 24;
                const float s  = dp.x * su8;
                const float lo = dp.y * float(mraw);

                for (uint r = 0u; r < BR; ++r) {
                    const uint ab = r * arow + avbase;
                    const float4 a0 = a[ab];      const float4 a1 = a[ab + 1u];
                    const float4 a2 = a[ab + 2u]; const float4 a3 = a[ab + 3u];
                    const float4 a4 = a[ab + 4u]; const float4 a5 = a[ab + 5u];
                    const float4 a6 = a[ab + 6u]; const float4 a7 = a[ab + 7u];
                    const float a_sum =
                          a0.x+a0.y+a0.z+a0.w + a1.x+a1.y+a1.z+a1.w
                        + a2.x+a2.y+a2.z+a2.w + a3.x+a3.y+a3.z+a3.w
                        + a4.x+a4.y+a4.z+a4.w + a5.x+a5.y+a5.z+a5.w
                        + a6.x+a6.y+a6.z+a6.w + a7.x+a7.y+a7.z+a7.w;
                    const float code_sum =
                          dot(a0, nlo0) + dot(a1, nhi0) + dot(a2, nlo1) + dot(a3, nhi1)
                        + dot(a4, nlo2) + dot(a5, nhi2) + dot(a6, nlo3) + dot(a7, nhi3);
                    acc[cc * ACC_ROWS + r] += s * code_sum + lo * a_sum;
                }
            }
        }
    } else {
        for (uint b = sgitg * 32u + tiisg; b < subs; b += NSGB * 32u) {
            const uint sblk   = b / 8u;
            const uint byte_b = b & 3u;
            const uint avbase = 8u * b;

            // ROW-OUTER / COLUMN-INNER. Each row's 8 float4 of activations are loaded ONCE into
            // NAMED registers and dotted against all COLS_B columns' weights. That is what divides
            // the activation traffic by COLS_B without the staging array that spilled in L197.
            for (uint r = 0u; r < BR; ++r) {
                const uint ab = r * arow + avbase;
                const float4 a0 = a[ab];      const float4 a1 = a[ab + 1u];
                const float4 a2 = a[ab + 2u]; const float4 a3 = a[ab + 3u];
                const float4 a4 = a[ab + 4u]; const float4 a5 = a[ab + 5u];
                const float4 a6 = a[ab + 6u]; const float4 a7 = a[ab + 7u];
                const float a_sum =
                      a0.x+a0.y+a0.z+a0.w + a1.x+a1.y+a1.z+a1.w
                    + a2.x+a2.y+a2.z+a2.w + a3.x+a3.y+a3.z+a3.w
                    + a4.x+a4.y+a4.z+a4.w + a5.x+a5.y+a5.z+a5.w
                    + a6.x+a6.y+a6.z+a6.w + a7.x+a7.y+a7.z+a7.w;

                for (uint cc = 0u; cc < COLS_B; ++cc) {
                    const uint col = col0 + cc;
                    if (col >= d.n) break;
                    // L357 — NOTE FOR THE LOOP-ORDER QUESTION. The weight load + 32-nibble unpack
                    // below sits INSIDE the `r` (row) loop, so at b>1 the same column's weights are
                    // re-unpacked once per row: 24 float4 constructions of pure ALU that the compiler
                    // cannot hoist out of a loop they are written inside. Activations are shared
                    // across columns here (a0..a7 in named registers — that is what COLS_B divides);
                    // weights are NOT shared across rows.
                    //
                    // Swapping the nesting (columns outer, weights unpacked once, rows inner) trades
                    // 24 ALU ops per row for 8 float4 reloads per column, and is register-neutral —
                    // NOT L197's failed staging array, which spilled 512 dynamically-indexed slots.
                    // Measured as `gemv_q4ks_batch_colmajor` (function constant 7), L357: 19% slower,
                    // kept off.
                    const uint cbase = col * subs;
                    const uint sbase = col * subs;
                    const uint ddbase = col * (subs / 8u);

                    // unpack this (column, sub-block)'s 32 nibbles
                    const uint4 w = codes[cbase + b];
                    const float4 nlo0 = float4(float(w.x & 0xFu), float((w.x>>4)&0xFu), float((w.x>>8)&0xFu), float((w.x>>12)&0xFu));
                    const float4 nhi0 = float4(float((w.x>>16)&0xFu), float((w.x>>20)&0xFu), float((w.x>>24)&0xFu), float((w.x>>28)&0xFu));
                    const float4 nlo1 = float4(float(w.y & 0xFu), float((w.y>>4)&0xFu), float((w.y>>8)&0xFu), float((w.y>>12)&0xFu));
                    const float4 nhi1 = float4(float((w.y>>16)&0xFu), float((w.y>>20)&0xFu), float((w.y>>24)&0xFu), float((w.y>>28)&0xFu));
                    const float4 nlo2 = float4(float(w.z & 0xFu), float((w.z>>4)&0xFu), float((w.z>>8)&0xFu), float((w.z>>12)&0xFu));
                    const float4 nhi2 = float4(float((w.z>>16)&0xFu), float((w.z>>20)&0xFu), float((w.z>>24)&0xFu), float((w.z>>28)&0xFu));
                    const float4 nlo3 = float4(float(w.w & 0xFu), float((w.w>>4)&0xFu), float((w.w>>8)&0xFu), float((w.w>>12)&0xFu));
                    const float4 nhi3 = float4(float((w.w>>16)&0xFu), float((w.w>>20)&0xFu), float((w.w>>24)&0xFu), float((w.w>>28)&0xFu));

                    const float2 dp = float2(dd[ddbase + sblk]);
                    const uint sword = scales[(sbase + b) >> 2];
                    const uint mword = mins[(sbase + b) >> 2];
                    const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
                    const int  mraw = int(mword << (24u - 8u * byte_b)) >> 24;
                    const float s  = dp.x * su8;
                    const float lo = dp.y * float(mraw);

                    const float code_sum =
                          dot(a0, nlo0) + dot(a1, nhi0) + dot(a2, nlo1) + dot(a3, nhi1)
                        + dot(a4, nlo2) + dot(a5, nhi2) + dot(a6, nlo3) + dot(a7, nhi3);
                    acc[cc * ACC_ROWS + r] += s * code_sum + lo * a_sum;
                }
            }
        }

    }

    // simd_sum each (column, row) partial; lane 0 writes [r, col].
    if (NSGB == 1u) {
        for (uint cc = 0u; cc < COLS_B; ++cc) {
            const uint col = col0 + cc;
            if (col >= d.n) break;
            for (uint r = 0u; r < BR; ++r) {
                const float total = simd_sum(acc[cc * ACC_ROWS + r]);
                if (tiisg == 0u) c[r * d.n + col] = total;
            }
        }
    } else {
        // NSGB simdgroups each hold a partial over a disjoint sub-block stride. Reduce within each
        // simdgroup, stage to threadgroup memory, then simdgroup 0 sums and writes. One column at
        // a time so `sgpart` stays [8][MAXB] rather than [8][MAXB*COLS_B].
        threadgroup float sgpart[8][MAXB];
        for (uint cc = 0u; cc < COLS_B; ++cc) {
            const uint col = col0 + cc;
            if (col >= d.n) break;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0u; r < BR; ++r) {
                const float part = simd_sum(acc[cc * ACC_ROWS + r]);
                if (tiisg == 0u) sgpart[sgitg][r] = part;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgitg == 0u && tiisg == 0u) {
                for (uint r = 0u; r < BR; ++r) {
                    float total = 0.0f;
                    for (uint g = 0u; g < NSGB; ++g) total += sgpart[g][r];
                    c[r * d.n + col] = total;
                }
            }
        }
    }
}
