// TILED Q8 simdgroup-GEMM for the batched lm_head — the ALU-optimal twin of gemv_q8_b.
//
// THE LEVER (measured): the shipped batched lm_head (gemv_q8_b, decode_ops_msl.metal) already
// streams the 311 MB Q8 vocab matrix ONCE (one dispatch, reused across all B rows via the
// acc[MAXB_Q8=64] register loop) — so it is NOT bandwidth-bound. It is ALU-bound: 64 SCALAR
// FMAs per column per lane (the acc[r] += ar[0]*w0 + ... loop). At conc64 that is the 37ms
// lm_head. This kernel does the SAME B×vocab matmul but feeds the (dequantized) weight tile +
// the activation tile to the simdgroup MATRIX units (simdgroup_float8x8 × multiply_accumulate),
// the M-series' highest-throughput FMA path → target lm_head 37 → ~5-10ms.
//
// STRUCTURE: clones matmul_mm_q4ks.metal (the SHIPPED tiled dense GEMM) — 32-row × 64-col tile,
// 128 threads = 4 simdgroups, simdgroup_float8x8 matrices, KC=32 K-chunks, col-major weight tile
// loaded with transpose=true. Only the DEQUANT differs: Q8 is per-ROW int8 (4 int8/u32, k/4
// words/row) with one f32 scale PER OUTPUT ROW (= per vocab column here). NO super-blocks, no
// mins — much simpler than Q4_K_S.
//
// SCALE PLACEMENT (parity with gemv_q8 / gemv_q8_b): both stream `acc += a·int8` and apply
//   c[col] = acc * scale[col]   (scale-AFTER-sum, once per output col).
// We do the SAME here: the staged weight tile holds the RAW int8 as float (NO scale folded), the
// matmul accumulates Σ_k a·int8 in f32, and scale[col] is multiplied at the SCATTER writeback.
// → identical math ordering to the GEMV oracle modulo the simdgroup f32 reassociation (≤1e-3
//   drift, greedy-token-safe). FLOAT8x8 (full f32 tile, like matmul_mm_q4ks) keeps the int8 EXACT
//   (no half rounding of the weight) — the safe-parity choice; HALF would round the int8 magnitudes.
//
// MATH:
//   logits[row, col] = scale[col] · Σ_k normed[row,k] · int8(W[col,k])
//   int8(W[col, 4w+j]) = (int)(qword << (24-8j)) >> 24   (sign-extend byte j of word w)
//
// BINDINGS (match gemv_q8_b so the host seam is a drop-in swap):
//   0 a[B,k] float row-major (normed)   1 q (u32-packed int8, [vocab,k] = [n,k])
//   2 c[B,n] float row-major (logits)   3 dims{m=B,k,n,_}   4 scale (f32 per output col / weight row)
//
// GRID (host): threadgroups = (ceil(B/BM), ceil(n/BN), 1), threads = 128 (4 simdgroups).
//   BM = 32 token rows, BN = 64 output cols. THREADGROUP MEM = 12288 B:
//     sA : BM(32) × KC(32) f32 = 4096 B   (activation tile, row-major)
//     sW : BN(64) × KC(32) f32 = 8192 B   (raw-int8-as-f32 weight tile, COL-major within tile)
//   At B=64 the row dim needs TWO BM=32 row-tiles → grid.x = ceil(64/32) = 2 (FULL tiles, no
//   under-fill — unlike MoE's 4-rows/expert). K iterated in KC=32 chunks; vocab tiled by BN=64.

#include <metal_stdlib>
using namespace metal;

struct Q8MmDims { uint m; uint k; uint n; uint _pad; };

constant constexpr uint BM = 32u;   // token rows per threadgroup tile
constant constexpr uint BN = 64u;   // output cols per threadgroup tile
constant constexpr uint KC = 32u;   // K-chunk (KC % 8 == 0); 8 int32 words = 32 int8
constant constexpr uint NSG = 4u;   // simdgroups per threadgroup (128 threads / 32)
constant constexpr uint SG_N = BN / NSG;   // = 16 cols per simdgroup
constant constexpr uint RFRAG = BM / 8u;   // = 4 row fragments (8 rows each)
constant constexpr uint CFRAG = SG_N / 8u; // = 2 col fragments (8 cols each)
constant constexpr uint KFRAG = KC / 8u;   // = 4 contraction fragments

kernel void matmul_mm_q8(
        device const float    *a     [[buffer(0)]],   // [B, k] row-major (normed)
        device const uint     *q     [[buffer(1)]],   // int8 weights [n, k] packed 4/u32
        device       float    *c     [[buffer(2)]],   // [B, n] row-major (logits)
        constant     Q8MmDims &d     [[buffer(3)]],   // {m=B, k, n, _}
        device const float    *scale [[buffer(4)]],   // f32 per output col (weight row)
        uint2  tgpig                 [[threadgroup_position_in_grid]],
        uint   tidx                  [[thread_index_in_threadgroup]],
        uint   sgid                  [[simdgroup_index_in_threadgroup]],
        threadgroup float *smem      [[threadgroup(0)]]) {
    const uint words = d.k / 4u;          // u32 words per weight row (4 int8 each)

    const uint row0 = tgpig.x * BM;       // first token row of this tile
    const uint col0 = tgpig.y * BN;       // first output col of this tile

    threadgroup float *sA = smem;                 // [BM][KC] activation (row-major)
    threadgroup float *sW = smem + (BM * KC);     // [BN][KC] weight (col-major within tile)

    // per-simdgroup accumulator fragments: RFRAG × CFRAG (4 × 2 = 8 fragments of 8×8). float-acc.
    simdgroup_float8x8 acc[RFRAG][CFRAG];
    for (uint i = 0; i < RFRAG; ++i)
        for (uint j = 0; j < CFRAG; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint kchunks = d.k / KC;
    const uint col_sg0 = col0 + sgid * SG_N;   // this simdgroup's first output column

    for (uint kc = 0u; kc < kchunks; ++kc) {
        const uint kbase = kc * KC;            // first K element of this chunk
        const uint wbase = kbase / 4u;         // first u32 word of this chunk (KC/4 = 8 words)

        // ---- stage weight tile sW[col_local*KC + kk] = int8(W[col0+col_local, kbase+kk]) ----
        // RAW int8 as float (NO scale folded — applied at writeback like gemv_q8). 128 threads
        // fill BN*KC = 2048 floats → 16 each.
        for (uint e = tidx; e < BN * KC; e += 128u) {
            const uint cl = e / KC;       // local column 0..63
            const uint kk = e % KC;       // element 0..31
            const uint col = col0 + cl;
            float wv = 0.0f;
            if (col < d.n) {
                const uint w = wbase + (kk >> 2);            // word holding K element kbase+kk
                const uint qv = q[col * words + w];
                const uint byte_b = kk & 3u;                 // which int8 within the word
                wv = float(int(qv << (24u - 8u * byte_b)) >> 24);
            }
            sW[cl * KC + kk] = wv;
        }
        // ---- stage activation tile sA[row_local*KC + kk] = a[row0+row_local, kbase+kk] ----
        // BM*KC = 1024 floats → 8 each.
        for (uint e = tidx; e < BM * KC; e += 128u) {
            const uint rl = e / KC;       // local row 0..31
            const uint kk = e % KC;       // 0..31
            const uint row = row0 + rl;
            float av = 0.0f;
            if (row < d.m) {
                av = a[row * d.k + kbase + kk];
            }
            sA[rl * KC + kk] = av;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- simdgroup matmul: acc[ri][cj] += A_frag(ri,kf) * W_frag(kf,cj) ----
        // A_frag: sA is [row][k] row-major (stride KC). W_frag: sW is [col][k] col-major (stride
        // KC); load with transpose=true → the (k × col) fragment the matmul needs.
        for (uint kf = 0u; kf < KFRAG; ++kf) {
            simdgroup_float8x8 af[RFRAG];
            for (uint ri = 0; ri < RFRAG; ++ri) {
                simdgroup_load(af[ri], sA + (ri * 8u) * KC + kf * 8u, KC);
            }
            for (uint cj = 0u; cj < CFRAG; ++cj) {
                const uint wcol_local = sgid * SG_N + cj * 8u;   // local col within tile
                simdgroup_float8x8 wf;
                simdgroup_load(wf, sW + wcol_local * KC + kf * 8u, KC, ulong2(0, 0), /*transpose=*/true);
                for (uint ri = 0; ri < RFRAG; ++ri) {
                    simdgroup_multiply_accumulate(acc[ri][cj], af[ri], wf, acc[ri][cj]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- store fragments to threadgroup, then scatter to c[B,n] with per-col scale (bounds-checked) ----
    threadgroup float *store = smem;   // reuse smem after the contraction loop
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ri = 0; ri < RFRAG; ++ri) {
        for (uint cj = 0u; cj < CFRAG; ++cj) {
            threadgroup float *reg = store + sgid * (BM * SG_N);
            simdgroup_store(acc[ri][cj], reg + (ri * 8u) * SG_N + cj * 8u, SG_N);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *reg = store + sgid * (BM * SG_N);
    const uint lane = tidx & 31u;
    for (uint e = lane; e < BM * SG_N; e += 32u) {
        const uint rl = e / SG_N;          // 0..31
        const uint cl = e % SG_N;          // 0..15
        const uint row = row0 + rl;
        const uint col = col_sg0 + cl;
        if (row < d.m && col < d.n) {
            // scale-after-sum, per output col — IDENTICAL ordering to gemv_q8 (c = acc * scale[col]).
            c[row * d.n + col] = reg[rl * SG_N + cl] * scale[col];
        }
    }
}
