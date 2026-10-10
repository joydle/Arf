// TILED Q4_K_S GEMM with dequant-on-load for the native Metal island megakernel — the B-row
// dense projection (q/k/v/o, optionally lm_head) accelerator. Replaces the per-row GEMV
// (gemv_q4ks_batch) on the dense projections when m>=8.
//
// NOTE ON THE REUSE PREMISE: the per-row GEMV (gemv_q4ks_batch) ALREADY reads each weight
// sub-block ONCE and reuses it across all B rows (its acc[MAXB] register loop). So the tiled
// GEMM does NOT win on weight DRAM traffic — both read weights once. The ONLY lever left is
// raw FMA throughput: the GEMV computes the B×n output with scalar dot() ALU, whereas this
// kernel feeds the dequantized tiles to the simdgroup matrix units (simdgroup_float8x8 ×
// simdgroup_multiply_accumulate), which is the M-series' highest-throughput FMA path. Whether
// that beats the (already bandwidth-bound) GEMV at these K sizes is the open question the
// probe measures.
//
// MATH (parity-identical to gemv_q4ks_batch within the 1e-3 logit bar):
//   c[row,col] = Σ_k a[row,k] · dequant(W[col,k])
//   dequant(W[col, 32b+j]) = s_b · nibble[col,b,j] + lo_b
//   s_b  = float2(dd[col*supers + (b/8)]).x (dv) · u8scale[col*subs+b]
//   lo_b = float2(dd[col*supers + (b/8)]).y (dmv) · i8min[col*subs+b]
//
// BINDINGS (identical to gemv_q4ks_batch — the host seam is unchanged):
//   0 a[B,k] float4 row-major   1 codes uint4/(col,sub-block)   2 c[B,n] float row-major
//   3 dims{m=B,k,n,_}   4 scales(u8×4/word)   5 mins(i8×4/word)   6 dd(half2 {dv,dmv}/super-block)
//
// GRID (host): threadgroups = (ceil(B/BM)=1, ceil(n/BN), 1), threads = 128 (4 simdgroups).
//   BM = 32 token rows, BN = 64 output cols. THREADGROUP MEM = 12288 B:
//     sA : BM(32) × KC(32) f32 = 4096 B   (activation tile, row-major)
//     sW : BN(64) × KC(32) f32 = 8192 B   (dequantized weight tile, COL-major within tile)
//   K iterated in KC=32 chunks (one Q4_K_S sub-block).
//
// SIMDGROUP MATMUL: output tile is 32(rows)×64(cols). With 4 simdgroups (128 threads), each
// simdgroup owns a 32×16 column-strip (cols [sg*16, sg*16+16)). It accumulates 4×2 = 8
// simdgroup_float8x8 fragments (32 rows / 8 = 4 row-frags × 16 cols / 8 = 2 col-frags) over
// the KC=32 contraction in 4 simdgroup_load/multiply_accumulate steps (KC/8 = 4).

#include <metal_stdlib>
using namespace metal;

struct Dims { uint m; uint k; uint n; uint _pad; };

// ❌ BM=64 (MM_BM64) IS 4x SLOWER — MEASURED, DO NOT SHIP IT. Kept only so the result is
// reproducible. ABA on muse-glimmer prefill: BM=32 15.7 best / 15.0 med, BM=64 4.0 / 3.9.
//
// The theory was sound and the outcome was not: doubling token rows per threadgroup doubles
// weight reuse (the staged sW tile serves 64 rows instead of 32) and halves the row-tile grid,
// with smem only going 12288 -> 16384 B of a 32 KB budget. But REGISTERS are the binding
// constraint, not smem: acc[RFRAG][CFRAG] goes 4x2=8 -> 8x2=16 simdgroup_float8x8 per
// simdgroup, which spills, and the occupancy loss swamps the reuse win by 4x.
//
// The lesson generalises: on this GPU the tile is register-bound, so more accumulator
// fragments is the wrong axis. Widening N or deepening K hits the same wall (see L169: the
// KC=64 v2 kernel is 40% slower for the analogous reason).
#ifdef MM_BM64
constant constexpr uint BM = 64u;
#else
constant constexpr uint BM = 32u;   // token rows per threadgroup tile
#endif
constant constexpr uint BN = 64u;   // output cols per threadgroup tile
constant constexpr uint KC = 32u;   // K-chunk = one Q4_K_S sub-block (KC % 8 == 0)
constant constexpr uint NSG = 4u;   // simdgroups per threadgroup (128 threads / 32)
constant constexpr uint SG_N = BN / NSG;   // = 16 cols per simdgroup
constant constexpr uint RFRAG = BM / 8u;   // = 4 row fragments (8 rows each)
constant constexpr uint CFRAG = SG_N / 8u; // = 2 col fragments (8 cols each)
constant constexpr uint KFRAG = KC / 8u;   // = 4 contraction fragments

kernel void matmul_mm_q4ks(
        device const float4 *a       [[buffer(0)]],   // [B, k] row-major
        device const uint4  *codes   [[buffer(1)]],
        device       float  *c       [[buffer(2)]],   // [B, n] row-major
        constant     Dims   &d       [[buffer(3)]],
        device const uint   *scales  [[buffer(4)]],   // u8 ×4/word
        device const uint   *mins    [[buffer(5)]],   // i8 ×4/word
        device const half2  *dd      [[buffer(6)]],   // {dv,dmv} per super-block
        uint2  tgpig                 [[threadgroup_position_in_grid]],
        uint   tidx                  [[thread_index_in_threadgroup]],
        uint   sgid                  [[simdgroup_index_in_threadgroup]],
        threadgroup float *smem      [[threadgroup(0)]]) {
    const uint subs = d.k / 32u;          // sub-blocks per column
    const uint supers = d.k / 256u;       // super-blocks per column
    const uint arow4 = d.k / 4u;          // float4 stride between activation rows

    const uint row0 = tgpig.x * BM;       // first token row of this tile (grid.x usually 1)
    const uint col0 = tgpig.y * BN;       // first output col of this tile

    threadgroup float *sA = smem;                 // [BM][KC] activation  (row-major)
    threadgroup float *sW = smem + (BM * KC);     // [BN][KC] weight  (col-major within tile)
#ifdef MM_HALF_FRAG
    // Same backing memory, viewed as half. The tiles are written as half and the simdgroup
    // loads read half, so smem HALVES too (12288 -> 6144 B) on top of the register saving.
    threadgroup half *sAh = (threadgroup half *)smem;
    threadgroup half *sWh = ((threadgroup half *)smem) + (BM * KC);
#endif

    // per-simdgroup accumulator fragments: RFRAG × CFRAG  (4 × 2 = 8 fragments of 8×8).
    simdgroup_float8x8 acc[RFRAG][CFRAG];
    for (uint i = 0; i < RFRAG; ++i)
        for (uint j = 0; j < CFRAG; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint kchunks = d.k / KC;        // = subs (KC == 32)
    const uint col_sg0 = col0 + sgid * SG_N;   // this simdgroup's first output column

    for (uint kc = 0u; kc < kchunks; ++kc) {
        const uint b = kc;                // sub-block index (KC==32 → chunk == sub-block)
        const uint sblk = b / 8u;
        const uint byte_b = b & 3u;

        // ---- stage weight tile sW[col_local*KC + kk] = dequant(W[col0+col_local, 32b+kk]) ----
        // 128 threads fill BN*KC = 2048 floats → 16 each.
#ifdef DENSE_MM_HOIST
        // BLOCK-DEQUANT HOIST: the sub-block b=kc is fixed for the whole chunk, and dd/scales/mins
        // depend only on `col` (not kk) — so s=dv*su8 and lo=dmv*mraw are constant across all 32 kk
        // of a column. Cache them once per local column (BN=64) in threadgroup mem, then the fill
        // loop does only nibble-extract + s*nib+lo. Kills the per-element dd/scales/mins re-reads +
        // dv/dmv/su8/mraw recompute (the same +24% producer-unlock proven on the MoE mm_id kernel).
        // Parity: keeps the (s)*nib + (lo) two-step op order → bit-identical to the per-element path.
        threadgroup float scol[BN];   // s  = dv*su8 per local column
        threadgroup float lcol[BN];   // lo = dmv*float(mraw) per local column
        for (uint cl = tidx; cl < BN; cl += 128u) {
            const uint col = col0 + cl;
            float s = 0.0f, lo = 0.0f;
            if (col < d.n) {
                const uint ddbase = col * supers;
                const float2 dp = float2(dd[ddbase + sblk]);
                const float dv  = dp.x;
                const float dmv = dp.y;
                const uint sidx = col * subs + b;
                const uint sword = scales[sidx >> 2];
                const uint mword = mins[sidx >> 2];
                const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
                const int   mraw = int(mword << (24u - 8u * byte_b)) >> 24;
                s  = dv * su8;
                lo = dmv * float(mraw);
            }
            scol[cl] = s; lcol[cl] = lo;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // L169 — ONE uint4 LOAD PER 8 NIBBLES. The old loop walked BN*KC=2048 elements and
        // re-read `codes[col*subs + b]` — the SAME 16-byte word — for EVERY element, to extract
        // one 4-bit nibble each: 16 bytes fetched per 0.5 bytes used = 32x read amplification,
        // and 32 threads hammering one address. Now each thread claims an (column, word) pair,
        // loads the uint4 ONCE, and writes all 8 nibbles that word carries.
        //
        // BN*KC/8 = 256 (column, word) pairs over 128 threads = 2 iterations each, vs 16.
        // Arithmetic is unchanged — same s*nib + lo, same order — so this stays bit-identical
        // to the per-element path and to the GEMV oracle.
        for (uint w = tidx; w < (BN * KC) / 8u; w += 128u) {
            const uint cl = w / 4u;        // local column 0..63  (4 words of 8 nibbles = KC=32)
            const uint wi = w & 3u;        // which word of the uint4: 0..3
            const uint col = col0 + cl;
            uint word = 0u;
            float sc = 0.0f, lo = 0.0f;
            if (col < d.n) {
                const uint4 qv = codes[col * subs + b];   // ONE load, serves 8 nibbles
                word = (wi == 0u) ? qv.x : (wi == 1u) ? qv.y : (wi == 2u) ? qv.z : qv.w;
                sc = scol[cl]; lo = lcol[cl];
            }
            const uint kk0 = wi * 8u;
            #pragma unroll(8)
            for (uint t8 = 0u; t8 < 8u; ++t8) {
                const float nib = float((word >> (4u * t8)) & 0xFu);
#ifdef MM_HALF_FRAG
                // K-MAJOR store: sWh[k][col] with row pitch BN, so the fragment load below is
                // FLAT (no transpose). moe_mm_id_q4ks — the half kernel that works here — never
                // transposes either; it pre-lays the tile. transpose=true on a half tile is what
                // produced the wrong output.
                sWh[(kk0 + t8) * BN + cl] = (col < d.n) ? half(sc * nib + lo) : half(0.0);
#else
                sW[cl * KC + kk0 + t8] = (col < d.n) ? (sc * nib + lo) : 0.0f;
#endif
            }
        }
#else
        for (uint e = tidx; e < BN * KC; e += 128u) {
            const uint cl = e / KC;       // local column 0..63
            const uint kk = e % KC;       // element 0..31
            const uint col = col0 + cl;
            float wv = 0.0f;
            if (col < d.n) {
                const uint4 qv = codes[col * subs + b];
                const uint word = (kk < 8u) ? qv.x : (kk < 16u) ? qv.y : (kk < 24u) ? qv.z : qv.w;
                const float nib = float((word >> (4u * (kk & 7u))) & 0xFu);
                const uint ddbase = col * supers;
                const float2 dp = float2(dd[ddbase + sblk]);
                const float dv  = dp.x;
                const float dmv = dp.y;
                const uint sidx = col * subs + b;
                const uint sword = scales[sidx >> 2];
                const uint mword = mins[sidx >> 2];
                const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
                const int   mraw = int(mword << (24u - 8u * byte_b)) >> 24;
                wv = (dv * su8) * nib + (dmv * float(mraw));
            }
#ifdef MM_HALF_FRAG
            sWh[kk * BN + cl] = half(wv);   // k-major, see the hoist branch
#else
            sW[cl * KC + kk] = wv;
#endif
        }
#endif
        // ---- stage activation tile sA[row_local*KC + kk] = a[row0+row_local, 32b+kk] ----
        // BM*KC = 1024 floats → 8 each.
        for (uint e = tidx; e < BM * KC; e += 128u) {
            const uint rl = e / KC;       // local row 0..31
            const uint kk = e % KC;       // 0..31
            const uint row = row0 + rl;
            float av = 0.0f;
            if (row < d.m) {
                const uint kelem = 32u * b + kk;
                const float4 v4 = a[row * arow4 + (kelem >> 2)];
                const uint lane = kelem & 3u;
                av = (lane == 0u) ? v4.x : (lane == 1u) ? v4.y : (lane == 2u) ? v4.z : v4.w;
            }
#ifdef MM_HALF_FRAG
            sAh[rl * KC + kk] = half(av);
#else
            sA[rl * KC + kk] = av;
#endif
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- simdgroup matmul: acc[ri][cj] += A_frag(ri,kf) * W_frag(kf,cj) ----
        // A_frag: rows [ri*8, ri*8+8), k [kf*8, kf*8+8). sA is [row][k] row-major, stride KC.
        // W_frag: we need Wᵀ as the rhs (k × col). sW is [col][k] col-major (stride KC); a
        // simdgroup_load with the col-major layout (transpose) yields the (k × col) fragment.
        for (uint kf = 0u; kf < KFRAG; ++kf) {
#ifdef MM_HALF_FRAG
            // HALF INPUT FRAGMENTS, FLOAT ACCUMULATE (L171). The kernel is REGISTER-bound
            // (L170: BM=64 spilled and lost 4x), so the lever is fewer registers per fragment,
            // not a bigger tile. af[RFRAG] + wf are STAGING registers — halving them frees
            // RFRAG*32 + 32 bytes per lane without touching acc, which stays float because the
            // K=19968 accumulation chain is exactly where half would lose precision. This is
            // llama.cpp's shape: half inputs, float accumulator.
            simdgroup_half8x8 af[RFRAG];
            for (uint ri = 0; ri < RFRAG; ++ri) {
                simdgroup_load(af[ri], sAh + (ri * 8u) * KC + kf * 8u, KC);
            }
#else
            simdgroup_float8x8 af[RFRAG];
            for (uint ri = 0; ri < RFRAG; ++ri) {
                simdgroup_load(af[ri], sA + (ri * 8u) * KC + kf * 8u, KC);
            }
#endif
            for (uint cj = 0u; cj < CFRAG; ++cj) {
                // weight fragment (k × 8cols): sW laid out [col][k] (stride KC). Loading with
                // transpose=true gives the k-major (k rows × 8 cols) fragment the matmul needs.
                const uint wcol_local = sgid * SG_N + cj * 8u;   // local col within tile
#ifdef MM_HALF_FRAG
                simdgroup_half8x8 wf;
                simdgroup_load(wf, sWh + (kf * 8u) * BN + wcol_local, BN);   // flat, k-major
#else
                simdgroup_float8x8 wf;
                simdgroup_load(wf, sW + wcol_local * KC + kf * 8u, KC, ulong2(0, 0), /*transpose=*/true);
#endif
                for (uint ri = 0; ri < RFRAG; ++ri) {
                    simdgroup_multiply_accumulate(acc[ri][cj], af[ri], wf, acc[ri][cj]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- store fragments to threadgroup, then scatter to c[B,n] (bounds-checked) ----
    // Reuse sA as the store staging area (BM*SG_N per simdgroup would collide across sg; instead
    // each simdgroup writes its own 32×16 strip directly through a small private store buffer).
    threadgroup float *store = smem;   // reuse smem after the contraction loop
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ri = 0; ri < RFRAG; ++ri) {
        for (uint cj = 0u; cj < CFRAG; ++cj) {
            // store this 8×8 fragment to a per-simdgroup region of smem.
            // region: sgid owns [sgid*(BM*SG_N), ...]; layout row-major 32×16 per simdgroup.
            threadgroup float *reg = store + sgid * (BM * SG_N);
            simdgroup_store(acc[ri][cj], reg + (ri * 8u) * SG_N + cj * 8u, SG_N);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scatter: each thread writes a slice of its simdgroup's 32×16 strip to global c.
    threadgroup float *reg = store + sgid * (BM * SG_N);
    const uint lane = tidx & 31u;
    for (uint e = lane; e < BM * SG_N; e += 32u) {
        const uint rl = e / SG_N;          // 0..31
        const uint cl = e % SG_N;          // 0..15
        const uint row = row0 + rl;
        const uint col = col_sg0 + cl;
        if (row < d.m && col < d.n) {
            c[row * d.n + col] = reg[rl * SG_N + cl];
        }
    }
}
