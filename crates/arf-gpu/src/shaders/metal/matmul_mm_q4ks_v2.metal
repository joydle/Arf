// TILED Q4_K_S GEMM v2 — KC=64 (2 sub-blocks/chunk) f32-tile variant of matmul_mm_q4ks. Drop-in
// replacement (same bindings, same grid, same math) gated by ARF_MM_V2. The dense q/k/v/o
// projection accelerator for the batched megakernel.
//
// WHY v2 (the measured bottleneck): a microbench (mm_q4ks_probe, PROF_SKIP_DEQUANT A/B) showed v1
// runs ~13-16x OFF the weight-read roofline AND the dequant is only ~5% of the time — so v1 is
// neither bandwidth- nor compute-bound. It is BARRIER-SERIALIZATION bound in the K-loop: v1 uses
// KC=32 (= one Q4_K_S sub-block) and pays 2 threadgroup_barriers PER chunk, so o_proj (k=4096 →
// 128 chunks) costs ~2x q_proj (k=2048 → 64 chunks) for the SAME FLOPs. HALF-storage (2x occupancy)
// gave ZERO speedup — occupancy was NOT the binding constraint; the per-chunk barrier count is.
//
// v2 FIX: KC=64 (span 2 Q4_K_S sub-blocks per chunk) → HALVES the chunk count → HALVES the
// threadgroup-barrier round-trips in the K-loop. Tiles stay f32 (parity-clean max_abs ~2.5e-6;
// half-storage flipped max_rel to 0.6 = token-flip risk, and bought nothing). tg mem grows to
// sA(32×64×4=8KB)+sW(64×64×4=16KB)=24KB; occupancy drops to ~1 threadgroup/core, which the
// half-storage A/B proved is harmless here.
//
// MATH (parity-identical to gemv_q4ks_batch within the 1e-3 logit bar):
//   c[row,col] = Σ_k a[row,k] · dequant(W[col,k])
//   dequant(W[col, 32b+j]) = (dv·su8)·nibble[col,b,j] + (dmv·mraw)
//
// BINDINGS (identical to gemv_q4ks_batch / matmul_mm_q4ks — host seam unchanged):
//   0 a[B,k] float4 row-major   1 codes uint4/(col,sub-block)   2 c[B,n] float row-major
//   3 dims{m=B,k,n,_}   4 scales(u8×4/word)   5 mins(i8×4/word)   6 dd(half2 {dv,dmv}/super-block)
//
// GRID (host, UNCHANGED): threadgroups = (ceil(B/BM), ceil(n/BN), 1), threads = 128 (4 simdgroups).
//   BM=32 rows, BN=64 cols, KC=64 (2 sub-blocks). THREADGROUP MEM = 24576 B (sA 8K + sW 16K).

#include <metal_stdlib>
using namespace metal;

struct Dims { uint m; uint k; uint n; uint _pad; };

constant constexpr uint BM = 32u;   // token rows per threadgroup tile
constant constexpr uint BN = 64u;   // output cols per threadgroup tile
constant constexpr uint KC = 64u;   // K-chunk = TWO Q4_K_S sub-blocks (KC % 8 == 0, KC % 32 == 0)
constant constexpr uint NSG = 4u;   // simdgroups per threadgroup (128 threads / 32)
constant constexpr uint SG_N = BN / NSG;   // = 16 cols per simdgroup
constant constexpr uint RFRAG = BM / 8u;   // = 4 row fragments (8 rows each)
constant constexpr uint CFRAG = SG_N / 8u; // = 2 col fragments (8 cols each)
constant constexpr uint KFRAG = KC / 8u;   // = 8 contraction fragments
constant constexpr uint SUBC  = KC / 32u;  // = 2 Q4_K_S sub-blocks per chunk

kernel void matmul_mm_q4ks_v2(
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

    // per-simdgroup accumulator fragments: RFRAG × CFRAG (4 × 2 = 8) of FLOAT 8x8 (parity).
    simdgroup_float8x8 acc[RFRAG][CFRAG];
    for (uint i = 0; i < RFRAG; ++i)
        for (uint j = 0; j < CFRAG; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint kchunks = d.k / KC;        // = subs/2 (KC == 64)
    const uint col_sg0 = col0 + sgid * SG_N;   // this simdgroup's first output column

    for (uint kc = 0u; kc < kchunks; ++kc) {
        // ---- stage weight tile sW[col_local*KC + kk] = dequant(W[col0+col_local, KC*kc+kk]) ----
        // 128 threads fill BN*KC = 4096 floats → 32 each. kk in [0,KC); sub-block = kc*SUBC + kk/32.
        for (uint e = tidx; e < BN * KC; e += 128u) {
            const uint cl = e / KC;       // local column 0..63
            const uint kk = e % KC;       // element 0..KC-1
            const uint col = col0 + cl;
            float wv = 0.0f;
            if (col < d.n) {
                const uint b = kc * SUBC + (kk >> 5);   // global sub-block index
                const uint jj = kk & 31u;               // element within the sub-block 0..31
                const uint sblk = b >> 3;
                const uint byte_b = b & 3u;
                const uint4 qv = codes[col * subs + b];
                const uint word = (jj < 8u) ? qv.x : (jj < 16u) ? qv.y : (jj < 24u) ? qv.z : qv.w;
                const float nib = float((word >> (4u * (jj & 7u))) & 0xFu);
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
            sW[cl * KC + kk] = wv;
        }
        // ---- stage activation tile sA[row_local*KC + kk] = a[row0+row_local, KC*kc+kk] ----
        // BM*KC = 2048 floats → 16 each.
        for (uint e = tidx; e < BM * KC; e += 128u) {
            const uint rl = e / KC;       // local row 0..31
            const uint kk = e % KC;       // 0..KC-1
            const uint row = row0 + rl;
            float av = 0.0f;
            if (row < d.m) {
                const uint kelem = KC * kc + kk;
                const float4 v4 = a[row * arow4 + (kelem >> 2)];
                const uint lane = kelem & 3u;
                av = (lane == 0u) ? v4.x : (lane == 1u) ? v4.y : (lane == 2u) ? v4.z : v4.w;
            }
            sA[rl * KC + kk] = av;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- simdgroup matmul over KFRAG=8 contraction fragments ----
        for (uint kf = 0u; kf < KFRAG; ++kf) {
            simdgroup_float8x8 af[RFRAG];
            for (uint ri = 0; ri < RFRAG; ++ri) {
                simdgroup_load(af[ri], sA + (ri * 8u) * KC + kf * 8u, KC);
            }
            for (uint cj = 0u; cj < CFRAG; ++cj) {
                const uint wcol_local = sgid * SG_N + cj * 8u;   // local col within tile
                simdgroup_float8x8 wf;
                // sW laid out [col][k] (stride KC); transpose=true → (k × 8cols) fragment.
                simdgroup_load(wf, sW + wcol_local * KC + kf * 8u, KC, ulong2(0, 0), /*transpose=*/true);
                for (uint ri = 0; ri < RFRAG; ++ri) {
                    simdgroup_multiply_accumulate(acc[ri][cj], af[ri], wf, acc[ri][cj]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- store fragments to threadgroup, then scatter to c[B,n] (bounds-checked) ----
    threadgroup float *store = smem;   // 4 sg × 32×16 floats = 8192 B (≤ 24576 staged alloc)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ri = 0; ri < RFRAG; ++ri) {
        for (uint cj = 0; cj < CFRAG; ++cj) {
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
            c[row * d.n + col] = reg[rl * SG_N + cl];
        }
    }
}
