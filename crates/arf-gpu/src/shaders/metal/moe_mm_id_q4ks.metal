// FAITHFUL port of llama.cpp's kernel_mul_mm_id (tiled per-expert MoE GEMM) for Q4_K_S, native
// Metal island. Closes the conc≥32 MoE expert-weight-TRAFFIC gap: the per-output indirect GEMV
// (gemv_q4ks_id_b) re-reads each expert's weight tile per (row,slot) = ~4× more expert bytes than
// llama @conc64. This kernel loads each expert's weight tile ONCE into threadgroup memory and
// reuses it across ALL the expert's grouped rows via the simdgroup matrix units. THE REUSE.
//
// EXACT llama structure (ggml-metal.metal kernel_mul_mm_id, non-TENSOR path), preserved:
//   * TILE: NR0=64 output cols, NR1=32 grouped rows, NK=32 K-chunk. 4 simdgroups (128 threads).
//   * HALF multiply + FLOAT accumulate: sa/sb staged as HALF (simdgroup_half8x8 ma[4], mb[2]);
//     mc[8] = simdgroup_float8x8 accumulators (make_filled_simdgroup_matrix<float,8>). Half-mul =
//     2× matmul throughput; the f32 accumulator keeps parity (only the staged tile values are
//     half-rounded — Q4_K_S dequant values are small, so the half rounding is < ~1e-3).
//   * DOUBLE-BUFFER: `il = (il + 2 < nl) ? il + 2 : il % 2;` advances the dequant sub-block index
//     to overlap the next load with the current compute.
//   * Threadgroup packed-8×8-block layout (sa + 64*ib + 8*ly + lx, ib = 8*sx + sy) — IDENTICAL to
//     llama so the simdgroup_load strides (lsma += 8*64, lsmb += 4*64) match exactly.
//
// GRID (our tile-list, BETTER than llama's per-expert grid.z): grid.x indexes the CSR-built
// tile-list tile_list[grid.x] = (expert<<16)|tile_idx → expert + tile_row0 = tile_idx*NR1 directly.
// No empty-expert grid.z waste. With NR1=32 @conc64 ~4 rows/expert → 1 tile/expert → ~128 tiles.
// grid.y*NR0 = col-tile. Tiles past the GPU-computed real n_tiles (csr_meta[1]) early-out.
//
// Q4_K_S dequant (dd super-block dv/dmv + scales u8 + mins i8 + codes uint4 nibbles) = IDENTICAL to
// moe_batch_msl.metal gemv_q4ks_id_b — copied verbatim so the grouped path is parity with the GEMV.

#include <metal_stdlib>
using namespace metal;

// llama's FOR_UNROLL (loop fully unrolled) — defined here since each island .metal compiles
// standalone (no llama prelude). The trip counts (16, 8, 4, 2) are all compile-time constants.
#define FOR_UNROLL _Pragma("clang loop unroll(full)") for

// TILE PRECISION. The faithful llama port stages the dequantized weight + the gathered activation as
// HALF and multiplies with simdgroup_half8x8 (2× matmul throughput); the accumulator mc[8] stays
// simdgroup_float8x8 (half-mul + FLOAT-acc). For Q4_K_S the half mantissa rounding shows ~2-3e-3
// abs vs the f32 GEMV oracle — under the greedy-token bar but OVER the harness 1e-4 bit-gate. So the
// tile type is a knob: the Rust host prepends `#define MOE_MM_F32 1` to compile the parity-tight
// f32-tile variant (simdgroup_float8x8 ma/mb) for the strict harness gate; the default half tile is
// the performance path (token-coherence-gated).
#ifdef MOE_MM_F32
typedef float           mm_tile_t;
typedef simdgroup_float8x8 mm_mat_t;
#define MM_TZERO 0.0f
#else
typedef half            mm_tile_t;
typedef simdgroup_half8x8  mm_mat_t;
#define MM_TZERO 0.0h
#endif

// {m=B, k, n, top_k}. m carries B; the CSR encodes every (row,slot), grid is keyed on the tile-list.
struct MmIdDims { uint m; uint k; uint n; uint top_k; };

constant constexpr short NR0 = 64;   // output cols per threadgroup tile
// NR1 = grouped rows per threadgroup tile (the llama NR1=32). At conc64 ~4 rows/expert fill the
// 32-row tile → ~87% masked padding. The host can compile a SHRUNK variant by prepending
// `#define MOE_MM_NR1 <8|16|32>` (ARF_MOE_NR1_8) to cut the wasted matmul once the block-dequant
// hoist removed the serial dequant gate. NR1 MUST be a multiple of 8 (simdgroup_float8x8 fragment).
// The simdgroup-fragment partition (lsma/lsmb/mc[]/writeback) is HARDCODED per-NR1, so each
// supported value gets its own #if branch below — the row-math (r1, masks, lr1) auto-scales off the
// const. CSR_MM_BM (moe_batch_msl.metal) MUST equal this (host drives both from the same define).
#ifndef MOE_MM_NR1
#define MOE_MM_NR1 32
#endif
constant constexpr short NR1 = MOE_MM_NR1;   // grouped rows per threadgroup tile
constant constexpr short NK  = 32;   // K-chunk = one Q4_K_S sub-block (32 elems)
constant constexpr short NL0 = NK / 16;  // = 2  (weight load lanes per col)
constant constexpr short NL1 = NK / 8;   // = 4  (activation load lanes per row)

// nl = sub-blocks per dequant step. We dequantize one Q4_K_S 32-elem sub-block per `il` step into a
// 4x4 half tile; NK=32 = one sub-block so nl effectively = 1 advance per K-chunk, but we keep the
// llama double-buffer machinery (il advances 2 per chunk, wrapping il%2) so the staged sub-block of
// the NEXT chunk overlaps the current simdgroup matmul.
constant constexpr short MM_NL = 2;

// ---- Q4_K_S dequant of ONE element: weight column `wcol`, K element `kelem` (0..k-1) ----
// Returns dequant(W[wcol, kelem]) as float. b = kelem/32 (sub-block), within = kelem%32.
static inline float dequant_q4ks_elem(
        device const uint4 *codes, device const uint *scales, device const uint *mins,
        device const half2 *dd, uint wcol, uint kelem, uint subs, uint supers) {
    const uint b = kelem >> 5;            // sub-block index (0..subs-1)
    const uint within = kelem & 31u;      // element within sub-block (0..31)
    const uint sblk = b >> 3;             // super-block index (b/8)
    const uint byte_b = b & 3u;
    const uint4 qv = codes[wcol * subs + b];
    const uint word = (within < 8u) ? qv.x : (within < 16u) ? qv.y : (within < 24u) ? qv.z : qv.w;
    const float nib = float((word >> (4u * (within & 7u))) & 0xFu);
    const uint ddbase = wcol * supers;
    const float2 dp = float2(dd[ddbase + sblk]);
    const float dv  = dp.x;
    const float dmv = dp.y;
    const uint sidx = wcol * subs + b;
    const uint sword = scales[sidx >> 2];
    const uint mword = mins[sidx >> 2];
    const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
    const int   mraw = int(mword << (24u - 8u * byte_b)) >> 24;
    return (dv * su8) * nib + (dmv * float(mraw));
}

// ===================== grouped gate/up tiled mm_id (Q4_K_S) =====================
// Per active expert (via tile-list), c[(row*top_k+slot)*n + col] = Σ_k normed[src_row*k + k] ·
// dequant(W_e[col,k]) for ALL (row,slot) routed to that expert. n = inter (gate/up), k = h.
// Run TWICE (gate weights, up weights). Router weight NOT folded here (applied in slot-reduce).
kernel void moe_mm_id_gu_q4ks(
        device const float  *a       [[buffer(0)]],   // [B, k] activation (normed), row-major
        device const uint4  *codes   [[buffer(1)]],   // all experts' codes
        device       float  *c       [[buffer(2)]],   // [B, top_k, n] (row,slot,col)-major
        constant     MmIdDims &d     [[buffer(3)]],
        device const uint   *scales  [[buffer(4)]],   // u8 ×4/word
        device const uint   *mins    [[buffer(5)]],   // i8 ×4/word
        device const half2  *dd      [[buffer(6)]],   // {dv,dmv} per super-block
        device const uint   *csr_offsets [[buffer(7)]],  // [ne+1]
        device const uint   *csr_row_slot[[buffer(8)]],  // [B*top_k] packs row*top_k+slot
        device const uint   *csr_active  [[buffer(9)]],  // unused (tile-list replaces grid.z)
        device const uint   *csr_meta    [[buffer(10)]], // [0]=n_active, [1]=n_tiles
        device const uint   *tile_list   [[buffer(11)]], // [max_tiles] packed (expert<<16)|tile_idx
        uint3  tgpig                 [[threadgroup_position_in_grid]],
        ushort tiitg                 [[thread_index_in_threadgroup]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]],
        threadgroup char *shmem      [[threadgroup(0)]]) {
    threadgroup mm_tile_t *sa = (threadgroup mm_tile_t *)(shmem);                       // weight tile
    threadgroup mm_tile_t *sb = (threadgroup mm_tile_t *)(shmem + NR0 * NK * sizeof(mm_tile_t)); // activation tile

    const uint subs   = d.k / 32u;          // sub-blocks per weight column
    const uint supers = d.k / 256u;         // super-blocks per weight column
    const uint arow   = d.k;                // float stride between activation rows

    // ---- tile assignment from the CSR-built tile-list ----
    if (tgpig.x >= csr_meta[1]) return;     // beyond the real tile count
    const uint packed   = tile_list[tgpig.x];
    const uint expert   = packed >> 16;
    const uint tile_idx = packed & 0xFFFFu;
    const uint csr_lo   = csr_offsets[expert];
    const uint csr_hi   = csr_offsets[expert + 1u];
    const int  neh1     = int(csr_hi - csr_lo);   // this expert's routed-row count

    const int r1 = int(tile_idx) * NR1;     // first group-local row of this tile
    if (r1 >= neh1) return;                  // empty row-tile early-out (llama parity)

    const int r0 = int(tgpig.y) * NR0;      // first output col of this tile
    const uint exp_col_base = expert * d.n; // gather the expert's weight block

    const short nr0 = (int(d.n) - r0 < NR0) ? short(int(d.n) - r0) : NR0;  // partial col mask
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;          // partial row mask

    // thread→tile-element mapping (llama): lr0 = weight row 0..63, lr1 = act row 0..31, il0 = lane.
    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : nr0 - 1;  // 0..63
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : nr1 - 1;  // 0..31
    const short il0 = (tiitg % NL0);        // 0..1  (NK/16 lanes for weight)
    short il = il0;

    // weight column this thread loads = r0 + lr0 (clamped to nr0).
    const uint wcol = exp_col_base + uint(r0 + lr0);
    const bool wcol_ok = (r0 + lr0) < int(d.n);

    // activation gather: group-local row r1+lr1 → CSR source index → src_row → a[src_row*k + ...].
    const uint grow = uint(r1 + lr1);
    const uint isrc = csr_row_slot[csr_lo + grow];   // i = row*top_k + slot
#ifdef MOE_MM_GATHERPROBE
    // MEASUREMENT-ONLY (ARF_MOE_MM_GATHERPROBE, L77): replace the SCATTERED activation row
    // index with a CONTIGUOUS one. Same byte count, same MMA count, same weight traffic — the ONLY
    // thing that changes is the locality of the `a[src_row*arow + kelem]` gather (line ~294).
    //   busy DROPS here → the scattered gather IS the cost → a row-compaction pass is the lever.
    //   busy FLAT here  → the gather is NOT the cost → the L76 gate+up anomaly is something else,
    //                     and the scattered-gather hypothesis dies.
    // Output is GARBAGE (wrong rows) — token-parity WILL fail; expected. Read [moe-flops] busy ONLY.
    const uint src_row = uint(r1 + lr1) % max(d.m, 1u);
#else
    const uint src_row = isrc / d.top_k;
#endif
    const short iy = 8 * (tiitg % NL1);     // 0,8,16,24 — activation K-offset for this lane

    // accumulators. The fragment partition differs per NR1 (see header):
    //   NR1=32 (llama): 4 sg = 2 col-halves × 2 row-halves; each sg = 16 rows × 32 cols = mc[8]
    //                   (2 row-frags mb[2] × 4 col-frags ma[4]).
    //   NR1=8 (shrunk): 4 sg ALL share the 8 rows, split the 64 cols 4-way (16 cols/sg) → 1 row-frag
    //                   mb[1] × 2 col-frags ma[2] = mc[2]. Keeps all 4 sg busy; no row-half split.
#if MOE_MM_NR1 == 8
    mm_mat_t  ma[2];
    mm_mat_t  mb[1];
    simdgroup_float8x8 mc[2];
    for (short i = 0; i < 2; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
#else
    mm_mat_t  ma[4];
    mm_mat_t  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
#endif

#ifndef MOE_MM_DBUF
    // ── PREFETCH PIPELINE (MOE_MM_PREFETCH, ARF_MOE_MM_PREFETCH): the dequant staging issues 4
    // DEPENDENT global loads/sub-block (codes→dd→scales→mins) then unpacks — the load-latency chain is
    // the conc64 stall (measured: dequant dominates, +30% w/o the hoist). Software-pipeline it: hoist
    // the NEXT iteration's global loads (codes/dd/scales/mins) into registers BEFORE this iteration's
    // barrier+matmul, so the loads are IN FLIGHT while the matmul runs → latency hidden. Bit-identical
    // (same values, fetched earlier — no math change). Only compiles the prefetch preamble under the
    // define; the default path is byte-for-byte the proven kernel.
#ifdef MOE_MM_PREFETCH
    // prefetch registers for the next sub-block's quant data (per-thread, one il half).
    uint4  pf_qv; float2 pf_dp; uint pf_sword; uint pf_mword; bool pf_valid;
    #define MOE_PF_LOAD(LK, IL) do { \
        const uint _kb = (LK) + 16u * uint(IL); \
        const uint _b  = _kb >> 5; const uint _sblk = _b >> 3; \
        pf_valid = wcol_ok && _kb < d.k; \
        pf_qv    = pf_valid ? codes[wcol * subs + _b] : uint4(0); \
        pf_dp    = float2(dd[wcol * supers + _sblk]); \
        pf_sword = scales[(wcol * subs + _b) >> 2]; \
        pf_mword = mins[(wcol * subs + _b) >> 2]; \
    } while (0)
    MOE_PF_LOAD(0u, il);   // prime the pipe with iteration 0's loads (issued before the loop)
#endif
    for (uint loop_k = 0; loop_k < d.k; loop_k += NK) {
        // ---- stage WEIGHT sub-block into sa (dequant → half), llama packed-8×8-block layout ----
        // il selects which 16-elem half of the 32-elem sub-block (il<2 → this chunk's sub-block).
        // We dequantize 16 elements (one il half) per thread into the tile.
        {
            const uint kbase = loop_k + 16u * uint(il);     // K element base for this il half
#ifdef MOE_MM_PREFETCH
            // Use the PREFETCHED quant data (loads already in flight from the previous iteration),
            // then immediately issue the NEXT iteration's loads so they overlap this matmul.
            const uint b      = kbase >> 5;
            const uint byte_b = b & 3u;
            const uint4 qv    = pf_qv;
            const float dv    = pf_dp.x;
            const float dmv   = pf_dp.y;
            const float su8   = float((pf_sword >> (8u * byte_b)) & 0xFFu);
            const int   mraw  = int(pf_mword << (24u - 8u * byte_b)) >> 24;
            const float s     = dv * su8;
            const float lo    = dmv * float(mraw);
            { const uint _nk = loop_k + NK; if (_nk < d.k) { MOE_PF_LOAD(_nk, il); } }  // next iter's loads IN FLIGHT
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                const uint within = (kbase & 31u) + uint(i);
                const uint w = (within < 8u) ? qv.x : (within < 16u) ? qv.y : (within < 24u) ? qv.z : qv.w;
                const float nib = float((w >> (4u * (within & 7u))) & 0xFu);
                mm_tile_t wv = MM_TZERO;
                if (wcol_ok && (kbase + uint(i)) < d.k) wv = (mm_tile_t)(s * nib + lo);
                *(sa + 64 * ib + 8 * ly + lx) = wv;
            }
#elif defined(MOE_MM_DEQSHORTCUT)
            // ⚠️ MEASUREMENT-ONLY, NOT CORRECT (ARF_MOE_DEQSHORTCUT crux probe). Isolates
            // whether the Q4_K dequant DEPENDENT-LOAD CHAIN is the conc64 latency stall. The shipped
            // HOIST path issues 4 scattered dependent global reads per sub-block (codes→dd→scales→mins,
            // all based at wcol=expert*d.n) then unpacks. This branch GUTS that chain: it reads ONLY
            // `codes` (the one unavoidable weight read — you can't matmul with no weights) and replaces
            // the scale/min derivation (dd/scales/mins — the 3 EXTRA scattered dependent reads) with a
            // constant. Same tile writes, same matmul, same occupancy — ONLY the dependent-load count
            // per sub-block drops 4→1. So an A/B of {HOIST default} vs {this} measures EXACTLY the
            // latency contribution of the dequant read-chain, with NO new buffer/plumbing.
            //   busy DROPS here → the dequant dependent-load chain IS the stall → baked-AoS-Q4
            //     / producer-consumer is the real lever (build it). util should also rise off 7%.
            //   busy FLAT here → the dequant reads are NOT the stall → kills 4 ideas at once; pivot to
            //     the ACTIVATION gather (a[src_row*arow+kelem], line ~294) or the DVFS/clock axis.
            // Output is GARBAGE (constant scale) — token-parity WILL fail; that is expected & fine, we
            // only read [moe-flops] busy/util here, never the logits. Guarded so default is untouched.
            const uint b      = kbase >> 5;
            const uint4 qv    = (wcol_ok && kbase < d.k) ? codes[wcol * subs + b] : uint4(0);  // the ONE kept read
            const float s     = 0.05f;   // constant stand-in for dv*su8 (the 3 dropped scattered reads)
            const float lo    = 0.0f;    // constant stand-in for dmv*mraw
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                const uint within = (kbase & 31u) + uint(i);
                const uint w = (within < 8u) ? qv.x : (within < 16u) ? qv.y : (within < 24u) ? qv.z : qv.w;
                const float nib = float((w >> (4u * (within & 7u))) & 0xFu);
                mm_tile_t wv = MM_TZERO;
                if (wcol_ok && (kbase + uint(i)) < d.k) wv = (mm_tile_t)(s * nib + lo);
                *(sa + 64 * ib + 8 * ly + lx) = wv;
            }
#elif defined(MOE_MM_HOIST)
            // BLOCK-DEQUANT HOIST: kbase = loop_k + 16*il, so all 16 unrolled i share b = kbase>>5
            // (NK=32 stride keeps loop_k 32-aligned; il in {0,1} → 16*il < 32). Lift the scale/min
            // derivation OUT of the per-nibble unroll: compute s=dv*su8 and lo=dmv*mraw ONCE per
            // (thread,sub-block) instead of 16×. Matches the GEMV oracle (moe_batch_msl.metal:73-82).
            const uint b      = kbase >> 5;                 // sub-block, CONSTANT across all 16 i
            const uint sblk   = b >> 3;                      // super-block (b/8)
            const uint byte_b = b & 3u;
            const uint4 qv    = (wcol_ok && kbase < d.k) ? codes[wcol * subs + b] : uint4(0);
            const uint ddbase = wcol * supers;
            const float2 dp   = float2(dd[ddbase + sblk]);
            const float dv    = dp.x;
            const float dmv   = dp.y;
            const uint sidx   = wcol * subs + b;
            const float su8   = float((scales[sidx >> 2] >> (8u * byte_b)) & 0xFFu);
            const int   mraw  = int(mins[sidx >> 2] << (24u - 8u * byte_b)) >> 24;
            const float s     = dv * su8;                    // hoisted product — IDENTICAL every i
            const float lo    = dmv * float(mraw);           // hoisted product
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                // within = element index inside the 32-elem sub-block. kbase&31 is 0 (il=0) or 16
                // (il=1); add i to get 0..31. Word-select + nibble extract are the only per-i parts.
                const uint within = (kbase & 31u) + uint(i);
                const uint w = (within < 8u) ? qv.x : (within < 16u) ? qv.y : (within < 24u) ? qv.z : qv.w;
                const float nib = float((w >> (4u * (within & 7u))) & 0xFu);
                mm_tile_t wv = MM_TZERO;
                // KEEP the two-step form s*nib+lo (mul-then-mul-add) — matches the GEMV oracle's
                // IEEE-754 op order (moe_batch_msl.metal:100). Do NOT collapse to fma / dv*su8*nib.
                if (wcol_ok && (kbase + uint(i)) < d.k) wv = (mm_tile_t)(s * nib + lo);
                *(sa + 64 * ib + 8 * ly + lx) = wv;
            }
#else
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                mm_tile_t wv = MM_TZERO;
                const uint kelem = kbase + uint(i);
                if (wcol_ok && kelem < d.k) {
                    wv = (mm_tile_t) dequant_q4ks_elem(codes, scales, mins, dd, wcol, kelem, subs, supers);
                }
                *(sa + 64 * ib + 8 * ly + lx) = wv;
            }
#endif
        }
        // ---- stage ACTIVATION into sb (gather → half), llama packed-8×8-block layout ----
        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            FOR_UNROLL (short i = 0; i < 8; i++) {
                const short lx = i;
                mm_tile_t av = MM_TZERO;
                const uint kelem = loop_k + uint(iy) + uint(i);
                if (kelem < d.k) av = (mm_tile_t) a[src_row * arow + kelem];
                *(sb + 64 * ib + 8 * ly + lx) = av;
            }
        }

        // double-buffer advance (overlap next load with current compute).
        il = (il + 2 < MM_NL) ? il + 2 : il % 2;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- simdgroup matmul over the NK chunk (llama outer-product layout) ----
#if MOE_MM_NR1 == 8
        // 4 sg split the NR0=64 cols 4-way (16 cols = 2 col-blocks/sg); ALL share rows 0-7 (1 row-
        // frag). lsma col-base = 2 col-blocks/sg; lsmb = row-block 0 (no row split). Per ik both
        // advance to the next K-group (lsma += 8*64 = +8 ib, lsmb += 4*64 = +4 ib) — IDENTICAL
        // per-K-element accumulation as NR1=32, only the row/col fragment ownership differs.
        threadgroup const mm_tile_t *lsma = (sa + 2 * 64 * sgitg);
        threadgroup const mm_tile_t *lsmb = (sb);
        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) simdgroup_multiply_accumulate(mc[i], mb[0], ma[i], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
#else
        threadgroup const mm_tile_t *lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const mm_tile_t *lsmb = (sb + 2 * 64 * (sgitg / 2));
        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
#endif

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#else
    // ══════════════════ DOUBLE-BUFFERED SOFTWARE PIPELINE (MOE_MM_DBUF) ══════════════════
    // The barrier between STAGE (Q4_K dequant → threadgroup tile) and MATMUL (simdgroup) serializes
    // them: dequant of chunk N must complete before the matmul of chunk N can read the tile, and the
    // matmul must complete before chunk N+1 overwrites the tile. The register-prefetch (above) can't
    // cross the barrier (it hides only the GLOBAL-load latency, not the dequant→matmul dependency).
    //
    // FIX: two threadgroup tile-pairs [sa0|sb0 | sa1|sb1]. Stage chunk N+1 into buffer B while the
    // matmul of chunk N runs from buffer A — the STAGE(N+1) and MATMUL(N) sit between the SAME pair of
    // barriers, so the GPU overlaps the dequant ALU/loads with the simdgroup matrix units. Only the
    // BUFFER a thread touches changes per chunk; every value staged and every matmul op is BIT-
    // IDENTICAL to the default path (same HOIST dequant `s*nib+lo`, same tile layout, same op order).
    //
    // Layout: sa0=sa (shmem+0), sb0=sb (shmem+NR0*NK*t), sa1 (shmem+(NR0+NR1)*NK*t), sb1 (+more).
    // Budget (half tile, NR1=32): sa 4K + sb 2K per pair ×2 = 12K; writeback sc = 8K reuses the front.
    // Host allocates max(12K, 8K) = 12K (concurrent_metal.rs already passes 12288). NR1=8 → 9K, fits.
    threadgroup mm_tile_t *sa1 = (threadgroup mm_tile_t *)(shmem + (NR0 + NR1) * NK * sizeof(mm_tile_t));
    threadgroup mm_tile_t *sb1 = (threadgroup mm_tile_t *)(shmem + (NR0 + NR1) * NK * sizeof(mm_tile_t) + NR0 * NK * sizeof(mm_tile_t));

    // STAGE one K-chunk's weight sub-block into tile SA_ — the HOIST dequant math, VERBATIM from the
    // default MOE_MM_HOIST path (lines above). PRESERVES `s*nib+lo` (mul-then-mul-add, NO fma/reorder).
    // KB = loop_k for this chunk; ILV = il for this chunk (constant il0 — the double-buffer advance is
    // a value-preserving no-op with MM_NL=2, kept for parity). SA_ = target weight tile (sa or sa1).
    #define MOE_STAGE_W(SA_, KB, ILV) do { \
        const uint _kbase  = (KB) + 16u * uint(ILV); \
        const uint _b      = _kbase >> 5; \
        const uint _sblk   = _b >> 3; \
        const uint _byte_b = _b & 3u; \
        const uint4 _qv    = (wcol_ok && _kbase < d.k) ? codes[wcol * subs + _b] : uint4(0); \
        const uint _ddbase = wcol * supers; \
        const float2 _dp   = float2(dd[_ddbase + _sblk]); \
        const float _dv    = _dp.x; \
        const float _dmv   = _dp.y; \
        const uint _sidx   = wcol * subs + _b; \
        const float _su8   = float((scales[_sidx >> 2] >> (8u * _byte_b)) & 0xFFu); \
        const int  _mraw   = int(mins[_sidx >> 2] << (24u - 8u * _byte_b)) >> 24; \
        const float _s     = _dv * _su8; \
        const float _lo    = _dmv * float(_mraw); \
        FOR_UNROLL (short _i = 0; _i < 16; _i++) { \
            const short _sx = 2 * il0 + _i / 8; \
            const short _sy = (tiitg / NL0) / 8; \
            const short _lx = (tiitg / NL0) % 8; \
            const short _ly = _i % 8; \
            const short _ib = 8 * _sx + _sy; \
            const uint _within = (_kbase & 31u) + uint(_i); \
            const uint _w = (_within < 8u) ? _qv.x : (_within < 16u) ? _qv.y : (_within < 24u) ? _qv.z : _qv.w; \
            const float _nib = float((_w >> (4u * (_within & 7u))) & 0xFu); \
            mm_tile_t _wv = MM_TZERO; \
            if (wcol_ok && (_kbase + uint(_i)) < d.k) _wv = (mm_tile_t)(_s * _nib + _lo); \
            *((SA_) + 64 * _ib + 8 * _ly + _lx) = _wv; \
        } \
    } while (0)

    // STAGE one K-chunk's activation gather into tile SB_ — VERBATIM from the default activation path.
    #define MOE_STAGE_A(SB_, KB) do { \
        const short _sx = (tiitg % NL1); \
        const short _sy = (tiitg / NL1) / 8; \
        const short _ly = (tiitg / NL1) % 8; \
        const short _ib = 4 * _sx + _sy; \
        FOR_UNROLL (short _i = 0; _i < 8; _i++) { \
            const short _lx = _i; \
            mm_tile_t _av = MM_TZERO; \
            const uint _kelem = (KB) + uint(iy) + uint(_i); \
            if (_kelem < d.k) _av = (mm_tile_t) a[src_row * arow + _kelem]; \
            *((SB_) + 64 * _ib + 8 * _ly + _lx) = _av; \
        } \
    } while (0)

    // MATMUL of one staged K-chunk from tiles SA_/SB_ into the mc[] accumulators — VERBATIM from the
    // default simdgroup-matmul block (both NR1 variants). Reads only from the passed tile pair.
#if MOE_MM_NR1 == 8
    #define MOE_MATMUL(SA_, SB_) do { \
        threadgroup const mm_tile_t *_lsma = ((SA_) + 2 * 64 * sgitg); \
        threadgroup const mm_tile_t *_lsmb = (SB_); \
        FOR_UNROLL (short _ik = 0; _ik < NK / 8; _ik++) { \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL (short _i = 0; _i < 2; _i++) simdgroup_load(ma[_i], _lsma + 64 * _i, 8, 0, false); \
            simdgroup_barrier(mem_flags::mem_none); \
            simdgroup_load(mb[0], _lsmb, 8, 0, false); \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL (short _i = 0; _i < 2; _i++) simdgroup_multiply_accumulate(mc[_i], mb[0], ma[_i], mc[_i]); \
            _lsma += 8 * 64; \
            _lsmb += 4 * 64; \
        } \
    } while (0)
#else
    #define MOE_MATMUL(SA_, SB_) do { \
        threadgroup const mm_tile_t *_lsma = ((SA_) + 4 * 64 * (sgitg % 2)); \
        threadgroup const mm_tile_t *_lsmb = ((SB_) + 2 * 64 * (sgitg / 2)); \
        FOR_UNROLL (short _ik = 0; _ik < NK / 8; _ik++) { \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL (short _i = 0; _i < 4; _i++) simdgroup_load(ma[_i], _lsma + 64 * _i, 8, 0, false); \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL (short _i = 0; _i < 2; _i++) simdgroup_load(mb[_i], _lsmb + 64 * _i, 8, 0, false); \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL (short _i = 0; _i < 8; _i++) simdgroup_multiply_accumulate(mc[_i], mb[_i / 4], ma[_i % 4], mc[_i]); \
            _lsma += 8 * 64; \
            _lsmb += 4 * 64; \
        } \
    } while (0)
#endif

    // PROLOGUE: stage chunk 0 into buffer A (sa0/sb0). `il` tracks the sub-block half (== il0 always
    // under MM_NL=2; the advance below is a no-op in value, kept for llama structural parity).
    if (d.k > 0u) {
        MOE_STAGE_W(sa,  0u, il);
        MOE_STAGE_A(sb,  0u);
        il = (il + 2 < MM_NL) ? il + 2 : il % 2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // STEADY STATE: for each chunk, MATMUL the CURRENT buffer and STAGE the NEXT chunk into the OTHER
    // buffer between the same barrier pair, so dequant(N+1) overlaps matmul(N). `cur`=0 → matmul A /
    // stage B; `cur`=1 → matmul B / stage A. Swap after each chunk.
    bool cur = false;   // false → current tile is A (sa/sb), stage into B (sa1/sb1)
    for (uint loop_k = 0; loop_k + NK < d.k; loop_k += NK) {
        const uint next_k = loop_k + NK;
        if (!cur) {
            MOE_STAGE_W(sa1, next_k, il);   // stage N+1 into buffer B
            MOE_STAGE_A(sb1, next_k);
            il = (il + 2 < MM_NL) ? il + 2 : il % 2;
            MOE_MATMUL(sa, sb);             // matmul N from buffer A (overlaps the stage above)
        } else {
            MOE_STAGE_W(sa,  next_k, il);   // stage N+1 into buffer A
            MOE_STAGE_A(sb,  next_k);
            il = (il + 2 < MM_NL) ? il + 2 : il % 2;
            MOE_MATMUL(sa1, sb1);           // matmul N from buffer B
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur = !cur;
    }

    // EPILOGUE: matmul the last-staged chunk (which lives in whichever buffer `cur` points at).
    if (d.k > 0u) {
        if (!cur) MOE_MATMUL(sa,  sb);
        else      MOE_MATMUL(sa1, sb1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #undef MOE_STAGE_W
    #undef MOE_STAGE_A
    #undef MOE_MATMUL
#endif  // MOE_MM_DBUF

    // ---- writeback: store mc[] to the [NR1 × NR0] row-major C tile in shmem, then SCATTER ----
#ifdef MOE_MM_LOWSMEM
    // ── LOW-SMEM WRITEBACK (MOE_MM_LOWSMEM, ARF_MOE_MM_LOWSMEM) ─────────────────────────────
    // The default writeback stages the FULL [NR1×NR0] float tile (8KB) then scatters — that 8KB
    // sets the kernel's PEAK threadgroup memory (host had to alloc 12288), capping occupancy at ~2
    // tiles/core so too few warps are in flight to hide the ~400ns dependent-load latency (the
    // conc64 stall). FIX: each simdgroup stages ONLY its own fragments into a SMALL per-sg slot and
    // scatters them itself — no sg reads another sg's data, so the full tile never needs to be
    // resident. Each sg owns rows [R, R+16) × cols [Cb, Cb+32) of the result (SAME ownership as the
    // default store, mc[i] → tile(R+8*(i/4), Cb+8*(i%4))); it scatters exactly those cells to the
    // SAME c[] addresses with the SAME values. Peak writeback smem = 4 sg × (8 rows × 32 cols × 4B)
    // = 4KB < the 6KB matmul staging → kernel PEAK drops to 6KB (matmul), raising occupancy. Split
    // per sg into 2 passes of 8 rows (mc[4*p..4*p+3]) so the per-sg slot is only 8×32×4 = 1KB.
    // BIT-IDENTICAL: the mc[] accumulator values are untouched; simdgroup_store writes the same
    // float each mc lane produced; the scatter writes value tile(row,col) → c[i_rs(row)*n+r0+col],
    // the SAME (value → address) pair the default path produces (no FP op reordered).
    threadgroup_barrier(mem_flags::mem_threadgroup);   // matmul reads of sa/sb (offset 0) must finish before ws stores reuse it
#if MOE_MM_NR1 == 8
    // NR1=8: all 4 sg share the 8 rows; sg owns cols [16*sgitg,16*sgitg+16). One pass, 8×16 slot
    // (512B/sg, 2KB total). Store mc[0..1] (2 col-frags) into the per-sg slot, then scatter.
    threadgroup float *ws = ((threadgroup float *) shmem) + 128 * sgitg;   // 8*16 floats/sg
    FOR_UNROLL (short i = 0; i < 2; i++) {
        simdgroup_store(mc[i], ws + 8 * i, 16, 0, false);   // slot stride = 16 (this sg's width)
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);   // simdgroup_store → this sg's scalar reads of ws (threadgroup-mem fence)
    const short cbase = 16 * sgitg;
    // 8 rows × 16 cols = 128 cells; thread tiisg walks cells tiisg, tiisg+32, ... (4 each).
    for (short e = short(tiisg); e < 128; e += 32) {
        const short lrow = e >> 4;          // 0..7  (row within the 8-row tile)
        const short lcol = e & 15;          // 0..15 (col within this sg's 16-col span)
        const short col  = cbase + lcol;
        if (lrow < nr1 && col < nr0) {
            const uint i_rs = csr_row_slot[csr_lo + uint(r1 + lrow)];
            ((device float *) c)[i_rs * d.n + uint(r0 + col)] = ws[lrow * 16 + lcol];
        }
    }
#else
    // NR1=32: sg owns rows [R,R+16) × cols [Cb,Cb+32), R=16*(sgitg>>1), Cb=32*(sgitg&1). Two passes
    // of 8 rows each (mc[4*p+0..3] = 4 col-frags across the 32-col span); 8×32 slot = 1KB/sg (4KB).
    threadgroup float *ws = ((threadgroup float *) shmem) + 256 * sgitg;   // 8*32 floats/sg
    const short R  = 16 * (sgitg >> 1);
    const short Cb = 32 * (sgitg & 1);
    FOR_UNROLL (short p = 0; p < 2; p++) {
        FOR_UNROLL (short i = 0; i < 4; i++) {
            simdgroup_store(mc[4 * p + i], ws + 8 * i, 32, 0, false);   // slot stride = 32
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);   // simdgroup_store → this sg's scalar reads of ws (threadgroup-mem fence)
        // 8 rows × 32 cols = 256 cells; thread tiisg walks cells tiisg, tiisg+32, ... (8 each).
        for (short e = short(tiisg); e < 256; e += 32) {
            const short lrow = e >> 5;              // 0..7  (row within this pass's 8-row block)
            const short lcol = e & 31;              // 0..31 (col within this sg's 32-col span)
            const short row  = R + 8 * p + lrow;    // absolute tile row
            const short col  = Cb + lcol;           // absolute tile col
            if (row < nr1 && col < nr0) {
                const uint i_rs = csr_row_slot[csr_lo + uint(r1 + row)];
                ((device float *) c)[i_rs * d.n + uint(r0 + col)] = ws[lrow * 32 + lcol];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);   // pass-1 reads of ws must finish before pass-2 stores overwrite it
    }
#endif
#else
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if MOE_MM_NR1 == 8
    // each sg owns cols [16*sgitg, 16*sgitg+16) of the 8 rows; 2 col-frags at 8-col offsets.
    threadgroup float *temp_str = ((threadgroup float *) shmem) + 16 * sgitg;
    FOR_UNROLL (short i = 0; i < 2; i++) {
        simdgroup_store(mc[i], temp_str + 8 * i, NR0, 0, false);
    }
#else
    threadgroup float *temp_str = ((threadgroup float *) shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    FOR_UNROLL (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // one simdgroup per row-block; scatter row j's NR0 cols back to (row,slot,col)-major output.
    for (short j = sgitg; j < nr1; j += 4) {
        const uint i_rs = csr_row_slot[csr_lo + uint(r1 + j)];   // i = row*top_k + slot
        device float *D = (device float *) c + i_rs * d.n + r0;
        threadgroup const float *C = (threadgroup const float *) shmem + j * NR0;
        for (short i = tiisg; i < nr0; i += 32) {
            D[i] = C[i];
        }
    }
#endif  // MOE_MM_LOWSMEM
}

// ===================== grouped down tiled mm_id (Q4_K_S) → down_slots =====================
// Same body as gu but k = inter, n = h, reading silu_all[(row*top_k+slot)*inter + ...] and writing
// down_slots[(row*top_k+slot)*h + h_j] WITHOUT folding the router weight and WITHOUT summing slots
// (PLACE — distinct cell per (row,slot)). The slot-sum + router fold is moe_reduce_slots_b's job.
// The activation gather differs: the down input is the (src_row,slot) silu segment, base =
// (src_row*top_k + slot)*inter — recover BOTH row AND slot from the CSR source index i.
kernel void moe_mm_id_down_q4ks(
        device const float  *a       [[buffer(0)]],   // [B, top_k, k] silu_all (k = inter)
        device const uint4  *codes   [[buffer(1)]],   // all experts' down codes
        device       float  *c       [[buffer(2)]],   // [B, top_k, n] down_slots (n = h)
        constant     MmIdDims &d     [[buffer(3)]],
        device const uint   *scales  [[buffer(4)]],
        device const uint   *mins    [[buffer(5)]],
        device const half2  *dd      [[buffer(6)]],
        device const uint   *csr_offsets [[buffer(7)]],
        device const uint   *csr_row_slot[[buffer(8)]],
        device const uint   *csr_active  [[buffer(9)]],  // unused
        device const uint   *csr_meta    [[buffer(10)]], // [0]=n_active, [1]=n_tiles
        device const uint   *tile_list   [[buffer(11)]], // [max_tiles] packed (expert<<16)|tile_idx
        uint3  tgpig                 [[threadgroup_position_in_grid]],
        ushort tiitg                 [[thread_index_in_threadgroup]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]],
        threadgroup char *shmem      [[threadgroup(0)]]) {
    threadgroup mm_tile_t *sa = (threadgroup mm_tile_t *)(shmem);
    threadgroup mm_tile_t *sb = (threadgroup mm_tile_t *)(shmem + NR0 * NK * sizeof(mm_tile_t));

    const uint subs   = d.k / 32u;          // sub-blocks per down-weight column (k = inter)
    const uint supers = d.k / 256u;
    const uint slot_seg = d.k;              // floats per slot's silu segment (= inter)
    const uint row_seg  = d.top_k * slot_seg;  // floats per batch row's silu (top_k*inter)

    if (tgpig.x >= csr_meta[1]) return;
    const uint packed   = tile_list[tgpig.x];
    const uint expert   = packed >> 16;
    const uint tile_idx = packed & 0xFFFFu;
    const uint csr_lo   = csr_offsets[expert];
    const uint csr_hi   = csr_offsets[expert + 1u];
    const int  neh1     = int(csr_hi - csr_lo);

    const int r1 = int(tile_idx) * NR1;
    if (r1 >= neh1) return;

    const int r0 = int(tgpig.y) * NR0;
    const uint exp_col_base = expert * d.n; // n = h: expert's down-weight block

    const short nr0 = (int(d.n) - r0 < NR0) ? short(int(d.n) - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : nr1 - 1;
    const short il0 = (tiitg % NL0);
    short il = il0;

    const uint wcol = exp_col_base + uint(r0 + lr0);
    const bool wcol_ok = (r0 + lr0) < int(d.n);

    // activation gather: recover (src_row, slot) from the CSR source index, silu base =
    // src_row*row_seg + slot*slot_seg.
    const uint grow = uint(r1 + lr1);
    const uint isrc = csr_row_slot[csr_lo + grow];   // i = row*top_k + slot
    const uint src_row = isrc / d.top_k;
    const uint slot    = isrc % d.top_k;
    const uint sbase = src_row * row_seg + slot * slot_seg;
    const short iy = 8 * (tiitg % NL1);

#if MOE_MM_NR1 == 8
    mm_mat_t  ma[2];
    mm_mat_t  mb[1];
    simdgroup_float8x8 mc[2];
    for (short i = 0; i < 2; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
#else
    mm_mat_t  ma[4];
    mm_mat_t  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
#endif

#ifndef MOE_MM_DBUF
    for (uint loop_k = 0; loop_k < d.k; loop_k += NK) {
        {
            const uint kbase = loop_k + 16u * uint(il);
#ifdef MOE_MM_HOIST
            // BLOCK-DEQUANT HOIST (down) — bit-for-bit identical to the gate/up hoist above.
            const uint b      = kbase >> 5;
            const uint sblk   = b >> 3;
            const uint byte_b = b & 3u;
            const uint4 qv    = (wcol_ok && kbase < d.k) ? codes[wcol * subs + b] : uint4(0);
            const uint ddbase = wcol * supers;
            const float2 dp   = float2(dd[ddbase + sblk]);
            const float dv    = dp.x;
            const float dmv   = dp.y;
            const uint sidx   = wcol * subs + b;
            const float su8   = float((scales[sidx >> 2] >> (8u * byte_b)) & 0xFFu);
            const int   mraw  = int(mins[sidx >> 2] << (24u - 8u * byte_b)) >> 24;
            const float s     = dv * su8;
            const float lo    = dmv * float(mraw);
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                const uint within = (kbase & 31u) + uint(i);
                const uint w = (within < 8u) ? qv.x : (within < 16u) ? qv.y : (within < 24u) ? qv.z : qv.w;
                const float nib = float((w >> (4u * (within & 7u))) & 0xFu);
                mm_tile_t wv = MM_TZERO;
                if (wcol_ok && (kbase + uint(i)) < d.k) wv = (mm_tile_t)(s * nib + lo);
                *(sa + 64 * ib + 8 * ly + lx) = wv;
            }
#else
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                mm_tile_t wv = MM_TZERO;
                const uint kelem = kbase + uint(i);
                if (wcol_ok && kelem < d.k) {
                    wv = (mm_tile_t) dequant_q4ks_elem(codes, scales, mins, dd, wcol, kelem, subs, supers);
                }
                *(sa + 64 * ib + 8 * ly + lx) = wv;
            }
#endif
        }
        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            FOR_UNROLL (short i = 0; i < 8; i++) {
                const short lx = i;
                mm_tile_t av = MM_TZERO;
                const uint kelem = loop_k + uint(iy) + uint(i);
                if (kelem < d.k) av = (mm_tile_t) a[sbase + kelem];
                *(sb + 64 * ib + 8 * ly + lx) = av;
            }
        }

        il = (il + 2 < MM_NL) ? il + 2 : il % 2;

        threadgroup_barrier(mem_flags::mem_threadgroup);

#if MOE_MM_NR1 == 8
        // down kernel: identical NR1=8 fragment re-partition as gate/up (see that kernel's note).
        threadgroup const mm_tile_t *lsma = (sa + 2 * 64 * sgitg);
        threadgroup const mm_tile_t *lsmb = (sb);
        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb[0], lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) simdgroup_multiply_accumulate(mc[i], mb[0], ma[i], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
#else
        threadgroup const mm_tile_t *lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const mm_tile_t *lsmb = (sb + 2 * 64 * (sgitg / 2));
        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
#endif

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#else
    // ══════════════════ DOUBLE-BUFFERED SOFTWARE PIPELINE (MOE_MM_DBUF) — down ══════════════════
    // Identical pipeline to the gate/up kernel above; the ONLY difference is the activation gather
    // reads the (src_row,slot) silu segment `a[sbase + kelem]` (down input) instead of `a[src_row*
    // arow + kelem]`. STAGE_W / MATMUL / barrier structure are bit-identical to the gate/up DBUF path.
    threadgroup mm_tile_t *sa1 = (threadgroup mm_tile_t *)(shmem + (NR0 + NR1) * NK * sizeof(mm_tile_t));
    threadgroup mm_tile_t *sb1 = (threadgroup mm_tile_t *)(shmem + (NR0 + NR1) * NK * sizeof(mm_tile_t) + NR0 * NK * sizeof(mm_tile_t));

    // STAGE weight sub-block into SA_ — HOIST dequant math, VERBATIM (preserves `s*nib+lo`, no fma).
    #define MOE_STAGE_W(SA_, KB, ILV) do { \
        const uint _kbase  = (KB) + 16u * uint(ILV); \
        const uint _b      = _kbase >> 5; \
        const uint _sblk   = _b >> 3; \
        const uint _byte_b = _b & 3u; \
        const uint4 _qv    = (wcol_ok && _kbase < d.k) ? codes[wcol * subs + _b] : uint4(0); \
        const uint _ddbase = wcol * supers; \
        const float2 _dp   = float2(dd[_ddbase + _sblk]); \
        const float _dv    = _dp.x; \
        const float _dmv   = _dp.y; \
        const uint _sidx   = wcol * subs + _b; \
        const float _su8   = float((scales[_sidx >> 2] >> (8u * _byte_b)) & 0xFFu); \
        const int  _mraw   = int(mins[_sidx >> 2] << (24u - 8u * _byte_b)) >> 24; \
        const float _s     = _dv * _su8; \
        const float _lo    = _dmv * float(_mraw); \
        FOR_UNROLL (short _i = 0; _i < 16; _i++) { \
            const short _sx = 2 * il0 + _i / 8; \
            const short _sy = (tiitg / NL0) / 8; \
            const short _lx = (tiitg / NL0) % 8; \
            const short _ly = _i % 8; \
            const short _ib = 8 * _sx + _sy; \
            const uint _within = (_kbase & 31u) + uint(_i); \
            const uint _w = (_within < 8u) ? _qv.x : (_within < 16u) ? _qv.y : (_within < 24u) ? _qv.z : _qv.w; \
            const float _nib = float((_w >> (4u * (_within & 7u))) & 0xFu); \
            mm_tile_t _wv = MM_TZERO; \
            if (wcol_ok && (_kbase + uint(_i)) < d.k) _wv = (mm_tile_t)(_s * _nib + _lo); \
            *((SA_) + 64 * _ib + 8 * _ly + _lx) = _wv; \
        } \
    } while (0)

    // STAGE activation gather into SB_ — down input `a[sbase + kelem]`, VERBATIM.
    #define MOE_STAGE_A(SB_, KB) do { \
        const short _sx = (tiitg % NL1); \
        const short _sy = (tiitg / NL1) / 8; \
        const short _ly = (tiitg / NL1) % 8; \
        const short _ib = 4 * _sx + _sy; \
        FOR_UNROLL (short _i = 0; _i < 8; _i++) { \
            const short _lx = _i; \
            mm_tile_t _av = MM_TZERO; \
            const uint _kelem = (KB) + uint(iy) + uint(_i); \
            if (_kelem < d.k) _av = (mm_tile_t) a[sbase + _kelem]; \
            *((SB_) + 64 * _ib + 8 * _ly + _lx) = _av; \
        } \
    } while (0)

#if MOE_MM_NR1 == 8
    #define MOE_MATMUL(SA_, SB_) do { \
        threadgroup const mm_tile_t *_lsma = ((SA_) + 2 * 64 * sgitg); \
        threadgroup const mm_tile_t *_lsmb = (SB_); \
        FOR_UNROLL (short _ik = 0; _ik < NK / 8; _ik++) { \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL (short _i = 0; _i < 2; _i++) simdgroup_load(ma[_i], _lsma + 64 * _i, 8, 0, false); \
            simdgroup_barrier(mem_flags::mem_none); \
            simdgroup_load(mb[0], _lsmb, 8, 0, false); \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL (short _i = 0; _i < 2; _i++) simdgroup_multiply_accumulate(mc[_i], mb[0], ma[_i], mc[_i]); \
            _lsma += 8 * 64; \
            _lsmb += 4 * 64; \
        } \
    } while (0)
#else
    #define MOE_MATMUL(SA_, SB_) do { \
        threadgroup const mm_tile_t *_lsma = ((SA_) + 4 * 64 * (sgitg % 2)); \
        threadgroup const mm_tile_t *_lsmb = ((SB_) + 2 * 64 * (sgitg / 2)); \
        FOR_UNROLL (short _ik = 0; _ik < NK / 8; _ik++) { \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL (short _i = 0; _i < 4; _i++) simdgroup_load(ma[_i], _lsma + 64 * _i, 8, 0, false); \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL (short _i = 0; _i < 2; _i++) simdgroup_load(mb[_i], _lsmb + 64 * _i, 8, 0, false); \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL (short _i = 0; _i < 8; _i++) simdgroup_multiply_accumulate(mc[_i], mb[_i / 4], ma[_i % 4], mc[_i]); \
            _lsma += 8 * 64; \
            _lsmb += 4 * 64; \
        } \
    } while (0)
#endif

    if (d.k > 0u) {
        MOE_STAGE_W(sa,  0u, il);
        MOE_STAGE_A(sb,  0u);
        il = (il + 2 < MM_NL) ? il + 2 : il % 2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool cur = false;
    for (uint loop_k = 0; loop_k + NK < d.k; loop_k += NK) {
        const uint next_k = loop_k + NK;
        if (!cur) {
            MOE_STAGE_W(sa1, next_k, il);
            MOE_STAGE_A(sb1, next_k);
            il = (il + 2 < MM_NL) ? il + 2 : il % 2;
            MOE_MATMUL(sa, sb);
        } else {
            MOE_STAGE_W(sa,  next_k, il);
            MOE_STAGE_A(sb,  next_k);
            il = (il + 2 < MM_NL) ? il + 2 : il % 2;
            MOE_MATMUL(sa1, sb1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur = !cur;
    }

    if (d.k > 0u) {
        if (!cur) MOE_MATMUL(sa,  sb);
        else      MOE_MATMUL(sa1, sb1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #undef MOE_STAGE_W
    #undef MOE_STAGE_A
    #undef MOE_MATMUL
#endif  // MOE_MM_DBUF

    // ---- writeback: store mc[] to the [NR1 × NR0] row-major C tile, then SCATTER (PLACE) ----
#ifdef MOE_MM_LOWSMEM
    // LOW-SMEM WRITEBACK (down) — bit-for-bit identical restructure to the gate/up kernel above.
    // Each sg stages ONLY its own fragments into a small per-sg slot and scatters them itself, so
    // the 8KB full-tile stage is avoided (peak drops to the 6KB matmul staging → higher occupancy).
    // Same c[] address = i_rs*d.n + r0 + col (PLACE, distinct cell per (row,slot)), same values.
    threadgroup_barrier(mem_flags::mem_threadgroup);   // matmul reads of sa/sb (offset 0) must finish before ws stores reuse it
#if MOE_MM_NR1 == 8
    threadgroup float *ws = ((threadgroup float *) shmem) + 128 * sgitg;   // 8*16 floats/sg
    FOR_UNROLL (short i = 0; i < 2; i++) {
        simdgroup_store(mc[i], ws + 8 * i, 16, 0, false);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);   // simdgroup_store → this sg's scalar reads of ws (threadgroup-mem fence)
    const short cbase = 16 * sgitg;
    for (short e = short(tiisg); e < 128; e += 32) {
        const short lrow = e >> 4;
        const short lcol = e & 15;
        const short col  = cbase + lcol;
        if (lrow < nr1 && col < nr0) {
            const uint i_rs = csr_row_slot[csr_lo + uint(r1 + lrow)];
            ((device float *) c)[i_rs * d.n + uint(r0 + col)] = ws[lrow * 16 + lcol];
        }
    }
#else
    threadgroup float *ws = ((threadgroup float *) shmem) + 256 * sgitg;   // 8*32 floats/sg
    const short R  = 16 * (sgitg >> 1);
    const short Cb = 32 * (sgitg & 1);
    FOR_UNROLL (short p = 0; p < 2; p++) {
        FOR_UNROLL (short i = 0; i < 4; i++) {
            simdgroup_store(mc[4 * p + i], ws + 8 * i, 32, 0, false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);   // simdgroup_store → this sg's scalar reads of ws (threadgroup-mem fence)
        for (short e = short(tiisg); e < 256; e += 32) {
            const short lrow = e >> 5;
            const short lcol = e & 31;
            const short row  = R + 8 * p + lrow;
            const short col  = Cb + lcol;
            if (row < nr1 && col < nr0) {
                const uint i_rs = csr_row_slot[csr_lo + uint(r1 + row)];
                ((device float *) c)[i_rs * d.n + uint(r0 + col)] = ws[lrow * 32 + lcol];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);   // pass-1 reads of ws must finish before pass-2 stores overwrite it
    }
#endif
#else
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if MOE_MM_NR1 == 8
    threadgroup float *temp_str = ((threadgroup float *) shmem) + 16 * sgitg;
    FOR_UNROLL (short i = 0; i < 2; i++) {
        simdgroup_store(mc[i], temp_str + 8 * i, NR0, 0, false);
    }
#else
    threadgroup float *temp_str = ((threadgroup float *) shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    FOR_UNROLL (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const uint i_rs = csr_row_slot[csr_lo + uint(r1 + j)];   // i = row*top_k + slot
        device float *D = (device float *) c + i_rs * d.n + r0;  // PLACE (distinct cell per (row,slot))
        threadgroup const float *C = (threadgroup const float *) shmem + j * NR0;
        for (short i = tiisg; i < nr0; i += 32) {
            D[i] = C[i];
        }
    }
#endif  // MOE_MM_LOWSMEM
}
