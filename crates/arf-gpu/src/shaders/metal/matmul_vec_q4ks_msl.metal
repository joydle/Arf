// Hand-tuned MSL Q4_K_S decode GEMV for the native Metal island — the REAL gemma-4-12B
// decode format (codes + u8 sub-scales + i8 sub-mins + per-super-block half2 d/dmin pairs).
// Bit-exact port of matmul_vec_q4ks.wgsl's math, with llama.cpp's roofline techniques:
//   - 2 output COLUMNS per simdgroup (activation sub-block loaded ONCE, reused across cols)
//   - pure simd_sum reduction (no threadgroup-barrier tree)
//   - 32-lane simdgroup strides the sub-blocks.
//
// c[col] = Σ_subblocks ( (d·u8scale)·Σ(nibble·a) + (dmin·i8min)·Σ(a) ).
// Bindings match matmul_vec_q4ks.wgsl: 0 a[k](f32x4), 1 codes(uint4/sub-block),
// 2 c[n], 3 dims{m,k,n,_}, 4 scales(u8×4/word), 5 mins(i8×4/word), 6 dd(half2 {d,dmin}/super).

#include <metal_stdlib>
using namespace metal;

struct Dims { uint m; uint k; uint n; uint _pad; };

// Output columns per simdgroup. Function constant so the host can compile a variant per shape
// (COLS=3/4 for the wide gemma FFN n=15360) WITHOUT branching in the hot loop. CRITICAL: a
// referenced function constant with NO default makes the PLAIN compile_msl path FAIL to build the
// pipeline (Metal requires newFunctionWithName:constantValues:). So we guard with
// is_function_constant_defined() → when the host passes NO constants (the default compile_msl
// path every model uses), COLS folds to 2u at compile time and the kernel builds normally,
// byte-identical to the pre-parametrization kernel. Max 4 (acc[4]).
constant uint COLS_FC [[function_constant(0)]];
constant uint COLS = is_function_constant_defined(COLS_FC) ? COLS_FC : 2u;

// ARF_Q4_DOTY (function_constant 2): dot-y nibble fold ported from the shipped M1 conc16 MoE
// lever (moe_batch_msl.metal:58-63). Instead of shifting each nibble down to bit 0 (24 down-shifts
// per uint4), keep nibbles MASKED-IN-PLACE (w & 0xF0 = nib·16, & 0xF00 = nib·256, & 0xF000 = nib·
// 4096 — all EXACT in fp32 since nib·4096 <= 61440 < 2^24) and pre-scale the activation ONCE by
// CORR = (1, 1/16, 1/256, 1/4096) (exact powers of two). dot(av·CORR, masked) == dot(av, shifted)
// bit-for-bit. Guarded so the plain compile folds to false → the shipped shift path is byte-exact.
constant bool DOTY_FC [[function_constant(2)]];
constant bool DOTY = is_function_constant_defined(DOTY_FC) ? DOTY_FC : false;

// ARF_GEMV_NSG (function_constant 3): simdgroups per threadgroup. Default 1 (32 threads, the
// shipped path, byte-identical). NSG=2 (64 threads) splits the sub-block weight stream across TWO
// simdgroups → ~2× in-flight memory requests to HIDE the Q4_K weight-stream latency (the measured
// bandwidth lever — the FFN GEMV runs at only ~46-55% of the ~255 GB/s wall because a single
// simdgroup can't keep enough memory requests in flight). Each simdgroup sums a DISJOINT stride of
// sub-blocks (sgitg, sgitg+NSG, ...); the two partials are combined via threadgroup memory before
// the write. Below-tie f32 reorder (different sub-block partition) — parity-gated, NOT bit-exact.
constant uint NSG_FC [[function_constant(3)]];
constant uint NSG = is_function_constant_defined(NSG_FC) ? NSG_FC : 1u;

kernel void gemv_q4ks(
        device const float4 *a       [[buffer(0)]],
        device const uint4  *codes   [[buffer(1)]],
        device       float  *c       [[buffer(2)]],
        constant     Dims   &d       [[buffer(3)]],
        device const uint   *scales  [[buffer(4)]],  // u8 sub-scale, 4/word
        device const uint   *mins    [[buffer(5)]],  // i8 sub-min, 4/word
        device const half2  *dd      [[buffer(6)]],  // {d,dmin} per super-block (8 sub-blocks)
        uint  tgpig                  [[threadgroup_position_in_grid]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]]) {
    const uint subs = d.k / 32u;
    const uint col0 = tgpig * COLS;
    if (col0 >= d.n) return;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // NSG=1: simdgroup 0 strides all sub-blocks (b += 32). NSG=2: each simdgroup takes a disjoint
    // stride (b starts at sgitg*32 + tiisg, steps NSG*32) so the 64 lanes issue ~2× memory requests.
    for (uint b = sgitg * 32u + tiisg; b < subs; b += NSG * 32u) {
        const uint abase = 8u * b;
        float4 av[8];
        for (short i = 0; i < 8; ++i) av[i] = a[abase + i];
        float a_sum = 0.0f;
        for (short i = 0; i < 8; ++i) a_sum += av[i].x + av[i].y + av[i].z + av[i].w;
        // dot-y fold: pre-scale the activation cache ONCE per sub-block (amortized over COLS), so the
        // masked-in-place nibbles (nib·16^lane) dot to the same value as the shifted nibbles.
        float4 avc[8];
        if (DOTY) {
            const float4 CORR = float4(1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f);
            for (short i = 0; i < 8; ++i) avc[i] = av[i] * CORR;
        }

        const uint sblk = b / 8u;            // super-block index within this column
        const uint byte_b = b & 3u;          // which byte in the u8/i8 packed word
        const uint word_b = b >> 2u;         // which packed word

        for (uint cc = 0u; cc < COLS; ++cc) {
            const uint col = col0 + cc;
            if (col >= d.n) break;
            const uint idx = col * subs + b;
            const uint4 qv = codes[idx];

            // two-level scale: u8 sub-scale × super d ; i8 sub-min × super dmin
            const uint sbase = col * subs;           // sub-block units
            const uint ddbase = col * (subs / 8u);
            const float2 dp = float2(dd[ddbase + sblk]);
            const float dv  = dp.x;
            const float dmv = dp.y;
            const uint sword = scales[(sbase + b) >> 2];
            const uint mword = mins[(sbase + b) >> 2];
            const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
            // signed i8 via arithmetic shift
            const int  mraw = int(mword << (24u - 8u * byte_b)) >> 24;
            const float mi8 = float(mraw);
            const float s  = dv * su8;
            const float lo = dmv * mi8;
            (void)word_b;

            float code_sum = 0.0f;
            uint4 w = qv;
            if (DOTY) {
                // Masked-in-place nibbles (nib·16^lane) dotted against the CORR-prescaled activation.
                // hi = w>>16 (single shift for the high half), then mask in place — no per-nibble shift.
                const uint hx = w.x >> 16, hy = w.y >> 16, hz = w.z >> 16, hw = w.w >> 16;
                code_sum += dot(avc[0], float4(float(w.x & 0xFu), float(w.x & 0xF0u), float(w.x & 0xF00u), float(w.x & 0xF000u)))
                          + dot(avc[1], float4(float(hx & 0xFu), float(hx & 0xF0u), float(hx & 0xF00u), float(hx & 0xF000u)));
                code_sum += dot(avc[2], float4(float(w.y & 0xFu), float(w.y & 0xF0u), float(w.y & 0xF00u), float(w.y & 0xF000u)))
                          + dot(avc[3], float4(float(hy & 0xFu), float(hy & 0xF0u), float(hy & 0xF00u), float(hy & 0xF000u)));
                code_sum += dot(avc[4], float4(float(w.z & 0xFu), float(w.z & 0xF0u), float(w.z & 0xF00u), float(w.z & 0xF000u)))
                          + dot(avc[5], float4(float(hz & 0xFu), float(hz & 0xF0u), float(hz & 0xF00u), float(hz & 0xF000u)));
                code_sum += dot(avc[6], float4(float(w.w & 0xFu), float(w.w & 0xF0u), float(w.w & 0xF00u), float(w.w & 0xF000u)))
                          + dot(avc[7], float4(float(hw & 0xFu), float(hw & 0xF0u), float(hw & 0xF00u), float(hw & 0xF000u)));
            } else {
            float4 nlo, nhi;
            nlo = float4(float(w.x & 0xFu), float((w.x>>4)&0xFu), float((w.x>>8)&0xFu), float((w.x>>12)&0xFu));
            nhi = float4(float((w.x>>16)&0xFu), float((w.x>>20)&0xFu), float((w.x>>24)&0xFu), float((w.x>>28)&0xFu));
            code_sum += dot(av[0], nlo) + dot(av[1], nhi);
            nlo = float4(float(w.y & 0xFu), float((w.y>>4)&0xFu), float((w.y>>8)&0xFu), float((w.y>>12)&0xFu));
            nhi = float4(float((w.y>>16)&0xFu), float((w.y>>20)&0xFu), float((w.y>>24)&0xFu), float((w.y>>28)&0xFu));
            code_sum += dot(av[2], nlo) + dot(av[3], nhi);
            nlo = float4(float(w.z & 0xFu), float((w.z>>4)&0xFu), float((w.z>>8)&0xFu), float((w.z>>12)&0xFu));
            nhi = float4(float((w.z>>16)&0xFu), float((w.z>>20)&0xFu), float((w.z>>24)&0xFu), float((w.z>>28)&0xFu));
            code_sum += dot(av[4], nlo) + dot(av[5], nhi);
            nlo = float4(float(w.w & 0xFu), float((w.w>>4)&0xFu), float((w.w>>8)&0xFu), float((w.w>>12)&0xFu));
            nhi = float4(float((w.w>>16)&0xFu), float((w.w>>20)&0xFu), float((w.w>>24)&0xFu), float((w.w>>28)&0xFu));
            code_sum += dot(av[6], nlo) + dot(av[7], nhi);
            }

            acc[cc] += s * code_sum + lo * a_sum;
        }
    }

    if (NSG == 1u) {
        // Single simdgroup: reduce 32 lanes, lane 0 writes (the shipped path, unchanged).
        for (uint cc = 0u; cc < COLS; ++cc) {
            const uint col = col0 + cc;
            const float total = simd_sum(acc[cc]);
            if (tiisg == 0u && col < d.n) c[col] = total;
        }
    } else {
        // NSG simdgroups each hold a partial over their disjoint sub-block stride. simd_sum within
        // each simdgroup → sgpart[sgitg][cc]; then simdgroup 0 sums the NSG partials and writes.
        threadgroup float sgpart[8][4];   // up to 8 simdgroups × COLS(<=4)
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
                for (uint g = 0u; g < NSG; ++g) total += sgpart[g][cc];
                c[col] = total;
            }
        }
    }
}
