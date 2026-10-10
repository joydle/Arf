// Hand-tuned MSL Q3_K (3-bit split-plane) decode GEMV for the native Metal island — the
// bandwidth-floor breaker (~3.4 bits/weight vs Q4_K_S 4.75). Same roofline structure as
// gemv_q4ks: 2 output COLUMNS per simdgroup (activations loaded ONCE, reused), pure simd_sum,
// 32-lane simdgroup strides the sub-blocks. Bit-exact to the canonical Q3KMatrix oracle.
//
// c[col] = Σ_subblocks ( (d·u8scale)·Σ(code·a) + (dmin·i8min)·Σa ), code = (qh_bit<<2)|ql_2bits.
// Per 32-weight sub-block: ql = 2 u32 (16 weights, 2 bits each), qh = 1 u32 (32 weights, 1 bit).
// Bindings: 0 a[k](f32x4), 1 ql, 2 qh, 3 c[n], 4 dims{m,k,n}, 5 scales(u8×4), 6 mins(i8×4),
// 7 dd(f32 [d,dmin]/super).

#include <metal_stdlib>
using namespace metal;

struct Dims { uint m; uint k; uint n; uint _pad; };

kernel void gemv_q3k(
        device const float4 *a      [[buffer(0)]],
        device const uint   *ql     [[buffer(1)]],
        device const uint   *qh     [[buffer(2)]],
        device       float  *c      [[buffer(3)]],
        constant     Dims   &d      [[buffer(4)]],
        device const uint   *scales [[buffer(5)]],
        device const uint   *mins   [[buffer(6)]],
        device const float  *dd     [[buffer(7)]],
        uint  tgpig                 [[threadgroup_position_in_grid]],
        ushort tiisg                [[thread_index_in_simdgroup]]) {
    const uint subs = d.k / 32u;
    const uint COLS = 2u;
    const uint col0 = tgpig * COLS;
    if (col0 >= d.n) return;

    float acc[2] = {0.0f, 0.0f};

    for (uint b = tiisg; b < subs; b += 32u) {
        // load this sub-block's 32 activations ONCE (8 float4), reuse across both columns.
        const uint abase = 8u * b;
        float4 av[8];
        for (short i = 0; i < 8; ++i) av[i] = a[abase + i];
        float a_sum = 0.0f;
        for (short i = 0; i < 8; ++i) a_sum += av[i].x + av[i].y + av[i].z + av[i].w;

        const uint sblk = b / 8u;
        const uint byte_b = b & 3u;

        for (uint cc = 0u; cc < COLS; ++cc) {
            const uint col = col0 + cc;
            if (col >= d.n) break;
            const uint sbase = col * subs;
            const uint ddbase = col * (subs / 8u) * 2u;
            const float dv  = dd[ddbase + sblk * 2u];
            const float dmv = dd[ddbase + sblk * 2u + 1u];
            const uint sword = scales[(sbase + b) >> 2];
            const uint mword = mins[(sbase + b) >> 2];
            const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
            const int  mraw = int(mword << (24u - 8u * byte_b)) >> 24;
            const float s  = dv * su8;
            const float lo = dmv * float(mraw);

            // this column's codes: ql 2 words (16 weights each), qh 1 word (32 high bits).
            const uint ql0 = ql[col * (d.k / 16u) + b * 2u];
            const uint ql1 = ql[col * (d.k / 16u) + b * 2u + 1u];
            const uint qhw = qh[col * (d.k / 32u) + b];

            // Reconstruct 32 codes vectorized: 8 groups of 4. Group g (g<4 → ql0, else ql1),
            // 2-bit lows at (g%4)*8 + 2j, 1-bit highs at g*4 + j. code = (hi<<2)|lo.
            float code_sum = 0.0f;
            #pragma unroll
            for (uint g = 0u; g < 8u; ++g) {
                const uint qlw = (g < 4u) ? ql0 : ql1;
                const uint shl = (g & 3u) * 8u;
                const uint shh = g * 4u;
                const uint4 l = uint4(qlw >> shl, qlw >> (shl+2u), qlw >> (shl+4u), qlw >> (shl+6u)) & uint4(3u);
                const uint4 h = uint4(qhw >> shh, qhw >> (shh+1u), qhw >> (shh+2u), qhw >> (shh+3u)) & uint4(1u);
                const float4 code = float4((h << uint4(2u)) | l);
                code_sum += dot(av[g], code);
            }
            acc[cc] += s * code_sum + lo * a_sum;
        }
    }

    for (uint cc = 0u; cc < COLS; ++cc) {
        const uint col = col0 + cc;
        const float total = simd_sum(acc[cc]);
        if (tiisg == 0u && col < d.n) c[col] = total;
    }
}
