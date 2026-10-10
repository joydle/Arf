// Hand-tuned MSL Q4_K-lite decode GEMV for the native Metal island. Computes
// c[col] = sum_b ( scale[b]·dot(a_sub, dequant(code[b])) + min[b]·sum(a_sub) ) over the
// k/32 sub-blocks, for our dequant-on-read Q4_K layout (codes/scales/mins in SEPARATE
// buffers — NOT ggml's interleaved block_q4_K). Matches the kernel_mul_mv_q4_K techniques
// that get llama.cpp to ~70% roofline (vs our WGSL's in-frame 40%):
//   - 2 output COLUMNS per simdgroup (COLS_PER_SG): the activation sub-block is loaded
//     ONCE into registers and reused across both columns' weight reads (the dominant
//     activation-bandwidth amortization).
//   - pure simdgroup reduction via simd_sum (one op) — no threadgroup-barrier tree.
//   - one simdgroup (32 lanes) per column-pair; the 32 lanes stride the sub-blocks.
//
// Buffer order matches matmul_vec_q4k_sg.wgsl bindings:
//   0 a[k] (f32), 1 codes[n*subs] (uint4 = 32 nibbles/sub-block), 2 c[n] (f32),
//   3 dims {m,k,n,_pad} (uint4), 4 scales[n*subs] (bf16 in low 16 of u32),
//   5 mins[n*subs] (bf16 in low 16 of u32).
//
// PARITY: same math as the WGSL (s·code_sum + lo·a_sum, bf16→f32 scale/min), only the
// reduction order differs (simd_sum vs tree) — f32 reassoc noise far below a logit tie,
// covered by the decode greedy-parity gate.

#include <metal_stdlib>
using namespace metal;

inline float bf16_to_f32(uint bits) {
    return as_type<float>((bits & 0xFFFFu) << 16);
}

struct Dims { uint m; uint k; uint n; uint _pad; };

kernel void gemv_q4k(
        device const float4 *a       [[buffer(0)]],  // activation [k], 4/vec
        device const uint4  *codes   [[buffer(1)]],  // 32 nibbles per sub-block
        device       float  *c       [[buffer(2)]],
        constant     Dims   &d       [[buffer(3)]],
        device const uint   *scales  [[buffer(4)]],
        device const uint   *mins    [[buffer(5)]],
        uint  tgpig                  [[threadgroup_position_in_grid]],
        ushort tiisg                 [[thread_index_in_simdgroup]]) {
    const uint subs = d.k / 32u;
    const uint COLS = 2u;                    // columns per simdgroup
    const uint col0 = tgpig * COLS;          // this simdgroup's first output column
    if (col0 >= d.n) return;

    float acc[2] = {0.0f, 0.0f};

    // 32 lanes stride the sub-blocks; each lane reads its sub-block's 8 activation vec4
    // ONCE and reuses across both columns.
    for (uint b = tiisg; b < subs; b += 32u) {
        const uint abase = 8u * b;
        // load the sub-block's 32 activations (8 vec4) once
        float4 av[8];
        for (short i = 0; i < 8; ++i) av[i] = a[abase + i];
        float a_sum = 0.0f;
        for (short i = 0; i < 8; ++i) a_sum += av[i].x + av[i].y + av[i].z + av[i].w;

        for (uint cc = 0u; cc < COLS; ++cc) {
            const uint col = col0 + cc;
            if (col >= d.n) break;
            const uint idx = col * subs + b;
            const uint4 qv = codes[idx];
            const float s  = bf16_to_f32(scales[idx]);
            const float lo = bf16_to_f32(mins[idx]);
            // dequant nibbles (lo/hi of each u32 word) and dot with the 8 activation vec4
            float code_sum = 0.0f;
            // word.x → av0 (lo nibbles), av1 (hi nibbles); word.y → av2,av3; etc.
            uint4 w = qv;
            float4 nlo, nhi;
            // x
            nlo = float4(float(w.x & 0xFu), float((w.x>>4)&0xFu), float((w.x>>8)&0xFu), float((w.x>>12)&0xFu));
            nhi = float4(float((w.x>>16)&0xFu), float((w.x>>20)&0xFu), float((w.x>>24)&0xFu), float((w.x>>28)&0xFu));
            code_sum += dot(av[0], nlo) + dot(av[1], nhi);
            // y
            nlo = float4(float(w.y & 0xFu), float((w.y>>4)&0xFu), float((w.y>>8)&0xFu), float((w.y>>12)&0xFu));
            nhi = float4(float((w.y>>16)&0xFu), float((w.y>>20)&0xFu), float((w.y>>24)&0xFu), float((w.y>>28)&0xFu));
            code_sum += dot(av[2], nlo) + dot(av[3], nhi);
            // z
            nlo = float4(float(w.z & 0xFu), float((w.z>>4)&0xFu), float((w.z>>8)&0xFu), float((w.z>>12)&0xFu));
            nhi = float4(float((w.z>>16)&0xFu), float((w.z>>20)&0xFu), float((w.z>>24)&0xFu), float((w.z>>28)&0xFu));
            code_sum += dot(av[4], nlo) + dot(av[5], nhi);
            // w
            nlo = float4(float(w.w & 0xFu), float((w.w>>4)&0xFu), float((w.w>>8)&0xFu), float((w.w>>12)&0xFu));
            nhi = float4(float((w.w>>16)&0xFu), float((w.w>>20)&0xFu), float((w.w>>24)&0xFu), float((w.w>>28)&0xFu));
            code_sum += dot(av[6], nlo) + dot(av[7], nhi);

            acc[cc] += s * code_sum + lo * a_sum;
        }
    }

    // simdgroup reduction: each column's partial summed across the 32 lanes.
    for (uint cc = 0u; cc < COLS; ++cc) {
        const uint col = col0 + cc;
        const float total = simd_sum(acc[cc]);
        if (tiisg == 0u && col < d.n) c[col] = total;
    }
}
