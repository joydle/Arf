// MSL RMSNorm (Llama/Gemma-style: no mean subtraction, no bias) for the per-token decode
// megakernel — bit-close port of rmsnorm.wgsl:  y = x / sqrt(mean(x²) + eps) * weight.
// One threadgroup per row (decode = 1 row). Reduction via simd_sum (simdgroup partial) +
// a threadgroup combine across simdgroups — the megakernel reduction idiom (same as the
// Q4_K_S GEMV). Reassoc differs from the WGSL tree but far below a logit tie; parity-gated
// vs a CPU oracle (the exact RmsNorm::forward math).
//
// Bindings match rmsnorm.wgsl: 0 x[hidden], 1 weight[hidden], 2 y[hidden],
// 3 dims {tokens, hidden, eps, _pad}.

#include <metal_stdlib>
using namespace metal;

struct Dims { uint tokens; uint hidden; float eps; uint _pad; };

kernel void rmsnorm(
        device const float *x       [[buffer(0)]],
        device const float *weight  [[buffer(1)]],
        device       float *y       [[buffer(2)]],
        constant     Dims  &d       [[buffer(3)]],
        uint  tgpig                 [[threadgroup_position_in_grid]],
        uint  lid                   [[thread_position_in_threadgroup]],
        uint  tg_size               [[threads_per_threadgroup]],
        ushort tiisg                [[thread_index_in_simdgroup]],
        ushort sgitg                [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig;
    const uint base = row * d.hidden;

    // Each lane sums squares of its strided slice.
    float acc = 0.0f;
    for (uint i = lid; i < d.hidden; i += tg_size) {
        const float v = x[base + i];
        acc += v * v;
    }
    // simdgroup partial, then combine across simdgroups via threadgroup memory.
    acc = simd_sum(acc);
    threadgroup float sg_partial[32]; // up to 32 simdgroups (1024/32)
    if (tiisg == 0) sg_partial[sgitg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg_size + 31u) / 32u;
    if (lid == 0) {
        for (uint s = 0; s < nsg; ++s) total += sg_partial[s];
        sg_partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = sg_partial[0];

    const float inv_rms = 1.0f / sqrt(total / float(d.hidden) + d.eps);
    for (uint i = lid; i < d.hidden; i += tg_size) {
        y[base + i] = x[base + i] * inv_rms * weight[i];
    }
}
