// Fused residual-add + RMSNorm (Llama/Qwen post-attention norm) for the megakernel — an MSL
// port of add_norm.wgsl. ONE threadgroup per row:
//   hidden[i] += residual[i]                         (store back to the residual stream)
//   normed[i]  = hidden[i] / sqrt(mean(hidden²)+eps) * weight[i]
// Bit-exact to add_norm.wgsl: same updated hidden, same strided sum-of-squares + tree reduce,
// same scale·weight. Unlike rmsnorm_add (which is `hidden += rmsnorm(src)·w`, the Gemma four-norm
// op), this is the Llama/Qwen flow: the residual is added FIRST, then the SUM is normalized into
// `normed` (the FFN input) while `hidden` keeps the un-normalized residual.
//
// Dispatch: grid (tokens, 1, 1), threadgroup 256 (one row per TG). dims {tokens, hidden, eps, _}.

#include <metal_stdlib>
using namespace metal;

struct AddNormDims { uint tokens; uint hidden; float eps; uint _pad; };

kernel void add_norm(
        device       float *hidden   [[buffer(0)]],  // read_write: += residual, feeds the reduce
        device const float *residual [[buffer(1)]],
        device const float *weight   [[buffer(2)]],
        device       float *normed   [[buffer(3)]],
        constant AddNormDims &d       [[buffer(4)]],
        uint  tgpig                  [[threadgroup_position_in_grid]],
        uint  lid                    [[thread_position_in_threadgroup]],
        uint  tg_size                [[threads_per_threadgroup]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig;
    const uint base = row * d.hidden;
    // Pass 1: add in place, accumulate sum-of-squares of the UPDATED value.
    float acc = 0.0f;
    for (uint i = lid; i < d.hidden; i += tg_size) {
        const float v = hidden[base + i] + residual[base + i];
        hidden[base + i] = v;
        acc += v * v;
    }
    acc = simd_sum(acc);
    threadgroup float sgp[32];
    if (tiisg == 0) sgp[sgitg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg_size + 31u) / 32u;
    if (lid == 0) { for (uint s = 0; s < nsg; ++s) total += sgp[s]; sgp[0] = total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = sgp[0];
    const float inv = 1.0f / sqrt(total / float(d.hidden) + d.eps);
    // Pass 3: write the normalized residual into `normed` (re-reads the updated hidden — it's
    // already published in DRAM via the in-place store above, same threadgroup).
    for (uint i = lid; i < d.hidden; i += tg_size) {
        normed[base + i] = hidden[base + i] * inv * weight[i];
    }
}

// Plain residual add in place: a[i] += b[i] (Llama/Qwen post-FFN residual, no norm). An MSL port
// of add.wgsl. Dispatch: grid (ceil(n/256),1,1), threadgroup 256. dims {n, _, _, _}.
struct AddDims { uint n; uint _p0; uint _p1; uint _p2; };

kernel void add_residual(
        device       float *a [[buffer(0)]],
        device const float *b [[buffer(1)]],
        constant AddDims   &d [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < d.n) a[gid] = a[gid] + b[gid];
}
