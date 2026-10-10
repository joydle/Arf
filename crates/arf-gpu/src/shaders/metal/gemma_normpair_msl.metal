// FUSED Gemma dense norm-pair (post-attn rmsnorm_add + pre-FFN rmsnorm) in ONE dispatch.
//
// The gemma dense flow between attention and FFN is two separate ops today:
//   (10) rmsnorm_add: hidden[i] += rmsnorm(attn_out)[i] · post_attn_norm[i]   (decode_ops:386)
//   (11) rmsnorm:     normed[i]  = rmsnorm(hidden)[i]    · pre_ffn_norm[i]      (rmsnorm_msl:16)
// = TWO dispatches + TWO full-hidden square-sum reductions per layer × 48 layers. llama.cpp's ggml
// graph CANNOT fuse these (they are distinct graph nodes with a data dependency); Arf's
// megakernel can, because it owns the whole frame. This is the dispatch-count / redundant-reduction
// cut that directly attacks the ~38-dispatch/token gemma frame — a STRUCTURAL beat-llama win.
//
// One threadgroup per row. Two sequential reductions, EACH byte-identical to its source kernel's
// reduction (same strided square-sum, same simd_sum, same sgp[32] combine, same
// 1/sqrt(total/hidden+eps)) so the fused result is bit-exact vs the rmsnorm_add→rmsnorm pair:
//   R1: reduce ss over attn_out → inv1 → publish hidden[i] += attn_out[i]·inv1·post_norm[i]
//   (barrier: R2 must read the UPDATED hidden all lanes wrote)
//   R2: reduce ss over the updated hidden → inv2 → normed[i] = hidden[i]·inv2·pre_ffn_norm[i]
//
// Bindings: 0 attn_out[hidden] (R1 src, read), 1 post_norm[hidden], 2 hidden[hidden] (read_write:
//   += published), 3 pre_ffn_norm[hidden], 4 normed[hidden] (write, FFN input), 5 dims.
// Dispatch: grid (tokens,1,1), threadgroup 256 (one row per TG). dims {tokens, hidden, eps, _}.

#include <metal_stdlib>
using namespace metal;

struct NormPairDims { uint tokens; uint hidden; float eps; uint _pad; };

kernel void gemma_normpair(
        device const float *attn_out [[buffer(0)]],   // R1 source
        device const float *post_norm[[buffer(1)]],
        device       float *hidden   [[buffer(2)]],    // read_write: += rmsnorm(attn_out)·post_norm
        device const float *pre_ffn  [[buffer(3)]],
        device       float *normed   [[buffer(4)]],    // = rmsnorm(hidden)·pre_ffn (FFN input)
        constant NormPairDims &d     [[buffer(5)]],
        uint  tgpig                  [[threadgroup_position_in_grid]],
        uint  lid                    [[thread_position_in_threadgroup]],
        uint  tg_size                [[threads_per_threadgroup]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig;
    const uint base = row * d.hidden;
    const uint nsg = (tg_size + 31u) / 32u;
    threadgroup float sgp[32];

    // ===== R1: post-attn rmsnorm_add — reduce ss(attn_out), publish hidden += norm·post_norm. =====
    // (identical reduction to rmsnorm_add: decode_ops_msl.metal:398-414)
    float acc = 0.0f;
    for (uint i = lid; i < d.hidden; i += tg_size) {
        const float v = attn_out[base + i];
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (tiisg == 0) sgp[sgitg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (lid == 0) { for (uint s = 0; s < nsg; ++s) total += sgp[s]; sgp[0] = total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = sgp[0];
    const float inv1 = 1.0f / sqrt(total / float(d.hidden) + d.eps);
    for (uint i = lid; i < d.hidden; i += tg_size) {
        hidden[base + i] += attn_out[base + i] * inv1 * post_norm[i];
    }
    // R2 reads the hidden all lanes just wrote — fence the in-place publish.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===== R2: pre-FFN rmsnorm — reduce ss(updated hidden), write normed = norm·pre_ffn. =====
    // (identical reduction to rmsnorm: rmsnorm_msl.metal:30-51)
    acc = 0.0f;
    for (uint i = lid; i < d.hidden; i += tg_size) {
        const float v = hidden[base + i];
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (tiisg == 0) sgp[sgitg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = 0.0f;
    if (lid == 0) { for (uint s = 0; s < nsg; ++s) total += sgp[s]; sgp[0] = total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = sgp[0];
    const float inv2 = 1.0f / sqrt(total / float(d.hidden) + d.eps);
    for (uint i = lid; i < d.hidden; i += tg_size) {
        normed[base + i] = hidden[base + i] * inv2 * pre_ffn[i];
    }
}
