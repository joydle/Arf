// MoE router top-k selection (single token, B=1) for the concurrent megakernel — a faithful MSL
// port of moe_route_b.wgsl's B=1 path. ONE threadgroup: softmax max + sum-exp over num_experts
// logits, then top_k passes of argmax over not-yet-taken experts (ties → LOWEST index), optional
// renorm (Qwen3 norm_topk). Writes ids[top_k] + wts[top_k]. Byte-identical selection to the WGSL
// route, so downstream indirect GEMVs read the same routed experts/weights.
//
// Dispatch: grid (1,1,1), threadgroup 128 lanes. dims {num_experts, top_k, norm_topk, _}.

#include <metal_stdlib>
using namespace metal;

struct RouteDims { uint num_experts; uint top_k; uint norm_topk; uint _pad; };

constant uint WG = 128u;
constant uint MAX_EXPERTS = 256u;
constant float NEG_INF = -3.4e38f;

kernel void moe_route(
        device const float *logits [[buffer(0)]],  // [num_experts]
        device       uint  *ids    [[buffer(1)]],  // [top_k]
        device       float *wts    [[buffer(2)]],  // [top_k]
        constant RouteDims &d      [[buffer(3)]],
        uint lane [[thread_position_in_threadgroup]]) {
    const uint n = d.num_experts;
    const uint k = min(d.top_k, n);

    threadgroup float red[128];
    threadgroup float sh_max;
    threadgroup float sh_sum;
    threadgroup bool taken[256];

    // MAX over logits (tree reduce).
    float m = NEG_INF;
    for (uint i = lane; i < n; i += WG) m = max(m, logits[i]);
    red[lane] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = WG/2u; stride > 0u; stride /= 2u) {
        if (lane < stride) red[lane] = max(red[lane], red[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) sh_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maxv = sh_max;

    // SUM of exp(logit - max).
    float s = 0.0f;
    for (uint i = lane; i < n; i += WG) s += exp(logits[i] - maxv);
    red[lane] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = WG/2u; stride > 0u; stride /= 2u) {
        if (lane < stride) red[lane] += red[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) sh_sum = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sum = sh_sum;

    // clear taken mask.
    for (uint i = lane; i < n; i += WG) taken[i] = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // k passes of argmax over not-yet-taken (ties → lowest index). Reuse `red` for val,
    // a second tg array for idx.
    threadgroup uint redi[128];
    for (uint slot = 0u; slot < k; ++slot) {
        float bv = NEG_INF; uint bi = 0xFFFFFFFFu;
        for (uint i = lane; i < n; i += WG) {
            if (!taken[i]) {
                float v = logits[i];
                if (v > bv) { bv = v; bi = i; }
            }
        }
        red[lane] = bv; redi[lane] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = WG/2u; stride > 0u; stride /= 2u) {
            if (lane < stride) {
                float ov = red[lane + stride]; uint oi = redi[lane + stride];
                float cv = red[lane]; uint ci = redi[lane];
                if (ov > cv || (ov == cv && oi < ci)) { red[lane] = ov; redi[lane] = oi; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0u) {
            uint best = redi[0];
            taken[best] = true;
            ids[slot] = best;
            wts[slot] = exp(logits[best] - maxv) / sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // renorm (lane 0, k tiny).
    if (lane == 0u && d.norm_topk != 0u) {
        float wsum = 0.0f;
        for (uint slot = 0u; slot < k; ++slot) wsum += wts[slot];
        if (wsum > 0.0f) for (uint slot = 0u; slot < k; ++slot) wts[slot] /= wsum;
    }
}
