// SwiGLU for the MoE megakernel: out[i] = silu(gate[i]) * up[i], silu(x)=x/(1+exp(-x)).
// qwen MoE is SILU-gated (NOT gemma's gelu-gated geglu). Elementwise over n = top_k*inter
// elements (the routed-slot gate/up outputs). Bit-exact to swiglu.wgsl.
//
// Dispatch: grid (ceil(n/256),1,1), threadgroup 256. dims {n, _, _, _}.

#include <metal_stdlib>
using namespace metal;

struct SwDims { uint n; uint _p0; uint _p1; uint _p2; };

kernel void moe_swiglu(
        device const float *gate [[buffer(0)]],  // gate_all [top_k*inter]
        device const float *up   [[buffer(1)]],  // up_all   [top_k*inter]
        device       float *out  [[buffer(2)]],  // silu_all [top_k*inter]
        constant SwDims    &d    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    const float g = gate[gid];
    out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}
