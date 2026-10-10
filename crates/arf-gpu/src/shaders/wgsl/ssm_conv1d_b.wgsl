// WGSL causal depthwise conv1d — B-PARALLEL (concurrency) twin of ssm_conv1d.wgsl. Decodes B seqs
// in lockstep: each seq carries its OWN conv ring; the conv_kernel (static weight) is shared.
// b-outer layout: conv_in[b*conv_dim + i1], conv_state[b*conv_dim*(d_conv-1) + i1*(d_conv-1) + i0].
// b=0 slice is byte-identical to the single-stream kernel → "B=1 ≡ M11".
//
// One invocation per (b, conv channel): grid = B*conv_dim. Read OLD ring BEFORE writing (the
// critical causal ordering). Same scalar math / register window / SiLU as ssm_conv1d.wgsl.
//
// Bindings: 0 conv_in[B*conv_dim] (READ), 1 conv_kernel[d_conv*conv_dim] (weight, SHARED, READ),
//           2 conv_state[B*conv_dim*(d_conv-1)] (READ+WRITE ring), 3 conv_out[B*conv_dim] (WRITE),
//           4 dims {conv_dim, d_conv, batch, _}.

struct ConvDimsB { conv_dim: u32, d_conv: u32, batch: u32, _p1: u32, };

@group(0) @binding(0) var<storage, read>        conv_in: array<f32>;
@group(0) @binding(1) var<storage, read>        conv_kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write>  conv_state: array<f32>;
@group(0) @binding(3) var<storage, read_write>  conv_out: array<f32>;
@group(0) @binding(4) var<uniform>              d: ConvDimsB;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = d.batch * d.conv_dim;
    let lane = gid.x;
    if (lane >= total) { return; }
    let b = lane / d.conv_dim;           // which sequence
    let i1 = lane % d.conv_dim;          // which conv channel

    let dc = d.d_conv;                   // taps (4)
    let cs = dc - 1u;                    // cached cols per channel (3)
    // b-outer ring/input strides. conv_kernel is SHARED across b (kbase has no b term).
    let sbase = b * d.conv_dim * cs + i1 * cs;   // this (b,channel)'s ring base
    let kbase = i1 * dc;                          // shared kernel base (tap-contiguous)
    let inbase = b * d.conv_dim;                  // this seq's conv_in base

    // 1) Assemble causal window from the OLD ring + the new input last (read before write).
    var window: array<f32, 8>;           // SSM_CONV_MAX_TAPS
    for (var i0: u32 = 0u; i0 < cs; i0 = i0 + 1u) {
        window[i0] = conv_state[sbase + i0];
    }
    let x_new = conv_in[inbase + i1];
    window[cs] = x_new;

    // 2) d_conv-tap dot.
    var sumf: f32 = 0.0;
    for (var i0: u32 = 0u; i0 < dc; i0 = i0 + 1u) {
        sumf = sumf + window[i0] * conv_kernel[kbase + i0];
    }

    // 3) SiLU epilogue (exact match to silu.wgsl).
    let z = clamp(sumf, -30.0, 30.0);
    conv_out[inbase + i1] = sumf / (1.0 + exp(-z));

    // 4) Advance the ring (drop oldest, append new). Write AFTER the dot.
    for (var i0: u32 = 0u; i0 < cs; i0 = i0 + 1u) {
        conv_state[sbase + i0] = window[i0 + 1u];
    }
}
