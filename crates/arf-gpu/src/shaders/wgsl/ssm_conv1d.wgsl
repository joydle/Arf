// WGSL causal depthwise conv1d for the Qwen3.6-27B gated-delta-net (M11 — single-queue port).
// BIT-EXACT WGSL twin of ssm_conv1d_msl.metal (itself a port of llama.cpp ggml_ssm_conv f32,
// decode step). Lets the conv record onto the wgpu CommandPass so the whole linear layer runs on
// ONE queue (no Metal island / no cross-queue wait). Same scalar math, same register window, same
// SiLU as silu.wgsl, same ring advance — Paris must stay identical.
//
// One invocation per conv channel (conv_dim total). Read OLD ring into registers BEFORE writing.
//
// Bindings: 0 conv_in[conv_dim] (new token, READ), 1 conv_kernel[d_conv*conv_dim] (weight, READ),
//           2 conv_state[conv_dim*(d_conv-1)] (READ+WRITE ring), 3 conv_out[conv_dim] (WRITE),
//           4 dims {conv_dim, d_conv, _, _}.

struct ConvDims { conv_dim: u32, d_conv: u32, _p0: u32, _p1: u32, };

@group(0) @binding(0) var<storage, read>        conv_in: array<f32>;
@group(0) @binding(1) var<storage, read>        conv_kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write>  conv_state: array<f32>;
@group(0) @binding(3) var<storage, read_write>  conv_out: array<f32>;
@group(0) @binding(4) var<uniform>              d: ConvDims;

const SSM_CONV_MAX_TAPS: u32 = 8u;   // d_conv is tiny (4 for the beast); register window cap.

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i1 = gid.x;                      // one invocation per channel
    if (i1 >= d.conv_dim) { return; }

    let dc = d.d_conv;                   // taps (4)
    let cs = dc - 1u;                    // cached cols per channel (3)
    let sbase = i1 * cs;                 // this channel's ring base in conv_state
    let kbase = i1 * dc;                 // this channel's kernel base (tap-contiguous)

    // 1) Assemble the causal window from the OLD ring (oldest..newest) + the new input last.
    //    Read old state into registers BEFORE any write (the critical ordering).
    var window: array<f32, 8>;           // SSM_CONV_MAX_TAPS
    for (var i0: u32 = 0u; i0 < cs; i0 = i0 + 1u) {
        window[i0] = conv_state[sbase + i0];
    }
    let x_new = conv_in[i1];
    window[cs] = x_new;                  // window[d_conv-1] = the new token's value

    // 2) The d_conv-tap dot with the channel's kernel.
    var sumf: f32 = 0.0;
    for (var i0: u32 = 0u; i0 < dc; i0 = i0 + 1u) {
        sumf = sumf + window[i0] * conv_kernel[kbase + i0];
    }

    // 3) SiLU epilogue (exact match to silu.wgsl: x/(1+exp(-clamp(x,-30,30)))).
    let z = clamp(sumf, -30.0, 30.0);
    conv_out[i1] = sumf / (1.0 + exp(-z));

    // 4) Advance the ring: shift left by 1 (drop oldest), append the new input. Write AFTER the dot.
    for (var i0: u32 = 0u; i0 < cs; i0 = i0 + 1u) {
        conv_state[sbase + i0] = window[i0 + 1u];
    }
}
