// WGSL fused g_decay for the gated-delta-net (M11 single-queue port). BIT-EXACT twin of
// gdn_g_decay in gdn_prologue_msl.metal: g_out[h] = softplus(alpha[h]+ssm_dt[h]) * ssm_a[h] over
// n = num_v_heads, where softplus = ggml op_softplus (x>20 ? x : log(1+exp(x))).
//
// Bindings: 0 g_out[n] (WRITE), 1 alpha[n] (READ), 2 ssm_dt[n] (READ), 3 ssm_a[n] (READ),
//           4 dims {n,_,_,_}.

struct Elem1 { n: u32, _p0: u32, _p1: u32, _p2: u32, };

@group(0) @binding(0) var<storage, read_write> g_out: array<f32>;
@group(0) @binding(1) var<storage, read>       alpha: array<f32>;
@group(0) @binding(2) var<storage, read>       ssm_dt: array<f32>;
@group(0) @binding(3) var<storage, read>       ssm_a: array<f32>;
@group(0) @binding(4) var<uniform>             d: Elem1;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= d.n) { return; }
    let v = alpha[i] + ssm_dt[i];
    var sp: f32;
    if (v > 20.0) { sp = v; } else { sp = log(1.0 + exp(v)); }   // ggml op_softplus
    g_out[i] = sp * ssm_a[i];
}
