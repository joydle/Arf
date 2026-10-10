// WGSL gdn_tile for the gated-delta-net (M11 single-queue port). BIT-EXACT twin of gdn_tile in
// gdn_prologue_msl.metal: split conv_out [q_conv|k_conv|v_conv] and TILE q,k across v-head groups
// (ggml_repeat_4d) into the [s*nvh] arrays the recurrence consumes. v passes through unchanged.
//
// One invocation per output index t in 0..s*nvh (== value_dim):
//   h = t/s ; i = t%s ; kh = h % nkh ; src = kh*s + i
//   q_rep[t] = conv_out[src] ; k_rep[t] = conv_out[key_dim+src] ; v[t] = conv_out[2*key_dim+t]
//
// Bindings: 0 conv_out[conv_dim] (READ), 1 q_rep[s*nvh] (WRITE), 2 k_rep[s*nvh] (WRITE),
//           3 v[value_dim] (WRITE), 4 dims {s, nkh, nvh, key_dim}.

struct TileDims { s: u32, nkh: u32, nvh: u32, key_dim: u32, };

@group(0) @binding(0) var<storage, read>       conv_out: array<f32>;
@group(0) @binding(1) var<storage, read_write> q_rep: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_rep: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_out: array<f32>;
@group(0) @binding(4) var<uniform>             d: TileDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = d.s * d.nvh;               // == value_dim
    let t = gid.x;
    if (t >= n) { return; }
    let h = t / d.s;
    let i = t % d.s;
    let kh = h % d.nkh;
    let src = kh * d.s + i;
    q_rep[t] = conv_out[src];
    k_rep[t] = conv_out[d.key_dim + src];
    v_out[t] = conv_out[2u * d.key_dim + t];
}
