// WGSL gdn_tile — B-PARALLEL twin of gdn_tile.wgsl. Per seq b: split conv_out[b] into
// [q_conv|k_conv|v_conv] and TILE q,k across v-head groups (ggml_repeat_4d) into [s*nvh] arrays.
// b-outer: conv_out[b*conv_dim + ...], q_rep/k_rep/v_out[b*value_dim + t]. b=0 slice ≡ single-stream.
//
// One invocation per (b, output index t in 0..value_dim): grid = B*value_dim.
//   h = t/s ; i = t%s ; kh = h % nkh ; src = kh*s + i
//   q_rep[b*vd + t] = conv_out[b*cd + src]
//   k_rep[b*vd + t] = conv_out[b*cd + key_dim + src]
//   v_out[b*vd + t] = conv_out[b*cd + 2*key_dim + t]
//
// Bindings: 0 conv_out[B*conv_dim] (READ), 1 q_rep[B*s*nvh] (WRITE), 2 k_rep[B*s*nvh] (WRITE),
//           3 v[B*value_dim] (WRITE), 4 dims {s, nkh, nvh, key_dim}, 5 dims2 {conv_dim, batch, _, _}.

struct TileDims { s: u32, nkh: u32, nvh: u32, key_dim: u32, };
struct TileDims2 { conv_dim: u32, batch: u32, _p0: u32, _p1: u32, };

@group(0) @binding(0) var<storage, read>       conv_out: array<f32>;
@group(0) @binding(1) var<storage, read_write> q_rep: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_rep: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_out: array<f32>;
@group(0) @binding(4) var<uniform>             d: TileDims;
@group(0) @binding(5) var<uniform>             d2: TileDims2;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vd = d.s * d.nvh;               // value_dim (== per-seq q_rep/k_rep/v length)
    let total = d2.batch * vd;
    let lane = gid.x;
    if (lane >= total) { return; }
    let b = lane / vd;                  // which sequence
    let t = lane % vd;                  // output index within the seq

    let h = t / d.s;
    let i = t % d.s;
    let kh = h % d.nkh;
    let src = kh * d.s + i;

    let cbase = b * d2.conv_dim;        // this seq's conv_out base
    let obase = b * vd;                 // this seq's output base
    q_rep[obase + t] = conv_out[cbase + src];
    k_rep[obase + t] = conv_out[cbase + d.key_dim + src];
    v_out[obase + t] = conv_out[cbase + 2u * d.key_dim + t];
}
