// WGSL non-in-place sigmoid for beta — B-PARALLEL twin of gdn_sigmoid_out.wgsl. Pure elementwise
// over the flat [B*n] buffer (b-outer is automatic: src/out are both [B*n], index passes through).
// b=0 slice ≡ single-stream. Caller sets dims.n = B*n_per_seq (e.g. B*num_v_heads).
//
// Bindings: 0 out[B*n] (WRITE), 1 src[B*n] (READ), 2 dims {n_total,_,_,_} where n_total = B*n.

struct Elem1 { n: u32, _p0: u32, _p1: u32, _p2: u32, };

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<uniform>             d: Elem1;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= d.n) { return; }
    out[i] = 1.0 / (1.0 + exp(-src[i]));
}
