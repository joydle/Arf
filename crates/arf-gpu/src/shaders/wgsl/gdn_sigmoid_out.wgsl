// WGSL non-in-place sigmoid for beta (M11 single-queue port). BIT-EXACT twin of gdn_sigmoid_out
// in gdn_prologue_msl.metal: out[i] = 1/(1+exp(-src[i])) (ggml op_sigmoid). Reads a separate src
// so the resident beta_raw projection is not mutated.
//
// Bindings: 0 out[n] (WRITE), 1 src[n] (READ), 2 dims {n,_,_,_}.

struct Elem1 { n: u32, _p0: u32, _p1: u32, _p2: u32, };

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<uniform>             d: Elem1;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= d.n) { return; }
    out[i] = 1.0 / (1.0 + exp(-src[i]));        // ggml op_sigmoid
}
