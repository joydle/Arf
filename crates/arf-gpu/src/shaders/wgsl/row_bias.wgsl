// Add a per-column bias `[n]` to every row of a row-major `[rows, n]` buffer, in place:
//   x[r*n + c] += bias[c]
// Used to apply a 1×1-conv (linear) bias to a pixel-major `[seq, c_out]` activation in the VAE
// mid-block attention (q/k/v/proj_out). 2D-safe flat index.

struct Dims { rows: u32, n: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read>       bias: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 256u) + gid.x;
    let total = d.rows * d.n;
    if (i < total) {
        x[i] = x[i] + bias[i % d.n];
    }
}
