// Add a per-column bias to every row of a [rows, cols] matrix, in place:
//   x[r*cols + c] += bias[c]
// Used after the SigLIP Q/K/V/O and MLP linears (the text path has no biases).

struct Dims { rows: u32, cols: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read>       bias: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;

// 2D-safe flat index (see add.wgsl): rows*cols can exceed the 65535-workgroup-per-dim
// limit (vision up-proj = 4096*4304 = 17.6M), so the caller spreads workgroups over Y.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 256u) + gid.x;
    let n = d.rows * d.cols;
    if (i < n) {
        x[i] = x[i] + bias[i % d.cols];
    }
}
