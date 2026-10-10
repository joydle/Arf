// Elementwise multiply in place: a[i] *= b[i]. (T5 gated-GELU: gate = gelu(gate) * up.)

struct Dims { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;

// 2D-safe flat index (see add.wgsl): Y carries the overflow past the 65535-per-dim limit.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 256u) + gid.x;
    if (i < d.n) {
        a[i] = a[i] * b[i];
    }
}
