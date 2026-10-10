// SwiGLU: out[i] = silu(gate[i]) * up[i], with silu(x) = x / (1 + exp(-x)).
// Pure elementwise; one lane per element. Matches Mlp::forward.

struct Dims { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= d.n) {
        return;
    }
    let g = gate[i];
    out[i] = (g / (1.0 + exp(-g))) * up[i];
}
