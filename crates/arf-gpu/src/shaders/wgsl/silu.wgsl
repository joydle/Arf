// Element-wise SiLU / swish, in place: x * sigmoid(x). Used by FLUX's adaLN modulation MLPs
// (SiLU(vec) before the Linear) and the single-stream activation. Mirrors gelu_tanh.wgsl's
// 2D-safe index + Metal exp overflow clamp.

struct Dims { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 256u) + gid.x;
    if (i < d.n) {
        let v = x[i];
        // clamp so Metal's exp() can't overflow→inf (sigmoid saturates well before |v|=30).
        let z = clamp(v, -30.0, 30.0);
        x[i] = v / (1.0 + exp(-z));
    }
}
