// Element-wise "quick GELU" (the activation in CLIP's text encoder MLP):
//   x * sigmoid(1.702 * x)
// In place over `x[0..n]`. Mirrors gelu_tanh.wgsl's 2D-safe index + Metal overflow guard.

struct Dims { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform>             d: Dims;

// 2D-safe flat index (see add.wgsl): n can exceed the 65535-workgroup-per-dim limit, so Y
// carries the overflow.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 256u) + gid.x;
    if (i < d.n) {
        let v = x[i];
        // sigmoid(z) = 1/(1+exp(-z)). Clamp z so Metal's exp() can't overflow→inf (sigmoid
        // saturates to 0/1 well before |z|=30); same NaN-safety discipline as gelu_tanh.wgsl.
        let z = clamp(1.702 * v, -30.0, 30.0);
        x[i] = v / (1.0 + exp(-z));
    }
}
