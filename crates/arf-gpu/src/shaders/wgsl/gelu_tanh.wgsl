// Element-wise GELU (tanh approximation — matches ggml `use_gelu` / the CPU vision path):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
// In place over `x[0..n]`.

struct Dims { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform>             d: Dims;

// 2D-safe flat index (see add.wgsl): n can exceed the 65535-workgroup-per-dim limit
// (vision MLP activation = 4096*4304 = 17.6M elements), so Y carries the overflow.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 256u) + gid.x;
    if (i < d.n) {
        let v = x[i];
        let k = 0.7978845608; // sqrt(2/pi)
        // Clamp the tanh argument: Metal's fast `tanh` overflows its internal exp() and
        // returns NaN for large arguments (observed at v≈10.5 → arg≈49). tanh saturates to
        // ±1 well before |arg|=10 in f32, so clamping to ±10 is exact and NaN-safe. This is
        // the same guard ggml/llama.cpp apply to their tanh-approx GELU.
        let arg = clamp(k * (v + 0.044715 * v * v * v), -10.0, 10.0);
        x[i] = 0.5 * v * (1.0 + tanh(arg));
    }
}
