// In-place elementwise scalar multiply: x[i] *= scale. Gemma 4 applies a learned
// per-layer scalar to the WHOLE layer output (`hidden *= layer_scalar`) at the very
// end of each decoder layer (after both sublayers + residuals + post-norms — see
// HF Gemma4TextDecoderLayer.forward). `scale_bits` is the f32 scale as u32 bits.
struct Dims { n: u32, scale_bits: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < d.n) {
        x[i] = x[i] * bitcast<f32>(d.scale_bits);
    }
}
