// GeGLU: out[i] = gelu_tanh(gate[i]) * up[i], the Gemma FFN gate.
//   gelu_tanh(x) = 0.5·x·(1 + tanh( √(2/π)·(x + 0.044715·x³) ))
// This is HF `gelu_pytorch_tanh` / ggml `ffn_geglu` — Gemma uses it, NOT SiLU.
// Pure elementwise; one lane per element. Bindings match swiglu.wgsl exactly so
// it's a drop-in dispatch.

struct Dims { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= d.n) {
        return;
    }
    let x = gate[i];
    // tanh saturates to ±1 by |arg|≈10; clamp before the call. The Metal/naga
    // lowering of tanh evaluates (e^a−e^−a)/(e^a+e^−a), so a large argument gives
    // inf/inf = NaN. QAT-quantized Gemma drives the gate to ~±34, making the cubic
    // term push `inner` past 1000 — without this clamp the whole forward NaNs.
    let inner = clamp(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x), -15.0, 15.0);
    let gelu = 0.5 * x * (1.0 + tanh(inner));
    out[i] = gelu * up[i];
}
