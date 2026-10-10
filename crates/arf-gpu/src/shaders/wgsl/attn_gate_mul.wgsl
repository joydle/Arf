// Muse Glimmer's ATTENTION OUTPUT GATE: `attn[i] *= sigmoid(gate[i])`, applied after SDPA and
// BEFORE o_proj (llama.cpp src/models/muse-glimmer.cpp:137-139 — `gate = ggml_sigmoid(gate);
// cur = ggml_mul(cur, gate)`). `gate` is a separate projection of the PRE-attention hidden state
// (`attn_gate @ attn_input`), so it is q_dim-wide, exactly like `attn`.
//
// Fused rather than sigmoid-then-multiply: the gate is dead after this and a separate sigmoid
// pass would cost an extra q_dim round-trip per layer per token on the hot path.
//
// In-place on `attn` — nothing reads the un-gated attention output afterwards.
//
// Bindings: 0 attn[n] (READ_WRITE), 1 gate[n] (READ), 2 dims {n,_,_,_}.

struct Elem1 { n: u32, _p0: u32, _p1: u32, _p2: u32, };

@group(0) @binding(0) var<storage, read_write> attn: array<f32>;
@group(0) @binding(1) var<storage, read>       gate: array<f32>;
@group(0) @binding(2) var<uniform>             d: Elem1;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= d.n) { return; }
    attn[i] = attn[i] * (1.0 / (1.0 + exp(-gate[i])));
}
