// Final-logit soft-capping in place: logit[i] = cap * tanh(logit[i] / cap).
//
// Gemma 4 sets `final_logit_softcapping = 30.0` and applies this element-wise to
// the lm_head output (every vocab logit, across every emitted row) BEFORE
// sampling/argmax. Squashes extreme logits into (-cap, +cap) without changing
// their order near the origin. Skipped entirely when `final_logit_softcap` is
// `None` (Gemma 3 / Llama / Qwen) — the dispatch is simply not recorded.
//
// One element per invocation, `ceil(n / 256)` workgroups. `n` = rows · vocab.

struct Dims { n: u32, cap: f32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < d.n) {
        logits[i] = d.cap * tanh(logits[i] / d.cap);
    }
}
