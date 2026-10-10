// RoPE: apply rotary position embeddings to x[tokens, heads, head_dim]
// given per-token positions and resident cos/sin tables.
//
// Applied in place: each invocation rotates one (a, b) pair, so disjoint
// invocations touch disjoint indices and a single read_write buffer is safe.
// For each token t at position p, head h, and frequency pair i in [0, half):
//   a = x[base + i],  b = x[base + i + half]
//   c = cos[p * half + i],  s = sin[p * half + i]
//   x[base + i] = a*c - b*s ;  x[base + i + half] = b*c + a*s

struct Dims { tokens: u32, heads: u32, head_dim: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read>       cos: array<f32>;
@group(0) @binding(2) var<storage, read>       sin: array<f32>;
@group(0) @binding(3) var<storage, read>       positions: array<u32>;
@group(0) @binding(4) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = d.tokens * d.heads * d.head_dim;
    if (idx >= total) { return; }

    let hd = d.head_dim;
    let half = hd / 2u;

    // Decompose linear index: idx = (t * heads + h) * head_dim + i
    let t = idx / (d.heads * hd);
    let rem = idx % (d.heads * hd);
    let h = rem / hd;
    let i = rem % hd;

    // Only process first-half elements (i < half); second-half is handled
    // by the paired first-half thread (idx + half).
    if (i >= half) { return; }

    let pos = positions[t];
    let base = (t * d.heads + h) * hd;

    let c = cos[pos * half + i];
    let s = sin[pos * half + i];

    let a = x[base + i];
    let b = x[base + i + half];

    x[base + i] = a * c - b * s;
    x[base + i + half] = b * c + a * s;
}
