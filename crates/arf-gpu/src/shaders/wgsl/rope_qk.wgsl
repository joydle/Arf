// Fused RoPE over q and k in one dispatch (replaces two rope.wgsl dispatches).
// q has q_heads, k has k_heads; both share head_dim, cos/sin, and per-token
// positions. The global index space is [0, q_total) -> q, then
// [q_total, q_total+k_total) -> k. Each invocation writes exactly one buffer, and
// within a buffer the (i >= half) guard keeps invocations on disjoint pairs — so
// q and k being distinct read_write buffers is safe (no dispatch reads one and
// writes the other). Bit-exact vs rope.wgsl run twice: identical (t,h,i)
// decomposition, identical cos/sin index, identical a*c-b*s / b*c+a*s pair math.

// `rotary_dim` is the count of LEADING head dims RoPE rotates: pairs (i, i+half)
// with i < rotary_dim/2 are rotated, the rest pass through unchanged. For full
// rotation (every model except Gemma 4 global layers) rotary_dim == head_dim, so
// rot_half == half and the guard never fires. Gemma 4 global = 128 of 512.
struct Dims { tokens: u32, q_heads: u32, k_heads: u32, head_dim: u32, rotary_dim: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read>       cos: array<f32>;
@group(0) @binding(3) var<storage, read>       sin: array<f32>;
@group(0) @binding(4) var<storage, read>       positions: array<u32>;
@group(0) @binding(5) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hd = d.head_dim;
    // Only the first rotary_dim/2 pairs of each head are rotated (partial RoPE on
    // Gemma 4 global layers); the rest pass through. rot_half == hd/2 for full RoPE.
    let rot_half = d.rotary_dim / 2u;
    let q_total = d.tokens * d.q_heads * hd;
    let k_total = d.tokens * d.k_heads * hd;
    let idx = gid.x;
    if (idx >= q_total + k_total) {
        return;
    }

    // Which buffer/region, and the head count for the local decomposition.
    let is_q = idx < q_total;
    var local = idx;
    var heads = d.q_heads;
    if (!is_q) {
        local = idx - q_total;
        heads = d.k_heads;
    }

    // local = (t * heads + h) * hd + i  — exactly rope.wgsl's decomposition.
    let t = local / (heads * hd);
    let rem = local % (heads * hd);
    let h = rem / hd;
    let i = rem % hd;
    // Rotate only the first rot_half pairs. Each pair is (i, i + rot_half) — HF's
    // split-half convention applied to the ROTARY sub-block: for full RoPE
    // rot_half == half == hd/2 so this is the original (i, i+half) pairing; for
    // partial RoPE (Gemma 4 global, rot_half = rotary_dim/2 = 64) the partner is
    // i + 64 and the tail dims [rotary_dim, head_dim) are left untouched.
    if (i >= rot_half) {
        return;
    }

    let pos = positions[t];
    let base = (t * heads + h) * hd;
    // Table row stride is rot_half (rotary_dim/2 frequencies); for full RoPE this
    // equals hd/2, matching the original layout. The global/partial table is built
    // with this stride (see Rope::with_theta_rotary / the loader).
    let c = cos[pos * rot_half + i];
    let s = sin[pos * rot_half + i];

    if (is_q) {
        let a = q[base + i];
        let b = q[base + i + rot_half];
        q[base + i] = a * c - b * s;
        q[base + i + rot_half] = b * c + a * s;
    } else {
        let a = k[base + i];
        let b = k[base + i + rot_half];
        k[base + i] = a * c - b * s;
        k[base + i + rot_half] = b * c + a * s;
    }
}
