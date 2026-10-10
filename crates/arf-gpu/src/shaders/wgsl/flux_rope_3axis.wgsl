// FLUX 3-axis RoPE, INTERLEAVED-pair convention (NOT the GPT-NeoX (i,i+half) rotate-half in
// rope.wgsl). Pairs are consecutive (x[2k], x[2k+1]); the host builds per-position cos/sin
// tables of length head_dim (already repeat-interleaved so cos[2k]==cos[2k+1]) with the 3-axis
// split [16,56,56] and axis-0's 16 dims = identity (cos1/sin0) baked in.
//
//   q'[2k]   = q[2k]·cos[2k]   - q[2k+1]·sin[2k]
//   q'[2k+1] = q[2k+1]·cos[2k] + q[2k]·sin[2k]
//
// Fused over q and k like qk_norm/rope_qk: workgroup ids [0,q_rows) rotate q, [q_rows, q_rows+
// k_rows) rotate k. Each row is one head's head_dim slice at (t*heads + head)*head_dim, and uses
// the cos/sin row for that token. cos/sin are [seq, head_dim].

struct Dims { q_rows: u32, k_rows: u32, heads: u32, head_dim: u32, seq: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read>       cos: array<f32>;   // [seq, head_dim]
@group(0) @binding(3) var<storage, read>       sin: array<f32>;
@group(0) @binding(4) var<uniform>             d: Dims;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    // 2D-safe row (q_rows+k_rows exceeds 65535 workgroups/dim at 1024px). Grid is [x,y,1];
    // reconstruct row-major with ng.x the x-extent. See dit.rs wg_rows.
    let row = wid.y * ng.x + wid.x;  // global (token,head) row across q then k
    let lane = lid.x;
    let total = d.q_rows + d.k_rows;
    if (row >= total) { return; }
    let is_k = row >= d.q_rows;
    let local = select(row, row - d.q_rows, is_k);   // row within q (or k)
    let hd = d.head_dim;
    let token = local / d.heads;     // which token this row belongs to
    let base = local * hd;
    let cbase = token * hd;          // cos/sin row for this token

    // each lane handles consecutive pairs (2k, 2k+1), strided.
    var p = lane;
    let pairs = hd / 2u;
    while (p < pairs) {
        let i0 = 2u * p;
        let i1 = i0 + 1u;
        let c = cos[cbase + i0];     // == cos[cbase+i1] (repeat-interleaved)
        let s = sin[cbase + i0];
        if (is_k) {
            let a = k[base + i0]; let b = k[base + i1];
            k[base + i0] = a * c - b * s;
            k[base + i1] = b * c + a * s;
        } else {
            let a = q[base + i0]; let b = q[base + i1];
            q[base + i0] = a * c - b * s;
            q[base + i1] = b * c + a * s;
        }
        p = p + 64u;
    }
}
