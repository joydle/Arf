// Batched MoE gate/up (bf16): the FUSED analog of matmul_vec_moe.wgsl that computes
// ALL top_k routed experts' gate (or up) projection in ONE dispatch, removing the
// per-expert serialization. Output is [top_k, n] (slot-major: c[slot*n + col]); the
// grid strides over the flat (slot, col) space, resolving the expert per slot from
// ids[slot]. Raw write only (the router weight is applied later, in the fused down).
// top_k is passed in Dims._p0. Inner j-reduction is byte-identical to
// matmul_vec_moe.wgsl, so each entry equals that kernel's per-slot scratch.

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, top_k: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;   // activation, 4 f32/vec
@group(0) @binding(1) var<storage, read>       b: array<vec4<u32>>;   // packed experts: 8 bf16/vec4
@group(0) @binding(2) var<storage, read_write> c: array<f32>;         // [top_k * n]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       ids: array<u32>;       // routed expert ids

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

fn bf16(bits: u32) -> f32 { return bitcast<f32>(bits << 16u); }
fn dot_word(word: u32, lo: f32, hi: f32) -> f32 {
    return lo * bf16(word & 0xffffu) + hi * bf16(word >> 16u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let vecs = d.k / 8u;                 // vec4 words per weight row (k % 8 == 0)
    let total = d.top_k * d.n;           // flat (slot, col) output count

    var idx = wid.x;                     // one workgroup per output element, grid-stride
    while (idx < total) {
        let slot = idx / d.n;
        let col = idx % d.n;
        let expert = ids[slot];          // expert selected on-GPU for this slot
        let row_base = (expert * d.n + col) * vecs;
        var acc = 0.0;
        var w = lane;
        while (w < vecs) {
            let bv = b[row_base + w];
            let a0 = a[2u * w];
            let a1 = a[2u * w + 1u];
            acc = acc + dot_word(bv.x, a0.x, a0.y)
                      + dot_word(bv.y, a0.z, a0.w)
                      + dot_word(bv.z, a1.x, a1.y)
                      + dot_word(bv.w, a1.z, a1.w);
            w = w + WG;
        }
        partial[lane] = acc;
        workgroupBarrier();

        var stride = WG / 2u;
        while (stride > 0u) {
            if (lane < stride) {
                partial[lane] = partial[lane] + partial[lane + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        if (lane == 0u) {
            c[idx] = partial[0];         // raw; router weight applied in fused down
        }
        workgroupBarrier();
        idx = idx + ngroups.x;
    }
}
