// BATCHED (B rows) fused MoE down-projection (bf16): the B-row generalization of
// matmul_vec_moe_down_reduce.wgsl. For each (row, h_j) output, weight-sums the down
// projection over THIS ROW's top_k routed experts in ONE dispatch:
//   mlp_down[row*h + h_j]
//     = Σ_slot wts[row*top_k+slot]·( Σ_c silu[row*top_k*mi + slot*mi + c]
//                                     · Wdown_{ids[row*top_k+slot]}[h_j,c] )
//
// One workgroup per (row, h_j); flat grid strides over B*h (oi → brow=oi/h,
// h_j=oi%h). Slots ASCENDING (bit-exact vs the sequential accumulate). Inner
// j-reduction byte-identical to matmul_vec_moe_down_reduce.wgsl; only the per-row
// offsets are added (ids/wts at row*top_k+slot, silu segment at row*top_k*mi +
// slot*mi, output at row*h + h_j). With B=1 (brow=0) it reduces exactly to the
// single-token kernel → bit-identical. Dims: k=mi, n=h, m=B, mode(1 write/2 accum),
// top_k=Dims.top_k. `a` is silu_all as vec4<f32>.

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, top_k: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;   // silu_all [B*top_k*mi], 4/vec
@group(0) @binding(1) var<storage, read>       b: array<vec4<u32>>;   // packed experts: 8 bf16/vec4
@group(0) @binding(2) var<storage, read_write> c: array<f32>;         // mlp_down [B*h]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       ids: array<u32>;       // [B*top_k]
@group(0) @binding(5) var<storage, read>       wts: array<f32>;       // [B*top_k]

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
    let vecs = d.k / 8u;                  // vec4 words per weight row (k = mi)
    let slot_vecs = d.k / 4u;             // vec4s per slot's silu segment (mi/4)
    let row_vecs = d.top_k * slot_vecs;   // vec4s per batch row's silu (top_k*mi/4)
    let total = d.m * d.n;                // B * h output elements

    var oi = wid.x;                       // one workgroup per output element, grid-stride
    while (oi < total) {
        let brow = oi / d.n;
        let h_j = oi % d.n;
        let ibase = brow * d.top_k;       // this row's ids/wts segment
        let rbase = brow * row_vecs;      // this row's silu segment (vec4 units)
        var dotacc = 0.0;
        var slot = 0u;
        while (slot < d.top_k) {
            let expert = ids[ibase + slot];
            let rw = wts[ibase + slot];
            let row_base = (expert * d.n + h_j) * vecs;
            let sbase = rbase + slot * slot_vecs;
            var w = lane;
            while (w < vecs) {
                let bv = b[row_base + w];
                let a0 = a[sbase + 2u * w];
                let a1 = a[sbase + 2u * w + 1u];
                let part = dot_word(bv.x, a0.x, a0.y)
                         + dot_word(bv.y, a0.z, a0.w)
                         + dot_word(bv.z, a1.x, a1.y)
                         + dot_word(bv.w, a1.z, a1.w);
                dotacc = dotacc + rw * part;
                w = w + WG;
            }
            slot = slot + 1u;
        }
        partial[lane] = dotacc;
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
            if (d.mode == 2u) {
                c[oi] = c[oi] + partial[0];
            } else {
                c[oi] = partial[0];
            }
        }
        workgroupBarrier();
        oi = oi + ngroups.x;
    }
}
