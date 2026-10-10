// Fused MoE down-projection (bf16): computes the WEIGHT-SUMMED down output for ALL
// top_k routed experts in ONE dispatch, removing the per-expert accumulate
// serialization on the shared mlp_down. One workgroup owns one output row h_j; it
// loops over slots (ASCENDING, to match the existing sequential accumulate order =>
// bit-exact), reading expert = ids[slot] and that slot's silu segment
// silu_all[slot*mi .. ], reducing over mi and accumulating wts[slot]*dot.
//
//   mlp_down[h_j] = Σ_{slot=0..top_k-1} wts[slot] · ( Σ_c silu_all[slot*mi+c] · Wdown_{ids[slot]}[h_j,c] )
//
// Dims: m, k(=mi), n(=h), slot(unused), mode(1 write / 2 accumulate), top_k(=_p0).
// `a` is silu_all as vec4<f32>; slot's segment starts at sbase = slot*(mi/4) vec4s.

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, top_k: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;   // silu_all [top_k*mi], 4/vec
@group(0) @binding(1) var<storage, read>       b: array<vec4<u32>>;   // packed experts: 8 bf16/vec4
@group(0) @binding(2) var<storage, read_write> c: array<f32>;         // mlp_down [n=h]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       ids: array<u32>;
@group(0) @binding(5) var<storage, read>       wts: array<f32>;

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

    var h_j = wid.x;                      // one workgroup per output row, grid-stride
    while (h_j < d.n) {
        // Each lane accumulates wts[slot]·(its strided dot share) over ALL slots
        // into a PRIVATE sum (no per-slot barriers), then ONE final tree reduction.
        var dotacc = 0.0;
        var slot = 0u;
        while (slot < d.top_k) {
            let expert = ids[slot];
            let rw = wts[slot];
            let row_base = (expert * d.n + h_j) * vecs;
            let sbase = slot * slot_vecs;
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
                c[h_j] = c[h_j] + partial[0];
            } else {
                c[h_j] = partial[0];
            }
        }
        workgroupBarrier();
        h_j = h_j + ngroups.x;
    }
}
