// Fused MoE down-projection (Q4_0 block-32): weight-summed down for ALL top_k
// routed experts in ONE dispatch (removes per-expert accumulate serialization).
// One workgroup per output row h_j; outer loop slots ASCENDING (bit-exact vs the
// old sequential accumulate); inner 64-lane reduce over mi sub-blocks of this
// slot's silu segment. mlp_down[h_j] = Σ_slot wts[slot]·dot(silu_all[slot], Wdown_e).
// Dims: k=mi, n=h, mode(1 write/2 accumulate), top_k=_p0. `a` = silu_all (vec4<f32>),
// slot segment at sbase = slot*(mi/4) vec4s. Q4_0 dequant byte-identical to
// matmul_vec_moe_q4.wgsl.

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, top_k: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // silu_all [top_k*mi]
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // packed experts' codes
@group(0) @binding(2) var<storage, read_write> c: array<f32>;           // mlp_down [n=h]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;
@group(0) @binding(5) var<storage, read>       ids: array<u32>;
@group(0) @binding(6) var<storage, read>       wts: array<f32>;

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

fn bf16_to_f32(bits: u32) -> f32 { return bitcast<f32>((bits & 0xFFFFu) << 16u); }
fn unpack_lo(word: u32) -> vec4<f32> {
    let n = vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(0xFu);
    let sign = select(vec4<f32>(0.0), vec4<f32>(16.0), n >= vec4<u32>(8u));
    return vec4<f32>(n) - sign;
}
fn unpack_hi(word: u32) -> vec4<f32> {
    let n = vec4<u32>(word >> 16u, word >> 20u, word >> 24u, word >> 28u) & vec4<u32>(0xFu);
    let sign = select(vec4<f32>(0.0), vec4<f32>(16.0), n >= vec4<u32>(8u));
    return vec4<f32>(n) - sign;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let blocks = d.k / 32u;               // blocks per weight row (k = mi)
    let slot_vecs = d.k / 4u;             // vec4s per slot's silu segment (mi/4)

    var h_j = wid.x;
    while (h_j < d.n) {
        // Each lane accumulates wts[slot]·(its strided block share) over ALL slots
        // into a PRIVATE sum (no per-slot barriers), then ONE final tree reduction.
        var dotacc = 0.0;
        var slot = 0u;
        while (slot < d.top_k) {
            let expert = ids[slot];
            let rw = wts[slot];
            let code_base = (expert * d.n + h_j) * blocks;
            let sbase = slot * slot_vecs;
            var b = lane;
            while (b < blocks) {
                let qv = codes[code_base + b];
                let s = bf16_to_f32(scales[code_base + b]);
                let abase = sbase + 8u * b;
                let blk = dot(a[abase],      unpack_lo(qv.x)) + dot(a[abase + 1u], unpack_hi(qv.x))
                        + dot(a[abase + 2u], unpack_lo(qv.y)) + dot(a[abase + 3u], unpack_hi(qv.y))
                        + dot(a[abase + 4u], unpack_lo(qv.z)) + dot(a[abase + 5u], unpack_hi(qv.z))
                        + dot(a[abase + 6u], unpack_lo(qv.w)) + dot(a[abase + 7u], unpack_hi(qv.w));
                dotacc = dotacc + rw * (s * blk);
                b = b + WG;
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
