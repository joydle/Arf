// Fused MoE down-projection (Q4_K-lite, super-block-256, per-32 asymmetric
// scale+min): weight-summed down for ALL top_k routed experts in ONE dispatch
// (removes per-expert accumulate serialization). One workgroup per output row h_j;
// outer loop slots ASCENDING (bit-exact vs the old sequential accumulate); inner
// 32-lane reduce over mi sub-blocks of this slot's silu segment.
//
// WG=32 (not 64): the inner reduction strides over subs = k/32 = mi/32 sub-blocks.
// For Qwen3-30B mi=768 → subs=24, so a 64-lane workgroup left 40 of 64 lanes
// (62.5%) permanently idle here — and this is the single most expensive MoE
// dispatch (expert weight streamed 8× over the slot loop). WG=32 leaves only
// lanes 24..31 idle (8, not 40), halves the tree-reduction depth, and is BIT-
// IDENTICAL: each lane `i` still owns exactly sub-block `i` (the `b += WG` stride
// never produces a second sub-block at either 32 or 64 when subs ≤ 32), and the
// final tree is the same ascending left-fold over the same per-lane partials
// (the now-absent lanes 32..63 contributed 0). Parity tests confirm.
//   mlp_down[h_j] = Σ_slot wts[slot]·( Σ_c silu_all[slot*mi+c]·Wdown_{ids[slot]}[h_j,c] )
// Dims: k=mi, n=h, mode(1 write/2 accumulate), top_k=_p0. `a` = silu_all (vec4<f32>),
// slot segment at sbase = slot*(mi/4) vec4s. Q4_K dequant byte-identical to
// matmul_vec_moe_q4k.wgsl (w = code·scale + min).

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, top_k: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // silu_all [top_k*mi]
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // packed experts' codes
@group(0) @binding(2) var<storage, read_write> c: array<f32>;           // mlp_down [n=h]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // bf16/word per sub-block
@group(0) @binding(5) var<storage, read>       ids: array<u32>;
@group(0) @binding(6) var<storage, read>       wts: array<f32>;
@group(0) @binding(7) var<storage, read>       mins: array<u32>;        // bf16/word per sub-block

const WG: u32 = 32u;
var<workgroup> partial: array<f32, 32>;

fn bf16_to_f32(bits: u32) -> f32 { return bitcast<f32>((bits & 0xFFFFu) << 16u); }
fn unpack_lo(word: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(0xFu));
}
fn unpack_hi(word: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(word >> 16u, word >> 20u, word >> 24u, word >> 28u) & vec4<u32>(0xFu));
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let subs = d.k / 32u;                 // sub-blocks per weight row (k = mi)
    let slot_vecs = d.k / 4u;             // vec4s per slot's silu segment (mi/4)

    var h_j = wid.x;
    while (h_j < d.n) {
        // Each lane accumulates wts[slot]·(its strided share of each slot's
        // sub-block dots) over ALL slots into a PRIVATE running sum — no per-slot
        // barriers. ONE workgroup tree-reduction at the end (8 barrier-reductions
        // collapse to 1). Slot order ascending; wts applied per term so the lane
        // sum equals Σ_slot wts[slot]·Σ_b(...) up to the final lane-sum reassoc,
        // within the 1e-3 parity bar.
        var dotacc = 0.0;
        var slot = 0u;
        while (slot < d.top_k) {
            let expert = ids[slot];
            let rw = wts[slot];
            let base = (expert * d.n + h_j) * subs;
            let sbase = slot * slot_vecs;
            var b = lane;
            while (b < subs) {
                // Fix #2: scale/min hoisted off the critical path; code·a and Σa
                // accumulated as vec4 partials (4-wide adds pipeline) then ONE dot
                // each — shorter dependency chain so dequant ALU overlaps the load.
                // Same per-sub-block code·scale + min·Σa; intra-sub-block summation
                // order changes only, within the 1e-3 parity bar.
                let s = bf16_to_f32(scales[base + b]);
                let lo = bf16_to_f32(mins[base + b]);
                let qv = codes[base + b];
                let abase = sbase + 8u * b;
                let a0 = a[abase];       let a1 = a[abase + 1u];
                let a2 = a[abase + 2u];  let a3 = a[abase + 3u];
                let a4 = a[abase + 4u];  let a5 = a[abase + 5u];
                let a6 = a[abase + 6u];  let a7 = a[abase + 7u];
                let code_v4 = a0 * unpack_lo(qv.x) + a1 * unpack_hi(qv.x)
                            + a2 * unpack_lo(qv.y) + a3 * unpack_hi(qv.y)
                            + a4 * unpack_lo(qv.z) + a5 * unpack_hi(qv.z)
                            + a6 * unpack_lo(qv.w) + a7 * unpack_hi(qv.w);
                let a_v4 = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
                let one = vec4<f32>(1.0);
                dotacc = dotacc + rw * (s * dot(code_v4, one) + lo * dot(a_v4, one));
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
