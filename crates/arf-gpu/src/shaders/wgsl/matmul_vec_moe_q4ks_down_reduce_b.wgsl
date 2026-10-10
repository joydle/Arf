// BATCHED (B rows) fused MoE down-projection (Q4_K_S): the B-row generalization of
// matmul_vec_moe_q4ks_down_reduce.wgsl. For each (row, h_j) output, weight-sums the
// down projection over THIS ROW's top_k routed experts in ONE dispatch:
//   mlp_down[row*h + h_j]
//     = Σ_slot wts[row*top_k+slot]·( Σ_c silu[row*top_k*mi + slot*mi + c]
//                                     · Wdown_{ids[row*top_k+slot]}[h_j,c] )
//
// One workgroup per (row, h_j) output element; the flat grid strides over B*h
// (oi → brow = oi/h, h_j = oi%h). The outer loop walks slots ASCENDING (bit-exact
// vs the old sequential accumulate); the inner 32-lane reduce strides over the
// subs = mi/32 sub-blocks of this (row,slot) silu segment. Per (row,slot,h_j) the
// per-expert dequant + dot is BYTE-IDENTICAL to the single-token kernel — only the
// per-row offsets change: ids/wts indexed at row*top_k+slot, the silu segment at
// row*(top_k*mi) + slot*mi, the output at row*h + h_j. With B=1 (brow=0) it reduces
// exactly to the single-token down_reduce → bit-identical to the oracle.
//
// WG=32, fix #2 vec4-partial accumulation, two-level Q4_K_S dequant — all
// IDENTICAL to the single-token kernel. Dims: k=mi, n=h, m=B, top_k=Dims.top_k.
// mode (Dims.slot field repurposed as in the single-token kernel: word[4]=mode):
//   1 write / 2 accumulate. `a` = silu_all (vec4<f32>).

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, top_k: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // silu_all [B*top_k*mi]
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // packed experts' codes
@group(0) @binding(2) var<storage, read_write> c: array<f32>;           // mlp_down [B*h]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // u8 sub-scale, 4/word
@group(0) @binding(5) var<storage, read>       ids: array<u32>;         // [B*top_k]
@group(0) @binding(6) var<storage, read>       wts: array<f32>;         // [B*top_k]
@group(0) @binding(7) var<storage, read>       mins: array<u32>;        // i8 sub-min, 4/word
@group(0) @binding(8) var<storage, read>       dd: array<f32>;          // [d,dmin] per super-block

const WG: u32 = 32u;
var<workgroup> partial: array<f32, 32>;

fn unpack_lo(word: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(0xFu));
}
fn unpack_hi(word: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(word >> 16u, word >> 20u, word >> 24u, word >> 28u) & vec4<u32>(0xFu));
}
fn u8_at(word: u32, b: u32) -> f32 {
    return f32((word >> (8u * b)) & 0xFFu);
}
fn i8_at(word: u32, b: u32) -> f32 {
    return f32(i32(word << (24u - 8u * b)) >> 24u);
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let subs = d.k / 32u;                 // sub-blocks per weight row (k = mi)
    let slot_vecs = d.k / 4u;             // vec4s per slot's silu segment (mi/4)
    let row_vecs = d.top_k * slot_vecs;   // vec4s per batch row's silu (top_k*mi/4)
    let total = d.m * d.n;                // B * h output elements

    var oi = wid.x;
    while (oi < total) {
        let brow = oi / d.n;              // batch row
        let h_j = oi % d.n;               // output hidden index
        let ibase = brow * d.top_k;       // this row's ids/wts segment
        let rbase = brow * row_vecs;      // this row's silu segment (vec4 units)

        var dotacc = 0.0;
        var slot = 0u;
        while (slot < d.top_k) {
            let expert = ids[ibase + slot];
            let rw = wts[ibase + slot];
            let row = expert * d.n + h_j;
            let base = row * subs;
            let dd_base = row * (subs / 8u) * 2u;
            let sbase = rbase + slot * slot_vecs;
            var b = lane;
            while (b < subs) {
                let sblk = b / 8u;
                let dv = dd[dd_base + sblk * 2u];
                let dmv = dd[dd_base + sblk * 2u + 1u];
                let s = dv * u8_at(scales[(base + b) / 4u], (base + b) % 4u);
                let lo = dmv * i8_at(mins[(base + b) / 4u], (base + b) % 4u);
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
                c[oi] = c[oi] + partial[0];
            } else {
                c[oi] = partial[0];
            }
        }
        workgroupBarrier();
        oi = oi + ngroups.x;
    }
}
