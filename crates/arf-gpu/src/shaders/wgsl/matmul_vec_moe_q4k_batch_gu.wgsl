// Batched MoE gate/up (Q4_K-lite, super-block-256, per-32 asymmetric scale+min):
// FUSED analog of matmul_vec_moe_q4k.wgsl — all top_k routed experts' gate (or up)
// in ONE dispatch. Output [top_k, n] (slot-major c[slot*n + col]); grid strides flat
// (slot, col); expert = ids[slot]. Raw write (router weight applied later in the
// fused down). top_k in Dims._p0.
//
// Fix #2 (ALU de-stall): the per-sub-block code·a and Σa terms are accumulated as
// vec4 partials (code_v4, a_v4) across the 8 packed sub-vectors, with ONE dot() at
// the end, instead of 8 serial scalar dots each. This shortens the dependency chain
// so the dequant ALU overlaps the weight load (Q4 was ~51% of peak vs int8's ~75%).
// The math is the same code·scale + min·Σa per sub-block; only the intra-sub-block
// summation order changes (4-wide vec adds then one dot, vs left-fold of 8 dots),
// which stays within the existing 1e-3 parity bar (verified — the kernel was
// already a 1e-3 reassociation of the scalar CPU oracle dot_q4k).
//
// MULTI-ROW (lever B, ROWS_PER_WG=2), same pattern as matmul_vec_q4k_mr.wgsl: each
// workgroup owns ROWS_PER_WG contiguous output indices WITHIN ONE SLOT. The group's
// first index is g*ROWS_PER_WG; slot = that / n, col0 = that % n. Because n is a
// multiple of ROWS_PER_WG and col0 is a multiple of ROWS_PER_WG, the ROWS columns
// col0..col0+ROWS-1 never straddle the slot boundary — so all ROWS share the SAME
// slot/expert (same activation segment, different weight columns). Each lane loads
// its sub-block's 8 activation vec4s + a_sum ONCE per sub-block and reuses them
// across all ROWS weight columns. Per-column dequant+dot is byte-identical to the
// single-row form, so this stays bit-identical to the oracle within the 1e-3 bar.

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, top_k: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // activation [k], 4/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // packed experts' codes
@group(0) @binding(2) var<storage, read_write> c: array<f32>;           // [top_k * n]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // bf16/word per sub-block
@group(0) @binding(5) var<storage, read>       ids: array<u32>;
@group(0) @binding(6) var<storage, read>       mins: array<u32>;        // bf16/word per sub-block

const WG: u32 = 64u;
const ROWS_PER_WG: u32 = 2u;
var<workgroup> partial: array<f32, 64>;

fn bf16_to_f32(bits: u32) -> f32 { return bitcast<f32>((bits & 0xFFFFu) << 16u); }
fn unpack_lo(word: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(0xFu));
}
fn unpack_hi(word: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(word >> 16u, word >> 20u, word >> 24u, word >> 28u) & vec4<u32>(0xFu));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let subs = d.k / 32u;               // sub-blocks per weight row
    let total = d.top_k * d.n;

    // Grid-stride over GROUPS of ROWS_PER_WG contiguous output indices. The group's
    // first index is idx0 = g*ROWS_PER_WG → slot = idx0/n, col0 = idx0%n. n is a
    // multiple of ROWS_PER_WG and col0 is a multiple of ROWS_PER_WG, so the ROWS
    // columns col0..col0+ROWS-1 stay within slot (same expert/activation segment).
    var idx0 = wid.x * ROWS_PER_WG;
    let stride_idx = ngroups.x * ROWS_PER_WG;
    while (idx0 < total) {
        let slot = idx0 / d.n;
        let col0 = idx0 % d.n;
        let expert = ids[slot];

        // Per-column running accumulators for the ROWS columns this group owns.
        var acc0 = 0.0;
        var acc1 = 0.0;

        var b = lane;
        while (b < subs) {
            // Load the activation sub-block + a_sum ONCE; reuse across all ROWS cols.
            let abase = 8u * b;
            let a0 = a[abase];       let a1 = a[abase + 1u];
            let a2 = a[abase + 2u];  let a3 = a[abase + 3u];
            let a4 = a[abase + 4u];  let a5 = a[abase + 5u];
            let a6 = a[abase + 6u];  let a7 = a[abase + 7u];
            let a_v4 = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
            let one = vec4<f32>(1.0);
            let a_sum = dot(a_v4, one);

            // Column 0 (col0): own codes/scales/mins at its packed-expert offset.
            if (col0 < d.n) {
                let base = (expert * d.n + col0) * subs;
                let s = bf16_to_f32(scales[base + b]);
                let lo = bf16_to_f32(mins[base + b]);
                let qv = codes[base + b];
                let code_v4 = a0 * unpack_lo(qv.x) + a1 * unpack_hi(qv.x)
                            + a2 * unpack_lo(qv.y) + a3 * unpack_hi(qv.y)
                            + a4 * unpack_lo(qv.z) + a5 * unpack_hi(qv.z)
                            + a6 * unpack_lo(qv.w) + a7 * unpack_hi(qv.w);
                acc0 = acc0 + s * dot(code_v4, one) + lo * a_sum;
            }
            // Column 1 (col0+1): guard the ragged tail (no-op if past n).
            if (col0 + 1u < d.n) {
                let base = (expert * d.n + col0 + 1u) * subs;
                let s = bf16_to_f32(scales[base + b]);
                let lo = bf16_to_f32(mins[base + b]);
                let qv = codes[base + b];
                let code_v4 = a0 * unpack_lo(qv.x) + a1 * unpack_hi(qv.x)
                            + a2 * unpack_lo(qv.y) + a3 * unpack_hi(qv.y)
                            + a4 * unpack_lo(qv.z) + a5 * unpack_hi(qv.z)
                            + a6 * unpack_lo(qv.w) + a7 * unpack_hi(qv.w);
                acc1 = acc1 + s * dot(code_v4, one) + lo * a_sum;
            }
            b = b + WG;
        }

        // Tree-reduce each column's 64 lane-partials in turn, sharing `partial`.
        // (Same left-fold associativity as the single-row kernel → bit-identical.)
        reduce_and_store(lane, slot * d.n + col0,      acc0, col0 < d.n);
        reduce_and_store(lane, slot * d.n + col0 + 1u, acc1, col0 + 1u < d.n);

        idx0 = idx0 + stride_idx;
    }
}

// Tree-reduce `acc` across the 64 lanes into `c[out]` (lane 0 writes when `valid`).
// Barriers bracket the shared-buffer use so the sequential reductions don't race.
fn reduce_and_store(lane: u32, out: u32, acc: f32, valid: bool) {
    workgroupBarrier();
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
    if (lane == 0u && valid && out < arrayLength(&c)) {
        c[out] = partial[0];
    }
}
