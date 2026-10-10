// BATCHED (B rows) MoE gate/up (Q4_K_S) — v2 nr0/NSG layout.
//
// A B-row generalization of matmul_vec_moe_q4ks_batch_gu.wgsl, byte-identical in
// arithmetic to v1 (matmul_vec_moe_q4ks_batch_gu_b.wgsl) but with the llama.cpp
// mul_mv_id "nr0/NSG" thread layout ported to PLAIN WGSL (no subgroupAdd — the
// SUBGROUP device feature is default-off and pessimizes every pipeline ~1.33x, and
// naga poisons simd_sum; we use a shared-memory 32-lane tree reduce instead).
//
// LAYOUT: a @workgroup_size(64) workgroup = GROUPS (=2) reducer groups of 32 lanes.
// Each group computes NR0 output columns. So one workgroup produces GROUPS*NR0
// output indices vs v1's ROWS_PER_WG=2. Each lane loads its activation sub-block
// ONCE per 32-stride and reuses it across all NR0 columns of its group — the same
// activation-load amortization as v1, but NR0-wide instead of 2-wide, and the final
// reduction is a shallow 32-lane tree (5 steps) instead of v1's 64-lane (6 steps).
//
// Output [B, top_k, n] laid out (row,slot,col)-major: c[(row*top_k + slot)*n + col].
// The router weight is applied later in the fused down (raw write here). top_k in
// Dims.top_k, B in Dims.m, n = moe_inter, k = hidden.
//
// Per output element (row, slot, col):
//   expert      = ids[row*top_k + slot]
//   activation  = a[row*k .. row*k + k]
//   weight col  = expert*n + col
// so each dot is BYTE-IDENTICAL to the single-token kernel (and to v1) for
// (row, slot's expert, col). With B=1 (row=0) it reduces to the single-token kernel
// → bit-identical to the oracle within the 1e-3 bar.
//
// GRID: one workgroup owns INDICES_PER_WG = GROUPS*NR0 contiguous output indices,
// grid-strided over total = B*top_k*n. n (=768) is a multiple of INDICES_PER_WG
// (768 = 2^8·3, and INDICES_PER_WG ∈ {8,16} both divide it), so the GROUPS*NR0
// columns a workgroup owns never straddle a (row,slot) boundary — they share the
// SAME row/slot/expert/activation segment, differing only in weight column. The
// dispatch (batch.rs batched_gu_b) divides the index cap by INDICES_PER_WG.
//
// Packed layout (per expert, concatenated over num_experts) — UNCHANGED vs v1:
//   codes:  vec4<u32> per sub-block; base = (expert*n + col) * (k/32) [sub-block units].
//   scales: u8 per sub-block, packed 4/u32; same sub-block base.
//   mins:   i8 per sub-block, packed 4/u32; same sub-block base.
//   dd:     f32 [d,dmin] per super-block; base = (expert*n + col) * (k/256) * 2.

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, top_k: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // activation [B*k], 4/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // packed experts' codes
@group(0) @binding(2) var<storage, read_write> c: array<f32>;           // [B * top_k * n]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // u8 sub-scale, 4/word
@group(0) @binding(5) var<storage, read>       ids: array<u32>;         // [B * top_k]
@group(0) @binding(6) var<storage, read>       mins: array<u32>;        // i8 sub-min, 4/word
@group(0) @binding(7) var<storage, read>       dd: array<f32>;          // [d,dmin] per super-block

// Two 32-lane reducer groups (GROUPS=2) per @workgroup_size(64). Each group owns
// NR0 output columns; INDICES_PER_WG = GROUPS*NR0 columns per workgroup.
const SG: u32 = 32u;                 // reducer-group lane count
const GROUPS: u32 = 2u;              // reducer groups per workgroup (64/32)
// NR0=4 is the measured-best default (qwen3-coder-30B Q4_K_S, M-series): tied with
// NR0=8 at conc16, marginally faster at conc32, half the shared memory (better
// occupancy headroom). To A/B NR0=8: set NR0=8u, partial to array<f32,512>, acc to
// array<f32,8>, and INDICES_PER_WG_V2 in batch.rs to 16.
const NR0: u32 = 4u;                 // output columns per reducer group
const INDICES_PER_WG: u32 = GROUPS * NR0;
var<workgroup> partial: array<f32, 256>;  // GROUPS*NR0*SG (=2*4*32=256)

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

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let grp = lane / SG;                 // which reducer group (0..GROUPS)
    let sl = lane % SG;                  // lane within the group (0..32)
    let subs = d.k / 32u;                // sub-blocks per weight row
    let supers = subs / 8u;              // super-blocks (256-wide) per weight row
    let arow = d.k / 4u;                 // vec4<f32> per activation row
    let per_row = d.top_k * d.n;         // output indices per batch row
    let total = d.m * per_row;           // B * top_k * n

    var idx0 = wid.x * INDICES_PER_WG;
    let stride_idx = ngroups.x * INDICES_PER_WG;
    while (idx0 < total) {
        // This group's first column index. All INDICES_PER_WG indices in this
        // workgroup share one (row,slot) since INDICES_PER_WG divides n.
        let gidx0 = idx0 + grp * NR0;
        let brow = gidx0 / per_row;      // batch row
        let rem = gidx0 % per_row;
        let slot = rem / d.n;            // routed-expert slot within this row
        let col0 = rem % d.n;            // this group's first output column
        let expert = ids[brow * d.top_k + slot];
        let arow_base = brow * arow;     // this row's activation segment (vec4 units)
        let obase = (brow * d.top_k + slot) * d.n;

        // Per-column running accumulators (NR0 columns this group owns).
        var acc: array<f32, 4>;          // NR0 ≤ 4
        for (var j = 0u; j < NR0; j = j + 1u) { acc[j] = 0.0; }

        var bsub = sl;
        while (bsub < subs) {
            // Load THIS ROW's activation sub-block + a_sum ONCE; reuse over NR0 cols.
            let abase = arow_base + 8u * bsub;
            let a0 = a[abase];       let a1 = a[abase + 1u];
            let a2 = a[abase + 2u];  let a3 = a[abase + 3u];
            let a4 = a[abase + 4u];  let a5 = a[abase + 5u];
            let a6 = a[abase + 6u];  let a7 = a[abase + 7u];
            let a_v4 = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
            let one = vec4<f32>(1.0);
            let a_sum = dot(a_v4, one);
            let sblk = bsub / 8u;

            for (var j = 0u; j < NR0; j = j + 1u) {
                let col = col0 + j;
                if (col < d.n) {
                    let row = expert * d.n + col;
                    let base = row * subs;
                    let dd_base = row * supers * 2u;
                    let dv = dd[dd_base + sblk * 2u];
                    let dmv = dd[dd_base + sblk * 2u + 1u];
                    let s = dv * u8_at(scales[(base + bsub) / 4u], (base + bsub) % 4u);
                    let lo = dmv * i8_at(mins[(base + bsub) / 4u], (base + bsub) % 4u);
                    let qv = codes[base + bsub];
                    let code_v4 = a0 * unpack_lo(qv.x) + a1 * unpack_hi(qv.x)
                                + a2 * unpack_lo(qv.y) + a3 * unpack_hi(qv.y)
                                + a4 * unpack_lo(qv.z) + a5 * unpack_hi(qv.z)
                                + a6 * unpack_lo(qv.w) + a7 * unpack_hi(qv.w);
                    acc[j] = acc[j] + s * dot(code_v4, one) + lo * a_sum;
                }
            }
            bsub = bsub + SG;
        }

        // Stage all GROUPS*NR0 lane-partials, then a 32-lane tree-reduce per
        // (group, column). partial slot for (grp, j, sl) = (grp*NR0 + j)*SG + sl.
        workgroupBarrier();
        for (var j = 0u; j < NR0; j = j + 1u) {
            partial[(grp * NR0 + j) * SG + sl] = acc[j];
        }
        workgroupBarrier();
        // Each group reduces its own NR0 columns independently over its 32 lanes.
        var stride = SG / 2u;
        while (stride > 0u) {
            if (sl < stride) {
                for (var j = 0u; j < NR0; j = j + 1u) {
                    let pbase = (grp * NR0 + j) * SG;
                    partial[pbase + sl] = partial[pbase + sl] + partial[pbase + sl + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (sl == 0u) {
            for (var j = 0u; j < NR0; j = j + 1u) {
                let col = col0 + j;
                let out = obase + col;
                if (col < d.n && out < arrayLength(&c)) {
                    c[out] = partial[(grp * NR0 + j) * SG];
                }
            }
        }
        workgroupBarrier();

        idx0 = idx0 + stride_idx;
    }
}
