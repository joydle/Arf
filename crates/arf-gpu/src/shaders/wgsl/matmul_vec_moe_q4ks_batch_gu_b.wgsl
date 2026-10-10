// BATCHED (B rows) MoE gate/up (Q4_K_S): the B-row generalization of
// matmul_vec_moe_q4ks_batch_gu.wgsl. Computes ALL B rows' top_k routed experts'
// gate (or up) projection in ONE dispatch. Output [B, top_k, n] laid out
// (row,slot,col)-major: c[(row*top_k + slot)*n + col]. Raw write (the router
// weight is applied later in the fused down). top_k in Dims.top_k, B in Dims.m.
//
// Per output element (row, slot, col):
//   expert      = ids[row*top_k + slot]            (this row's slot-th routed expert)
//   activation  = a[row*k .. row*k + k]            (this row's rmsnorm'd hidden)
//   weight col  = expert*n + col                   (UNCHANGED per-expert packing)
// so the dot is BYTE-IDENTICAL to the single-token kernel run for (row, slot's
// expert, col). With B=1 (row=0) it reduces exactly to the single-token batch_gu
// kernel → bit-identical to the oracle within the 1e-3 bar.
//
// MULTI-ROW (lever B, ROWS_PER_WG=2) is preserved OVER COLUMNS within one
// (row,slot): each workgroup owns ROWS_PER_WG contiguous output indices, its first
// index idx0 = g*ROWS_PER_WG → row = idx0/(top_k*n), rem = idx0%(top_k*n),
// slot = rem/n, col0 = rem%n. n is a multiple of ROWS_PER_WG and each (row,slot)
// block spans exactly n indices, so col0 is a multiple of ROWS_PER_WG and the
// ROWS columns col0..col0+ROWS-1 never straddle a (row,slot) boundary — they share
// the SAME row/slot/expert/activation segment, different weight columns. Each lane
// loads its sub-block's 8 activation vec4s (from THIS row's segment) + a_sum ONCE
// and reuses them across the ROWS weight columns. Same fix #2 vec4-partial
// accumulation + two-level Q4_K_S unpack as the single-token kernel.
//
// Packed layout (per expert, concatenated over num_experts):
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

const WG: u32 = 64u;
const ROWS_PER_WG: u32 = 2u;
var<workgroup> partial: array<f32, 64>;

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
    let subs = d.k / 32u;               // sub-blocks per weight row
    let supers = subs / 8u;             // super-blocks (256-wide) per weight row
    let arow = d.k / 4u;                // vec4<f32> per activation row
    let per_row = d.top_k * d.n;        // output indices per batch row
    let total = d.m * per_row;          // B * top_k * n

    var idx0 = wid.x * ROWS_PER_WG;
    let stride_idx = ngroups.x * ROWS_PER_WG;
    while (idx0 < total) {
        let brow = idx0 / per_row;      // batch row
        let rem = idx0 % per_row;
        let slot = rem / d.n;           // routed-expert slot within this row
        let col0 = rem % d.n;           // first output column (multiple of ROWS_PER_WG)
        let expert = ids[brow * d.top_k + slot];
        let arow_base = brow * arow;    // this row's activation segment (vec4 units)

        // Per-column running accumulators for the ROWS columns this group owns.
        var acc0 = 0.0;
        var acc1 = 0.0;

        var b = lane;
        while (b < subs) {
            // Load THIS ROW's activation sub-block + a_sum ONCE; reuse across cols.
            let abase = arow_base + 8u * b;
            let a0 = a[abase];       let a1 = a[abase + 1u];
            let a2 = a[abase + 2u];  let a3 = a[abase + 3u];
            let a4 = a[abase + 4u];  let a5 = a[abase + 5u];
            let a6 = a[abase + 6u];  let a7 = a[abase + 7u];
            let a_v4 = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
            let one = vec4<f32>(1.0);
            let a_sum = dot(a_v4, one);
            let sblk = b / 8u;

            // Column 0 (col0): own codes/scales/mins/dd at its packed-expert offset.
            if (col0 < d.n) {
                let row = expert * d.n + col0;
                let base = row * subs;
                let dd_base = row * supers * 2u;
                let dv = dd[dd_base + sblk * 2u];
                let dmv = dd[dd_base + sblk * 2u + 1u];
                let s = dv * u8_at(scales[(base + b) / 4u], (base + b) % 4u);
                let lo = dmv * i8_at(mins[(base + b) / 4u], (base + b) % 4u);
                let qv = codes[base + b];
                let code_v4 = a0 * unpack_lo(qv.x) + a1 * unpack_hi(qv.x)
                            + a2 * unpack_lo(qv.y) + a3 * unpack_hi(qv.y)
                            + a4 * unpack_lo(qv.z) + a5 * unpack_hi(qv.z)
                            + a6 * unpack_lo(qv.w) + a7 * unpack_hi(qv.w);
                acc0 = acc0 + s * dot(code_v4, one) + lo * a_sum;
            }
            // Column 1 (col0+1): guard the ragged tail (no-op if past n).
            if (col0 + 1u < d.n) {
                let row = expert * d.n + col0 + 1u;
                let base = row * subs;
                let dd_base = row * supers * 2u;
                let dv = dd[dd_base + sblk * 2u];
                let dmv = dd[dd_base + sblk * 2u + 1u];
                let s = dv * u8_at(scales[(base + b) / 4u], (base + b) % 4u);
                let lo = dmv * i8_at(mins[(base + b) / 4u], (base + b) % 4u);
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
        // Output index (row,slot,col)-major: (brow*top_k + slot)*n + col.
        let obase = (brow * d.top_k + slot) * d.n;
        reduce_and_store(lane, obase + col0,      acc0, col0 < d.n);
        reduce_and_store(lane, obase + col0 + 1u, acc1, col0 + 1u < d.n);

        idx0 = idx0 + stride_idx;
    }
}

// Tree-reduce `acc` across the 64 lanes into `c[out]` (lane 0 writes when `valid`).
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
