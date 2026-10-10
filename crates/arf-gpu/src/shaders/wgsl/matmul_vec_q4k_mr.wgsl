// Q4_K-lite GEMV, MULTI-ROW variant of matmul_vec_q4k.wgsl (lever B).
//
// Same math, same per-sub-block (scale·Σ(code·a) + min·Σ(a)) accumulation, same
// term order — so it is BIT-IDENTICAL to the CPU `matmul_nt_q4k` oracle and the
// single-row kernel; the parity tests cover it unchanged.
//
// The difference is the MEMORY ACCESS PATTERN, not the arithmetic. The single-row
// kernel runs one workgroup per output column, and every column's workgroup
// re-reads the WHOLE activation vector from global memory (n × k activation
// traffic). Here each workgroup computes ROWS_PER_WG (=4) output columns at once:
// each lane loads its sub-block's 8 activation vec4s ONCE and reuses them across
// all 4 columns (4 different weight rows). That cuts activation traffic ~4× and
// raises arithmetic intensity per global load — the weight traffic (the dominant
// ~1.86 GB/token term) is unchanged, each weight still read exactly once.
//
// Reduction stays the shared-memory TREE (no subgroup/simd ops): on this M4 Max,
// simd_sum measured 3.9× SLOWER than the tree for this GEMV shape (lever A), so we
// keep the tree deliberately. Each of the 4 columns gets its own tree-reduce over
// the 64 lanes, sharing one `partial` buffer sequentially.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // activation [k], 4/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // 32 nibbles/vec4 = one sub-block
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // one bf16/word per sub-block
@group(0) @binding(5) var<storage, read>       mins: array<u32>;        // one bf16/word per sub-block

const WG: u32 = 64u;
const ROWS_PER_WG: u32 = 4u;
var<workgroup> partial: array<f32, 64>;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>((bits & 0xFFFFu) << 16u);
}
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
    let subs = d.k / 32u;            // sub-blocks per row (one vec4 code-load each)

    // Grid-stride over column GROUPS of ROWS_PER_WG. Each group computes columns
    // [col0, col0 + ROWS_PER_WG).
    var col0 = wid.x * ROWS_PER_WG;
    let stride_cols = ngroups.x * ROWS_PER_WG;
    while (col0 < d.n) {
        // Per-row running accumulators for the up-to-4 columns this group owns.
        var acc0 = 0.0;
        var acc1 = 0.0;
        var acc2 = 0.0;
        var acc3 = 0.0;

        var b = lane;
        while (b < subs) {
            // Load the activation sub-block ONCE; reuse across all 4 weight rows.
            let abase = 8u * b;
            let a0 = a[abase];       let a1 = a[abase + 1u];
            let a2 = a[abase + 2u];  let a3 = a[abase + 3u];
            let a4 = a[abase + 4u];  let a5 = a[abase + 5u];
            let a6 = a[abase + 6u];  let a7 = a[abase + 7u];
            let one = vec4<f32>(1.0);
            let a_sum = dot(a0, one) + dot(a1, one) + dot(a2, one) + dot(a3, one)
                      + dot(a4, one) + dot(a5, one) + dot(a6, one) + dot(a7, one);

            // Column 0 (always valid: n is a multiple of ROWS_PER_WG for our shapes,
            // but guard anyway so a ragged tail can't read past the row).
            if (col0 < d.n) {
                let base = (col0) * subs + b;
                let qv = codes[base];
                let s = bf16_to_f32(scales[base]);
                let lo = bf16_to_f32(mins[base]);
                let code_sum = dot(a0, unpack_lo(qv.x)) + dot(a1, unpack_hi(qv.x))
                             + dot(a2, unpack_lo(qv.y)) + dot(a3, unpack_hi(qv.y))
                             + dot(a4, unpack_lo(qv.z)) + dot(a5, unpack_hi(qv.z))
                             + dot(a6, unpack_lo(qv.w)) + dot(a7, unpack_hi(qv.w));
                acc0 = acc0 + s * code_sum + lo * a_sum;
            }
            if (col0 + 1u < d.n) {
                let base = (col0 + 1u) * subs + b;
                let qv = codes[base];
                let s = bf16_to_f32(scales[base]);
                let lo = bf16_to_f32(mins[base]);
                let code_sum = dot(a0, unpack_lo(qv.x)) + dot(a1, unpack_hi(qv.x))
                             + dot(a2, unpack_lo(qv.y)) + dot(a3, unpack_hi(qv.y))
                             + dot(a4, unpack_lo(qv.z)) + dot(a5, unpack_hi(qv.z))
                             + dot(a6, unpack_lo(qv.w)) + dot(a7, unpack_hi(qv.w));
                acc1 = acc1 + s * code_sum + lo * a_sum;
            }
            if (col0 + 2u < d.n) {
                let base = (col0 + 2u) * subs + b;
                let qv = codes[base];
                let s = bf16_to_f32(scales[base]);
                let lo = bf16_to_f32(mins[base]);
                let code_sum = dot(a0, unpack_lo(qv.x)) + dot(a1, unpack_hi(qv.x))
                             + dot(a2, unpack_lo(qv.y)) + dot(a3, unpack_hi(qv.y))
                             + dot(a4, unpack_lo(qv.z)) + dot(a5, unpack_hi(qv.z))
                             + dot(a6, unpack_lo(qv.w)) + dot(a7, unpack_hi(qv.w));
                acc2 = acc2 + s * code_sum + lo * a_sum;
            }
            if (col0 + 3u < d.n) {
                let base = (col0 + 3u) * subs + b;
                let qv = codes[base];
                let s = bf16_to_f32(scales[base]);
                let lo = bf16_to_f32(mins[base]);
                let code_sum = dot(a0, unpack_lo(qv.x)) + dot(a1, unpack_hi(qv.x))
                             + dot(a2, unpack_lo(qv.y)) + dot(a3, unpack_hi(qv.y))
                             + dot(a4, unpack_lo(qv.z)) + dot(a5, unpack_hi(qv.z))
                             + dot(a6, unpack_lo(qv.w)) + dot(a7, unpack_hi(qv.w));
                acc3 = acc3 + s * code_sum + lo * a_sum;
            }
            b = b + WG;
        }

        // Tree-reduce each column's 64 lane-partials in turn, sharing `partial`.
        // (Same left-fold associativity as the single-row kernel → bit-identical.)
        reduce_and_store(lane, col0,      acc0);
        reduce_and_store(lane, col0 + 1u, acc1);
        reduce_and_store(lane, col0 + 2u, acc2);
        reduce_and_store(lane, col0 + 3u, acc3);

        col0 = col0 + stride_cols;
    }
}

// Tree-reduce `acc` across the 64 lanes into `c[col]` (lane 0 writes). Guards the
// ragged tail (col >= n is a no-op). Barriers bracket the shared-buffer use so the
// four sequential reductions don't race.
fn reduce_and_store(lane: u32, col: u32, acc: f32) {
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
    if (lane == 0u && col < arrayLength(&c)) {
        c[col] = partial[0];
    }
}
