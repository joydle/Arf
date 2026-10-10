// Q4_K-lite decode GEMV, faithful port of ggml's mul_mv_q4_K SIMDGROUP structure
// (ggml-metal.metal kernel_mul_mv_q4_K_f32_impl). Differs fundamentally from both
// matmul_vec_q4k.wgsl (1 column/workgroup, tree-reduce) and matmul_vec_q4k_sg.wgsl
// (1 column/workgroup, subgroupAdd only on the final reduce):
//
//   * Each 32-lane SIMDGROUP independently produces NR0 whole output columns. The
//     32 lanes split the K dimension (each owns sub-blocks {sg_lane, sg_lane+32,…});
//     a single `subgroupAdd` per column reduces the lane partials → the column value
//     (no shared memory, no workgroup barrier).
//   * NR0 columns per simdgroup SHARE the per-lane register-resident activation
//     (loaded once, reused across the NR0 columns) — ggml's nr0=2 trick: more
//     arithmetic per byte of activation, and the weight stream is what bounds us.
//   * A workgroup is NSG simdgroups → NSG·NR0 columns per workgroup, many rows in
//     flight (vs our 1 col/wg) → fills the GPU on narrow projections.
//
// naga 29 lowers `subgroupAdd` to Metal `simd_sum` (the exact ggml primitive); the
// earlier "subgroup slower" finding was on the column-per-workgroup _sg shape, NOT
// this row-per-simdgroup one. NR0=2, NSG=4 → workgroup_size 128, 8 cols/workgroup.
//
// NOT bitwise identical to the tree kernel (subgroup reduction reassociates f32);
// far below a logit tie — gated by the decode parity / greedy-cap tests.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;
@group(0) @binding(5) var<storage, read>       mins: array<u32>;

// NR0 columns/simdgroup share the per-lane register activation. ggml hardcodes
// nr0=2 for ALL Apple GPUs; the M4 Max has more registers/bandwidth so a wider NR0
// raises arithmetic-per-activation-byte — the M4-Max-specific tune ggml leaves open.
const NR0: u32 = 4u;   // columns per simdgroup
const NSG: u32 = 2u;   // simdgroups per workgroup (NSG*32 = 64-lane workgroup)
const WGN: u32 = 64u;  // NSG * 32

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
        @builtin(subgroup_size) sg_size: u32,
        @builtin(subgroup_invocation_id) sg_lane: u32,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let sg_id = lid.x / sg_size;          // which simdgroup in this workgroup (0..NSG)
    let subs = d.k / 32u;                  // sub-blocks per column
    let cols_per_wg = NSG * NR0;

    // Grid-stride over column GROUPS of cols_per_wg. This simdgroup owns the NR0
    // columns starting at col0.
    var grp = wid.x;
    let n_groups_cols = (d.n + cols_per_wg - 1u) / cols_per_wg;
    while (grp < n_groups_cols) {
        let col0 = grp * cols_per_wg + sg_id * NR0;

        var acc: array<f32, 4>;  // NR0
        for (var r = 0u; r < NR0; r = r + 1u) { acc[r] = 0.0; }

        // Each lane strides sub-blocks by sg_size; loads the activation slice ONCE
        // and reuses it across the NR0 columns (ggml's shared-activation trick).
        var b = sg_lane;
        while (b < subs) {
            let abase = 8u * b;
            let a0 = a[abase];       let a1 = a[abase + 1u];
            let a2 = a[abase + 2u];  let a3 = a[abase + 3u];
            let a4 = a[abase + 4u];  let a5 = a[abase + 5u];
            let a6 = a[abase + 6u];  let a7 = a[abase + 7u];
            let one = vec4<f32>(1.0);
            let a_sum = dot(a0, one) + dot(a1, one) + dot(a2, one) + dot(a3, one)
                      + dot(a4, one) + dot(a5, one) + dot(a6, one) + dot(a7, one);
            for (var r = 0u; r < NR0; r = r + 1u) {
                let col = col0 + r;
                if (col < d.n) {
                    let cb = col * subs + b;
                    let qv = codes[cb];
                    let s = bf16_to_f32(scales[cb]);
                    let lo = bf16_to_f32(mins[cb]);
                    let code_sum = dot(a0, unpack_lo(qv.x)) + dot(a1, unpack_hi(qv.x))
                                 + dot(a2, unpack_lo(qv.y)) + dot(a3, unpack_hi(qv.y))
                                 + dot(a4, unpack_lo(qv.z)) + dot(a5, unpack_hi(qv.z))
                                 + dot(a6, unpack_lo(qv.w)) + dot(a7, unpack_hi(qv.w));
                    acc[r] = acc[r] + s * code_sum + lo * a_sum;
                }
            }
            b = b + sg_size;
        }

        // One simd_sum per column reduces the 32 lane partials → the column value.
        for (var r = 0u; r < NR0; r = r + 1u) {
            let total = subgroupAdd(acc[r]);
            let col = col0 + r;
            if (sg_lane == 0u && col < d.n) {
                c[col] = total;
            }
        }
        grp = grp + ngroups.x;
    }
}
