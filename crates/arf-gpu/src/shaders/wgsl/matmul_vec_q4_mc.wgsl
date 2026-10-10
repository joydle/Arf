// Multi-COLUMN Q4_0 GEMV for decode: each workgroup computes C=4 output columns
// instead of 1. The activation vector a[] (shared by every output column) is loaded
// ONCE per block and dotted against C weight columns, and the workgroup count drops
// C×. Targets the medium-n FFN matmuls (n=15360) where the 1-column-per-workgroup
// matmul_vec_q4 under-occupies the GPU (46% roofline vs 68% on the huge lm_head).
//
// Bit-IDENTICAL math to matmul_vec_q4.wgsl (same vec4 block load, same branchless
// nibble unpack, same per-block bf16 scale, same f32 dot + tree reduction), just
// folded over C columns — so the CPU matmul_nt_q4 oracle still gates it.
//
// Grid: ceil(n / C) workgroups. Workgroup g owns columns [g*C, g*C + C).

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // activation [k], 4/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // 32 nibbles/vec4 = one block
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // 1 bf16/word per block

const WG: u32 = 64u;
const C: u32 = 4u;                       // output columns per workgroup
var<workgroup> partial: array<f32, 256>; // WG * C = 64 * 4 (one bank per column)

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>((bits & 0xFFFFu) << 16u);
}
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
    let blocks = d.k / 32u;          // blocks per row (one vec4 code-load each)
    let col0 = wid.x * C;            // first output column this workgroup owns

    // Per-column accumulators for this lane.
    var acc: array<f32, 4>;
    acc[0] = 0.0; acc[1] = 0.0; acc[2] = 0.0; acc[3] = 0.0;

    var b = lane;
    while (b < blocks) {
        // Load the C weight blocks for this block index ONCE-per-column, but the
        // activation vec4s are loaded ONCE and reused across all C columns — the
        // amortization win over the 1-column kernel.
        let abase = 8u * b;          // 32 activations = 8 vec4<f32>
        let a0 = a[abase];       let a1 = a[abase + 1u];
        let a2 = a[abase + 2u];  let a3 = a[abase + 3u];
        let a4 = a[abase + 4u];  let a5 = a[abase + 5u];
        let a6 = a[abase + 6u];  let a7 = a[abase + 7u];

        for (var j = 0u; j < C; j = j + 1u) {
            let col = col0 + j;
            if (col >= d.n) { continue; }
            let base = col * blocks; // code + scale base (same units: one per block)
            let qv = codes[base + b];
            let s = bf16_to_f32(scales[base + b]);
            let blk = dot(a0, unpack_lo(qv.x)) + dot(a1, unpack_hi(qv.x))
                    + dot(a2, unpack_lo(qv.y)) + dot(a3, unpack_hi(qv.y))
                    + dot(a4, unpack_lo(qv.z)) + dot(a5, unpack_hi(qv.z))
                    + dot(a6, unpack_lo(qv.w)) + dot(a7, unpack_hi(qv.w));
            acc[j] = acc[j] + s * blk;
        }
        b = b + WG;
    }

    // Stash each column's partial in its own bank, then tree-reduce per column.
    partial[lane]            = acc[0];
    partial[lane + 64u]      = acc[1];
    partial[lane + 128u]     = acc[2];
    partial[lane + 192u]     = acc[3];
    workgroupBarrier();

    var stride = WG / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            partial[lane]        = partial[lane]        + partial[lane + stride];
            partial[lane + 64u]  = partial[lane + 64u]  + partial[lane + 64u + stride];
            partial[lane + 128u] = partial[lane + 128u] + partial[lane + 128u + stride];
            partial[lane + 192u] = partial[lane + 192u] + partial[lane + 192u + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        if (col0 + 0u < d.n) { c[col0 + 0u] = partial[0]; }
        if (col0 + 1u < d.n) { c[col0 + 1u] = partial[64u]; }
        if (col0 + 2u < d.n) { c[col0 + 2u] = partial[128u]; }
        if (col0 + 3u < d.n) { c[col0 + 3u] = partial[192u]; }
    }
}
