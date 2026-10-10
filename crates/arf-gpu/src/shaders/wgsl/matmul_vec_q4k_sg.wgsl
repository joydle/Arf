// Q4_K-lite matrix·vector for decode, SUBGROUP-reduction variant of
// matmul_vec_q4k.wgsl. Same per-sub-block dequant + dot; the across-lane reduction
// uses `subgroupAdd` (one op) + a tiny sg_partial[16] cross-subgroup combine (one
// barrier) instead of the 6-step shared-memory tree. On a reduction-bound decode
// GEMV that tree is the dominant cost; subgroup ops are ~1.85-5× on it (MLX).
//
// No `enable subgroups;` (naga 29 rejects the directive but implements subgroupAdd;
// wgpu enables the SUBGROUP feature at pipeline creation when the adapter has it).
// Used only when has_subgroups(); else the caller dispatches matmul_vec_q4k.wgsl.
// The reduction order differs from the tree → NOT bitwise identical (f32 reassoc),
// but far below a logit tie; covered by the decode parity / greedy-cap tests.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // activation [k], 4/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // 32 nibbles/vec4 = one sub-block
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // one bf16/word per sub-block
@group(0) @binding(5) var<storage, read>       mins: array<u32>;        // one bf16/word per sub-block

const WG: u32 = 64u;
var<workgroup> sg_partial: array<f32, 16>;

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
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sg_size: u32,
        @builtin(subgroup_invocation_id) sg_lane: u32) {
    let lane = lid.x;
    let sg_id = lane / sg_size;
    let num_sg = WG / sg_size;
    let subs = d.k / 32u;
    var col = wid.x;
    while (col < d.n) {
        let code_base = col * subs;
        let scale_base = col * subs;
        var acc = 0.0;
        var b = lane;
        while (b < subs) {
            let qv = codes[code_base + b];
            let s = bf16_to_f32(scales[scale_base + b]);
            let lo = bf16_to_f32(mins[scale_base + b]);
            let abase = 8u * b;
            let a0 = a[abase];       let a1 = a[abase + 1u];
            let a2 = a[abase + 2u];  let a3 = a[abase + 3u];
            let a4 = a[abase + 4u];  let a5 = a[abase + 5u];
            let a6 = a[abase + 6u];  let a7 = a[abase + 7u];
            let code_sum = dot(a0, unpack_lo(qv.x)) + dot(a1, unpack_hi(qv.x))
                         + dot(a2, unpack_lo(qv.y)) + dot(a3, unpack_hi(qv.y))
                         + dot(a4, unpack_lo(qv.z)) + dot(a5, unpack_hi(qv.z))
                         + dot(a6, unpack_lo(qv.w)) + dot(a7, unpack_hi(qv.w));
            let one = vec4<f32>(1.0);
            let a_sum = dot(a0, one) + dot(a1, one) + dot(a2, one) + dot(a3, one)
                      + dot(a4, one) + dot(a5, one) + dot(a6, one) + dot(a7, one);
            acc = acc + s * code_sum + lo * a_sum;
            b = b + WG;
        }

        let sg_sum = subgroupAdd(acc);
        if (sg_lane == 0u) {
            sg_partial[sg_id] = sg_sum;
        }
        workgroupBarrier();
        if (lane == 0u) {
            var total = 0.0;
            for (var sgx = 0u; sgx < num_sg; sgx = sgx + 1u) {
                total = total + sg_partial[sgx];
            }
            c[col] = total;
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
