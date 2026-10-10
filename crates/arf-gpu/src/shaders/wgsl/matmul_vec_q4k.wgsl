// Q4_K-lite (super-block-256, 4-bit UNSIGNED, per-32 asymmetric scale+min) matrix·
// vector for decode: c[n] = Σ_subblocks ( scale[n,s]·Σ(code·a) + min[n,s]·Σ(a) ).
// The asymmetric min-offset (vs Q4_0's symmetric [-8,7]) is the accuracy win —
// it fits each sub-block's range exactly, the same idea ollama's Q4_K_M uses.
//
// Bit-identical to the CPU `matmul_nt_q4k` oracle (same per-sub-block scale·code +
// min·1 accumulation, same unsigned nibble, same term order), so CPU Q4_K is the
// parity oracle.
//
// Layout (see Q4KMatrix): one 32-weight sub-block = one vec4<u32> of codes (8
// unsigned nibbles/word). scales/mins: one bf16 per sub-block (low 16 bits of a u32
// word). k % 256 == 0. Sub-block count = k/32.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // activation [k], 4/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // 32 nibbles/vec4 = one sub-block
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // one bf16/word per sub-block
@group(0) @binding(5) var<storage, read>       mins: array<u32>;        // one bf16/word per sub-block

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>((bits & 0xFFFFu) << 16u);
}
// Unpack the LOW four UNSIGNED nibbles of `word` (weights 0..4) to f32 in [0,15].
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
    var col = wid.x;                 // grid-stride over output columns
    while (col < d.n) {
        let code_base = col * subs;
        let scale_base = col * subs;
        var acc = 0.0;
        var b = lane;
        while (b < subs) {
            let qv = codes[code_base + b];
            let s = bf16_to_f32(scales[scale_base + b]);
            let lo = bf16_to_f32(mins[scale_base + b]);
            let abase = 8u * b;       // 32 activations = 8 vec4<f32>
            // Σ(code·a) over the 32-weight sub-block, plus Σ(a) for the min offset.
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

        if (lane == 0u) {
            c[col] = partial[0];
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
