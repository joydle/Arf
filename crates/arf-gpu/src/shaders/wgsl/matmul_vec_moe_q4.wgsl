// MoE expert matrix·vector, Q4_0 (block-32 4-bit), batch-1 decode. The Q4 analog of
// matmul_vec_moe.wgsl: the weight matrix is one expert SELECTED AT GPU RUNTIME from
// PACKED all-experts Q4 buffers (codes + bf16 block-scales), indexed by ids[slot].
// Only the routed experts' Q4 bytes stream — fits the 30B (Q4 ≈ half bf16) and runs
// at Q4 bandwidth. Bit-exact vs the CPU Q4 MoE oracle (matmul_vec_q4's dot math,
// per matmul_vec_moe's expert-offset + write-mode contract).
//
// Packed layout (Q4Matrix per expert, concatenated over num_experts):
//   codes:  vec4<u32>, 32 nibbles/vec4 = one 32-weight block; row = k/32 vec4s.
//   scales: u32, ONE bf16 block-scale per word; row = k/32 words.
//   For expert e, output col: code/scale base = (e*n + col) * (k/32).
//
// Write mode (same as matmul_vec_moe): 0 raw, 1 write rw·dot, 2 accumulate rw·dot.

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // activation [k], 4/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // packed experts' codes
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // packed experts' scales
@group(0) @binding(5) var<storage, read>       ids: array<u32>;
@group(0) @binding(6) var<storage, read>       wts: array<f32>;

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

fn bf16_to_f32(bits: u32) -> f32 { return bitcast<f32>((bits & 0xFFFFu) << 16u); }

// Unpack the LOW four nibbles of `word` to signed f32 in [-8,7] (vec4-wide).
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
    let blocks = d.k / 32u;            // blocks per weight row
    let expert = ids[d.slot];
    let rw = wts[d.slot];
    // Packed offset: this expert's matrix starts at expert*n rows; +col rows in.
    let expert_row = expert * d.n;

    var col = wid.x;                   // grid-stride over output columns
    while (col < d.n) {
        let code_base = (expert_row + col) * blocks;   // vec4<u32> per block
        let scale_base = (expert_row + col) * blocks;  // one bf16 scale per block
        var acc = 0.0;
        var b = lane;
        while (b < blocks) {
            let qv = codes[code_base + b];
            let s = bf16_to_f32(scales[scale_base + b]);
            let abase = 8u * b;        // 32 activations = 8 vec4<f32>
            let blk = dot(a[abase],      unpack_lo(qv.x)) + dot(a[abase + 1u], unpack_hi(qv.x))
                    + dot(a[abase + 2u], unpack_lo(qv.y)) + dot(a[abase + 3u], unpack_hi(qv.y))
                    + dot(a[abase + 4u], unpack_lo(qv.z)) + dot(a[abase + 5u], unpack_hi(qv.z))
                    + dot(a[abase + 6u], unpack_lo(qv.w)) + dot(a[abase + 7u], unpack_hi(qv.w));
            acc = acc + s * blk;
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
            if (d.mode == 2u) {
                c[col] = c[col] + rw * partial[0];
            } else if (d.mode == 1u) {
                c[col] = rw * partial[0];
            } else {
                c[col] = partial[0];
            }
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
