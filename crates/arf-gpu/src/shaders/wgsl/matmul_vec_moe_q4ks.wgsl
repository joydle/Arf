// MoE expert matrix·vector, Q4_K_S (super-block-256, 4-bit UNSIGNED, two-level
// u8/i8 scales against a per-super-block f32 d/dmin), batch-1 decode. The Q4_K_S
// analog of matmul_vec_moe_q4k.wgsl: the weight matrix is one expert SELECTED AT
// GPU RUNTIME from PACKED all-experts Q4_K_S buffers (codes + u8 per-sub-block
// scales + i8 per-sub-block mins + f32 [d,dmin] per super-block), indexed by
// ids[slot]. Only the routed experts' bytes stream — fits the 30B and is ~16%
// smaller than Q4_K-lite (u8/i8 sub-scales vs bf16). The dequant is byte-identical
// to the dense matmul_vec_q4ks.wgsl, just with the packed-expert offset added.
//
// Dequant per 32-weight sub-block b (sblk = b/8 super-block): two-level scale
//   s  = dd[dd_base + sblk*2]      · u8(scales at sub-block b)
//   lo = dd[dd_base + sblk*2 + 1]  · i8(mins   at sub-block b)
// contribution = s·Σ(code·a) + lo·Σ(a). Bit-identical to the CPU Q4_K_S MoE oracle
// (same per-sub-block term order, same expert-offset + write-mode contract).
//
// Packed layout (Q4KSMatrix per expert, concatenated over num_experts):
//   codes:  vec4<u32>, 32 unsigned nibbles/vec4 = one 32-weight sub-block; row = k/32 vec4s.
//   scales: u8 per sub-block, packed 4/u32; row = k/32 sub-blocks.
//   mins:   i8 per sub-block, packed 4/u32; row = k/32 sub-blocks.
//   dd:     f32 pairs [d, dmin] per super-block (256 weights = 8 sub-blocks); row = k/256 pairs.
//   For expert e, output col: code/scale/min base = (e*n + col) * (k/32) [sub-block units];
//   dd base = (e*n + col) * (k/256) * 2 [f32 units].
//
// Write mode (same as matmul_vec_moe): 0 raw, 1 write rw·dot, 2 accumulate rw·dot.

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // activation [k], 4/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // packed experts' codes
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // packed experts' u8 sub-scales, 4/word
@group(0) @binding(5) var<storage, read>       ids: array<u32>;
@group(0) @binding(6) var<storage, read>       wts: array<f32>;
@group(0) @binding(7) var<storage, read>       mins: array<u32>;        // packed experts' i8 sub-mins, 4/word
@group(0) @binding(8) var<storage, read>       dd: array<f32>;          // packed experts' [d,dmin] per super-block

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

// Unpack the LOW / HIGH four UNSIGNED nibbles of `word` to f32 in [0,15] (vec4-wide).
fn unpack_lo(word: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(0xFu));
}
fn unpack_hi(word: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(word >> 16u, word >> 20u, word >> 24u, word >> 28u) & vec4<u32>(0xFu));
}
// Extract byte `b` (0..4) of a packed word as an UNSIGNED value (0..255).
fn u8_at(word: u32, b: u32) -> f32 {
    return f32((word >> (8u * b)) & 0xFFu);
}
// Extract byte `b` as a SIGNED i8 value (-128..127) via arithmetic shift.
fn i8_at(word: u32, b: u32) -> f32 {
    return f32(i32(word << (24u - 8u * b)) >> 24u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let subs = d.k / 32u;             // sub-blocks per weight row (one vec4 code-load each)
    let expert = ids[d.slot];
    let rw = wts[d.slot];
    // Packed offset: this expert's matrix starts at expert*n rows; +col rows in.
    let expert_row = expert * d.n;

    var col = wid.x;                  // grid-stride over output columns
    while (col < d.n) {
        let row = expert_row + col;
        let base = row * subs;              // sub-block base for codes/scales/mins
        let dd_base = row * (subs / 8u) * 2u; // f32 base for [d,dmin] pairs
        var acc = 0.0;
        var b = lane;
        while (b < subs) {
            let qv = codes[base + b];
            // Two-level scale: sub-scale u8 × super d; sub-min i8 × super dmin.
            let sblk = b / 8u;
            let dv = dd[dd_base + sblk * 2u];
            let dmv = dd[dd_base + sblk * 2u + 1u];
            let su8 = u8_at(scales[(base + b) / 4u], (base + b) % 4u);
            let mi8 = i8_at(mins[(base + b) / 4u], (base + b) % 4u);
            let s = dv * su8;
            let lo = dmv * mi8;
            let abase = 8u * b;       // 32 activations = 8 vec4<f32>
            let a0 = a[abase];       let a1 = a[abase + 1u];
            let a2 = a[abase + 2u];  let a3 = a[abase + 3u];
            let a4 = a[abase + 4u];  let a5 = a[abase + 5u];
            let a6 = a[abase + 6u];  let a7 = a[abase + 7u];
            // Σ(code·a) over the 32-weight sub-block, plus Σ(a) for the min offset.
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
