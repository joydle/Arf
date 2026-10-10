// Q3_K (super-block-256, 3-bit split-plane, u8/i8 two-level scales) matrix·vector
// for decode: c[n] = Σ_subblocks ( (d·scale_u8)·Σ(code·a) + (dmin·min_i8)·Σ(a) ).
// 3-bit codes are split (ggml Q3_K style): ql = 2 low bits (16/u32), qh = 1 high
// bit (32/u32); code = (qh<<2)|ql, range [0,7]. ~25% fewer code bytes than Q4 →
// faster decode (bandwidth-bound), at a real 3-bit accuracy cost. Bit-identical to
// the CPU `matmul_nt_q3k` oracle.
//
// Layout per 32-weight sub-block: ql is 2 u32 words (16 weights each), qh is 1 u32
// word (32 weights). scales: u8 packed 4/u32. mins: i8 packed 4/u32. dd: f32 pairs
// [d, dmin] per super-block. k % 256 == 0.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;  // [k], 4/vec
@group(0) @binding(1) var<storage, read>       ql: array<u32>;       // 2 low bits, 16/u32
@group(0) @binding(2) var<storage, read>       qh: array<u32>;       // 1 high bit, 32/u32
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<uniform>             d: Dims;
@group(0) @binding(5) var<storage, read>       scales: array<u32>;   // u8, 4/word
@group(0) @binding(6) var<storage, read>       mins: array<u32>;     // i8, 4/word
@group(0) @binding(7) var<storage, read>       dd: array<f32>;       // [d,dmin]/super-block

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

fn u8_at(word: u32, b: u32) -> f32 {
    return f32((word >> (8u * b)) & 0xFFu);
}
fn i8_at(word: u32, b: u32) -> f32 {
    return f32(i32(word << (24u - 8u * b)) >> 24u);
}
// Reconstruct 4 consecutive 3-bit codes into a vec4<f32>, VECTORIZED (P-Q4b-style):
// `qlw`/`shl` give the 2 low bits of group `g` (4 weights), `qhw`/`shh` the 1 high
// bit; combine code = (hi<<2)|lo for all four lanes at once. `shl = (g%4)*8` into
// the right ql word; `shh = g*4` into qhw.
fn codes4(qlw: u32, shl: u32, qhw: u32, shh: u32) -> vec4<f32> {
    let l = vec4<u32>(qlw >> shl, qlw >> (shl + 2u), qlw >> (shl + 4u), qlw >> (shl + 6u))
        & vec4<u32>(0x3u);
    let h = vec4<u32>(qhw >> shh, qhw >> (shh + 1u), qhw >> (shh + 2u), qhw >> (shh + 3u))
        & vec4<u32>(0x1u);
    return vec4<f32>((h << vec4<u32>(2u)) | l);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let subs = d.k / 32u;                 // sub-blocks per row
    let ql_per_row = d.k / 16u;           // ql u32 words per row
    let qh_per_row = d.k / 32u;           // qh u32 words per row
    var col = wid.x;
    while (col < d.n) {
        let ql_base = col * ql_per_row;
        let qh_base = col * qh_per_row;
        let scale_base = col * subs;
        let dd_base = col * (subs / 8u) * 2u;
        var acc = 0.0;
        var sb = lane;                    // one sub-block per lane (grid-strided)
        while (sb < subs) {
            let sblk = sb / 8u;
            let dv = dd[dd_base + sblk * 2u];
            let dmv = dd[dd_base + sblk * 2u + 1u];
            let s = dv * u8_at(scales[(scale_base + sb) / 4u], (scale_base + sb) % 4u);
            let lo = dmv * i8_at(mins[(scale_base + sb) / 4u], (scale_base + sb) % 4u);
            // This sub-block's 32 weights: 2 ql words (16 each) + 1 qh word (32).
            let ql0 = ql[ql_base + sb * 2u];
            let ql1 = ql[ql_base + sb * 2u + 1u];
            let qhw = qh[qh_base + sb];
            let abase = 8u * sb;          // 32 activations = 8 vec4<f32>; 8 groups of 4
            // 8 vec4 groups: groups 0..4 use ql0, 4..8 use ql1 (each 16 weights = 4
            // groups). qh holds all 32 high bits; group g at bit g*4. Vectorized
            // unpack + `dot` per group — the P-Q4b fix (was a scalar 32-iter loop +
            // switch running at ~43 GB/s; this restores Q4-class bandwidth).
            let c0 = codes4(ql0, 0u, qhw, 0u);
            let c1 = codes4(ql0, 8u, qhw, 4u);
            let c2 = codes4(ql0, 16u, qhw, 8u);
            let c3 = codes4(ql0, 24u, qhw, 12u);
            let c4 = codes4(ql1, 0u, qhw, 16u);
            let c5 = codes4(ql1, 8u, qhw, 20u);
            let c6 = codes4(ql1, 16u, qhw, 24u);
            let c7 = codes4(ql1, 24u, qhw, 28u);
            let a0 = a[abase];      let a1 = a[abase + 1u];
            let a2 = a[abase + 2u]; let a3 = a[abase + 3u];
            let a4 = a[abase + 4u]; let a5 = a[abase + 5u];
            let a6 = a[abase + 6u]; let a7 = a[abase + 7u];
            let code_sum = dot(a0, c0) + dot(a1, c1) + dot(a2, c2) + dot(a3, c3)
                         + dot(a4, c4) + dot(a5, c5) + dot(a6, c6) + dot(a7, c7);
            let one = vec4<f32>(1.0);
            let x_sum = dot(a0, one) + dot(a1, one) + dot(a2, one) + dot(a3, one)
                      + dot(a4, one) + dot(a5, one) + dot(a6, one) + dot(a7, one);
            acc = acc + s * code_sum + lo * x_sum;
            sb = sb + WG;
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
