// Batched Q4_K_S GEMV: C[m,n] = Σ_subblocks ( (d·scale_u8)·Σ(code·A[r]) +
// (dmin·min_i8)·Σ(A[r]) ) for all m rows. Batched analog of matmul_vec_q4ks.wgsl —
// streams + UNPACKS each 32-weight sub-block ONCE (the batch CSE), reuses across m
// rows. Powers Q4_K_S prefill. Bit-identical to the m==1 kernel / matmul_nt_q4ks.

struct Dims { m: u32, k: u32, n: u32, row_off: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // [m,k], 4/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // 32 nibbles/vec4
@group(0) @binding(2) var<storage, read_write> c: array<f32>;           // [m,n]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // u8, 4/word
@group(0) @binding(5) var<storage, read>       mins: array<u32>;        // i8, 4/word
@group(0) @binding(6) var<storage, read>       dd: array<f32>;          // [d,dmin]/super-block

const WG: u32 = 64u;
const MAXM: u32 = 16u;
var<workgroup> partial: array<f32, 1024>;

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
    let m = min(d.m, MAXM);
    let subs = d.k / 32u;
    let arow = d.k / 4u;

    var col = wid.x;
    while (col < d.n) {
        let code_base = col * subs;
        let scale_base = col * subs;
        let dd_base = col * (subs / 8u) * 2u;

        var acc: array<f32, 16>;
        for (var r = 0u; r < m; r = r + 1u) {
            acc[r] = 0.0;
        }

        var b = lane;
        while (b < subs) {
            let qv = codes[code_base + b];
            let sblk = b / 8u;
            let dv = dd[dd_base + sblk * 2u];
            let dmv = dd[dd_base + sblk * 2u + 1u];
            let s = dv * u8_at(scales[(scale_base + b) / 4u], (scale_base + b) % 4u);
            let lo = dmv * i8_at(mins[(scale_base + b) / 4u], (scale_base + b) % 4u);
            let f0 = unpack_lo(qv.x); let f1 = unpack_hi(qv.x);
            let f2 = unpack_lo(qv.y); let f3 = unpack_hi(qv.y);
            let f4 = unpack_lo(qv.z); let f5 = unpack_hi(qv.z);
            let f6 = unpack_lo(qv.w); let f7 = unpack_hi(qv.w);
            let one = vec4<f32>(1.0);
            for (var r = 0u; r < m; r = r + 1u) {
                let abase = (d.row_off + r) * arow + 8u * b;
                let a0 = a[abase];      let a1 = a[abase + 1u];
                let a2 = a[abase + 2u]; let a3 = a[abase + 3u];
                let a4 = a[abase + 4u]; let a5 = a[abase + 5u];
                let a6 = a[abase + 6u]; let a7 = a[abase + 7u];
                let code_sum = dot(a0, f0) + dot(a1, f1) + dot(a2, f2) + dot(a3, f3)
                             + dot(a4, f4) + dot(a5, f5) + dot(a6, f6) + dot(a7, f7);
                let a_sum = dot(a0, one) + dot(a1, one) + dot(a2, one) + dot(a3, one)
                          + dot(a4, one) + dot(a5, one) + dot(a6, one) + dot(a7, one);
                acc[r] = acc[r] + s * code_sum + lo * a_sum;
            }
            b = b + WG;
        }

        for (var r = 0u; r < m; r = r + 1u) {
            partial[lane * MAXM + r] = acc[r];
        }
        workgroupBarrier();
        var stride = WG / 2u;
        while (stride > 0u) {
            if (lane < stride) {
                let src = (lane + stride) * MAXM;
                let dst = lane * MAXM;
                for (var r = 0u; r < m; r = r + 1u) {
                    partial[dst + r] = partial[dst + r] + partial[src + r];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (lane == 0u) {
            for (var r = 0u; r < m; r = r + 1u) {
                c[(d.row_off + r) * d.n + col] = partial[r];
            }
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
