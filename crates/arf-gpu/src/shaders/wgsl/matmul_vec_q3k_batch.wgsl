// Batched Q3_K GEMV: C[m,n] = Σ_subblocks ( (d·scale_u8)·Σ(code·A[r]) +
// (dmin·min_i8)·Σ(A[r]) ) for all m rows. Batched analog of matmul_vec_q3k.wgsl —
// decodes each sub-block's 32 codes (from ql/qh) ONCE into a small array, reuses
// across m rows (the batch CSE). Powers Q3_K prefill. Bit-identical to the m==1
// kernel / matmul_nt_q3k.

struct Dims { m: u32, k: u32, n: u32, row_off: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       ql: array<u32>;
@group(0) @binding(2) var<storage, read>       qh: array<u32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<uniform>             d: Dims;
@group(0) @binding(5) var<storage, read>       scales: array<u32>;
@group(0) @binding(6) var<storage, read>       mins: array<u32>;
@group(0) @binding(7) var<storage, read>       dd: array<f32>;

const WG: u32 = 64u;
const MAXM: u32 = 16u;
var<workgroup> partial: array<f32, 1024>;

fn u8_at(word: u32, b: u32) -> f32 {
    return f32((word >> (8u * b)) & 0xFFu);
}
fn i8_at(word: u32, b: u32) -> f32 {
    return f32(i32(word << (24u - 8u * b)) >> 24u);
}
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
    let m = min(d.m, MAXM);
    let subs = d.k / 32u;
    let ql_per_row = d.k / 16u;
    let qh_per_row = d.k / 32u;
    let arow = d.k / 4u;

    var col = wid.x;
    while (col < d.n) {
        let ql_base = col * ql_per_row;
        let qh_base = col * qh_per_row;
        let scale_base = col * subs;
        let dd_base = col * (subs / 8u) * 2u;

        var acc: array<f32, 16>;
        for (var r = 0u; r < m; r = r + 1u) {
            acc[r] = 0.0;
        }

        var sb = lane;
        while (sb < subs) {
            let sblk = sb / 8u;
            let s = dd[dd_base + sblk * 2u] * u8_at(scales[(scale_base + sb) / 4u], (scale_base + sb) % 4u);
            let lo = dd[dd_base + sblk * 2u + 1u] * i8_at(mins[(scale_base + sb) / 4u], (scale_base + sb) % 4u);
            let ql0 = ql[ql_base + sb * 2u];
            let ql1 = ql[ql_base + sb * 2u + 1u];
            let qhw = qh[qh_base + sb];
            // Decode the 8 vec4 code-groups ONCE (hoisted across the m rows — the
            // batch CSE), vectorized (P-Q4b). Reused for every activation row.
            let c0 = codes4(ql0, 0u, qhw, 0u);
            let c1 = codes4(ql0, 8u, qhw, 4u);
            let c2 = codes4(ql0, 16u, qhw, 8u);
            let c3 = codes4(ql0, 24u, qhw, 12u);
            let c4 = codes4(ql1, 0u, qhw, 16u);
            let c5 = codes4(ql1, 8u, qhw, 20u);
            let c6 = codes4(ql1, 16u, qhw, 24u);
            let c7 = codes4(ql1, 24u, qhw, 28u);
            let one = vec4<f32>(1.0);
            for (var r = 0u; r < m; r = r + 1u) {
                let abase = (d.row_off + r) * arow + 8u * sb;
                let a0 = a[abase];      let a1 = a[abase + 1u];
                let a2 = a[abase + 2u]; let a3 = a[abase + 3u];
                let a4 = a[abase + 4u]; let a5 = a[abase + 5u];
                let a6 = a[abase + 6u]; let a7 = a[abase + 7u];
                let code_sum = dot(a0, c0) + dot(a1, c1) + dot(a2, c2) + dot(a3, c3)
                             + dot(a4, c4) + dot(a5, c5) + dot(a6, c6) + dot(a7, c7);
                let x_sum = dot(a0, one) + dot(a1, one) + dot(a2, one) + dot(a3, one)
                          + dot(a4, one) + dot(a5, one) + dot(a6, one) + dot(a7, one);
                acc[r] = acc[r] + s * code_sum + lo * x_sum;
            }
            sb = sb + WG;
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
