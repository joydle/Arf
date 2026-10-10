// Cooperative-matrix GEMM for Q4_K-lite, register-blocked RB=2 × CB=2: one
// workgroup computes a 16×16 output (a 2×2 grid of 8×8 tiles) with FOUR
// coopMultiplyAdd per K-step instead of the two in matmul_coop_q4k_rb2. The two
// staged A row-tiles are each reused across both column-tiles, and the two
// dequantized weight column-tiles are each reused across both row-tiles, so the
// per-barrier matrix-unit work doubles (4 mma / 2 barriers vs 2 mma / 2 barriers)
// while the A staging + weight dequant per output element is unchanged. This
// lifts the matrix-unit occupancy that capped the RB=2 GEMM at ~15% of roofline
// on the M3 Pro (profiled batched decode, B=32). Same per-element math and
// boundary contract as matmul_coop_q4k_rb2 — the matmul_nt_q4k oracle holds.
// Caller dispatches for m ≥ 16 with grid (ceil(m/16), ceil(n/16)).

enable wgpu_cooperative_matrix;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;
@group(0) @binding(5) var<storage, read>       mins: array<u32>;

const T: u32 = 8u;
var<workgroup> as0: array<f32, 64>; // A rows [row0 .. row0+8)
var<workgroup> as1: array<f32, 64>; // A rows [row0+8 .. row0+16)
var<workgroup> bs0: array<f32, 64>; // weight cols [col0 .. col0+8)   (logical B [kk,nn])
var<workgroup> bs1: array<f32, 64>; // weight cols [col0+8 .. col0+16)
var<workgroup> cs_t: array<f32, 64>; // output stage for the coopStore→global copy

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>((bits & 0xFFFFu) << 16u);
}

fn qweight(col: u32, kk: u32, k: u32) -> f32 {
    let subs = k / 32u;
    let b = kk / 32u;
    let j = kk % 32u;
    let word = codes[(col * subs + b) * 4u + j / 8u];
    let nib = f32((word >> ((j % 8u) * 4u)) & 0xFu);
    let idx = col * subs + b;
    return bf16_to_f32(scales[idx]) * nib + bf16_to_f32(mins[idx]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = d.m;
    let K = d.k;
    let N = d.n;
    let row0 = wid.x * (T * 2u);
    let col0 = wid.y * (T * 2u);
    let lr = lid.x;
    let lc = lid.y;

    cs_t[lr * T + lc] = 0.0;
    workgroupBarrier();
    var acc00 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc01 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc10 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc11 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);

    let num_tiles = (K + T - 1u) / T;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let ka = t * T + lc;
        let r0 = row0 + lr;
        let r1 = row0 + T + lr;
        as0[lr * T + lc] = select(0.0, a[r0 * K + ka], r0 < M && ka < K);
        as1[lr * T + lc] = select(0.0, a[r1 * K + ka], r1 < M && ka < K);
        let kb = t * T + lr;
        let c0 = col0 + lc;
        let c1 = col0 + T + lc;
        bs0[lr * T + lc] = select(0.0, qweight(c0, kb, K), c0 < N && kb < K);
        bs1[lr * T + lc] = select(0.0, qweight(c1, kb, K), c1 < N && kb < K);
        workgroupBarrier();

        let at0 = coopLoad<coop_mat8x8<f32, A>>(&as0[0], T);
        let at1 = coopLoad<coop_mat8x8<f32, A>>(&as1[0], T);
        let bt0 = coopLoad<coop_mat8x8<f32, B>>(&bs0[0], T);
        let bt1 = coopLoad<coop_mat8x8<f32, B>>(&bs1[0], T);
        acc00 = coopMultiplyAdd(at0, bt0, acc00);
        acc01 = coopMultiplyAdd(at0, bt1, acc01);
        acc10 = coopMultiplyAdd(at1, bt0, acc10);
        acc11 = coopMultiplyAdd(at1, bt1, acc11);
        workgroupBarrier();
    }

    // Drain the four accumulators through the shared stage one at a time (the
    // coopStore writes a full 8×8, the guarded scalar copy masks the ragged edge).
    coopStore(acc00, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + lr < M && col0 + lc < N) {
        c[(row0 + lr) * N + col0 + lc] = cs_t[lr * T + lc];
    }
    workgroupBarrier();
    coopStore(acc01, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + lr < M && col0 + T + lc < N) {
        c[(row0 + lr) * N + col0 + T + lc] = cs_t[lr * T + lc];
    }
    workgroupBarrier();
    coopStore(acc10, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + T + lr < M && col0 + lc < N) {
        c[(row0 + T + lr) * N + col0 + lc] = cs_t[lr * T + lc];
    }
    workgroupBarrier();
    coopStore(acc11, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + T + lr < M && col0 + T + lc < N) {
        c[(row0 + T + lr) * N + col0 + T + lc] = cs_t[lr * T + lc];
    }
}
