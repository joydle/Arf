// Cooperative-matrix GEMM for Q4_K-lite, RB=2 × CB=8: one workgroup computes a
// 16×64 output (2×8 grid of 8×8 tiles) with SIXTEEN coopMultiplyAdd per K-step —
// the widest column block, halving the barrier count again vs CB=4. Each A row-tile
// is reused across all eight column-tiles; each dequantized weight column-tile across
// both row-tiles. Register pressure is the risk (16 live f32 8×8 accumulators ≈ 32
// f32/lane), so this is an A/B candidate, not an unconditional default. Same math +
// boundary contract as matmul_coop_q4k_rb2 (matmul_nt_q4k oracle). Caller dispatches
// for m ≥ 16 with grid (ceil(m/16), ceil(n/64)).

enable wgpu_cooperative_matrix;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;
@group(0) @binding(5) var<storage, read>       mins: array<u32>;

const T: u32 = 8u;
var<workgroup> as0: array<f32, 64>;
var<workgroup> as1: array<f32, 64>;
var<workgroup> bs: array<f32, 512>; // 8 column-tiles laid out [col_tile][kk*T+nn]
var<workgroup> cs_t: array<f32, 64>;

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

fn store_tile(acc: coop_mat8x8<f32, C>, row_base: u32, col_base: u32, M: u32, N: u32, lr: u32, lc: u32) {
    coopStore(acc, &cs_t[0], T);
    workgroupBarrier();
    if (row_base + lr < M && col_base + lc < N) {
        c[(row_base + lr) * N + col_base + lc] = cs_t[lr * T + lc];
    }
    workgroupBarrier();
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = d.m;
    let K = d.k;
    let N = d.n;
    let row0 = wid.x * (T * 2u);
    let col0 = wid.y * (T * 8u);
    let lr = lid.x;
    let lc = lid.y;

    cs_t[lr * T + lc] = 0.0;
    workgroupBarrier();
    var acc00 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc01 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc02 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc03 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc04 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc05 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc06 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc07 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc10 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc11 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc12 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc13 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc14 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc15 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc16 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc17 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);

    let num_tiles = (K + T - 1u) / T;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let ka = t * T + lc;
        let r0 = row0 + lr;
        let r1 = row0 + T + lr;
        as0[lr * T + lc] = select(0.0, a[r0 * K + ka], r0 < M && ka < K);
        as1[lr * T + lc] = select(0.0, a[r1 * K + ka], r1 < M && ka < K);
        let kb = t * T + lr;
        for (var cc = 0u; cc < 8u; cc = cc + 1u) {
            let col = col0 + cc * T + lc;
            bs[cc * 64u + lr * T + lc] = select(0.0, qweight(col, kb, K), col < N && kb < K);
        }
        workgroupBarrier();

        let at0 = coopLoad<coop_mat8x8<f32, A>>(&as0[0], T);
        let at1 = coopLoad<coop_mat8x8<f32, A>>(&as1[0], T);
        let b0 = coopLoad<coop_mat8x8<f32, B>>(&bs[0u], T);
        acc00 = coopMultiplyAdd(at0, b0, acc00);
        acc10 = coopMultiplyAdd(at1, b0, acc10);
        let b1 = coopLoad<coop_mat8x8<f32, B>>(&bs[64u], T);
        acc01 = coopMultiplyAdd(at0, b1, acc01);
        acc11 = coopMultiplyAdd(at1, b1, acc11);
        let b2 = coopLoad<coop_mat8x8<f32, B>>(&bs[128u], T);
        acc02 = coopMultiplyAdd(at0, b2, acc02);
        acc12 = coopMultiplyAdd(at1, b2, acc12);
        let b3 = coopLoad<coop_mat8x8<f32, B>>(&bs[192u], T);
        acc03 = coopMultiplyAdd(at0, b3, acc03);
        acc13 = coopMultiplyAdd(at1, b3, acc13);
        let b4 = coopLoad<coop_mat8x8<f32, B>>(&bs[256u], T);
        acc04 = coopMultiplyAdd(at0, b4, acc04);
        acc14 = coopMultiplyAdd(at1, b4, acc14);
        let b5 = coopLoad<coop_mat8x8<f32, B>>(&bs[320u], T);
        acc05 = coopMultiplyAdd(at0, b5, acc05);
        acc15 = coopMultiplyAdd(at1, b5, acc15);
        let b6 = coopLoad<coop_mat8x8<f32, B>>(&bs[384u], T);
        acc06 = coopMultiplyAdd(at0, b6, acc06);
        acc16 = coopMultiplyAdd(at1, b6, acc16);
        let b7 = coopLoad<coop_mat8x8<f32, B>>(&bs[448u], T);
        acc07 = coopMultiplyAdd(at0, b7, acc07);
        acc17 = coopMultiplyAdd(at1, b7, acc17);
        workgroupBarrier();
    }

    store_tile(acc00, row0,     col0,          M, N, lr, lc);
    store_tile(acc01, row0,     col0 + T,      M, N, lr, lc);
    store_tile(acc02, row0,     col0 + 2u * T, M, N, lr, lc);
    store_tile(acc03, row0,     col0 + 3u * T, M, N, lr, lc);
    store_tile(acc04, row0,     col0 + 4u * T, M, N, lr, lc);
    store_tile(acc05, row0,     col0 + 5u * T, M, N, lr, lc);
    store_tile(acc06, row0,     col0 + 6u * T, M, N, lr, lc);
    store_tile(acc07, row0,     col0 + 7u * T, M, N, lr, lc);
    store_tile(acc10, row0 + T, col0,          M, N, lr, lc);
    store_tile(acc11, row0 + T, col0 + T,      M, N, lr, lc);
    store_tile(acc12, row0 + T, col0 + 2u * T, M, N, lr, lc);
    store_tile(acc13, row0 + T, col0 + 3u * T, M, N, lr, lc);
    store_tile(acc14, row0 + T, col0 + 4u * T, M, N, lr, lc);
    store_tile(acc15, row0 + T, col0 + 5u * T, M, N, lr, lc);
    store_tile(acc16, row0 + T, col0 + 6u * T, M, N, lr, lc);
    store_tile(acc17, row0 + T, col0 + 7u * T, M, N, lr, lc);
}
