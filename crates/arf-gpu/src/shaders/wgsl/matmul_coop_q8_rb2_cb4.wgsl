// Cooperative-matrix GEMM for INT8 weights, RB=2 × CB=4: one workgroup computes a
// 16×32 output (2×4 grid of 8×8 tiles) with EIGHT coopMultiplyAdd per K-step. Each
// of the four staged int8 weight column-tiles is reused across both row-tiles, each
// A row-tile across all four column-tiles — 4× the matrix-unit work per barrier vs
// matmul_coop_q8_rb2. The per-column dequant scale[n] factors out of the dot and is
// applied at each guarded store (per column-tile). Same math/boundary contract;
// matmul_nt_q8 oracle. Caller dispatches for m ≥ 16 with grid (ceil(m/16), ceil(n/32)).

enable wgpu_cooperative_matrix;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       q: array<u32>;     // int8 [n,k], 4/word
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scale: array<f32>; // per output col (n)

const T: u32 = 8u;
var<workgroup> as0: array<f32, 64>;
var<workgroup> as1: array<f32, 64>;
var<workgroup> bs0: array<f32, 64>;
var<workgroup> bs1: array<f32, 64>;
var<workgroup> bs2: array<f32, 64>;
var<workgroup> bs3: array<f32, 64>;
var<workgroup> cs_t: array<f32, 64>;

fn i8_at(word: u32, j: u32) -> i32 {
    return (i32(word << (24u - 8u * j)) >> 24u);
}

fn qweight(col: u32, kk: u32, k: u32) -> f32 {
    let word = q[col * (k / 4u) + kk / 4u];
    return f32(i8_at(word, kk % 4u));
}

// Store one 8×8 tile, applying the per-column int8 scale (col_base+lc).
fn store_tile(acc: coop_mat8x8<f32, C>, row_base: u32, col_base: u32, M: u32, N: u32, lr: u32, lc: u32) {
    coopStore(acc, &cs_t[0], T);
    workgroupBarrier();
    let col = col_base + lc;
    if (row_base + lr < M && col < N) {
        c[(row_base + lr) * N + col] = cs_t[lr * T + lc] * scale[col];
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
    let col0 = wid.y * (T * 4u);
    let lr = lid.x;
    let lc = lid.y;

    cs_t[lr * T + lc] = 0.0;
    workgroupBarrier();
    var acc00 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc01 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc02 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc03 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc10 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc11 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc12 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc13 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);

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
        let c2 = col0 + 2u * T + lc;
        let c3 = col0 + 3u * T + lc;
        bs0[lr * T + lc] = select(0.0, qweight(c0, kb, K), c0 < N && kb < K);
        bs1[lr * T + lc] = select(0.0, qweight(c1, kb, K), c1 < N && kb < K);
        bs2[lr * T + lc] = select(0.0, qweight(c2, kb, K), c2 < N && kb < K);
        bs3[lr * T + lc] = select(0.0, qweight(c3, kb, K), c3 < N && kb < K);
        workgroupBarrier();

        let at0 = coopLoad<coop_mat8x8<f32, A>>(&as0[0], T);
        let at1 = coopLoad<coop_mat8x8<f32, A>>(&as1[0], T);
        let bt0 = coopLoad<coop_mat8x8<f32, B>>(&bs0[0], T);
        let bt1 = coopLoad<coop_mat8x8<f32, B>>(&bs1[0], T);
        let bt2 = coopLoad<coop_mat8x8<f32, B>>(&bs2[0], T);
        let bt3 = coopLoad<coop_mat8x8<f32, B>>(&bs3[0], T);
        acc00 = coopMultiplyAdd(at0, bt0, acc00);
        acc01 = coopMultiplyAdd(at0, bt1, acc01);
        acc02 = coopMultiplyAdd(at0, bt2, acc02);
        acc03 = coopMultiplyAdd(at0, bt3, acc03);
        acc10 = coopMultiplyAdd(at1, bt0, acc10);
        acc11 = coopMultiplyAdd(at1, bt1, acc11);
        acc12 = coopMultiplyAdd(at1, bt2, acc12);
        acc13 = coopMultiplyAdd(at1, bt3, acc13);
        workgroupBarrier();
    }

    store_tile(acc00, row0,     col0,          M, N, lr, lc);
    store_tile(acc01, row0,     col0 + T,      M, N, lr, lc);
    store_tile(acc02, row0,     col0 + 2u * T, M, N, lr, lc);
    store_tile(acc03, row0,     col0 + 3u * T, M, N, lr, lc);
    store_tile(acc10, row0 + T, col0,          M, N, lr, lc);
    store_tile(acc11, row0 + T, col0 + T,      M, N, lr, lc);
    store_tile(acc12, row0 + T, col0 + 2u * T, M, N, lr, lc);
    store_tile(acc13, row0 + T, col0 + 3u * T, M, N, lr, lc);
}
