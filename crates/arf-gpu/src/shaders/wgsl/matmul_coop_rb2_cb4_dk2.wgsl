// Cooperative-matrix GEMM for bf16 weights, RB=2 × CB=4 × DK=2 (deeper-K): the
// CB=4 16×32 tile (8 accumulators) with TWO 8-deep K-slices staged per round and
// SIXTEEN coopMultiplyAdd between barriers — halving the barrier count without
// CB=8's register spill. Same math/boundary contract as matmul_coop_rb2; the
// matmul_nt oracle holds. Same grid as CB=4: (ceil(m/16), ceil(n/32)).

enable wgpu_cooperative_matrix;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;  // [m,k] row-major
@group(0) @binding(1) var<storage, read>       b: array<u32>;  // bf16-packed [n,k], 2/word
@group(0) @binding(2) var<storage, read_write> c: array<f32>;  // [m,n] row-major
@group(0) @binding(3) var<uniform>             d: Dims;

const T: u32 = 8u;
var<workgroup> as0a: array<f32, 64>;
var<workgroup> as1a: array<f32, 64>;
var<workgroup> as0b: array<f32, 64>;
var<workgroup> as1b: array<f32, 64>;
var<workgroup> bs0a: array<f32, 64>;
var<workgroup> bs1a: array<f32, 64>;
var<workgroup> bs2a: array<f32, 64>;
var<workgroup> bs3a: array<f32, 64>;
var<workgroup> bs0b: array<f32, 64>;
var<workgroup> bs1b: array<f32, 64>;
var<workgroup> bs2b: array<f32, 64>;
var<workgroup> bs3b: array<f32, 64>;
var<workgroup> cs_t: array<f32, 64>;

fn weight_at(col: u32, kk: u32, k: u32) -> f32 {
    let word = b[col * (k / 2u) + kk / 2u];
    let half = select(word >> 16u, word & 0xffffu, (kk & 1u) == 0u);
    return bitcast<f32>(half << 16u);
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

    let num_rounds = (K + 2u * T - 1u) / (2u * T);
    let r0 = row0 + lr;
    let r1 = row0 + T + lr;
    let c0 = col0 + lc;
    let c1 = col0 + T + lc;
    let c2 = col0 + 2u * T + lc;
    let c3 = col0 + 3u * T + lc;
    for (var t = 0u; t < num_rounds; t = t + 1u) {
        let kbase = t * 2u * T;
        let kaa = kbase + lc;
        let kab = kbase + T + lc;
        as0a[lr * T + lc] = select(0.0, a[r0 * K + kaa], r0 < M && kaa < K);
        as1a[lr * T + lc] = select(0.0, a[r1 * K + kaa], r1 < M && kaa < K);
        as0b[lr * T + lc] = select(0.0, a[r0 * K + kab], r0 < M && kab < K);
        as1b[lr * T + lc] = select(0.0, a[r1 * K + kab], r1 < M && kab < K);
        let kba = kbase + lr;
        let kbb = kbase + T + lr;
        bs0a[lr * T + lc] = select(0.0, weight_at(c0, kba, K), c0 < N && kba < K);
        bs1a[lr * T + lc] = select(0.0, weight_at(c1, kba, K), c1 < N && kba < K);
        bs2a[lr * T + lc] = select(0.0, weight_at(c2, kba, K), c2 < N && kba < K);
        bs3a[lr * T + lc] = select(0.0, weight_at(c3, kba, K), c3 < N && kba < K);
        bs0b[lr * T + lc] = select(0.0, weight_at(c0, kbb, K), c0 < N && kbb < K);
        bs1b[lr * T + lc] = select(0.0, weight_at(c1, kbb, K), c1 < N && kbb < K);
        bs2b[lr * T + lc] = select(0.0, weight_at(c2, kbb, K), c2 < N && kbb < K);
        bs3b[lr * T + lc] = select(0.0, weight_at(c3, kbb, K), c3 < N && kbb < K);
        workgroupBarrier();

        let at0a = coopLoad<coop_mat8x8<f32, A>>(&as0a[0], T);
        let at1a = coopLoad<coop_mat8x8<f32, A>>(&as1a[0], T);
        let b0a = coopLoad<coop_mat8x8<f32, B>>(&bs0a[0], T);
        acc00 = coopMultiplyAdd(at0a, b0a, acc00);
        acc10 = coopMultiplyAdd(at1a, b0a, acc10);
        let b1a = coopLoad<coop_mat8x8<f32, B>>(&bs1a[0], T);
        acc01 = coopMultiplyAdd(at0a, b1a, acc01);
        acc11 = coopMultiplyAdd(at1a, b1a, acc11);
        let b2a = coopLoad<coop_mat8x8<f32, B>>(&bs2a[0], T);
        acc02 = coopMultiplyAdd(at0a, b2a, acc02);
        acc12 = coopMultiplyAdd(at1a, b2a, acc12);
        let b3a = coopLoad<coop_mat8x8<f32, B>>(&bs3a[0], T);
        acc03 = coopMultiplyAdd(at0a, b3a, acc03);
        acc13 = coopMultiplyAdd(at1a, b3a, acc13);

        let at0b = coopLoad<coop_mat8x8<f32, A>>(&as0b[0], T);
        let at1b = coopLoad<coop_mat8x8<f32, A>>(&as1b[0], T);
        let b0b = coopLoad<coop_mat8x8<f32, B>>(&bs0b[0], T);
        acc00 = coopMultiplyAdd(at0b, b0b, acc00);
        acc10 = coopMultiplyAdd(at1b, b0b, acc10);
        let b1b = coopLoad<coop_mat8x8<f32, B>>(&bs1b[0], T);
        acc01 = coopMultiplyAdd(at0b, b1b, acc01);
        acc11 = coopMultiplyAdd(at1b, b1b, acc11);
        let b2b = coopLoad<coop_mat8x8<f32, B>>(&bs2b[0], T);
        acc02 = coopMultiplyAdd(at0b, b2b, acc02);
        acc12 = coopMultiplyAdd(at1b, b2b, acc12);
        let b3b = coopLoad<coop_mat8x8<f32, B>>(&bs3b[0], T);
        acc03 = coopMultiplyAdd(at0b, b3b, acc03);
        acc13 = coopMultiplyAdd(at1b, b3b, acc13);
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
