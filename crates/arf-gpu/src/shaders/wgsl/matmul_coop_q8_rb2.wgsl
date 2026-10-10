// Cooperative-matrix GEMM for INT8 weights, M register-blocked RB=2: one workgroup
// computes two stacked 8×8 output tiles and stages the shared int8 weight tile ONCE
// per K-step, reusing it for both row accumulators. The RB=2 analog of
// matmul_coop_q8 — the per-column dequant scale[n] is applied at each guarded store
// (factors out of the dot). Same math/boundary contract; matmul_nt_q8 oracle. Caller
// dispatches for m ≥ 16 (ceil(m/16) row-groups).

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
var<workgroup> bs_t: array<f32, 64>;
var<workgroup> cs_t: array<f32, 64>;

fn i8_at(word: u32, j: u32) -> i32 {
    return (i32(word << (24u - 8u * j)) >> 24u);
}

fn qweight(col: u32, kk: u32, k: u32) -> f32 {
    let word = q[col * (k / 4u) + kk / 4u];
    return f32(i8_at(word, kk % 4u));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = d.m;
    let K = d.k;
    let N = d.n;
    let row0 = wid.x * (T * 2u);
    let col0 = wid.y * T;
    let lr = lid.x;
    let lc = lid.y;

    cs_t[lr * T + lc] = 0.0;
    workgroupBarrier();
    var acc0 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc1 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);

    let num_tiles = (K + T - 1u) / T;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let ka = t * T + lc;
        let r0 = row0 + lr;
        let r1 = row0 + T + lr;
        as0[lr * T + lc] = select(0.0, a[r0 * K + ka], r0 < M && ka < K);
        as1[lr * T + lc] = select(0.0, a[r1 * K + ka], r1 < M && ka < K);
        let kb = t * T + lr;
        bs_t[lr * T + lc] = select(0.0, qweight(col0 + lc, kb, K), col0 + lc < N && kb < K);
        workgroupBarrier();

        let bt = coopLoad<coop_mat8x8<f32, B>>(&bs_t[0], T);
        let at0 = coopLoad<coop_mat8x8<f32, A>>(&as0[0], T);
        let at1 = coopLoad<coop_mat8x8<f32, A>>(&as1[0], T);
        acc0 = coopMultiplyAdd(at0, bt, acc0);
        acc1 = coopMultiplyAdd(at1, bt, acc1);
        workgroupBarrier();
    }

    let sc = scale[col0 + lc];
    coopStore(acc0, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + lr < M && col0 + lc < N) {
        c[(row0 + lr) * N + col0 + lc] = cs_t[lr * T + lc] * sc;
    }
    workgroupBarrier();
    coopStore(acc1, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + T + lr < M && col0 + lc < N) {
        c[(row0 + T + lr) * N + col0 + lc] = cs_t[lr * T + lc] * sc;
    }
}
