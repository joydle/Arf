// Cooperative-matrix GEMM, M register-blocked RB=2: C[m,n] = A[m,k] · Bᵀ (B stored
// [n,k], bf16-packed). Same math + boundary contract as matmul_coop.wgsl, but one
// workgroup computes TWO stacked 8×8 output tiles (16 rows × 8 cols) and stages the
// shared 8×8 B (weight) tile ONCE per K-step, reusing it for both row accumulators.
//
// Register-blocking along M is the throughput lever at batched-decode m, where the
// cost is streaming the weight: each weight tile is now read once per 16 output rows
// instead of once per 8, ~halving weight bandwidth. Measured +10–20% over RB=1 at
// m ≥ 16 (N16 129→143, N32 130→149 tok/s, Llama-1B). The caller dispatches this only
// for m ≥ 16 (ceil(m/16) row-groups); m ∈ [8,16) keeps RB=1 (no wasted second tile).

enable wgpu_cooperative_matrix;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;  // [m,k] row-major
@group(0) @binding(1) var<storage, read>       b: array<u32>;  // bf16-packed [n,k], 2/word
@group(0) @binding(2) var<storage, read_write> c: array<f32>;  // [m,n] row-major
@group(0) @binding(3) var<uniform>             d: Dims;

const T: u32 = 8u;
var<workgroup> as0: array<f32, 64>; // A tile, row-block 0
var<workgroup> as1: array<f32, 64>; // A tile, row-block 1
var<workgroup> bs_t: array<f32, 64>; // shared logical-B tile [kk,nn] (transposed)
var<workgroup> cs_t: array<f32, 64>; // C staging (zero init + guarded store), reused

fn weight_at(col: u32, kk: u32, k: u32) -> f32 {
    let word = b[col * (k / 2u) + kk / 2u];
    let half = select(word >> 16u, word & 0xffffu, (kk & 1u) == 0u);
    return bitcast<f32>(half << 16u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = d.m;
    let K = d.k;
    let N = d.n;
    let row0 = wid.x * (T * 2u); // 16 rows tall
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
        bs_t[lr * T + lc] = select(0.0, weight_at(col0 + lc, kb, K), col0 + lc < N && kb < K);
        workgroupBarrier();

        let bt = coopLoad<coop_mat8x8<f32, B>>(&bs_t[0], T); // weight read ONCE for 2 rows
        let at0 = coopLoad<coop_mat8x8<f32, A>>(&as0[0], T);
        let at1 = coopLoad<coop_mat8x8<f32, A>>(&as1[0], T);
        acc0 = coopMultiplyAdd(at0, bt, acc0);
        acc1 = coopMultiplyAdd(at1, bt, acc1);
        workgroupBarrier();
    }

    coopStore(acc0, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + lr < M && col0 + lc < N) {
        c[(row0 + lr) * N + col0 + lc] = cs_t[lr * T + lc];
    }
    workgroupBarrier();
    coopStore(acc1, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + T + lr < M && col0 + lc < N) {
        c[(row0 + T + lr) * N + col0 + lc] = cs_t[lr * T + lc];
    }
}
