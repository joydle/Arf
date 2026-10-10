// Cooperative-matrix tiled GEMM for INT8 weights: C[m,n] = scale[n] · (A[m,k]·Qᵀ),
// Q per-row symmetric int8 stored [n,k], scale[n] the per-output-row dequant.
//
// The int8 analog of matmul_coop.wgsl and the coop analog of matmul_vec_q8_batch:
// it runs the dot on the matrix units while keeping the int8 weight bandwidth.
// The raw int8·activation product accumulates in f32; the per-column dequant
// `scale[n]` is applied once at the guarded store (it factors out of the dot, as
// in the scalar int8 path — so matmul_nt_q8 stays the exact oracle). One workgroup
// per 8×8 output tile; coop handles any m (no MAXM chunking). Edges guarded.

enable wgpu_cooperative_matrix;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;     // [m,k] row-major
@group(0) @binding(1) var<storage, read>       q: array<u32>;     // int8 [n,k], 4/word
@group(0) @binding(2) var<storage, read_write> c: array<f32>;     // [m,n] row-major
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scale: array<f32>; // per output col (n)

const T: u32 = 8u;
var<workgroup> as_t: array<f32, 64>;
var<workgroup> bs_t: array<f32, 64>;
var<workgroup> cs_t: array<f32, 64>;

fn i8_at(word: u32, j: u32) -> i32 {
    return (i32(word << (24u - 8u * j)) >> 24u);
}

// Raw int8 weight (col, kk) as f32, UNSCALED (scale applied at store). Row-major
// [n,k] int8 packed 4-per-word: word col*(k/4)+kk/4, byte kk%4.
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
    let row0 = wid.x * T;
    let col0 = wid.y * T;
    let lr = lid.x;
    let lc = lid.y;

    cs_t[lr * T + lc] = 0.0;
    workgroupBarrier();
    var acc = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);

    let num_tiles = (K + T - 1u) / T;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let ka = t * T + lc;
        as_t[lr * T + lc] = select(0.0, a[(row0 + lr) * K + ka], row0 + lr < M && ka < K);
        let kb = t * T + lr;
        bs_t[lr * T + lc] = select(0.0, qweight(col0 + lc, kb, K), col0 + lc < N && kb < K);
        workgroupBarrier();

        let at = coopLoad<coop_mat8x8<f32, A>>(&as_t[0], T);
        let bt = coopLoad<coop_mat8x8<f32, B>>(&bs_t[0], T);
        acc = coopMultiplyAdd(at, bt, acc);
        workgroupBarrier();
    }

    coopStore(acc, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + lr < M && col0 + lc < N) {
        c[(row0 + lr) * N + col0 + lc] = cs_t[lr * T + lc] * scale[col0 + lc];
    }
}
