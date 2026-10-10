// Cooperative-matrix tiled GEMM for native **Q8_0** (32-weight blocks, one f16 scale +
// 32 i8 codes) weights: C[m,n] = A[m,k] · dequant(Q8_0)ᵀ. The Q8_0 analog of
// matmul_coop_q4ks — folds the per-block dequant `W = d_blk · q` into the staged f32
// weight element, so Σ a·W matches the scalar `Σ_blk d·Σ(q·a)` of matmul_nt_q8_0 modulo
// f32 reassociation. One workgroup per 8×8 output tile, any m; matmul_nt_q8_0 oracle.
//
// Layout (split from the raw 34-byte ggml block on upload, for aligned bindings):
//   q  = i8 codes packed 4/u32, row-major [n, k] (q[(col*k + kk)/4] byte kk%4).
//   d  = one f32 scale per 32-weight block, row-major [n, k/32] (d[col*(k/32) + kk/32]).
// k % 32 == 0.

enable wgpu_cooperative_matrix;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;     // [m,k] row-major
@group(0) @binding(1) var<storage, read>       q: array<u32>;     // i8 codes, 4/word [n,k]
@group(0) @binding(2) var<storage, read_write> c: array<f32>;     // [m,n] row-major
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       dscale: array<f32>; // one scale per 32-block

const T: u32 = 8u;
var<workgroup> as_t: array<f32, 64>;
var<workgroup> bs_t: array<f32, 64>;
var<workgroup> cs_t: array<f32, 64>;

// Extract byte `b` (0..4) of a packed word as SIGNED i8 (-128..127).
fn i8_at(word: u32, b: u32) -> f32 {
    return f32(i32(word << (24u - 8u * b)) >> 24u);
}

// Dequantized Q8_0 weight (col, kk) as f32: d_block · code.
fn qweight(col: u32, kk: u32, k: u32) -> f32 {
    let lin = col * k + kk;            // element index into the [n,k] code stream
    let code = i8_at(q[lin / 4u], lin % 4u);
    let dblk = dscale[col * (k / 32u) + kk / 32u];
    return dblk * code;
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
        c[(row0 + lr) * N + col0 + lc] = cs_t[lr * T + lc];
    }
}
