// Cooperative-matrix tiled GEMM for Q4_K-lite (32-weight sub-blocks, 4-bit, per-
// sub-block asymmetric scale+min) weights: C[m,n] = A[m,k] · dequant(Q4_K)ᵀ. The
// Q4_K analog of matmul_coop_q4 — unpacks each unsigned nibble and folds the per-
// sub-block (scale, min) into the staged f32 weight element (W = scale·nib + min),
// so Σ a·W matches the scalar `s·Σ(nib·a) + min·Σa` of matmul_vec_q4k modulo f32
// reassociation. One workgroup per 8×8 output tile, any m; matmul_nt_q4k oracle.
//
// Layout (Q4KMatrix-lite): codes = 4 u32 per 32-weight sub-block (8 nibbles/word,
// unsigned); scales/mins = one bf16 (low 16 bits) per sub-block. k-index kk →
// sub-block b=kk/32, j=kk%32; nib = (codes[(col*subs+b)*4 + j/8] >> ((j%8)*4)) & 0xF.

enable wgpu_cooperative_matrix;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;       // [m,k] row-major
@group(0) @binding(1) var<storage, read>       codes: array<u32>;   // Q4_K [n,k], 4 u32/sub-block
@group(0) @binding(2) var<storage, read_write> c: array<f32>;       // [m,n] row-major
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;  // one bf16/word per sub-block
@group(0) @binding(5) var<storage, read>       mins: array<u32>;    // one bf16/word per sub-block

const T: u32 = 8u;
var<workgroup> as_t: array<f32, 64>;
var<workgroup> bs_t: array<f32, 64>;
var<workgroup> cs_t: array<f32, 64>;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>((bits & 0xFFFFu) << 16u);
}

// Dequantized Q4_K weight (col, kk) as f32: scale·nibble + min (unsigned nibble).
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
