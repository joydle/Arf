// Cooperative-matrix tiled GEMM: C[m,n] = A[m,k] · Bᵀ  (B stored [n,k], bf16-packed).
//
// Same contract as matmul.wgsl (the scalar tiled GEMM it replaces when the device
// has cooperative matrix), so it is a drop-in for the batched/prefill bf16 path
// (m ≥ 8). It runs the inner product on the hardware matrix units (Apple
// `simdgroup_matrix` / tensor cores), measured ~3.5× the scalar tiled GEMM.
//
// Tiling: one workgroup computes one 8×8 output tile. Each K-step stages an 8×8
// tile of A (row-major) and of the LOGICAL B (= Bᵀ of the stored weight) into
// workgroup memory — the weight is transposed in the scalar copy and expanded
// bf16→f32 there — then `coopLoad`s both from workgroup memory and accumulates
// with `coopMultiplyAdd`. Edges (m,n,k not multiples of 8) are zero-padded in the
// staging copy and the store is guarded, so this matches the scalar kernel's
// boundary behaviour. The accumulator is initialised from a zeroed workgroup tile
// (the result is computed fresh, like matmul.wgsl — it does not read prior C).

enable wgpu_cooperative_matrix;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;  // [m,k] row-major
@group(0) @binding(1) var<storage, read>       b: array<u32>;  // bf16-packed [n,k], 2/word
@group(0) @binding(2) var<storage, read_write> c: array<f32>;  // [m,n] row-major
@group(0) @binding(3) var<uniform>             d: Dims;

const T: u32 = 8u;
var<workgroup> as_t: array<f32, 64>; // 8×8 A tile  [mm,kk]
var<workgroup> bs_t: array<f32, 64>; // 8×8 logical-B tile [kk,nn] (transposed from Bᵀ)
var<workgroup> cs_t: array<f32, 64>; // 8×8 C tile staging (zero init + guarded store)

// bf16 -> f32 (bit-exact: bf16 is the high 16 bits of f32). Weight (col,kk) of the
// row-major [n,k] matrix is packed two-per-word, low half for even kk.
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
    let row0 = wid.x * T; // output row base (m)
    let col0 = wid.y * T; // output col base (n)
    let lr = lid.x;       // 0..7
    let lc = lid.y;       // 0..7

    // Zero accumulator via a zeroed workgroup tile.
    cs_t[lr * T + lc] = 0.0;
    workgroupBarrier();
    var acc = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);

    let num_tiles = (K + T - 1u) / T;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let ka = t * T + lc;
        as_t[lr * T + lc] = select(0.0, a[(row0 + lr) * K + ka], row0 + lr < M && ka < K);
        let kb = t * T + lr;
        bs_t[lr * T + lc] = select(0.0, weight_at(col0 + lc, kb, K), col0 + lc < N && kb < K);
        workgroupBarrier();

        let at = coopLoad<coop_mat8x8<f32, A>>(&as_t[0], T);
        let bt = coopLoad<coop_mat8x8<f32, B>>(&bs_t[0], T);
        acc = coopMultiplyAdd(at, bt, acc);
        workgroupBarrier();
    }

    // Store to a workgroup tile, then guarded scalar copy to C (rows past M / cols
    // past N are skipped so we never write outside the [m,n] buffer).
    coopStore(acc, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + lr < M && col0 + lc < N) {
        c[(row0 + lr) * N + col0 + lc] = cs_t[lr * T + lc];
    }
}
