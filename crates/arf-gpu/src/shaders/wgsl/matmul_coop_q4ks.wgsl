// Cooperative-matrix tiled GEMM for Q4_K_S (super-block-256, 4-bit, two-level u8/i8
// scales) weights: C[m,n] = A[m,k] · dequant(Q4_K_S)ᵀ. The GEMM analog of
// matmul_vec_q4ks — folds the two-level dequant into the staged f32 weight element
// W = (d·scale_u8)·nib + (dmin·min_i8), so Σ a·W matches the scalar
// `s·Σ(nib·a) + lo·Σa` of matmul_vec_q4ks modulo f32 reassociation. One workgroup
// per 8×8 output tile, any m; matmul_nt_q4ks oracle. Mirrors matmul_coop_q4k with
// the q4k-lite (bf16 scale+min) dequant swapped for the q4ks two-level dequant.
//
// Layout (Q4KSMatrix): codes = 4 u32 per 32-weight sub-block (8 unsigned nibbles/
// word); scales = u8 sub-scale packed 4/u32 (one per sub-block); mins = i8 sub-min
// packed 4/u32; dd = f32 pairs [d, dmin] per super-block (256 weights = 8 sub-
// blocks). k % 256 == 0. k-index kk → sub-block b=kk/32, super sblk=b/8, j=kk%32;
// nib = (codes[(col*subs+b)*4 + j/8] >> ((j%8)*4)) & 0xF.

enable wgpu_cooperative_matrix;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;       // [m,k] row-major
@group(0) @binding(1) var<storage, read>       codes: array<u32>;   // Q4_K_S [n,k], 4 u32/sub-block
@group(0) @binding(2) var<storage, read_write> c: array<f32>;       // [m,n] row-major
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;  // u8 sub-scale, 4/word
@group(0) @binding(5) var<storage, read>       mins: array<u32>;    // i8 sub-min, 4/word
@group(0) @binding(6) var<storage, read>       dd: array<f32>;      // [d,dmin] per super-block

const T: u32 = 8u;
// K-UNROLL (L163). The hardware fragment is fixed at 8x8, but the K LOOP does not have to
// advance 8 at a time. Staging KU=4 fragments per pass cuts the barrier count 4x while doing
// 4x the arithmetic between barriers — the same 8x8 mma primitive, just fed properly.
//
// WHY IT MATTERS: down_proj on muse-glimmer is K=19968, so at KU=1 the loop ran 2496 times
// with TWO workgroupBarriers each = ~5k barriers per workgroup, times 53,248 workgroups at
// m=512. That is ~265 MILLION barriers for ONE projection, with 64 threads of work between
// them. Arithmetic intensity was 512 FLOP per 128 staged elements = 4 FLOP/element — a GEMM
// doing memory-bound work. Measured effect of the whole prefill path: 0.26 TFLOP/s, 1.5% of
// the M4 Max matrix units, vs llama.cpp at 5.03 TFLOP/s.
const KU: u32 = 4u;            // K fragments staged per barrier pair
const KC: u32 = T * KU;        // = 32 contraction elements per pass
// N-WIDENING (L164). Each workgroup now owns NT=4 column fragments (32 output cols) instead
// of one. The staged A tile is IDENTICAL for all of them, so widening N is pure weight reuse:
// the same activation staging feeds 4x the output. It also cuts the grid 4x — down_proj at
// m=512,n=6656 went from 53,248 workgroups to 13,312 — which is where the barrier count
// actually lives (barriers are PER WORKGROUP).
//
// This mirrors matmul_mm_q4ks.metal, the island GEMM that is already fast: BM=32, BN=64,
// KC=32, 4 simdgroups — a 32x larger tile than this kernel had. That kernel is wired into the
// batched megakernel but not into the wgpu prefill path, so prefill was left on the narrow one.
const NT: u32 = 4u;            // column fragments per workgroup (NT * T = 32 output cols)
var<workgroup> as_t: array<f32, 256>;        // T * KC
var<workgroup> bs_t: array<f32, 1024>;       // KC * (T * NT)
var<workgroup> cs_t: array<f32, 256>;        // T * (T * NT)

// Extract byte `b` (0..4) of a packed word as an UNSIGNED value (0..255).
fn u8_at(word: u32, b: u32) -> f32 {
    return f32((word >> (8u * b)) & 0xFFu);
}
// Extract byte `b` as a SIGNED i8 value (-128..127) via arithmetic shift.
fn i8_at(word: u32, b: u32) -> f32 {
    return f32(i32(word << (24u - 8u * b)) >> 24u);
}

// Dequantized Q4_K_S weight (col, kk) as f32: (d·scale_u8)·nibble + (dmin·min_i8).
fn qweight(col: u32, kk: u32, k: u32) -> f32 {
    let subs = k / 32u;          // sub-blocks per row
    let b = kk / 32u;            // sub-block index in this row
    let j = kk % 32u;            // weight within sub-block
    let sblk = b / 8u;           // super-block index (8 sub-blocks/super)
    let word = codes[(col * subs + b) * 4u + j / 8u];
    let nib = f32((word >> ((j % 8u) * 4u)) & 0xFu);
    let sidx = col * subs + b;                  // sub-block units (u8/i8 are 4/word)
    let dd_base = col * (subs / 8u) * 2u + sblk * 2u;
    let dv = dd[dd_base];                        // super-block scale-of-scales
    let dmv = dd[dd_base + 1u];                  // super-block scale-of-mins
    let su8 = u8_at(scales[sidx / 4u], sidx % 4u);
    let mi8 = i8_at(mins[sidx / 4u], sidx % 4u);
    return (dv * su8) * nib + (dmv * mi8);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = d.m;
    let K = d.k;
    let N = d.n;
    let row0 = wid.x * T;
    let col0 = wid.y * (T * NT);      // this workgroup owns NT column fragments
    let lr = lid.x;
    let lc = lid.y;

    // NT accumulators, one per column fragment. All share the SAME staged A tile.
    for (var v = 0u; v < NT; v = v + 1u) { cs_t[v * 64u + lr * T + lc] = 0.0; }
    workgroupBarrier();
    var acc0 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc1 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[64], T);
    var acc2 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[128], T);
    var acc3 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[192], T);

    let num_passes = (K + KC - 1u) / KC;
    for (var t = 0u; t < num_passes; t = t + 1u) {
        let kbase = t * KC;
        // Stage A once: [T rows][KC]. Stage B for ALL NT column fragments: [KC][T*NT].
        for (var u = 0u; u < KU; u = u + 1u) {
            let ka = kbase + u * T + lc;
            as_t[lr * KC + u * T + lc] =
                select(0.0, a[(row0 + lr) * K + ka], row0 + lr < M && ka < K);
            let kb = kbase + u * T + lr;
            for (var v = 0u; v < NT; v = v + 1u) {
                let cj = col0 + v * T + lc;
                bs_t[(u * T + lr) * (T * NT) + v * T + lc] =
                    select(0.0, qweight(cj, kb, K), cj < N && kb < K);
            }
        }
        workgroupBarrier();

        // KU * NT mmas under ONE barrier pair, all reusing the same A fragments.
        for (var u = 0u; u < KU; u = u + 1u) {
            let at = coopLoad<coop_mat8x8<f32, A>>(&as_t[u * T], KC);
            let bo = u * T * (T * NT);
            acc0 = coopMultiplyAdd(at, coopLoad<coop_mat8x8<f32, B>>(&bs_t[bo], T * NT), acc0);
            acc1 = coopMultiplyAdd(at, coopLoad<coop_mat8x8<f32, B>>(&bs_t[bo + T], T * NT), acc1);
            acc2 = coopMultiplyAdd(at, coopLoad<coop_mat8x8<f32, B>>(&bs_t[bo + 2u * T], T * NT), acc2);
            acc3 = coopMultiplyAdd(at, coopLoad<coop_mat8x8<f32, B>>(&bs_t[bo + 3u * T], T * NT), acc3);
        }
        workgroupBarrier();
    }

    coopStore(acc0, &cs_t[0], T);
    coopStore(acc1, &cs_t[64], T);
    coopStore(acc2, &cs_t[128], T);
    coopStore(acc3, &cs_t[192], T);
    workgroupBarrier();
    if (row0 + lr < M) {
        for (var v = 0u; v < NT; v = v + 1u) {
            let cj = col0 + v * T + lc;
            if (cj < N) { c[(row0 + lr) * N + cj] = cs_t[v * 64u + lr * T + lc]; }
        }
    }
}
