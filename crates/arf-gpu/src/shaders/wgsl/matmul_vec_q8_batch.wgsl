// Batched int8 matrix·vector for decode: C[m,n] = scale[n] · (A[m,k] · Qᵀ), where
// Q is per-row symmetric int8 ([n,k]) and scale[n] the per-row dequant factor.
//
// The int8 analog of matmul_vec_batch.wgsl, and the batched analog of
// matmul_vec_q8.wgsl — it combines the two wins: int8 weights (a quarter of f32
// bytes) AND reading each weight row ONCE for all m activation rows (the
// continuous-batching amortization). Decode is bandwidth-bound on the weight
// read, so this is the narrowest batched decode kernel.
//
// Per-row arithmetic is bit-identical to the m=1 int8 GEMV (same i8_at unpack,
// same 4× dot_word term order, same scale[col] applied after the lane reduction),
// so the CPU matmul_nt_q8 path stays the exact oracle. Output is row-major [m,n]:
// c[r*n + col]. Caller MUST gate d.m <= MAXM (it shares MATMUL_VEC_BATCH_MAXM with
// the bf16 batched GEMV); the in-shader min() is defense in depth.
//
// Weights: 16 int8 per vec4<u32>, k/16 vec4 words per row (k % 16 == 0).

// `m` is the rows THIS dispatch handles (<= MAXM); `row_off` is the first batch
// row (so m > MAXM is covered by chunked dispatches advancing row_off by MAXM).
struct Dims { m: u32, k: u32, n: u32, row_off: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;  // [m,k], 4 f32/vec
@group(0) @binding(1) var<storage, read>       q: array<vec4<u32>>;  // int8: 16 weights/vec
@group(0) @binding(2) var<storage, read_write> c: array<f32>;        // [m,n]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scale: array<f32>;    // per output row (column of C)

const WG: u32 = 64u;   // lanes per column (same as the m==1 int8 GEMV)
const MAXM: u32 = 16u; // compile-time accumulator bound; MUST equal MATMUL_VEC_BATCH_MAXM

// partial[lane*MAXM + r] — one slot per (lane, batch-row). 64*16 f32 = 4 KiB.
var<workgroup> partial: array<f32, 1024>;

// Sign-extend the j-th byte (j in 0..4) of `word` to an i32 via arithmetic shift.
fn i8_at(word: u32, j: u32) -> i32 {
    return (i32(word << (24u - 8u * j)) >> 24u);
}

// Unpack a packed word (4 int8) to a vec4<f32> — done ONCE per weight chunk and
// reused across all m activation rows (the CSE that makes batching pay off; the
// bf16 batch kernel hoists its unpack the same way).
fn unpack4(word: u32) -> vec4<f32> {
    return vec4<f32>(f32(i8_at(word, 0u)), f32(i8_at(word, 1u)),
                     f32(i8_at(word, 2u)), f32(i8_at(word, 3u)));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let m = min(d.m, MAXM);    // clamp: caller MUST also gate d.m <= MAXM
    let vecs = d.k / 16u;      // vec4<u32> words per weight row (k % 16 == 0)
    let arow = d.k / 4u;       // vec4<f32> per activation row

    var col = wid.x;           // grid-stride over output columns (n may exceed cap)
    while (col < d.n) {
        let row_base = col * vecs;

        var acc: array<f32, 16>;
        for (var r = 0u; r < m; r = r + 1u) {
            acc[r] = 0.0;
        }

        // Stream the weight row ONCE; reuse each loaded chunk for all m rows.
        // Activation row index is row_off + r (chunked dispatch for m > MAXM).
        var w = lane;
        while (w < vecs) {
            let qv = q[row_base + w];   // 16 int8 weights, one coalesced load
            // Unpack the 16 weights ONCE (hoisted out of the r-loop), then reuse
            // them for every activation row — without this the sign-extend ran m
            // times and the batch barely amortized.
            let f0 = unpack4(qv.x);
            let f1 = unpack4(qv.y);
            let f2 = unpack4(qv.z);
            let f3 = unpack4(qv.w);
            for (var r = 0u; r < m; r = r + 1u) {
                let abase = (d.row_off + r) * arow + 4u * w;
                acc[r] = acc[r]
                    + dot(a[abase], f0)
                    + dot(a[abase + 1u], f1)
                    + dot(a[abase + 2u], f2)
                    + dot(a[abase + 3u], f3);
            }
            w = w + WG;
        }

        for (var r = 0u; r < m; r = r + 1u) {
            partial[lane * MAXM + r] = acc[r];
        }
        workgroupBarrier();

        var stride = WG / 2u;
        while (stride > 0u) {
            if (lane < stride) {
                let src = (lane + stride) * MAXM;
                let dst = lane * MAXM;
                for (var r = 0u; r < m; r = r + 1u) {
                    partial[dst + r] = partial[dst + r] + partial[src + r];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        // Lane 0 writes column `col` for every batch row, scaled by the row's
        // (= column-of-C's) dequant factor.
        if (lane == 0u) {
            let s = scale[col];
            for (var r = 0u; r < m; r = r + 1u) {
                c[(d.row_off + r) * d.n + col] = partial[r] * s;
            }
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
