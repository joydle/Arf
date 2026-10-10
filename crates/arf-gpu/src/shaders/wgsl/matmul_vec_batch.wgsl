// Batched matrix·vector for decode: C[m, n] = A[m, k] · Bᵀ, B stored [n, k]
// (Linearᵀ, bf16-packed). For SMALL m (2..32, the continuous-decode batch).
//
// The win over the tiled 16×16 GEMM (matmul.wgsl): each weight row B[col,:] is
// streamed from global memory ONCE and dotted against ALL m activation rows.
// Decode is bandwidth-bound on the weight read; amortizing that read across the
// batch is the point of continuous batching. The tiled GEMM re-streams weights
// per 16-col tile and wastes lanes for m < 16 (profiled: ~98% of batched-decode
// GPU time). This kernel makes the amortization in-kernel/contractual rather
// than relying on L2 to hold a weight row across separate per-row workgroups.
//
// Layout MUST match matmul_vec.wgsl's m==1 path and forward_batch:
//   a : row-major [m, k]  -> row r at a[r*k ..], read as vec4<f32> (k % 8 == 0)
//   c : row-major [m, n]  -> c[r*n + col]   (NOT c[col*m + r])
// so a single dispatch is bit-exact identical to running the m==1 GEMV m times.
// The per-row accumulation order is IDENTICAL to matmul_vec.wgsl: same lane-
// striding over the same vec4<u32> chunks, same 4× dot term order, same
// bf16(bits<<16) unpack, same lane-stride tree reduction. The only change is
// hoisting the 8 weight unpacks out of the r-loop (pure CSE; alters no row's
// arithmetic). The CPU matmul_nt (bf16) remains the oracle.
//
// Caller MUST gate d.m <= MAXM and fall back to the tiled GEMM for larger m
// (prefill passes m = full prompt length); the in-shader min() is defense in
// depth so a stale m>MAXM clamps instead of writing other lanes' partials.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;  // [m,k], 4 f32/vec
@group(0) @binding(1) var<storage, read>       b: array<vec4<u32>>;  // bf16: 8 weights/vec
@group(0) @binding(2) var<storage, read_write> c: array<f32>;        // [m,n]
@group(0) @binding(3) var<uniform>             d: Dims;

const WG: u32 = 64u; // lanes per column (same as the m==1 GEMV)
// Compile-time accumulator / shared-array bound. The per-lane private `acc`
// array is always this size in registers regardless of runtime m, so it sets the
// register footprint: MAXM=32 spilled and collapsed throughput past m~20
// (measured), MAXM=16 stays in registers. Batches above MAXM fall through to the
// tiled GEMM in add_matmul_m. Decision: MAXM=16 over 32 — the spill cliff at
// 32 cost more than the extra coverage was worth; 16 covers the useful batch range.
const MAXM: u32 = 16u;

// partial[lane*MAXM + r] — one slot per (lane, batch-row). 64*16 f32 = 4 KiB.
var<workgroup> partial: array<f32, 1024>;

fn bf16(bits: u32) -> f32 { return bitcast<f32>(bits << 16u); }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let m = min(d.m, MAXM); // clamp: caller MUST also gate d.m <= MAXM
    let vecs = d.k / 8u;    // vec4<u32> words per weight row (k % 8 == 0)
    let arow = d.k / 4u;    // vec4<f32> per activation row

    var col = wid.x; // grid-stride over columns (n may exceed the dispatch cap)
    while (col < d.n) {
        let row_base = col * vecs;

        // m partial dot-products in registers, one per batch row. Const-sized
        // private array (WGSL requires a const dimension == MAXM); live count `m`.
        var acc: array<f32, 16>;
        for (var r = 0u; r < m; r = r + 1u) {
            acc[r] = 0.0;
        }

        // Stream the weight row ONCE; reuse each loaded chunk for all m rows.
        var w = lane;
        while (w < vecs) {
            let bv = b[row_base + w]; // 8 bf16 weights, one coalesced load
            // Unpack the 8 weights once (hoisted out of the r-loop).
            let w0 = bf16(bv.x & 0xffffu);
            let w1 = bf16(bv.x >> 16u);
            let w2 = bf16(bv.y & 0xffffu);
            let w3 = bf16(bv.y >> 16u);
            let w4 = bf16(bv.z & 0xffffu);
            let w5 = bf16(bv.z >> 16u);
            let w6 = bf16(bv.w & 0xffffu);
            let w7 = bf16(bv.w >> 16u);

            for (var r = 0u; r < m; r = r + 1u) {
                let abase = r * arow + 2u * w; // row r's two vec4<f32> for this chunk
                let a0 = a[abase];
                let a1 = a[abase + 1u];
                // Accumulate in the SAME associative grouping as the m==1 oracle
                // (matmul_vec's `dot_word` pairs): (lo·w + hi·w) per word, then sum
                // the four words. A flat left-to-right 8-term sum is a different f32
                // reassociation — tiny per-chunk, but it compounds over K/8 chunks ×
                // layers and flips argmax once K>256 (softcap shrinks the logit
                // margin). Keeping the oracle's grouping makes batched decode bit-match
                // the single-stream path (the batched-Gemma-4 divergence root cause).
                acc[r] = acc[r]
                    + (a0.x * w0 + a0.y * w1)
                    + (a0.z * w2 + a0.w * w3)
                    + (a1.x * w4 + a1.y * w5)
                    + (a1.z * w6 + a1.w * w7);
            }
            w = w + WG;
        }

        // Stage all m partials, then one m-way reduction over the 64 lanes.
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

        // Lane 0 writes column `col` for every batch row, row-major [m, n].
        if (lane == 0u) {
            for (var r = 0u; r < m; r = r + 1u) {
                c[r * d.n + col] = partial[r]; // partial[0*MAXM + r]
            }
        }
        workgroupBarrier(); // protect `partial` before the next col reuses it
        col = col + ngroups.x;
    }
}
