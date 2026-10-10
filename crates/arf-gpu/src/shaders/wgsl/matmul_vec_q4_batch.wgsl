// Batched Q4_0 (block-32, 4-bit) matrix·vector: C[m,n] = Σ_blocks scale[n,b] ·
// Σ_{w in block} (codes · A[r]), for all m activation rows. The batched analog of
// matmul_vec_q4.wgsl and the Q4 analog of matmul_vec_q8_batch.wgsl — it combines
// the two wins: Q4 weights (half the int8 bytes) AND reading + UNPACKING each
// weight block ONCE for all m rows (the continuous-batching amortization). The
// nibble unpack is the costly ALU (P-Q4b), so hoisting it out of the r-loop is
// what makes batching pay; without it the unpack would run m times.
//
// This unblocks (a) Q4 batched throughput and (b) the spec-decode verify, which
// needs to stream the weights ONCE for a small k+1-row window (the sequential
// per-row verify cost ~k+1 full forwards; this costs ~one).
//
// Per-row arithmetic is bit-identical to the m=1 Q4 GEMV (same vec4 block load,
// same unpack_lo/hi, same per-block bf16 scale folded into the block sub-dot, same
// term order), so the CPU matmul_nt_q4 path stays the exact oracle. Output is
// row-major [m,n]: c[r*n + col]. Caller MUST gate d.m <= MAXM.
//
// Layout (see Q4Matrix): codes = one vec4<u32> per 32-weight block (8 nibbles/word,
// 4 words/block); scales = one bf16 per block (low 16 bits of a u32 word). k%32==0.

// `m` rows THIS dispatch handles (<= MAXM); `row_off` is the first batch row.
struct Dims { m: u32, k: u32, n: u32, row_off: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // [m,k], 4 f32/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // 32 nibbles/vec4 = one block
@group(0) @binding(2) var<storage, read_write> c: array<f32>;           // [m,n]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // one bf16/word per block

const WG: u32 = 64u;    // lanes per column (same as the m==1 Q4 GEMV)
const MAXM: u32 = 16u;  // accumulator bound; MUST equal MATMUL_VEC_BATCH_MAXM

// partial[lane*MAXM + r] — one slot per (lane, batch-row). 64*16 f32 = 4 KiB.
var<workgroup> partial: array<f32, 1024>;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>((bits & 0xFFFFu) << 16u);
}
fn unpack_lo(word: u32) -> vec4<f32> {
    let n = vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(0xFu);
    let sign = select(vec4<f32>(0.0), vec4<f32>(16.0), n >= vec4<u32>(8u));
    return vec4<f32>(n) - sign;
}
fn unpack_hi(word: u32) -> vec4<f32> {
    let n = vec4<u32>(word >> 16u, word >> 20u, word >> 24u, word >> 28u) & vec4<u32>(0xFu);
    let sign = select(vec4<f32>(0.0), vec4<f32>(16.0), n >= vec4<u32>(8u));
    return vec4<f32>(n) - sign;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let m = min(d.m, MAXM);    // clamp: caller MUST also gate d.m <= MAXM
    let blocks = d.k / 32u;    // blocks per weight row (one vec4 code-load each)
    let arow = d.k / 4u;       // vec4<f32> per activation row

    var col = wid.x;           // grid-stride over output columns (n may exceed cap)
    while (col < d.n) {
        let code_base = col * blocks;
        let scale_base = col * blocks;

        var acc: array<f32, 16>;
        for (var r = 0u; r < m; r = r + 1u) {
            acc[r] = 0.0;
        }

        // Stream + UNPACK each weight block ONCE; reuse the 8 unpacked vec4s for
        // every activation row. The per-block bf16 scale folds into each row's
        // block sub-dot (Q4 scale is per-block, unlike int8's per-row).
        var b = lane;
        while (b < blocks) {
            let qv = codes[code_base + b];
            let s = bf16_to_f32(scales[scale_base + b]);
            // 8 unpacks, hoisted out of the r-loop (the batch CSE that pays off).
            let f0 = unpack_lo(qv.x); let f1 = unpack_hi(qv.x);
            let f2 = unpack_lo(qv.y); let f3 = unpack_hi(qv.y);
            let f4 = unpack_lo(qv.z); let f5 = unpack_hi(qv.z);
            let f6 = unpack_lo(qv.w); let f7 = unpack_hi(qv.w);
            for (var r = 0u; r < m; r = r + 1u) {
                let abase = (d.row_off + r) * arow + 8u * b; // 32 acts = 8 vec4
                let blk = dot(a[abase],      f0) + dot(a[abase + 1u], f1)
                        + dot(a[abase + 2u], f2) + dot(a[abase + 3u], f3)
                        + dot(a[abase + 4u], f4) + dot(a[abase + 5u], f5)
                        + dot(a[abase + 6u], f6) + dot(a[abase + 7u], f7);
                acc[r] = acc[r] + s * blk;
            }
            b = b + WG;
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

        // Lane 0 writes column `col` for every batch row (scale already applied
        // per block during accumulation — no final per-row scale, unlike int8).
        if (lane == 0u) {
            for (var r = 0u; r < m; r = r + 1u) {
                c[(d.row_off + r) * d.n + col] = partial[r];
            }
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
