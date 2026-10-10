// int8 matrix·vector for decode: c[n] = scale[n] · (A[1,k] · Qᵀ), where Q is
// per-row symmetric int8 ([n, k]) and scale[n] is the per-row dequant factor.
//
// The int8 companion to matmul_vec.wgsl. Decode is bandwidth-bound on the weight
// read; int8 is a quarter of f32 and half of bf16, so this is the narrowest
// resident-weight decode kernel. Dequant is exactly the CPU matmul_nt_q8 path
// (w ≈ q · scale[row]), so CPU int8 generation is the parity oracle.
//
// Weights are packed 4 int8 per u32 word, k/4 words per row (k % 4 == 0 for every
// Llama weight). Each int8 is sign-extended with an arithmetic shift.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;   // activation, 4 f32 per vec
@group(0) @binding(1) var<storage, read>       q: array<vec4<u32>>;   // int8 packed: 16 weights per vec4
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scale: array<f32>;     // per output row

// 64 lanes per column, not 256: at k=2048 a row is only 128 vec4 words, so 256
// lanes left half idle yet still paid the full reduction. 64 lanes keep every
// lane busy (grid-striding the words) with a shorter reduction; we launch one
// workgroup per column (thousands), so total occupancy stays high.
const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

// Sign-extend the j-th byte (j in 0..4) of `word` to an i32 via arithmetic shift.
fn i8_at(word: u32, j: u32) -> i32 {
    return (i32(word << (24u - 8u * j)) >> 24u);
}

// Dot one packed word (4 int8 weights) with 4 activations.
fn dot_word(word: u32, av: vec4<f32>) -> f32 {
    return av.x * f32(i8_at(word, 0u))
         + av.y * f32(i8_at(word, 1u))
         + av.z * f32(i8_at(word, 2u))
         + av.w * f32(i8_at(word, 3u));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    // One vec4<u32> = 4 words = 16 int8 weights, paired with 4 vec4<f32> of `a`.
    let vecs = d.k / 16u;            // vec4 words per weight row (k % 16 == 0)
    var col = wid.x;                 // grid-stride over output columns (n > cap)
    while (col < d.n) {
        let row_base = col * vecs;
        var acc = 0.0;
        var w = lane;
        while (w < vecs) {
            let qv = q[row_base + w];   // 16 int8 weights in one coalesced load
            let abase = 4u * w;         // 4 vec4<f32> of activations
            acc = acc + dot_word(qv.x, a[abase])
                      + dot_word(qv.y, a[abase + 1u])
                      + dot_word(qv.z, a[abase + 2u])
                      + dot_word(qv.w, a[abase + 3u]);
            w = w + WG;
        }
        partial[lane] = acc;
        workgroupBarrier();

        var stride = WG / 2u;
        while (stride > 0u) {
            if (lane < stride) {
                partial[lane] = partial[lane] + partial[lane + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        if (lane == 0u) {
            c[col] = partial[0] * scale[col];
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
