// Q4_0 (block-32, 4-bit) matrix·vector for decode: c[n] = Σ_blocks scale[n,b] ·
// Σ_{w in block} (codes·a). Half the int8 bytes on the bandwidth-bound matmuls —
// the lever to beat ollama on the same precision class it runs (Q4_0).
//
// The Q4 companion to matmul_vec_q8.wgsl. Bit-identical to the CPU `matmul_nt_q4`
// oracle (same block scale applied to the same integer sub-dot, same nibble
// sign-extend), so CPU Q4 generation is the parity oracle.
//
// Layout (see Q4Matrix):
//   codes:  array<u32>, 8 signed-4-bit codes per word, low nibble = weight 0.
//           A row is k/8 words; a 32-weight block is exactly 4 words.
//   scales: array<u32>, ONE bf16 block-scale per word (low 16 bits). A row is
//           k/32 scale-words (one per block) — no packing, so per-row indexing
//           never straddles a word (a 2-per-word layout breaks when a row's block
//           count is odd, e.g. k=32).
//   k % 32 == 0 (true for every Llama weight dim).

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // activation [k], 4/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // 32 nibbles/vec4 = one block
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;    // 2 bf16/word

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

// Widen a bf16 bit pattern (low 16 bits of `bits`) to f32: bf16 is the top 16
// bits of an f32, so shift left 16. Matches dtype::bf16_to_f32.
fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>((bits & 0xFFFFu) << 16u);
}

// Unpack the LOW four nibbles of `word` (weights 0..4) to signed f32 in [-8,7],
// vector-wide: four shifts + one mask + one branchless sign-extend, vs four
// scalar nib() calls. `select(0,16,n>=8)` realizes the two's-complement fold for
// all four lanes at once (P-Q4b: the dequant was ALU-bound, not bandwidth-bound).
fn unpack_lo(word: u32) -> vec4<f32> {
    let n = vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(0xFu);
    let sign = select(vec4<f32>(0.0), vec4<f32>(16.0), n >= vec4<u32>(8u));
    return vec4<f32>(n) - sign;
}
// Unpack the HIGH four nibbles (weights 4..8).
fn unpack_hi(word: u32) -> vec4<f32> {
    let n = vec4<u32>(word >> 16u, word >> 20u, word >> 24u, word >> 28u) & vec4<u32>(0xFu);
    let sign = select(vec4<f32>(0.0), vec4<f32>(16.0), n >= vec4<u32>(8u));
    return vec4<f32>(n) - sign;
}

// Block scale for block index `b` of this row (one bf16 per scales word).
fn block_scale(scale_base: u32, b: u32) -> f32 {
    return bf16_to_f32(scales[scale_base + b]);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let blocks = d.k / 32u;          // blocks per row (one vec4 code-load each)
    var col = wid.x;                 // grid-stride over output columns
    while (col < d.n) {
        let code_base = col * blocks;   // in vec4<u32> units (one per block)
        let scale_base = col * blocks;  // one bf16 scale per block
        var acc = 0.0;
        var b = lane;
        while (b < blocks) {
            // P-Q4b: load a whole 32-weight block as ONE vec4<u32> (matching int8's
            // 16-weight vec4 load width — the scalar-u32 load was the throttle, not
            // the unpack ALU). 8 vec4 unpacks + 8 `dot`s over the block's 4 words,
            // and the bf16 scale read ONCE per block instead of per word.
            let qv = codes[code_base + b];
            let s = block_scale(scale_base, b);
            let abase = 8u * b;   // 32 activations = 8 vec4<f32>
            let blk = dot(a[abase],      unpack_lo(qv.x)) + dot(a[abase + 1u], unpack_hi(qv.x))
                    + dot(a[abase + 2u], unpack_lo(qv.y)) + dot(a[abase + 3u], unpack_hi(qv.y))
                    + dot(a[abase + 4u], unpack_lo(qv.z)) + dot(a[abase + 5u], unpack_hi(qv.z))
                    + dot(a[abase + 6u], unpack_lo(qv.w)) + dot(a[abase + 7u], unpack_hi(qv.w));
            acc = acc + s * blk;
            b = b + WG;
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
            c[col] = partial[0];
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
