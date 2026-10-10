// Matrix·vector: c[n] = A[1, k] · Bᵀ  (B stored [n, k], i.e. Linearᵀ).
//
// The decode-time companion to matmul.wgsl. When m == 1 (one token), the tiled
// 16×16 GEMM wastes 15/16 of every workgroup on zero-padded rows. Here one
// workgroup owns one output column n and all 256 lanes cooperate on the dot
// product over k: each lane sums a strided slice, then a tree reduction in
// shared memory combines them. The activation row A is tiny (k floats) and hot
// in cache; the cost is streaming the weight row B[n, :] once — so this kernel
// runs at weight-bandwidth, which is the decode roofline.
//
// Dims is identical to matmul.wgsl (m is ignored here; m == 1 assumed) so this
// is a drop-in for the m == 1 dispatch. Bindings match matmul.wgsl exactly.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;   // activation, 4 f32 per vec
@group(0) @binding(1) var<storage, read>       b: array<vec4<u32>>;   // bf16-packed: 8 weights per vec4
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

// 64 lanes per column (see matmul_vec_q8.wgsl for why 64, not 256).
const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

// bf16 -> f32: bf16 is the high 16 bits of an f32, so shift back up. Bit-exact.
fn bf16(bits: u32) -> f32 { return bitcast<f32>(bits << 16u); }

// Dot one packed word (2 bf16 weights) with 2 activations.
fn dot_word(word: u32, lo: f32, hi: f32) -> f32 {
    return lo * bf16(word & 0xffffu) + hi * bf16(word >> 16u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    // One vec4<u32> = 4 words = 8 bf16 weights, paired with 2 vec4<f32> of `a`.
    // Grid-stride over output columns when n exceeds the launch width (the
    // lm_head's n = vocab = 128256 is past wgpu's 65535 per-dim dispatch cap).
    let vecs = d.k / 8u;            // vec4 words per weight row (k % 8 == 0)
    var col = wid.x;
    while (col < d.n) {
        let row_base = col * vecs;
        var acc = 0.0;
        var w = lane;
        while (w < vecs) {
            let bv = b[row_base + w];    // 8 bf16 weights in one coalesced load
            let a0 = a[2u * w];          // activations [8w..8w+4)
            let a1 = a[2u * w + 1u];     // activations [8w+4..8w+8)
            acc = acc + dot_word(bv.x, a0.x, a0.y)
                      + dot_word(bv.y, a0.z, a0.w)
                      + dot_word(bv.z, a1.x, a1.y)
                      + dot_word(bv.w, a1.z, a1.w);
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
            c[col] = partial[0];
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
