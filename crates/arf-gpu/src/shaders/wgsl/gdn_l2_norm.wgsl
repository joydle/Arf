// WGSL per-head L2 normalize for the gated-delta-net (M11 single-queue port). BIT-EXACT twin of
// gdn_l2_norm in gdn_prologue_msl.metal (ggml_l2_norm): y = x / max(sqrt(sum_i x_i^2), eps), NO
// learned gain. Serial f32 sum-of-squares in invocation 0 (matches ggml's left-to-right order),
// then each invocation scales its element. One workgroup per head, `dim` threads.
//
// Bindings: 0 x[dim*n_heads] (READ+WRITE in place), 1 dims {dim, n_heads, eps_bits, _}.

struct L2Dims { dim: u32, n_heads: u32, eps_bits: u32, _p: u32, };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform>             d: L2Dims;

var<workgroup> scale_sh: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wg.x;
    if (head >= d.n_heads) { return; }
    let dim = d.dim;
    let base = head * dim;
    let eps = bitcast<f32>(d.eps_bits);
    let t = lid.x;

    if (t == 0u) {
        // Serial f32 accumulation (ggml left-to-right order; f32 within tol — see probe).
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < dim; i = i + 1u) {
            let xi = x[base + i];
            sum = sum + xi * xi;
        }
        scale_sh = 1.0 / max(sqrt(sum), eps);   // ggml: 1/fmaxf(sqrtf(sum), eps)
    }
    workgroupBarrier();
    if (t < dim) {
        x[base + t] = x[base + t] * scale_sh;
    }
}
