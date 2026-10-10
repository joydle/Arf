// WGSL per-head L2 normalize — B-PARALLEL twin of gdn_l2_norm.wgsl. One workgroup per (b, head):
// grid = B*n_heads. b-outer: head's data at (b*n_heads + head)*dim. b=0 slice ≡ single-stream.
// Serial f32 sum-of-squares in lane 0 (ggml left-to-right order), then scale. No learned gain.
//
// Bindings: 0 x[B*dim*n_heads] (READ+WRITE in place), 1 dims {dim, n_heads, eps_bits, batch}.

struct L2DimsB { dim: u32, n_heads: u32, eps_bits: u32, batch: u32, };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform>             d: L2DimsB;

var<workgroup> scale_sh: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = wg.x;
    if (lane >= d.batch * d.n_heads) { return; }
    // b-outer: the (b,head) block. base over the flat [B * n_heads * dim] buffer.
    let dim = d.dim;
    let base = lane * dim;           // (b*n_heads + head)*dim, lane already = b*n_heads + head
    let eps = bitcast<f32>(d.eps_bits);
    let t = lid.x;

    if (t == 0u) {
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < dim; i = i + 1u) {
            let xi = x[base + i];
            sum = sum + xi * xi;
        }
        scale_sh = 1.0 / max(sqrt(sum), eps);
    }
    workgroupBarrier();
    if (t < dim) {
        x[base + t] = x[base + t] * scale_sh;
    }
}
