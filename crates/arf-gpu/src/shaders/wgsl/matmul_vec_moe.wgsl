// MoE expert matrix·vector (bf16, batch-1 decode): like matmul_vec.wgsl, but the
// weight matrix is one expert SELECTED AT GPU RUNTIME out of a packed buffer of all
// experts. `b` holds every expert's [n, k] matrix back-to-back; the active expert
// id is read from `ids[slot]` (written by moe_route), so the row base is
// (expert * n + col) * vecs. No CPU readback: the CPU encodes top_k fixed
// iterations differing only by `slot`, and the kernel resolves the expert on-GPU.
//
// Write mode `d.mode` controls how the dot result lands in c:
//   0 = write raw            (gate/up: fresh per-expert scratch, no router weight)
//   1 = write scaled by rw   (down, FIRST selected expert: initializes the sum)
//   2 = accumulate scaled    (down, later experts/shared: weight-sums into the sum)
// So the down-projection of each selected expert weight-sums into the MLP output
// in one pass, no separate zero-fill. Bit-exact vs matmul_vec for a single expert.

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;   // activation, 4 f32/vec
@group(0) @binding(1) var<storage, read>       b: array<vec4<u32>>;   // packed experts: 8 bf16/vec4
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       ids: array<u32>;       // routed expert ids
@group(0) @binding(5) var<storage, read>       wts: array<f32>;       // routed weights

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

fn bf16(bits: u32) -> f32 { return bitcast<f32>(bits << 16u); }
fn dot_word(word: u32, lo: f32, hi: f32) -> f32 {
    return lo * bf16(word & 0xffffu) + hi * bf16(word >> 16u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let vecs = d.k / 8u;                 // vec4 words per weight row (k % 8 == 0)
    let expert = ids[d.slot];           // expert selected on-GPU
    let expert_base = expert * d.n * vecs;
    let rw = wts[d.slot];               // router weight for this expert

    var col = wid.x;
    while (col < d.n) {
        let row_base = expert_base + col * vecs;
        var acc = 0.0;
        var w = lane;
        while (w < vecs) {
            let bv = b[row_base + w];
            let a0 = a[2u * w];
            let a1 = a[2u * w + 1u];
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
            if (d.mode == 2u) {
                c[col] = c[col] + rw * partial[0];   // down (accumulate): weight-sum
            } else if (d.mode == 1u) {
                c[col] = rw * partial[0];            // down (first): init the sum
            } else {
                c[col] = partial[0];                 // gate/up: fresh per-expert scratch
            }
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
