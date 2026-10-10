// Matrix·vector for decode (m == 1), subgroup-reduction variant of matmul_vec.wgsl.
//
// Same dot product, but the across-lane reduction uses `subgroupAdd` instead of a
// shared-memory tree (log2(64)=6 barrier'd steps). On a reduction-bound small
// matvec this is the dominant cost — MLX measured ~5× swapping the tree for
// subgroup/quad ops. WG=64 spans >=2 subgroups (Apple simd width 32): each
// subgroup reduces with one subgroupAdd, then the (<=16) subgroup totals are
// combined through a tiny shared array (one barrier instead of six).
//
// NOTE: no `enable subgroups;` directive — naga 29 rejects that directive but DOES
// implement the subgroupAdd op, and wgpu enables the SUBGROUP device feature at the
// pipeline level (requested in GpuContext::new when the adapter has it). Used only
// when has_subgroups(); otherwise the caller dispatches matmul_vec.wgsl.
//
// The reduction order differs from the tree, so this is NOT bitwise identical — f32
// reassociation. The decode parity tests tolerate that (the difference is far below
// a logit tie); confirmed against the CPU oracle and the greedy cap tests.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       b: array<vec4<u32>>;   // bf16-packed
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

const WG: u32 = 64u;
// One slot per possible subgroup (WG / min subgroup size; min is 4 → <=16).
var<workgroup> sg_partial: array<f32, 16>;

fn bf16(bits: u32) -> f32 { return bitcast<f32>(bits << 16u); }

fn dot_word(word: u32, lo: f32, hi: f32) -> f32 {
    return lo * bf16(word & 0xffffu) + hi * bf16(word >> 16u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sg_size: u32,
        @builtin(subgroup_invocation_id) sg_lane: u32) {
    let lane = lid.x;
    let sg_id = lane / sg_size;   // which subgroup this lane belongs to
    let num_sg = WG / sg_size;    // subgroups per workgroup
    let vecs = d.k / 8u;

    var col = wid.x;
    while (col < d.n) {
        let row_base = col * vecs;
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

        // Reduce within each subgroup in one op; lane 0 of each stages its total.
        let sg_sum = subgroupAdd(acc);
        if (sg_lane == 0u) {
            sg_partial[sg_id] = sg_sum;
        }
        workgroupBarrier();

        // Combine the few subgroup totals (workgroup lane 0).
        if (lane == 0u) {
            var total = 0.0;
            for (var s = 0u; s < num_sg; s = s + 1u) {
                total = total + sg_partial[s];
            }
            c[col] = total;
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
