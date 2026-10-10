// int8 matrix·vector for decode (m == 1), subgroup-reduction variant of
// matmul_vec_q8.wgsl. Identical dot (i8 sign-extend, ×scale[col]) but the
// across-lane reduction is subgroupAdd instead of the shared-mem tree. See
// matmul_vec_sg.wgsl for the why (MLX ~5× on reduction-bound matvec) and the
// no-`enable subgroups;` note (naga implements the op, not the directive). Used
// only when has_subgroups(); else the caller dispatches matmul_vec_q8.wgsl.
//
// The reduction order differs from the tree (f32 reassociation) — not bitwise
// identical, but well below a logit tie; the greedy cap tests pass unchanged.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       q: array<vec4<u32>>;   // int8: 16/vec
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scale: array<f32>;     // per output row

const WG: u32 = 64u;
var<workgroup> sg_partial: array<f32, 16>;

fn i8_at(word: u32, j: u32) -> i32 {
    return (i32(word << (24u - 8u * j)) >> 24u);
}
fn unpack4(word: u32) -> vec4<f32> {
    return vec4<f32>(f32(i8_at(word, 0u)), f32(i8_at(word, 1u)),
                     f32(i8_at(word, 2u)), f32(i8_at(word, 3u)));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sg_size: u32,
        @builtin(subgroup_invocation_id) sg_lane: u32) {
    let lane = lid.x;
    let sg_id = lane / sg_size;
    let num_sg = WG / sg_size;
    let vecs = d.k / 16u;

    var col = wid.x;
    while (col < d.n) {
        let row_base = col * vecs;
        var acc = 0.0;
        var w = lane;
        while (w < vecs) {
            let qv = q[row_base + w];
            let abase = 4u * w;
            acc = acc + dot(a[abase], unpack4(qv.x))
                      + dot(a[abase + 1u], unpack4(qv.y))
                      + dot(a[abase + 2u], unpack4(qv.z))
                      + dot(a[abase + 3u], unpack4(qv.w));
            w = w + WG;
        }

        let sg_sum = subgroupAdd(acc);
        if (sg_lane == 0u) {
            sg_partial[sg_id] = sg_sum;
        }
        workgroupBarrier();

        if (lane == 0u) {
            var total = 0.0;
            for (var s = 0u; s < num_sg; s = s + 1u) {
                total = total + sg_partial[s];
            }
            c[col] = total * scale[col];
        }
        workgroupBarrier();
        col = col + ngroups.x;
    }
}
