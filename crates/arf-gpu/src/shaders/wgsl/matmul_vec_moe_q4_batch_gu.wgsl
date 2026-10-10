// Batched MoE gate/up (Q4_0 block-32): FUSED analog of matmul_vec_moe_q4.wgsl —
// all top_k routed experts' gate (or up) in ONE dispatch. Output [top_k, n]
// (slot-major c[slot*n + col]); grid strides flat (slot, col); expert = ids[slot].
// Raw write (router weight applied later in the fused down). top_k in Dims._p0.
// Inner dequant/reduction is byte-identical to matmul_vec_moe_q4.wgsl.

struct Dims { m: u32, k: u32, n: u32, slot: u32, mode: u32, top_k: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;     // activation [k], 4/vec
@group(0) @binding(1) var<storage, read>       codes: array<vec4<u32>>; // packed experts' codes
@group(0) @binding(2) var<storage, read_write> c: array<f32>;           // [top_k * n]
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;      // packed experts' scales
@group(0) @binding(5) var<storage, read>       ids: array<u32>;

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

fn bf16_to_f32(bits: u32) -> f32 { return bitcast<f32>((bits & 0xFFFFu) << 16u); }
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
    let blocks = d.k / 32u;              // blocks per weight row
    let total = d.top_k * d.n;

    var idx = wid.x;
    while (idx < total) {
        let slot = idx / d.n;
        let col = idx % d.n;
        let expert = ids[slot];
        let code_base = (expert * d.n + col) * blocks;
        let scale_base = (expert * d.n + col) * blocks;
        var acc = 0.0;
        var b = lane;
        while (b < blocks) {
            let qv = codes[code_base + b];
            let s = bf16_to_f32(scales[scale_base + b]);
            let abase = 8u * b;
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
            c[idx] = partial[0];
        }
        workgroupBarrier();
        idx = idx + ngroups.x;
    }
}
