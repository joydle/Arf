// Split-KV (flash-decoding) attention — PASS 2 of 2: combine the `splitk` partial
// online-softmax results per (head,row) into the final attention output.
//
// Each partial s holds (acc_s[hd], m_s, l_s) where acc_s = Σ_{key in chunk s}
// exp(score − m_s)·V[key] and l_s = Σ exp(score − m_s). The standard flash merge:
//   m = max_s m_s
//   out[dim] = ( Σ_s exp(m_s − m)·acc_s[dim] ) / ( Σ_s exp(m_s − m)·l_s )
// Empty splits wrote m_s = −inf, l_s = 0 → exp(m_s − m) = 0, contributing nothing.
//
// One workgroup per (head,row); 256 lanes, one output dim per lane (DPL strided for
// hd=512). splitk is small (≤ ~64), so the per-dim loop over splits is cheap.

struct Dims {
    q_len: u32, ctx: u32, num_heads: u32, kv_heads: u32, head_dim: u32,
    past_len: u32, scale_bits: u32, group: u32, q_start: u32, window: u32,
    splitk: u32, _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       partials: array<f32>; // [head * splitk * (hd+2)]
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;

const WG: u32 = 256u;
const DPL: u32 = 2u;
var<workgroup> mg: f32;   // global max over splits
var<workgroup> lg: f32;   // global denom Σ exp(m_s−mg)·l_s

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let head = wid.x;
    let hd = d.head_dim;
    let nsp = d.splitk;
    let abs_row = d.q_start; // decode row 0

    // Lane 0 computes the global max and denominator over the splits.
    if (lane == 0u) {
        var m = -3.4e38;
        for (var s = 0u; s < nsp; s = s + 1u) {
            m = max(m, partials[(head * nsp + s) * (hd + 2u) + hd]);
        }
        var l = 0.0;
        for (var s = 0u; s < nsp; s = s + 1u) {
            let pb = (head * nsp + s) * (hd + 2u);
            let ms = partials[pb + hd];
            let ls = partials[pb + hd + 1u];
            l = l + exp(ms - m) * ls;
        }
        mg = m;
        lg = l;
    }
    workgroupBarrier();

    let inv_l = select(0.0, 1.0 / lg, lg > 0.0);
    for (var j = 0u; j < DPL; j = j + 1u) {
        let dim = lane + j * WG;
        if (dim < hd) {
            var num = 0.0;
            for (var s = 0u; s < nsp; s = s + 1u) {
                let pb = (head * nsp + s) * (hd + 2u);
                let ms = partials[pb + hd];
                num = num + exp(ms - mg) * partials[pb + dim];
            }
            let out_base = (abs_row * d.num_heads + head) * hd;
            out[out_base + dim] = num * inv_l;
        }
    }
}
