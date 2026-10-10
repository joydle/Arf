// Greedy argmax over logits[vocab] -> one token id (lowest index on ties).
// Matches the greedy path of Sampler::sample.
//
// Single-workgroup reduction: each lane scans a strided slice tracking the best
// (value, lowest-index); a tree reduction combines them; lane 0 writes out[0].

struct Dims { vocab: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read>       logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform>             d: Dims;

const WG: u32 = 256u;
var<workgroup> best_vals: array<f32, 256>;
var<workgroup> best_idxs: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;

    // Each lane scans its strided slice; ties keep the lower index.
    var bv = -3.4e38;
    var bi = 0u;
    var i = lane;
    while (i < d.vocab) {
        let v = logits[i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
        i = i + WG;
    }
    best_vals[lane] = bv;
    best_idxs[lane] = bi;
    workgroupBarrier();

    // Tree reduction; strictly-greater wins, equal keeps the lower index.
    var stride = WG / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            let ov = best_vals[lane + stride];
            let oi = best_idxs[lane + stride];
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
            best_vals[lane] = bv;
            best_idxs[lane] = bi;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        out[0] = bi;
    }
}
