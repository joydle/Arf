// BATCHED greedy argmax: for each of `rows` output rows, find argmax over its `vocab`-wide logits
// slice and write the token id to out[row]. One WORKGROUP per row (workgroup_id.x = row), so the
// whole batch's sampling is ONE dispatch with NO full-vocab CPU readback — the caller reads back
// only `rows` u32s (rows*4 bytes) instead of rows*vocab*4 floats. This is the on-GPU-sampling
// upgrade that removes the per-token blocking readback stall (the dominant single-stream gap).
//
// Bit-identical to running sample.wgsl per row: same strided scan + tie-keeps-lower-index tree
// reduction, just offset by row*vocab into the flat logits buffer.

struct Dims { vocab: u32, rows: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read>       logits: array<f32>;  // [rows * vocab], row-major
@group(0) @binding(1) var<storage, read_write> out: array<u32>;     // [rows] token ids
@group(0) @binding(2) var<uniform>             d: Dims;

const WG: u32 = 256u;
var<workgroup> best_vals: array<f32, 256>;
var<workgroup> best_idxs: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= d.rows) { return; }
    let base = row * d.vocab;          // this row's logits slice start
    let lane = lid.x;

    // Each lane scans its strided slice of THIS row; ties keep the lower (global) index.
    var bv = -3.4e38;
    var bi = 0u;
    var i = lane;
    while (i < d.vocab) {
        let v = logits[base + i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
        i = i + WG;
    }
    best_vals[lane] = bv;
    best_idxs[lane] = bi;
    workgroupBarrier();

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
        out[row] = bi;
    }
}
