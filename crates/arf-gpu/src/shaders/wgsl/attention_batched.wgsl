// Batched grouped-query causal attention — STREAMED (online) softmax over a
// RAGGED batch of sequences in ONE dispatch. This replaces the per-sequence loop
// over `attention.wgsl` that capped continuous-batched decode: there, B sequences
// meant B tiny `[num_heads, q_len, 1]` dispatches (each ~B× under-occupied at
// decode q_len==1) serialized with per-dispatch overhead. Here a single
// `[num_heads, total_rows, 1]` grid covers every (head, query row) across all
// sequences, so the GPU stays filled and attention cost stops growing per-stream.
//
// The per-sequence geometry (which physical pool slots a row attends, and how far)
// is carried in `row_meta`, two u32 per GLOBAL query row:
//   row_meta[row*2 + 0] = slot_base — offset into the flattened `slots` buffer
//                         where this row's sequence's slot table begins
//   row_meta[row*2 + 1] = last      — inclusive last key position the row attends
//                         (= past_len + row_within_seq, the causal frontier)
// `slots[slot_base + key]` is then the physical pool row for context position
// `key`, exactly like the single-stream kernel's `slots[key]`. Rows of the same
// sequence share a `slot_base`; `last` advances by one per row. `window` (uniform,
// per-layer) reproduces Gemma local masking; 0 = global/causal.
//
// Numerics, tiling, and the K-once/V-once bandwidth are identical to
// `attention.wgsl` — only the indexing is generalized from one sequence to a
// per-row (slot_base, last) pair, so parity with the per-seq loop is exact.

struct Dims {
    num_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    scale_bits: u32,
    group: u32,      // num_heads / kv_heads (exact)
    window: u32,     // Gemma local layers: attend only the last `window` keys; 0 = global
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       key_pool: array<f32>;
@group(0) @binding(2) var<storage, read>       value_pool: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read>       slots: array<u32>;     // flattened per-seq slot tables
@group(0) @binding(5) var<storage, read>       row_meta: array<u32>;  // 2 u32 per global query row
@group(0) @binding(6) var<uniform>             d: Dims;

const WG: u32 = 256u;
// Max output dims a lane accumulates = ceil(MAX_HEAD_DIM / WG). Gemma-4 GLOBAL
// head_dim 512 → DPL 2; ≤256 models use only slot 0 (one dim/lane).
const DPL: u32 = 2u;
const TILE: u32 = 2048u;
var<workgroup> scores: array<f32, 2048>;
var<workgroup> red: array<f32, 256>;
var<workgroup> m_run: f32;
var<workgroup> l_run: f32;
var<workgroup> corr: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let head = wid.x;
    let row = wid.y;                         // GLOBAL query row across all sequences

    let slot_base = row_meta[row * 3u];
    let causal_last = row_meta[row * 3u + 1u]; // inclusive causal frontier for this row
    let bidir_last = row_meta[row * 3u + 2u];  // Gemma-3 image span end (0 = none)

    let kv_head = head / d.group;
    let scale = bitcast<f32>(d.scale_bits);
    let hd = d.head_dim;
    let row_floats = d.kv_heads * hd;
    let head_col = kv_head * hd;
    let q_base = (row * d.num_heads + head) * hd;

    // Causal range [lo, last]; lo = 0 (global) or last+1−window (Gemma local). For a query
    // INSIDE an image span (bidir_last>0), extend `last` to the span end (attend forward over
    // the whole image block) and force lo=0 (the local window must not clip the block).
    let is_image = bidir_last > 0u;
    let last = max(causal_last, bidir_last);
    var lo = 0u;
    if (!is_image && d.window > 0u && causal_last + 1u > d.window) {
        lo = causal_last + 1u - d.window;
    }

    // Per-lane output accumulators: slot j owns output dim `lane + j·WG`. head_dim
    // may exceed WG (256): Gemma-4 GLOBAL layers use head_dim 512 → DPL 2 (a strided
    // fan-out, two dims per lane). The ≤256 models use only slot 0 (one dim/lane).
    // Without this, the upper dims of a 512-wide head were never written → garbage
    // (the batched-gemma4 global-layer corruption).
    var acc: array<f32, DPL>;
    for (var j = 0u; j < DPL; j = j + 1u) { acc[j] = 0.0; }

    if (lane == 0u) {
        m_run = -3.4e38;
        l_run = 0.0;
    }
    workgroupBarrier();

    var tile0 = lo;
    loop {
        if (tile0 > last) { break; }
        let tile_end = min(tile0 + TILE - 1u, last);
        let tile_len = tile_end - tile0 + 1u;

        // --- Phase A1: scores[t] = scale · Q · K[tile0+t] (one key per lane).
        var t = lane;
        while (t < tile_len) {
            let key = tile0 + t;
            let k_base = slots[slot_base + key] * row_floats + head_col;
            var acc = 0.0;
            for (var i = 0u; i < hd; i = i + 1u) {
                acc = acc + q[q_base + i] * key_pool[k_base + i];
            }
            scores[t] = acc * scale;
            t = t + WG;
        }
        workgroupBarrier();

        // --- Phase A2: tile max via tree reduction.
        var lmax = -3.4e38;
        var c = lane;
        while (c < tile_len) {
            lmax = max(lmax, scores[c]);
            c = c + WG;
        }
        red[lane] = lmax;
        workgroupBarrier();
        var stride = WG / 2u;
        while (stride > 0u) {
            if (lane < stride) {
                red[lane] = max(red[lane], red[lane + stride]);
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (lane == 0u) {
            let tile_max = red[0];
            let m_new = max(m_run, tile_max);
            corr = exp(m_run - m_new);
            m_run = m_new;
        }
        workgroupBarrier();

        // --- Phase A3: exp(score − m_run); tile sum.
        var lsum = 0.0;
        var e = lane;
        while (e < tile_len) {
            let ex = exp(scores[e] - m_run);
            scores[e] = ex;
            lsum = lsum + ex;
            e = e + WG;
        }
        red[lane] = lsum;
        workgroupBarrier();
        stride = WG / 2u;
        while (stride > 0u) {
            if (lane < stride) {
                red[lane] = red[lane] + red[lane + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (lane == 0u) {
            l_run = l_run * corr + red[0];
        }
        workgroupBarrier();

        // --- Phase B: fold tile into the per-dim output accumulators (DPL strided).
        for (var j = 0u; j < DPL; j = j + 1u) {
            let dim = lane + j * WG;
            if (dim < hd) {
                var add = 0.0;
                for (var tt = 0u; tt < tile_len; tt = tt + 1u) {
                    let key = tile0 + tt;
                    add = add + scores[tt] * value_pool[slots[slot_base + key] * row_floats + head_col + dim];
                }
                acc[j] = acc[j] * corr + add;
            }
        }
        workgroupBarrier();

        tile0 = tile0 + TILE;
    }

    let out_base = (row * d.num_heads + head) * hd;
    for (var j = 0u; j < DPL; j = j + 1u) {
        let dim = lane + j * WG;
        if (dim < hd) {
            out[out_base + dim] = acc[j] / l_run;
        }
    }
}
