// Grouped-query causal attention for one sequence — STREAMED (online) softmax.
//   For each (head, query row): scores = scale·Q·Kᵀ, causal-mask, softmax, ·V.
// Matches attend_sequence + causal_softmax (the CPU oracle).
//
// One workgroup per (head, query row). The context is processed in TILEs: the
// workgroup keeps a running max `m`, running denominator `l`, and a per-output-
// dim accumulator, updating them tile-by-tile with the flash-attention rescale
// (`exp(m_old − m_new)`). This caps shared memory at O(TILE) instead of O(ctx),
// so context is bounded only by the KV pool, NOT by a fixed score row. (The old
// kernel held the whole score row in shared memory → ctx ≤ 2048; this removes
// that cap, which is what the long-context / TurboQuant work needs.)
//
// K/V are read DIRECTLY from the paged pool by slot (FlashInfer-style in-kernel
// gather) — `slots[pos]` is the physical pool row for context position `pos`,
// pool row stride `row_floats = kv_heads * head_dim` (matches kv_scatter, the
// writer). Each key is read once (score phase) and each value once (output
// phase), per tile — the same optimal K-once/V-once bandwidth as before.
//
// Numerics: for ctx ≤ TILE this is a single tile and reduces to the exact
// max→exp→sum→normalize order (parity with the CPU oracle to tolerance); for
// ctx > TILE the online rescale is numerically stable (that is its purpose).
//
// head_dim may exceed WG (256): Gemma 4 GLOBAL layers use head_dim 512. Each lane
// owns the output dims `lane, lane+WG, lane+2·WG, …` (a strided fan-out), held in
// the `acc` array — DPL (dims-per-lane) = ceil(head_dim/WG) slots, 2 for hd=512,
// 1 for the ≤256 models (then this is exactly the old one-dim-per-lane kernel).

struct Dims {
    q_len: u32,
    ctx: u32,        // past_len + q_len (kept for the uniform layout; the loop
                     // bound is `last`, derived from past_len + row + window)
    num_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    past_len: u32,
    scale_bits: u32,
    group: u32,      // num_heads / kv_heads (exact; the config guarantees divisibility)
    q_start: u32,    // first query row of this sequence within the (batched) q/out buffers
    window: u32,     // Gemma local layers: attend only the last `window` keys; 0 = global/causal
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       key_pool: array<f32>;
@group(0) @binding(2) var<storage, read>       value_pool: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read>       slots: array<u32>;
@group(0) @binding(5) var<uniform>             d: Dims;

const WG: u32 = 256u;
// Max output dims a single lane accumulates = ceil(MAX_HEAD_DIM / WG). MAX_HEAD_DIM
// 512 (Gemma 4 global) → DPL 2; the ≤256 models use only slot 0 (one dim/lane).
const DPL: u32 = 2u;
// Tile width: the score row is computed and consumed one TILE at a time. Shared
// memory is O(TILE), independent of context length.
const TILE: u32 = 2048u;
var<workgroup> scores: array<f32, 2048>;
// Per-lane scratch for the workgroup tree reductions (tile max, then tile sum).
var<workgroup> red: array<f32, 256>;
// Running softmax state for this (head, row), shared across the workgroup.
var<workgroup> m_run: f32;   // running max of scores seen so far
var<workgroup> l_run: f32;   // running Σ exp(score − m_run)
var<workgroup> corr: f32;    // this tile's rescale factor exp(m_old − m_new)

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let head = wid.x;
    let row = wid.y;

    let kv_head = head / d.group;           // GQA mapping
    let scale = bitcast<f32>(d.scale_bits);
    let hd = d.head_dim;
    let row_floats = d.kv_heads * hd;       // floats per cached token
    let head_col = kv_head * hd;            // this kv head's offset within a pool row
    let abs_row = d.q_start + row;          // row within the (batched) q/out buffers
    let q_base = (abs_row * d.num_heads + head) * hd;

    // Causal range [lo, last]: query `row` (absolute position past_len+row) attends
    // keys lo..=last, where lo = 0 (global/causal) or last+1−window (Gemma local).
    let last = d.past_len + row;
    var lo = 0u;
    if (d.window > 0u && last + 1u > d.window) {
        lo = last + 1u - d.window;
    }

    // Per-lane output accumulators: slot j owns output dim `lane + j·WG`. Σ over
    // keys of the softmax weight times V[key, that dim], in running-max space,
    // divided by l_run at the end. Only dims < hd are valid.
    var acc: array<f32, DPL>;
    for (var j = 0u; j < DPL; j = j + 1u) { acc[j] = 0.0; }

    if (lane == 0u) {
        m_run = -3.4e38;
        l_run = 0.0;
    }
    workgroupBarrier();

    // Stream the causal range [lo, last] in tiles of TILE keys.
    var tile0 = lo;
    loop {
        if (tile0 > last) { break; }
        let tile_end = min(tile0 + TILE - 1u, last);   // inclusive last key in tile
        let tile_len = tile_end - tile0 + 1u;

        // --- Phase A1: scores[t] = scale · Q · K[tile0+t] (one key per lane, strided).
        var t = lane;
        while (t < tile_len) {
            let key = tile0 + t;
            let k_base = slots[key] * row_floats + head_col;
            var acc = 0.0;
            for (var i = 0u; i < hd; i = i + 1u) {
                acc = acc + q[q_base + i] * key_pool[k_base + i];
            }
            scores[t] = acc * scale;
            t = t + WG;
        }
        workgroupBarrier();

        // --- Phase A2: tile max via shared-memory tree reduction (idle lanes -inf).
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
        // Update the running max and this tile's rescale factor (single lane,
        // broadcast via workgroup memory). First tile: m_run=-inf → corr=0 (the
        // zero acc/l it scales are already 0).
        if (lane == 0u) {
            let tile_max = red[0];
            let m_new = max(m_run, tile_max);
            corr = exp(m_run - m_new);
            m_run = m_new;
        }
        workgroupBarrier();

        // --- Phase A3: overwrite scores[t] with exp(score − m_run); sum them.
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

        // --- Phase B: fold this tile into the per-dim output accumulators.
        // acc[j] = acc[j]·corr + Σ_key exp(score−m_run)·V[key, dim] for dim=lane+j·WG.
        // Each lane owns DPL strided dims; V is read once per key per owned dim.
        for (var j = 0u; j < DPL; j = j + 1u) {
            let dim = lane + j * WG;
            if (dim < hd) {
                var add = 0.0;
                for (var tt = 0u; tt < tile_len; tt = tt + 1u) {
                    let key = tile0 + tt;
                    add = add + scores[tt] * value_pool[slots[key] * row_floats + head_col + dim];
                }
                acc[j] = acc[j] * corr + add;
            }
        }
        workgroupBarrier();   // all reads of `scores` done before the next tile overwrites it

        tile0 = tile0 + TILE;
    }

    // Finalize: out[row, head, dim] = acc[j] / l_run for each owned dim.
    let out_base = (abs_row * d.num_heads + head) * hd;
    for (var j = 0u; j < DPL; j = j + 1u) {
        let dim = lane + j * WG;
        if (dim < hd) {
            out[out_base + dim] = acc[j] / l_run;
        }
    }
}
