// Split-KV (flash-decoding) attention — PASS 1 of 2. Splits the causal key range
// across `splitk` workgroups per (head,row) so NO single dispatch streams the whole
// context: at long ctx (e.g. 8192) the one-workgroup attention.wgsl exceeds Metal's
// ~10s per-dispatch watchdog and wedges the GPU. Each split computes a PARTIAL
// online-softmax over its key chunk and writes (acc[hd], m, l) to `partials`; the
// combine kernel (attention_splitk_combine.wgsl) merges the `splitk` partials per
// (head,row) into the final output. Adds splitk× more workgroups → also better
// occupancy at long ctx.
//
// Numerically a standard flash-decoding split: partial m_i/l_i/acc_i, then combine
// rescales by exp(m_i − m_global). Matches attention.wgsl / the CPU oracle to the
// online-softmax tolerance. Grid: wid.x = head, wid.y = split index (decode row 0).
//
// head_dim ≤ 256 (one lane per output dim, DPL strided fan-out for hd=512 globals).

struct Dims {
    q_len: u32, ctx: u32, num_heads: u32, kv_heads: u32, head_dim: u32,
    past_len: u32, scale_bits: u32, group: u32, q_start: u32, window: u32,
    splitk: u32, _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       key_pool: array<f32>;
@group(0) @binding(2) var<storage, read>       value_pool: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>; // [head * splitk * (hd+2)]
@group(0) @binding(4) var<storage, read>       slots: array<u32>;
@group(0) @binding(5) var<uniform>             d: Dims;

const WG: u32 = 256u;
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
    let sp = wid.y;           // split index 0..splitk
    let row = 0u;             // decode: single query row

    let kv_head = head / d.group;
    let scale = bitcast<f32>(d.scale_bits);
    let hd = d.head_dim;
    let row_floats = d.kv_heads * hd;
    let head_col = kv_head * hd;
    let abs_row = d.q_start + row;
    let q_base = (abs_row * d.num_heads + head) * hd;

    // Full causal range [lo, last] for this (head,row).
    let last = d.past_len + row;
    var lo = 0u;
    if (d.window > 0u && last + 1u > d.window) {
        lo = last + 1u - d.window;
    }
    // This split owns the key sub-range [s_lo, s_hi] of [lo, last], chunked evenly.
    let total = last + 1u - lo;
    let chunk = (total + d.splitk - 1u) / d.splitk;   // ceil
    let s_lo = lo + sp * chunk;
    var s_hi = s_lo + chunk;                           // exclusive
    if (s_hi > last + 1u) { s_hi = last + 1u; }

    let pbase = (head * d.splitk + sp) * (hd + 2u);

    var acc: array<f32, DPL>;
    for (var j = 0u; j < DPL; j = j + 1u) { acc[j] = 0.0; }
    if (lane == 0u) { m_run = -3.4e38; l_run = 0.0; }
    workgroupBarrier();

    // Empty split (chunk overshoots the range): write a neutral partial (l=0).
    if (s_lo >= s_hi) {
        if (lane == 0u) {
            partials[pbase + hd] = -3.4e38; // m
            partials[pbase + hd + 1u] = 0.0; // l
        }
        for (var j = 0u; j < DPL; j = j + 1u) {
            let dim = lane + j * WG;
            if (dim < hd) { partials[pbase + dim] = 0.0; }
        }
        return;
    }

    var tile0 = s_lo;
    loop {
        if (tile0 >= s_hi) { break; }
        let tile_end = min(tile0 + TILE, s_hi);        // exclusive
        let tile_len = tile_end - tile0;

        // Phase A1: scores[t] = scale · Q · K[tile0+t].
        var t = lane;
        while (t < tile_len) {
            let key = tile0 + t;
            let k_base = slots[key] * row_floats + head_col;
            var dotp = 0.0;
            for (var i = 0u; i < hd; i = i + 1u) {
                dotp = dotp + q[q_base + i] * key_pool[k_base + i];
            }
            scores[t] = dotp * scale;
            t = t + WG;
        }
        workgroupBarrier();

        // Phase A2: tile max.
        var lmax = -3.4e38;
        var c = lane;
        while (c < tile_len) { lmax = max(lmax, scores[c]); c = c + WG; }
        red[lane] = lmax;
        workgroupBarrier();
        var stride = WG / 2u;
        while (stride > 0u) {
            if (lane < stride) { red[lane] = max(red[lane], red[lane + stride]); }
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

        // Phase A3: exp(score − m_run); tile sum.
        var lsum = 0.0;
        var e = lane;
        while (e < tile_len) {
            let ex = exp(scores[e] - m_run);
            scores[e] = ex; lsum = lsum + ex; e = e + WG;
        }
        red[lane] = lsum;
        workgroupBarrier();
        stride = WG / 2u;
        while (stride > 0u) {
            if (lane < stride) { red[lane] = red[lane] + red[lane + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (lane == 0u) { l_run = l_run * corr + red[0]; }
        workgroupBarrier();

        // Phase B: fold this tile into per-dim accumulators (rescaled by corr).
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
        workgroupBarrier();
        tile0 = tile0 + TILE;
    }

    // Write the UN-normalized partial: acc (Σ exp·V in m_run space), m_run, l_run.
    for (var j = 0u; j < DPL; j = j + 1u) {
        let dim = lane + j * WG;
        if (dim < hd) { partials[pbase + dim] = acc[j]; }
    }
    if (lane == 0u) {
        partials[pbase + hd] = m_run;
        partials[pbase + hd + 1u] = l_run;
    }
}
