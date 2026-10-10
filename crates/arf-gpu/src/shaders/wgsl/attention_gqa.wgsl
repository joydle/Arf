// GQA-grouped batched attention — STREAMED (online) softmax, one workgroup per
// (kv_head, query row), serving ALL `group = num_heads / kv_heads` query heads
// that share this KV head IN ONE PASS. This is the bandwidth lever on top of
// `attention_batched`: there, one workgroup per (query head, row) re-reads the
// SAME K/V slots `group` times (4× on Llama-1B, 2× on gemma3). Decode attention
// is KV-bandwidth-bound, so reading each K and each V slot ONCE and fanning it
// out across the group's query heads cuts the dominant traffic by `group`.
//
// Ragged-batch geometry is identical to `attention_batched`: `row_meta` carries
// per GLOBAL query row (slot_base, last); `slots[slot_base + key]` is the physical
// pool row; `window` (uniform) reproduces Gemma local masking (0 = global).
//
// Per-group state lives in small workgroup arrays (m_run/l_run/corr indexed by g);
// the score tile is one shared buffer partitioned into `group` rows of
// `TW = SCORES_CAP / group` each, so a fixed 16 KB covers group ∈ {2,4,8} (TW =
// 2048/1024/512). Numerics match `attention.wgsl` per query head exactly, so this
// is bit-parity with the per-seq / single-stream path (the continuous-batching
// test exercises group=2).
//
// Assumes head_dim ≤ 256 (one output dim per lane) and group ≤ 8 — true for every
// supported model; the host falls back to `attention_batched` otherwise.

struct Dims {
    num_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    scale_bits: u32,
    group: u32,      // num_heads / kv_heads
    window: u32,     // Gemma local layers: last `window` keys; 0 = global
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       key_pool: array<f32>;
@group(0) @binding(2) var<storage, read>       value_pool: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read>       slots: array<u32>;
@group(0) @binding(5) var<storage, read>       row_meta: array<u32>;
@group(0) @binding(6) var<uniform>             d: Dims;

const WG: u32 = 256u;
const SCORES_CAP: u32 = 4096u;  // 16 KB; partitioned into `group` rows of TW each
const MAXGROUP: u32 = 8u;
var<workgroup> scores: array<f32, 4096>;
var<workgroup> red: array<f32, 256>;
var<workgroup> m_run: array<f32, 8>;
var<workgroup> l_run: array<f32, 8>;
var<workgroup> corr: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let kv_head = wid.x;
    let row = wid.y;
    let group = d.group;
    let tw = SCORES_CAP / group;          // per-query-head score tile width

    let slot_base = row_meta[row * 3u];
    let causal_last = row_meta[row * 3u + 1u];
    let bidir_last = row_meta[row * 3u + 2u]; // Gemma-3 image span end (0 = none)

    let scale = bitcast<f32>(d.scale_bits);
    let hd = d.head_dim;
    let row_floats = d.kv_heads * hd;
    let head_col = kv_head * hd;           // this KV head's offset within a pool row
    let head0 = kv_head * group;           // first query head sharing this KV head

    // Bidirectional image-span attention: a query inside an image span extends `last` to the
    // span end and ignores the local window (lo=0). See attention_batched.wgsl.
    let is_image = bidir_last > 0u;
    let last = max(causal_last, bidir_last);
    var lo = 0u;
    if (!is_image && d.window > 0u && causal_last + 1u > d.window) {
        lo = causal_last + 1u - d.window;
    }

    // Per-lane output accumulator, one per query head in the group (lane < hd).
    var out_acc: array<f32, 8>;
    for (var g = 0u; g < group; g = g + 1u) {
        out_acc[g] = 0.0;
    }
    if (lane < MAXGROUP) {
        m_run[lane] = -3.4e38;
        l_run[lane] = 0.0;
    }
    workgroupBarrier();

    var tile0 = lo;
    loop {
        if (tile0 > last) { break; }
        let tile_end = min(tile0 + tw - 1u, last);
        let tile_len = tile_end - tile0 + 1u;

        // --- Phase A1: read K[key] ONCE per key; score it against all `group`
        // query heads. scores[g*tw + t] = scale · Q_g · K[tile0+t].
        var t = lane;
        while (t < tile_len) {
            let key = tile0 + t;
            let k_base = slots[slot_base + key] * row_floats + head_col;
            for (var g = 0u; g < group; g = g + 1u) {
                let q_base = (row * d.num_heads + head0 + g) * hd;
                var acc = 0.0;
                for (var i = 0u; i < hd; i = i + 1u) {
                    acc = acc + q[q_base + i] * key_pool[k_base + i];
                }
                scores[g * tw + t] = acc * scale;
            }
            t = t + WG;
        }
        workgroupBarrier();

        // --- Phase A2/A3 per query head: tile max → running max + rescale → exp+sum.
        for (var g = 0u; g < group; g = g + 1u) {
            let goff = g * tw;
            var lmax = -3.4e38;
            var c = lane;
            while (c < tile_len) {
                lmax = max(lmax, scores[goff + c]);
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
                let m_new = max(m_run[g], red[0]);
                corr[g] = exp(m_run[g] - m_new);
                m_run[g] = m_new;
            }
            workgroupBarrier();

            var lsum = 0.0;
            var e = lane;
            while (e < tile_len) {
                let ex = exp(scores[goff + e] - m_run[g]);
                scores[goff + e] = ex;
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
                l_run[g] = l_run[g] * corr[g] + red[0];
            }
            workgroupBarrier();
        }

        // --- Phase B: read V[key] ONCE per (key, dim); fold into every group head.
        if (lane < hd) {
            // Rescale each head's accumulator for this tile's new max.
            for (var g = 0u; g < group; g = g + 1u) {
                out_acc[g] = out_acc[g] * corr[g];
            }
            for (var tt = 0u; tt < tile_len; tt = tt + 1u) {
                let key = tile0 + tt;
                let v = value_pool[slots[slot_base + key] * row_floats + head_col + lane];
                for (var g = 0u; g < group; g = g + 1u) {
                    out_acc[g] = out_acc[g] + scores[g * tw + tt] * v;
                }
            }
        }
        workgroupBarrier();   // all score reads done before the next tile overwrites

        tile0 = tile0 + tw;
    }

    // Finalize: out[row, head0+g, lane] = out_acc[g] / l_run[g].
    if (lane < hd) {
        for (var g = 0u; g < group; g = g + 1u) {
            let out_base = (row * d.num_heads + head0 + g) * hd;
            out[out_base + lane] = out_acc[g] / l_run[g];
        }
    }
}
