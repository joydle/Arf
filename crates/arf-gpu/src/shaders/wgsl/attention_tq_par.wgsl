// attention_tq.wgsl with the Walsh-Hadamard rotations PARALLELIZED across the
// workgroup (the original ran the whole O(d·log d) butterfly on lane 0 only,
// idling 255 lanes — the bulk of the TurboQuant decode tax). Each Hadamard stage's
// hd/2 butterflies touch disjoint (j, j+h) pairs, so one lane per butterfly is
// exact; a workgroupBarrier separates the log2(hd) sequential stages. Result is
// numerically identical to the serial butterfly (same disjoint adds per stage,
// same stage order). Everything else (R-space scoring/accumulation, inline
// TurboQuant dequant, streamed softmax) is unchanged from attention_tq.wgsl.

struct Dims {
    q_len: u32, ctx: u32, num_heads: u32, kv_heads: u32, head_dim: u32,
    past_len: u32, scale_bits: u32, group: u32, q_start: u32, window: u32,
    bits: u32, _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       key_codes: array<u32>;
@group(0) @binding(2) var<storage, read>       value_codes: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read>       slots: array<u32>;
@group(0) @binding(5) var<uniform>             d: Dims;
@group(0) @binding(6) var<storage, read>       key_norms: array<f32>;
@group(0) @binding(7) var<storage, read>       value_norms: array<f32>;
@group(0) @binding(8) var<storage, read>       levels: array<f32>;

const WG: u32 = 256u;
const TILE: u32 = 2048u;
var<workgroup> scores: array<f32, 2048>;
var<workgroup> kvi: array<u32, 2048>;
var<workgroup> red: array<f32, 256>;
var<workgroup> qrot: array<f32, 256>;
var<workgroup> orot: array<f32, 256>;
var<workgroup> lev: array<f32, 256>;
var<workgroup> m_run: f32;
var<workgroup> l_run: f32;
var<workgroup> corr: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let head = wid.x;
    let row = wid.y;

    let kv_head = head / d.group;
    let scale = bitcast<f32>(d.scale_bits);
    let hd = d.head_dim;
    let bits = d.bits;
    let mask = (1u << bits) - 1u;
    let kv_heads = d.kv_heads;
    let abs_row = d.q_start + row;
    let q_base = (abs_row * d.num_heads + head) * hd;
    let half = hd / 2u;
    let inv_sqrt = 1.0 / sqrt(f32(hd));

    let last = d.past_len + row;
    var lo = 0u;
    if (d.window > 0u && last + 1u > d.window) {
        lo = last + 1u - d.window;
    }

    if (lane <= mask) { lev[lane] = levels[lane]; }
    workgroupBarrier();

    // --- Parallel Hadamard on the query: qrot = R·q. One lane per butterfly.
    if (lane < hd) { qrot[lane] = q[q_base + lane]; }
    workgroupBarrier();
    var h = 1u;
    loop {
        if (h >= hd) { break; }
        // butterfly index `lane` (0..half): block = lane/h, w = lane%h, j = block*2h + w.
        if (lane < half) {
            let j = (lane / h) * (h * 2u) + (lane % h);
            let x = qrot[j];
            let y = qrot[j + h];
            qrot[j] = x + y;
            qrot[j + h] = x - y;
        }
        workgroupBarrier();
        h = h * 2u;
    }
    if (lane < hd) { qrot[lane] = qrot[lane] * inv_sqrt; }
    workgroupBarrier();

    var out_acc = 0.0;
    if (lane == 0u) { m_run = -3.4e38; l_run = 0.0; }
    workgroupBarrier();

    var tile0 = lo;
    loop {
        if (tile0 > last) { break; }
        let tile_end = min(tile0 + TILE - 1u, last);
        let tile_len = tile_end - tile0 + 1u;

        // Phase A1: scores[t] = scale · norm_k · Σ_i qrot[i]·levels[code_k].
        var t = lane;
        while (t < tile_len) {
            let key = tile0 + t;
            let kvidx = slots[key] * kv_heads + kv_head;
            kvi[t] = kvidx;
            var dot = 0.0;
            let base_bit = (kvidx * hd) * bits;
            var wi = base_bit >> 5u;
            var off = base_bit & 31u;
            var cur = key_codes[wi];
            var nxt = key_codes[wi + 1u];
            for (var i = 0u; i < hd; i = i + 1u) {
                var code = cur >> off;
                if (off + bits > 32u) { code = code | (nxt << (32u - off)); }
                dot = dot + qrot[i] * lev[code & mask];
                off = off + bits;
                if (off >= 32u) { off = off - 32u; wi = wi + 1u; cur = nxt; nxt = key_codes[wi + 1u]; }
            }
            scores[t] = dot * key_norms[kvidx] * scale;
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

        // Phase A3: exp; tile sum.
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

        // Phase A3.5: fold value norm into weight once.
        var p = lane;
        while (p < tile_len) { scores[p] = scores[p] * value_norms[kvi[p]]; p = p + WG; }
        workgroupBarrier();

        // Phase B: out_acc (R-space).
        if (lane < hd) {
            var add = 0.0;
            for (var tt = 0u; tt < tile_len; tt = tt + 1u) {
                let bitpos = (kvi[tt] * hd + lane) * bits;
                let word = bitpos >> 5u;
                let off = bitpos & 31u;
                var code = value_codes[word] >> off;
                if (off + bits > 32u) { code = code | (value_codes[word + 1u] << (32u - off)); }
                add = add + scores[tt] * lev[code & mask];
            }
            out_acc = out_acc * corr + add;
        }
        workgroupBarrier();
        tile0 = tile0 + TILE;
    }

    // --- Parallel inverse Hadamard on the output: out = Rᵀ·(out_acc/l_run).
    if (lane < hd) { orot[lane] = out_acc / l_run; }
    workgroupBarrier();
    h = 1u;
    loop {
        if (h >= hd) { break; }
        if (lane < half) {
            let j = (lane / h) * (h * 2u) + (lane % h);
            let x = orot[j];
            let y = orot[j + h];
            orot[j] = x + y;
            orot[j + h] = x - y;
        }
        workgroupBarrier();
        h = h * 2u;
    }
    if (lane < hd) {
        let out_base = (abs_row * d.num_heads + head) * hd;
        out[out_base + lane] = orot[lane] * inv_sqrt;
    }
}
