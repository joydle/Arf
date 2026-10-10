// Grouped-query causal attention over a TurboQuant-packed KV pool — STREAMED
// (online) softmax with INLINE dequant + the rotation trick.
//
// This is `attention.wgsl` (the tiled online-softmax kernel) with the K/V reads
// replaced by TurboQuant reconstruction, fused into the kernel (no separate
// dequant pass). Per head-vector the pool stores: a `bits`-wide code per
// coordinate (packed, LSB-first, in `key_codes`/`value_codes`) and one f32 norm
// (`key_norms`/`value_norms`). The codebook reconstruction `levels[code]` is a
// small storage array (length 2^bits).
//
// The rotation trick (R orthogonal ⇒ q·k = (Rq)·(Rk)): keys/values are stored
// already ROTATED (the CPU writer packs quant(R·k̂)·‖k‖). So we:
//   1. pre-rotate the query once: q' = R·q   (R = normalized Walsh-Hadamard),
//   2. score q' against the dequantized rotated keys (dot in R-space),
//   3. accumulate the output in R-space (Σ a·(R·v)),
//   4. inverse-rotate the output once: out = Rᵀ·out_rot  (Rᵀ = R for Hadamard).
// Keys/values are never rotated back — only the per-(head,row) query and output
// are, a single O(d log d) Hadamard each.
//
// Matches the CPU oracle: gather → reconstruct K/V to f32 → attend_sequence.
// One workgroup per (head, query row); head_dim ≤ 256 (one lane per output dim).

struct Dims {
    q_len: u32,
    ctx: u32,
    num_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    past_len: u32,
    scale_bits: u32,
    group: u32,
    q_start: u32,
    window: u32,
    bits: u32,        // TurboQuant bits per coordinate (2/3/4)
    _pad2: u32,
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
var<workgroup> kvi: array<u32, 2048>;   // per-tile packed-vector index (slots→kvidx), computed once
var<workgroup> red: array<f32, 256>;
var<workgroup> qrot: array<f32, 256>;   // the rotated query q' = R·q (persists across tiles)
var<workgroup> orot: array<f32, 256>;   // the R-space output, inverse-rotated at the end
var<workgroup> lev: array<f32, 256>;    // codebook (2^bits ≤ 16 entries) cached out of storage
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

    let last = d.past_len + row;
    var lo = 0u;
    if (d.window > 0u && last + 1u > d.window) {
        lo = last + 1u - d.window;
    }

    // --- Cache the tiny codebook (2^bits entries) out of storage into workgroup
    //     memory: it is read on every coordinate in both hot loops below.
    if (lane <= mask) {
        lev[lane] = levels[lane];
    }
    workgroupBarrier();

    // --- Rotate the query once: qrot = R·q (normalized Walsh-Hadamard, in place).
    if (lane < hd) {
        qrot[lane] = q[q_base + lane];
    }
    workgroupBarrier();
    if (lane == 0u) {
        var h = 1u;
        loop {
            if (h >= hd) { break; }
            var i = 0u;
            loop {
                if (i >= hd) { break; }
                var j = i;
                loop {
                    if (j >= i + h) { break; }
                    let x = qrot[j];
                    let y = qrot[j + h];
                    qrot[j] = x + y;
                    qrot[j + h] = x - y;
                    j = j + 1u;
                }
                i = i + h * 2u;
            }
            h = h * 2u;
        }
        let s = 1.0 / sqrt(f32(hd));
        var k = 0u;
        loop {
            if (k >= hd) { break; }
            qrot[k] = qrot[k] * s;
            k = k + 1u;
        }
    }
    workgroupBarrier();

    var out_acc = 0.0;
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

        // --- Phase A1: scores[t] = scale · norm_k · Σ_i qrot[i]·levels[code_k].
        // One lane owns a whole key vector, so its `bits`-wide codes are contiguous
        // in `key_codes`: stream them with a carry buffer (one word load per 32 bits)
        // instead of re-deriving word/offset (and reloading the same word) per coord.
        var t = lane;
        while (t < tile_len) {
            let key = tile0 + t;
            let kvidx = slots[key] * kv_heads + kv_head;   // packed vector index
            kvi[t] = kvidx;                                // reused in Phase A3.5 / B
            var dot = 0.0;
            let base_bit = (kvidx * hd) * bits;
            var wi = base_bit >> 5u;
            var off = base_bit & 31u;
            var cur = key_codes[wi];
            var nxt = key_codes[wi + 1u];
            for (var i = 0u; i < hd; i = i + 1u) {
                var code = cur >> off;
                if (off + bits > 32u) {
                    code = code | (nxt << (32u - off));
                }
                dot = dot + qrot[i] * lev[code & mask];
                off = off + bits;
                if (off >= 32u) {
                    off = off - 32u;
                    wi = wi + 1u;
                    cur = nxt;
                    nxt = key_codes[wi + 1u];
                }
            }
            scores[t] = dot * key_norms[kvidx] * scale;
            t = t + WG;
        }
        workgroupBarrier();

        // --- Phase A2: tile max (tree reduction).
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

        // --- Phase A3.5: fold the per-key value norm into the weight ONCE.
        // Phase B replicates its inner loop over all `hd` lanes, so reading
        // `value_norms[kvidx]` there costs it `hd`× per key; do it here instead.
        var p = lane;
        while (p < tile_len) {
            scores[p] = scores[p] * value_norms[kvi[p]];
            p = p + WG;
        }
        workgroupBarrier();

        // --- Phase B: out_acc = out_acc·corr + Σ_key weight·levels[code_v].
        // Each active lane owns one output dim (= lane), in R-space. `kvi`/`scores`
        // come from workgroup memory; only the packed value code is read per key.
        if (lane < hd) {
            var add = 0.0;
            for (var tt = 0u; tt < tile_len; tt = tt + 1u) {
                let bitpos = (kvi[tt] * hd + lane) * bits;
                let word = bitpos >> 5u;
                let off = bitpos & 31u;
                var code = value_codes[word] >> off;
                if (off + bits > 32u) {
                    code = code | (value_codes[word + 1u] << (32u - off));
                }
                add = add + scores[tt] * lev[code & mask];
            }
            out_acc = out_acc * corr + add;
        }
        workgroupBarrier();

        tile0 = tile0 + TILE;
    }

    // --- Inverse-rotate the R-space output once: out = Rᵀ·(out_acc/l_run) = H(·).
    if (lane < hd) {
        orot[lane] = out_acc / l_run;
    }
    workgroupBarrier();
    if (lane == 0u) {
        var h = 1u;
        loop {
            if (h >= hd) { break; }
            var i = 0u;
            loop {
                if (i >= hd) { break; }
                var j = i;
                loop {
                    if (j >= i + h) { break; }
                    let x = orot[j];
                    let y = orot[j + h];
                    orot[j] = x + y;
                    orot[j + h] = x - y;
                    j = j + 1u;
                }
                i = i + h * 2u;
            }
            h = h * 2u;
        }
        let s = 1.0 / sqrt(f32(hd));
        var k = 0u;
        loop {
            if (k >= hd) { break; }
            orot[k] = orot[k] * s;
            k = k + 1u;
        }
    }
    workgroupBarrier();

    if (lane < hd) {
        let out_base = (abs_row * d.num_heads + head) * hd;
        out[out_base + lane] = orot[lane];
    }
}
