// TurboQuant KV scatter: quantize this step's new keys/values and write them
// PACKED into the physical pool at the slots given by the block table. The
// TurboQuant analogue of kv_scatter.wgsl — matches PagedKvCache::write under
// KvQuant::Tq (TqKvBlock::set_vector).
//
// One invocation per head-vector (q_len · kv_heads of them). For global id g:
//   pos     = g / kv_heads          (which new token)
//   kv_head = g % kv_heads          (which kv head within the row)
//   src     = (src_row + pos) · row_floats + kv_head · head_dim   (raw f32 vector)
//   vidx    = slots[pos] · kv_heads + kv_head                     (packed slot)
// Per vector: norm = ‖x‖ (stored f32), then quantize R·(x/norm) coordinate-wise
// (R = normalized Walsh-Hadamard) to `bits`-wide codes packed LSB-first into
// key_codes/value_codes at bit (vidx·head_dim + i)·bits.
//
// head_dim·bits is a multiple of 32 for every supported model (head_dim ∈
// {64,128,256}, bits ∈ {2,3,4}), so each vector owns a disjoint run of u32 words
// — distinct invocations never write the same word, so the clear-then-set
// read-modify-write below is race-free without atomics.

struct Dims {
    q_len: u32,
    kv_heads: u32,
    head_dim: u32,
    src_row: u32,
    bits: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read>       new_k: array<f32>;
@group(0) @binding(1) var<storage, read>       new_v: array<f32>;
@group(0) @binding(2) var<storage, read>       slots: array<u32>;
@group(0) @binding(3) var<storage, read_write> key_codes: array<u32>;
@group(0) @binding(4) var<storage, read_write> value_codes: array<u32>;
@group(0) @binding(5) var<storage, read_write> key_norms: array<f32>;
@group(0) @binding(6) var<storage, read_write> value_norms: array<f32>;
@group(0) @binding(7) var<storage, read>       levels: array<f32>;
@group(0) @binding(8) var<uniform>             d: Dims;

// Normalize + rotate `buf[0..n]` in place (R = normalized Walsh-Hadamard);
// returns ‖buf‖ (the pre-normalization norm). For a zero vector returns 0 and
// leaves buf untouched (its codes are ignored downstream).
fn prep(buf: ptr<function, array<f32, 256>>, n: u32) -> f32 {
    var s = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let x = (*buf)[i];
        s = s + x * x;
    }
    let norm = sqrt(s);
    if (norm <= 0.0) {
        return 0.0;
    }
    let inv = 1.0 / norm;
    for (var i = 0u; i < n; i = i + 1u) {
        (*buf)[i] = (*buf)[i] * inv;
    }
    // In-place normalized Walsh-Hadamard butterfly.
    var h = 1u;
    loop {
        if (h >= n) { break; }
        var i = 0u;
        loop {
            if (i >= n) { break; }
            var j = i;
            loop {
                if (j >= i + h) { break; }
                let a = (*buf)[j];
                let b = (*buf)[j + h];
                (*buf)[j] = a + b;
                (*buf)[j + h] = a - b;
                j = j + 1u;
            }
            i = i + h * 2u;
        }
        h = h * 2u;
    }
    let sc = 1.0 / sqrt(f32(n));
    for (var k = 0u; k < n; k = k + 1u) {
        (*buf)[k] = (*buf)[k] * sc;
    }
    return norm;
}

// Nearest codebook level index to `x` (levels sorted; linear scan, ≤16 levels).
fn nearest(x: f32, bits: u32) -> u32 {
    let nl = 1u << bits;
    var best = 0u;
    var bd = 1.0e30;
    var c = 0u;
    loop {
        if (c >= nl) { break; }
        let dd = abs(x - levels[c]);
        if (dd < bd) {
            bd = dd;
            best = c;
        }
        c = c + 1u;
    }
    return best;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let n_vecs = d.q_len * d.kv_heads;
    if (g >= n_vecs) {
        return;
    }
    let hd = d.head_dim;
    let bits = d.bits;
    let mask = (1u << bits) - 1u;
    let pos = g / d.kv_heads;
    let kv_head = g % d.kv_heads;
    let row_floats = d.kv_heads * hd;
    let src_base = (d.src_row + pos) * row_floats + kv_head * hd;
    let vidx = slots[pos] * d.kv_heads + kv_head;
    let base_bit = vidx * hd * bits;

    var buf: array<f32, 256>;

    // --- KEY ---
    for (var i = 0u; i < hd; i = i + 1u) {
        buf[i] = new_k[src_base + i];
    }
    let kn = prep(&buf, hd);
    key_norms[vidx] = kn;
    for (var i = 0u; i < hd; i = i + 1u) {
        var code = 0u;
        if (kn > 0.0) {
            code = nearest(buf[i], bits) & mask;
        }
        let bitpos = base_bit + i * bits;
        let word = bitpos / 32u;
        let off = bitpos % 32u;
        key_codes[word] = (key_codes[word] & ~(mask << off)) | (code << off);
        if (off + bits > 32u) {
            let hm = mask >> (32u - off);
            key_codes[word + 1u] = (key_codes[word + 1u] & ~hm) | (code >> (32u - off));
        }
    }

    // --- VALUE ---
    for (var i = 0u; i < hd; i = i + 1u) {
        buf[i] = new_v[src_base + i];
    }
    let vn = prep(&buf, hd);
    value_norms[vidx] = vn;
    for (var i = 0u; i < hd; i = i + 1u) {
        var code = 0u;
        if (vn > 0.0) {
            code = nearest(buf[i], bits) & mask;
        }
        let bitpos = base_bit + i * bits;
        let word = bitpos / 32u;
        let off = bitpos % 32u;
        value_codes[word] = (value_codes[word] & ~(mask << off)) | (code << off);
        if (off + bits > 32u) {
            let hm = mask >> (32u - off);
            value_codes[word + 1u] = (value_codes[word + 1u] & ~hm) | (code >> (32u - off));
        }
    }
}
