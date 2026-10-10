// Q4_K twin of embed_tok_bf16.wgsl: the source row comes from `tok[0]`.
struct Dims { hidden: u32, scale_bits: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       table: array<u32>; // Q4_K blocks [vocab, hidden/256]
@group(0) @binding(1) var<storage, read_write> out: array<f32>;   // hidden[0]
@group(0) @binding(2) var<storage, read>       tok: array<u32>;   // tok[0] = source row
@group(0) @binding(3) var<uniform>             d: Dims;

// Q4_K gather from the GGUF's own token-embedding blocks (144 bytes per 256 weights, read as
// u32 words; every block and row offset is 4-aligned). Element-for-element ggml
// `dequantize_row_q4_K` — see embed_tok_q4k in decode_ops_msl.metal (2026-09-23).
fn byte_at(idx: u32) -> u32 { return (table[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xffu; }
fn q4k_elem(base: u32, i: u32) -> f32 {
    let j = i / 32u;
    let l = i % 32u;
    let dm = unpack2x16float(table[base >> 2u]);
    let sc = base + 4u;
    var s: u32;
    var m: u32;
    if (j < 4u) {
        s = byte_at(sc + j) & 63u;
        m = byte_at(sc + j + 4u) & 63u;
    } else {
        s = (byte_at(sc + j + 4u) & 0xfu) | ((byte_at(sc + j - 4u) >> 6u) << 4u);
        m = (byte_at(sc + j + 4u) >> 4u) | ((byte_at(sc + j) >> 6u) << 4u);
    }
    let qb = byte_at(base + 16u + (j / 2u) * 32u + l);
    let q = select(qb & 0xfu, qb >> 4u, (j & 1u) == 1u);
    return dm.x * f32(s) * f32(q) - dm.y * f32(m);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < d.hidden) {
        let base = (tok[0] * (d.hidden / 256u) + i / 256u) * 144u;
        out[i] = q4k_elem(base, i % 256u) * bitcast<f32>(d.scale_bits);
    }
}
