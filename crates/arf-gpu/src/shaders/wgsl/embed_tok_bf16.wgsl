// bf16 token embed with the source row taken from a resident buffer (`tok[0]`),
// not a uniform — the decode loop's on-GPU feedback path (see embed_tok.wgsl).
// Identical to embed_tok.wgsl but the `table` is bf16-packed (two values per u32),
// widened to f32 on read; see embed_bf16.wgsl for why the table is bf16.
//
// `scale_bits` multiplies the gathered row (Gemma scales embeddings by √hidden).
struct Dims { hidden: u32, scale_bits: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       table: array<u32>; // bf16-packed [vocab, hidden]
@group(0) @binding(1) var<storage, read_write> out: array<f32>;   // hidden[0]
@group(0) @binding(2) var<storage, read>       tok: array<u32>;   // tok[0] = source row
@group(0) @binding(3) var<uniform>             d: Dims;

fn bf16(bits: u32) -> f32 { return bitcast<f32>(bits << 16u); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < d.hidden) {
        let elem = tok[0] * d.hidden + i;
        let word = table[elem >> 1u];
        let half = select(word >> 16u, word & 0xffffu, (elem & 1u) == 0u);
        out[i] = bf16(half) * bitcast<f32>(d.scale_bits);
    }
}
