// bf16 row gather: copy row `src_row` of a bf16-PACKED `table` ([_, hidden]) into
// row `dst_row` of the f32 `out`. Identical to embed.wgsl but the table is stored
// bf16 (two values per u32 word), widened to f32 on read.
//
// WHY bf16: a large vocab × wide hidden embedding (Gemma 4: 262144 × 5376) is
// 5.6 GB at f32 — over Metal's ~4 GB single-buffer-binding limit. Stored bf16 it is
// 2.8 GB (fits), and bf16 is the embedding's native checkpoint dtype anyway, so this
// is closer to the source than the f32 widening, not lossier.
//
// `scale_bits` multiplies the gathered row (Gemma scales embeddings by √hidden);
// pass 1.0 for models with no embedding scale — `x * 1.0` is bit-identical.
struct Dims { src_row: u32, hidden: u32, dst_row: u32, scale_bits: u32 };

@group(0) @binding(0) var<storage, read>       table: array<u32>; // bf16-packed [vocab, hidden]
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;

// The engine-standard bf16→f32 widen (matches matmul_vec.wgsl etc.).
fn bf16(bits: u32) -> f32 { return bitcast<f32>(bits << 16u); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < d.hidden) {
        // Element (src_row*hidden + i) lives in word j/2, low half if even else high.
        let elem = d.src_row * d.hidden + i;
        let word = table[elem >> 1u];
        let half = select(word >> 16u, word & 0xffffu, (elem & 1u) == 0u);
        out[d.dst_row * d.hidden + i] = bf16(half) * bitcast<f32>(d.scale_bits);
    }
}
