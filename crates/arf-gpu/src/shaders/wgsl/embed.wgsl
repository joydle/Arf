// Row gather: copy row `src_row` of `table` ([_, hidden]) into row `dst_row` of
// `out`. Used to embed a token (table = embedding) and to gather a sequence's
// last hidden row (table = hidden states) — same operation, both directions.

// `scale_bits` multiplies the gathered row (Gemma scales embeddings by √hidden);
// pass 1.0 for models with no embedding scale — `x * 1.0` is bit-identical.
struct Dims { src_row: u32, hidden: u32, dst_row: u32, scale_bits: u32 };

@group(0) @binding(0) var<storage, read>       table: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < d.hidden) {
        out[d.dst_row * d.hidden + i] = table[d.src_row * d.hidden + i] * bitcast<f32>(d.scale_bits);
    }
}
