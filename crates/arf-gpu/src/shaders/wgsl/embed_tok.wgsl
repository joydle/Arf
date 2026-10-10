// Token embed with the source row taken from a resident buffer, not a uniform.
//
// The decode loop's on-GPU feedback path: the sample kernel writes the just-
// emitted token id into `tok[0]`, and this kernel embeds that row — so the
// sampled token feeds straight into the next step's embed with no CPU round-trip
// between tokens. Otherwise identical to embed.wgsl (a row gather into out[0]).

// `scale_bits` multiplies the gathered row (Gemma scales embeddings by √hidden);
// pass 1.0 for models with no embedding scale — `x * 1.0` is bit-identical.
struct Dims { hidden: u32, scale_bits: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       table: array<f32>; // embedding [vocab, hidden]
@group(0) @binding(1) var<storage, read_write> out: array<f32>;   // hidden[0]
@group(0) @binding(2) var<storage, read>       tok: array<u32>;   // tok[0] = source row
@group(0) @binding(3) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < d.hidden) {
        out[i] = table[tok[0] * d.hidden + i] * bitcast<f32>(d.scale_bits);
    }
}
