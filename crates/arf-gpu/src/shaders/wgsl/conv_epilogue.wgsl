// Conv epilogue: the im2col→matmul step produces `mm[out_pixels, c_out]` (pixel-major, the
// natural matmul output). The rest of the VAE keeps activations CHANNEL-major `[c_out, H, W]`
// (so the next im2col reads contiguous per-channel planes). This kernel transposes mm into
// channel-major AND adds the per-channel conv bias in one pass:
//     out[c, p] = mm[p, c] + bias[c]          (p over out_pixels, c over c_out)
// One invocation per output element. 2D-safe flat index.
//
// TILED: `mm` holds only THIS tile's `[tile_pixels, c_out]` (tile_pixels = out_pixels here), but
// `out` is the FULL `[c_out, full_pixels]` plane. Scatter each tile pixel into its absolute
// column `pixel_off + p`, so the conv can be streamed row-band by row-band without ever
// materialising the full im2col `cols`. Non-tiled callers pass pixel_off=0, full_pixels=out_pixels.

struct Dims { out_pixels: u32, c_out: u32, pixel_off: u32, full_pixels: u32 };

@group(0) @binding(0) var<storage, read>       mm: array<f32>;    // [tile_pixels, c_out]
@group(0) @binding(1) var<storage, read>       bias: array<f32>;  // [c_out]
@group(0) @binding(2) var<storage, read_write> out: array<f32>;   // [c_out, full_pixels]
@group(0) @binding(3) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 256u) + gid.x;
    let n = d.out_pixels * d.c_out;      // out_pixels = this tile's pixel count
    if (i >= n) { return; }
    let c = i / d.out_pixels;            // (c, p_local) within the tile's mm
    let p = i % d.out_pixels;
    out[c * d.full_pixels + d.pixel_off + p] = mm[p * d.c_out + c] + bias[c];
}
