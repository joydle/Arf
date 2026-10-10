// im2col for a 2D conv (stride 1, symmetric padding `pad`). Lowers a conv into a matmul:
// for an input `[C_in, H, W]` (channel-major, row-major per channel) and a `KH×KW` kernel, build
// the patch matrix `cols[out_pixels, C_in*KH*KW]` so that
//     conv_out = cols · Wᵀ     with W the resident weight `[C_out, C_in*KH*KW]` (bf16, matmul.wgsl)
// out_pixels = H_out*W_out (= H*W for stride-1 'same' padding). Column order within a patch is
// (c_in, ky, kx) with kx fastest — this MUST match how the conv weight `[C_out,C_in,KH,KW]` is
// flattened to `[C_out, C_in*KH*KW]` on load (also c_in,ky,kx with kx fastest → row-major).
//
// One invocation per output element of `cols` (tile_rows*W_out * C_in*KH*KW). Pixels outside the
// padded border read 0. 2D-safe flat index (the patch matrix is large: 16384*4608 at 128² mid).
//
// TILED: `h_out` is the number of output ROWS in THIS tile and `row_off` the first output row
// of the tile, so the conv can be streamed row-band by row-band — bounding `cols` to
// tile_rows*W_out*patch instead of the full H*W*patch (the multi-GB VAE im2col spike at 1024px).
// The input `inp` stays fully resident, so a tile's patches read across its band boundary freely.

struct Dims { c_in: u32, h: u32, w: u32, kh: u32, kw: u32, pad: u32, h_out: u32, w_out: u32, row_off: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;   // [c_in, h, w]
@group(0) @binding(1) var<storage, read_write> cols: array<f32>;  // [h_out*w_out, c_in*kh*kw]
@group(0) @binding(2) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 256u) + gid.x;
    let pwidth = d.c_in * d.kh * d.kw;       // columns per output pixel
    let tile_pixels = d.h_out * d.w_out;     // h_out = rows in THIS tile
    let n = tile_pixels * pwidth;
    if (i >= n) { return; }

    let pix = i / pwidth;                     // which output pixel WITHIN the tile
    let col = i % pwidth;                     // which (c_in,ky,kx) within the patch
    let oy = pix / d.w_out + d.row_off;       // absolute output row
    let ox = pix % d.w_out;
    // decode col -> (c_in, ky, kx), kx fastest
    let kx = col % d.kw;
    let ky = (col / d.kw) % d.kh;
    let ci = col / (d.kh * d.kw);
    // input coordinate (stride 1): iy = oy + ky - pad
    let iy = i32(oy) + i32(ky) - i32(d.pad);
    let ix = i32(ox) + i32(kx) - i32(d.pad);
    var v = 0.0;
    if (iy >= 0 && iy < i32(d.h) && ix >= 0 && ix < i32(d.w)) {
        v = inp[ci * d.h * d.w + u32(iy) * d.w + u32(ix)];
    }
    cols[i] = v;
}
