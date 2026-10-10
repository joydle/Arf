// Nearest-neighbor 2× upsample of channel-major `[C, H, W]` → `[C, 2H, 2W]`.
// out[c, oy, ox] = in[c, oy/2, ox/2]   (each input pixel replicated into a 2×2 block).
// The VAE Upsample2D follows this with a 3×3 conv (a separate im2col→matmul step). One
// invocation per OUTPUT element; 2D-safe flat index (output can be 512*256*256 = 33M).

struct Dims { c: u32, h: u32, w: u32, _p: u32 };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;   // [c, h, w]
@group(0) @binding(1) var<storage, read_write> out: array<f32>;   // [c, 2h, 2w]
@group(0) @binding(2) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 256u) + gid.x;
    let oh = d.h * 2u;
    let ow = d.w * 2u;
    let n = d.c * oh * ow;
    if (i >= n) { return; }
    let ch = i / (oh * ow);
    let rem = i % (oh * ow);
    let oy = rem / ow;
    let ox = rem % ow;
    let iy = oy / 2u;
    let ix = ox / 2u;
    out[i] = inp[ch * d.h * d.w + iy * d.w + ix];
}
