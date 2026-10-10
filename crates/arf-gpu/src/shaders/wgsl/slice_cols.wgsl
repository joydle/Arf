// Move a column-block between a row-major matrix `[rows, stride]` and a dense `[*, width]`.
// A fused projection writes `[rows, stride]` row-major (qkv = [tokens, 3*h], or qkv+mlp =
// [tokens, 3*h+mlp]); the q/k/v/mlp parts are COLUMN slices of each row, NOT contiguous
// row-blocks. Two modes (so one kernel does both the split and the re-concat for linear2):
//
//   mode 0 (gather):  dense[(dst_tok+r)*width + c] = mat[r*stride + col_off + c]
//   mode 1 (scatter): mat[r*stride + col_off + c]  = dense[(dst_tok+r)*width + c]
//
// `dst_tok` ALWAYS offsets the dense side (e.g. scatter a per-stream v slice straight into the
// joint vj buffer at its token offset). `a` is always the [rows,stride] matrix side, `b` the
// dense side.

struct Dims { rows: u32, stride: u32, col_off: u32, width: u32, dst_tok: u32, mode: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read_write> a: array<f32>;   // [rows, stride] matrix side
@group(0) @binding(1) var<storage, read_write> b: array<f32>;   // dense [dst_tok+rows, width] side
@group(0) @binding(2) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 256u) + gid.x;       // 2D-safe flat index (n can exceed 65535*256)
    let n = d.rows * d.width;
    if (i >= n) { return; }
    let r = i / d.width;
    let c = i % d.width;
    let mat_idx = r * d.stride + d.col_off + c;
    let dense_idx = (d.dst_tok + r) * d.width + c;
    if (d.mode == 0u) {
        b[dense_idx] = a[mat_idx];               // gather column-block → dense (at dst_tok)
    } else {
        a[mat_idx] = b[dense_idx];               // scatter dense (at dst_tok) → column-block
    }
}
