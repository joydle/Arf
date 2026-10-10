// FLUX adaLN-Zero modulation. The modulation Linear produces all chunks (scale/shift/gate) in
// ONE [1, nchunk*hidden] buffer `params`; this kernel reads a chunk by its offset (so we never
// bind sub-ranges — sidesteps the 256B storage-offset alignment limit). Two modes:
//   mode 0 (modulate): out[t,c] = x[t,c] * (1 + params[scale_off + c]) + params[shift_off + c]
//   mode 1 (gate-add): x[t,c]   = x[t,c] + params[gate_off + c] * y[t,c]
// `scale_off`/`shift_off` (or `gate_off` in p1) are in ELEMENTS into `params`. col = i % hidden.

struct Dims { tokens: u32, hidden: u32, mode: u32, off_a: u32, off_b: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;       // modulate: in→out ; gate: x (in-place)
@group(0) @binding(1) var<storage, read>       y: array<f32>;       // modulate: dummy ; gate: y
@group(0) @binding(2) var<storage, read>       params: array<f32>;  // [1, nchunk*hidden]
@group(0) @binding(3) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
    let i = gid.y * (ng.x * 256u) + gid.x;
    let n = d.tokens * d.hidden;
    if (i < n) {
        let c = i % d.hidden;
        if (d.mode == 0u) {
            x[i] = x[i] * (1.0 + params[d.off_a + c]) + params[d.off_b + c];
        } else {
            x[i] = x[i] + params[d.off_a + c] * y[i];
        }
    }
}
