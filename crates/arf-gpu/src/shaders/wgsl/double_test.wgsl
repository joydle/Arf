// Minimal test fixture: out[i] = 2 * in[i]. Used by the GPU-foundation and
// CommandPass smoke tests (tests/gpu.rs) to exercise dispatch + one-submit
// chaining without depending on a real model kernel.

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write>  out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&inp)) {
        return;
    }
    out[i] = 2.0 * inp[i];
}
