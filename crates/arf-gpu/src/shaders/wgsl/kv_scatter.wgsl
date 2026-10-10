// KV scatter: write this step's new keys/values into the physical block pool at
// the slots given by the block table. Matches PagedKvCache::write.
//
// One invocation per float, indexed over q_len * row_floats. For global id i:
//   position = i / row_floats   (which query token)
//   col      = i % row_floats   (column within the kv row = kv_heads*head_dim)
//   phys_row = slots[position]  (physical slot for that position)
//   pool[phys_row * row_floats + col] = new[i]

// `src_row` lets the new K/V be read from a row offset within a larger batched
// buffer (whole-buffer binding avoids storage offset-alignment limits).
struct Dims { q_len: u32, row_floats: u32, src_row: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read>       new_k: array<f32>;
@group(0) @binding(1) var<storage, read>       new_v: array<f32>;
@group(0) @binding(2) var<storage, read>       slots: array<u32>;
@group(0) @binding(3) var<storage, read_write> key_pool: array<f32>;
@group(0) @binding(4) var<storage, read_write> value_pool: array<f32>;
@group(0) @binding(5) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= d.q_len * d.row_floats) {
        return;
    }
    let pos = i / d.row_floats;
    let col = i % d.row_floats;
    let pool_idx = slots[pos] * d.row_floats + col;
    let src = (d.src_row * d.row_floats) + i;
    key_pool[pool_idx] = new_k[src];
    value_pool[pool_idx] = new_v[src];
}
