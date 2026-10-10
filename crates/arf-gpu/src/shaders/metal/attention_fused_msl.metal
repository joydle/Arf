// MSL decode attention WITH the new-token KV scatter FOLDED IN (kv_scatter + attention → 1
// dispatch) for the megakernel island — saves a dispatch + a barrier per layer (the ggml-metal
// flash path reads the cache directly; this is the equivalent fold). Bit-identical to running
// kv_scatter then attention_decode.
//
// THE FOLD IS SAFE despite no cross-threadgroup sync: each head-threadgroup writes ITS OWN
// kv_head's new K/V row into the pool at the new slot, then reads only the slots it (or a
// sibling head in the same GQA group writing the IDENTICAL bytes) produced. Heads sharing a
// kv_head write the same value to the same slot → idempotent, not a race. A local
// threadgroup_barrier orders the write before this threadgroup's reads. No threadgroup ever
// reads a slot another threadgroup is concurrently writing to a DIFFERENT value.
//
// Bindings: 0 q, 1 key_pool, 2 value_pool, 3 out, 4 slots, 5 dims, 6 new_k, 7 new_v, 8 new_slot
// (new_slot[0] = this token's pool slot; new_k/new_v = kv_dim floats for the new token).

#include <metal_stdlib>
using namespace metal;

struct Dims {
    uint q_len; uint ctx; uint num_heads; uint kv_heads; uint head_dim; uint past_len;
    uint scale_bits; uint group; uint q_start; uint window; uint _p1; uint _p2;
};

constant constexpr uint TG = 256u;
constant constexpr uint TILE = 1024u;

kernel void attention_decode_fused(
        device const float *q          [[buffer(0)]],
        device       float *key_pool   [[buffer(1)]],
        device       float *value_pool [[buffer(2)]],
        device       float *out        [[buffer(3)]],
        device const uint  *slots      [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const float *new_k      [[buffer(6)]],
        device const float *new_v      [[buffer(7)]],
        device const uint  *new_slot   [[buffer(8)]],
        uint  head                     [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_position_in_threadgroup]]) {
    const uint hd = d.head_dim;
    const uint kv_head = head / d.group;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_floats = d.kv_heads * hd;
    const uint head_col = kv_head * hd;
    const uint q_base = (d.q_start * d.num_heads + head) * hd;

    // --- A0 (FOLDED SCATTER): write this token's K/V for THIS kv_head into the pool slot.
    // new_k/new_v are laid out [kv_head*hd + i]; pool row at new_slot[0]. Each head-threadgroup
    // writes its own kv_head's hd floats (idempotent across the GQA group), then barriers.
    const uint pslot = new_slot[0] * row_floats + head_col;
    for (uint i = lane; i < hd; i += TG) {
        key_pool[pslot + i]   = new_k[head_col + i];
        value_pool[pslot + i] = new_v[head_col + i];
    }
    threadgroup_barrier(mem_flags::mem_device); // device scope: pool write visible to this TG's reads

    const uint last = d.past_len;
    uint lo = 0u;
    if (d.window > 0u && last + 1u > d.window) lo = last + 1u - d.window;

    threadgroup float scores[TILE];
    threadgroup float red[TG];
    threadgroup float m_run; threadgroup float l_run; threadgroup float corr;

    const uint DPL = (hd + TG - 1u) / TG;
    float acc[2] = {0.0f, 0.0f};
    if (lane == 0) { m_run = -3.4e38f; l_run = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile0 = lo;
    while (tile0 <= last) {
        const uint tile_end = min(tile0 + TILE - 1u, last);
        const uint tile_len = tile_end - tile0 + 1u;

        for (uint t = lane; t < tile_len; t += TG) {
            const uint kb = slots[tile0 + t] * row_floats + head_col;
            float s = 0.0f;
            for (uint i = 0; i < hd; ++i) s += q[q_base + i] * key_pool[kb + i];
            scores[t] = s * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float lmax = -3.4e38f;
        for (uint c = lane; c < tile_len; c += TG) lmax = max(lmax, scores[c]);
        red[lane] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG / 2u; st > 0u; st >>= 1) {
            if (lane < st) red[lane] = max(red[lane], red[lane + st]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float m_new = max(m_run, red[0]);
            corr = exp(m_run - m_new);
            m_run = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float lsum = 0.0f;
        for (uint e = lane; e < tile_len; e += TG) {
            const float ex = exp(scores[e] - m_run);
            scores[e] = ex;
            lsum += ex;
        }
        red[lane] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG / 2u; st > 0u; st >>= 1) {
            if (lane < st) red[lane] = red[lane] + red[lane + st];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) l_run = l_run * corr + red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < DPL; ++j) {
            const uint dim = lane + j * TG;
            if (dim < hd) {
                float add = 0.0f;
                for (uint tt = 0; tt < tile_len; ++tt) {
                    add += scores[tt] * value_pool[slots[tile0 + tt] * row_floats + head_col + dim];
                }
                acc[j] = acc[j] * corr + add;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile0 += TILE;
    }

    const uint out_base = (d.q_start * d.num_heads + head) * hd;
    for (uint j = 0; j < DPL; ++j) {
        const uint dim = lane + j * TG;
        if (dim < hd) out[out_base + dim] = acc[j] / l_run;
    }
}
