// MSL decode attention (q_len == 1) for the per-token island megakernel — flash-style
// online softmax over the paged KV pool. Faithful port of attention.wgsl's tile structure
// for the single-query decode case (the proven-correct shape): one threadgroup per head,
// streams keys in tiles, keeps a running max m_run / denominator l_run (flash rescale),
// each lane owns the output dims {lane, lane+TG, ...}. Reductions via threadgroup tree
// (same as WGSL) to keep parity exact. GQA: kv_head = head/group. Causal window supported.
//
// Bindings match attention.wgsl: 0 q, 1 key_pool, 2 value_pool, 3 out, 4 slots, 5 dims.

#include <metal_stdlib>
using namespace metal;

struct Dims {
    uint q_len; uint ctx; uint num_heads; uint kv_heads; uint head_dim; uint past_len;
    uint scale_bits; uint group; uint q_start; uint window; uint _p1; uint _p2;
};

// ARF_ATTN_VEC4 gate (function_constant 0, default OFF → scalar path, byte-identical when the
// constant is unset since compile_msl passes no function-constant values). When true, the A1 QK
// dot reads via float4 loads and accumulates 4 explicit fmas PER GROUP-OF-4 in the SAME order
// (i, i+1, i+2, i+3) as the scalar loop — bit-exact by construction (identical op DAG; only the
// memory access is vectorized). head_dim is always a multiple of 4 (gemma 256/512, qwen 128), and
// q_base/kb are multiples of head_dim (>=4 floats = 16B) off 256B-aligned buffers → float4-aligned.
constant bool ATTN_VEC4_fc [[function_constant(0)]];
constant bool ATTN_VEC4 = is_function_constant_defined(ATTN_VEC4_fc) ? ATTN_VEC4_fc : false;

// Bit-exact QK dot: float4 group-of-4 fmas (i,i+1,i+2,i+3) when ATTN_VEC4, else the scalar loop.
inline float qk_dot(device const float *q, uint q_base, device const float *kp, uint kb, uint hd) {
    float s = 0.0f;
    if (ATTN_VEC4) {
        device const float4 *q4 = reinterpret_cast<device const float4 *>(q + q_base);
        device const float4 *k4 = reinterpret_cast<device const float4 *>(kp + kb);
        const uint hd4 = hd >> 2;
        for (uint i = 0; i < hd4; ++i) {
            const float4 qv = q4[i]; const float4 kv = k4[i];
            s += qv.x * kv.x; s += qv.y * kv.y; s += qv.z * kv.z; s += qv.w * kv.w;
        }
    } else {
        for (uint i = 0; i < hd; ++i) s += q[q_base + i] * kp[kb + i];
    }
    return s;
}

constant constexpr uint TG = 256u;     // threads per threadgroup (matches WGSL WG)
constant constexpr uint TILE = 1024u;  // keys per tile (decode ctx usually fits a few tiles)
constant constexpr uint TILE_S = 16u;  // GQA head-group kernel staged-KV tile (must fit 32KB tg-mem)

kernel void attention_decode(
        device const float *q          [[buffer(0)]],
        device const float *key_pool   [[buffer(1)]],
        device const float *value_pool [[buffer(2)]],
        device       float *out        [[buffer(3)]],
        device const uint  *slots      [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        uint  head                     [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_position_in_threadgroup]]) {
    const uint hd = d.head_dim;
    const uint kv_head = head / d.group;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_floats = d.kv_heads * hd;
    const uint head_col = kv_head * hd;
    const uint q_base = (d.q_start * d.num_heads + head) * hd;

    const uint last = d.past_len;            // decode: q_len==1, row 0
    uint lo = 0u;
    if (d.window > 0u && last + 1u > d.window) lo = last + 1u - d.window;

    threadgroup float scores[TILE];
    threadgroup float red[TG];
    threadgroup float m_run; threadgroup float l_run; threadgroup float corr;

    const uint DPL = (hd + TG - 1u) / TG;    // output dims per lane (1 for ≤256, 2 for 512)
    float acc[2] = {0.0f, 0.0f};
    if (lane == 0) { m_run = -3.4e38f; l_run = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile0 = lo;
    while (tile0 <= last) {
        const uint tile_end = min(tile0 + TILE - 1u, last);
        const uint tile_len = tile_end - tile0 + 1u;

        // A1: scores[t] = scale·Q·K[tile0+t], one key per lane (strided).
        for (uint t = lane; t < tile_len; t += TG) {
            const uint kb = slots[tile0 + t] * row_floats + head_col;
            float s = 0.0f;
            s += qk_dot(q, q_base, key_pool, kb, hd);
            scores[t] = s * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A2: tile max via tree reduction.
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

        // A3: scores = exp(score − m_run); tile sum.
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

        // B: fold tile into per-dim accumulators (each lane owns DPL strided dims).
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

// ============================================================================================
// B-ROW DECODE ATTENTION — collapses the per-sequence dispatch loop into ONE dispatch.
//
// The per-seq megakernel did `for r in 0..b { dispatchThreadgroups(grid=nh) }` (B×nh threadgroups
// in B separate dispatches). This kernel uses a 2D grid (head, seq) = (nh, B): one dispatch, B×nh
// threadgroups. grid.x = head, grid.y = seq r. Each threadgroup computes the IDENTICAL math the
// m=1 attention_decode does for seq r — same flash online-softmax, same reductions — so the output
// is BIT-IDENTICAL to the per-seq loop (only the dispatch packaging differs).
//
// PER-SEQ data is read via the seq index r = grid.y:
//   q     : q[r*q_dim + head*hd + i]                    (q laid out [B, nh*hd] row-major)
//   out   : out[r*q_dim + head*hd + dim]
//   slots : slots_all[r*max_slots + key_idx]            (one [B, max_slots] buffer, padded)
//   past_len_b[r], q_start UNUSED (decode q_start=0 baked into r*q_dim offset)
// UNIFORM data (same model, same step) comes from `d`: nh, nkv, hd, scale, group, window, max_slots.
//
// Bindings: 0 q, 1 key_pool, 2 value_pool, 3 out, 4 slots_all, 5 dims_b, 6 past_len_b.
// dims_b reuses the Dims struct but: q_len=1, num_heads=nh, kv_heads=nkv, head_dim=hd,
// scale_bits, group, window in their usual slots; q_start REPURPOSED as max_slots (the per-seq
// slot-table stride). past_len is NOT read from `d` — it comes from past_len_b[r].

kernel void attention_decode_b(
        device const float *q          [[buffer(0)]],
        device const float *key_pool   [[buffer(1)]],
        device const float *value_pool [[buffer(2)]],
        device       float *out        [[buffer(3)]],
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]]) {
    const uint head = gid.x;        // head index (0..nh)
    const uint r    = gid.y;        // sequence index (0..B)
    const uint hd = d.head_dim;
    const uint kv_head = head / d.group;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_floats = d.kv_heads * hd;
    const uint head_col = kv_head * hd;
    const uint max_slots = d.q_start;            // q_start REPURPOSED as the slot-table stride
    const uint q_base = (r * d.num_heads + head) * hd;   // q[r*q_dim + head*hd]
    device const uint *slots = slots_all + r * max_slots; // this seq's slot table

    const uint last = past_len_b[r];             // this seq's OWN past_len (decode: q_len==1, row 0)
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
            s += qk_dot(q, q_base, key_pool, kb, hd);
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

    const uint out_base = (r * d.num_heads + head) * hd;
    for (uint j = 0; j < DPL; ++j) {
        const uint dim = lane + j * TG;
        if (dim < hd) out[out_base + dim] = acc[j] / l_run;
    }
}

// ============================================================================================
// SPLIT-KV DECODE ATTENTION (ARF_ATTN_SPLITK) — flatten the KV-growth ramp.
//
// attention_decode_b walks the WHOLE growing KV in one threadgroup per (head,seq): cost is
// O(past_len), so per-token latency ramps as context grows (the conc64 median-vs-best gap).
// Split-KV fans the KV range across `splitk` work-groups (grid dim z), each doing an
// UN-normalized online-softmax over its own chunk and writing a partial (acc[hd], m, l). A
// second `combine` pass merges the partials with the standard flash rescale. Each chunk is
// ~past_len/splitk keys → the critical path stops growing, flattening the ramp.
//
// IDENTITY at splitk==1: one chunk covers [lo,last] exactly like _b, and combine with one
// partial reduces to out = acc / l — bit-for-bit the _b output. So the two-pass plumbing is
// parity-testable BEFORE the fan-out is enabled (pin splitk=1, expect max_abs 0).
//
// Partials layout: partials[((r*nh + head)*splitk + sp) * (hd+2) + k], k in [0,hd) = acc,
//   k==hd = m (running max), k==hd+1 = l (running denom).
// Bindings pass1 (partials): 0..6 IDENTICAL to attention_decode_b, + 7 = partials (write).
// d._p1 REPURPOSED as `splitk` (the chunk count).

kernel void attention_decode_splitk_b(
        device const float *q          [[buffer(0)]],
        device const float *key_pool   [[buffer(1)]],
        device const float *value_pool [[buffer(2)]],
        device       float *out        [[buffer(3)]],   // unused in pass1 (combine writes out)
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        device       float *partials   [[buffer(7)]],
        uint3 gid                      [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]]) {
    const uint head = gid.x;
    const uint r    = gid.y;
    const uint sp   = gid.z;                     // this work-group's KV chunk index
    const uint splitk = max(d._p1, 1u);
    const uint hd = d.head_dim;
    const uint kv_head = head / d.group;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_floats = d.kv_heads * hd;
    const uint head_col = kv_head * hd;
    const uint max_slots = d.q_start;
    const uint q_base = (r * d.num_heads + head) * hd;
    device const uint *slots = slots_all + r * max_slots;

    const uint last = past_len_b[r];
    uint lo = 0u;
    if (d.window > 0u && last + 1u > d.window) lo = last + 1u - d.window;

    // This chunk's key range [s_lo, s_hi) within [lo, last]. Even split, remainder to the last.
    const uint total = last + 1u - lo;
    const uint chunk = (total + splitk - 1u) / splitk;
    const uint s_lo = lo + sp * chunk;
    const uint s_hi = min(s_lo + chunk, last + 1u);

    const uint pbase = ((r * d.num_heads + head) * splitk + sp) * (hd + 2u);
    const uint DPL = (hd + TG - 1u) / TG;

    // Empty chunk (splitk > keys): write a neutral partial (m=-inf, l=0, acc=0) so combine skips it.
    if (s_lo >= s_hi) {
        for (uint j = 0; j < DPL; ++j) { const uint dim = lane + j * TG; if (dim < hd) partials[pbase + dim] = 0.0f; }
        if (lane == 0) { partials[pbase + hd] = -3.4e38f; partials[pbase + hd + 1u] = 0.0f; }
        return;
    }

    threadgroup float scores[TILE];
    threadgroup float red[TG];
    threadgroup float m_run; threadgroup float l_run; threadgroup float corr;
    float acc[2] = {0.0f, 0.0f};
    if (lane == 0) { m_run = -3.4e38f; l_run = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile0 = s_lo;
    while (tile0 < s_hi) {
        const uint tile_end = min(tile0 + TILE, s_hi);   // exclusive
        const uint tile_len = tile_end - tile0;

        for (uint t = lane; t < tile_len; t += TG) {
            const uint kb = slots[tile0 + t] * row_floats + head_col;
            scores[t] = qk_dot(q, q_base, key_pool, kb, hd) * scale;
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
        if (lane == 0) { const float m_new = max(m_run, red[0]); corr = exp(m_run - m_new); m_run = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float lsum = 0.0f;
        for (uint e = lane; e < tile_len; e += TG) { const float ex = exp(scores[e] - m_run); scores[e] = ex; lsum += ex; }
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

    // Write the UN-normalized partial: acc (not divided by l), plus m and l for the combine.
    for (uint j = 0; j < DPL; ++j) { const uint dim = lane + j * TG; if (dim < hd) partials[pbase + dim] = acc[j]; }
    if (lane == 0) { partials[pbase + hd] = m_run; partials[pbase + hd + 1u] = l_run; }
}

// Merge the per-chunk partials into the final attention output (the flash rescale across chunks).
// Grid (nh, B), tg256. Bindings: 0 partials(read), 1 out(write), 2 dims_b, 3 past_len_b (unused).
kernel void attention_splitk_combine_b(
        device const float *partials   [[buffer(0)]],
        device       float *out        [[buffer(1)]],
        constant     Dims  &d          [[buffer(2)]],
        device const uint  *past_len_b [[buffer(3)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]]) {
    const uint head = gid.x;
    const uint r    = gid.y;
    const uint splitk = max(d._p1, 1u);
    const uint hd = d.head_dim;
    const uint DPL = (hd + TG - 1u) / TG;

    threadgroup float mg; threadgroup float lg;

    // Global max across chunks, then the rescaled global denom.
    if (lane == 0) {
        float m = -3.4e38f;
        for (uint s = 0; s < splitk; ++s) m = max(m, partials[((r * d.num_heads + head) * splitk + s) * (hd + 2u) + hd]);
        float l = 0.0f;
        for (uint s = 0; s < splitk; ++s) {
            const uint pb = ((r * d.num_heads + head) * splitk + s) * (hd + 2u);
            l += exp(partials[pb + hd] - m) * partials[pb + hd + 1u];
        }
        mg = m; lg = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out_base = (r * d.num_heads + head) * hd;
    for (uint j = 0; j < DPL; ++j) {
        const uint dim = lane + j * TG;
        if (dim < hd) {
            float num = 0.0f;
            for (uint s = 0; s < splitk; ++s) {
                const uint pb = ((r * d.num_heads + head) * splitk + s) * (hd + 2u);
                num += exp(partials[pb + hd] - mg) * partials[pb + dim];
            }
            out[out_base + dim] = (lg > 0.0f) ? (num / lg) : 0.0f;
        }
    }
}

// ============================================================================================
// B-ROW KV SCATTER — collapses the per-sequence scatter loop into ONE dispatch.
//
// The per-seq megakernel did `for r in 0..b { dispatchThreadgroups(grid=kv_dim/256) }`. This uses
// a 2D grid (col, seq): grid.x covers the kv_dim columns, grid.y = seq r. Each (col,seq) thread
// writes seq r's new K/V (read from new_k[r*row_floats + col]) into the shared pool at the row
// write_slot_b[r]. BIT-IDENTICAL to the per-seq scatter (same write per element).
//
// Bindings: 0 new_k [B,row_floats], 1 new_v [B,row_floats], 2 write_slot_b [B] (per-seq pool row),
//           3 key_pool, 4 value_pool, 5 dims {q_len(unused), row_floats, B, _}.
struct ScatterDimsB { uint q_len; uint row_floats; uint b; uint _p; };

kernel void kv_scatter_b(
        device const float *new_k        [[buffer(0)]],
        device const float *new_v        [[buffer(1)]],
        device const uint  *write_slot_b [[buffer(2)]],
        device       float *key_pool     [[buffer(3)]],
        device       float *value_pool   [[buffer(4)]],
        constant ScatterDimsB &d         [[buffer(5)]],
        uint2 gid                        [[thread_position_in_grid]]) {
    const uint col = gid.x;          // column within the kv row (0..row_floats)
    const uint r   = gid.y;          // sequence index (0..B)
    if (col >= d.row_floats || r >= d.b) return;
    const uint pool_idx = write_slot_b[r] * d.row_floats + col;
    const uint src = r * d.row_floats + col;
    key_pool[pool_idx]   = new_k[src];
    value_pool[pool_idx] = new_v[src];
}

// ============================================================================================
// SHARED-PREFIX VERIFY ATTENTION — the MTP / speculative-decode verify pass, prefix staged ONCE.
//
// The verify shape: ONE sequence, `k` query rows (window = [committed, draft_0..draft_{k-1}]), all
// sharing the SAME prefix slot table [0..prefix_len). Row i's query position is prefix_len+i, and
// its causal frontier is prefix_len+i — it attends the whole shared prefix [0, prefix_len) PLUS the
// drafted rows 0..i (their freshly-scattered K/V, at slots[prefix_len+0..prefix_len+i]) PLUS itself.
//
// THE INEFFICIENCY IT KILLS: attention_decode_b / attention_batched run one threadgroup per
// (head, query row); every one of the k rows RE-STREAMS the entire prefix KV from the pool
// independently (k× prefix DRAM traffic). Here ONE threadgroup per head streams the shared prefix
// [0, prefix_len) EXACTLY ONCE — for each prefix tile, cooperatively stage K (and, in the fold,
// V) into threadgroup memory, then let all k queries consume that staged tile. Prefix KV DRAM
// traffic drops k× → 1×. The k×k causal window [prefix_len, prefix_len+k) is done per-row (small).
//
// NUMERICS: EXACT online-softmax rescale of attention_decode_b — per query row i keeps m_run[i]/
// l_run[i]/acc, m_new/corr per tile, l_run = l_run*corr + tile_sum. The prefix and the window are
// two consecutive tile-groups fed through the SAME flash accumulator, in the SAME key order
// (ascending position), so the result is bit-associated identically to the per-row oracle
// (attention_batched.wgsl with row_meta[row].last = prefix_len+i). f32 throughout.
//
// Bindings (byte-identical shape to attention_decode_b): 0 q [k, nh*hd], 1 key_pool, 2 value_pool,
// 3 out [k, nh*hd], 4 slots [prefix_len+k] (the ONE seq's flat slot table: prefix ++ drafted write
// slots), 5 dims. Dims repurposing: q_start = k (# verify rows); past_len = prefix_len; window @9
// (0 = global/qwen-coder; sliding-window deferred, guarded below); num_heads/kv_heads/head_dim/
// scale_bits/group in their usual slots. Grid: (nh, 1), 256 threads. head_dim <= 256 (DPL 1) —
// qwen3-coder hd=128; a >256 head would need DPL 2 (add a slot, as attention_decode does).
//
// KMAX caps the verify window width the threadgroup arrays hold; k must be <= KMAX (host gates it).

constant constexpr uint KMAX = 32u;      // max verify rows (k). Was 16 (spec windows are tiny);
// widened for PREFILL windows (attack plan L1): each window streams the FULL active weights
// once, so window WIDTH sets prefill cost. tg-mem +1.2KB (checked, ~19.5KB of 32KB); the real
// risk is acc[KMAX][DPL] register pressure — MEASURED via the full conc curve, not assumed.
constant constexpr uint TILE_V = 16u;    // shared-prefix staged tile (16*128*4*2 = 16KB tg-mem, hd<=128)
// 🔴 TWO OVERRUNS FIXED 2026-09-20 — and the first is the 2026-08-30 bug ("with the shared-prefix
// kernel both rows lose the prompt", which is why it has been switched off on the 27B ever since):
//  1. `ksh`/`vsh` hold TILE_V*128 floats but are indexed by the REAL head_dim. Qwen3.8-27B has
//     head_dim 256, so every staged tile wrote past both arrays into the softmax state beside them.
//     The kernel was written for qwen3-coder (hd=128), where it fits exactly. The staged tile is now
//     `tile_v` KEYS long — TILE_V at hd<=128, 8 at hd=256 — so it always fits the same memory.
//  2. `scores` was [KMAX][TILE_V] = 32x16, but the WINDOW phase writes up to k <= KMAX entries a
//     row; the comment there claimed KMAX <= TILE_V, untrue since KMAX went 16 -> 32. Harmless for
//     2-4-row spec windows, an overrun for anything wider. Now [KMAX][KMAX].
//
// MEASURED-OUT THE SAME DAY — THIS KERNEL FOR PREFILL WINDOWS. With both fixes in, a 128-row
// prefill window was cut into 32-row sub-windows and run through it (correct: recall 3/3, text
// identical to the per-row path). Time to first token on the 27B, per-row attention vs this:
// 1,829 tokens 14.0 s vs 31.5 s; 5,880 tokens 52.3 s vs 209.7 s; 11,730 tokens 136.9 s vs
// 750.9 s. It is 2-5x SLOWER and the gap widens with length. One threadgroup per head walks the
// prefix in tiles of 8-16 keys with a chain of threadgroup barriers per row per tile — built for
// k <= 4 verify rows, latency-bound on barriers at k = 32. The bytes argument ("per-row attention
// re-reads 1.3 GB of KV per prompt token at 10K") was right about the traffic and wrong about the
// remedy. Long-prompt prefill needs a real tiled query-block x key-block kernel. DO NOT RE-CHASE
// this one for prefill; the branch that armed it was deleted.

// L95 f16-KV VERIFY: the pool element type is a macro so ONE body compiles as both the f32 kernel
// and an f16 twin. The twin exists to remove the verify path's f32 tax (L93): today verify forces
// the f32 pools (`use_kv_f16 = ... && !use_verify`, concurrent_metal.rs:2360) AND imports the whole
// prefix f16->f32 per call (batch.rs:1504-1523) — together ~4x the KV bytes of a decode step, which
// is why verify cost 4.35x a decode step (L94) instead of the ~2x the batched path achieves at the
// same row count. Loads convert to float on read, so the softmax/accumulate math is UNCHANGED and
// f32 throughout; only the bytes moved shrink. Parity vs the f32 kernel is therefore the gate.
#ifndef VERIFY_KV_T
#define VERIFY_KV_T float
#define VERIFY_KERNEL_NAME attention_verify_shared_prefix
#endif
kernel void VERIFY_KERNEL_NAME(
        device const float *q          [[buffer(0)]],
        device const VERIFY_KV_T *key_pool   [[buffer(1)]],
        device const VERIFY_KV_T *value_pool [[buffer(2)]],
        device       float *out        [[buffer(3)]],
        device const uint  *slots      [[buffer(4)]],   // ONE seq: prefix[0..prefix_len) ++ drafted
        constant     Dims  &d          [[buffer(5)]],
        uint  head                     [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]]) {
    const uint hd = d.head_dim;
    const uint kv_head = head / d.group;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_floats = d.kv_heads * hd;
    const uint head_col = kv_head * hd;
    // Defensive clamps: a corrupted dims buffer must degrade to WRONG NUMBERS (parity catches it),
    // never an unbounded loop / KMAX-array overrun that wedges the AGX firmware.
    const uint prefix_len = min(d.past_len, 1048576u);   // shared prefix length (repurposed field)
    const uint k = min(d.q_start, KMAX);                 // # verify query rows (repurposed field)

    // Per-query-row online-softmax state (KMAX rows; only [0,k) live).
    threadgroup float scores[KMAX][KMAX];        // per-row scores: a staged prefix tile (<= TILE_V) or the window (<= KMAX)
    threadgroup float red[TG];
    threadgroup float m_run[KMAX];
    threadgroup float l_run[KMAX];
    threadgroup float corr[KMAX];                // per-row rescale this tile
    threadgroup float ksh[TILE_V * 128];         // staged K rows for the tile (read ONCE from pool)
    threadgroup float vsh[TILE_V * 128];         // staged V rows for the tile (read ONCE from pool)

    const uint DPL = (hd + TG - 1u) / TG;        // hd<=256 → DPL 1 (qwen3-coder hd=128)
    // Per (query-row, dim-slot) accumulator. KMAX rows × DPL(<=2) dims/lane.
    float acc[KMAX][2];
    for (uint i = 0; i < k; ++i) { acc[i][0] = 0.0f; acc[i][1] = 0.0f; }
    if (lane < k) { m_run[lane] = -3.4e38f; l_run[lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===== PHASE 1: SHARED PREFIX [0, prefix_len) — STAGED ONCE per tile, consumed by all k rows. ==
    // For window==0 (global) EVERY verify row attends the whole prefix, so the prefix contributes
    // IDENTICALLY to every row's flash accumulator. We stream it in TILE_V-key tiles; each tile's K
    // AND V are cooperatively staged into threadgroup memory ONCE (all 256 lanes stride over
    // tile_len*hd elements → pool touched EXACTLY ONCE per (head,tile)). Then, for EACH query row i,
    // we (A1) score the tile from the STAGED ksh, (A2/A3) per-row max/exp/sum, (B) fold the STAGED
    // vsh into acc[i]. The k×→1× prefix DRAM reuse is exactly this stage-once/consume-k-times.
    // (Same numerics as attention_decode_b: online-softmax rescale, ascending key order.)
    // Sliding-window is deferred: window>0 would clip each row's prefix lo independently.
    // keys per staged tile: what TILE_V*128 floats can hold at THIS head_dim (16 @128, 8 @256)
    const uint tile_v = min(TILE_V, (TILE_V * 128u) / max(hd, 1u));
    if (tile_v == 0u) return; // head_dim > 2048: nothing fits; the host must not dispatch this
    uint tile0 = 0u;
    while (tile0 < prefix_len) {
        const uint tile_len = min(tile0 + tile_v, prefix_len) - tile0;

        // STAGE the tile's K and V into tg-mem ONCE (the prefix reuse). slots[] read once (shared).
        for (uint idx = lane; idx < tile_len * hd; idx += TG) {
            const uint t = idx / hd, jj = idx % hd;
            const uint kb = slots[tile0 + t] * row_floats + head_col;
            ksh[idx] = (float)key_pool[kb + jj];
            vsh[idx] = (float)value_pool[kb + jj];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A1 scores per row, from the STAGED ksh (one key per lane, strided; inner loop over rows).
        for (uint t = lane; t < tile_len; t += TG) {
            for (uint i = 0; i < k; ++i) {
                const uint q_base = (i * d.num_heads + head) * hd;
                float s = 0.0f;
                for (uint jj = 0; jj < hd; ++jj) s += q[q_base + jj] * ksh[t * hd + jj];
                scores[i][t] = s * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A2/A3 per-row online softmax — k SEQUENTIAL tree reductions reusing red[256] (SAME order
        // as the per-row oracle → association UNCHANGED).
        for (uint i = 0; i < k; ++i) {
            float lmax = -3.4e38f;
            for (uint c = lane; c < tile_len; c += TG) lmax = max(lmax, scores[i][c]);
            red[lane] = lmax;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint st = TG / 2u; st > 0u; st >>= 1) {
                if (lane < st) red[lane] = max(red[lane], red[lane + st]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) {
                const float m_new = max(m_run[i], red[0]);
                corr[i] = exp(m_run[i] - m_new);
                m_run[i] = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float lsum = 0.0f;
            for (uint e = lane; e < tile_len; e += TG) {
                const float ex = exp(scores[i][e] - m_run[i]);
                scores[i][e] = ex;
                lsum += ex;
            }
            red[lane] = lsum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint st = TG / 2u; st > 0u; st >>= 1) {
                if (lane < st) red[lane] = red[lane] + red[lane + st];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) l_run[i] = l_run[i] * corr[i] + red[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // B fold the STAGED vsh into acc[i] (each lane owns DPL strided dims), per row.
        for (uint i = 0; i < k; ++i) {
            for (uint j = 0; j < DPL; ++j) {
                const uint dim = lane + j * TG;
                if (dim < hd) {
                    float add = 0.0f;
                    for (uint tt = 0; tt < tile_len; ++tt) add += scores[i][tt] * vsh[tt * hd + dim];
                    acc[i][j] = acc[i][j] * corr[i] + add;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile0 += tile_v;
    }

    // ===== PHASE 2: CAUSAL WINDOW [prefix_len, prefix_len+k) — small (k×k), per row. =====
    // Row i additionally attends the drafted rows 0..i (their freshly-scattered K/V at pool slots
    // slots[prefix_len+0 .. prefix_len+i]) INCLUDING itself (slots[prefix_len+i]). These keys were
    // written this step by the verify scatter into the shared pool; the slot table's tail carries
    // their pool rows. This is the strictly-causal part: row i sees window positions [0, i].
    for (uint i = 0; i < k; ++i) {
        const uint q_base = (i * d.num_heads + head) * hd;
        const uint win_len = i + 1u;             // positions prefix_len+0 .. prefix_len+i
        // A1 window scores (into scores[i][*]; win_len <= k <= KMAX, the row's width).
        for (uint t = lane; t < win_len; t += TG) {
            const uint kb = slots[prefix_len + t] * row_floats + head_col;
            float s = 0.0f;
            for (uint jj = 0; jj < hd; ++jj) s += q[q_base + jj] * (float)key_pool[kb + jj];
            scores[i][t] = s * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A2 max.
        float lmax = -3.4e38f;
        for (uint c = lane; c < win_len; c += TG) lmax = max(lmax, scores[i][c]);
        red[lane] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG / 2u; st > 0u; st >>= 1) {
            if (lane < st) red[lane] = max(red[lane], red[lane + st]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float m_new = max(m_run[i], red[0]);
            corr[i] = exp(m_run[i] - m_new);
            m_run[i] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A3 exp + sum.
        float lsum = 0.0f;
        for (uint e = lane; e < win_len; e += TG) {
            const float ex = exp(scores[i][e] - m_run[i]);
            scores[i][e] = ex;
            lsum += ex;
        }
        red[lane] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG / 2u; st > 0u; st >>= 1) {
            if (lane < st) red[lane] = red[lane] + red[lane + st];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) l_run[i] = l_run[i] * corr[i] + red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // B fold.
        for (uint j = 0; j < DPL; ++j) {
            const uint dim = lane + j * TG;
            if (dim < hd) {
                float add = 0.0f;
                for (uint tt = 0; tt < win_len; ++tt) {
                    add += scores[i][tt] * value_pool[slots[prefix_len + tt] * row_floats + head_col + dim];
                }
                acc[i][j] = acc[i][j] * corr[i] + add;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===== OUTPUT: out[(i*nh + head)*hd + dim] = acc[i]/l_run[i], per query row. =====
    for (uint i = 0; i < k; ++i) {
        const uint out_base = (i * d.num_heads + head) * hd;
        for (uint j = 0; j < DPL; ++j) {
            const uint dim = lane + j * TG;
            if (dim < hd) out[out_base + dim] = acc[i][j] / l_run[i];
        }
    }
}

// ── f16 twin of the SAME body (L95). Generated by re-including the source above with the pool
// element type swapped to `half`. Nothing else differs: q stays f32, all math stays f32, only the
// KV loads shrink 4B->2B. Any divergence from the f32 kernel is therefore a pure precision effect
// and must be caught by batched_mega_parity before this is enabled.
#undef VERIFY_KV_T
#undef VERIFY_KERNEL_NAME
#define VERIFY_KV_T half
#define VERIFY_KERNEL_NAME attention_verify_shared_prefix_f16
kernel void VERIFY_KERNEL_NAME(
        device const float *q          [[buffer(0)]],
        device const VERIFY_KV_T *key_pool   [[buffer(1)]],
        device const VERIFY_KV_T *value_pool [[buffer(2)]],
        device       float *out        [[buffer(3)]],
        device const uint  *slots      [[buffer(4)]],   // ONE seq: prefix[0..prefix_len) ++ drafted
        constant     Dims  &d          [[buffer(5)]],
        uint  head                     [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]]) {
    const uint hd = d.head_dim;
    const uint kv_head = head / d.group;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_floats = d.kv_heads * hd;
    const uint head_col = kv_head * hd;
    // Defensive clamps: a corrupted dims buffer must degrade to WRONG NUMBERS (parity catches it),
    // never an unbounded loop / KMAX-array overrun that wedges the AGX firmware.
    const uint prefix_len = min(d.past_len, 1048576u);   // shared prefix length (repurposed field)
    const uint k = min(d.q_start, KMAX);                 // # verify query rows (repurposed field)

    // Per-query-row online-softmax state (KMAX rows; only [0,k) live).
    threadgroup float scores[KMAX][KMAX];        // per-row scores: a staged prefix tile (<= TILE_V) or the window (<= KMAX)
    threadgroup float red[TG];
    threadgroup float m_run[KMAX];
    threadgroup float l_run[KMAX];
    threadgroup float corr[KMAX];                // per-row rescale this tile
    threadgroup float ksh[TILE_V * 128];         // staged K rows for the tile (read ONCE from pool)
    threadgroup float vsh[TILE_V * 128];         // staged V rows for the tile (read ONCE from pool)

    const uint DPL = (hd + TG - 1u) / TG;        // hd<=256 → DPL 1 (qwen3-coder hd=128)
    // Per (query-row, dim-slot) accumulator. KMAX rows × DPL(<=2) dims/lane.
    float acc[KMAX][2];
    for (uint i = 0; i < k; ++i) { acc[i][0] = 0.0f; acc[i][1] = 0.0f; }
    if (lane < k) { m_run[lane] = -3.4e38f; l_run[lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===== PHASE 1: SHARED PREFIX [0, prefix_len) — STAGED ONCE per tile, consumed by all k rows. ==
    // For window==0 (global) EVERY verify row attends the whole prefix, so the prefix contributes
    // IDENTICALLY to every row's flash accumulator. We stream it in TILE_V-key tiles; each tile's K
    // AND V are cooperatively staged into threadgroup memory ONCE (all 256 lanes stride over
    // tile_len*hd elements → pool touched EXACTLY ONCE per (head,tile)). Then, for EACH query row i,
    // we (A1) score the tile from the STAGED ksh, (A2/A3) per-row max/exp/sum, (B) fold the STAGED
    // vsh into acc[i]. The k×→1× prefix DRAM reuse is exactly this stage-once/consume-k-times.
    // (Same numerics as attention_decode_b: online-softmax rescale, ascending key order.)
    // Sliding-window is deferred: window>0 would clip each row's prefix lo independently.
    // keys per staged tile: what TILE_V*128 floats can hold at THIS head_dim (16 @128, 8 @256)
    const uint tile_v = min(TILE_V, (TILE_V * 128u) / max(hd, 1u));
    if (tile_v == 0u) return; // head_dim > 2048: nothing fits; the host must not dispatch this
    uint tile0 = 0u;
    while (tile0 < prefix_len) {
        const uint tile_len = min(tile0 + tile_v, prefix_len) - tile0;

        // STAGE the tile's K and V into tg-mem ONCE (the prefix reuse). slots[] read once (shared).
        for (uint idx = lane; idx < tile_len * hd; idx += TG) {
            const uint t = idx / hd, jj = idx % hd;
            const uint kb = slots[tile0 + t] * row_floats + head_col;
            ksh[idx] = (float)key_pool[kb + jj];
            vsh[idx] = (float)value_pool[kb + jj];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A1 scores per row, from the STAGED ksh (one key per lane, strided; inner loop over rows).
        for (uint t = lane; t < tile_len; t += TG) {
            for (uint i = 0; i < k; ++i) {
                const uint q_base = (i * d.num_heads + head) * hd;
                float s = 0.0f;
                for (uint jj = 0; jj < hd; ++jj) s += q[q_base + jj] * ksh[t * hd + jj];
                scores[i][t] = s * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A2/A3 per-row online softmax — k SEQUENTIAL tree reductions reusing red[256] (SAME order
        // as the per-row oracle → association UNCHANGED).
        for (uint i = 0; i < k; ++i) {
            float lmax = -3.4e38f;
            for (uint c = lane; c < tile_len; c += TG) lmax = max(lmax, scores[i][c]);
            red[lane] = lmax;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint st = TG / 2u; st > 0u; st >>= 1) {
                if (lane < st) red[lane] = max(red[lane], red[lane + st]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) {
                const float m_new = max(m_run[i], red[0]);
                corr[i] = exp(m_run[i] - m_new);
                m_run[i] = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float lsum = 0.0f;
            for (uint e = lane; e < tile_len; e += TG) {
                const float ex = exp(scores[i][e] - m_run[i]);
                scores[i][e] = ex;
                lsum += ex;
            }
            red[lane] = lsum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint st = TG / 2u; st > 0u; st >>= 1) {
                if (lane < st) red[lane] = red[lane] + red[lane + st];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) l_run[i] = l_run[i] * corr[i] + red[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // B fold the STAGED vsh into acc[i] (each lane owns DPL strided dims), per row.
        for (uint i = 0; i < k; ++i) {
            for (uint j = 0; j < DPL; ++j) {
                const uint dim = lane + j * TG;
                if (dim < hd) {
                    float add = 0.0f;
                    for (uint tt = 0; tt < tile_len; ++tt) add += scores[i][tt] * vsh[tt * hd + dim];
                    acc[i][j] = acc[i][j] * corr[i] + add;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile0 += tile_v;
    }

    // ===== PHASE 2: CAUSAL WINDOW [prefix_len, prefix_len+k) — small (k×k), per row. =====
    // Row i additionally attends the drafted rows 0..i (their freshly-scattered K/V at pool slots
    // slots[prefix_len+0 .. prefix_len+i]) INCLUDING itself (slots[prefix_len+i]). These keys were
    // written this step by the verify scatter into the shared pool; the slot table's tail carries
    // their pool rows. This is the strictly-causal part: row i sees window positions [0, i].
    for (uint i = 0; i < k; ++i) {
        const uint q_base = (i * d.num_heads + head) * hd;
        const uint win_len = i + 1u;             // positions prefix_len+0 .. prefix_len+i
        // A1 window scores (into scores[i][*]; win_len <= k <= KMAX, the row's width).
        for (uint t = lane; t < win_len; t += TG) {
            const uint kb = slots[prefix_len + t] * row_floats + head_col;
            float s = 0.0f;
            for (uint jj = 0; jj < hd; ++jj) s += q[q_base + jj] * (float)key_pool[kb + jj];
            scores[i][t] = s * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A2 max.
        float lmax = -3.4e38f;
        for (uint c = lane; c < win_len; c += TG) lmax = max(lmax, scores[i][c]);
        red[lane] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG / 2u; st > 0u; st >>= 1) {
            if (lane < st) red[lane] = max(red[lane], red[lane + st]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float m_new = max(m_run[i], red[0]);
            corr[i] = exp(m_run[i] - m_new);
            m_run[i] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A3 exp + sum.
        float lsum = 0.0f;
        for (uint e = lane; e < win_len; e += TG) {
            const float ex = exp(scores[i][e] - m_run[i]);
            scores[i][e] = ex;
            lsum += ex;
        }
        red[lane] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG / 2u; st > 0u; st >>= 1) {
            if (lane < st) red[lane] = red[lane] + red[lane + st];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) l_run[i] = l_run[i] * corr[i] + red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // B fold.
        for (uint j = 0; j < DPL; ++j) {
            const uint dim = lane + j * TG;
            if (dim < hd) {
                float add = 0.0f;
                for (uint tt = 0; tt < win_len; ++tt) {
                    add += scores[i][tt] * value_pool[slots[prefix_len + tt] * row_floats + head_col + dim];
                }
                acc[i][j] = acc[i][j] * corr[i] + add;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===== OUTPUT: out[(i*nh + head)*hd + dim] = acc[i]/l_run[i], per query row. =====
    for (uint i = 0; i < k; ++i) {
        const uint out_base = (i * d.num_heads + head) * hd;
        for (uint j = 0; j < DPL; ++j) {
            const uint dim = lane + j * TG;
            if (dim < hd) out[out_base + dim] = acc[i][j] / l_run[i];
        }
    }
}


// ============================================================================================
// COALESCED N-simdgroup DECODE ATTENTION (ARF_ATTN_COALESCED) — the conc64 WIN.
//
// attention_decode_b reads KV UNCOALESCED (each thread gathers a whole 128-dim key row alone;
// adjacent lanes' keys are row_floats apart), runs ~8 threadgroup_barriers/tile for its tree
// softmax, and strides every value per thread in PV. That is memory-latency bound → low
// arithmetic intensity → the DVFS governor holds the GPU at ~380MHz even at 99% busy.
//
// The FIRST cut of this kernel used ONE 32-lane simdgroup per (head,seq): it coalesced the KV
// access but ran at 1/8 the oracle's occupancy (the oracle is 256 threads = 8 simdgroups). The
// serial online-softmax dependency chain over ALL keys, with no latency-hiding, cancelled the
// coalescing gain → a perf WASH. THE FIX (this kernel): NSG=8 simdgroups per threadgroup (256
// threads = the oracle's occupancy), each simdgroup COALESCING its own strided KV subset, then a
// ONE-barrier flash merge of the NSG partial online-softmax states — llama's split-KV geometry
// folded INTO the threadgroup. Coalesced access AT the oracle's occupancy = the real win.
//
//   * grid (nh, B), NSG=8 simdgroups (256 lanes) per (head, seq) threadgroup.
//     sg = lane / 32 (which simdgroup); lin = lane % 32 (lane within the simdgroup, 0..31).
//   * SPLIT: simdgroup `sg` processes keys t = lo+sg, lo+sg+NSG, lo+sg+2*NSG, ... (stride NSG).
//     Each SG runs the SAME coalesced-load + simd_sum + in-register online-softmax over ITS
//     subset → a partial (m_sg, l_sg, acc_sg[]). NO barriers during the per-SG stream.
//   * Q staged in registers ONCE as float4-per-lin chunks, reused across every key of the SG.
//   * QK: the 32 lins of a SG CO-LOAD one key's 128-dim row as float4/lin at CONSECUTIVE
//     addresses (lin L reads key[... + (L + c*32)*4 .. +3]); local fma vs the matching Q float4,
//     simd_sum across the 32 lins → the scalar score (broadcast to all 32 lins of the SG).
//   * PV: the value row co-loaded coalesced (same layout as K), fma'd into per-lin acc chunks.
//   * MERGE (the ONLY barrier): each SG writes (m_sg, l_sg) to threadgroup arrays; lin 0 of each
//     SG writes its acc chunks to a threadgroup acc grid. After one barrier, EVERY lane computes
//     the flash combine (mirroring attention_splitk_combine_b EXACTLY for bit-parity):
//       m  = max_sg m_sg ;  l = Σ_sg exp(m_sg − m)·l_sg
//       out[dim] = ( Σ_sg exp(m_sg − m)·acc_sg[dim] ) / l
//     Empty SGs (fewer keys than NSG) kept m_sg=-inf, l_sg=0 → exp(m_sg−m)=0, contribute nothing.
//   * out[dim] = combine / l in the SAME layout attention_decode_b writes.
//
// BINDINGS 0-6 are BYTE-IDENTICAL to attention_decode_b (q, key_pool, value_pool, out,
// slots_all, Dims, past_len_b), so the dispatch site swaps only the PSO + grid + threadgroup
// size (now 256 threads = 8 simdgroups). NUMERICS: f32 throughout (KV stays f32 this pass). The
// softmax is the classic online flash formulation; keys are re-associated across SGs (strided
// subsets) then merged with the SAME rescale math as the split-KV combine — stays within the
// 1e-4 parity bar (argmax-identical vs the _b oracle).
//
// Requires hd % 128 == 0 (32 lanes × float4). qwen3 hd=128 (1 chunk/lin), gemma hd=256/512
// (2/4 chunks/lin). MAXCH caps the per-lin chunk count (hd<=512 → 4). MAXHD caps the merge
// threadgroup acc grid (NSG*hd floats; hd<=512 → 8*512 = 4096 floats = 16KB).

constant constexpr uint SIMD_W = 32u;   // one simdgroup = 32 lanes
constant constexpr uint MAXCH  = 4u;    // hd/128 chunks per lane (hd<=512)
// L154 — max q-heads a single simdgroup owns in the kvhead kernel = max_group / NSG.
// group 8 -> 1, 16 -> 2, 32 -> 4. Sizes the per-q-head register arrays.
constant constexpr uint MAXQPG = 4u;
constant constexpr uint NSG    = 8u;    // simdgroups per threadgroup (256 threads = oracle occupancy)
// L39 OCCUPANCY FIX — MAXHD sizes `threadgroup float tg_acc[NSG*MAXHD]`, and at the 512 default
// that is 8*512*4 = 16384 B. MEASURED (ARF_OCCUPANCY=1): this kernel's staticSmem is 16448 B,
// i.e. MORE THAN HALF of an Apple core's 32 KiB threadgroup memory → exactly ONE threadgroup
// resident per core, so a DRAM stall has nothing to switch to. That is the direct mechanism
// behind L37's "2% of bandwidth AND 6% of compute → STALLED" reading, and attention is 47% of
// the step at B=16.
//
// The 512 is a WORST-CASE bound for gemma (hd=512). qwen3-coder is hd=128 → the grid only needs
// 8*128*4 = 4096 B, and residency goes 1 → 7 threadgroups per core.
//
// The host compiles this source PER MODEL, so it can pass the real head_dim:
//   #define ATTN_MAXHD <head_dim rounded up to a multiple of 128>
// Undefined → 512u, byte-identical to the shipped behavior. Correctness: MAXHD only bounds the
// tg_acc array; every access is indexed by the RUNTIME hd (d.head_dim), so a tighter bound is
// safe as long as ATTN_MAXHD >= the model's actual head_dim (the host guarantees this).
#ifndef ATTN_MAXHD
#define ATTN_MAXHD 512u
#endif
constant constexpr uint MAXHD  = ATTN_MAXHD;  // max head_dim supported by the merge tg-mem grid

kernel void attention_decode_coalesced_b(
        device const float *q          [[buffer(0)]],
        device const float *key_pool   [[buffer(1)]],
        device const float *value_pool [[buffer(2)]],
        device       float *out        [[buffer(3)]],
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]]) {
    const uint head = gid.x;        // head index (0..nh)
    const uint r    = gid.y;        // sequence index (0..B)
    const uint hd = d.head_dim;
    const uint kv_head = head / d.group;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_floats = d.kv_heads * hd;
    const uint head_col = kv_head * hd;
    const uint max_slots = d.q_start;            // q_start REPURPOSED as the slot-table stride
    const uint q_base = (r * d.num_heads + head) * hd;   // q[r*q_dim + head*hd]
    device const uint *slots = slots_all + r * max_slots; // this seq's slot table

    const uint last = past_len_b[r];             // this seq's OWN past_len (decode: q_len==1, row 0)
    uint lo = 0u;
    if (d.window > 0u && last + 1u > d.window) lo = last + 1u - d.window;

    // Threadgroup geometry: 256 threads = NSG simdgroups of 32 lanes. sg picks the simdgroup;
    // lin (0..31) is the lane within it (also the float4-chunk offset for the coalesced loads).
    const uint sg  = lane / SIMD_W;   // 0..NSG-1
    const uint lin = lane % SIMD_W;   // 0..31

    // Chunks of float4 each lin owns: chunk c covers dims [(lin + c*SIMD_W)*4 .. +3].
    // SIMD_W*4 = 128 dims/chunk, so nch = hd/128. Each chunk's 4 dims are that lin's output dims.
    const uint nch = (hd + (SIMD_W * 4u) - 1u) / (SIMD_W * 4u);

    // Stage Q float4 per chunk ONCE (reused across every key). q_base+dim is float4-aligned:
    // q_base is a multiple of hd (>=128 floats = 512B) and (lin+c*32)*4 is a multiple of 4.
    device const float4 *q4 = reinterpret_cast<device const float4 *>(q + q_base);
    float4 qv[MAXCH];
    for (uint c = 0; c < nch; ++c) qv[c] = q4[lin + c * SIMD_W];

    // Per-simdgroup in-register online-softmax state (identical scalar update on every lin of the
    // SG; score is broadcast by simd_sum). Empty SGs keep m=-inf, l=0 → contribute nothing at merge.
    float m_run = -3.4e38f;
    float l_run = 0.0f;
    float4 acc[MAXCH];
    for (uint c = 0; c < nch; ++c) acc[c] = float4(0.0f);

    // SPLIT: simdgroup `sg` streams its strided subset of keys in ASCENDING order:
    // t = lo+sg, lo+sg+NSG, lo+sg+2*NSG, ... — each SG walks its OWN keys, no cross-SG sync.
    for (uint t = lo + sg; t <= last; t += NSG) {
        const uint kb = slots[t] * row_floats + head_col;   // this key's row base (float index)
        device const float4 *k4 = reinterpret_cast<device const float4 *>(key_pool + kb);
        // COALESCED QK: lin L loads float4 at chunk offset (L + c*32); adjacent lins → adjacent
        // float4s → one contiguous burst per chunk. Local dot, then simd_sum → the scalar score.
        float partial = 0.0f;
        for (uint c = 0; c < nch; ++c) {
            const float4 kv = k4[lin + c * SIMD_W];
            partial += qv[c].x * kv.x; partial += qv[c].y * kv.y;
            partial += qv[c].z * kv.z; partial += qv[c].w * kv.w;
        }
        const float s = simd_sum(partial) * scale;          // broadcast to all 32 lins of the SG

        // Online flash rescale (barrier-free; every lin of the SG runs the identical scalar update).
        const float m_new = max(m_run, s);
        const float corr  = exp(m_run - m_new);              // 0 on the first key (m_run=-inf)
        const float p     = exp(s - m_new);
        m_run = m_new;
        l_run = l_run * corr + p;

        // COALESCED PV: value row co-loaded the same way; fma p*v into the lin's own acc chunks.
        device const float4 *v4 = reinterpret_cast<device const float4 *>(value_pool + kb);
        for (uint c = 0; c < nch; ++c) {
            const float4 vv = v4[lin + c * SIMD_W];
            acc[c] = acc[c] * corr + p * vv;
        }
    }

    // ===== MERGE the NSG partial (m_sg, l_sg, acc_sg[]) states — the ONLY barrier. =====
    // Mirror attention_splitk_combine_b EXACTLY: m = max_sg m_sg; l = Σ exp(m_sg−m)·l_sg;
    // out[dim] = (Σ exp(m_sg−m)·acc_sg[dim]) / l. Threadgroup arrays hold the per-SG partials.
    threadgroup float tg_m[NSG];
    threadgroup float tg_l[NSG];
    threadgroup float tg_acc[NSG * MAXHD];   // per-SG un-normalized acc, dim-major within an SG
    // Each SG's m/l is replicated across its 32 lins; write once from lin 0. acc is distributed
    // across the SG's lins by chunk — every lin writes ITS chunks into the SG's acc row.
    if (lin == 0u) { tg_m[sg] = m_run; tg_l[sg] = l_run; }
    threadgroup float *acc_row = tg_acc + sg * hd;
    for (uint c = 0; c < nch; ++c) {
        const uint d0 = (lin + c * SIMD_W) * 4u;   // this lin's 4 dims for chunk c
        acc_row[d0 + 0u] = acc[c].x; acc_row[d0 + 1u] = acc[c].y;
        acc_row[d0 + 2u] = acc[c].z; acc_row[d0 + 3u] = acc[c].w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global max + rescaled denom over the NSG partials (identical on every lane).
    float mg = -3.4e38f;
    for (uint s = 0; s < NSG; ++s) mg = max(mg, tg_m[s]);
    float lg = 0.0f;
    for (uint s = 0; s < NSG; ++s) lg += exp(tg_m[s] - mg) * tg_l[s];
    const float inv_l = (lg > 0.0f) ? (1.0f / lg) : 0.0f;

    // Combine + write in attention_decode_b's layout: out[(r*nh + head)*hd + dim]. The dims are
    // covered by lin (chunk c → float4 at lin+c*32). ONE simdgroup (sg==0) does the write; its 32
    // lins tile all hd dims (nch chunks/lin). The other SGs' acc rows are read from tg_acc.
    if (sg == 0u) {
        const uint out_base = (r * d.num_heads + head) * hd;
        device float4 *out4 = reinterpret_cast<device float4 *>(out + out_base);
        for (uint c = 0; c < nch; ++c) {
            const uint d0 = (lin + c * SIMD_W) * 4u;
            float4 num = float4(0.0f);
            for (uint s = 0; s < NSG; ++s) {
                const float w = exp(tg_m[s] - mg);          // 0 for empty/low SGs
                threadgroup float *ar = tg_acc + s * hd + d0;
                num.x += w * ar[0]; num.y += w * ar[1];
                num.z += w * ar[2]; num.w += w * ar[3];
            }
            out4[lin + c * SIMD_W] = num * inv_l;
        }
    }
}

// ============================================================================================
// f16 KV CACHE (the 3rd leg: bandwidth). ARF_KV_F16-gated variants of the batched
// scatter + the coalesced attention that store/read the KV pool as `half` instead of `float`,
// halving the KV-cache DRAM traffic. Orthogonal to kv_quant (kv_quant stays None; f16 is just a
// separate pool FORMAT flag, not a lossy quant in the KvQuant sense). ONLY the memory format
// changes — the scatter converts f32 new-token K/V → half on store; the attention loads half4 and
// converts to float4 IN-REGISTER; the QK dot + online-softmax math stay f32 (byte-identical DAG to
// attention_decode_coalesced_b). f16 KV is LOSSY (f32->f16 round-trip), so this is NOT bit-exact vs
// the f32 oracle — the parity gate is IDENTICAL ARGMAX (greedy-robust; llama proves it), with
// max_abs on the attn output at the f16 round-trip magnitude (~1e-3). Default OFF → these kernels
// are never dispatched and the f32 pool + oracle are byte-unchanged.

// f16 SCATTER — byte-identical to kv_scatter_b except key_pool/value_pool are `half` and the store
// converts f32 -> half (implicit narrowing on assignment). Same 2D (col, seq) grid, same write.
kernel void kv_scatter_b_f16(
        device const float *new_k        [[buffer(0)]],
        device const float *new_v        [[buffer(1)]],
        device const uint  *write_slot_b [[buffer(2)]],
        device       half  *key_pool     [[buffer(3)]],
        device       half  *value_pool   [[buffer(4)]],
        constant ScatterDimsB &d         [[buffer(5)]],
        uint2 gid                        [[thread_position_in_grid]]) {
    const uint col = gid.x;          // column within the kv row (0..row_floats)
    const uint r   = gid.y;          // sequence index (0..B)
    if (col >= d.row_floats || r >= d.b) return;
    const uint pool_idx = write_slot_b[r] * d.row_floats + col;
    const uint src = r * d.row_floats + col;
    key_pool[pool_idx]   = half(new_k[src]);   // f32 -> f16 on store (the lossy round-trip)
    value_pool[pool_idx] = half(new_v[src]);
}

// ============================================================================================
// KV-HEAD-CENTRIC coalesced f16 decode attention — the group× KV-traffic lever. The per-q-head
// kernels above re-read each GQA group's KV pool once PER Q-HEAD (group=8 ⇒ 8×). This kernel
// dispatches ONE threadgroup per (kv_head, seq): 256 threads = NSG=8 simdgroups.
//
// L154 — GROUP IS NO LONGER PINNED TO 8. Simdgroup `sg` owns the q-heads
// `kv_head*group + sg + j*NSG` for j = 0..group/NSG-1, i.e. ONE q-head at group=8 (byte-identical
// to the old mapping, since the j-loop runs once) and TWO at group=16. Every piece of per-q-head
// state — qv, m_run, l_run, acc — was already private to the simdgroup, so serving a second
// q-head is just another pass over the SAME staged tile. The traffic saving therefore IMPROVES
// with group (16× at group=16), which is exactly where a wide-GQA model needs it. Requires
// group % NSG == 0 and group >= NSG; the host gates that. KV
// tiles are staged into threadgroup memory ONCE by all 256 threads, then every SG scores its own
// q-head from the shared tile — device KV traffic drops 8×, compounding with f16's 2× (16× vs
// the original per-q-head f32 kernel). Each SG keeps a PRIVATE in-register online softmax for
// its q-head, so there is NO cross-SG merge — the only barriers fence the tile staging.
// Bindings 0-6 byte-identical to attention_decode_coalesced_b_f16; the dispatch swaps the PSO
// and the grid x-dim (nkv instead of nh). Requires hd%128==0 and hd<=KVH_MAXHD (tg-mem budget:
// 2 tiles × KVH_TILE × hd halfs; hd=256 ⇒ 16KB). Same f32 QK/softmax/PV math as the f16
// coalesced kernel — parity gate is IDENTICAL ARGMAX vs the full-precision f32 oracle.

// L40 — KEYS PER TILE. llama's flash_attn_ext blocks 64 keys per simdgroup pass
// (OP_FLASH_ATTN_EXT_NCPSG=64, vec variant 32; ggml-metal-impl.h:110,113); we stage 16. A
// smaller tile means more loop iterations, more barriers and more reduction rounds per key for
// the SAME work. Before L39 this bound was not tunable in practice — the tg-mem budget was
// already spent on the MAXHD=512 over-allocation. With that fixed (hd=128 → 8192 B here) there
// is room: KVH_TILE=32 costs 16384 B (2 TG/core, what we shipped BEFORE L39), 64 costs the whole
// 32 KiB (1 TG/core).
// L40 MEASURED FLAT (2026-08-05): swept 8/16/32/64 — every value inside the 3.3% floor at every
// tier, so matching llama's 64 bought nothing. The tunable was deleted under the env ratchet and
// 16 is the shipped constant. DO NOT re-sweep this without a new mechanism to justify it.
constant constexpr uint KVH_TILE  = 16u;   // keys staged per tile
// L39 OCCUPANCY FIX (same mechanism as ATTN_MAXHD above). KVH_MAXHD sizes the two staged-KV
// arrays `half4 k4t/v4t[KVH_TILE*KVH_MAXHD/4]` = 2*16*256*2B = 16384 B at the 256 default.
// MEASURED staticSmem for attention_decode_kvhead_b_f16 / _splitk_b_f16 = 16384 B → only 2
// threadgroups resident per 32 KiB core. qwen3-coder is hd=128, which needs 2*16*128*2 = 8192 B
// and doubles residency to 4. These are the kernels the conc8/conc16 tiers ship.
// Host passes `#define ATTN_KVH_MAXHD <head_dim>`; undefined → 256u (byte-identical default).
// Safe for the same reason: KVH_MAXHD only bounds the arrays, all indexing uses the runtime hd.
#ifndef ATTN_KVH_MAXHD
#define ATTN_KVH_MAXHD 256u
#endif
constant constexpr uint KVH_MAXHD = ATTN_KVH_MAXHD;  // max head_dim (tg-mem: 2*16*hd*2B)

// ============================================================================================
// PREFILL ATTENTION, KV-HEAD x ROW-BLOCK (2026-09-23). The windowed-prefill form of the kvhead
// kernel, f32 KV. WHY: at long context prefill is attention-bound — per-token prefill cost fits
// 7.2 ms + 0.46 us x context (measured 2026-09-23), 626 s to first token at 29.8K against
// llama.cpp ~150 s on the same GGUF — because the per-(q_head, row) kernels re-read a KV head's
// whole history for every q-head AND every prompt row: at 30K that is ~61 MB per KV head per
// layer, far past the caches, read 6 x 96 times per window.
//
// Here ONE threadgroup owns (kv_head, a block of PF_ROWS window rows); simdgroup sg owns the pair
// (q_head = kv_head*group + sg % group, row = row0 + sg / group). Each KV tile is staged into
// threadgroup memory ONCE, in f32, and scored by every (q_head, row) of the block: device KV
// traffic drops group x PF_ROWS (24x at group 6). Each simdgroup keeps a PRIVATE online softmax
// over the keys in ASCENDING order (no cross-SG merge), f32 throughout.
//
// CAUSAL: the rows of one window see different key counts — row r sees [lo, past_len_b[r]] —
// so a tile is staged up to the block's LAST row and every SG skips keys past its own row. The
// slot table used for staging is the block's last row's: the rows are one sequence, so every
// row's table is a prefix of it.
//
// Bindings 0-6 byte-identical to attention_decode_coalesced_b. Grid (nkv, ceil(B / PF_ROWS)),
// threads group * PF_ROWS * 32 (host: <= 1024). Host gate: a COMMITTED prefill window only (never
// the speculative verify, whose logits must match the 1-row path), f32 KV, hd % 128 == 0,
// hd <= PF_MAXHD. ARF_NO_PREFILL_ATTN=1 opts out.
constant constexpr uint PF_TILE  = 8u;     // keys staged per tile (2 x 8 x 256 x 4 B = 16 KB)
constant constexpr uint PF_MAXHD = 256u;
kernel void attention_prefill_kvhead_f32(
        device const float *q          [[buffer(0)]],
        device const float *key_pool   [[buffer(1)]],
        device const float *value_pool [[buffer(2)]],
        device       float *out        [[buffer(3)]],
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        constant     uint  &nrows      [[buffer(7)]],   // B: rows in the window
        constant     uint  &pf_rows    [[buffer(8)]],   // rows per threadgroup
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]],
        uint2 tptg                     [[threads_per_threadgroup]]) {
    // uint2, not uint: Metal refuses a scalar threads_per_threadgroup beside the uint2 grid
    // position, and the failure takes down EVERY kernel compiled from this source at load.
    const uint nthreads = tptg.x;
    const uint kv_head = gid.x;
    const uint row0 = gid.y * pf_rows;
    const uint hd = d.head_dim;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_floats = d.kv_heads * hd;
    const uint head_col = kv_head * hd;
    const uint max_slots = d.q_start;
    const uint group = d.group;

    const uint sg  = lane / SIMD_W;
    const uint lin = lane % SIMD_W;
    const uint nch = (hd + (SIMD_W * 4u) - 1u) / (SIMD_W * 4u);
    const uint my_row = row0 + sg / group;
    const uint qh = kv_head * group + sg % group;
    const bool live = (sg / group) < pf_rows && my_row < nrows;

    // The block's last live row bounds the staging; its table covers every row of the block.
    const uint last_row = min(row0 + pf_rows, nrows) - 1u;
    const uint block_last = past_len_b[last_row];
    device const uint *slots = slots_all + last_row * max_slots;
    const uint my_last = live ? past_len_b[my_row] : 0u;
    uint lo = 0u;
    if (d.window > 0u && past_len_b[row0] + 1u > d.window) lo = past_len_b[row0] + 1u - d.window;
    uint my_lo = 0u;
    if (live && d.window > 0u && my_last + 1u > d.window) my_lo = my_last + 1u - d.window;

    float4 qv[MAXCH];
    float4 acc[MAXCH];
    float m_run = -3.4e38f, l_run = 0.0f;
    if (live) {
        device const float4 *q4 = reinterpret_cast<device const float4 *>(q + (my_row * d.num_heads + qh) * hd);
        for (uint c = 0; c < nch; ++c) qv[c] = q4[lin + c * SIMD_W];
    }
    for (uint c = 0; c < nch; ++c) acc[c] = float4(0.0f);

    threadgroup float4 k4t[PF_TILE * PF_MAXHD / 4u];
    threadgroup float4 v4t[PF_TILE * PF_MAXHD / 4u];
    device const float4 *kp = reinterpret_cast<device const float4 *>(key_pool);
    device const float4 *vp = reinterpret_cast<device const float4 *>(value_pool);
    const uint h4 = hd / 4u;

    for (uint base = lo; base <= block_last; base += PF_TILE) {
        const uint ntile = min(PF_TILE, block_last - base + 1u);
        const uint total4 = ntile * h4;
        for (uint i = lane; i < total4; i += nthreads) {
            const uint row = i / h4, col = i % h4;
            const uint kb = (slots[base + row] * row_floats + head_col) / 4u;
            k4t[row * h4 + col] = kp[kb + col];
            v4t[row * h4 + col] = vp[kb + col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            for (uint t = 0; t < ntile; ++t) {
                const uint key = base + t;
                if (key < my_lo || key > my_last) continue;   // causal (and window) mask
                threadgroup const float4 *kr = k4t + t * h4;
                float partial = 0.0f;
                for (uint c = 0; c < nch; ++c) {
                    const float4 kv = kr[lin + c * SIMD_W];
                    partial += qv[c].x * kv.x; partial += qv[c].y * kv.y;
                    partial += qv[c].z * kv.z; partial += qv[c].w * kv.w;
                }
                const float sc = simd_sum(partial) * scale;
                const float m_new = max(m_run, sc);
                const float corr  = exp(m_run - m_new);
                const float pr    = exp(sc - m_new);
                m_run = m_new;
                l_run = l_run * corr + pr;
                threadgroup const float4 *vr = v4t + t * h4;
                for (uint c = 0; c < nch; ++c) acc[c] = acc[c] * corr + pr * vr[lin + c * SIMD_W];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (live) {
        const float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
        device float4 *out4 = reinterpret_cast<device float4 *>(out + (my_row * d.num_heads + qh) * hd);
        for (uint c = 0; c < nch; ++c) out4[lin + c * SIMD_W] = acc[c] * inv_l;
    }
}

// ============================================================================================
// PREFILL FLASH ATTENTION on simdgroup_matrix (2026-09-23). The scalar kv-head x row-block kernel
// above cut KV traffic 24x but measured only 1.5x on the attention term: every simdgroup walks
// every key with a simd_sum and a threadgroup barrier per 8-key tile. This one does the scores
// and the value sum as 8x8 matrix products:
//   simdgroup sg of a threadgroup owns q-head (kv_head*group + sg) for 8 window rows r0..r0+7.
//   Per 8-key tile: S(8q x 8k) = sum over 32 dim-steps of Q(8x8) * Kt(8x8), Kt loaded straight
//   from the paged pool (8 keys starting at a multiple of 8 lie in ONE 16-token block, so their
//   rows are consecutive slots: one strided simdgroup_load, transposed); causal mask per row;
//   online softmax per row (a row's 8 scores live on 4 lanes: xor 1 and xor 8); then
//   O(8 x hd) = O*corr + P(8x8) * V(8 x hd) as hd/8 matrix products.
// No threadgroup memory, no barriers. f32 throughout. SPLIT over key tiles (grid.z) with
// un-normalized partials merged by attention_prefill_fa_combine, for occupancy.
struct PfaParams { uint nrows; uint nsplit; uint _p0; uint _p1; };
inline thread vec<float, 2> &pfa_te(thread simdgroup_float8x8 &m) {
    return reinterpret_cast<thread vec<float, 2> &>(m.thread_elements());
}
inline float pfa_rowmax(float v) { v = max(v, simd_shuffle_xor(v, 1)); return max(v, simd_shuffle_xor(v, 8)); }
inline float pfa_rowsum(float v) { v += simd_shuffle_xor(v, 1); return v + simd_shuffle_xor(v, 8); }
kernel void attention_prefill_fa_f32(
        device const float *q          [[buffer(0)]],
        device const float *key_pool   [[buffer(1)]],
        device const float *value_pool [[buffer(2)]],
        device       float *part_o     [[buffer(3)]],   // [nsplit][nrows][nh][hd]
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        device       float *part_ml    [[buffer(7)]],   // [nsplit][nrows][nh][2]
        constant PfaParams &pp         [[buffer(8)]],
        uint3 gid                      [[threadgroup_position_in_grid]],
        uint  lane_all                 [[thread_index_in_threadgroup]]) {
    const uint kv_head = gid.x, r0 = gid.y * 8u, sp = gid.z;
    const uint sg = lane_all / 32u, lane = lane_all % 32u;
    const uint hd = d.head_dim, nh = d.num_heads;
    const uint qh = kv_head * d.group + sg;
    const uint row_floats = d.kv_heads * hd, head_col = kv_head * hd;
    const float scale = as_type<float>(d.scale_bits);
    const uint nrows = pp.nrows;
    // lane -> (fm, fn): this lane holds M[fm][fn], M[fm][fn+1] (the driver-verified mapping).
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) | ((lane >> 1) & 3u);
    const uint fn = ((qid & 2u) << 1) | ((lane & 1u) << 1);
    const uint my_row = r0 + fm;
    const uint rows_here = min(8u, nrows - r0);
    const uint last_row = r0 + rows_here - 1u;
    const uint block_last = past_len_b[last_row];
    const uint my_last = past_len_b[min(my_row, last_row)];
    device const uint *slots = slots_all + last_row * d.q_start;   // q_start = slot-table stride

    const uint ntiles = block_last / 8u + 1u;
    const uint per = (ntiles + pp.nsplit - 1u) / pp.nsplit;
    const uint t_lo = sp * per, t_hi = min(ntiles, t_lo + per);

    constexpr uint MAXDB = 32u;                     // hd <= 256
    const uint ndb = hd / 8u;
    simdgroup_float8x8 O[MAXDB];
    for (uint j = 0; j < ndb; ++j) O[j] = simdgroup_float8x8(0.0f);
    float m_run = -3.4e38f, l_run = 0.0f;
    const float MASK = -3.4e38f;
    // Q rows beyond the window are clamped to the last row (their outputs are never written).
    const uint q_row0 = r0;
    device const float *qbase = q + (q_row0 * nh + qh) * hd;
    const bool full8 = rows_here == 8u;

    for (uint t = t_lo; t < t_hi; ++t) {
        const uint key0 = t * 8u;
        const ulong kbase = ulong(slots[key0]) * row_floats + head_col;
        simdgroup_float8x8 S = simdgroup_float8x8(0.0f);
        for (uint kk = 0; kk < ndb; ++kk) {
            simdgroup_float8x8 Qm, Kt;
            if (full8) {
                simdgroup_load(Qm, qbase + kk * 8u, nh * hd);
            } else {
                // partial row block: load row-by-row into thread elements, clamping rows
                const uint rq = min(my_row, last_row);
                device const float *qr = q + (rq * nh + qh) * hd + kk * 8u + fn;
                pfa_te(Qm)[0] = qr[0];
                pfa_te(Qm)[1] = qr[1];
            }
            simdgroup_load(Kt, key_pool + kbase + kk * 8u, row_floats, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(S, Qm, Kt, S);
        }
        // scale + causal mask; this lane's two scores are keys key0+fn, key0+fn+1 for row fm
        float s0 = pfa_te(S)[0] * scale, s1 = pfa_te(S)[1] * scale;
        if (key0 + fn > my_last) s0 = MASK;
        if (key0 + fn + 1u > my_last) s1 = MASK;
        const float m_new = max(m_run, pfa_rowmax(max(s0, s1)));
        const float p0 = (s0 == MASK) ? 0.0f : exp(s0 - m_new);
        const float p1 = (s1 == MASK) ? 0.0f : exp(s1 - m_new);
        const float corr = exp(m_run - m_new);
        l_run = l_run * corr + pfa_rowsum(p0 + p1);
        m_run = m_new;
        simdgroup_float8x8 P;
        pfa_te(P)[0] = p0;
        pfa_te(P)[1] = p1;
        for (uint j = 0; j < ndb; ++j) {
            pfa_te(O[j])[0] *= corr;
            pfa_te(O[j])[1] *= corr;
            simdgroup_float8x8 Vm;
            simdgroup_load(Vm, value_pool + kbase + j * 8u, row_floats);
            simdgroup_multiply_accumulate(O[j], P, Vm, O[j]);
        }
    }
    if (my_row < nrows) {
        const ulong ob = ((ulong(sp) * nrows + my_row) * nh + qh) * hd;
        for (uint j = 0; j < ndb; ++j) {
            part_o[ob + j * 8u + fn] = pfa_te(O[j])[0];
            part_o[ob + j * 8u + fn + 1u] = pfa_te(O[j])[1];
        }
        if (fn == 0u) {
            const ulong mb = ((ulong(sp) * nrows + my_row) * nh + qh) * 2u;
            part_ml[mb] = m_run;
            part_ml[mb + 1u] = l_run;
        }
    }
}

kernel void attention_prefill_fa32_f32(
        device const float *q          [[buffer(0)]],
        device const float *key_pool   [[buffer(1)]],
        device const float *value_pool [[buffer(2)]],
        device       float *part_o     [[buffer(3)]],   // [nsplit][nrows][nh][hd]
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        device       float *part_ml    [[buffer(7)]],   // [nsplit][nrows][nh][2]
        constant PfaParams &pp         [[buffer(8)]],
        uint3 gid                      [[threadgroup_position_in_grid]],
        uint  lane_all                 [[thread_index_in_threadgroup]]) {
    const uint kv_head = gid.x, r0 = gid.y * 8u, sp = gid.z;
    const uint sg = lane_all / 32u, lane = lane_all % 32u;
    const uint hd = d.head_dim, nh = d.num_heads;
    const uint qh = kv_head * d.group + sg;
    const uint row_floats = d.kv_heads * hd, head_col = kv_head * hd;
    const float scale = as_type<float>(d.scale_bits);
    const uint nrows = pp.nrows;
    // lane -> (fm, fn): this lane holds M[fm][fn], M[fm][fn+1] (the driver-verified mapping).
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) | ((lane >> 1) & 3u);
    const uint fn = ((qid & 2u) << 1) | ((lane & 1u) << 1);
    const uint my_row = r0 + fm;
    const uint rows_here = min(8u, nrows - r0);
    const uint last_row = r0 + rows_here - 1u;
    const uint block_last = past_len_b[last_row];
    const uint my_last = past_len_b[min(my_row, last_row)];
    device const uint *slots = slots_all + last_row * d.q_start;   // q_start = slot-table stride

    // 32-KEY TILES: four 8-key sub-tiles share each Q load (Q traffic / 4) and one softmax
    // update. A 32-key tile may straddle two 16-token blocks, so every 8-key sub-tile has its
    // own base slot; a sub-tile past the block's last key reuses a valid slot (it is masked).
    const uint ntiles = block_last / 32u + 1u;
    const uint per = (ntiles + pp.nsplit - 1u) / pp.nsplit;
    const uint t_lo = sp * per, t_hi = min(ntiles, t_lo + per);

    constexpr uint MAXDB = 32u;                     // hd <= 256
    const uint ndb = hd / 8u;
    simdgroup_float8x8 O[MAXDB];
    for (uint j = 0; j < ndb; ++j) O[j] = simdgroup_float8x8(0.0f);
    float m_run = -3.4e38f, l_run = 0.0f;
    const float MASK = -3.4e38f;
    device const float *qbase = q + (r0 * nh + qh) * hd;
    const bool full8 = rows_here == 8u;

    for (uint t = t_lo; t < t_hi; ++t) {
        const uint key0 = t * 32u;
        ulong kb[4];
        for (uint i = 0; i < 4u; ++i) {
            const uint k8 = key0 + i * 8u;
            kb[i] = ulong(slots[k8 <= block_last ? k8 : key0]) * row_floats + head_col;
        }
        simdgroup_float8x8 S[4];
        for (uint i = 0; i < 4u; ++i) S[i] = simdgroup_float8x8(0.0f);
        for (uint kk = 0; kk < ndb; ++kk) {
            simdgroup_float8x8 Qm;
            if (full8) {
                simdgroup_load(Qm, qbase + kk * 8u, nh * hd);
            } else {
                const uint rq = min(my_row, last_row);
                device const float *qr = q + (rq * nh + qh) * hd + kk * 8u + fn;
                pfa_te(Qm)[0] = qr[0];
                pfa_te(Qm)[1] = qr[1];
            }
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_float8x8 Kt;
                simdgroup_load(Kt, key_pool + kb[i] + kk * 8u, row_floats, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[i], Qm, Kt, S[i]);
            }
        }
        float sv[8];
        float mx = MASK;
        for (uint i = 0; i < 4u; ++i) {
            const uint k = key0 + i * 8u + fn;
            float a0 = pfa_te(S[i])[0] * scale, a1 = pfa_te(S[i])[1] * scale;
            if (k > my_last) a0 = MASK;
            if (k + 1u > my_last) a1 = MASK;
            sv[2 * i] = a0; sv[2 * i + 1] = a1;
            mx = max(mx, max(a0, a1));
        }
        const float m_new = max(m_run, pfa_rowmax(mx));
        const float corr = exp(m_run - m_new);
        float psum = 0.0f;
        simdgroup_float8x8 P[4];
        for (uint i = 0; i < 4u; ++i) {
            const float p0 = (sv[2 * i] == MASK) ? 0.0f : exp(sv[2 * i] - m_new);
            const float p1 = (sv[2 * i + 1] == MASK) ? 0.0f : exp(sv[2 * i + 1] - m_new);
            pfa_te(P[i])[0] = p0; pfa_te(P[i])[1] = p1;
            psum += p0 + p1;
        }
        l_run = l_run * corr + pfa_rowsum(psum);
        m_run = m_new;
        for (uint j = 0; j < ndb; ++j) {
            pfa_te(O[j])[0] *= corr;
            pfa_te(O[j])[1] *= corr;
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_float8x8 Vm;
                simdgroup_load(Vm, value_pool + kb[i] + j * 8u, row_floats);
                simdgroup_multiply_accumulate(O[j], P[i], Vm, O[j]);
            }
        }
    }
    if (my_row < nrows) {
        const ulong ob = ((ulong(sp) * nrows + my_row) * nh + qh) * hd;
        for (uint j = 0; j < ndb; ++j) {
            part_o[ob + j * 8u + fn] = pfa_te(O[j])[0];
            part_o[ob + j * 8u + fn + 1u] = pfa_te(O[j])[1];
        }
        if (fn == 0u) {
            const ulong mb = ((ulong(sp) * nrows + my_row) * nh + qh) * 2u;
            part_ml[mb] = m_run;
            part_ml[mb + 1u] = l_run;
        }
    }
}

kernel void attention_prefill_fa_combine(
        device const float *part_o [[buffer(0)]],
        device const float *part_ml [[buffer(1)]],
        device       float *out    [[buffer(2)]],
        constant     Dims  &d      [[buffer(3)]],
        constant PfaParams &pp     [[buffer(4)]],
        uint2 gid                  [[threadgroup_position_in_grid]],
        uint  dim                  [[thread_index_in_threadgroup]]) {
    const uint r = gid.x, qh = gid.y, nh = d.num_heads, hd = d.head_dim;
    if (dim >= hd) return;
    float mg = -3.4e38f;
    for (uint s = 0; s < pp.nsplit; ++s) mg = max(mg, part_ml[((ulong(s) * pp.nrows + r) * nh + qh) * 2u]);
    float lg = 0.0f, acc = 0.0f;
    for (uint s = 0; s < pp.nsplit; ++s) {
        const ulong mb = ((ulong(s) * pp.nrows + r) * nh + qh) * 2u;
        const float l = part_ml[mb + 1u];
        if (l <= 0.0f) continue;
        const float w = exp(part_ml[mb] - mg);
        lg += w * l;
        acc += w * part_o[((ulong(s) * pp.nrows + r) * nh + qh) * hd + dim];
    }
    out[(ulong(r) * nh + qh) * hd + dim] = lg > 0.0f ? acc / lg : 0.0f;
}

// ============================================================================================
// 8-BIT KV CACHE (2026-09-23; default on the hybrid arch, ARF_NO_KV_Q8=1 for f32). The f32 pool is 128 KiB a token on Qwen3.8-27B (16 attention
// layers x 4 kv heads x 256 x K,V x 4 B) and is what caps context on a 36 GB Mac at 32K. another engine keeps
// KV as int8 with one f32 scale per (token, kv head) — 8.1 bits an element, 4x the context in the
// same memory. Layout: k8/v8 [slot][kv_heads*hd] int8, ks/vs [slot][kv_heads] f32. Symmetric:
// scale = max|x| / 127, q = round(x / scale). Only the batched record's kernels read it (the host
// routes single-row decode through the record when Q8 is on).
struct Q8ScatterDims { uint kv_heads; uint hd; uint b; uint _p; };
kernel void kv_scatter_b_q8(
        device const float *new_k        [[buffer(0)]],   // [b, kv_heads*hd]
        device const float *new_v        [[buffer(1)]],
        device const uint  *write_slot_b [[buffer(2)]],
        device       char  *k8           [[buffer(3)]],
        device       float *ks           [[buffer(4)]],
        device       char  *v8           [[buffer(5)]],
        device       float *vs           [[buffer(6)]],
        constant Q8ScatterDims &d        [[buffer(7)]],
        uint2 gid  [[threadgroup_position_in_grid]],      // (kv_head, row)
        uint  tid  [[thread_index_in_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  lane [[thread_index_in_simdgroup]]) {
    const uint h = gid.x, r = gid.y, hd = d.hd;
    if (h >= d.kv_heads || r >= d.b) return;
    const uint src = r * d.kv_heads * hd + h * hd;
    const ulong dst = ulong(write_slot_b[r]) * d.kv_heads * hd + h * hd;
    const ulong sslot = ulong(write_slot_b[r]) * d.kv_heads + h;
    threadgroup float red[2][8];
    float mk = 0.0f, mv = 0.0f;
    for (uint i = tid; i < hd; i += 256u) { mk = max(mk, fabs(new_k[src + i])); mv = max(mv, fabs(new_v[src + i])); }
    mk = simd_max(mk); mv = simd_max(mv);
    if (lane == 0) { red[0][sgi] = mk; red[1][sgi] = mv; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mk = 0.0f; mv = 0.0f;
    for (uint s = 0; s < 8u; ++s) { mk = max(mk, red[0][s]); mv = max(mv, red[1][s]); }
    const float sk = mk > 0.0f ? mk / 127.0f : 1.0f, sv = mv > 0.0f ? mv / 127.0f : 1.0f;
    for (uint i = tid; i < hd; i += 256u) {
        k8[dst + i] = char(clamp(rint(new_k[src + i] / sk), -127.0f, 127.0f));
        v8[dst + i] = char(clamp(rint(new_v[src + i] / sv), -127.0f, 127.0f));
    }
    if (tid == 0) { ks[sslot] = sk; vs[sslot] = sv; }
}

// attention_decode_coalesced_b over the int8 pool: bindings 0 q, 1 k8, 2 v8, 3 out, 4 slots_all,
// 5 dims, 6 past_len_b, 7 ks, 8 vs. Same (nh, B) grid and 8-simdgroup split + merge.
kernel void attention_decode_coalesced_b_q8(
        device const float *q          [[buffer(0)]],
        device const char  *k8         [[buffer(1)]],
        device const char  *v8         [[buffer(2)]],
        device       float *out        [[buffer(3)]],
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        device const float *ks         [[buffer(7)]],
        device const float *vs         [[buffer(8)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]]) {
    const uint head = gid.x, r = gid.y, hd = d.head_dim;
    const uint kv_head = head / d.group;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_elems = d.kv_heads * hd, head_col = kv_head * hd;
    device const uint *slots = slots_all + r * d.q_start;
    const uint last = past_len_b[r];
    uint lo = 0u;
    if (d.window > 0u && last + 1u > d.window) lo = last + 1u - d.window;
    const uint sg = lane / SIMD_W, lin = lane % SIMD_W;
    const uint nch = (hd + (SIMD_W * 4u) - 1u) / (SIMD_W * 4u);
    device const float4 *q4 = reinterpret_cast<device const float4 *>(q + (r * d.num_heads + head) * hd);
    float4 qv[MAXCH];
    for (uint c = 0; c < nch; ++c) qv[c] = q4[lin + c * SIMD_W];
    float m_run = -3.4e38f, l_run = 0.0f;
    float4 acc[MAXCH];
    for (uint c = 0; c < nch; ++c) acc[c] = float4(0.0f);
    for (uint t = lo + sg; t <= last; t += NSG) {
        const ulong slot = slots[t];
        device const char4 *k4 = reinterpret_cast<device const char4 *>(k8 + slot * row_elems + head_col);
        const float kscale = ks[slot * d.kv_heads + kv_head];
        float partial = 0.0f;
        for (uint c = 0; c < nch; ++c) {
            const float4 kv = float4(k4[lin + c * SIMD_W]);
            partial += dot(qv[c], kv);
        }
        const float s = simd_sum(partial) * kscale * scale;
        const float m_new = max(m_run, s);
        const float corr = exp(m_run - m_new);
        const float p = exp(s - m_new);
        m_run = m_new;
        l_run = l_run * corr + p;
        device const char4 *v4 = reinterpret_cast<device const char4 *>(v8 + slot * row_elems + head_col);
        const float pv = p * vs[slot * d.kv_heads + kv_head];
        for (uint c = 0; c < nch; ++c) acc[c] = acc[c] * corr + pv * float4(v4[lin + c * SIMD_W]);
    }
    threadgroup float tg_m[NSG];
    threadgroup float tg_l[NSG];
    threadgroup float tg_acc[NSG * MAXHD];
    if (lin == 0u) { tg_m[sg] = m_run; tg_l[sg] = l_run; }
    threadgroup float *acc_row = tg_acc + sg * hd;
    for (uint c = 0; c < nch; ++c) {
        const uint d0 = (lin + c * SIMD_W) * 4u;
        acc_row[d0] = acc[c].x; acc_row[d0 + 1u] = acc[c].y; acc_row[d0 + 2u] = acc[c].z; acc_row[d0 + 3u] = acc[c].w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mg = -3.4e38f;
    for (uint s2 = 0; s2 < NSG; ++s2) mg = max(mg, tg_m[s2]);
    float lg = 0.0f;
    for (uint s2 = 0; s2 < NSG; ++s2) lg += exp(tg_m[s2] - mg) * tg_l[s2];
    const float inv_l = (lg > 0.0f) ? (1.0f / lg) : 0.0f;
    if (sg == 0u) {
        device float4 *out4 = reinterpret_cast<device float4 *>(out + (r * d.num_heads + head) * hd);
        for (uint c = 0; c < nch; ++c) {
            const uint d0 = (lin + c * SIMD_W) * 4u;
            float4 num = float4(0.0f);
            for (uint s2 = 0; s2 < NSG; ++s2) {
                const float w = exp(tg_m[s2] - mg);
                threadgroup const float *ar = tg_acc + s2 * hd;
                num += w * float4(ar[d0], ar[d0 + 1u], ar[d0 + 2u], ar[d0 + 3u]);
            }
            out4[lin + c * SIMD_W] = num * inv_l;
        }
    }
}

// 8-BIT KV, KV-HEAD SPLIT-K DECODE (2026-10-08). `attention_decode_coalesced_b_q8` runs one
// threadgroup per (query head, row): on the 27B a single stream gets 24 threadgroups for a 40-core
// GPU, each walking every key alone, and each key/value row is read once per query head of its
// group (6 times). Long agent sessions decoded at 10-31 tok/s at 33-52K tokens of context against
// 60-90 at short ones. Here a threadgroup serves one kv head for one row and one chunk of the key
// range: it stages each 16-key tile of int8 K/V (and their per-slot scales) in threadgroup memory
// ONCE, and its simdgroups score their query heads against it; grid (nkv, B, S), S = d._p1 key
// chunks, merged by the existing `attention_splitk_combine_b`. The f16 twin
// (`attention_decode_kvhead_splitk_b_f16`) assumed group >= NSG; this one handles a group smaller
// than NSG (6 on the 27B): simdgroups past the group stage tiles and score nothing.
// Bindings: 0 q, 1 k8, 2 v8, 3 out (unused), 4 slots_all, 5 dims, 6 past_len_b, 7 partials,
// 8 ks, 9 vs. Same f32 QK / online softmax / PV math as the coalesced kernel.
kernel void attention_decode_kvhead_splitk_b_q8(
        device const float *q          [[buffer(0)]],
        device const char  *k8         [[buffer(1)]],
        device const char  *v8         [[buffer(2)]],
        device       float *out        [[buffer(3)]],
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        device       float *partials   [[buffer(7)]],
        device const float *ks         [[buffer(8)]],
        device const float *vs         [[buffer(9)]],
        uint3 gid                      [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]]) {
    const uint kv_head = gid.x;
    const uint r       = gid.y;
    const uint sp      = gid.z;
    const uint splitk  = max(d._p1, 1u);
    const uint hd = d.head_dim;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_elems = d.kv_heads * hd;
    const uint head_col = kv_head * hd;
    device const uint *slots = slots_all + r * d.q_start;
    const uint last = past_len_b[r];
    uint lo = 0u;
    if (d.window > 0u && last + 1u > d.window) lo = last + 1u - d.window;

    const uint sg  = lane / SIMD_W;
    const uint lin = lane % SIMD_W;
    // Query heads per simdgroup (ceil): with group < NSG some simdgroups own none.
    const uint qpg = min((d.group + NSG - 1u) / NSG, MAXQPG);
    const uint nch = (hd + (SIMD_W * 4u) - 1u) / (SIMD_W * 4u);

    const uint total = last + 1u - lo;
    const uint chunk = (total + splitk - 1u) / splitk;
    const uint s_lo = lo + sp * chunk;
    const uint s_hi = min(s_lo + chunk, last + 1u);

    float4 qv[MAXQPG][MAXCH];
    float  m_run[MAXQPG];
    float  l_run[MAXQPG];
    float4 acc[MAXQPG][MAXCH];
    for (uint j = 0; j < qpg; ++j) {
        const uint gh = sg + j * NSG;  // query head within the group
        m_run[j] = -3.4e38f;
        l_run[j] = 0.0f;
        for (uint c = 0; c < nch; ++c) acc[j][c] = float4(0.0f);
        if (gh < d.group) {
            device const float4 *q4 = reinterpret_cast<device const float4 *>(
                q + (r * d.num_heads + kv_head * d.group + gh) * hd);
            for (uint c = 0; c < nch; ++c) qv[j][c] = q4[lin + c * SIMD_W];
        }
    }

    threadgroup char4 k4t[KVH_TILE * KVH_MAXHD / 4u];
    threadgroup char4 v4t[KVH_TILE * KVH_MAXHD / 4u];
    threadgroup float kst[KVH_TILE];
    threadgroup float vst[KVH_TILE];
    device const char4 *kp = reinterpret_cast<device const char4 *>(k8);
    device const char4 *vp = reinterpret_cast<device const char4 *>(v8);
    const uint h4 = hd / 4u;

    for (uint base = s_lo; base < s_hi; base += KVH_TILE) {
        const uint ntile = min(KVH_TILE, s_hi - base);
        for (uint i = lane; i < ntile * h4; i += NSG * SIMD_W) {
            const uint row = i / h4, col = i % h4;
            const uint kb = (slots[base + row] * row_elems + head_col) / 4u;
            k4t[row * h4 + col] = kp[kb + col];
            v4t[row * h4 + col] = vp[kb + col];
        }
        if (lane < ntile) {
            const uint sl = slots[base + lane] * d.kv_heads + kv_head;
            kst[lane] = ks[sl];
            vst[lane] = vs[sl];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < qpg; ++j) {
            if (sg + j * NSG >= d.group) continue;  // uniform per simdgroup
            for (uint t = 0; t < ntile; ++t) {
                threadgroup const char4 *kr = k4t + t * h4;
                threadgroup const char4 *vr = v4t + t * h4;
                float partial = 0.0f;
                for (uint c = 0; c < nch; ++c) partial += dot(qv[j][c], float4(kr[lin + c * SIMD_W]));
                const float sc = simd_sum(partial) * kst[t] * scale;
                const float m_new = max(m_run[j], sc);
                const float corr  = exp(m_run[j] - m_new);
                const float p     = exp(sc - m_new);
                m_run[j] = m_new;
                l_run[j] = l_run[j] * corr + p;
                const float pv = p * vst[t];
                for (uint c = 0; c < nch; ++c) acc[j][c] = acc[j][c] * corr + pv * float4(vr[lin + c * SIMD_W]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The un-normalized partial in the split-K layout; an empty chunk writes the neutral one.
    for (uint j = 0; j < qpg; ++j) {
        const uint gh = sg + j * NSG;
        if (gh >= d.group) continue;
        const uint pb = ((r * d.num_heads + kv_head * d.group + gh) * splitk + sp) * (hd + 2u);
        for (uint c = 0; c < nch; ++c) {
            const uint d0 = (lin + c * SIMD_W) * 4u;
            partials[pb + d0 + 0u] = acc[j][c].x; partials[pb + d0 + 1u] = acc[j][c].y;
            partials[pb + d0 + 2u] = acc[j][c].z; partials[pb + d0 + 3u] = acc[j][c].w;
        }
        if (lin == 0u) { partials[pb + hd] = m_run[j]; partials[pb + hd + 1u] = l_run[j]; }
    }
}

// REMOVED 2026-09-24: the simdgroup 8-bit flash kernels (`attention_prefill_fa_q8`, 16-key tiles
// dequantized into threadgroup half, and its multi-row-block forms `attention_prefill_fa_q8r` /
// `_qd`). `attention_q8_mpp_msl.metal` replaced them — int8 K/V as MPP matmul2d operands, no
// dequantize — 23.5K TTFT 224.2 -> 177.4 s interleaved. The multi-row measurements are kept at
// the host's selection site (island.rs, `p_kvq8`); the kernels themselves were removed.

kernel void attention_decode_kvhead_b_f16(
        device const float *q          [[buffer(0)]],
        device const half  *key_pool   [[buffer(1)]],
        device const half  *value_pool [[buffer(2)]],
        device       float *out        [[buffer(3)]],
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]]) {
    const uint kv_head = gid.x;     // KV-head index (0..nkv) — grid.x is nkv here, NOT nh
    const uint r       = gid.y;     // sequence index (0..B)
    const uint hd = d.head_dim;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_floats = d.kv_heads * hd;     // pool row stride (elements)
    const uint head_col = kv_head * hd;
    const uint max_slots = d.q_start;            // q_start REPURPOSED as the slot-table stride
    device const uint *slots = slots_all + r * max_slots;
    const uint last = past_len_b[r];
    uint lo = 0u;
    if (d.window > 0u && last + 1u > d.window) lo = last + 1u - d.window;

    const uint sg  = lane / SIMD_W;              // 0..NSG-1
    const uint lin = lane % SIMD_W;              // 0..31 → float4-chunk offset
    const uint nch = (hd + (SIMD_W * 4u) - 1u) / (SIMD_W * 4u);
    // L154 — q-heads owned by this SG: 1 at group=8, 2 at group=16. Host guarantees
    // group % NSG == 0 && group >= NSG, so the division is exact.
    const uint qpg = max(d.group / NSG, 1u);

    // Per-owned-q-head Q staging + private online-softmax state. At qpg==1 every loop below
    // runs exactly once, so the emitted code is the pre-L154 kernel.
    float4 qv[MAXQPG][MAXCH];
    float  m_run[MAXQPG];
    float  l_run[MAXQPG];
    float4 acc[MAXQPG][MAXCH];
    for (uint j = 0; j < qpg; ++j) {
        const uint qhj = kv_head * d.group + sg + j * NSG;
        device const float4 *q4 = reinterpret_cast<device const float4 *>(q + (r * d.num_heads + qhj) * hd);
        for (uint c = 0; c < nch; ++c) qv[j][c] = q4[lin + c * SIMD_W];
        m_run[j] = -3.4e38f;
        l_run[j] = 0.0f;
        for (uint c = 0; c < nch; ++c) acc[j][c] = float4(0.0f);
    }

    // Declared as half4 natively (NOT a cast from a half array — threadgroup half is only
    // 2B-aligned and a half4* cast would be UB). Indexed in half4 units: row stride = h4.
    threadgroup half4 k4t[KVH_TILE * KVH_MAXHD / 4u];
    threadgroup half4 v4t[KVH_TILE * KVH_MAXHD / 4u];
    device const half4 *kp = reinterpret_cast<device const half4 *>(key_pool);
    device const half4 *vp = reinterpret_cast<device const half4 *>(value_pool);
    const uint h4 = hd / 4u;                     // half4s per KV row

    for (uint base = lo; base <= last; base += KVH_TILE) {
        const uint ntile = min(KVH_TILE, last - base + 1u);
        // Cooperative stage: all 256 threads co-load the tile's rows. Thread i loads half4
        // (row=i/h4, col=i%h4): adjacent threads → adjacent half4s within a row → coalesced
        // bursts. This is the ONLY device KV read — once per tile for all 8 q-heads.
        const uint total4 = ntile * h4;
        for (uint i = lane; i < total4; i += 256u) {
            const uint row = i / h4, col = i % h4;
            const uint kb = (slots[base + row] * row_floats + head_col) / 4u;
            k4t[row * h4 + col] = kp[kb + col];
            v4t[row * h4 + col] = vp[kb + col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Score + accumulate the tile for THIS SG's q-head (keys in ascending order — the same
        // sequential online-softmax association as the CPU oracle).
        // The tile is staged ONCE and scored for EVERY q-head this SG owns. That reuse is the
        // whole lever: device KV traffic drops by `group`, so it gets BETTER at group=16.
        for (uint j = 0; j < qpg; ++j) {
            for (uint t = 0; t < ntile; ++t) {
                threadgroup const half4 *kr = k4t + t * h4;
                float partial = 0.0f;
                for (uint c = 0; c < nch; ++c) {
                    const float4 kv = float4(kr[lin + c * SIMD_W]);
                    partial += qv[j][c].x * kv.x; partial += qv[j][c].y * kv.y;
                    partial += qv[j][c].z * kv.z; partial += qv[j][c].w * kv.w;
                }
                const float s = simd_sum(partial) * scale;
                const float m_new = max(m_run[j], s);
                const float corr  = exp(m_run[j] - m_new);
                const float p     = exp(s - m_new);
                m_run[j] = m_new;
                l_run[j] = l_run[j] * corr + p;
                threadgroup const half4 *vr = v4t + t * h4;
                for (uint c = 0; c < nch; ++c) {
                    const float4 vv = float4(vr[lin + c * SIMD_W]);
                    acc[j][c] = acc[j][c] * corr + p * vv;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // tile consumed before the next overwrite
    }

    // No merge — each SG normalizes and writes each q-head it owns.
    for (uint j = 0; j < qpg; ++j) {
        const uint qhj = kv_head * d.group + sg + j * NSG;
        const float inv_l = (l_run[j] > 0.0f) ? (1.0f / l_run[j]) : 0.0f;
        device float4 *out4 = reinterpret_cast<device float4 *>(out + (r * d.num_heads + qhj) * hd);
        for (uint c = 0; c < nch; ++c) out4[lin + c * SIMD_W] = acc[j][c] * inv_l;
    }
}

// ============================================================================================
// KV-HEAD-CENTRIC + SPLIT-K (small-batch occupancy) — the B=1..16 form of the kvhead kernel.
// The plain kvhead grid (nkv, B) starves below ~64 threadgroups (measured: conc8 = 32 tgs
// REGRESSED). This variant adds grid.z = sp over `splitk = d._p1` chunks of the key range:
// (nkv, B, S) threadgroups, each staging ONLY its chunk's KV tiles (still ONE device read per
// chunk for all 8 q-heads), each SG emitting an UN-normalized partial (acc[hd], m, l) in the
// EXACT attention_decode_splitk_b partials layout — so the EXISTING, parity-green
// attention_splitk_combine_b merges per (q-head, seq) untouched. Occupancy AND the 8× traffic
// cut at small B. Same gates as kvhead: group == NSG == 8, hd%128==0, hd<=KVH_MAXHD.
kernel void attention_decode_kvhead_splitk_b_f16(
        device const float *q          [[buffer(0)]],
        device const half  *key_pool   [[buffer(1)]],
        device const half  *value_pool [[buffer(2)]],
        device       float *out        [[buffer(3)]],   // unused in pass1 (combine writes out)
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        device       float *partials   [[buffer(7)]],
        uint3 gid                      [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]]) {
    const uint kv_head = gid.x;
    const uint r       = gid.y;
    const uint sp      = gid.z;                  // this tg's KV chunk index
    const uint splitk  = max(d._p1, 1u);
    const uint hd = d.head_dim;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_floats = d.kv_heads * hd;
    const uint head_col = kv_head * hd;
    const uint max_slots = d.q_start;
    device const uint *slots = slots_all + r * max_slots;
    const uint last = past_len_b[r];
    uint lo = 0u;
    if (d.window > 0u && last + 1u > d.window) lo = last + 1u - d.window;

    const uint sg  = lane / SIMD_W;
    const uint lin = lane % SIMD_W;
    // L155 — GENERALIZED past group==NSG, the same way L154 generalized the non-splitk twin
    // (which was the ONLY one it fixed; this variant kept `qh = kv_head*group + sg` and its
    // stale "group==NSG enforced host-side" comment). At group=16 that computed just 8 of the
    // 16 q-heads and left the rest as whatever the partials buffer held — finite, plausible,
    // WRONG. Each SG now owns `qpg = group/NSG` q-heads, all sharing ONE staged KV tile, so the
    // traffic saving still scales with the group.
    const uint qpg = max(d.group / NSG, 1u);
    const uint nch = (hd + (SIMD_W * 4u) - 1u) / (SIMD_W * 4u);

    // This chunk's key range [s_lo, s_hi) within [lo, last] (even split, remainder to the last).
    const uint total = last + 1u - lo;
    const uint chunk = (total + splitk - 1u) / splitk;
    const uint s_lo = lo + sp * chunk;
    const uint s_hi = min(s_lo + chunk, last + 1u);

    // Empty chunk: every SG writes a neutral partial for ITS q-head so the combine skips it.
    if (s_lo >= s_hi) {
        for (uint j = 0; j < qpg; ++j) {
            const uint qhj = kv_head * d.group + sg + j * NSG;
            const uint pb  = ((r * d.num_heads + qhj) * splitk + sp) * (hd + 2u);
            for (uint c = 0; c < nch; ++c) {
                const uint d0 = (lin + c * SIMD_W) * 4u;
                partials[pb + d0 + 0u] = 0.0f; partials[pb + d0 + 1u] = 0.0f;
                partials[pb + d0 + 2u] = 0.0f; partials[pb + d0 + 3u] = 0.0f;
            }
            if (lin == 0u) { partials[pb + hd] = -3.4e38f; partials[pb + hd + 1u] = 0.0f; }
        }
        return;
    }

    float4 qv[MAXQPG][MAXCH];
    float  m_run[MAXQPG];
    float  l_run[MAXQPG];
    float4 acc[MAXQPG][MAXCH];
    for (uint j = 0; j < qpg; ++j) {
        const uint qhj = kv_head * d.group + sg + j * NSG;
        device const float4 *q4 = reinterpret_cast<device const float4 *>(q + (r * d.num_heads + qhj) * hd);
        for (uint c = 0; c < nch; ++c) { qv[j][c] = q4[lin + c * SIMD_W]; acc[j][c] = float4(0.0f); }
        m_run[j] = -3.4e38f;
        l_run[j] = 0.0f;
    }

    threadgroup half4 k4t[KVH_TILE * KVH_MAXHD / 4u];
    threadgroup half4 v4t[KVH_TILE * KVH_MAXHD / 4u];
    device const half4 *kp = reinterpret_cast<device const half4 *>(key_pool);
    device const half4 *vp = reinterpret_cast<device const half4 *>(value_pool);
    const uint h4 = hd / 4u;

    for (uint base = s_lo; base < s_hi; base += KVH_TILE) {
        const uint ntile = min(KVH_TILE, s_hi - base);
        const uint total4 = ntile * h4;
        for (uint i = lane; i < total4; i += 256u) {
            const uint row = i / h4, col = i % h4;
            const uint kb = (slots[base + row] * row_floats + head_col) / 4u;
            k4t[row * h4 + col] = kp[kb + col];
            v4t[row * h4 + col] = vp[kb + col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < ntile; ++t) {
            threadgroup const half4 *kr = k4t + t * h4;
            threadgroup const half4 *vr = v4t + t * h4;
            for (uint j = 0; j < qpg; ++j) {
                float partial = 0.0f;
                for (uint c = 0; c < nch; ++c) {
                    const float4 kv = float4(kr[lin + c * SIMD_W]);
                    partial += qv[j][c].x * kv.x; partial += qv[j][c].y * kv.y;
                    partial += qv[j][c].z * kv.z; partial += qv[j][c].w * kv.w;
                }
                const float s = simd_sum(partial) * scale;
                const float m_new = max(m_run[j], s);
                const float corr  = exp(m_run[j] - m_new);
                const float p     = exp(s - m_new);
                m_run[j] = m_new;
                l_run[j] = l_run[j] * corr + p;
                for (uint c = 0; c < nch; ++c) {
                    const float4 vv = float4(vr[lin + c * SIMD_W]);
                    acc[j][c] = acc[j][c] * corr + p * vv;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Emit the UN-normalized partial in the splitk layout: acc dims by lin, m/l from lin 0.
    for (uint j = 0; j < qpg; ++j) {
        const uint qhj = kv_head * d.group + sg + j * NSG;
        const uint pb  = ((r * d.num_heads + qhj) * splitk + sp) * (hd + 2u);
        for (uint c = 0; c < nch; ++c) {
            const uint d0 = (lin + c * SIMD_W) * 4u;
            partials[pb + d0 + 0u] = acc[j][c].x; partials[pb + d0 + 1u] = acc[j][c].y;
            partials[pb + d0 + 2u] = acc[j][c].z; partials[pb + d0 + 3u] = acc[j][c].w;
        }
        if (lin == 0u) { partials[pb + hd] = m_run[j]; partials[pb + hd + 1u] = l_run[j]; }
    }
}

// ============================================================================================
// f16 KV PREFILL-MIRROR. Prefill writes the f32 KV pool (WGSL kv_scatter_batched, which
// aliases keys_mtl[li]); the SEPARATE f16 pool (keys_mtl_f16[li]) is written ONLY by the decode
// island f16 scatter, so a real prompt's PREFILL slots stay zero in the f16 pool → argmax diverges
// on the first decode step that reads them. This kernel mirrors the just-written f32 pool → the f16
// pool over the valid KV range [0, n_elems), so the f16 pool sees prefill's K/V. Dispatched AFTER
// the prefill scatter (host-side, gated ARF_KV_F16), per layer, over n_elems = valid_slots*
// row_floats elements. The f32 pool + oracle are byte-UNCHANGED (this is a pure f32-read → f16-write
// copy; nothing writes the f32 pool). The store is the SAME narrowing as the f16 scatter (half(x)),
// so a slot mirrored here is bit-identical to the same slot had it been written by kv_scatter_b_f16
// — the decode f16 scatter and this prefill mirror produce a coherent f16 pool.
//
// Bindings: 0 f32 key_pool (source), 1 f32 value_pool (source), 2 f16 key_pool (dest),
//           3 f16 value_pool (dest), 4 dims {n_elems, _, _, _}. Grid: 1D over n_elems, 256/tg.
struct ConvertDims { uint n_elems; uint _p0; uint _p1; uint _p2; };

kernel void kv_f32_to_f16_convert(
        device const float *key_pool_f32   [[buffer(0)]],
        device const float *value_pool_f32 [[buffer(1)]],
        device       half  *key_pool_f16   [[buffer(2)]],
        device       half  *value_pool_f16 [[buffer(3)]],
        constant ConvertDims &d            [[buffer(4)]],
        uint gid                           [[thread_position_in_grid]]) {
    const uint i = gid;
    if (i >= d.n_elems) return;
    key_pool_f16[i]   = half(key_pool_f32[i]);    // SAME narrowing as kv_scatter_b_f16's store
    value_pool_f16[i] = half(value_pool_f32[i]);
}

// SLOT-LIST KV convert (f16-mode pool sync). In f16 mode the f16 pool is the SOURCE OF TRUTH
// (decode scatters ONLY f16); the f32 pool is a prefill staging area. These kernels sync exactly
// the listed slots' rows — never the whole pool (a whole-pool f32→f16 mirror would overwrite
// decoded-token f16 slots with STALE f32, since decode never writes f32):
//   kv_f32_to_f16_slots — EXPORT the slots a prefill batch just WROTE (f32 staging → f16 truth).
//   kv_f16_to_f32_slots — IMPORT the PAST slots a prefill batch will READ (f16 truth → f32
//                         staging, f16-rounded by construction — the same values decode reads).
// Grid: 2D [kv_dim, n_slots]; slot_list[y] = pool slot; element i = slot*kv_dim + x.
struct SlotConvDims { uint n_slots; uint kv_dim; uint _p1; uint _p2; };

kernel void kv_f32_to_f16_slots(
        device const float *key_pool_f32   [[buffer(0)]],
        device const float *value_pool_f32 [[buffer(1)]],
        device       half  *key_pool_f16   [[buffer(2)]],
        device       half  *value_pool_f16 [[buffer(3)]],
        device const uint  *slot_list      [[buffer(4)]],
        constant SlotConvDims &d           [[buffer(5)]],
        uint2 gid                          [[thread_position_in_grid]]) {
    if (gid.x >= d.kv_dim || gid.y >= d.n_slots) return;
    const uint i = slot_list[gid.y] * d.kv_dim + gid.x;
    key_pool_f16[i]   = half(key_pool_f32[i]);    // SAME narrowing as kv_scatter_b_f16's store
    value_pool_f16[i] = half(value_pool_f32[i]);
}

kernel void kv_f16_to_f32_slots(
        device       float *key_pool_f32   [[buffer(0)]],
        device       float *value_pool_f32 [[buffer(1)]],
        device const half  *key_pool_f16   [[buffer(2)]],
        device const half  *value_pool_f16 [[buffer(3)]],
        device const uint  *slot_list      [[buffer(4)]],
        constant SlotConvDims &d           [[buffer(5)]],
        uint2 gid                          [[thread_position_in_grid]]) {
    if (gid.x >= d.kv_dim || gid.y >= d.n_slots) return;
    const uint i = slot_list[gid.y] * d.kv_dim + gid.x;
    key_pool_f32[i]   = float(key_pool_f16[i]);   // exact widening — no further loss
    value_pool_f32[i] = float(value_pool_f16[i]);
}

// f16 COALESCED ATTENTION — an exact clone of attention_decode_coalesced_b with ONE difference:
// key_pool/value_pool are `device const half*`, loaded as `half4` and promoted to `float4` in-
// register before the SAME f32 QK dot + f32 online-softmax + f32 combine. The op DAG downstream of
// the load is identical to the f32 kernel; only the load type (and thus the KV DRAM bytes) changes.
// Bindings 0-6 are byte-identical to attention_decode_coalesced_b so the dispatch swaps only the
// PSO + the pool buffers (f16 views). hd % 128 == 0 required (32 lanes × half4/float4). NSG=8.
kernel void attention_decode_coalesced_b_f16(
        device const float *q          [[buffer(0)]],
        device const half  *key_pool   [[buffer(1)]],
        device const half  *value_pool [[buffer(2)]],
        device       float *out        [[buffer(3)]],
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint  lane                     [[thread_index_in_threadgroup]]) {
    const uint head = gid.x;        // head index (0..nh)
    const uint r    = gid.y;        // sequence index (0..B)
    const uint hd = d.head_dim;
    const uint kv_head = head / d.group;
    const float scale = as_type<float>(d.scale_bits);
    const uint row_floats = d.kv_heads * hd;    // elements per KV row (half elements now)
    const uint head_col = kv_head * hd;
    const uint max_slots = d.q_start;            // q_start REPURPOSED as the slot-table stride
    const uint q_base = (r * d.num_heads + head) * hd;   // q[r*q_dim + head*hd]
    device const uint *slots = slots_all + r * max_slots; // this seq's slot table

    const uint last = past_len_b[r];             // this seq's OWN past_len (decode: q_len==1, row 0)
    uint lo = 0u;
    if (d.window > 0u && last + 1u > d.window) lo = last + 1u - d.window;

    const uint sg  = lane / SIMD_W;   // 0..NSG-1
    const uint lin = lane % SIMD_W;   // 0..31
    const uint nch = (hd + (SIMD_W * 4u) - 1u) / (SIMD_W * 4u);

    // Stage Q float4 per chunk ONCE (Q stays f32; only the KV pool is f16).
    device const float4 *q4 = reinterpret_cast<device const float4 *>(q + q_base);
    float4 qv[MAXCH];
    for (uint c = 0; c < nch; ++c) qv[c] = q4[lin + c * SIMD_W];

    float m_run = -3.4e38f;
    float l_run = 0.0f;
    float4 acc[MAXCH];
    for (uint c = 0; c < nch; ++c) acc[c] = float4(0.0f);

    // SPLIT: simdgroup `sg` streams its strided subset of keys in ASCENDING order.
    for (uint t = lo + sg; t <= last; t += NSG) {
        const uint kb = slots[t] * row_floats + head_col;   // this key's row base (half index)
        device const half4 *k4 = reinterpret_cast<device const half4 *>(key_pool + kb);
        float partial = 0.0f;
        for (uint c = 0; c < nch; ++c) {
            const float4 kv = float4(k4[lin + c * SIMD_W]);   // half4 -> float4 IN-REGISTER
            partial += qv[c].x * kv.x; partial += qv[c].y * kv.y;
            partial += qv[c].z * kv.z; partial += qv[c].w * kv.w;
        }
        const float s = simd_sum(partial) * scale;          // broadcast to all 32 lins of the SG

        const float m_new = max(m_run, s);
        const float corr  = exp(m_run - m_new);
        const float p     = exp(s - m_new);
        m_run = m_new;
        l_run = l_run * corr + p;

        device const half4 *v4 = reinterpret_cast<device const half4 *>(value_pool + kb);
        for (uint c = 0; c < nch; ++c) {
            const float4 vv = float4(v4[lin + c * SIMD_W]);   // half4 -> float4 IN-REGISTER
            acc[c] = acc[c] * corr + p * vv;
        }
    }

    // ===== MERGE the NSG partials — IDENTICAL to attention_decode_coalesced_b (f32 throughout). ==
    threadgroup float tg_m[NSG];
    threadgroup float tg_l[NSG];
    threadgroup float tg_acc[NSG * MAXHD];
    if (lin == 0u) { tg_m[sg] = m_run; tg_l[sg] = l_run; }
    threadgroup float *acc_row = tg_acc + sg * hd;
    for (uint c = 0; c < nch; ++c) {
        const uint d0 = (lin + c * SIMD_W) * 4u;
        acc_row[d0 + 0u] = acc[c].x; acc_row[d0 + 1u] = acc[c].y;
        acc_row[d0 + 2u] = acc[c].z; acc_row[d0 + 3u] = acc[c].w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mg = -3.4e38f;
    for (uint s = 0; s < NSG; ++s) mg = max(mg, tg_m[s]);
    float lg = 0.0f;
    for (uint s = 0; s < NSG; ++s) lg += exp(tg_m[s] - mg) * tg_l[s];
    const float inv_l = (lg > 0.0f) ? (1.0f / lg) : 0.0f;

    if (sg == 0u) {
        const uint out_base = (r * d.num_heads + head) * hd;
        device float4 *out4 = reinterpret_cast<device float4 *>(out + out_base);
        for (uint c = 0; c < nch; ++c) {
            const uint d0 = (lin + c * SIMD_W) * 4u;
            float4 num = float4(0.0f);
            for (uint s = 0; s < NSG; ++s) {
                const float w = exp(tg_m[s] - mg);
                threadgroup float *ar = tg_acc + s * hd + d0;
                num.x += w * ar[0]; num.y += w * ar[1];
                num.z += w * ar[2]; num.w += w * ar[3];
            }
            out4[lin + c * SIMD_W] = num * inv_l;
        }
    }
}
