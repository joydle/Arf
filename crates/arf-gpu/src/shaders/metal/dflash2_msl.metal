// DFlash 2 draft — kernels that are the draft's own.
//
// Everything the draft shares with the target (RMSNorm, the Q4_K_S row-major matmuls, SwiGLU,
// RoPE) is dispatched from the target's shaders. This file holds only what has no twin there.
#include <metal_stdlib>
using namespace metal;

// ---------- dflash_tap: one tapped target layer -> its slot of the context feature ----------
// The draft is conditioned on the target's residual stream after 5 of its layers
// (`target_layer_ids`). `fc` reads them CONCATENATED per row — [row][slot][hidden], the layout
// the checkpoint's `fc.weight` [hidden, taps*hidden] was trained on — so the tap interleaves as it
// copies and stage 3 can hand `dst` to one matmul with no gather.
//
// The ROW COUNT IS THE GRID HEIGHT, not a uniform. A record's uniforms are read when the GPU runs
// the dispatch, not when the host encodes it, so a `rows` field rewritten per step would be read
// at its LAST value by every command buffer still in flight (this codebase has paid for that
// twice). `TapDims` is therefore constant for the life of the draft: one buffer per slot, written
// once when taps are enabled.
struct TapDims { uint hidden; uint slot; uint taps; uint _p; };
kernel void dflash_tap(
        device       float   *dst [[buffer(0)]],
        device const float   *src [[buffer(1)]],
        constant     TapDims &d   [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= d.hidden) return;
    dst[(gid.y * d.taps + d.slot) * d.hidden + gid.x] = src[gid.y * d.hidden + gid.x];
}

// ---------- dflash_rmsnorm: y[row] = x[row] * rsqrt(mean(x[row]^2) + eps) * w ----------
// PLAIN weight (the draft is Qwen3-style; the published `hidden_norm.weight` averages ~1, a
// (1+w) norm's would average ~0). One thread per row: at most 128 rows of 5120, run once per
// context commit — not worth a threadgroup reduction. Uniforms arrive by `setBytes`, i.e. copied
// at ENCODE time, so a later commit can never rewrite what an in-flight one reads.
struct NormDims { uint rows; uint hidden; float eps; uint _p; };
kernel void dflash_rmsnorm(
        device const float    *x [[buffer(0)]],
        device const float    *w [[buffer(1)]],
        device       float    *y [[buffer(2)]],
        constant     NormDims &d [[buffer(3)]],
        uint row [[thread_position_in_grid]]) {
    if (row >= d.rows) return;
    device const float *xr = x + row * d.hidden;
    float ss = 0.0f;
    for (uint i = 0; i < d.hidden; ++i) ss += xr[i] * xr[i];
    const float inv = rsqrt(ss / float(d.hidden) + d.eps);
    for (uint i = 0; i < d.hidden; ++i) y[row * d.hidden + i] = xr[i] * inv * w[i];
}

// ---------- dflash_rmsnorm_tg: the same norm, ONE THREADGROUP PER ROW (2026-09-23) ----------
// `dflash_rmsnorm` above runs one THREAD per row — 8 threads on a 40-core GPU walking 5120
// floats twice, serially. The draft block calls it 11 times per cycle and the context commit once;
// the block's MLP sub-block measured ~3.2 ms of NON-matmul time (dflash2_block, stage
// subtraction), most of it this. 256 threads per row, simd + threadgroup reduction. Same maths;
// the sum is reduced in a different order.
kernel void dflash_rmsnorm_tg(
        device const float    *x [[buffer(0)]],
        device const float    *w [[buffer(1)]],
        device       float    *y [[buffer(2)]],
        constant     NormDims &d [[buffer(3)]],
        uint row  [[threadgroup_position_in_grid]],
        uint tid  [[thread_index_in_threadgroup]],
        uint ntg  [[threads_per_threadgroup]],
        uint sgi  [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (row >= d.rows) return;
    device const float *xr = x + row * d.hidden;
    float ss = 0.0f;
    for (uint i = tid; i < d.hidden; i += ntg) ss += xr[i] * xr[i];
    threadgroup float part[32];
    ss = simd_sum(ss);
    if (lane == 0) part[sgi] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nsg = (ntg + 31) / 32;
    float tot = 0.0f;
    for (uint k = 0; k < nsg; ++k) tot += part[k];
    const float inv = rsqrt(tot / float(d.hidden) + d.eps);
    for (uint i = tid; i < d.hidden; i += ntg) y[row * d.hidden + i] = xr[i] * inv * w[i];
}

// ---------- dflash_ctx_commit: one draft layer's context K/V -> its ring ----------
// For each of `rows` newly committed positions and each KV head: K = rope(rms(k_raw) * k_norm),
// V = v_raw, stored at ring slot (start + row) % window. Half-split RoPE over the whole head
// (`rope_type: default`, theta 1e7) at the LOGICAL position — the draft is a text model over
// logical positions. `inv_freq` is [head_dim/2], computed once on the host in f64.
// Ring layout [kv_head][slot][head_dim]: stage 4's attention walks one head's slots contiguously.
// Only rows < `rows` are written: a verify window commits its ACCEPTED prefix and nothing else,
// and the matmuls upstream run on padded row counts whose tail is garbage.
// One thread per (row, head).
struct CtxDims { uint rows; uint start; uint window; uint kv_heads; uint head_dim; float eps; uint _p0; uint _p1; };
kernel void dflash_ctx_commit(
        device const float   *k_raw    [[buffer(0)]],   // [rows, kv_heads*head_dim]
        device const float   *v_raw    [[buffer(1)]],
        device const float   *k_norm   [[buffer(2)]],   // [head_dim]
        device const float   *inv_freq [[buffer(3)]],   // [head_dim/2]
        device       float   *ring_k   [[buffer(4)]],   // [kv_heads, window, head_dim]
        device       float   *ring_v   [[buffer(5)]],
        constant     CtxDims &d        [[buffer(6)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint row = gid.y, head = gid.x;
    if (row >= d.rows || head >= d.kv_heads) return;
    const uint hd = d.head_dim, half_d = hd / 2;
    const uint pos = d.start + row;
    const uint slot = pos % d.window;
    device const float *ks = k_raw + (row * d.kv_heads + head) * hd;
    device const float *vs = v_raw + (row * d.kv_heads + head) * hd;
    device float *kd = ring_k + (head * d.window + slot) * hd;
    device float *vd = ring_v + (head * d.window + slot) * hd;
    float ss = 0.0f;
    for (uint i = 0; i < hd; ++i) ss += ks[i] * ks[i];
    const float inv = rsqrt(ss / float(hd) + d.eps);
    for (uint i = 0; i < half_d; ++i) {
        const float a = ks[i] * inv * k_norm[i];
        const float b = ks[i + half_d] * inv * k_norm[i + half_d];
        const float ang = float(pos) * inv_freq[i];
        const float c = cos(ang), s = sin(ang);
        kd[i] = a * c - b * s;
        kd[i + half_d] = b * c + a * s;
    }
    for (uint i = 0; i < hd; ++i) vd[i] = vs[i];
}

// =====================================================================================
// STAGE 4 — the block forward. One pass proposes `block_size` positions: row 0 is the anchor
// (the pending token), rows 1.. are mask tokens; row i's logits predict the token AT position
// anchor + i (no next-token shift). All kernels below take uniforms by `setBytes`.
// =====================================================================================

// ---------- dflash_conv: the two-tap dynamic depthwise conv ----------
// out[row, ch] = in[row, ch]   * (base[stage][0][ch] + dyn[row][(2*stage+0)*G + ch/group])
//              + in[row-1, ch] * (base[stage][1][ch] + dyn[row][(2*stage+1)*G + ch/group])   (row > 0)
//              + residual[row, ch]                                                           (stage 1)
// G = hidden / group. `dyn` is `kernel_projection(normed)`, `[rows, 4*G]`, computed ONCE per
// sub-block and used by both its stages (0 = prepare, before the projections; 1 = finish, after
// them, where the residual is added). Row 0 has no previous row: the block does not reach back
// into the context through the conv, only through attention.
struct ConvDims { uint rows; uint hidden; uint group; uint stage; };
kernel void dflash_conv(
        device const float    *x        [[buffer(0)]],
        device const float    *dyn      [[buffer(1)]],
        device const float    *base     [[buffer(2)]],   // [2][2][hidden]
        device const float    *residual [[buffer(3)]],
        device       float    *y        [[buffer(4)]],
        constant     ConvDims &d        [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint ch = gid.x, row = gid.y;
    if (ch >= d.hidden || row >= d.rows) return;
    const uint G = d.hidden / d.group, g = ch / d.group;
    device const float *dr = dyn + row * 4u * G;
    const uint s2 = d.stage * 2u;
    float v = x[row * d.hidden + ch] * (base[s2 * d.hidden + ch] + dr[s2 * G + g]);
    if (row > 0u)
        v += x[(row - 1u) * d.hidden + ch] * (base[(s2 + 1u) * d.hidden + ch] + dr[(s2 + 1u) * G + g]);
    if (d.stage == 1u) v += residual[row * d.hidden + ch];
    y[row * d.hidden + ch] = v;
}

// ---------- dflash_head_norm_rope: per-head RMS norm (plain weight) + half-split RoPE, in place ----------
// x is [rows, heads, head_dim]; row r sits at position start + r. One thread per (head, row);
// each thread owns its head, so reading the pair before writing it makes in-place safe.
struct HeadDims { uint rows; uint heads; uint head_dim; uint start; float eps; uint _p0; uint _p1; uint _p2; };
kernel void dflash_head_norm_rope(
        device       float    *x        [[buffer(0)]],
        device const float    *w        [[buffer(1)]],   // [head_dim]
        device const float    *inv_freq [[buffer(2)]],   // [head_dim/2]
        constant     HeadDims &d        [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint head = gid.x, row = gid.y;
    if (head >= d.heads || row >= d.rows) return;
    const uint hd = d.head_dim, half_d = hd / 2u;
    device float *p = x + (row * d.heads + head) * hd;
    float ss = 0.0f;
    for (uint i = 0; i < hd; ++i) ss += p[i] * p[i];
    const float inv = rsqrt(ss / float(hd) + d.eps);
    const float pos = float(d.start + row);
    for (uint i = 0; i < half_d; ++i) {
        const float a = p[i] * inv * w[i];
        const float b = p[i + half_d] * inv * w[i + half_d];
        const float ang = pos * inv_freq[i];
        const float c = cos(ang), s = sin(ang);
        p[i] = a * c - b * s;
        p[i + half_d] = b * c + a * s;
    }
}

// ---------- block attention: block queries over [context ring ++ the block's own K/V] ----------
// NOT causal inside the block: every block row sees all `rows` block keys. Over the context, the
// query at position P = ctx_len + row sees context positions [P - (window-1), ctx_len) — a
// sliding window that INCLUDES the query's own position, as the draft was trained. The ring holds
// position p at slot p % window, layout [kv_head][slot][head_dim]; it must hold exactly positions
// [0, ctx_len) (the host refuses to draft otherwise). GQA: q head h reads kv head h / (heads/kv).
//
// 🔴 MEASURED 2026-09-21: THIS IS THE SLOWEST THING IN THE ENGINE RELATIVE TO ITS FLOOR.
// 5.66 ms at 1024 context for 0.68 GFLOP over 0.039 GB — 59x the byte floor and 100x the
// arithmetic time. Neither compute- nor bandwidth-bound: the SHAPE starves the GPU. With 8
// queries x 32 heads there are only 256 output rows, and `dflash_attn_softmax` runs ONE THREAD
// PER QUERY (256 threads total), each walking ~1040 keys serially. The fix is a threadgroup per
// (query, head) with a parallel reduction over keys, as `attention_msl.metal` does for the
// target. Worth ~4% of single stream on its own (measured, same date).
//
// THREE PASSES, each one thread per OUTPUT ELEMENT. MEASURED-OUT (2026-09-20): the first build
// was one kernel, one thread per (q head, row), walking every key with an online softmax into a
// 128-wide thread-local accumulator — 13.6 ms of a 32.8 ms block at only 128 context positions
// (~3 GFLOP/s), growing with context. Key j of the `n_keys = n_ctx + rows` a query can see:
// j < n_ctx is context position lo0 + j (lo0 = the lowest position ANY row sees), else block row
// j - n_ctx. Rows further along see a shorter context; their hidden keys score -inf.
struct AttnDims { uint rows; uint heads; uint kv_heads; uint head_dim; uint ctx_len; uint window; uint n_keys; uint n_ctx; };

// scores[(row*heads + head) * n_keys + j] = q . k_j / sqrt(hd), or -inf if row cannot see key j.
kernel void dflash_attn_scores(
        device const float    *q      [[buffer(0)]],   // [rows, heads, hd]
        device const float    *bk     [[buffer(1)]],   // [rows, kv_heads, hd]
        device const float    *ring_k [[buffer(2)]],   // [kv_heads, window, hd]
        device       float    *scores [[buffer(3)]],   // [rows*heads, n_keys]
        constant     AttnDims &d      [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x, rh = gid.y;
    if (j >= d.n_keys || rh >= d.rows * d.heads) return;
    const uint row = rh / d.heads, head = rh % d.heads, hd = d.head_dim;
    const uint kvh = head / (d.heads / d.kv_heads);
    const uint lo0 = d.ctx_len - d.n_ctx;
    device const float *kp;
    if (j < d.n_ctx) {
        const uint pos = lo0 + j;
        const uint P1 = d.ctx_len + row + 1u;
        const uint lo = P1 > d.window ? P1 - d.window : 0u;
        if (pos < lo) { scores[rh * d.n_keys + j] = -INFINITY; return; }
        kp = ring_k + (kvh * d.window + pos % d.window) * hd;
    } else {
        kp = bk + ((j - d.n_ctx) * d.kv_heads + kvh) * hd;
    }
    device const float *qp = q + rh * hd;
    float4 acc = float4(0.0f);
    for (uint i = 0; i < hd; i += 4u)
        acc += float4(qp[i], qp[i + 1], qp[i + 2], qp[i + 3]) * float4(kp[i], kp[i + 1], kp[i + 2], kp[i + 3]);
    scores[rh * d.n_keys + j] = (acc.x + acc.y + acc.z + acc.w) * rsqrt(float(hd));
}

// In-place softmax of each query's n_keys scores. One thread per query (256 of them): n_keys
// scalar ops each, not worth a reduction.
kernel void dflash_attn_softmax(
        device       float    *scores [[buffer(0)]],
        constant     AttnDims &d      [[buffer(1)]],
        uint rh [[thread_position_in_grid]]) {
    if (rh >= d.rows * d.heads) return;
    device float *s = scores + rh * d.n_keys;
    float m = -INFINITY;
    for (uint j = 0; j < d.n_keys; ++j) m = max(m, s[j]);
    float sum = 0.0f;
    for (uint j = 0; j < d.n_keys; ++j) { s[j] = exp(s[j] - m); sum += s[j]; }
    const float inv = 1.0f / sum;
    for (uint j = 0; j < d.n_keys; ++j) s[j] *= inv;
}

// out[(row*heads + head) * hd + i] = sum_j w_j * v_j[i]. One thread per output element.
kernel void dflash_attn_mix(
        device const float    *scores [[buffer(0)]],
        device const float    *bv     [[buffer(1)]],   // [rows, kv_heads, hd]
        device const float    *ring_v [[buffer(2)]],
        device       float    *out    [[buffer(3)]],   // [rows, heads, hd]
        constant     AttnDims &d      [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, rh = gid.y;
    if (i >= d.head_dim || rh >= d.rows * d.heads) return;
    const uint head = rh % d.heads, hd = d.head_dim;
    const uint kvh = head / (d.heads / d.kv_heads);
    const uint lo0 = d.ctx_len - d.n_ctx;
    device const float *w = scores + rh * d.n_keys;
    float acc = 0.0f;
    for (uint j = 0; j < d.n_ctx; ++j)
        acc += w[j] * ring_v[(kvh * d.window + (lo0 + j) % d.window) * hd + i];
    for (uint j = 0; j < d.rows; ++j)
        acc += w[d.n_ctx + j] * bv[(j * d.kv_heads + kvh) * hd + i];
    out[rh * hd + i] = acc;
}

// ---------- dflash_attn_fused: ONE THREADGROUP PER (row, head), online softmax ----------
// Replaces the scores/softmax/mix trio above, which measured 5.66 ms at 1024 context against a
// 0.10 ms byte floor and 0.05 ms of arithmetic (2026-09-21) — starved, not slow: 8 queries x 32
// heads is 256 output rows, and the softmax pass ran ONE THREAD PER QUERY walking ~1040 keys.
//
// Here each threadgroup owns one (row, head) and its TG_N threads stride the keys together:
// every thread keeps a running max/sum and a partial output over the head_dim slots it owns,
// then one threadgroup reduction merges them. No scores buffer round-trip, one dispatch instead
// of three. Same arithmetic, same masking rule, f32 throughout.
constant constexpr uint DFLASH_ATTN_TG = 128;
kernel void dflash_attn_fused(
        device const float    *q      [[buffer(0)]],   // [rows, heads, hd]
        device const float    *bk     [[buffer(1)]],   // [rows, kv_heads, hd]
        device const float    *bv     [[buffer(2)]],
        device const float    *ring_k [[buffer(3)]],   // [kv_heads, window, hd]
        device const float    *ring_v [[buffer(4)]],
        device       float    *out    [[buffer(5)]],   // [rows, heads, hd]
        constant     AttnDims &d      [[buffer(6)]],
        uint  rh  [[threadgroup_position_in_grid]],
        uint  lid [[thread_position_in_threadgroup]]) {
    const uint row = rh / d.heads, head = rh % d.heads, hd = d.head_dim;
    const uint kvh = head / (d.heads / d.kv_heads);
    const uint lo0 = d.ctx_len - d.n_ctx;
    // This row's own sliding window: keys before `lo` are not visible to it.
    const uint P1 = d.ctx_len + row + 1u;
    const uint lo = P1 > d.window ? P1 - d.window : 0u;
    device const float *qp = q + rh * hd;

    threadgroup float tg_m[DFLASH_ATTN_TG];
    threadgroup float tg_l[DFLASH_ATTN_TG];

    // Pass 1: this thread's running max and sum over the keys it strides.
    float m = -INFINITY, l = 0.0f;
    for (uint j = lid; j < d.n_keys; j += DFLASH_ATTN_TG) {
        device const float *kp;
        if (j < d.n_ctx) {
            const uint pos = lo0 + j;
            if (pos < lo) continue;
            kp = ring_k + (kvh * d.window + pos % d.window) * hd;
        } else {
            kp = bk + ((j - d.n_ctx) * d.kv_heads + kvh) * hd;
        }
        float4 a = float4(0.0f);
        for (uint i = 0; i < hd; i += 4u)
            a += float4(qp[i], qp[i+1], qp[i+2], qp[i+3]) * float4(kp[i], kp[i+1], kp[i+2], kp[i+3]);
        const float sc = (a.x + a.y + a.z + a.w) * rsqrt(float(hd));
        const float mn = max(m, sc);
        l = l * exp(m - mn) + exp(sc - mn);
        m = mn;
    }
    tg_m[lid] = m; tg_l[lid] = l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Merge the per-thread (max, sum) pairs into the threadgroup's.
    for (uint s = DFLASH_ATTN_TG / 2; s > 0; s >>= 1) {
        if (lid < s) {
            const float ma = tg_m[lid], mb = tg_m[lid + s];
            const float mm = max(ma, mb);
            // exp(-inf - -inf) is nan; an empty side contributes nothing.
            const float la = (ma == -INFINITY) ? 0.0f : tg_l[lid] * exp(ma - mm);
            const float lb = (mb == -INFINITY) ? 0.0f : tg_l[lid + s] * exp(mb - mm);
            tg_m[lid] = mm; tg_l[lid] = la + lb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float gm = tg_m[0];
    const float inv = (tg_l[0] > 0.0f) ? 1.0f / tg_l[0] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: KEYS IN TILES. A first version had every one of the 128 threads recompute the
    // q.k dot product for each key so it could own one output channel — 128x redundant
    // arithmetic, and it measured 41 ms against the 23 ms three-kernel path (2026-09-21).
    // Instead: the threads compute a TILE of weights cooperatively into threadgroup memory
    // (one thread per key), then each thread owns one channel and accumulates that tile.
    threadgroup float tg_w[DFLASH_ATTN_TG];
    threadgroup uint  tg_j[DFLASH_ATTN_TG];
    float acc = 0.0f;
    for (uint base = 0; base < d.n_keys; base += DFLASH_ATTN_TG) {
        const uint j = base + lid;
        float w = 0.0f;
        uint valid = 0u;
        if (j < d.n_keys) {
            device const float *kp = nullptr;
            if (j < d.n_ctx) {
                const uint pos = lo0 + j;
                if (pos >= lo) kp = ring_k + (kvh * d.window + pos % d.window) * hd;
            } else {
                kp = bk + ((j - d.n_ctx) * d.kv_heads + kvh) * hd;
            }
            if (kp != nullptr) {
                float4 a = float4(0.0f);
                for (uint c = 0; c < hd; c += 4u)
                    a += float4(qp[c], qp[c+1], qp[c+2], qp[c+3]) * float4(kp[c], kp[c+1], kp[c+2], kp[c+3]);
                w = exp((a.x + a.y + a.z + a.w) * rsqrt(float(hd)) - gm);
                valid = 1u;
            }
        }
        tg_w[lid] = w; tg_j[lid] = valid ? j : 0xffffffffu;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < hd) {
            const uint n = min(DFLASH_ATTN_TG, d.n_keys - base);
            for (uint t = 0; t < n; ++t) {
                const uint jj = tg_j[t];
                if (jj == 0xffffffffu) continue;
                device const float *vp = (jj < d.n_ctx)
                    ? ring_v + (kvh * d.window + (lo0 + jj) % d.window) * hd
                    : bv + ((jj - d.n_ctx) * d.kv_heads + kvh) * hd;
                acc += tg_w[t] * vp[lid];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid < hd) {
        out[rh * hd + lid] = acc * inv;
    }
}

// ---------- dflash_attn_gqa: SPLIT-K over the keys, GQA-shared (2026-09-24) ----------
// `dflash_attn_fused` above gives each (row, head) its own threadgroup, so every key is read by
// rows x (heads/kv_heads) = 32 threadgroups, TWICE (the second pass recomputes q.k): 3.32 ms of
// an 11.0 ms draft block at 1,024 context (examples/dflash2_block), against ~0.33 ms for another engine's
// whole draft attention on this box. Here a threadgroup owns ONE kv head and ONE chunk of
// DFLASH_GQA_CHUNK keys; its 8 simdgroups are the block's 8 rows, each carrying that kv head's 4
// query heads, so a key is loaded once per chunk and the lanes split head_dim (4 dims a lane).
// One pass, online softmax per (row, head), partial (m, l, o) per chunk; `dflash_attn_gqa_combine`
// merges the chunks. f32 throughout; only the ORDER of the softmax sums differs from the kernel
// above. Masking is the same rule: row r sees context positions >= its own window start, and
// every block key. Requires rows == 8, heads == 4 * kv_heads, head_dim == 128 (host-checked).
constant constexpr uint DFLASH_GQA_CHUNK = 64;
kernel void dflash_attn_gqa(
        device const float    *q      [[buffer(0)]],   // [rows, heads, hd]
        device const float    *bk     [[buffer(1)]],   // [rows, kv_heads, hd]
        device const float    *bv     [[buffer(2)]],
        device const float    *ring_k [[buffer(3)]],   // [kv_heads, window, hd]
        device const float    *ring_v [[buffer(4)]],
        device       float    *o_part [[buffer(5)]],   // [chunks, rows*heads, hd]
        device       float2   *ml     [[buffer(6)]],   // [chunks, rows*heads] (max, sum)
        constant     AttnDims &d      [[buffer(7)]],
        uint2 tg   [[threadgroup_position_in_grid]],   // x = kv head, y = key chunk
        uint  lane [[thread_index_in_simdgroup]],
        uint  sg   [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = tg.x, chunk = tg.y, row = sg, hd = d.head_dim;
    constexpr uint G = 4;                               // query heads per kv head
    const uint lo0 = d.ctx_len - d.n_ctx;
    const uint P1 = d.ctx_len + row + 1u;
    const uint lo = P1 > d.window ? P1 - d.window : 0u;
    const float scale = rsqrt(float(hd));
    float4 qv[G], acc[G];
    float m[G], l[G];
    for (uint g = 0; g < G; ++g) {
        const uint head = kvh * G + g;
        qv[g] = *reinterpret_cast<device const float4 *>(q + (row * d.heads + head) * hd + lane * 4);
        acc[g] = float4(0.0f); m[g] = -INFINITY; l[g] = 0.0f;
    }
    const uint j0 = chunk * DFLASH_GQA_CHUNK, j1 = min(j0 + DFLASH_GQA_CHUNK, d.n_keys);
    for (uint j = j0; j < j1; ++j) {
        device const float *kp, *vp;
        if (j < d.n_ctx) {
            const uint pos = lo0 + j;
            if (pos < lo) continue;                     // uniform across the simdgroup (one row)
            const ulong off = (ulong(kvh) * d.window + pos % d.window) * hd;
            kp = ring_k + off; vp = ring_v + off;
        } else {
            const ulong off = (ulong(j - d.n_ctx) * d.kv_heads + kvh) * hd;
            kp = bk + off; vp = bv + off;
        }
        const float4 k4 = *reinterpret_cast<device const float4 *>(kp + lane * 4);
        const float4 v4 = *reinterpret_cast<device const float4 *>(vp + lane * 4);
        for (uint g = 0; g < G; ++g) {
            const float sc = simd_sum(dot(qv[g], k4)) * scale;
            const float mn = max(m[g], sc);
            const float c = exp(m[g] - mn), p = exp(sc - mn);
            l[g] = l[g] * c + p;
            acc[g] = acc[g] * c + p * v4;
            m[g] = mn;
        }
    }
    const uint nq = d.rows * d.heads;
    for (uint g = 0; g < G; ++g) {
        const uint qi = row * d.heads + kvh * G + g;
        *reinterpret_cast<device float4 *>(o_part + (ulong(chunk) * nq + qi) * hd + lane * 4) = acc[g];
        if (lane == 0) ml[chunk * nq + qi] = float2(m[g], l[g]);
    }
}

// out[q] = sum over chunks of o_c * exp(m_c - M) / sum of l_c * exp(m_c - M). A chunk a row could
// not see at all has l = 0 and m = -inf and contributes nothing. One threadgroup per query, one
// thread per output dim.
kernel void dflash_attn_gqa_combine(
        device const float    *o_part [[buffer(0)]],
        device const float2   *ml     [[buffer(1)]],
        device       float    *out    [[buffer(2)]],   // [rows, heads, hd]
        constant     AttnDims &d      [[buffer(3)]],
        constant     uint     &chunks [[buffer(4)]],
        uint qi  [[threadgroup_position_in_grid]],
        uint lid [[thread_position_in_threadgroup]]) {
    const uint nq = d.rows * d.heads, hd = d.head_dim;
    if (lid >= hd) return;
    float M = -INFINITY;
    for (uint c = 0; c < chunks; ++c) {
        const float2 e = ml[c * nq + qi];
        if (e.y > 0.0f) M = max(M, e.x);
    }
    float L = 0.0f, o = 0.0f;
    for (uint c = 0; c < chunks; ++c) {
        const float2 e = ml[c * nq + qi];
        if (e.y > 0.0f) {
            const float w = exp(e.x - M);
            L += e.y * w;
            o += o_part[(ulong(c) * nq + qi) * hd + lid] * w;
        }
    }
    out[qi * hd + lid] = L > 0.0f ? o / L : 0.0f;
}

// ---------- dflash_swiglu: y = silu(gate) * up ----------
kernel void dflash_swiglu(
        device const float *gate [[buffer(0)]],
        device const float *up   [[buffer(1)]],
        device       float *y    [[buffer(2)]],
        constant     uint  &n    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    const float g = gate[gid];
    y[gid] = g / (1.0f + exp(-g)) * up[gid];
}

// =================================================================================================
// THE GPU CANDIDATE SELECTOR (2026-09-22). Why it exists: the draft block ended in
// `waitUntilCompleted` and a CPU selector, and the verify record was only encoded after both — so
// the GPU sat idle for the draft's launch, its drain, the selector and the verify's encode, every
// cycle. MEASURED: the 8-row verify pass is 48% GPU-busy with a 62 ms gap before layer 0, of
// which ~42 ms is that idle. another engine never round-trips: draft -> select -> verify -> accept is one
// command graph (`draft_select_top16_sharded` / `_edges` / `_dflash` in sampling.metal). These
// three kernels are the same shape, ported against OUR buffers: the chosen window
// `[anchor, t1..t7]` lands in `tok_win`, which the verify record reads via TokenSrc::GpuBank.
// The CPU selector (`dflash_select`) is the reference these are gated against.
// =================================================================================================
struct DflashSelDims { uint vocab; uint rank; uint rows; uint n_mask; };
constant constexpr uint DFLASH_TOPK = 16u;
constant constexpr uint DFLASH_TOPK_TG = 128u;

// THE SPECIAL-TOKEN TAIL (2026-09-28, `DflashTail` in dflash2_block.rs): the lm_head rows
// `[start, start + stride)` scored by their own small matmul into `[rows, stride]` logits, so the
// draft can propose `</think>` / `<|im_end|>` without widening the candidate range. Tail index i
// IS token id start + i; only i in [lo, hi) is a candidate (lo skips the main range and anything
// below the special floor, hi stops at the vocabulary). lo == hi (all zero) = no tail.
struct DflashTailDims { uint start; uint stride; uint lo; uint hi; };

static inline bool dflash_masked(device const uint *mask, uint n_mask, uint t) {
  uint lo = 0, hi = n_mask;
  while (lo < hi) { const uint m = (lo + hi) >> 1; const uint x = mask[m];
    if (x < t) lo = m + 1; else if (x > t) hi = m; else return true; }
  return false;
}

static inline void dflash_topk_insert(thread float *lv, thread uint *li, float v, uint t) {
  if (v > lv[DFLASH_TOPK - 1] || (v == lv[DFLASH_TOPK - 1] && t < li[DFLASH_TOPK - 1])) {
    uint p = DFLASH_TOPK - 1;
    while (p > 0 && (v > lv[p - 1] || (v == lv[p - 1] && t < li[p - 1]))) {
      lv[p] = lv[p - 1]; li[p] = li[p - 1]; --p;
    }
    lv[p] = v; li[p] = t;
  }
}

// Top-16 per position over `[rows, vocab]` logits. 128 threads per row: each keeps a sorted local
// top-16 over its strided slice, then 16 rounds of a parallel argmax over the 128 list heads merge
// them. Ties -> the lower token id (the CPU selector's rule). `mask` = sorted ids the draft must
// never propose (Dflash2State::no_propose; empty unless the guard is opted in). Then the tail's
// live window, as TRUE ids (the candidates feed the codebooks and the verify window).
kernel void dflash_topk(device const float *logits [[buffer(0)]],
                        device const uint  *mask   [[buffer(1)]],
                        device uint        *cands  [[buffer(2)]],   // [rows, 16]
                        device float       *unary  [[buffer(3)]],   // [rows, 16]
                        constant DflashSelDims &d  [[buffer(4)]],
                        device const float *tail   [[buffer(5)]],   // [rows, td.stride]
                        constant DflashTailDims &td [[buffer(6)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint tg  [[threads_per_threadgroup]]) {
  device const float *L = logits + ulong(row) * d.vocab;
  float lv[DFLASH_TOPK];
  uint  li[DFLASH_TOPK];
  for (uint i = 0; i < DFLASH_TOPK; ++i) { lv[i] = -INFINITY; li[i] = 0xFFFFFFFFu; }
  for (uint t = lid; t < d.vocab; t += tg) {
    if (d.n_mask && dflash_masked(mask, d.n_mask, t)) continue;
    dflash_topk_insert(lv, li, L[t], t);
  }
  device const float *T = tail + ulong(row) * td.stride;
  for (uint i = td.lo + lid; i < td.hi; i += tg) {
    const uint t = td.start + i;
    if (d.n_mask && dflash_masked(mask, d.n_mask, t)) continue;
    dflash_topk_insert(lv, li, T[i], t);
  }
  threadgroup float sv[DFLASH_TOPK_TG * DFLASH_TOPK];
  threadgroup uint  si[DFLASH_TOPK_TG * DFLASH_TOPK];
  threadgroup uint  head[DFLASH_TOPK_TG];
  threadgroup float rv[DFLASH_TOPK_TG];
  threadgroup uint  ri[DFLASH_TOPK_TG];
  threadgroup uint  rw[DFLASH_TOPK_TG];
  for (uint i = 0; i < DFLASH_TOPK; ++i) { sv[lid * DFLASH_TOPK + i] = lv[i]; si[lid * DFLASH_TOPK + i] = li[i]; }
  head[lid] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint round = 0; round < DFLASH_TOPK; ++round) {
    const uint h = head[lid];
    rv[lid] = h < DFLASH_TOPK ? sv[lid * DFLASH_TOPK + h] : -INFINITY;
    ri[lid] = h < DFLASH_TOPK ? si[lid * DFLASH_TOPK + h] : 0xFFFFFFFFu;
    rw[lid] = lid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        const float ov = rv[lid + stride]; const uint oi = ri[lid + stride];
        if (ov > rv[lid] || (ov == rv[lid] && oi < ri[lid])) { rv[lid] = ov; ri[lid] = oi; rw[lid] = rw[lid + stride]; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
      cands[row * DFLASH_TOPK + round] = ri[0];
      unary[row * DFLASH_TOPK + round] = rv[0];
      head[rw[0]] += 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The plain top-1 with the tail: `argmax_b` has written the main range's winner to `out[row]`;
// scan the tail's live window (first maximum, ties -> the lower index) and replace the winner
// only when the tail's best is STRICTLY greater (ties -> the lower id: every tail id is above the
// main range). Dispatched only when a tail is live. One 256-thread threadgroup per row.
kernel void dflash_tail_argmax(device const float *logits [[buffer(0)]],   // [rows, main_w]
                               device const float *tail   [[buffer(1)]],   // [rows, td.stride]
                               device uint        *out    [[buffer(2)]],   // [rows]
                               constant DflashTailDims &td [[buffer(3)]],
                               constant uint &main_w      [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]],
                               uint tg  [[threads_per_threadgroup]]) {
  device const float *T = tail + ulong(row) * td.stride;
  float bv = -3.4e38f; uint bi = 0xFFFFFFFFu;
  for (uint i = td.lo + lid; i < td.hi; i += tg) {
    const float v = T[i];
    if (v > bv) { bv = v; bi = i; }
  }
  threadgroup float sv[256]; threadgroup uint si[256];
  sv[lid] = bv; si[lid] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tg / 2u; stride > 0u; stride >>= 1) {
    if (lid < stride) {
      const float ov = sv[lid + stride]; const uint oi = si[lid + stride];
      if (ov > sv[lid] || (ov == sv[lid] && oi < si[lid])) { sv[lid] = ov; si[lid] = oi; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0 && si[0] != 0xFFFFFFFFu) {
    const uint m = out[row];
    if (sv[0] > logits[ulong(row) * main_w + m]) out[row] = td.start + si[0];
  }
}

// Pairwise edge table: edges[pos][i][j] = < predecessor_codebook[pred_i] * sel[pos],
// successor_codebook[cand_j] >, for pos in 1..rows. The predecessor of position 1 is the anchor
// (`tok_in[0]`, i = 0 only); of position p > 1, the 16 candidates of p-1. One thread per
// (pos, i, j), a rank-256 dot each. Codebooks are raw bf16 bits.
kernel void dflash_edges(device const uint   *cands   [[buffer(0)]],  // [rows, 16]
                         device const float  *sel     [[buffer(1)]],  // [rows, rank]
                         device const ushort *pred_cb [[buffer(2)]],  // [vocab, rank] bf16
                         device const ushort *succ_cb [[buffer(3)]],  // [vocab, rank] bf16
                         device const uint   *tok_in  [[buffer(4)]],  // [0] = anchor
                         device float        *edges   [[buffer(5)]],  // [rows, 16, 16]
                         constant DflashSelDims &d    [[buffer(6)]],
                         uint gid [[thread_position_in_grid]]) {
  const uint j = gid % DFLASH_TOPK, i = (gid / DFLASH_TOPK) % DFLASH_TOPK, pos = gid / (DFLASH_TOPK * DFLASH_TOPK);
  if (pos == 0 || pos >= d.rows) return;
  if (pos == 1 && i != 0) return;
  const uint pred = pos == 1 ? tok_in[0] : cands[(pos - 1) * DFLASH_TOPK + i];
  const uint succ = cands[pos * DFLASH_TOPK + j];
  device const float *hs = sel + ulong(pos) * d.rank;
  device const ushort *pc = pred_cb + ulong(pred) * d.rank;
  device const ushort *sc = succ_cb + ulong(succ) * d.rank;
  float acc = 0.0f;
  for (uint r = 0; r < d.rank; ++r) {
    const float q = as_type<float>(uint(pc[r]) << 16) * hs[r];
    acc += q * as_type<float>(uint(sc[r]) << 16);
  }
  edges[(pos * DFLASH_TOPK + i) * DFLASH_TOPK + j] = acc;
}

// The greedy chain, exactly `dflash_select`: position by position, argmax_j of
// unary[pos][j] + edges[pos][prev][j], ties -> the lower token id; `prev` is the chosen
// candidate's INDEX (its token is what the edge table was built from). One thread. Writes the
// verify window: tok_win[0] = anchor, tok_win[pos] = the chosen token.
kernel void dflash_chain(device const uint  *cands  [[buffer(0)]],
                         device const float *unary  [[buffer(1)]],
                         device const float *edges  [[buffer(2)]],
                         device const uint  *tok_in [[buffer(3)]],
                         device uint        *tok_win [[buffer(4)]],
                         constant DflashSelDims &d  [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid != 0) return;
  tok_win[0] = tok_in[0];
  uint prev = 0;
  for (uint pos = 1; pos < d.rows; ++pos) {
    uint best = 0; float bs = -INFINITY; uint bt = 0xFFFFFFFFu;
    for (uint j = 0; j < DFLASH_TOPK; ++j) {
      const uint t = cands[pos * DFLASH_TOPK + j];
      if (t == 0xFFFFFFFFu) continue;
      const float s = unary[pos * DFLASH_TOPK + j] + edges[(pos * DFLASH_TOPK + prev) * DFLASH_TOPK + j];
      if (s > bs || (s == bs && t < bt)) { bs = s; bt = t; best = j; }
    }
    tok_win[pos] = bt;
    prev = best;
  }
}

// THE SAMPLED CHAIN (2026-09-26) — `dflash_chain` for a SAMPLED request (speculative sampling,
// step 2), so the sampled draft keeps the GPU-select path instead of waiting for the CPU selector.
// MEASURED before it (measured 2026-09-26): the CPU-selected sampled draft accepted 3.770
// tokens a window vs the point mass's 3.453 on 60 sampled requests, yet decoded 1-3% SLOWER — it
// leaves this path for a waited draft + host selector. NOT MEASURED with this kernel yet.
//
// Mirrors `dflash_select`'s sampled branch (`arf_core::sampling::sampled_draft_pick`) op for op:
// score = unary[pos][j] + edges[pos][prev][j]; the argmax (ties -> the lower id) is the fallback;
// candidates in TOKEN-ID order; m = max; w = exp((s - m) / max(T, 1e-6)); z = sequential sum;
// q = w / z unless z is NaN/inf/<= 0, then the point mass q = [(best, 1.0)]; the draw is
// `draw_from`'s inverse CDF over q with the host-keyed uniform (keyed_uniform(seed, past_len + pos,
// DRAFT) — computed on the CPU, never here), falling back to the last positive entry and then to
// the argmax. `prev` = the DRAWN candidate's index (its token is what the edge table was built
// from). `precise::` exp/divide and bit-level finiteness tests because the library compiles with
// fast math, which may drop NaN/inf checks; the rare last-bit exp difference from Rust's is the
// one expected CPU/GPU disagreement. Outputs the verify window and, per position, q as
// `[rows, 16]` ids (0xFFFFFFFF = no entry) + probabilities — the host reads them back after the
// verify record and hands them to the min(1, p/q) acceptor. Row 0 of q is unused. One thread.
static inline bool dflash_finite(float x) { return (as_type<uint>(x) & 0x7F800000u) != 0x7F800000u; }
static inline bool dflash_nan(float x) {
  return (as_type<uint>(x) & 0x7F800000u) == 0x7F800000u && (as_type<uint>(x) & 0x007FFFFFu) != 0u;
}
kernel void dflash_chain_sampled(device const uint  *cands   [[buffer(0)]],
                                 device const float *unary   [[buffer(1)]],
                                 device const float *edges   [[buffer(2)]],
                                 device const uint  *tok_in  [[buffer(3)]],
                                 device uint        *tok_win [[buffer(4)]],
                                 constant DflashSelDims &d   [[buffer(5)]],
                                 constant float     *unif    [[buffer(6)]],  // [rows]; [0] unused
                                 constant float     &temp    [[buffer(7)]],
                                 device uint        *q_id    [[buffer(8)]],  // [rows, 16]
                                 device float       *q_p     [[buffer(9)]],  // [rows, 16]
                                 uint gid [[thread_position_in_grid]]) {
  if (gid != 0) return;
  tok_win[0] = tok_in[0];
  for (uint j = 0; j < DFLASH_TOPK; ++j) { q_id[j] = 0xFFFFFFFFu; q_p[j] = 0.0f; }
  const float tdiv = max(temp, 1e-6f);
  uint prev = 0;
  for (uint pos = 1; pos < d.rows; ++pos) {
    device const uint *C = cands + pos * DFLASH_TOPK;
    float s[DFLASH_TOPK];
    uint ord[DFLASH_TOPK];  // candidate indices, token id ascending
    uint n = 0;
    uint best = 0; float bs = -INFINITY; uint bt = 0xFFFFFFFFu;
    for (uint j = 0; j < DFLASH_TOPK; ++j) {
      const uint t = C[j];
      if (t == 0xFFFFFFFFu) continue;
      s[j] = unary[pos * DFLASH_TOPK + j] + edges[(pos * DFLASH_TOPK + prev) * DFLASH_TOPK + j];
      if (s[j] > bs || (s[j] == bs && t < bt)) { bs = s[j]; bt = t; best = j; }
      uint p = n;
      while (p > 0 && C[ord[p - 1]] > t) { ord[p] = ord[p - 1]; --p; }
      ord[p] = j;
      ++n;
    }
    // softmax(score / T) in id order — f32::max semantics (a NaN operand yields the other)
    float m = -INFINITY;
    for (uint i = 0; i < n; ++i) { const float x = s[ord[i]]; if (!dflash_nan(x) && x > m) m = x; }
    float w[DFLASH_TOPK];
    float z = 0.0f;
    for (uint i = 0; i < n; ++i) { w[i] = precise::exp(precise::divide(s[ord[i]] - m, tdiv)); z += w[i]; }
    uint qt[DFLASH_TOPK]; float qv[DFLASH_TOPK]; uint qj[DFLASH_TOPK]; uint nq;
    if (dflash_finite(z) && z > 0.0f) {
      for (uint i = 0; i < n; ++i) { qt[i] = C[ord[i]]; qv[i] = precise::divide(w[i], z); qj[i] = ord[i]; }
      nq = n;
    } else {
      qt[0] = bt; qv[0] = 1.0f; qj[0] = best; nq = 1;
    }
    // draw_from(q, u): inverse CDF; no mass -> the argmax
    uint pick = bt, pick_j = best;
    float total = 0.0f;
    for (uint i = 0; i < nq; ++i) total += qv[i];
    if (!(dflash_nan(total) || total <= 0.0f)) {
      const float target = unif[pos] * total;
      float cum = 0.0f;
      bool hit = false;
      for (uint i = 0; i < nq; ++i) {
        cum += qv[i];
        if (cum > target) { pick = qt[i]; pick_j = qj[i]; hit = true; break; }
      }
      if (!hit) {
        for (uint i = nq; i > 0; --i) {
          if (qv[i - 1] > 0.0f) { pick = qt[i - 1]; pick_j = qj[i - 1]; break; }
        }
      }
    }
    for (uint i = 0; i < DFLASH_TOPK; ++i) {
      q_id[pos * DFLASH_TOPK + i] = i < nq ? qt[i] : 0xFFFFFFFFu;
      q_p[pos * DFLASH_TOPK + i] = i < nq ? qv[i] : 0.0f;
    }
    tok_win[pos] = pick;
    prev = pick_j;
  }
}

