// B-ROW (batched) decode ops for the native Metal island batched megakernel — the B-row
// generalizations of the per-row kernels in decode_ops_msl.metal / rmsnorm_msl.metal.
// Each is bit-faithful to its m==1 sibling: at B=1 the math + reduction order are identical.
//
// Buffer layouts are ROW-MAJOR over B sequences:
//   normed/hidden : [B, hidden]
//   q             : [B, q_heads*head_dim]
//   k             : [B, k_heads*head_dim]
// Per-row position is read from a B-length `positions` buffer (rope).

#include <metal_stdlib>
using namespace metal;

// ---------- rmsnorm_b: y[r,:] = rmsnorm(x[r,:]) * weight  for all B rows ----------
// One threadgroup per (row) — grid.x = B. Identical reduction to rmsnorm_msl (simd_sum +
// threadgroup combine). dims.tokens carries B (informational; the kernel keys off tgpig).
struct Dims { uint tokens; uint hidden; float eps; uint _pad; };

kernel void rmsnorm_b(
        device const float *x       [[buffer(0)]],   // [B, hidden]
        device const float *weight  [[buffer(1)]],   // [hidden]
        device       float *y       [[buffer(2)]],   // [B, hidden]
        constant     Dims  &d       [[buffer(3)]],
        uint  row                   [[threadgroup_position_in_grid]],
        uint  lid                   [[thread_position_in_threadgroup]],
        uint  tg_size               [[threads_per_threadgroup]],
        ushort tiisg                [[thread_index_in_simdgroup]],
        ushort sgitg                [[simdgroup_index_in_threadgroup]]) {
    const uint base = row * d.hidden;
    float acc = 0.0f;
    for (uint i = lid; i < d.hidden; i += tg_size) { const float v = x[base + i]; acc += v * v; }
    acc = simd_sum(acc);
    threadgroup float sgp[32];
    if (tiisg == 0) sgp[sgitg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg_size + 31u) / 32u;
    if (lid == 0) { for (uint s = 0; s < nsg; ++s) total += sgp[s]; sgp[0] = total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = sgp[0];
    const float inv = 1.0f / sqrt(total / float(d.hidden) + d.eps);
    for (uint i = lid; i < d.hidden; i += tg_size) y[base + i] = x[base + i] * inv * weight[i];
}

// ---------- add_residual_b: hidden[r,:] += src[r,:]  for all B rows ----------
// grid.x = ceil(B*hidden / tg). dims.tokens=B, dims.hidden=hidden.
kernel void add_residual_b(
        device       float *hidden [[buffer(0)]],   // [B, hidden] (rw)
        device const float *src    [[buffer(1)]],   // [B, hidden]
        constant     Dims  &d      [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = d.tokens * d.hidden;
    if (gid < total) hidden[gid] += src[gid];
}

// ---------- copy_b: dst[i] = src[i] over B*hidden elements ----------
// Needed because the pre-trunk embedding norm must land in `hidden` — the RESIDUAL STREAM —
// and rmsnorm_b cannot write it in place (x is `device const float*`, y is `device float*`;
// aliasing both to one buffer is UB in MSL and was measured as a silent no-op). So the norm
// writes a scratch and this folds it back in ONE dispatch, rather than zero-then-add in two.
kernel void copy_b(
        device       float *dst [[buffer(0)]],
        device const float *src [[buffer(1)]],
        constant     Dims  &d   [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = d.tokens * d.hidden;
    if (gid < total) dst[gid] = src[gid];
}

// ---------- bank_copy: one ROW of a recurrent-state bank <-> one checkpoint slot ----------
// The bank row is resolved on the GPU from `s_copy[0]` — the same batch-row -> state-row map the
// stateful kernels use — so a checkpoint can never name a different row than the record wrote
// (2026-08-30: a host-side lookup did exactly that, and the restore landed on another sequence's
// row). `to_bank` = 0 checkpoints (bank -> slot), 1 restores (slot -> bank). L309.
struct BankCopy { uint n; uint slot; uint to_bank; uint _p; };
kernel void bank_copy(
        device       float    *bank   [[buffer(0)]],
        device       float    *ckpt   [[buffer(1)]],
        device const uint     *s_copy [[buffer(2)]],
        constant     BankCopy &d      [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    const uint b = s_copy[0] * d.n + gid;
    const uint c = d.slot * d.n + gid;
    if (d.to_bank != 0u) bank[b] = ckpt[c]; else ckpt[c] = bank[b];
}

// ---------- swiglu_b: out[r,i] = silu(gate[r,i]) * up[r,i], DENSE FFN, all B rows ----------
// The dense twin of moe_swiglu_b, which is laid out by (row, expert-slot, col) and cannot serve a
// dense FFN. Plain elementwise over B*inter. dims.tokens=B, dims.hidden=inter.
// silu(x) = x*sigmoid(x); no clamp needed (exp(-x) saturates gracefully, unlike geglu's cubic).
kernel void swiglu_b(
        device const float *gate [[buffer(0)]],   // [B, inter]
        device const float *up   [[buffer(1)]],   // [B, inter]
        device       float *out  [[buffer(2)]],   // [B, inter]
        constant     Dims  &d    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = d.tokens * d.hidden;
    if (gid >= total) return;
    const float x = gate[gid];
    out[gid] = (x / (1.0f + exp(-x))) * up[gid];
}

// ---------- geglu_b: out[r,i] = gelu_tanh(gate[r,i]) * up[r,i], DENSE FFN, all B rows ----------
// Gemma's activation; same bindings as swiglu_b so the dispatch only swaps the pipeline.
// The clamp before tanh is load-bearing: QAT gemma drives the gate to ~±34 and the cubic
// overflows to NaN without it (the m=1 geglu carries the same clamp for the same reason).
kernel void geglu_b(
        device const float *gate [[buffer(0)]],
        device const float *up   [[buffer(1)]],
        device       float *out  [[buffer(2)]],
        constant     Dims  &d    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = d.tokens * d.hidden;
    if (gid >= total) return;
    const float x = gate[gid];
    const float inner = clamp(0.7978845608028654f * (x + 0.044715f * x * x * x), -15.0f, 15.0f);
    out[gid] = (0.5f * x * (1.0f + tanh(inner))) * up[gid];
}

// ---------- rmsnorm_add_b: hidden[r,:] += rmsnorm(src[r,:]) * weight, all B rows ----------
// The FOUR-NORM post-attention step (gemma / muse-glimmer): normalize the sublayer output and
// THEN add it to the residual — unlike llama/qwen, which add first and fuse the next norm in
// (add_norm_b). One threadgroup per row; same simd reduction as rmsnorm_b.
// Binding order matches the m=1 `rmsnorm_add` EXACTLY (src, weight, hidden, dims) — that one is
// proven correct on this model, and having two orders for the same op is how a silent mismatch
// gets in.
kernel void rmsnorm_add_b(
        device const float *src    [[buffer(0)]],   // [B, hidden] (the sublayer output)
        device const float *weight [[buffer(1)]],   // [hidden]
        device       float *hidden [[buffer(2)]],   // [B, hidden] (rw, the residual stream)
        constant     Dims  &d      [[buffer(3)]],
        uint  row                   [[threadgroup_position_in_grid]],
        uint  lid                   [[thread_position_in_threadgroup]],
        uint  tg_size               [[threads_per_threadgroup]],
        ushort tiisg                [[thread_index_in_simdgroup]],
        ushort sgitg                [[simdgroup_index_in_threadgroup]]) {
    // Reduction idiom copied verbatim from rmsnorm_b above — same builtins, same barriers.
    const uint base = row * d.hidden;
    float acc = 0.0f;
    for (uint i = lid; i < d.hidden; i += tg_size) { const float v = src[base + i]; acc += v * v; }
    acc = simd_sum(acc);
    threadgroup float sgp[32];
    if (tiisg == 0) sgp[sgitg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg_size + 31u) / 32u;
    if (lid == 0) { for (uint s = 0; s < nsg; ++s) total += sgp[s]; sgp[0] = total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = sgp[0];
    const float inv = 1.0f / sqrt(total / float(d.hidden) + d.eps);
    for (uint i = lid; i < d.hidden; i += tg_size)
        hidden[base + i] += src[base + i] * inv * weight[i];
}

// ---------- attn_gate_mul_b: attn[r,i] *= sigmoid(gate[r,i]), all B rows ----------
// Muse Glimmer's attention output gate, batched. dims.tokens=B, dims.hidden=q_dim.
kernel void attn_gate_mul_b(
        device       float *attn [[buffer(0)]],   // [B, q_dim] (rw)
        device const float *gate [[buffer(1)]],   // [B, q_dim]
        constant     Dims  &d    [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = d.tokens * d.hidden;
    if (gid < total) attn[gid] *= 1.0f / (1.0f + exp(-gate[gid]));
}

// ---------- qg_split_b: de-interleave Qwen3.8's JOINT query+gate projection ----------
// L155. On a full-attention layer of the hybrid SSM family (Qwen3.8 / "Qwen3Next"), `wq` is ONE
// projection emitting BOTH the query and its output gate — `blk.3.attn_q.weight` is [5120, 12288]
// where 12288 = n_embd_head(256) * 2 * n_head(24). llama.cpp says so outright
// (src/models/qwen35moe.cpp:294 "Qwen3Next uses a single Q projection that outputs query + gate")
// and takes two strided views of it (:297 and :317).
//
// LAYOUT IS INTERLEAVED PER HEAD, not two contiguous halves:
//     [ q(hd) | gate(hd) ][ q(hd) | gate(hd) ] ... x n_head
// Both llama views use row stride `n_embd_head*2` and differ only by the start offset — q at 0,
// gate at `n_embd_head`. Reading it as "first half q, second half gate" would silently scramble
// heads, which is the fluent-garbage failure mode, so the interleave is the whole point here.
//
// This writes the two CONTIGUOUS [B, hd*n_head] buffers the rest of our pipeline expects
// (qk_norm_b / rope / SDPA all assume head-major contiguous q), so no downstream kernel changes.
// dims.tokens = B, dims.hidden = hd * n_head (the OUTPUT width, i.e. half the joint width).
struct QgDims { uint tokens; uint hidden; uint head_dim; uint _pad; };
kernel void qg_split_b(
        device const float *qg   [[buffer(0)]],   // [B, hd*2*n_head] joint projection
        device       float *q    [[buffer(1)]],   // [B, hd*n_head]   query out
        device       float *gate [[buffer(2)]],   // [B, hd*n_head]   gate out (pre-sigmoid)
        constant     QgDims &d   [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = d.tokens * d.hidden;
    if (gid >= total) return;
    const uint row = gid / d.hidden;          // which B row
    const uint col = gid % d.hidden;          // index within this row's [hd*n_head]
    const uint head = col / d.head_dim;       // which head
    const uint lane = col % d.head_dim;       // element within the head
    // Source row stride is DOUBLE the destination's; each head occupies 2*hd there.
    const uint src = row * (d.hidden * 2u) + head * (d.head_dim * 2u) + lane;
    q[gid]    = qg[src];
    gate[gid] = qg[src + d.head_dim];
}

// ---------- qk_norm_b: per-head RMSNorm on q,k (q/k only, no V) for all B rows ----------
// grid.x = B*(q_rows + k_rows). One threadgroup per (row, q-or-k head). q laid [B,q_rows*hd],
// k laid [B,k_rows*hd]. Same simd reduction as qk_norm. q_rows/k_rows are PER-ROW head counts.
struct QkDims { uint q_rows; uint k_rows; uint head_dim; float eps; };

kernel void qk_norm_b(
        device       float *q        [[buffer(0)]],   // [B, q_rows*hd]
        device       float *k        [[buffer(1)]],   // [B, k_rows*hd]
        device const float *q_weight [[buffer(2)]],   // [hd]
        device const float *k_weight [[buffer(3)]],   // [hd]
        constant QkDims    &d        [[buffer(4)]],
        uint  tgpig                  [[threadgroup_position_in_grid]],
        uint  lid                    [[thread_position_in_threadgroup]],
        uint  tg_size                [[threads_per_threadgroup]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]]) {
    const uint per_seq = d.q_rows + d.k_rows;   // q+k heads per sequence
    const uint seq = tgpig / per_seq;           // which sequence (0..B)
    const uint within = tgpig % per_seq;        // which head within q+k block
    const bool is_q = within < d.q_rows;
    const uint head = is_q ? within : (within - d.q_rows);
    const uint hd = d.head_dim;
    // base offset into the row-major [B, rows*hd] buffer for this (seq, head)
    const uint rows = is_q ? d.q_rows : d.k_rows;
    const uint base = (seq * rows + head) * hd;
    device float *buf = is_q ? q : k;
    device const float *w = is_q ? q_weight : k_weight;

    float acc = 0.0f;
    for (uint i = lid; i < hd; i += tg_size) { const float v = buf[base + i]; acc += v * v; }
    acc = simd_sum(acc);
    threadgroup float sgp[32];
    if (tiisg == 0) sgp[sgitg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg_size + 31u) / 32u;
    if (lid == 0) { for (uint s = 0; s < nsg; ++s) total += sgp[s]; sgp[0] = total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = sgp[0];
    const float inv = 1.0f / sqrt(total / float(hd) + d.eps);
    for (uint i = lid; i < hd; i += tg_size) buf[base + i] = buf[base + i] * inv * w[i];
}

// ---------- qkv_norm_b: per-head RMSNorm on q,k AND v, for all B rows ----------
// L360g — the B-row twin of `qkv_norm`. Gemma normalizes V as well as Q/K (`value_norm`), with V
// WEIGHTLESS — there is no v_norm tensor, only an RMS rescale. The batched record had no such
// kernel and unconditionally dispatched `qk_norm_b`, whose own comment says "q/k only —
// qwen3moe value_norm=false". So on the batched path Gemma's V was never normalized, on every
// layer. (The m=1 record discriminates per layer on `ly.value_norm` and always has.)
//
// Layout matches qk_norm_b exactly: q is [B, q_rows*hd], k is [B, k_rows*hd], v is
// [B, v_rows*hd] with v_rows == k_rows, and grid.x = B*(q_rows + k_rows + v_rows). The only
// difference from qk_norm_b is the three-way split and the weightless V branch, so the reduction
// below is character-for-character the one that kernel already uses.
struct QkvDimsB { uint q_rows; uint k_rows; uint v_rows; uint head_dim; float eps; uint _p0; uint _p1; uint _p2; };

kernel void qkv_norm_b(
        device       float *q        [[buffer(0)]],   // [B, q_rows*hd]
        device       float *k        [[buffer(1)]],   // [B, k_rows*hd]
        device       float *v        [[buffer(2)]],   // [B, v_rows*hd]
        device const float *q_weight [[buffer(3)]],   // [hd]
        device const float *k_weight [[buffer(4)]],   // [hd]
        constant QkvDimsB  &d        [[buffer(5)]],
        uint  tgpig                  [[threadgroup_position_in_grid]],
        uint  lid                    [[thread_position_in_threadgroup]],
        uint  tg_size                [[threads_per_threadgroup]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]]) {
    const uint per_seq = d.q_rows + d.k_rows + d.v_rows;
    const uint seq = tgpig / per_seq;
    const uint within = tgpig % per_seq;
    const bool is_q = within < d.q_rows;
    const bool is_k = (within >= d.q_rows) && (within < d.q_rows + d.k_rows);
    uint head = within;
    if (is_k)       head = within - d.q_rows;
    else if (!is_q) head = within - d.q_rows - d.k_rows;
    const uint hd = d.head_dim;
    const uint rows = is_q ? d.q_rows : (is_k ? d.k_rows : d.v_rows);
    const uint base = (seq * rows + head) * hd;
    device float *buf = is_q ? q : (is_k ? k : v);

    float acc = 0.0f;
    for (uint i = lid; i < hd; i += tg_size) { const float x = buf[base + i]; acc += x * x; }
    acc = simd_sum(acc);
    threadgroup float sgp[32];
    if (tiisg == 0) sgp[sgitg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg_size + 31u) / 32u;
    if (lid == 0) { for (uint s = 0; s < nsg; ++s) total += sgp[s]; sgp[0] = total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = sgp[0];
    const float inv = 1.0f / sqrt(total / float(hd) + d.eps);
    for (uint i = lid; i < hd; i += tg_size) {
        if (is_q)      buf[base + i] = buf[base + i] * inv * q_weight[i];
        else if (is_k) buf[base + i] = buf[base + i] * inv * k_weight[i];
        else           buf[base + i] = buf[base + i] * inv;  // V is WEIGHTLESS
    }
}

// ---------- rope_qk_b: fused RoPE on q,k for all B rows (per-row position) ----------
// q laid [B, q_heads*hd], k laid [B, k_heads*hd]. positions[seq] = pos of sequence seq.
// One lane per rotated element across all B sequences (grid covers B*(q+k)*rot_half region,
// but uses the q_total/k_total split exactly like rope_qk with tokens=B and the t index = seq).
struct RopeDims { uint tokens; uint q_heads; uint k_heads; uint head_dim; uint rotary_dim; uint _p0; uint _p1; uint _p2; };

// ---------- rope_qk_b_interleaved: batched RoPE, INTERLEAVED (ggml "NORM") pairing ----------
// The batched twin of rope_qk_interleaved (decode_ops_msl.metal). L155 added the INTERLEAVED rope
// for the m=1 island but NOT for this batched kernel, so the batched path silently kept applying
// SPLIT-HALF pairing to muse-glimmer — scrambling every q and k in all 52 layers. Same bug, second
// kernel; the wgpu path has had both variants since the port landed.
//     split-half (NEOX):   (x[i],  x[i + rot_half])   for i < rot_half
//     interleaved (NORM):  (x[2p], x[2p + 1])         for p < rot_half
// Identical angle per frequency, so the cos/sin tables are reused unchanged; only the stride
// between partners differs. Bindings/RopeDims are byte-identical to rope_qk_b so the dispatch
// site only swaps the pipeline.
kernel void rope_qk_b_interleaved(
        device       float *q         [[buffer(0)]],
        device       float *k         [[buffer(1)]],
        device const float *cosb      [[buffer(2)]],
        device const float *sinb      [[buffer(3)]],
        device const uint  *positions [[buffer(4)]],
        constant RopeDims  &d         [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hd = d.head_dim;
    const uint rot_half = d.rotary_dim / 2u;
    const uint q_total = d.tokens * d.q_heads * hd;
    const uint k_total = d.tokens * d.k_heads * hd;
    if (gid >= q_total + k_total) return;
    const bool is_q = gid < q_total;
    const uint local = is_q ? gid : (gid - q_total);
    const uint heads = is_q ? d.q_heads : d.k_heads;
    const uint t = local / (heads * hd);
    const uint rem = local % (heads * hd);
    const uint hh = rem / hd;
    const uint i = rem % hd;
    // EVEN elements drive the pair and write both halves; dims past rotary_dim pass through.
    if ((i & 1u) != 0u || i >= d.rotary_dim) return;
    const uint p = i / 2u;
    const uint pos = positions[t];
    const uint base = (t * heads + hh) * hd;
    const float c = cosb[pos * rot_half + p];
    const float s = sinb[pos * rot_half + p];
    device float *buf = is_q ? q : k;
    const float a = buf[base + i];
    const float bb = buf[base + i + 1u];
    buf[base + i]      = a * c - bb * s;
    buf[base + i + 1u] = bb * c + a * s;
}

kernel void rope_qk_b(
        device       float *q         [[buffer(0)]],   // [B, q_heads*hd]
        device       float *k         [[buffer(1)]],   // [B, k_heads*hd]
        device const float *cosb      [[buffer(2)]],
        device const float *sinb      [[buffer(3)]],
        device const uint  *positions [[buffer(4)]],   // [B]
        constant RopeDims  &d         [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hd = d.head_dim;
    const uint rot_half = d.rotary_dim / 2u;
    const uint q_total = d.tokens * d.q_heads * hd;
    const uint k_total = d.tokens * d.k_heads * hd;
    if (gid >= q_total + k_total) return;
    const bool is_q = gid < q_total;
    const uint local = is_q ? gid : (gid - q_total);
    const uint heads = is_q ? d.q_heads : d.k_heads;
    const uint t = local / (heads * hd);        // sequence index (0..B)
    const uint rem = local % (heads * hd);
    const uint h = rem / hd;
    const uint i = rem % hd;
    if (i >= rot_half) return;
    const uint pos = positions[t];
    const uint base = (t * heads + h) * hd;
    const float c = cosb[pos * rot_half + i];
    const float s = sinb[pos * rot_half + i];
    device float *buf = is_q ? q : k;
    const float a = buf[base + i];
    const float b = buf[base + i + rot_half];
    buf[base + i] = a * c - b * s;
    buf[base + i + rot_half] = b * c + a * s;
}

// ---------- rope_qk_b_mrope: rope_qk_b with INTERLEAVED M-RoPE positions (Qwen3.8 images) ----------
// 2026-09-27. Dispatched ONLY for a batch that carries an image (`RowOverride::mrope`); every text
// record keeps `rope_qk_b`. Same split-half (NeoX) pairing and the same cos/sin tables - the angle
// of pair i at position p is table[p * rot_half + i] - but row t has THREE positions (t, h, w) and
// pair i reads the one ggml's `ggml_mrope_cache_init` (is_imrope) assigns it:
//     sector = i % (s0+s1+s2+s3)
//     axis   = h if sector%3==1 && sector<3*s1;  w if sector%3==2 && sector<3*s2;
//              t if sector%3==0 && sector<3*s0;  else the extra axis (unused: position 0)
// For Qwen3.8's sections [11,11,10,0] over 32 pairs that is `i % 3`. A text row sent through this
// kernel has t==h==w and rotates exactly as rope_qk_b does. CPU reference:
// arf_core::model::mrope::apply_imrope.
kernel void rope_qk_b_mrope(
        device       float *q         [[buffer(0)]],   // [B, q_heads*hd]
        device       float *k         [[buffer(1)]],   // [B, k_heads*hd]
        device const float *cosb      [[buffer(2)]],
        device const float *sinb      [[buffer(3)]],
        device const uint  *positions [[buffer(4)]],   // [B*4]: t, h, w, _
        constant RopeDims  &d         [[buffer(5)]],
        device const uint  *sections  [[buffer(6)]],   // [4]
        uint gid [[thread_position_in_grid]]) {
    const uint hd = d.head_dim;
    const uint rot_half = d.rotary_dim / 2u;
    const uint q_total = d.tokens * d.q_heads * hd;
    const uint k_total = d.tokens * d.k_heads * hd;
    if (gid >= q_total + k_total) return;
    const bool is_q = gid < q_total;
    const uint local = is_q ? gid : (gid - q_total);
    const uint heads = is_q ? d.q_heads : d.k_heads;
    const uint t = local / (heads * hd);
    const uint rem = local % (heads * hd);
    const uint h = rem / hd;
    const uint i = rem % hd;
    if (i >= rot_half) return;
    const uint s0 = sections[0], s1 = sections[1], s2 = sections[2];
    const uint sect_dims = max(s0 + s1 + s2 + sections[3], 1u);
    const uint sector = i % sect_dims;
    uint pos;
    if (sector % 3u == 1u && sector < 3u * s1)      pos = positions[t * 4u + 1u];
    else if (sector % 3u == 2u && sector < 3u * s2) pos = positions[t * 4u + 2u];
    else if (sector % 3u == 0u && sector < 3u * s0) pos = positions[t * 4u + 0u];
    else                                            pos = 0u;
    const uint base = (t * heads + h) * hd;
    const float c = cosb[pos * rot_half + i];
    const float s = sinb[pos * rot_half + i];
    device float *buf = is_q ? q : k;
    const float a = buf[base + i];
    const float b = buf[base + i + rot_half];
    buf[base + i] = a * c - b * s;
    buf[base + i + rot_half] = b * c + a * s;
}

// ---------- gdn_norm_gate_b: build_norm_gated for the gated-delta-net readout ----------
// L136 (Qwen3.8 hybrid). out[h*S + i] = rmsnorm(out[h*S .. h*S+S], ssm_norm)[i] * silu(z[h*S+i])
//
// This is step 6 of the GDN layer: a PER-HEAD RMSNorm over head_v_dim (== S) with a SHARED
// `ssm_norm` gain of length S, multiplied by silu of the output-gate projection z. Mirrors the
// CPU reference exactly (ssm_qwen35.rs:2126-2132) — same per-head slicing, same eps placement,
// same silu.
//
// ONE threadgroup per v-head (grid.x = n_v_heads); each thread walks stride-wise over S.
// NOT in place: `src` (the recurrence readout) and `out` (the gated result) are separate buffers,
// so the kernel never read+writes one allocation on a Concurrent encoder.
// dims: {n_v_heads, S, eps, _}.
// ROWS ON grid.y (2026-09-27): row r of every buffer starts at r * n_v_heads * S (the tight stride
// L163 names), so ONE dispatch of (n_v_heads, rows) covers a window. The host issued one dispatch
// per row at byte offsets (L162) — 256 of them, 48 threadgroups each, per GDN layer of a 256-row
// prefill window. At grid height 1 this is the old kernel exactly (row 0, base unchanged).
struct GdnNormDims { uint n_vheads; uint s; float eps; uint _pad; };
kernel void gdn_norm_gate_b(
        device       float      *out    [[buffer(0)]],  // [rows][n_vheads * S] DESTINATION (gated)
        device const float      *src    [[buffer(1)]],  // [rows][n_vheads * S] recurrence readout
        device const float      *weight [[buffer(2)]],  // [S] ssm_norm gain
        device const float      *z      [[buffer(3)]],  // [rows][n_vheads * S] output gate (pre-silu)
        constant     GdnNormDims &d     [[buffer(4)]],
        uint2 tgp                       [[threadgroup_position_in_grid]],
        uint2 lid2                      [[thread_position_in_threadgroup]],
        uint2 tgs2                      [[threads_per_threadgroup]],
        ushort tiisg                    [[thread_index_in_simdgroup]],
        ushort sgitg                    [[simdgroup_index_in_threadgroup]]) {
    // (position attributes must be all scalar or all the same vector width, so 2-D throughout)
    const uint lid = lid2.x, tg_size = tgs2.x;
    const uint head = tgp.x;
    if (head >= d.n_vheads) return;
    const uint base = tgp.y * d.n_vheads * d.s + head * d.s;
    float acc = 0.0f;
    for (uint i = lid; i < d.s; i += tg_size) { const float v = src[base + i]; acc += v * v; }
    acc = simd_sum(acc);
    threadgroup float sgp[32];
    if (tiisg == 0) sgp[sgitg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg_size + 31u) / 32u;
    if (lid == 0) { for (uint s = 0; s < nsg; ++s) total += sgp[s]; sgp[0] = total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = sgp[0];
    const float inv = 1.0f / sqrt(total / float(d.s) + d.eps);
    for (uint i = lid; i < d.s; i += tg_size) {
        const float zi = z[base + i];
        const float sil = zi / (1.0f + exp(-zi));           // silu(z)
        out[base + i] = src[base + i] * inv * weight[i] * sil;
    }
}

// ---------- scale_inplace_b: x[0..b*h) *= layer_scalar, for ALL B rows ----------
// L363e — the B-row twin of scale_inplace. The m=1 kernel guards `gid < d.n` with n = h from the
// layer's ScaleDims; the batched record (L360f) dispatched b*h/256 threadgroups but bound THAT
// same dims buffer, so the guard stopped at h and only row 0 was ever scaled. The capture showed
// it exactly: row 1 = row 0 × 18.4 = 1/layer_scalar, to three decimals, on every channel, from
// layer 0. Two dims buffers: `d` = the record's {b, h, _, _} (bd.resid), `sc` = the layer's
// ScaleDims {_, scale_bits, _, _} — nothing new is allocated per layer.
struct ResidDimsB { uint b; uint h; uint _p0; uint _p1; };
struct ScaleDimsB { uint n; uint scale_bits; uint _p1; uint _p2; };
kernel void scale_inplace_b(
        device float *x           [[buffer(0)]],
        constant ResidDimsB &d    [[buffer(1)]],
        constant ScaleDimsB &sc   [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < d.b * d.h) x[gid] = x[gid] * as_type<float>(sc.scale_bits);
}
