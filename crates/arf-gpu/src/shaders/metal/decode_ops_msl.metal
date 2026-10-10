// MSL ports of the small decode element-wise ops, for the per-token island megakernel.
// Each is a bit-close port of its WGSL twin (parity-gated vs a CPU oracle). They use the
// megakernel reduction idiom (simd_sum + threadgroup combine) where a reduction is needed.
//
// Ops: geglu (gemma FFN gate), qk_norm (per-head RMSNorm on q+k), rope_qk (fused RoPE on
// q+k), rmsnorm_add (fused norm + residual-add). Bindings match the WGSL kernels exactly.

#include <metal_stdlib>
using namespace metal;

// ---------- GeGLU: out[i] = gelu_tanh(gate[i]) * up[i] (gemma FFN) ----------
struct GegluDims { uint n; uint _p0; uint _p1; uint _p2; };
constant float SQRT_2_OVER_PI = 0.7978845608028654f;

kernel void geglu(
        device const float *gate [[buffer(0)]],
        device const float *up   [[buffer(1)]],
        device       float *out  [[buffer(2)]],
        constant GegluDims &d    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    const float x = gate[gid];
    // clamp before tanh (QAT gemma drives the gate to ~±34 → cubic overflows → NaN).
    const float inner = clamp(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x), -15.0f, 15.0f);
    const float gelu = 0.5f * x * (1.0f + tanh(inner));
    out[gid] = gelu * up[gid];
}

// ---------- SwiGLU: out[i] = silu(gate[i]) * up[i] (llama / qwen / muse-glimmer FFN) ----------
// The activation twin of geglu above. WHICH ONE A MODEL USES IS NOT COSMETIC: running a SiLU
// model through the GELU kernel corrupts every FFN in every layer — coherent-looking machinery
// with a wrong distribution. batch.rs already learned this once (see its GateAct match: "k.swiglu
// ran Gemma's FFN with the WRONG activation"); the island had only the GELU kernel because every
// dense model it had served until now was Gemma.
//
// silu(x) = x * sigmoid(x). No clamp is needed (unlike geglu's tanh cubic, which overflows on
// QAT gemma's ~±34 gate): expf(-x) saturates to 0/inf gracefully and the product stays finite.
// Same Dims/bindings as geglu so the dispatch site only swaps the pipeline.
kernel void swiglu(
        device const float *gate [[buffer(0)]],
        device const float *up   [[buffer(1)]],
        device       float *out  [[buffer(2)]],
        constant GegluDims &d    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    const float x = gate[gid];
    out[gid] = (x / (1.0f + exp(-x))) * up[gid];
}

// ---------- attn_gate_mul: Muse Glimmer's attention output gate ----------
// `attn[i] *= sigmoid(gate[i])`, applied after SDPA and BEFORE o_proj
// (llama.cpp src/models/muse-glimmer.cpp:137-139). `gate` is a separate projection of the
// PRE-attention hidden state, so it is q_dim-wide exactly like `attn`.
//
// Fused rather than sigmoid-then-multiply: the gate is dead after this, and a separate
// sigmoid pass would cost an extra q_dim round-trip per layer per token on the hot path.
// In place on `attn` — nothing reads the un-gated attention output afterwards.
// The MSL twin of shaders/attn_gate_mul.wgsl; same math, same binding order.
struct AttnGateDims { uint n; };

kernel void attn_gate_mul(
        device       float *attn [[buffer(0)]],
        device const float *gate [[buffer(1)]],
        constant AttnGateDims &d [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    attn[gid] *= 1.0f / (1.0f + exp(-gate[gid]));
}

// ---------- qk_norm: per-head RMSNorm on q and k (rows of head_dim), in place ----------
struct QkDims { uint q_rows; uint k_rows; uint head_dim; float eps; };

kernel void qk_norm(
        device       float *q        [[buffer(0)]],
        device       float *k        [[buffer(1)]],
        device const float *q_weight [[buffer(2)]],
        device const float *k_weight [[buffer(3)]],
        constant QkDims    &d        [[buffer(4)]],
        uint  tgpig                  [[threadgroup_position_in_grid]],
        uint  lid                    [[thread_position_in_threadgroup]],
        uint  tg_size                [[threads_per_threadgroup]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig;
    if (row >= d.q_rows + d.k_rows) return;
    const bool is_q = row < d.q_rows;
    const uint local_row = is_q ? row : (row - d.q_rows);
    const uint hd = d.head_dim;
    const uint base = local_row * hd;
    device float *buf = is_q ? q : k;
    device const float *w = is_q ? q_weight : k_weight;

    float acc = 0.0f;
    for (uint i = lid; i < hd; i += tg_size) {
        const float v = buf[base + i];
        acc += v * v;
    }
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
        buf[base + i] = buf[base + i] * inv * w[i];
    }
}

// ---------- qkv_norm: fused per-head RMSNorm on q,k (weighted) + v (weightless) ----------
// Gemma-4 default: one threadgroup per row over [q_rows)+[k_rows)+[v_rows) of head_dim.
// Bindings match qkv_norm.wgsl: 0 q, 1 k, 2 v, 3 q_weight, 4 k_weight, 5 dims
// {q_rows, k_rows, v_rows, head_dim, eps, _, _, _}.
struct QkvDims { uint q_rows; uint k_rows; uint v_rows; uint head_dim; float eps; uint _p0; uint _p1; uint _p2; };

kernel void qkv_norm(
        device       float *q        [[buffer(0)]],
        device       float *k        [[buffer(1)]],
        device       float *v        [[buffer(2)]],
        device const float *q_weight [[buffer(3)]],
        device const float *k_weight [[buffer(4)]],
        constant QkvDims   &d        [[buffer(5)]],
        uint  row                    [[threadgroup_position_in_grid]],
        uint  lid                    [[thread_position_in_threadgroup]],
        uint  tg_size                [[threads_per_threadgroup]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]]) {
    const uint total = d.q_rows + d.k_rows + d.v_rows;
    if (row >= total) return;
    const bool is_q = row < d.q_rows;
    const bool is_k = (row >= d.q_rows) && (row < d.q_rows + d.k_rows);
    uint local_row = row;
    if (is_k) local_row = row - d.q_rows;
    else if (!is_q) local_row = row - d.q_rows - d.k_rows;
    const uint hd = d.head_dim;
    const uint base = local_row * hd;
    device float *buf = is_q ? q : (is_k ? k : v);

    float acc = 0.0f;
    for (uint i = lid; i < hd; i += tg_size) { const float x = buf[base + i]; acc += x * x; }
    acc = simd_sum(acc);
    threadgroup float sgp[32];
    if (tiisg == 0) sgp[sgitg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_ss = 0.0f;
    const uint nsg = (tg_size + 31u) / 32u;
    if (lid == 0) { for (uint s = 0; s < nsg; ++s) total_ss += sgp[s]; sgp[0] = total_ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sqrt(sgp[0] / float(hd) + d.eps);
    for (uint i = lid; i < hd; i += tg_size) {
        if (is_q) buf[base + i] = buf[base + i] * inv * q_weight[i];
        else if (is_k) buf[base + i] = buf[base + i] * inv * k_weight[i];
        else buf[base + i] = buf[base + i] * inv; // v weightless
    }
}

// ---------- rope_qk: fused RoPE on q and k (one lane per rotated element) ----------
struct RopeDims { uint tokens; uint q_heads; uint k_heads; uint head_dim; uint rotary_dim; uint _p0; uint _p1; uint _p2; };

kernel void rope_qk(
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
    uint local = is_q ? gid : (gid - q_total);
    const uint heads = is_q ? d.q_heads : d.k_heads;
    const uint t = local / (heads * hd);
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

// ---------- rope_qk_interleaved: RoPE with INTERLEAVED (ggml "NORM") pairing ----------
// The twin of rope_qk above, which is split-half ("NEOX"). THE ONLY DIFFERENCE is which two
// elements form a rotation pair:
//     split-half (NEOX):   (x[i],  x[i + rot_half])   for i < rot_half
//     interleaved (NORM):  (x[2p], x[2p + 1])         for p < rot_half
// The ANGLE for frequency p is identical, so the cos/sin tables are reused unchanged
// (`cos[pos*rot_half + p]`). Only the stride between partners differs.
//
// WHY THE ISLAND NEEDS IT: llama.cpp returns LLAMA_ROPE_TYPE_NORM for muse-glimmer, whose
// converter permutes q/k so the GGUF is ALREADY interleaved. Running split-half rope over those
// weights scrambles every q and k in every layer — finite, plausible-looking values with a wrong
// distribution, not a crash. The wgpu path has had rope_qk_interleaved.wgsl since the port
// landed; the island only ever compiled the split-half kernel, so the megakernel was silently
// applying the wrong pairing on all 52 layers.
//
// Same RopeDims/bindings as rope_qk so the dispatch site only swaps the pipeline.
kernel void rope_qk_interleaved(
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
    uint local = is_q ? gid : (gid - q_total);
    const uint heads = is_q ? d.q_heads : d.k_heads;
    const uint t = local / (heads * hd);
    const uint rem = local % (heads * hd);
    const uint h = rem / hd;
    const uint i = rem % hd;
    // EVEN elements drive the pair and write both halves; dims past rotary_dim pass through.
    if ((i & 1u) != 0u || i >= d.rotary_dim) return;
    const uint p = i / 2u;

    const uint pos = positions[t];
    const uint base = (t * heads + h) * hd;
    const float c = cosb[pos * rot_half + p];
    const float s = sinb[pos * rot_half + p];
    device float *buf = is_q ? q : k;
    const float a = buf[base + i];
    const float b = buf[base + i + 1u];
    buf[base + i]      = a * c - b * s;
    buf[base + i + 1u] = b * c + a * s;
}

// ---------- softcap: logit[i] = cap·tanh(logit[i]/cap) (gemma final-logit softcap) ----------
struct SoftcapDims { uint n; float cap; uint _p0; uint _p1; };
kernel void softcap(
        device float *logits  [[buffer(0)]],
        constant SoftcapDims &d [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < d.n) logits[gid] = d.cap * tanh(logits[gid] / d.cap);
}

// ---------- argmax: greedy sample over logits[vocab] → out[0] (lowest index on ties) ----------
// Single threadgroup; each lane scans a strided slice (value + lowest-index), simd-then-
// threadgroup reduction. Matches sample.wgsl's greedy path (strictly-greater wins, tie→lower idx).
struct ArgmaxDims { uint vocab; uint _p0; uint _p1; uint _p2; };
kernel void argmax(
        device const float *logits [[buffer(0)]],
        device       uint  *out    [[buffer(1)]],
        constant ArgmaxDims &d     [[buffer(2)]],
        uint  lid                  [[thread_position_in_threadgroup]],
        uint  tg_size              [[threads_per_threadgroup]]) {
    float bv = -3.4e38f; uint bi = 0u;
    for (uint i = lid; i < d.vocab; i += tg_size) {
        const float v = logits[i];
        if (v > bv) { bv = v; bi = i; }
    }
    threadgroup float sv[256]; threadgroup uint si[256];
    sv[lid] = bv; si[lid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            const float ov = sv[lid + stride]; const uint oi = si[lid + stride];
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; sv[lid] = bv; si[lid] = bi; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) out[0] = si[0];
}

// ---------- store_word: out_tokens[dims[0]] = src[0] (island collect → no wgpu copy) ----------
struct StoreDims { uint slot; uint _p0; uint _p1; uint _p2; };
kernel void store_word(
        device const uint  *src        [[buffer(0)]],
        device       uint  *out_tokens [[buffer(1)]],
        constant StoreDims &d          [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid == 0) out_tokens[d.slot] = src[0];
}

// ---------- embed_tok: hidden[i] = bf16(table[tok[0]*hidden + i]) * scale (gemma √hidden) ----------
// Steady-decode on-GPU feedback embed: reads the sampled id from tok[0]. Runs FIRST on the
// island so the whole token is single-queue (kills the wgpu→island cross-queue handoff).
// Bindings match embed_tok_bf16.wgsl: 0 table(u32 bf16-packed), 1 out(f32 hidden), 2 tok(u32),
// 3 dims{hidden, scale_bits, _, _}.
struct EmbedDims { uint hidden; uint scale_bits; uint _p1; uint _p2; };

kernel void embed_tok(
        device const uint  *table [[buffer(0)]],
        device       float *out   [[buffer(1)]],
        device const uint  *tok   [[buffer(2)]],
        constant EmbedDims &d     [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= d.hidden) return;
    const uint elem = tok[0] * d.hidden + gid;
    const uint word = table[elem >> 1];
    const uint lo = ((elem & 1u) == 0u) ? (word & 0xffffu) : (word >> 16);
    out[gid] = as_type<float>(lo << 16) * as_type<float>(d.scale_bits);
}

// ---------- embed_tok_q4k: the same gather from the GGUF's OWN Q4_K table (2026-09-23) ----------
// Qwen3.8-27B ships token_embd as Q4_K (0.67 GB); widened to bf16 it was 2.54 GB held resident to
// gather 1-8 rows a step. Same ABI as `embed_tok` (the host compiles this under the `embed_tok`
// key when the table is Q4_K, so every call site is unchanged): 0 table (Q4_K blocks, 144 B per
// 256), 1 out, 2 tok, 3 dims. Element-for-element ggml `dequantize_row_q4_K`:
//   y = (d * sc) * q - dmin * m, sub-block j of 32: q from byte 16 + (j/2)*32 + l, low nibble for
//   even j; (sc, m) = get_scale_min_k4(j). Exact f32 — the bf16 table rounded these values.
inline float q4k_elem(device const uchar *blk, uint i) {
    const uint j = i / 32u, l = i % 32u;
    const float dd = float(as_type<half>(ushort(blk[0] | (uint(blk[1]) << 8))));
    const float dm = float(as_type<half>(ushort(blk[2] | (uint(blk[3]) << 8))));
    device const uchar *sc = blk + 4;
    uint s, m;
    if (j < 4u) { s = sc[j] & 63u; m = sc[j + 4u] & 63u; }
    else { s = (sc[j + 4u] & 0xFu) | ((sc[j - 4u] >> 6) << 4); m = (sc[j + 4u] >> 4) | ((sc[j] >> 6) << 4); }
    const uint qb = blk[16u + (j / 2u) * 32u + l];
    const uint q = (j & 1u) ? (qb >> 4) : (qb & 0xFu);
    return dd * float(s) * float(q) - dm * float(m);
}
kernel void embed_tok_q4k(
        device const uchar *table [[buffer(0)]],
        device       float *out   [[buffer(1)]],
        device const uint  *tok   [[buffer(2)]],
        constant EmbedDims &d     [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= d.hidden) return;
    const ulong blk = ulong(tok[0]) * (d.hidden / 256u) + gid / 256u;
    out[gid] = q4k_elem(table + blk * 144ul, gid % 256u) * as_type<float>(d.scale_bits);
}

// ---------- gemv_q8: int8 per-row GEMV (c[n] = scale[n]·(a·qᵀ)) — for the lm_head ----------
// Parity-safe lm_head (Q8 ≈ bf16 for greedy argmax) at half the bf16 bandwidth. One
// threadgroup per output column, simd_sum reduction. Weights packed 4 int8/u32, k/4 words/row.
// Bindings match matmul_vec_q8.wgsl: 0 a(f32), 1 q(u32-packed i8), 2 c(f32), 3 dims{m,k,n},
// 4 scale(f32 per row).
struct Q8Dims { uint m; uint k; uint n; uint _pad; };

// One SIMDGROUP (32 lanes) per output column, MANY columns per threadgroup, grid-strided so
// the launch is ~thousands of threadgroups not 262144 (vocab). Each simdgroup reduces its
// column's row with simd_sum (no threadgroup barrier). Far better occupancy than 1-tg/column.
kernel void gemv_q8(
        device const float *a      [[buffer(0)]],
        device const uint  *q      [[buffer(1)]],
        device       float *c      [[buffer(2)]],
        constant Q8Dims    &d      [[buffer(3)]],
        device const float *scale  [[buffer(4)]],
        uint  tgid                 [[threadgroup_position_in_grid]],
        uint  ntg                  [[threadgroups_per_grid]],
        uint  tg_size              [[threads_per_threadgroup]],
        ushort tiisg               [[thread_index_in_simdgroup]],
        ushort sgitg               [[simdgroup_index_in_threadgroup]]) {
    const uint words = d.k / 4u;            // u32 words per row (4 int8 each)
    const uint nsg = tg_size / 32u;         // simdgroups per threadgroup
    // global simdgroup index = this column; grid-stride over all columns.
    for (uint col = tgid * nsg + sgitg; col < d.n; col += ntg * nsg) {
        const uint row_base = col * words;
        float acc = 0.0f;
        for (uint w = tiisg; w < words; w += 32u) {
            const uint qv = q[row_base + w];
            const uint ab = 4u * w;
            acc += a[ab+0] * float(int(qv << 24) >> 24)
                 + a[ab+1] * float(int(qv << 16) >> 24)
                 + a[ab+2] * float(int(qv <<  8) >> 24)
                 + a[ab+3] * float(int(qv) >> 24);
        }
        acc = simd_sum(acc);
        if (tiisg == 0) c[col] = acc * scale[col];
    }
}

// ---------- gemv_q8_b: B-ROW batched int8 GEMV (lm_head) — stream the weight ONCE ----------
// The B-row generalization of gemv_q8 and the MSL twin of matmul_vec_q8_batch.wgsl: computes
// c[r*n + col] = scale[col] · (a[r*k] · qᵀ[col]) for ALL B activation rows, reading each weight
// row's int8 bytes EXACTLY ONCE (loaded into the simdgroup, reused across the B rows). At conc64
// that turns the per-row loop's 64× redundant 311 MB vocab-matrix reload into a single stream.
//
// Layout matches gemv_q8: one SIMDGROUP per output column, many columns/threadgroup, grid-strided.
// Each lane streams a strided slice of the column's weight words; for each word it unpacks the 4
// int8 ONCE and accumulates the contribution into B per-row partials (acc[r]); a per-row simd_sum
// then scales by scale[col] and lane 0 writes the B outputs. Per-row arithmetic (i8 unpack order,
// 4-term dot, scale-after-reduce) is bit-identical to gemv_q8 — same greedy-argmax oracle holds.
//
// Bindings match gemv_q8 (0 a, 1 q, 2 c, 3 dims{m=B,k,n}, 4 scale). d.m = B (the real batch),
// capped at MAXB_Q8 (defense in depth; the conc gate keeps B<=64).
constant uint MAXB_Q8 = 64u;
kernel void gemv_q8_b(
        device const float *a      [[buffer(0)]],   // [B,k]
        device const uint  *q      [[buffer(1)]],   // int8 weights [n,k] packed 4/u32
        device       float *c      [[buffer(2)]],   // [B,n] row-major
        constant Q8Dims    &d      [[buffer(3)]],   // {m=B, k, n, _}
        device const float *scale  [[buffer(4)]],   // per output column (weight row)
        uint  tgid                 [[threadgroup_position_in_grid]],
        uint  ntg                  [[threadgroups_per_grid]],
        uint  tg_size              [[threads_per_threadgroup]],
        ushort tiisg               [[thread_index_in_simdgroup]],
        ushort sgitg               [[simdgroup_index_in_threadgroup]]) {
    const uint words = d.k / 4u;            // u32 words per weight row (4 int8 each)
    const uint nsg   = tg_size / 32u;       // simdgroups per threadgroup
    const uint m     = min(d.m, MAXB_Q8);   // active batch rows (caller gates B<=MAXB_Q8)
    const uint arow  = d.k;                 // f32 per activation row
    // global simdgroup index = this column; grid-stride over all columns.
    for (uint col = tgid * nsg + sgitg; col < d.n; col += ntg * nsg) {
        const uint row_base = col * words;
        float acc[MAXB_Q8];
        for (uint r = 0u; r < m; ++r) acc[r] = 0.0f;
        // Stream the weight row ONCE; reuse each loaded/unpacked chunk for all m activation rows.
        for (uint w = tiisg; w < words; w += 32u) {
            const uint qv = q[row_base + w];          // 4 int8, one load
            const float w0 = float(int(qv << 24) >> 24);
            const float w1 = float(int(qv << 16) >> 24);
            const float w2 = float(int(qv <<  8) >> 24);
            const float w3 = float(int(qv) >> 24);
            const uint ab = 4u * w;
            for (uint r = 0u; r < m; ++r) {
                device const float *ar = a + r * arow + ab;
                acc[r] += ar[0]*w0 + ar[1]*w1 + ar[2]*w2 + ar[3]*w3;
            }
        }
        const float s = scale[col];
        for (uint r = 0u; r < m; ++r) {
            const float v = simd_sum(acc[r]);
            if (tiisg == 0) c[r * d.n + col] = v * s;
        }
    }
}

// ---------- argmax_b: B-ROW greedy argmax over logits[B,vocab] → out[B] ----------
// One THREADGROUP per batch row (grid.x = B): threadgroup `r` argmaxes its own logits[r*vocab..]
// row into out[r]. Identical greedy reduction to argmax (strictly-greater wins, tie → lower idx);
// reading logits, not the 311 MB weight, so this is cheap — it just packages the B argmaxes into
// one dispatch (no per-row hard fence).
// BLOCK-32 int8 GEMV, B rows (2026-09-22). Same contract as gemv_q8_b except `scale` is
// [n][k/32]: one scale per 32-INPUT BLOCK of each weight row, not one per row. WHY: per-row int8
// gives a whole 17,408-wide `ffn_down` row ONE scale, and that tensor family carries single
// enormous "super weights" — the outlier sets the scale and its row-mates keep ~2 bits; served
// that way the 27B degenerated to `<think></think>assistant`. A block scale contains the outlier
// to its own 32 weights, which is also why Q4_K_S (32-wide sub-blocks) survives it.
kernel void gemv_q8b32_b(
        device const float *a      [[buffer(0)]],   // [B,k]
        device const uint  *q      [[buffer(1)]],   // int8 weights [n,k] packed 4/u32
        device       float *c      [[buffer(2)]],   // [B,n] row-major
        constant Q8Dims    &d      [[buffer(3)]],   // {m=B, k, n, _}
        device const float *scale  [[buffer(4)]],   // [n][k/32]
        uint  tgid                 [[threadgroup_position_in_grid]],
        uint  ntg                  [[threadgroups_per_grid]],
        uint  tg_size              [[threads_per_threadgroup]],
        ushort tiisg               [[thread_index_in_simdgroup]],
        ushort sgitg               [[simdgroup_index_in_threadgroup]]) {
    const uint words = d.k / 4u;
    const uint nblk  = d.k / 32u;
    const uint nsg   = tg_size / 32u;
    const uint m     = min(d.m, MAXB_Q8);
    const uint arow  = d.k;
    for (uint col = tgid * nsg + sgitg; col < d.n; col += ntg * nsg) {
        const uint row_base = col * words;
        device const float *sc = scale + col * nblk;
        float acc[MAXB_Q8];
        for (uint r = 0u; r < m; ++r) acc[r] = 0.0f;
        for (uint w = tiisg; w < words; w += 32u) {
            const uint qv = q[row_base + w];
            const float w0 = float(int(qv << 24) >> 24);
            const float w1 = float(int(qv << 16) >> 24);
            const float w2 = float(int(qv <<  8) >> 24);
            const float w3 = float(int(qv) >> 24);
            const float s  = sc[w >> 3];            // 8 words of 4 int8 per 32-block
            const uint ab = 4u * w;
            for (uint r = 0u; r < m; ++r) {
                device const float *ar = a + r * arow + ab;
                acc[r] += (ar[0]*w0 + ar[1]*w1 + ar[2]*w2 + ar[3]*w3) * s;
            }
        }
        for (uint r = 0u; r < m; ++r) {
            const float v = simd_sum(acc[r]);
            if (tiisg == 0) c[r * d.n + col] = v;
        }
    }
}

kernel void argmax_b(
        device const float *logits [[buffer(0)]],   // [B, vocab] row-major
        device       uint  *out    [[buffer(1)]],   // [B]
        constant ArgmaxDims &d     [[buffer(2)]],    // {vocab, _, _, _}
        uint  row                  [[threadgroup_position_in_grid]],
        uint  lid                  [[thread_position_in_threadgroup]],
        uint  tg_size              [[threads_per_threadgroup]]) {
    device const float *row_logits = logits + row * d.vocab;
    float bv = -3.4e38f; uint bi = 0u;
    for (uint i = lid; i < d.vocab; i += tg_size) {
        const float v = row_logits[i];
        if (v > bv) { bv = v; bi = i; }
    }
    threadgroup float sv[256]; threadgroup uint si[256];
    sv[lid] = bv; si[lid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            const float ov = sv[lid + stride]; const uint oi = si[lid + stride];
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; sv[lid] = bv; si[lid] = bi; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) out[row] = si[0];
}

// ---------- scale_inplace: x[i] *= scale (gemma per-layer layer_scalar) ----------
struct ScaleDims { uint n; uint scale_bits; uint _p1; uint _p2; };

kernel void scale_inplace(
        device float *x        [[buffer(0)]],
        constant ScaleDims &d  [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < d.n) x[gid] = x[gid] * as_type<float>(d.scale_bits);
}

// ---------- kv_scatter: write this token's k,v into the paged pool at slots[pos] ----------
// Decode (q_len=1): pool[slots[0]*row_floats + col] = new[col], col in [0,row_floats).
// Bindings match kv_scatter.wgsl: 0 new_k, 1 new_v, 2 slots, 3 key_pool, 4 value_pool, 5 dims
// {q_len, row_floats, src_row, _}.
struct ScatterDims { uint q_len; uint row_floats; uint src_row; uint _p; };

kernel void kv_scatter(
        device const float *new_k      [[buffer(0)]],
        device const float *new_v      [[buffer(1)]],
        device const uint  *slots      [[buffer(2)]],
        device       float *key_pool   [[buffer(3)]],
        device       float *value_pool [[buffer(4)]],
        constant ScatterDims &d        [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= d.q_len * d.row_floats) return;
    const uint pos = gid / d.row_floats;
    const uint col = gid % d.row_floats;
    const uint pool_idx = slots[pos] * d.row_floats + col;
    const uint src = d.src_row * d.row_floats + gid;
    key_pool[pool_idx] = new_k[src];
    value_pool[pool_idx] = new_v[src];
}

// ---------- rmsnorm_add: normed = rmsnorm(src); hidden += normed (gemma post-norm) ----------
struct RmsAddDims { uint tokens; uint hidden; float eps; uint _pad; };

kernel void rmsnorm_add(
        device const float *src    [[buffer(0)]],
        device const float *weight [[buffer(1)]],
        device       float *hidden [[buffer(2)]],
        constant RmsAddDims &d     [[buffer(3)]],
        uint  tgpig                [[threadgroup_position_in_grid]],
        uint  lid                  [[thread_position_in_threadgroup]],
        uint  tg_size              [[threads_per_threadgroup]],
        ushort tiisg               [[thread_index_in_simdgroup]],
        ushort sgitg               [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig;
    const uint base = row * d.hidden;
    float acc = 0.0f;
    for (uint i = lid; i < d.hidden; i += tg_size) {
        const float v = src[base + i];
        acc += v * v;
    }
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
    for (uint i = lid; i < d.hidden; i += tg_size) {
        hidden[base + i] += src[base + i] * inv * weight[i];
    }
}

// ---------- rmsnorm_add_scale: hidden = (hidden + rmsnorm(src)·weight) · s ----------
// L361 — Gemma 4's post-FFN pair fused into ONE dispatch. The four-norm block ends every layer
// with `hidden += rmsnorm(mlp_down)·post_ffn_norm` (rmsnorm_add) then `hidden *= layer_scalar`
// (scale_inplace): two dispatches over the same h floats, back to back — 96/token on a 48-layer
// model whose decode is latency-bound on dispatch COUNT (measured out), not bytes. This is
// NOT the already-measured FUSE_ADDNORM2 (the post-ATTN pair, +2.1% on 2026-07-03); it is the
// other pair. buffer(4) is the layer's EXISTING per-layer ScaleDims — the same buffer
// scale_inplace binds — so nothing new is allocated and the shared RmsAddDims stays shared. Body
// is rmsnorm_add verbatim; only the final write differs. Fires only for a layer with BOTH
// post_ffn_norm and layer_scalar, i.e. Gemma 4 alone; every other model is untouched.
//
// Placed at END OF FILE deliberately: MSL needs both structs declared above, and rmsnorm_add
// (whose RmsAddDims this reuses) is the file's last section — two insert attempts anchored on
// "the next section header" found none and wrote nothing, leaving Rust registering a kernel the
// library did not contain: `island compile failed … falling back to wgpu GEMV` for EVERY model.
kernel void rmsnorm_add_scale(
        device const float *src    [[buffer(0)]],
        device const float *weight [[buffer(1)]],
        device       float *hidden [[buffer(2)]],
        constant RmsAddDims &d     [[buffer(3)]],
        constant ScaleDims  &sc    [[buffer(4)]],
        uint  tgpig                [[threadgroup_position_in_grid]],
        uint  lid                  [[thread_position_in_threadgroup]],
        uint  tg_size              [[threads_per_threadgroup]],
        ushort tiisg               [[thread_index_in_simdgroup]],
        ushort sgitg               [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig;
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
    const float s = as_type<float>(sc.scale_bits);
    for (uint i = lid; i < d.hidden; i += tg_size) {
        hidden[base + i] = (hidden[base + i] + src[base + i] * inv * weight[i]) * s;
    }
}
