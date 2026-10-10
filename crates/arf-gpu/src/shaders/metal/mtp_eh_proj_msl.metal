// L241 — MTP ("nextn") input projection for the Qwen3.8 multi-token-prediction head.
//
// The head predicts token t+1 from TWO things: the trunk's hidden state at position t, and the
// embedding of the token actually emitted at t. The GGUF's `nextn.eh_proj.weight` is
// [2*hidden, hidden] (measured [10240, 5120]), which is what fixes the formulation: each input
// is RMS-normed by its own weight, the two are CONCATENATED, and the pair is projected back to
// hidden width.
//
//     x = eh_proj( [ enorm(embed(tok_t))  ‖  hnorm(h_t) ] )
//
// ORDER CONFIRMED: embedding first, hidden second — verified against a working llama.cpp MTP
// implementation for this model family. Getting it backwards would not crash; it would just
// draft badly, which is indistinguishable from "the head is weak" without a reference.
//
// This kernel builds the concatenated, normed [2*hidden] vector. The projection itself is an
// ordinary GEMV against eh_proj (Q8_0), so it reuses the shipped Q8 path rather than a new one.
//
// One threadgroup per batch row; threads stride the hidden dim. Two RMS reductions per row
// (one per half) done cooperatively in threadgroup memory — the same shape as `rmsnorm_b`.

#include <metal_stdlib>
using namespace metal;

struct MtpDims { uint b; uint hidden; float eps; uint swap; uint plus1; uint _p[3]; };

// L259 — `swap` reorders the two halves. eh_proj is [2*hidden -> hidden] and consumes ONE
// [2*hidden] row, so if the weight expects [hnorm(h) || enorm(e)] rather than the documented
// [enorm(e) || hnorm(h)], every output is plausible-but-wrong — which is exactly the symptom
// (the head drafts real-looking tokens that verify always rejects). A one-flag A/B settles it.

kernel void mtp_eh_concat(
        device const float *embed_tok [[buffer(0)]],  // [B, hidden] embedding of the emitted token
        device const float *h_trunk   [[buffer(1)]],  // [B, hidden] trunk residual at position t
        device const float *enorm_w   [[buffer(2)]],  // [hidden]
        device const float *hnorm_w   [[buffer(3)]],  // [hidden]
        device       float *out       [[buffer(4)]],  // [B, 2*hidden] concatenated + normed
        constant     MtpDims &d       [[buffer(5)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const uint row = tgpig.x;
    if (row >= d.b) return;
    const uint H = d.hidden;
    const uint t = uint(tpitg.x);
    const uint nt = uint(ntg.x);

    threadgroup float red[256];

    // ---- half 0: enorm(embed) ----
    const device float *e = embed_tok + row * H;
    float acc = 0.0f;
    for (uint i = t; i < H; i += nt) { const float v = e[i]; acc += v * v; }
    red[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2u; s > 0u; s >>= 1u) {
        if (t < s) red[t] += red[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float e_scale = rsqrt(red[0] / float(H) + d.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // L260 — ⚠️ THE (1+w) OFFSET IS NOW OPT-IN AND DEFAULTS OFF.
//
// L247's web search said enorm/hnorm need `(1 + w) * rmsnorm(x)`. llama.cpp's WORKING MTP
// forward (src/models/bailingmoe3.cpp:444-446) does NOT: it calls the ordinary
// `build_norm(x, w, nullptr, LLM_NORM_RMS, il)` with no add1 anywhere in the file. A shipped
// implementation beats a search summary, so the default flipped. ARF_MTP_PLUS1=1 restores it.
//
// Historic note on the search's claim: (1 + w), NOT w. enorm/hnorm store the weight OFFSET BY -1, unlike every other norm
    // in this model — confirmed against a working llama.cpp MTP implementation, which applies
    // it at runtime via ggml_add1() precisely because the naming does not match the usual
    // convention. Using `w` directly produces plausible-but-wrong drafts (they would simply be
    // rejected by verify, so this would have shown up as terrible acceptance, not as a crash).
    const uint e_off = d.swap != 0u ? H : 0u;
    const uint h_off = d.swap != 0u ? 0u : H;
    for (uint i = t; i < H; i += nt) out[row * 2u * H + e_off + i] = e[i] * e_scale * (d.plus1 != 0u ? 1.0f + enorm_w[i] : enorm_w[i]);

    // ---- half 1: hnorm(h_trunk) ----
    const device float *h = h_trunk + row * H;
    acc = 0.0f;
    for (uint i = t; i < H; i += nt) { const float v = h[i]; acc += v * v; }
    red[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2u; s > 0u; s >>= 1u) {
        if (t < s) red[t] += red[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float h_scale = rsqrt(red[0] / float(H) + d.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < H; i += nt) out[row * 2u * H + h_off + i] = h[i] * h_scale * (d.plus1 != 0u ? 1.0f + hnorm_w[i] : hnorm_w[i]);
}
