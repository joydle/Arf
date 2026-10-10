// MSL causal depthwise conv1d for the Qwen3.6-27B gated-delta-net (M2). Bit-exact port of
// llama.cpp ggml-cpu/ops.cpp ggml_compute_forward_ssm_conv_f32, decode step (one new token).
//
// EMBARRASSINGLY PARALLEL: one GPU thread per conv channel (conv_dim threads total). Each
// thread does a `d_conv`-tap dot over its channel's window + a SiLU epilogue, then advances
// that channel's conv ring (shift-left + append the new input). Tiny + fast (~10240 threads).
//
// GEOMETRY (the beast): conv_dim = 10240 channels, d_conv = 4 taps. conv_state holds the
// previous d_conv-1 = 3 values per channel, laid out [conv_dim, d_conv-1] row-major (channel
// i1's 3 prior values at conv_state[i1*(d_conv-1) + 0..d_conv-1], OLDEST first). The kernel
// weight is [d_conv, conv_dim] in ggml dims [in,out] → flat index k[i0 + i1*d_conv] (the tap
// i0 is contiguous per channel i1), matching ggml's `c[i0 + i1*ncs]` with ncs=d_conv.
//
// THE WINDOW (causal, length d_conv): the d_conv-1 cached prior values (oldest..newest) then
// the new input value last:  window = [state[0], state[1], state[2], conv_in[i1]].
// out[i1] = silu( sum_{i0<d_conv} window[i0] * k[i0 + i1*d_conv] ).
//
// CRITICAL ORDERING: read the OLD state into local registers for the dot BEFORE overwriting
// it, then write the shifted+appended ring back (drop the oldest, append the new input) so the
// next token's window is correct. d_conv is small (<=8) so the window lives in a register array.
//
// SiLU matches silu.wgsl exactly: silu(x) = x / (1 + exp(-clamp(x,-30,30)))  (clamp so Metal's
// exp() can't overflow → inf). This is the conv epilogue (ggml applies silu right after).
//
// Bindings: 0 conv_in[conv_dim] (new token), 1 conv_kernel[d_conv*conv_dim] (weight),
//           2 conv_state[conv_dim*(d_conv-1)] (READ+WRITE ring), 3 conv_out[conv_dim] (post-silu),
//           4 dims {conv_dim, d_conv, row0, _}.

#include <metal_stdlib>
using namespace metal;

// L309 — `row0`: the batch row this dispatch starts at (see gated_delta_net's GdnDims). The ring
// is stateful, so k rows of ONE sequence must be pushed serially: height-1 dispatches with
// row0 = 0..k and a barrier between. row0 = 0 with a full-height grid is the unchanged default.
// L313 — `serial_rows` folds the per-row dispatch loop into the kernel (see gated_delta_net's
// GdnDims): each thread owns one channel's ring cache, so rows chain through its own registers
// with no barrier at all. `ckpt` (buffer 6) writes the post-row ring to checkpoint slot r.
struct ConvDims { uint conv_dim; uint d_conv; uint row0; uint serial_rows; };

#define SSM_CONV_MAX_TAPS 8u   // d_conv is tiny (4 for the beast); register window cap.

kernel void ssm_conv1d(
        device const float *conv_in     [[buffer(0)]],
        device const float *conv_kernel [[buffer(1)]],
        device       float *conv_state  [[buffer(2)]],
        device       float *conv_out    [[buffer(3)]],
        constant ConvDims  &d           [[buffer(4)]],
        // L175 — s_copy: batch row -> PINNED STATE ROW (see gated_delta_net). conv_in/conv_out are
        // indexed by the batch row; only conv_state follows the sequence's pinned bank.
        device const uint  *s_copy      [[buffer(5)]],
        // L313 — checkpoint slots [SPEC_CKPT_ROWS][conv_dim*(d_conv-1)]; written only when
        // serial_rows > 0. Dummy-bound (the state itself) on the ordinary path, never touched.
        device       float *ckpt        [[buffer(6)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    // L174 — SEQUENCE AXIS, ggml's attribute set (see gdn_prologue_msl.metal for the three
    // constraints: no mixing grid-position attributes, never hardcode the threadgroup width,
    // read it from [[threads_per_threadgroup]]).
    //
    // Each sequence advances ITS OWN conv ring — that is the entire point. conv_in/conv_out are
    // [n_seqs, conv_dim]; conv_state is [n_seqs, conv_dim*(d_conv-1)]. A 1-high grid gives seq==0
    // and is byte-identical to the single-stream kernel.
    const uint i1  = tgpig.x * uint(ntg.x) + uint(tpitg.x);   // one thread per channel
    if (i1 >= d.conv_dim) return;

    const uint dc = d.d_conv;            // taps (4)
    const uint cs = dc - 1u;             // cached cols per channel (3)
    const uint kbase = i1 * dc;          // kernel is STATIC — shared by every sequence
    // BIT 31 of `serial_rows` = NO CHECKPOINTS (2026-09-19, windowed prefill). The struct has no
    // spare word. A 16-row PREFILL window commits every row, so it needs no rollback slots — and
    // it MUST NOT write them: there are only SPEC_CKPT_ROWS (3), and on a no-checkpoint dispatch
    // the host dummy-binds the STATE ITSELF at buffer 6, so an unconditional write here would
    // land in other sequences' rings.
    const uint srows = d.serial_rows & 0x7fffffffu;
    const bool write_ckpt = (d.serial_rows & 0x80000000u) == 0u;
    const uint iters = srows > 0u ? srows : 1u;
    for (uint r = 0u; r < iters; ++r) {
    const uint seq = srows > 0u ? r : (tgpig.y + d.row0);
    const uint io    = seq * d.conv_dim;                       // this seq's conv_in/out slab
    const uint bank  = s_copy[seq];                            // pinned state row
    const uint sbase = bank * d.conv_dim * cs + i1 * cs;       // this (bank,channel)'s ring base

    // 1) Assemble the causal window from the OLD ring (oldest..newest) + the new input last.
    //    Read old state into registers BEFORE any write (the critical ordering).
    float window[SSM_CONV_MAX_TAPS];
    for (uint i0 = 0u; i0 < cs; ++i0) {
        window[i0] = conv_state[sbase + i0];
    }
    const float x_new = conv_in[io + i1];
    window[cs] = x_new;                  // window[d_conv-1] = the new token's value

    // 2) The d_conv-tap dot with the channel's kernel.
    float sumf = 0.0f;
    for (uint i0 = 0u; i0 < dc; ++i0) {
        sumf += window[i0] * conv_kernel[kbase + i0];
    }

    // 3) SiLU epilogue (exact match to silu.wgsl: x/(1+exp(-clamp(x,-30,30)))).
    const float z = clamp(sumf, -30.0f, 30.0f);
    conv_out[io + i1] = sumf / (1.0f + exp(-z));

    // 4) Advance the ring: shift left by 1 (drop oldest), append the new input value.
    //    window[1..d_conv] are the new (oldest..newest) cached cols. Write AFTER the dot.
    for (uint i0 = 0u; i0 < cs; ++i0) {
        conv_state[sbase + i0] = window[i0 + 1u];
        // L313 — checkpoint slot r: the post-row ring, written by the channel's owner.
        if (write_ckpt && srows > 0u && r + 1u < srows) {
            ckpt[r * d.conv_dim * cs + i1 * cs + i0] = window[i0 + 1u];
        }
    }
    } // serial-rows loop (single iteration on the ordinary path)
}

// -------------------------------------------------------------------------------------------
// FLOAT4-OVER-CHANNELS variant (ARF_SSM_CONV_BATCHED). ONE thread processes FOUR adjacent
// conv channels via float4 vectorized loads/stores → grid = conv_dim/4 (guard the tail scalar).
//
// BIT-EXACTNESS: the per-channel conv is INDEPENDENT and each float4 lane carries the SAME
// IEEE scalar arithmetic in the SAME order as `ssm_conv1d` above. float4 componentwise
// (fma-free) `+`/`*` are per-lane identical to 4 separate scalar ops — no reassociation, no
// cross-lane mixing. So lane c of the vectorized thread == the scalar kernel's channel (base+c)
// bit-for-bit. Only difference is memory access shape (coalesced float4 vs 4 scalar), which
// does not change the produced float values.
//
// conv_in[i1] is channel-contiguous → a single float4 load covers channels [base..base+3].
// conv_kernel[i0 + i1*d_conv] is tap-contiguous per channel (strided by d_conv across
// channels), so a tap across 4 channels is a GATHER — we build a float4 of that tap from the 4
// per-channel scalars (same values the scalar path reads). conv_state[i1*cs + c] is likewise
// strided; we gather/scatter its cols as float4 across the 4 channels.
//
// Same bindings + ConvDims as `ssm_conv1d`. Grid must cover ceil(conv_dim/4) threads.
kernel void ssm_conv1d_f4(
        device const float *conv_in     [[buffer(0)]],
        device const float *conv_kernel [[buffer(1)]],
        device       float *conv_state  [[buffer(2)]],
        device       float *conv_out    [[buffer(3)]],
        constant ConvDims  &d           [[buffer(4)]],
        uint gid                         [[thread_position_in_grid]]) {
    const uint base = gid * 4u;          // first of the 4 channels this thread owns
    if (base >= d.conv_dim) return;

    const uint dc = d.d_conv;            // taps (4)
    const uint cs = dc - 1u;             // cached cols per channel (3)

    // Fast float4 path only when a full quad of channels fits (conv_dim % 4 == 0 for the beast:
    // 10240). Tail (base+3 >= conv_dim) falls back to the identical scalar math per channel so
    // odd conv_dims stay bit-exact.
    if (base + 3u < d.conv_dim) {
        // window per lane: 4 float4s (one per tap position), lane c = channel (base+c).
        float4 window[SSM_CONV_MAX_TAPS];
        // 1) OLD ring cols (oldest..newest), gathered across the 4 channels. Read BEFORE write.
        for (uint i0 = 0u; i0 < cs; ++i0) {
            window[i0] = float4(
                conv_state[(base + 0u) * cs + i0],
                conv_state[(base + 1u) * cs + i0],
                conv_state[(base + 2u) * cs + i0],
                conv_state[(base + 3u) * cs + i0]);
        }
        // new input: channel-contiguous → single coalesced float4 load.
        const float4 x_new = *((device const float4 *)(conv_in + base));
        window[cs] = x_new;

        // 2) d_conv-tap dot, per-lane (each lane's `+`/`*` == the scalar kernel's channel).
        float4 sumf = float4(0.0f);
        for (uint i0 = 0u; i0 < dc; ++i0) {
            const float4 kv = float4(
                conv_kernel[(base + 0u) * dc + i0],
                conv_kernel[(base + 1u) * dc + i0],
                conv_kernel[(base + 2u) * dc + i0],
                conv_kernel[(base + 3u) * dc + i0]);
            sumf += window[i0] * kv;
        }

        // 3) SiLU epilogue — per-lane, identical clamp/exp to the scalar kernel.
        const float4 z = clamp(sumf, -30.0f, 30.0f);
        const float4 res = sumf / (1.0f + exp(-z));
        // conv_out channel-contiguous → single coalesced float4 store.
        *((device float4 *)(conv_out + base)) = res;

        // 4) Advance each channel's ring (drop oldest, append new). Scatter across channels.
        for (uint i0 = 0u; i0 < cs; ++i0) {
            const float4 nxt = window[i0 + 1u];
            conv_state[(base + 0u) * cs + i0] = nxt.x;
            conv_state[(base + 1u) * cs + i0] = nxt.y;
            conv_state[(base + 2u) * cs + i0] = nxt.z;
            conv_state[(base + 3u) * cs + i0] = nxt.w;
        }
        return;
    }

    // ---- tail: fewer than 4 channels remain — identical scalar math per channel ----
    for (uint c = base; c < d.conv_dim; ++c) {
        const uint sbase = c * cs;
        const uint kbase = c * dc;
        float window[SSM_CONV_MAX_TAPS];
        for (uint i0 = 0u; i0 < cs; ++i0) window[i0] = conv_state[sbase + i0];
        window[cs] = conv_in[c];
        float sf = 0.0f;
        for (uint i0 = 0u; i0 < dc; ++i0) sf += window[i0] * conv_kernel[kbase + i0];
        const float zz = clamp(sf, -30.0f, 30.0f);
        conv_out[c] = sf / (1.0f + exp(-zz));
        for (uint i0 = 0u; i0 < cs; ++i0) conv_state[sbase + i0] = window[i0 + 1u];
    }
}
