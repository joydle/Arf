// Qwen3-Omni audio encoder ("AuT") on the GPU — the kernels the vision file does not already have,
// behind crates/arf-gpu/src/gpu/metal/qwen_audio.rs. NOT A STANDALONE FILE: the host compiles it
// APPENDED to qwen_vision_msl.metal (one library), so it uses that file's tile constants (MM_*),
// `vit_w` and `vit_gelu_erf`, and the host reuses `vit_mm_*` (the dense layers, bias + erf GELU +
// residual fused) and `vit_layernorm` (eps from the host: 1e-5 here) unchanged.
//
// The CPU encoder (arf-core model/qwen_audio_encoder.rs, rel <= 1.01e-5 vs transformers) is the
// reference: f32 activations, bf16 weights widened exactly, f32 accumulation. bf16 activations
// are NOT an option: the bf16 envelope measured on speech had a worst-token cosine of 0.875
//.
//
// Three kernels:
//   aud_conv_mm_*     Conv2d 3x3 stride 2 pad 1 + bias + erf GELU as an IMPLICIT-im2col GEMM:
//                     vit_mm's tile with the A operand gathered from the NHWC input on the fly.
//                     An explicit im2col of conv2 is [chunks*32*25, 4320] f32 — 13.8 MB per
//                     second of audio — which is why it is not materialised.
//   aud_flatten       conv3's NHWC output -> the [token, c*F + f] rows conv_out reads, for the
//                     VALID token positions only (M:830-832, 840)
//   aud_attn_win_hd*  vit_attn with per-block key ranges: block-diagonal (windowed) non-causal
//                     attention (M:704-743). A copy of vit_attn rather than an edit of it: the
//                     vision kernel is measured (rel 1.95e-5, measured 2026-09-27) and
//                     stays byte-identical; the only differences are the key range [k_begin,
//                     k_end) and the store guard (row < k_end) taken from `blocks`.

// ------------------------------------------------------------------------------------------
// Conv2d(3x3, stride 2, pad 1) over `imgs` NHWC images x[img, h, w, cin] -> c[img, ho, wo, cout]
// (row = (img*ho + y)*wo + x, column = output channel), + bias, then act (2 = erf GELU).
// GEMM view: M = imgs*ho*wo, K = cin*9 in PyTorch's weight order (ci, ky, kx), N = cout.
// Grid (ceil(M/32), ceil(N/64)), 128 threads — vit_mm's shape.
// ------------------------------------------------------------------------------------------
struct AudConvDims {
    uint m; uint k; uint n; uint act;
    uint h; uint w; uint cin; uint ho;
    uint wo; uint _p0; uint _p1; uint _p2;
};

template <typename WT>
kernel void aud_conv_mm(device const float *x    [[buffer(0)]],   // [imgs, h, w, cin]
                        device const WT    *w    [[buffer(1)]],   // [cout, cin*9]
                        device       float *c    [[buffer(2)]],   // [imgs*ho*wo, cout]
                        constant AudConvDims &d  [[buffer(3)]],
                        device const float *bias [[buffer(4)]],   // [cout]
                        uint2 tgpig [[threadgroup_position_in_grid]],
                        uint  tidx  [[thread_index_in_threadgroup]],
                        uint  sgid  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[MM_BM * MM_KC + MM_BN * MM_KC];
    threadgroup float *sA = smem;
    threadgroup float *sW = smem + MM_BM * MM_KC;
    const uint M = d.m, K = d.k, N = d.n;
    const uint H = d.h, W = d.w, CIN = d.cin, HO = d.ho, WO = d.wo;
    const uint row0 = tgpig.x * MM_BM;
    const uint col0 = tgpig.y * MM_BN;

    simdgroup_float8x8 acc[MM_RFRAG][MM_CFRAG];
    for (uint i = 0; i < MM_RFRAG; ++i)
        for (uint j = 0; j < MM_CFRAG; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint kchunks = (K + MM_KC - 1u) / MM_KC;
    for (uint kc = 0u; kc < kchunks; ++kc) {
        const uint kbase = kc * MM_KC;
        for (uint e = tidx; e < MM_BN * MM_KC; e += 128u) {
            const uint cl = e / MM_KC, kk = e % MM_KC;
            const uint col = col0 + cl, kab = kbase + kk;
            sW[cl * MM_KC + kk] = (col < N && kab < K) ? vit_w(w, ulong(col) * K + kab) : 0.0f;
        }
        for (uint e = tidx; e < MM_BM * MM_KC; e += 128u) {
            const uint rl = e / MM_KC, kk = e % MM_KC;
            const uint row = row0 + rl, kab = kbase + kk;
            float v = 0.0f;
            if (row < M && kab < K) {
                const uint img = row / (HO * WO), p = row % (HO * WO);
                const uint y = p / WO, xo = p % WO;
                const uint ci = kab / 9u, r9 = kab % 9u;
                const int iy = int(2u * y + r9 / 3u) - 1;
                const int ix = int(2u * xo + r9 % 3u) - 1;
                if (iy >= 0 && iy < int(H) && ix >= 0 && ix < int(W)) {
                    v = x[((ulong(img) * H + uint(iy)) * W + uint(ix)) * CIN + ci];
                }
            }
            sA[rl * MM_KC + kk] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kf = 0u; kf < MM_KFRAG; ++kf) {
            simdgroup_float8x8 af[MM_RFRAG];
            for (uint ri = 0; ri < MM_RFRAG; ++ri)
                simdgroup_load(af[ri], sA + (ri * 8u) * MM_KC + kf * 8u, MM_KC);
            for (uint cj = 0u; cj < MM_CFRAG; ++cj) {
                simdgroup_float8x8 wf;
                simdgroup_load(wf, sW + (sgid * MM_SG_N + cj * 8u) * MM_KC + kf * 8u, MM_KC,
                               ulong2(0, 0), true);
                for (uint ri = 0; ri < MM_RFRAG; ++ri)
                    simdgroup_multiply_accumulate(acc[ri][cj], af[ri], wf, acc[ri][cj]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float *reg = smem + sgid * (MM_BM * MM_SG_N);
    for (uint ri = 0; ri < MM_RFRAG; ++ri)
        for (uint cj = 0u; cj < MM_CFRAG; ++cj)
            simdgroup_store(acc[ri][cj], reg + (ri * 8u) * MM_SG_N + cj * 8u, MM_SG_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lane = tidx & 31u;
    const uint col_sg0 = col0 + sgid * MM_SG_N;
    for (uint e = lane; e < MM_BM * MM_SG_N; e += 32u) {
        const uint rl = e / MM_SG_N, cl = e % MM_SG_N;
        const uint row = row0 + rl, col = col_sg0 + cl;
        if (row < M && col < N) {
            float v = reg[rl * MM_SG_N + cl] + bias[col];
            if (d.act == 2u) v = vit_gelu_erf(v);
            c[ulong(row) * N + col] = v;
        }
    }
}

template [[host_name("aud_conv_mm_bf16")]] kernel void aud_conv_mm<ushort>(
    device const float *, device const ushort *, device float *, constant AudConvDims &,
    device const float *, uint2, uint, uint);
template [[host_name("aud_conv_mm_f32")]] kernel void aud_conv_mm<float>(
    device const float *, device const float *, device float *, constant AudConvDims &,
    device const float *, uint2, uint, uint);

// ------------------------------------------------------------------------------------------
// Flatten for conv_out: out[i, cc*F + f] = a3[chunk, f, t, cc] where token i came from
// (chunk, t): src[i] = chunk*wo + t (global over the call), minus row_base = the first chunk of
// this stem group times wo. One thread per output element, 256-wide threadgroups.
// ------------------------------------------------------------------------------------------
struct AudFlatDims { uint ntok; uint ch; uint fq; uint wo; uint row_base; uint _p0; uint _p1; uint _p2; };

kernel void aud_flatten(device const float *a3  [[buffer(0)]],   // [chunks, fq, wo, ch]
                        device const uint  *src [[buffer(1)]],   // [ntok]
                        device       float *out [[buffer(2)]],   // [ntok, ch*fq]
                        constant AudFlatDims &d [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    const uint flat = d.ch * d.fq;
    if (gid >= d.ntok * flat) return;
    const uint i = gid / flat, j = gid % flat;
    const uint cc = j / d.fq, f = j % d.fq;
    const uint local = src[i] - d.row_base;
    const uint lc = local / d.wo, t = local % d.wo;
    out[gid] = a3[((ulong(lc) * d.fq + f) * d.wo + t) * d.ch + cc];
}

// ------------------------------------------------------------------------------------------
// Windowed attention: vit_attn (see qwen_vision_msl.metal for the algorithm and its invariants)
// where threadgroup tg.x takes blocks[tg.x] = (k_begin, k_end, q_start, 0): 32 query rows from
// q_start attend to keys [k_begin, k_end) — one attention window — and rows >= k_end are not
// stored. The host emits ceil(len/32) blocks per window.
//
// Query rows read past the window come from the next window (finite) or from pad rows: the qkv
// buffer MUST hold n + 32 rows, with the rows past n zeroed (a NaN there would reach stored rows
// through the diag multiply's 0 * x terms, exactly as in vit_attn).
// Grid (blocks, heads), 128 threads.
// ------------------------------------------------------------------------------------------
struct AudAttnDims { uint n; uint hidden; uint heads; uint scale_bits; };

template <uint HD>
kernel void aud_attn_win(device const float *qkv    [[buffer(0)]],
                         device       float *out    [[buffer(1)]],
                         constant AudAttnDims &d    [[buffer(2)]],
                         device const uint4 *blocks [[buffer(3)]],
                         uint2 tg   [[threadgroup_position_in_grid]],
                         uint  tid  [[thread_index_in_threadgroup]],
                         uint  sg   [[simdgroup_index_in_threadgroup]],
                         uint  lane [[thread_index_in_simdgroup]]) {
    constexpr uint KB = 32u, DF = HD / 8u, NSG = 4u;
    threadgroup float Ks[KB * HD];
    threadgroup float Vs[KB * HD];
    threadgroup float Ssm[NSG * 8u * KB];
    threadgroup float Dsm[NSG * 64u];
    const uint4 blk = blocks[tg.x];
    const uint k_begin = blk.x, k_end = blk.y;
    const uint D = d.hidden, head = tg.y;
    const ulong ld = 3ul * D;
    const float scale = as_type<float>(d.scale_bits);
    const uint q0 = blk.z + sg * 8u;
    threadgroup float *S = Ssm + sg * 8u * KB;
    threadgroup float *Dg = Dsm + sg * 64u;
    device const float *qbase = qkv + ulong(q0) * ld + head * HD;

    simdgroup_float8x8 O[DF];
    for (uint c = 0; c < DF; ++c) O[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint r = lane / 4u, qd = lane % 4u;
    float m_run = -1e30f, l_run = 0.0f;          // finite sentinels: the build is fast-math

    for (uint k0 = k_begin; k0 < k_end; k0 += KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < KB * HD; e += 128u) {
            const uint kr = e / HD, cc = e % HD, key = k0 + kr;
            float kv = 0.0f, vv = 0.0f;
            if (key < k_end) {
                const ulong o = ulong(key) * ld + head * HD + cc;
                kv = qkv[o + D];
                vv = qkv[o + 2u * D];
            }
            Ks[e] = kv;
            Vs[e] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 Sf[KB / 8u];
        for (uint j = 0; j < KB / 8u; ++j) Sf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint df = 0; df < DF; ++df) {
            simdgroup_float8x8 qf;
            simdgroup_load(qf, qbase + df * 8u, ld);
            for (uint j = 0; j < KB / 8u; ++j) {
                simdgroup_float8x8 kf;
                simdgroup_load(kf, Ks + (j * 8u) * HD + df * 8u, HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(Sf[j], qf, kf, Sf[j]);
            }
        }
        for (uint j = 0; j < KB / 8u; ++j) simdgroup_store(Sf[j], S + j * 8u, KB);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float s[8];
        float mx = -1e30f;
        for (uint t = 0; t < 8u; ++t) {
            const uint col = qd * 8u + t;
            float v = S[r * KB + col] * scale;
            if (k0 + col >= k_end) v = -1e30f;
            s[t] = v;
            mx = max(mx, v);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1u));
        mx = max(mx, simd_shuffle_xor(mx, 2u));
        const float m_new = max(m_run, mx);
        const float alpha = (m_run <= -1e29f) ? 0.0f : exp(m_run - m_new);
        float ps = 0.0f;
        for (uint t = 0; t < 8u; ++t) {
            const float p = (s[t] <= -1e29f) ? 0.0f : exp(s[t] - m_new);
            S[r * KB + qd * 8u + t] = p;
            ps += p;
        }
        ps += simd_shuffle_xor(ps, 1u);
        ps += simd_shuffle_xor(ps, 2u);
        l_run = l_run * alpha + ps;
        m_run = m_new;
        for (uint t = 0; t < 2u; ++t) {
            const uint e = lane * 2u + t, rr = e / 8u, cc = e % 8u;
            const float a_r = simd_shuffle(alpha, ushort(rr * 4u));
            Dg[e] = (rr == cc) ? a_r : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 Df;
        simdgroup_load(Df, Dg, 8u);
        for (uint c = 0; c < DF; ++c) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, Df, O[c]);
            O[c] = t;
        }
        for (uint j = 0; j < KB / 8u; ++j) {
            simdgroup_float8x8 pf;
            simdgroup_load(pf, S + j * 8u, KB);
            for (uint c = 0; c < DF; ++c) {
                simdgroup_float8x8 vf;
                simdgroup_load(vf, Vs + (j * 8u) * HD + c * 8u, HD);
                simdgroup_multiply_accumulate(O[c], pf, vf, O[c]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *Os = Ks + sg * 8u * HD;
    for (uint c = 0; c < DF; ++c) simdgroup_store(O[c], Os + c * 8u, HD);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 8u * HD; e += 32u) {
        const uint rr = e / HD, cc = e % HD;
        const float lr = simd_shuffle(l_run, ushort(rr * 4u));
        const uint row = q0 + rr;
        if (row < k_end) out[ulong(row) * D + head * HD + cc] = Os[e] / lr;
    }
}

#define AUD_ATTN(HD)                                                                              \
    template [[host_name("aud_attn_win_hd" #HD)]] kernel void aud_attn_win<HD>(                  \
        device const float *, device float *, constant AudAttnDims &, device const uint4 *,     \
        uint2, uint, uint, uint);
AUD_ATTN(64)

#undef AUD_ATTN
