// Qwen3.8 vision encoder (qwen3vl_merger mmproj) on the GPU — the kernels behind
// crates/arf-gpu/src/gpu/metal/qwen_vision.rs. The CPU encoder (arf-core model/qwen_vision.rs)
// is the reference: same op order, f32 activations everywhere, bf16 weights widened EXACTLY
// (bf16 is the top half of an f32), f32 accumulation. The only differences from the CPU are
// summation order (tiles vs the CPU's loops), f32 LayerNorm sums (the CPU accumulates in f64),
// softmax normalised at the end instead of per weight, and an f32 erf in the merger GELU.
//
// Five kernels:
//   vit_mm_*       C = act(A . W^T + bias) [+ R]   the dense layers (tiled, simdgroup matrices)
//   vit_layernorm  y = (x - mean) / sqrt(var + eps) * w + b, one threadgroup per row
//   vit_rope       2-D vision rope on the q and k thirds of the fused qkv rows, in place
//   vit_attn_hd*   full (non-causal) attention, flash-style (online softmax), per (32 rows, head)
//
// Why float tiles and not half: the activations are not bounded (the residual stream of a ViT
// has large outliers) and a measurement records the one half-fragment GEMM tried here as FASTER
// BUT WRONG (L171). The shape of `vit_mm` is the shipped matmul_mm_q8 / matmul_mm_q4ks tile
// (32 rows x 64 columns, 4 simdgroups, KC=32, weight tile col-major + transpose=true load), NOT
// the BM=64 variant: that one is recorded in matmul_mm_q4ks.metal as 4x SLOWER (16 float
// accumulator fragments spill).

#include <metal_stdlib>
using namespace metal;

// ------------------------------------------------------------------------------------------
// Dense layer: C[m, n] = act(A[m, k] . W[n, k]^T + bias[n]) (+ R[m, n] when has_res).
//   act 0 = none, 1 = GELU tanh (the ViT MLP), 2 = GELU erf (the merger).
//   R may be the SAME buffer as C (the residual add: h += o): each output element is read and
//   written by the one thread that owns it, and A is never C.
// Any m, k, n (edges are zero-padded in the staging copy and the store is guarded).
// Grid (ceil(m/32), ceil(n/64)), 128 threads.
// ------------------------------------------------------------------------------------------
struct VitMmDims { uint m; uint k; uint n; uint act; uint has_res; uint _p0; uint _p1; uint _p2; };

constant constexpr uint MM_BM = 32u;
constant constexpr uint MM_BN = 64u;
constant constexpr uint MM_KC = 32u;
constant constexpr uint MM_NSG = 4u;
constant constexpr uint MM_SG_N = MM_BN / MM_NSG;   // 16 columns per simdgroup
constant constexpr uint MM_RFRAG = MM_BM / 8u;      // 4
constant constexpr uint MM_CFRAG = MM_SG_N / 8u;    // 2
constant constexpr uint MM_KFRAG = MM_KC / 8u;      // 4

inline float vit_w(device const ushort *w, ulong i) { return as_type<float>(uint(w[i]) << 16); }
inline float vit_w(device const float *w, ulong i) { return w[i]; }

inline float vit_gelu_tanh(float x) {
    const float k = 0.7978846f; // sqrt(2/pi), as the CPU's 0.797_884_6
    return 0.5f * x * (1.0f + precise::tanh(k * (x + 0.044715f * x * x * x)));
}

// erf to ~1.5e-7 absolute (Abramowitz & Stegun 7.1.26). The CPU computes erf in f64; GELU adds
// erf to 1, so an absolute error is what matters and this one is below f32 resolution there.
inline float vit_erf(float x) {
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                             - 0.284496736f) * t + 0.254829592f) * t * precise::exp(-ax * ax);
    return copysign(y, x);
}

inline float vit_gelu_erf(float x) { return 0.5f * x * (1.0f + vit_erf(x * 0.70710678118654752f)); }

template <typename WT>
kernel void vit_mm(device const float *a    [[buffer(0)]],   // [m, k] row-major
                   device const WT    *w    [[buffer(1)]],   // [n, k] row-major (bf16 bits or f32)
                   device       float *c    [[buffer(2)]],   // [m, n] row-major
                   constant VitMmDims &d    [[buffer(3)]],
                   device const float *bias [[buffer(4)]],   // [n]
                   device const float *res  [[buffer(5)]],   // [m, n] (read only when has_res)
                   uint2 tgpig [[threadgroup_position_in_grid]],
                   uint  tidx  [[thread_index_in_threadgroup]],
                   uint  sgid  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[MM_BM * MM_KC + MM_BN * MM_KC];   // 12288 B
    threadgroup float *sA = smem;                    // [BM][KC] row-major
    threadgroup float *sW = smem + MM_BM * MM_KC;    // [BN][KC] (col-major within the tile)
    const uint M = d.m, K = d.k, N = d.n;
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
            sA[rl * MM_KC + kk] = (row < M && kab < K) ? a[ulong(row) * K + kab] : 0.0f;
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

    // Epilogue: each simdgroup stores its 32x16 strip to its own quarter of smem (512 floats,
    // 4 x 512 = 2048 <= 3072), then each lane finishes 16 elements.
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
            if (d.act == 1u) v = vit_gelu_tanh(v);
            else if (d.act == 2u) v = vit_gelu_erf(v);
            const ulong o = ulong(row) * N + col;
            if (d.has_res != 0u) v = res[o] + v;
            c[o] = v;
        }
    }
}

template [[host_name("vit_mm_bf16")]] kernel void vit_mm<ushort>(
    device const float *, device const ushort *, device float *, constant VitMmDims &,
    device const float *, device const float *, uint2, uint, uint);
template [[host_name("vit_mm_f32")]] kernel void vit_mm<float>(
    device const float *, device const float *, device float *, constant VitMmDims &,
    device const float *, device const float *, uint2, uint, uint);

// ------------------------------------------------------------------------------------------
// LayerNorm with bias over rows of `cols`: one 256-thread threadgroup per row. Two passes
// (mean, then the variance of x - mean) like the CPU; sums in f32 (the CPU uses f64).
// ------------------------------------------------------------------------------------------
struct VitLnDims { uint rows; uint cols; uint eps_bits; uint _p; };

kernel void vit_layernorm(device const float *x [[buffer(0)]],
                          device const float *w [[buffer(1)]],
                          device const float *b [[buffer(2)]],
                          device       float *y [[buffer(3)]],
                          constant VitLnDims &d [[buffer(4)]],
                          uint row  [[threadgroup_position_in_grid]],
                          uint tid  [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    if (row >= d.rows) return;
    const uint cols = d.cols;
    device const float *xr = x + ulong(row) * cols;
    device float *yr = y + ulong(row) * cols;
    float s = 0.0f;
    for (uint e = tid; e < cols; e += 256u) s += xr[e];
    s = simd_sum(s);
    if (lane == 0u) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8u; ++i) tot += red[i];
    const float mean = tot / float(cols);
    threadgroup_barrier(mem_flags::mem_threadgroup);   // everyone has read red[] before reuse
    float s2 = 0.0f;
    for (uint e = tid; e < cols; e += 256u) {
        const float v = xr[e] - mean;
        s2 += v * v;
    }
    s2 = simd_sum(s2);
    if (lane == 0u) red[sg] = s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (uint i = 0; i < 8u; ++i) var += red[i];
    var /= float(cols);
    const float scale = 1.0f / precise::sqrt(var + as_type<float>(d.eps_bits));
    for (uint e = tid; e < cols; e += 256u) yr[e] = (xr[e] - mean) * scale * w[e] + b[e];
}

// ------------------------------------------------------------------------------------------
// Vision rope, in place on the q and k thirds of qkv rows [n, 3*hidden]: NeoX pairs (j, j+half)
// within each head, cos/sin from the host tables [n, half] (the CPU's own `rope_tables`).
// One thread per (token, q-or-k head, pair); 256-wide threadgroups over the flat count.
// ------------------------------------------------------------------------------------------
struct VitRopeDims { uint n; uint hidden; uint heads; uint half_dim; };

kernel void vit_rope(device float *qkv        [[buffer(0)]],
                     device const float *cs   [[buffer(1)]],
                     device const float *sn   [[buffer(2)]],
                     constant VitRopeDims &d  [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    const uint half_dim = d.half_dim;
    const uint total = d.n * 2u * d.heads * half_dim;
    if (gid >= total) return;
    const uint j = gid % half_dim;
    const uint t = gid / half_dim;
    const uint hp = t % (2u * d.heads);
    const uint i = t / (2u * d.heads);
    const uint part = hp / d.heads, head = hp % d.heads;
    const ulong base = ulong(i) * 3u * d.hidden + part * d.hidden + head * 2u * half_dim;
    const float a = qkv[base + j], b = qkv[base + j + half_dim];
    const float c = cs[i * half_dim + j], s = sn[i * half_dim + j];
    qkv[base + j] = a * c - b * s;
    qkv[base + j + half_dim] = a * s + b * c;
}

// ------------------------------------------------------------------------------------------
// Attention: out[n, hidden] = softmax(Q K^T * scale) V per head, all keys (bidirectional).
// Q, K, V are the three thirds of qkv [n, 3*hidden] (already roped).
//
// One threadgroup = 32 query rows x one head, 4 simdgroups x 8 rows. Keys go in blocks of 32:
// the block's K and V are staged in threadgroup memory (zero past n), each simdgroup computes
// its 8x32 scores S = Q K^T with Q read straight from device (8x8 fragments), the online
// softmax runs on 4 lanes per row (8 scores each), the running output O (8 x HD, HD/8
// fragments) is rescaled by diag(alpha) with one 8x8 multiply per fragment, and O += P V.
// At the end O is divided by the row sum.
//
// Query rows at or past n are computed (from the zeroed pad rows the host guarantees — a NaN
// there would reach real rows through the 0 * x terms of the diag multiply) and not stored.
// The qkv buffer MUST hold ceil(n/32)*32 rows.
//
// Threadgroup memory at HD=72: K 9216 + V 9216 + S 4096 + diag 1024 = 23552 B.
// Grid (ceil(n/32), heads), 128 threads.
// ------------------------------------------------------------------------------------------
struct VitAttnDims { uint n; uint hidden; uint heads; uint scale_bits; };

template <uint HD>
kernel void vit_attn(device const float *qkv   [[buffer(0)]],
                     device       float *out   [[buffer(1)]],
                     constant VitAttnDims &d   [[buffer(2)]],
                     uint2 tg   [[threadgroup_position_in_grid]],
                     uint  tid  [[thread_index_in_threadgroup]],
                     uint  sg   [[simdgroup_index_in_threadgroup]],
                     uint  lane [[thread_index_in_simdgroup]]) {
    constexpr uint KB = 32u, DF = HD / 8u, NSG = 4u;
    threadgroup float Ks[KB * HD];
    threadgroup float Vs[KB * HD];
    threadgroup float Ssm[NSG * 8u * KB];
    threadgroup float Dsm[NSG * 64u];
    const uint n = d.n, D = d.hidden, head = tg.y;
    const ulong ld = 3ul * D;
    const float scale = as_type<float>(d.scale_bits);
    const uint q0 = tg.x * 32u + sg * 8u;
    threadgroup float *S = Ssm + sg * 8u * KB;
    threadgroup float *Dg = Dsm + sg * 64u;
    device const float *qbase = qkv + ulong(q0) * ld + head * HD;

    simdgroup_float8x8 O[DF];
    for (uint c = 0; c < DF; ++c) O[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint r = lane / 4u, qd = lane % 4u;   // this lane's score row and column quarter
    float m_run = -1e30f, l_run = 0.0f;          // finite sentinels: the build is fast-math

    for (uint k0 = 0; k0 < n; k0 += KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);   // last block's readers are done
        for (uint e = tid; e < KB * HD; e += 128u) {
            const uint kr = e / HD, cc = e % HD, key = k0 + kr;
            float kv = 0.0f, vv = 0.0f;
            if (key < n) {
                const ulong o = ulong(key) * ld + head * HD + cc;
                kv = qkv[o + D];
                vv = qkv[o + 2u * D];
            }
            Ks[e] = kv;
            Vs[e] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q K^T: 8 x 32, as 4 fragments.
        simdgroup_float8x8 Sf[KB / 8u];
        for (uint j = 0; j < KB / 8u; ++j) Sf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint df = 0; df < DF; ++df) {
            simdgroup_float8x8 qf;
            simdgroup_load(qf, qbase + df * 8u, ld);
            for (uint j = 0; j < KB / 8u; ++j) {
                simdgroup_float8x8 kf;   // kf[dd][key] = K[k0 + 8j + key][8df + dd]
                simdgroup_load(kf, Ks + (j * 8u) * HD + df * 8u, HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(Sf[j], qf, kf, Sf[j]);
            }
        }
        for (uint j = 0; j < KB / 8u; ++j) simdgroup_store(Sf[j], S + j * 8u, KB);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: lane (r, qd) owns S[r][8qd .. 8qd+8]; row stats reduced over the 4
        // lanes of the row (lane ^ 1, lane ^ 2 stay in it).
        float s[8];
        float mx = -1e30f;
        for (uint t = 0; t < 8u; ++t) {
            const uint col = qd * 8u + t;
            float v = S[r * KB + col] * scale;
            if (k0 + col >= n) v = -1e30f;
            s[t] = v;
            mx = max(mx, v);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1u));
        mx = max(mx, simd_shuffle_xor(mx, 2u));
        const float m_new = max(m_run, mx);
        // The sentinels are zeroed explicitly rather than trusted to exp(-1e30) under fast math.
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
        // diag(alpha) as an 8x8 matrix: 2 entries per lane, alpha of row rr from lane 4*rr.
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
        // O += P V
        for (uint j = 0; j < KB / 8u; ++j) {
            simdgroup_float8x8 pf;
            simdgroup_load(pf, S + j * 8u, KB);
            for (uint c = 0; c < DF; ++c) {
                simdgroup_float8x8 vf;   // vf[key][dd] = V[k0 + 8j + key][8c + dd]
                simdgroup_load(vf, Vs + (j * 8u) * HD + c * 8u, HD);
                simdgroup_multiply_accumulate(O[c], pf, vf, O[c]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);   // S / Dg are rewritten next block
    }

    // O / l -> out. Each simdgroup parks its 8 x HD rows in its quarter of Ks (4 x 8 x HD = KB x HD).
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *Os = Ks + sg * 8u * HD;
    for (uint c = 0; c < DF; ++c) simdgroup_store(O[c], Os + c * 8u, HD);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 8u * HD; e += 32u) {   // 8*HD is a multiple of 32: uniform trips
        const uint rr = e / HD, cc = e % HD;
        const float lr = simd_shuffle(l_run, ushort(rr * 4u));
        const uint row = q0 + rr;
        if (row < n) out[ulong(row) * D + head * HD + cc] = Os[e] / lr;
    }
}

#define VIT_ATTN(HD)                                                                              \
    template [[host_name("vit_attn_hd" #HD)]] kernel void vit_attn<HD>(                          \
        device const float *, device float *, constant VitAttnDims &, uint2, uint, uint, uint);
VIT_ATTN(8)
VIT_ATTN(16)
VIT_ATTN(32)
VIT_ATTN(64)
VIT_ATTN(72)
VIT_ATTN(80)

#undef VIT_ATTN
