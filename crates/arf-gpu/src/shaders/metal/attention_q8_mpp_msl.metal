// FLASH PREFILL ATTENTION OVER THE 8-BIT KV POOL ON MPP matmul2d (2026-09-23).
//
// The int8 K and V are MATMUL OPERANDS read straight from device memory (half x int8 -> float is a
// supported MPP combination): no dequantize pass into threadgroup memory. The per-token scales are
// folded around the products instead — each score column is multiplied by its key's scale, and
// each probability by its value's scale before PV, which is exact: the scale is per (token, kv
// head), so it factors out of the D-long dot product and out of the V row.
//
// Same scheme as another engine's `prefill_attention_q8_split` (runtime/metal/kernels/common/
// q8_attention_tile.h), read before this was written: one threadgroup = one KV head x 8 query rows
// x one history split, the group's query heads FUSED into M = 8 x GROUP rows, 8 simdgroups running
// the MPP ops cooperatively, scores and probabilities through threadgroup memory, the running
// output in a cooperative tensor. What differs is our layout: the pool is token-major
// `[slot][kv_heads * D]` in 16-slot blocks, so a page is ONE block (N = 16), K and V are both
// `{D, N}` tensors with row stride kv_heads * D (V without the transpose), and the scales are
// `[slot][kv_heads]`.
//
// Kept in its own source: MPP needs the Metal 4 language, and one kernel that fails to compile
// takes every kernel of its source down with it at load.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Dims {
    uint q_len; uint ctx; uint num_heads; uint kv_heads; uint head_dim; uint past_len;
    uint scale_bits; uint group; uint q_start; uint window; uint _p1; uint _p2;
};
struct PfaParams { uint nrows; uint nsplit; uint _p0; uint _p1; };

constant constexpr uint MQ_ROWS = 8u;   // query rows per threadgroup
constant constexpr uint MQ_G = 6u;      // query heads per KV head (Qwen3.8-27B: 24 / 4)
constant constexpr uint MQ_M = MQ_ROWS * MQ_G;
constant constexpr uint MQ_N = 16u;     // one KV block
constant constexpr uint MQ_D = 256u;

// Q (f32 [row][num_heads][D]) -> half, KV-head-major [kv][row (padded to 8)][g][D], so one
// threadgroup's fused M x D query tile is contiguous. Rows past nrows are written as zero.
// Grid (rows_pad, num_heads), D threads.
kernel void pfa_q8_mpp_stage_q(
        device const float *q      [[buffer(0)]],
        device       half  *qs     [[buffer(1)]],
        constant     Dims  &d      [[buffer(2)]],
        constant PfaParams &pp     [[buffer(3)]],
        uint2 gid                  [[threadgroup_position_in_grid]],
        uint  t                    [[thread_index_in_threadgroup]]) {
    const uint r = gid.x, qh = gid.y, D = d.head_dim;
    const uint kv = qh / d.group, g = qh % d.group;
    const uint rows_pad = (pp.nrows + MQ_ROWS - 1u) / MQ_ROWS * MQ_ROWS;
    const float v = r < pp.nrows ? q[(r * d.num_heads + qh) * D + t] : 0.0f;
    qs[((ulong(kv) * rows_pad + r) * d.group + g) * D + t] = half(v);
}

// Grid (kv_heads, ceil(nrows / 8), nsplit), 256 threads. Partials: [slot][M][D] floats and
// [slot][M]{max, sum}, slot = (tile * kv_heads + kv) * nsplit + split. Splits that own no block
// write nothing; the combine recomputes the partition.
//
// NB 16-key blocks per iteration — SHIPPED AT NB=1. MEASURED-OUT 2026-09-24: sharing one barrier
// set and one softmax pass across NB blocks (NB QK runs into column slices of one M x 16NB score
// tile, one softmax, NB PV runs) does not pay. Threadgroup memory 5.2 / 9.8 / 19.0 KB at NB 1/2/4;
// teacher-forced pass 97.7 / 92.6 / 154.6 s (NB=4 starves occupancy); 23.5K TTFT interleaved x2
// NB=2 160.4/195.4/170.9/176.8 s vs NB=1 150.0/178.3/171.0/176.9 s — no gain. The kernel still
// runs ~3.8 TFLOPS of attention (13.3 ms of GPU per 1K context per window) against MPP's 11.6;
// the barrier count was not what holds it. DO NOT RE-CHASE multi-block iteration.
template <uint NB>
kernel void attention_prefill_fa_q8_mpp(
        device const half  *qs         [[buffer(0)]],
        device       int8_t *k8        [[buffer(1)]],
        device       int8_t *v8        [[buffer(2)]],
        device       float *part_o     [[buffer(3)]],
        device const uint  *slots_all  [[buffer(4)]],
        constant     Dims  &d          [[buffer(5)]],
        device const uint  *past_len_b [[buffer(6)]],
        device       float *part_ml    [[buffer(7)]],
        constant PfaParams &pp         [[buffer(8)]],
        device const float *ks         [[buffer(9)]],
        device const float *vs         [[buffer(10)]],
        uint3 gid                      [[threadgroup_position_in_grid]],
        uint  tid                      [[thread_index_in_threadgroup]]) {
    constexpr int M = int(MQ_M), N = int(MQ_N), D = int(MQ_D);
    constexpr uint KT = MQ_N * NB;                    // keys per iteration
    constexpr uint PER = KT / 4u;                     // keys per softmax lane
    const uint kv = gid.x, tile = gid.y, sp = gid.z;
    const uint nrows = pp.nrows, nsplit = pp.nsplit, KVH = d.kv_heads;
    const uint r0 = tile * MQ_ROWS;
    if (r0 >= nrows) return;
    const uint active = min(MQ_ROWS, nrows - r0);
    const uint last_row = r0 + active - 1u;
    const uint block_last = past_len_b[last_row];
    device const uint *slots = slots_all + ulong(last_row) * d.q_start;
    const uint ntiles = block_last / MQ_N + 1u;
    // splits own whole NB-block groups, so a wide iteration never straddles two splits
    const uint per = ((ntiles + nsplit - 1u) / nsplit + NB - 1u) / NB * NB;
    const uint t_lo = sp * per;
    if (t_lo >= ntiles) return;
    const uint t_hi = min(ntiles, t_lo + per);
    const float scale = as_type<float>(d.scale_bits);
    const uint row_elems = KVH * uint(D);

    threadgroup float scores[MQ_M * KT];
    threadgroup half probs[MQ_M * KT];
    threadgroup float row_max[MQ_M];
    threadgroup float row_sum[MQ_M];
    threadgroup float prev_scale[MQ_M];
    threadgroup uint blk_slot[NB];
    threadgroup atomic_uint rescale;
    if (tid < MQ_M) { row_max[tid] = -INFINITY; row_sum[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rows_pad = (nrows + MQ_ROWS - 1u) / MQ_ROWS * MQ_ROWS;
    device half *qtile = const_cast<device half *>(qs) + (ulong(kv) * rows_pad + r0) * MQ_G * uint(D);
    auto qt = tensor(qtile, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    auto st = tensor(scores, dextents<int, 2>{int(KT), M}, array<int, 2>{1, int(KT)});
    auto pt = tensor(probs, dextents<int, 2>{int(KT), M}, array<int, 2>{1, int(KT)});
    auto q0 = qt.slice<D, M>(0, 0);
    auto p0 = pt.template slice<N, M>(0, 0);
    auto k_proto = tensor(k8, dextents<int, 2>{D, N}, array<int, 2>{1, int(row_elems)});
    auto v_proto = tensor(v8, dextents<int, 2>{D, N}, array<int, 2>{1, int(row_elems)});
    auto k0 = k_proto.slice<D, N>(0, 0);
    auto v0 = v_proto.slice<D, N>(0, 0);
    constexpr auto qk_desc = matmul2d_descriptor(M, N, D, false, true, false,
                                                 matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(M, D, N, false, false, false,
                                                 matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<8>> qk;
    matmul2d<pv_desc, execution_simdgroups<8>> pv;
    // THE WIDE PATH (NB > 1): when an iteration's NB blocks are full and adjacent in the pool, Q.K^T
    // runs once over all KT keys and P.V once with K = KT — not NB runs with K = 16, the shape the
    // 2026-09-24 NB experiment kept (and measured level)
    constexpr auto qkw_desc = matmul2d_descriptor(M, int(KT), D, false, true, false,
                                                  matmul2d_descriptor::mode::multiply);
    constexpr auto pvw_desc = matmul2d_descriptor(M, D, int(KT), false, false, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qkw_desc, execution_simdgroups<8>> qkw;
    matmul2d<pvw_desc, execution_simdgroups<8>> pvw;
    auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();
    const bool running_full = uint(running.get_capacity()) * 256u == uint(M * D);
    #pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i)
        if (running_full || running.is_valid_element(i)) running[i] = 0.0f;

    // softmax lane mapping: 4 lanes x PER keys per fused row (192 of the 256 threads)
    const uint fr = tid / 4u, col = (tid % 4u) * PER;
    const bool smx = tid < MQ_M * 4u;
    const uint qrow = fr / MQ_G;
    const uint my_last = smx ? past_len_b[min(r0 + qrow, last_row)] : 0u;

    for (uint t = t_lo; t < t_hi; t += NB) {
        const uint nb = min(NB, t_hi - t);
        const uint key0 = t * MQ_N;
        if (tid < NB) blk_slot[tid] = tid < nb ? slots[key0 + tid * MQ_N] : 0u;
        const uint wslot = slots[key0];
        bool wide = NB > 1u && nb == NB;
        for (uint j = 1; wide && j < NB; ++j) wide = slots[key0 + j * MQ_N] == wslot + j * MQ_N;
        if (wide) {
            auto kt = tensor(k8 + ulong(wslot) * row_elems + kv * uint(D),
                             dextents<int, 2>{D, int(KT)}, array<int, 2>{1, int(row_elems)});
            auto kslice = kt.template slice<D, int(KT)>(0, 0);
            auto sc = qkw.template get_destination_cooperative_tensor<decltype(q0), decltype(kslice), float>();
            qkw.run(q0, kslice, sc);
            sc.store(st.template slice<int(KT), M>(0, 0));
        }
        for (uint j = 0; !wide && j < nb; ++j) {
            const uint slot0 = slots[key0 + j * MQ_N];     // block-aligned: its keys are slot0..
            auto kt = tensor(k8 + ulong(slot0) * row_elems + kv * uint(D),
                             dextents<int, 2>{D, N}, array<int, 2>{1, int(row_elems)});
            auto kslice = kt.template slice<D, N>(0, 0);
            auto sc = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
            qk.run(q0, kslice, sc);
            sc.store(st.template slice<N, M>(int(j * MQ_N), 0));
        }
        if (tid == 0u) atomic_store_explicit(&rescale, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (smx) {
            float s[PER];
            float lmax = -INFINITY;
            for (uint j = 0; j < PER; ++j) {
                const uint c = col + j, blk = c / MQ_N;
                const uint key = key0 + c;
                const bool ok = blk < nb && key <= my_last;
                const float kscale = ok ? ks[ulong(blk_slot[blk] + c % MQ_N) * KVH + kv] : 0.0f;
                s[j] = ok ? scores[fr * KT + c] * kscale * scale : -INFINITY;
                lmax = max(lmax, s[j]);
            }
            lmax = max(lmax, simd_shuffle_xor(lmax, 1));
            lmax = max(lmax, simd_shuffle_xor(lmax, 2));
            const float pmax = row_max[fr];
            const float nmax = max(pmax, lmax);
            float lsum = 0.0f;
            for (uint j = 0; j < PER; ++j) {
                const uint c = col + j, blk = c / MQ_N;
                const bool ok = blk < nb && key0 + c <= my_last;
                const float pj = ok ? exp(s[j] - nmax) : 0.0f;
                lsum += pj;
                // value scale folded into P; a masked key stays EXACTLY zero whatever its slot holds
                const float vsc = ok ? vs[ulong(blk_slot[blk] + c % MQ_N) * KVH + kv] : 0.0f;
                probs[fr * KT + c] = half(ok ? pj * vsc : 0.0f);
            }
            lsum += simd_shuffle_xor(lsum, 1);
            lsum += simd_shuffle_xor(lsum, 2);
            if (col == 0u) {
                const float cc = (nmax == -INFINITY || nmax == pmax) ? 1.0f : exp(pmax - nmax);
                prev_scale[fr] = cc;
                row_sum[fr] = row_sum[fr] * cc + lsum;
                row_max[fr] = nmax;
                if (cc != 1.0f) atomic_store_explicit(&rescale, 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&rescale, memory_order_relaxed)) {
            #pragma unroll
            for (ushort i = 0; i < running.get_capacity(); ++i) {
                if (!running_full && !running.is_valid_element(i)) continue;
                const auto c = running.get_multidimensional_index(i);
                running[i] *= prev_scale[c[1]];
            }
        }
        if (wide) {
            auto vt = tensor(v8 + ulong(wslot) * row_elems + kv * uint(D),
                             dextents<int, 2>{D, int(KT)}, array<int, 2>{1, int(row_elems)});
            auto vslice = vt.template slice<D, int(KT)>(0, 0);
            auto pw = pt.template slice<int(KT), M>(0, 0);
            pvw.run(pw, vslice, running);
        }
        for (uint j = 0; !wide && j < nb; ++j) {
            const uint slot0 = slots[key0 + j * MQ_N];
            auto vt = tensor(v8 + ulong(slot0) * row_elems + kv * uint(D),
                             dextents<int, 2>{D, N}, array<int, 2>{1, int(row_elems)});
            auto vslice = vt.template slice<D, N>(0, 0);
            auto pj = pt.template slice<N, M>(int(j * MQ_N), 0);
            pv.run(pj, vslice, running);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const ulong slot = (ulong(tile) * KVH + kv) * nsplit + sp;
    auto target = tensor(part_o + slot * uint(M * D), dextents<int, 2>{D, M}, array<int, 2>{1, D});
    running.store(target.slice<D, M>(0, 0));
    if (tid < MQ_M) {
        part_ml[(slot * MQ_M + tid) * 2u] = row_max[tid];
        part_ml[(slot * MQ_M + tid) * 2u + 1u] = row_sum[tid];
    }
}
#define PFA_Q8_MPP(NB)                                                                              \
  template [[host_name("attention_prefill_fa_q8_mpp_b" #NB)]] kernel void                            \
  attention_prefill_fa_q8_mpp<NB>(device const half *, device int8_t *, device int8_t *,            \
      device float *, device const uint *, constant Dims &, device const uint *, device float *,    \
      constant PfaParams &, device const float *, device const float *, uint3, uint);
PFA_Q8_MPP(1)
PFA_Q8_MPP(2)
PFA_Q8_MPP(4)
#undef PFA_Q8_MPP

// Combine the splits of one (row, q-head): grid (nrows, num_heads), D threads. Writes the f32
// attention output [row][num_heads][D] the rest of the record reads.
kernel void attention_prefill_fa_q8_mpp_combine(
        device const float *part_o     [[buffer(0)]],
        device const float *part_ml    [[buffer(1)]],
        device       float *out        [[buffer(2)]],
        constant     Dims  &d          [[buffer(3)]],
        constant PfaParams &pp         [[buffer(4)]],
        device const uint  *past_len_b [[buffer(5)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint  t                        [[thread_index_in_threadgroup]]) {
    const uint r = gid.x, qh = gid.y, D = d.head_dim, nrows = pp.nrows, nsplit = pp.nsplit;
    const uint kv = qh / d.group, g = qh % d.group;
    const uint tile = r / MQ_ROWS;
    const uint r0 = tile * MQ_ROWS;
    const uint last_row = min(r0 + MQ_ROWS, nrows) - 1u;
    const uint ntiles = past_len_b[last_row] / MQ_N + 1u;
    const uint nbk = max(pp._p0, 1u);
    const uint per = ((ntiles + nsplit - 1u) / nsplit + nbk - 1u) / nbk * nbk;
    const uint written = (ntiles + per - 1u) / per;
    const uint m = (r - r0) * d.group + g;
    const ulong base = (ulong(tile) * d.kv_heads + kv) * nsplit;
    float mx = -INFINITY;
    for (uint s = 0; s < written; ++s) mx = max(mx, part_ml[((base + s) * MQ_M + m) * 2u]);
    float num = 0.0f, den = 0.0f;
    for (uint s = 0; s < written; ++s) {
        const ulong st = ((base + s) * MQ_M + m) * 2u;
        const float w = part_ml[st] == -INFINITY ? 0.0f : exp(part_ml[st] - mx);
        num += w * part_o[((base + s) * MQ_M + m) * D + t];
        den += w * part_ml[st + 1u];
    }
    out[(ulong(r) * d.num_heads + qh) * D + t] = den > 0.0f ? num / den : 0.0f;
}
