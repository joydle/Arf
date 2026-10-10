// GROUP-64 AFFINE Q4 MATMUL for 8-row lanes (2026-09-25) — a port of Splash 1.0.2's Apple9
// one-lane decode matmul (incoai/splash, Apache-2.0: runtime/metal/kernels/common/q4_sgmatrix.h,
// decode/linear_q4_sgmatrix.metal; see NOTICE). Weights `w = scale * q + bias`, one f16 scale and
// bias per 64 inputs, in Splash's StorageN=256 tiling (`arf_core::model::weights::g64_from_q4_1`).
//
// WHY (measured 2026-09-25): with Q4_K_S two-level group-32 scales our 8-row
// matmul sustained ~20% slower than Splash's on the same shapes; the format is the difference
// (layout ~5%, group width + scale arithmetic the rest). Projected: a verify pass's matmuls
// 56.3 -> 46.7 ms.
//
// Differences from Splash: activations arrive as f32 [rows][K] (Splash's are bf16), so
// `g64_prepare` rounds them to the bf16 fragment table here; the output is f32 (Splash rounds
// its accumulator to bf16); scales/biases are f16 (the GGUF Q4_1 carrier's d/m) instead of bf16.
// A lane is 8 rows; tg.z indexes lanes, so one dispatch serves 8, 16, or a 256-row prefill window.
#include <metal_stdlib>
using namespace metal;

struct G64Params { uint output_size; uint input_size; uint splits; uint _pad; };

// Groups of weight loads in flight per thread in `g64_mm` / `g64_mm_gate_up`. 2 since 2026-09-25:
// a standalone probe (sustained, 1 lane, bit-identical across depths) 17408x5120
// 312 -> 317 GB/s, 10240x5120 276 -> 293, 5120x6144 268 -> 266 (a plain read of the same bytes: ~360);
// in context the 8-row verify 56.2 / 56.4 -> 55.8 / 56.0 ms (depth 3: 57.4 / 57.3, worse). 1 = load
// the next group while multiplying this one. The engine compiles depths 1 and 3 beside it
// (ARF_G64_PF=1|3).
#ifndef G64_PF
#define G64_PF 2
#endif

namespace g64 {
struct Lane { ushort fm; ushort fn; };
inline Lane lane_map(uint lane) {
  const uint qid = lane >> 2;
  Lane l;
  l.fm = ushort((qid & 4) | ((lane >> 1) & 3));
  l.fn = ushort(((qid & 2) << 1) | ((lane & 1) << 1));
  return l;
}
// physical k (0..63 within a group) -> (fragment j, slot k')
inline uint2 klogical(uint k) {
  const uint c = k >> 4, r = k & 15;
  return uint2((r >> 3) * 4 + (r & 3), 2 * c + ((r >> 2) & 1));
}
template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2> &te(thread simdgroup_matrix<T, 8, 8> &m) {
  return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}
template <typename T>
__attribute__((always_inline)) inline void mma_acc(thread float2 &c, vec<T, 2> a, vec<T, 2> b) {
  simdgroup_matrix<T, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, D;
  te(A) = a; te(B) = b; te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}
constexpr constant uint kXtPerGroup = 512;   // 64 inputs x 8 rows, bf16
inline uint xt_offset(uint j, uint kp, uint m) {
  return (((j >> 2) * 8 + kp) * 4 + (m >> 1)) * 8 + (j & 3) * 2 + (m & 1);
}
} // namespace g64

// f32 activations -> bf16 fragment table + per-(group, row) sum of the bf16 values.
// Grid: (groups * 8 / 4, lanes) threadgroups of 128 threads; one simdgroup per (group, row).
kernel void g64_prepare(device const float *input [[buffer(0)]],
                        device bfloat *table       [[buffer(1)]],
                        device float *sums         [[buffer(2)]],
                        constant uint &width       [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const uint group = (tg.x * 4 + sg) / 8, row = (tg.x * 4 + sg) % 8;
  input += ulong(tg.y) * width * 8;
  table += ulong(tg.y) * width * 8;
  sums += ulong(tg.y) * width / 8;
  const uint offset = row * width + group * 64 + 2 * lane;
  const bfloat a = bfloat(input[offset]), b = bfloat(input[offset + 1]);
  const uint2 logical = g64::klogical(2 * lane);
  table[group * g64::kXtPerGroup + g64::xt_offset(logical.x, logical.y, row)] = a;
  table[group * g64::kXtPerGroup + g64::xt_offset(logical.x + 1, logical.y, row)] = b;
  const float sum = simd_sum(float(a) + float(b));
  if (lane == 0) sums[group * 8 + row] = sum;
}

// y[lane rows][N] (f32) = table x W^T. Grid: (N / tileN, splits, lanes) threadgroups of 128
// threads (4 simdgroups). Split-K partials meet in `partials` and the LAST-ARRIVING threadgroup of
// a column tile adds them in split order (its own from registers) — Splash's fixed-order
// reduction, no spinning. `counters` must start at zero; every use leaves them zero.
//
// GATE_UP (2026-09-25, Splash's `decode_linear_q4_sg_gate_up`): the two chains a simdgroup runs
// are the SAME 8 columns of two weights (gate = w0, up = w1) instead of 16 columns of one, and the
// epilogue writes `silu(gate) * up` — the FFN's gate matmul, up matmul and `swiglu_b` in one
// dispatch. tileN 32 instead of 64; the SiLU is swiglu_b's expression exactly.
// BITS 3 (2026-09-26): GROUP PAIRS. Per (column, groups 2p and 2p+1) a 48-byte record; lane-quarter
// c (inputs 16c..16c+15 of each group) reads 12 contiguous bytes at [12c]: the two groups' u32s of
// 2-bit lows, then their two u16s of high bits — ONE 12-byte load per column per two groups (the
// first layout, a 4-byte and a 2-byte load per group, ran 0.57-0.75x the 4-bit kernel: this kernel
// is bound by loads in flight, not bytes). Step j's pair (inputs o and o+4, o = (j&3) + 8(j>>2)) sits
// at bits 2j / 2j+16 of the lows and j / j+8 of the highs; the highs are spread once a group
// (h32 = lo byte | hi byte << 16) so each step is one shift and mask of each. Codes 0..7; everything
// after the pair is the 4-bit kernel's arithmetic, so the same codes give BIT-IDENTICAL outputs
// (a standalone probe). Split parts must hold whole group pairs (host-checked).
// LPT (2026-09-26): 8-row LANES PER THREADGROUP. 1 = one lane a threadgroup (grid z = lanes). 2 = two
// lanes' simdgroups (4 each, same 64 columns) share a 256-thread threadgroup (grid z = lanes / 2): the
// same weight bytes are read by both halves at the same moment, so the second lane's reads come from
// on-chip cache instead of DRAM — a second lane in its own threadgroup re-read every weight (1.76-1.97x
// one lane, a standalone probe -lanes). Split-K counters are per lane pair; the last
// threadgroup of a tile reduces both lanes (each simdgroup its own).
// ⛔ MEASURED-OUT 2026-09-26: g64_mm_l2 costs the SAME as two separate lanes (1.65-1.92x one lane vs
// 1.65-1.96x) — the second lane was never paying DRAM; a lane costs its own loads-in-flight and MMA
// issue, from cache or not. g64_mm_r32 (4 lanes in one simdgroup's registers) is worse (4.8-7.2x vs
// 3.3-4.0x). Extra lanes are ~linear with this kernel family — as they are in Splash (B2 114 ms, B4 228).
// Probe-only kernels; not dispatched by the engine.
template <bool GATE_UP, uint BITS = 4, uint LPT = 1>
inline void g64_mm_impl(device const bfloat *table, device const uchar *w0,
                        device const half *sc0, device const half *bi0, device float *output,
                        device const float *sums, device float *partials,
                        device atomic_uint *counters, constant G64Params &p,
                        device const uchar *w1, device const half *sc1, device const half *bi1,
                        threadgroup uint &arrival, uint3 tg, uint tid, uint sg_raw, uint lane) {
  constexpr uint tileN = GATE_UP ? 32 : 64;
  const uint N = p.output_size, groups = p.input_size / 64, splits = p.splits;
  const uint lane_i = tg.z * LPT + (LPT > 1 ? sg_raw / 4 : 0);
  const uint sg = LPT > 1 ? sg_raw % 4 : sg_raw;
  table += ulong(lane_i) * p.input_size * 8;
  sums += ulong(lane_i) * p.input_size / 8;
  output += ulong(lane_i) * 8 * N;
  if (splits > 1) {
    partials += ulong(lane_i) * splits * 16 * N;
    counters += ulong(tg.z) * N / tileN;
  }
  const uint first = tg.y * (groups / splits);
  const uint end = tg.y + 1 == splits ? groups : first + groups / splits;
  const g64::Lane l = g64::lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * tileN + sg * (GATE_UP ? 8 : 16);
  const uint tile = base / 256;
  const uint col0 = base % 256 + fm, col1 = GATE_UP ? col0 : col0 + 8;
  // bytes per group of a tile (3-bit: 256 x 24, the pair record being 48 bytes per column)
  constexpr uint GSTRIDE = BITS == 3 ? 256 * 24 : 256 * 32;
  device const uchar *tile0 = w0 + ulong(tile) * groups * GSTRIDE;
  device const uchar *tile1 = w1 + ulong(tile) * groups * GSTRIDE;
  float2 acc[2] = {float2(0), float2(0)};
  float2 dot[2][2] = {{float2(0), float2(0)}, {float2(0), float2(0)}};
  uint2 words[2];
  auto load = [&](uint g, thread uint2 (&w)[2]) __attribute__((always_inline)) {
    w[0] = *reinterpret_cast<device const uint2 *>(tile0 + ulong(g) * GSTRIDE + col0 * 32 + c * 8);
    w[1] = *reinterpret_cast<device const uint2 *>(tile1 + ulong(g) * GSTRIDE + col1 * 32 + c * 8);
  };
  // 3-bit: both groups of a pair for both columns, as (lows, spread highs) per group
  auto load3 = [&](uint g2, thread uint2 (&wa)[2], thread uint2 (&wb)[2]) __attribute__((always_inline)) {
    const packed_uint3 r0 = *reinterpret_cast<device const packed_uint3 *>(
        tile0 + ulong(g2) * GSTRIDE + col0 * 48 + c * 12);
    const packed_uint3 r1 = *reinterpret_cast<device const packed_uint3 *>(
        tile1 + ulong(g2) * GSTRIDE + col1 * 48 + c * 12);
    auto spread = [](uint h) __attribute__((always_inline)) { return (h & 0xFFu) | ((h & 0xFF00u) << 8); };
    wa[0] = uint2(r0.x, spread(r0.z & 0xFFFFu)); wb[0] = uint2(r0.y, spread(r0.z >> 16));
    wa[1] = uint2(r1.x, spread(r1.z & 0xFFFFu)); wb[1] = uint2(r1.y, spread(r1.z >> 16));
  };
  auto run_group = [&](uint g, thread uint2 (&w)[2]) __attribute__((always_inline)) {
    const float2 sum = float2(sums[g * 8 + fn], sums[g * 8 + fn + 1]);
    device const vec<bfloat, 8> *xt =
        reinterpret_cast<device const vec<bfloat, 8> *>(table + ulong(g) * g64::kXtPerGroup);
    vec<bfloat, 8> bq[2];
    bq[0] = xt[fm * 4 + c];
    bq[1] = xt[(8 + fm) * 4 + c];
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
      const bfloat2 b = reinterpret_cast<thread bfloat2 *>(&bq[j >> 2])[j & 3];
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        uint pair;
        if (BITS == 3) {
#ifdef G64_DIAG_CHEAP_EXTRACT
          // PROBE-ONLY diagnostic: the 4-bit extraction on the 3-bit words (wrong values) — separates
          // the extraction's arithmetic from the loads in the 3-bit kernel's time
          pair = ((w[nf].x >> (4 * (j & 3))) & 0x000F000Fu) | 0x43004300u;
#else
          pair = ((w[nf].x >> (2 * j)) & 0x00030003u) | (((w[nf].y >> j) & 0x00010001u) << 2) | 0x43004300u;
#endif
        } else {
          const uint word = j < 4 ? w[nf].x : w[nf].y;
          pair = ((word >> (4 * (j & 3))) & 0x000F000Fu) | 0x43004300u;   // bf16 128+q
        }
        if (j < 2) dot[nf][j & 1] = float2(0);
        g64::mma_acc<bfloat>(dot[nf][j & 1], as_type<bfloat2>(pair), b);
      }
    }
    const ulong prm0 = (ulong(tile) * groups + g) * 256 + col0;
    const ulong prm1 = (ulong(tile) * groups + g) * 256 + col1;
    const float2 d0 = fma(-128.0f, sum, dot[0][0] + dot[0][1]);
    const float2 d1 = fma(-128.0f, sum, dot[1][0] + dot[1][1]);
    acc[0] = fma(d0, float(sc0[prm0]), acc[0]);
    acc[0] = fma(sum, float(bi0[prm0]), acc[0]);
    acc[1] = fma(d1, float(sc1[prm1]), acc[1]);
    acc[1] = fma(sum, float(bi1[prm1]), acc[1]);
  };
  if (BITS == 3) {
    // group pairs, the next pair's load in flight while this one is multiplied
    uint2 a0[2], b0[2], a1[2], b1[2];
    load3(first, a0, b0);
    uint g = first;
    for (; g + 4 <= end; g += 4) {
      load3(g + 2, a1, b1);
      run_group(g, a0); run_group(g + 1, b0);
      if (g + 4 < end) load3(g + 4, a0, b0);
      run_group(g + 2, a1); run_group(g + 3, b1);
    }
    if (g < end) { run_group(g, a0); run_group(g + 1, b0); }
  } else {
#if G64_PF <= 1
  load(first, words);
  for (uint g = first; g < end; ++g) {
    run_group(g, words);
    if (g + 1 < end) load(g + 1, words);
  }
#else
  // G64_PF groups of weight loads in flight per thread (a ring, fully unrolled so it stays in
  // registers): slot i holds group g + i; its reload for g + i + G64_PF issues before group g + i
  // is multiplied. Same arithmetic, same order — only the load schedule differs.
  (void)words;
  uint2 ring[G64_PF][2];
#pragma unroll
  for (uint i = 0; i < G64_PF; ++i)
    if (first + i < end) load(first + i, ring[i]);
  uint g = first;
  for (; g + G64_PF <= end; g += G64_PF) {
#pragma unroll
    for (uint i = 0; i < G64_PF; ++i) {
      uint2 cur[2] = {ring[i][0], ring[i][1]};
      if (g + i + G64_PF < end) load(g + i + G64_PF, ring[i]);
      run_group(g + i, cur);
    }
  }
#pragma unroll
  for (uint i = 0; i < G64_PF; ++i)
    if (g + i < end) run_group(g + i, ring[i]);
#endif
  }
  if (splits > 1) {
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      const uint n = base + fm + (GATE_UP ? 0 : nf * 8);
      device float *slot = partials + ulong(tg.y * 2 + nf) * 8 * N + n;
      slot[fn * N] = acc[nf].x;
      slot[(fn + 1) * N] = acc[nf].y;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      arrival = atomic_fetch_add_explicit(counters + tg.x, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (arrival != splits - 1) return;
    float2 total[2] = {float2(0), float2(0)};
    for (uint s = 0; s < splits; ++s) {
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        const uint n = base + fm + (GATE_UP ? 0 : nf * 8);
        device const float *slot = partials + ulong(s * 2 + nf) * 8 * N + n;
        total[nf] += s == tg.y ? acc[nf] : float2(slot[fn * N], slot[(fn + 1) * N]);
      }
    }
    acc[0] = total[0]; acc[1] = total[1];
    if (tid == 0) atomic_store_explicit(counters + tg.x, 0u, memory_order_relaxed);
  }
  if (GATE_UP) {
    const uint n = base + fm;
    const float2 x = acc[0];
    output[fn * N + n] = (x.x / (1.0f + exp(-x.x))) * acc[1].x;
    output[(fn + 1) * N + n] = (x.y / (1.0f + exp(-x.y))) * acc[1].y;
  } else {
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      const uint n = base + nf * 8 + fm;
      output[fn * N + n] = acc[nf].x;
      output[(fn + 1) * N + n] = acc[nf].y;
    }
  }
}

kernel void g64_mm(device const bfloat *table   [[buffer(0)]],
                   device const uchar *weights  [[buffer(1)]],
                   device const half *scales    [[buffer(2)]],
                   device const half *biases    [[buffer(3)]],
                   device float *output         [[buffer(4)]],
                   device const float *sums     [[buffer(5)]],
                   device float *partials       [[buffer(6)]],
                   device atomic_uint *counters [[buffer(7)]],
                   constant G64Params &p        [[buffer(8)]],
                   uint3 tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint arrival;
  g64_mm_impl<false>(table, weights, scales, biases, output, sums, partials, counters, p,
                     weights, scales, biases, arrival, tg, tid, sg, lane);
}

// output = silu(table x Wg^T) * (table x Wu^T); Grid (N / 32, splits, lanes).
kernel void g64_mm_gate_up(device const bfloat *table   [[buffer(0)]],
                           device const uchar *gate     [[buffer(1)]],
                           device const half *gscales   [[buffer(2)]],
                           device const half *gbiases   [[buffer(3)]],
                           device float *output         [[buffer(4)]],
                           device const float *sums     [[buffer(5)]],
                           device float *partials       [[buffer(6)]],
                           device atomic_uint *counters [[buffer(7)]],
                           constant G64Params &p        [[buffer(8)]],
                           device const uchar *up       [[buffer(9)]],
                           device const half *uscales   [[buffer(10)]],
                           device const half *ubiases   [[buffer(11)]],
                           uint3 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint arrival;
  g64_mm_impl<true>(table, gate, gscales, gbiases, output, sums, partials, counters, p, up,
                    uscales, ubiases, arrival, tg, tid, sg, lane);
}

// Two lanes per threadgroup (LPT 2): same bindings as g64_mm / g64_mm_gate_up; grid (N / tileN,
// splits, lanes / 2), 256 threads.
kernel void g64_mm_l2(device const bfloat *table   [[buffer(0)]],
                      device const uchar *weights  [[buffer(1)]],
                      device const half *scales    [[buffer(2)]],
                      device const half *biases    [[buffer(3)]],
                      device float *output         [[buffer(4)]],
                      device const float *sums     [[buffer(5)]],
                      device float *partials       [[buffer(6)]],
                      device atomic_uint *counters [[buffer(7)]],
                      constant G64Params &p        [[buffer(8)]],
                      uint3 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint arrival;
  g64_mm_impl<false, 4, 2>(table, weights, scales, biases, output, sums, partials, counters, p,
                           weights, scales, biases, arrival, tg, tid, sg, lane);
}
kernel void g64_mm_gate_up_l2(device const bfloat *table   [[buffer(0)]],
                              device const uchar *gate     [[buffer(1)]],
                              device const half *gscales   [[buffer(2)]],
                              device const half *gbiases   [[buffer(3)]],
                              device float *output         [[buffer(4)]],
                              device const float *sums     [[buffer(5)]],
                              device float *partials       [[buffer(6)]],
                              device atomic_uint *counters [[buffer(7)]],
                              constant G64Params &p        [[buffer(8)]],
                              device const uchar *up       [[buffer(9)]],
                              device const half *uscales   [[buffer(10)]],
                              device const half *ubiases   [[buffer(11)]],
                              uint3 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint arrival;
  g64_mm_impl<true, 4, 2>(table, gate, gscales, gbiases, output, sums, partials, counters, p, up,
                          uscales, ubiases, arrival, tg, tid, sg, lane);
}

// The 3-bit builds (24-byte records): same bindings and grids as g64_mm / g64_mm_gate_up.
kernel void g64_mm3(device const bfloat *table   [[buffer(0)]],
                    device const uchar *weights  [[buffer(1)]],
                    device const half *scales    [[buffer(2)]],
                    device const half *biases    [[buffer(3)]],
                    device float *output         [[buffer(4)]],
                    device const float *sums     [[buffer(5)]],
                    device float *partials       [[buffer(6)]],
                    device atomic_uint *counters [[buffer(7)]],
                    constant G64Params &p        [[buffer(8)]],
                    uint3 tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint arrival;
  g64_mm_impl<false, 3>(table, weights, scales, biases, output, sums, partials, counters, p,
                        weights, scales, biases, arrival, tg, tid, sg, lane);
}
kernel void g64_mm3_gate_up(device const bfloat *table   [[buffer(0)]],
                            device const uchar *gate     [[buffer(1)]],
                            device const half *gscales   [[buffer(2)]],
                            device const half *gbiases   [[buffer(3)]],
                            device float *output         [[buffer(4)]],
                            device const float *sums     [[buffer(5)]],
                            device float *partials       [[buffer(6)]],
                            device atomic_uint *counters [[buffer(7)]],
                            constant G64Params &p        [[buffer(8)]],
                            device const uchar *up       [[buffer(9)]],
                            device const half *uscales   [[buffer(10)]],
                            device const half *ubiases   [[buffer(11)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint arrival;
  g64_mm_impl<true, 3>(table, gate, gscales, gbiases, output, sums, partials, counters, p, up,
                       uscales, ubiases, arrival, tg, tid, sg, lane);
}

// PREFILL: FOUR 8-ROW LANES PER SIMDGROUP (`g64_mm_r32`, 2026-09-25). Prefill is ~86-99% matmul
// (ARF_DUP_MM=all doubles TTFT) and the lane kernel above unpacks each weight nibble pair for ONE
// 8-row lane: at a 256-row window every weight is unpacked 32 times, 8 MACs per unpack. Here each
// simdgroup holds 4 lanes (32 rows) of accumulators, so each unpacked pair feeds 4 MMAs. Same table
// and sums (g64_prepare, lanes = rows / 8), same StorageN=256 weights, no split-K (a window fills
// the GPU). Grid (N / 64, 1, lanes / 4) threadgroups of 128 threads.
kernel void g64_mm_r32(device const bfloat *table   [[buffer(0)]],
                       device const uchar *weights  [[buffer(1)]],
                       device const half *scales    [[buffer(2)]],
                       device const half *biases    [[buffer(3)]],
                       device float *output         [[buffer(4)]],
                       device const float *sums     [[buffer(5)]],
                       constant G64Params &p        [[buffer(8)]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint R = 4;
  const uint N = p.output_size, K = p.input_size, groups = K / 64;
  const uint lane0 = tg.z * R;
  const g64::Lane l = g64::lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * 64 + sg * 16;
  const uint tile = base / 256;
  const uint col0 = base % 256 + fm, col1 = col0 + 8;
  device const uchar *tile0 = weights + ulong(tile) * groups * 8192;
  float2 acc[R][2], dot[R][2][2];
#pragma unroll
  for (uint r = 0; r < R; ++r) {
    acc[r][0] = acc[r][1] = float2(0);
    dot[r][0][0] = dot[r][0][1] = dot[r][1][0] = dot[r][1][1] = float2(0);
  }
  for (uint g = 0; g < groups; ++g) {
    uint2 w[2];
    w[0] = *reinterpret_cast<device const uint2 *>(tile0 + ulong(g) * 8192 + col0 * 32 + c * 8);
    w[1] = *reinterpret_cast<device const uint2 *>(tile0 + ulong(g) * 8192 + col1 * 32 + c * 8);
    vec<bfloat, 8> bq[R][2];
    float2 sum[R];
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      const ulong lr = lane0 + r;
      device const vec<bfloat, 8> *xt = reinterpret_cast<device const vec<bfloat, 8> *>(
          table + lr * K * 8 + ulong(g) * g64::kXtPerGroup);
      bq[r][0] = xt[fm * 4 + c];
      bq[r][1] = xt[(8 + fm) * 4 + c];
      device const float *sr = sums + lr * K / 8;
      sum[r] = float2(sr[g * 8 + fn], sr[g * 8 + fn + 1]);
    }
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        const uint word = j < 4 ? w[nf].x : w[nf].y;
        const bfloat2 pair = as_type<bfloat2>(((word >> (4 * (j & 3))) & 0x000F000Fu) | 0x43004300u);
#pragma unroll
        for (uint r = 0; r < R; ++r) {
          if (j < 2) dot[r][nf][j & 1] = float2(0);
          g64::mma_acc<bfloat>(dot[r][nf][j & 1], pair,
                               reinterpret_cast<thread bfloat2 *>(&bq[r][j >> 2])[j & 3]);
        }
      }
    }
    const ulong prm0 = (ulong(tile) * groups + g) * 256 + col0;
    const ulong prm1 = (ulong(tile) * groups + g) * 256 + col1;
    const float s0 = float(scales[prm0]), b0 = float(biases[prm0]);
    const float s1 = float(scales[prm1]), b1 = float(biases[prm1]);
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      const float2 d0 = fma(-128.0f, sum[r], dot[r][0][0] + dot[r][0][1]);
      const float2 d1 = fma(-128.0f, sum[r], dot[r][1][0] + dot[r][1][1]);
      acc[r][0] = fma(sum[r], b0, fma(d0, s0, acc[r][0]));
      acc[r][1] = fma(sum[r], b1, fma(d1, s1, acc[r][1]));
    }
  }
#pragma unroll
  for (uint r = 0; r < R; ++r) {
    device float *o = output + ulong(lane0 + r) * 8 * N;
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      const uint n = base + nf * 8 + fm;
      o[fn * N + n] = acc[r][nf].x;
      o[(fn + 1) * N + n] = acc[r][nf].y;
    }
  }
}
