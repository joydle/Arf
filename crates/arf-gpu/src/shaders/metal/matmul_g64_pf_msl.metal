// GROUP-64 PREFILL MATMUL (2026-09-25) — a port of Splash 1.0.2's prefill linear
// (incoai/splash, Apache-2.0: runtime/metal/kernels/prefill/linear_q4.metal, `q4_mpp_prefill_tile`;
// see NOTICE). The 4-bit codes go straight into Metal 4's `matmul2d` as a `uint4b_format` tensor
// over the StorageN=256 tiled layout (`arf_core::model::weights::g64_from_q4_1`); each 64-wide
// group's integer product is scaled and offset in the epilogue: acc += (x . q) * scale + sum(x) * bias.
//
// WHY: the decode kernel (matmul_g64_msl.metal) serves a prefill window as independent 8-row lanes,
// so every weight slice is fetched once per 8 rows; here once per 32. Separate file because
// MetalPerformancePrimitives needs macOS 26 — a failure here leaves prefill on the lane kernel.
//
// Differences from Splash: f32 activations in (g64pf_prepare rounds them to bf16 and takes the
// per-group sums of the ROUNDED values, as the decode path does), f32 out, f16 scale/bias.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct G64PfParams { uint output_size; uint input_size; uint _p0; uint _p1; };

// f32 [rows][K] -> bf16 [rows][K] + sums[row_tile][group][32]. One simdgroup per (row, group);
// grid (rows * groups / 4) threadgroups of 128.
kernel void g64pf_prepare(device const float *input  [[buffer(0)]],
                          device bfloat *xb          [[buffer(1)]],
                          device float *sums         [[buffer(2)]],
                          constant uint &width       [[buffer(3)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint groups = width / 64;
  const uint id = tg * 4 + sg, row = id / groups, g = id % groups;
  const ulong o = ulong(row) * width + g * 64 + lane;
  const bfloat a = bfloat(input[o]), b = bfloat(input[o + 32]);
  xb[o] = a;
  xb[o + 32] = b;
  const float s = simd_sum(float(a) + float(b));
  if (lane == 0) sums[(ulong(row / 32) * groups + g) * 32 + row % 32] = s;
}

constant constexpr ushort G64PfSumBatch = 256;   // 32 rows x 256 groups x 4 B = 32 KiB staged

// out[32 rows][TileN columns] per threadgroup; grid (rows / 32, N / TileN), 8 simdgroups.
template <ushort TileN>
inline void g64pf_tile(device bfloat *input, device uchar *weights, device const half *scales,
                       device const half *biases, device float *output, uint output_size,
                       uint input_size, device const float *sums, uint output_origin,
                       uint simd_lane, uint simd_group, threadgroup float *input_sums) {
  constexpr ushort TileM = 32, Simdgroups = 8;
  auto a = tensor(input, dextents<int, 2>{int(input_size), TileM}, array<int, 2>{1, int(input_size)});
  auto c = tensor(output, dextents<int, 2>{int(output_size), TileM}, array<int, 2>{1, int(output_size)});
  constexpr auto descriptor = matmul2d_descriptor(TileM, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a0 = a.slice<64, TileM>(0, 0);
  const uint groups = input_size / 64;
  constexpr ushort WeightTileN = 256;
  const uint tile = output_origin / WeightTileN, tile_column = output_origin % WeightTileN;
  device uchar *tile_weights = weights + (ulong(tile) * groups * WeightTileN + tile_column) * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b(
      tile_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
  auto b0 = first_b.slice<64, TileN>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  auto load_sums = [&](uint start) {
    const uint count = min(uint(G64PfSumBatch), groups - start);
    for (uint i = simd_group * 32 + simd_lane; i < count * TileM; i += Simdgroups * 32)
      input_sums[i] = sums[(start + i / TileM) * TileM + i % TileM];
  };
  load_sums(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint g = 0; g < groups; ++g) {
    auto a_slice = a.slice<64, TileM>(g * 64, 0);
    device uchar *gw = tile_weights + ulong(g) * WeightTileN * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b(gw, dextents<int, 2>{64, TileN},
                                                                    array<int, 2>{1, 64});
    auto b_slice = b.slice<64, TileN>(0, 0);
    auto partial = operation.template get_destination_cooperative_tensor<decltype(a_slice),
                                                                         decltype(b_slice), float>();
    operation.run(a_slice, b_slice, partial);
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      auto idx = acc.get_multidimensional_index(i);
      const ulong prm = (ulong(tile) * groups + g) * WeightTileN + tile_column + idx[0];
      const float sum = input_sums[(g % G64PfSumBatch) * TileM + idx[1]];
      acc[i] += partial[i] * float(scales[prm]) + sum * float(biases[prm]);
    }
    if (g % G64PfSumBatch == G64PfSumBatch - 1 && g + 1 < groups) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_sums(g + 1);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  acc.store(c.slice<TileN, TileM>(output_origin, 0));
}

kernel void g64pf_mm_n128(device bfloat *input         [[buffer(0)]],
                          device uchar *weights        [[buffer(1)]],
                          device const half *scales    [[buffer(2)]],
                          device const half *biases    [[buffer(3)]],
                          device float *output         [[buffer(4)]],
                          device const float *sums     [[buffer(5)]],
                          constant G64PfParams &p      [[buffer(6)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint simd_lane [[thread_index_in_simdgroup]],
                          uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[32 * G64PfSumBatch];
  const uint row_tile = group.x, groups = p.input_size / 64;
  g64pf_tile<128>(input + ulong(row_tile) * 32 * p.input_size, weights, scales, biases,
                  output + ulong(row_tile) * 32 * p.output_size, p.output_size, p.input_size,
                  sums + ulong(row_tile) * 32 * groups, group.y * 128, simd_lane, simd_group,
                  input_sums);
}

// ---------------------------------------------------------------------------------------------
// GROUP-64 PREFILL v2 (`g64pf2_mm`, 2026-09-25): our `q4pf2_mm` (matmul_mpp_q4ks_msl.metal) with the
// group-64 weight read — DEQUANTIZE ONCE INTO THREADGROUP MEMORY, NO EPILOGUE. `g64pf_mm_n128` above
// (Splash's per-group epilogue form) measured a TIE with the 8-row-lane decode kernel on prefill
// windows (256 rows, quiet GPU: 0.94-0.97x); q4pf2 is what runs our Q4_K_S prefill at ~10.9 TFLOPS.
// Inputs are exactly q4pf2's: the window narrowed to half (`xh`) by the q4tile chain, its per-row
// inverse scale applied once at the store. Each 64-column x 128-input slice is dequantized as
// half(scale * q + bias) — one thread per (column, 16 inputs), one 8-byte load — and one MPP op
// multiplies the window by it. Grid (n / TN, 1), 256 threads.
struct G64Q4TileDims { uint input_size; uint output_size; uint subs; uint supers; };

template <int ROWS, int TN, int KB, int BITS = 4>
kernel void g64pf2_mm(device half *xh                [[buffer(0)]],
                      device const uchar *w          [[buffer(1)]],
                      device const half *scales      [[buffer(2)]],
                      device const half *biases      [[buffer(3)]],
                      device const float *inv_scale  [[buffer(6)]],
                      device float *y                [[buffer(7)]],
                      constant G64Q4TileDims &d      [[buffer(8)]],
                      uint tile [[threadgroup_position_in_grid]],
                      uint tid  [[thread_index_in_threadgroup]]) {
  constexpr int HPC = KB / 16, ITEMS = TN * HPC;
  static_assert(ITEMS % 256 == 0, "whole items per thread");
  static_assert(KB % 64 == 0, "whole groups per chunk");
  if (tile >= d.output_size / TN) return;
  threadgroup half wt[TN * KB];
  auto a = tensor(xh, dextents<int, 2>{int(d.input_size), ROWS}, array<int, 2>{1, int(d.input_size)});
  auto bt = tensor(wt, dextents<int, 2>{KB, TN}, array<int, 2>{1, KB});
  auto c = tensor(y, dextents<int, 2>{int(d.output_size), ROWS}, array<int, 2>{1, int(d.output_size)});
  constexpr auto descriptor = matmul2d_descriptor(ROWS, TN, KB, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  auto a0 = a.template slice<KB, ROWS>(0, 0);
  auto b0 = bt.template slice<KB, TN>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  const bool full = uint(acc.get_capacity()) * 256u == uint(ROWS * TN);
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (full || acc.is_valid_element(i)) acc[i] = 0.0f;
  const uint groups = d.input_size / 64, chunks = d.input_size / KB;
  for (uint ck = 0; ck < chunks; ++ck) {
    for (uint item = tid; item < uint(ITEMS); item += 256u) {
      const uint col = item / uint(HPC), h = item % uint(HPC);
      const uint gcol = tile * TN + col;
      const uint g = ck * (KB / 64) + h / 4u;                       // the 64-group of these 16 inputs
      const ulong p = (ulong(gcol / 256u) * groups + g) * 256u + gcol % 256u;
      const float s = float(scales[p]), b = float(biases[p]);
      threadgroup half *dst = wt + col * KB + h * 16u;
      if (BITS == 3) {
        // the 24-byte record (see g64_mm_impl): lows of inputs 16c.. at [4c], highs at [16 + 2c]
        // the group-pair record (g64_mm_impl BITS 3): 48 bytes per (column, groups 2p, 2p+1);
        // lane-quarter cq at [12cq]: lows of the even / odd group, then their highs
        const uint cq = h % 4u, odd = g & 1u;
        const ulong rec = ((ulong(gcol / 256u) * groups + (g & ~1u)) * 256u + gcol % 256u * 2u) * 24u
                          + cq * 12u;
        const uint lo = *reinterpret_cast<device const uint *>(w + rec + odd * 4u);
        const uint hi = uint(*reinterpret_cast<device const ushort *>(w + rec + 8u + odd * 2u));
        for (uint o = 0; o < 16u; ++o) {
          const uint i = (o & 3u) + 4u * (o >> 3), hf = (o >> 2) & 1u;
          const uint q = ((lo >> (2u * i + 16u * hf)) & 3u) | (((hi >> (i + 8u * hf)) & 1u) << 2);
          dst[o] = half(fma(s, float(q), b));
        }
      } else {
      const uint2 q = *reinterpret_cast<device const uint2 *>(w + p * 32u + (h % 4u) * 8u);
      const uint words[2] = {q.x, q.y};
      for (uint wi = 0; wi < 2; ++wi) {
        const uint v = words[wi];
        for (uint by = 0; by < 4; ++by) {
          const uint byte = (v >> (8u * by)) & 0xFFu;
          dst[wi * 8u + by * 2u]      = half(fma(s, float(byte & 0xFu), b));
          dst[wi * 8u + by * 2u + 1u] = half(fma(s, float(byte >> 4), b));
        }
      }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto a_s = a.template slice<KB, ROWS>(int(ck * KB), 0);
    operation.run(a_s, b0, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!full && !acc.is_valid_element(i)) continue;
    const auto idx = acc.get_multidimensional_index(i);
    acc[i] *= inv_scale[min(uint(idx[1]), uint(ROWS - 1))];
  }
  acc.store(c.template slice<TN, ROWS>(int(tile) * TN, 0));
}
#define G64PF2(NAME, R, TN, KB, BITS)                                                               \
  template [[host_name(NAME)]] kernel void g64pf2_mm<R, TN, KB, BITS>(                              \
      device half *, device const uchar *, device const half *, device const half *,                \
      device const float *, device float *, constant G64Q4TileDims &, uint, uint);
G64PF2("g64pf2_mm_r32_n64", 32, 64, 128, 4)
G64PF2("g64pf2_mm_r128_n64", 128, 64, 128, 4)
G64PF2("g64pf2_mm3_r32_n64", 32, 64, 128, 3)
G64PF2("g64pf2_mm3_r128_n64", 128, 64, 128, 3)
#undef G64PF2
