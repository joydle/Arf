// GROUP-64 PREFILL MATMUL, SPLASH'S sg4 FORM (2026-09-27) — a port of Splash's prefill linear
// (incoai/splash, Apache-2.0: runtime/metal/kernels/prefill/linear_q4.metal — `q4_mpp_prefill_tile`
// at Simdgroups = 4 and the kernels `prefill_linear_q4_n128_sg4`, `prefill_linear_q4_n128_residual_sg4`,
// `prefill_linear_q4_n128_up_silu_sums_sg4`, `q4_prefill_write_output_sums`, plus the operand pass
// `prefill_linear_q4_sums32`; see NOTICE). The file is IDENTICAL at tags 1.0.2 and 1.1.0
// (`git diff 1.0.2 1.1.0 -- runtime/metal/kernels/prefill/linear_q4.metal` is empty).
//
// WHY: prefill is Arf's one scoreboard row with no fix in flight — Splash 1.0.2 prefills 7-13% faster
// up to 14K tokens (measured 2026-09-26), and these three
// kernels are 43.4 + 30.5 + 22.5% of Splash's prefill GPU time (its decode-profile, 512 rows). On a
// <= 32-core Apple9 GPU (this M4 Max has 32) Splash's plan (`Linear::baseline`, runtime/ops/Linear.cpp)
// sends EVERY prefill projection here: "the four-simdgroup N128 tile ahead of N256 on every prefill
// shape and probed row count: +6..10% GPU".
//
// NOT THE SAME AS `g64pf_mm_n128` (matmul_g64_pf_msl.metal, measured out 2026-09-25 at 0.94-0.97x
// the lane kernel at 256 rows). That port was the EIGHT-simdgroup N128 kernel, which stages 32 rows x
// 256 groups of input sums in 32 KiB of threadgroup memory. The sg4 form is 128 threads, reads the
// sums straight from device memory and uses NO threadgroup memory, so "four of them fit a core at the
// 512-thread occupancy knee" (q4_mpp_tiles.h). The per-element arithmetic is unchanged — Splash states
// every Simdgroups instance of one tile is bit-identical to the others. Unmeasured here until the
// probe (examples/g64_pfs_probe.rs) runs.
//
// WHAT IT DOES (per threadgroup: 32 rows x 128 output columns, 4 simdgroups):
//   for each 64-input quant group g:  partial = x[32 x 64] . q[64 x 128]   (one matmul2d; the 4-bit
//       codes go into MPP as a `uint4b_format` tensor straight from device memory, x as bf16)
//     acc += partial * scale[col, g] + sum_x[row, g] * bias[col, g]         (register epilogue)
// with sum_x the per-(row, group) sum of the SAME bf16 x, precomputed by the operand pass.
// Grid (rows / 32, N / 128), 128 threads; rows padded to whole 32-row tiles (the padding rows are
// computed and never read). Weights in Splash's StorageN=256 tiling
// (`arf_core::model::weights::g64_from_q4_1`): a 128-column tile is half of a 256-column storage tile.
//
// DIFFERENCES FROM SPLASH (each the same choice matmul_g64_msl.metal's decode port made):
//   * Arf's activations are f32 [rows][K]; Splash's norm emits bf16 + the sums
//     (`addRmsWithQ4Sums`). `g64pfs_prepare` is that operand pass: it rounds x to bf16 and takes each
//     (row, group) sum of the ROUNDED values — `prefill_linear_q4_sums32` plus the conversion.
//   * scale / bias are f16 (the GGUF Q4_1 carrier's d / m) instead of bf16.
//   * the plain and residual outputs are f32 and UNROUNDED (Splash rounds the accumulator to bf16
//     and writes bf16); the residual is f32. `g64pfs_mm_up_silu_sums` keeps a bf16 output, because
//     that output IS the down projection's operand, with its sums — exactly Splash's contract. Its
//     gate is Arf's f32 gate projection (Splash's is bf16); the accumulator is not rounded before the
//     SiLU product.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Splash's Q4PrefillParams {outputSize, inputSize}, padded to 16 bytes.
struct G64PfsParams { uint output_size; uint input_size; uint _p0; uint _p1; };

// THE OPERAND PASS: f32 [rows][K] -> bf16 [rows][K] + sums[row_tile][group][32] of the bf16 values
// (the layout `prefill_linear_q4_sums32` writes). One simdgroup per (row, group); grid
// (rows * groups / 4) threadgroups of 128. Same arithmetic as `g64pf_prepare`.
kernel void g64pfs_prepare(device const float *input  [[buffer(0)]],
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

constant constexpr ushort G64PfsTileM = 32, G64PfsTileN = 128, G64PfsSimdgroups = 4;

// `q4_mpp_prefill_tile<32, 128, 4, AddResidual, MultiplySiluGate>`: the Simdgroups == 4 instance,
// whose sums come from device memory (`StagedSums` is false there, so no threadgroup memory and no
// barrier in the K loop). `Epilogue`: 0 plain, 1 + residual, 2 silu(gate) * value.
template <int Epilogue, class Out>
inline void g64pfs_tile(device bfloat *input, device uchar *weights, device const half *scales,
                        device const half *biases, device Out *output,
                        device const float *auxiliary, uint output_size, uint input_size,
                        device const float *precomputed_sums, uint output_origin) {
  constexpr ushort TileM = G64PfsTileM, TileN = G64PfsTileN;
  auto a = tensor(input, dextents<int, 2>{int(input_size), TileM},
                  array<int, 2>{1, int(input_size)});
  auto c = tensor(output, dextents<int, 2>{int(output_size), TileM},
                  array<int, 2>{1, int(output_size)});
  constexpr auto descriptor = matmul2d_descriptor(TileM, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<G64PfsSimdgroups>> operation;
  auto a0 = a.slice<64, TileM>(0, 0);
  const uint quant_groups = input_size / 64;
  constexpr ushort WeightTileN = 256; // weights are stored in 256-column tiles
  const uint tile = output_origin / WeightTileN;
  const uint tile_column = output_origin % WeightTileN;
  device uchar *tile_weights =
      weights + (ulong(tile) * quant_groups * WeightTileN + tile_column) * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b(
      tile_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
  auto b0 = first_b.slice<64, TileN>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;

  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    auto a_slice = a.slice<64, TileM>(quant_group * 64, 0);
    device uchar *group_weights = tile_weights + ulong(quant_group) * WeightTileN * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b(
        group_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b_slice = b.slice<64, TileN>(0, 0);
    auto partial = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b_slice), float>();
    operation.run(a_slice, b_slice, partial);
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
      auto index = accumulated.get_multidimensional_index(i);
      const uint row = index[1];
      const ulong parameter =
          (ulong(tile) * quant_groups + quant_group) * WeightTileN + tile_column + index[0];
      const float sum = precomputed_sums[quant_group * TileM + row];
      accumulated[i] += partial[i] * float(scales[parameter]) + sum * float(biases[parameter]);
    }
  }

  if constexpr (Epilogue == 2) {
    // bf16 out (the next matmul's operand), as Splash stores it
    auto converted = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), bfloat>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
      auto index = accumulated.get_multidimensional_index(i);
      const float gate = auxiliary[index[1] * output_size + output_origin + index[0]];
      converted[i] = bfloat(gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * accumulated[i]);
    }
    converted.store(c.template slice<TileN, TileM>(output_origin, 0));
  } else {
    if constexpr (Epilogue == 1) {
#pragma unroll
      for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        auto index = accumulated.get_multidimensional_index(i);
        accumulated[i] += auxiliary[index[1] * output_size + output_origin + index[0]];
      }
    }
    accumulated.store(c.template slice<TileN, TileM>(output_origin, 0));
  }
}

// `prefill_linear_q4_n128_sg4`: out f32 [rows][N] = x . W^T.
kernel void g64pfs_mm(device bfloat *input         [[buffer(0)]],
                      device uchar *weights        [[buffer(1)]],
                      device const half *scales    [[buffer(2)]],
                      device const half *biases    [[buffer(3)]],
                      device float *output         [[buffer(4)]],
                      device const float *sums     [[buffer(5)]],
                      constant G64PfsParams &p     [[buffer(6)]],
                      uint2 group [[threadgroup_position_in_grid]]) {
  const uint row_tile = group.x, output_tile = group.y;
  g64pfs_tile<0>(input + ulong(row_tile) * G64PfsTileM * p.input_size, weights, scales, biases,
                 output + ulong(row_tile) * G64PfsTileM * p.output_size, output, p.output_size,
                 p.input_size, sums + ulong(row_tile) * G64PfsTileM * (p.input_size / 64),
                 output_tile * G64PfsTileN);
}

// `prefill_linear_q4_n128_residual_sg4`: out f32 = x . W^T + residual (f32 [rows][N]).
kernel void g64pfs_mm_residual(device bfloat *input         [[buffer(0)]],
                               device uchar *weights        [[buffer(1)]],
                               device const half *scales    [[buffer(2)]],
                               device const half *biases    [[buffer(3)]],
                               device const float *residual [[buffer(4)]],
                               device float *output         [[buffer(5)]],
                               device const float *sums     [[buffer(6)]],
                               constant G64PfsParams &p     [[buffer(7)]],
                               uint2 group [[threadgroup_position_in_grid]]) {
  const uint row_tile = group.x, output_tile = group.y;
  const ulong output_offset = ulong(row_tile) * G64PfsTileM * p.output_size;
  g64pfs_tile<1>(input + ulong(row_tile) * G64PfsTileM * p.input_size, weights, scales, biases,
                 output + output_offset, residual + output_offset, p.output_size, p.input_size,
                 sums + ulong(row_tile) * G64PfsTileM * (p.input_size / 64),
                 output_tile * G64PfsTileN);
}

// `q4_prefill_write_output_sums<32, 128, 4>`: the per-(row, 64-group) sums of THIS tile's bf16
// output, in the operand pass's [row_tile][group][32] layout — the next matmul's sums, so it needs
// no operand pass of its own. Reads back what the other simdgroups just stored.
inline void g64pfs_write_output_sums(device const bfloat *output, device float *output_sums,
                                     uint output_size, uint output_origin, uint simd_lane,
                                     uint simd_group) {
  constexpr uint QuantGroups = G64PfsTileN / 64;
  threadgroup_barrier(mem_flags::mem_device);
  for (uint task = simd_group; task < G64PfsTileM * QuantGroups; task += G64PfsSimdgroups) {
    const uint row = task / QuantGroups;
    const uint local_group = task % QuantGroups;
    const uint origin = row * output_size + output_origin + local_group * 64 + simd_lane;
    const float sum = simd_sum(float(output[origin]) + float(output[origin + 32]));
    if (simd_lane == 0) {
      const uint quant_group = output_origin / 64 + local_group;
      output_sums[quant_group * G64PfsTileM + row] = sum;
    }
  }
}

// `prefill_linear_q4_n128_up_silu_sums_sg4`: out bf16 [rows][N] = silu(gate) * (x . W_up^T), gate
// f32 [rows][N] (the gate projection's output), and out_sums [row_tile][N/64][32] of the bf16 out —
// the down projection's operand AND its sums in one pass.
kernel void g64pfs_mm_up_silu_sums(device bfloat *input         [[buffer(0)]],
                                   device uchar *weights        [[buffer(1)]],
                                   device const half *scales    [[buffer(2)]],
                                   device const half *biases    [[buffer(3)]],
                                   device const float *gate     [[buffer(4)]],
                                   device bfloat *output        [[buffer(5)]],
                                   device const float *sums     [[buffer(6)]],
                                   device float *output_sums    [[buffer(7)]],
                                   constant G64PfsParams &p     [[buffer(8)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint simd_lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint row_tile = group.x, output_tile = group.y;
  const ulong output_offset = ulong(row_tile) * G64PfsTileM * p.output_size;
  g64pfs_tile<2>(input + ulong(row_tile) * G64PfsTileM * p.input_size, weights, scales, biases,
                 output + output_offset, gate + output_offset, p.output_size, p.input_size,
                 sums + ulong(row_tile) * G64PfsTileM * (p.input_size / 64),
                 output_tile * G64PfsTileN);
  g64pfs_write_output_sums(output + output_offset,
                           output_sums + ulong(row_tile) * G64PfsTileM * (p.output_size / 64),
                           p.output_size, output_tile * G64PfsTileN, simd_lane, simd_group);
}
