// Batched Q4 matmul through Metal 4's MPP matmul2d over a native 4-bit tensor — reading the SHIPPED
// row-major Q4KS buffers through a STRIDED view, so no second copy of the weights exists.
//
// ON THE DECODE PATH, DEFAULT ON (since 2026-09-19; `ARF_NO_MPP_Q4=1` opts out). The batched
// record's `gemv_b` sends a live batch of 5..=16 rows here, per weight; b=1 and b<5 stay on the
// shipped GEMV, as does any shape that is not whole tiles (the 48-wide GDN beta/alpha).
// MEASURED on Qwen3.8-27B, 36 GB M4 Max, interleaved, rounds agreeing to 0.2%:
//   8 streams  276.6 -> 188.9 ms/step, 28.9 -> 42.4 tok/s aggregate (1.46x)
//   16 streams 551.5 -> 259.4 ms/step, 29.0 -> 61.7 tok/s aggregate (2.13x)
// 16/16 answers faithful in 4 of 4 bursts on both arms; 8/8 character-identical at 8 streams.
//
// WHY IT EXISTS (measured 2026-09-19). Our batched GEMV pays ~33 ms per extra
// decode row on an M4 Max — 0.67 of a step — and every reshaping of it has lost (L345/L346/L356/
// L357/L363r/L363s). The technique is Splash's (incoai/splash, Apache-2.0, see NOTICE); reading OUR
// layout through strides {1, k}, the two-level scales and the narrowing passes are ours.
//
// WHAT WAS TRIED AND DELETED, so nobody rebuilds it to find out:
//   - a contiguous TILE COPY of each weight (`Q4TileMatrix`, strides {1, 32}): ~40% faster per
//     matmul (371 vs 511 us on 5120->17408) and 2.17x at 16 streams on gemma-4-12B — but it is a
//     second copy of the weights, and the 27B has no room for one on a 36 GB Mac. Deleted in
//     favour of one path that works everywhere.
//   - a tile-native b=1 GEMV, so tiles could be the ONLY layout: 1.5-2.25x slower than the shipped
//     GEMV even with its exact thread structure (the shipped one runs at the bandwidth roofline,
//     136 us on 5120->17408). One column's sub-blocks sit 4 KB apart in tiles; L363s again.
//
// THREE THINGS THE PRIMITIVE IMPOSES, each measured, none negotiable:
//   1. ROWS is a multiple of 8 — a static_assert in MPPTensorOpsMatMul2dImpl.h ("M must be a
//      multiple of 8 or 16"). There is no 1/2/4-row form. 5..=8 live rows run as 8, 9..=16 as 16;
//      padded rows are computed and ignored. NaN/Inf in them do not leak into live rows (measured).
//   2. The left operand is `half` or `bfloat`. `float` does not compile ("Unsupported type").
//      Our activations are f32, hence the three `q4tile_*` narrowing passes.
//   3. Nibbles are LOW-FIRST — which is exactly the byte order of the island's u32 `codes` stream.
// It also needs MSL 4.0, which is the DEFAULT language version on macOS 26 (raw 262144) — no
// compile option is set. On an older OS this file fails to compile and the feature stays off.
//
// DISPATCH CONTRACT for q4rm_mm_r8 / _r16: one threadgroup per 256 output rows, EXACTLY 256 threads
// per threadgroup — `execution_simdgroups<8>` is 8 simdgroups x 32 threads cooperating on one tile.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Q4TileDims {
  uint input_size;   // k — a multiple of 256
  uint output_size;  // n — a multiple of 256
  uint subs;         // input_size / 32
  uint supers;       // input_size / 256
};

// f32 activations [8][k] -> per-row power-of-two-scaled half [8][k], the 8 inverse scales, and the
// per-sub-block input sums [subs][8] the affine term needs. THREE passes, each embarrassingly
// parallel, because this runs before EVERY matmul of a step (hundreds of times):
//   q4tile_blockmax     one thread per (row, sub-block): max |x| over 32 elements
//   q4tile_scale        one thread per row: reduce <= 544 block maxima to one power-of-two scale
//   q4tile_narrow       one thread per (row, sub-block): write 32 halves and the block's sum
// (A first version did all three in one thread per row — ~35K serial iterations at k=17408. Fine
// for a selftest, tens of milliseconds a step in a real one.)
//
// WHY SCALED, not a plain cast: half overflows above 65,504 and an LLM's residual stream has
// outlier channels; bfloat never overflows but keeps 8 bits of mantissa, and unlike Splash — whose
// values are bfloat16 to begin with — we would be ADDING that rounding. Scaling each row so its
// largest magnitude lands in [8192, 16384) gives every element half's full 11 bits with no
// overflow, and a POWER-OF-TWO scale is exact in floating point, so it comes back out exactly.
// MEASURED (mpp_q4tile_parity): 0.03-0.05% RMS per matmul, finite with 2e5 outliers present.
// The sums are taken from the f32 ORIGINALS: the affine term then carries no narrowing error.
// Rows past the live batch hold stale data; a non-finite maximum falls back to scale 1 so a NaN
// there stays in ITS row and cannot poison the others.
template <int ROWS>
kernel void q4tile_blockmax(device const float *x        [[buffer(0)]],
                               device float *blockmax        [[buffer(1)]],
                               constant Q4TileDims &d        [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
  uint sub = gid.x, row = gid.y;
  if (sub >= d.subs || row >= uint(ROWS)) return;
  device const float *xb = x + ulong(row) * d.input_size + sub * 32;
  float m = 0.0f;
  for (uint k = 0; k < 32; ++k) m = max(m, abs(xb[k]));
  blockmax[row * d.subs + sub] = m;
}

// ONE THREADGROUP PER ROW (2026-09-24). This was one THREAD per row walking all k/32 block maxima
// serially — the draft RMSnorm's pathology again: 10.3 / 11.8 / 34.4 us at k = 5120 / 6144 / 17408
// against ~2 us for each of the other three passes (a standalone probe), and
// it runs before every unique matmul input, ~256 times an 8-row verify pass (~4.3 ms of it).
// max() is exact and order-free, so the scales are BIT-IDENTICAL to the serial loop.
// DISPATCH: ROWS threadgroups (one per row) of Q4TILE_SCALE_TG threads.
constant constexpr uint Q4TILE_SCALE_TG = 256;
template <int ROWS>
kernel void q4tile_scale(device const float *blockmax    [[buffer(0)]],
                            device float *inv_scale         [[buffer(1)]],
                            constant Q4TileDims &d          [[buffer(2)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint t   [[thread_index_in_threadgroup]],
                            uint sl  [[thread_index_in_simdgroup]],
                            uint sg  [[simdgroup_index_in_threadgroup]]) {
  if (row >= uint(ROWS)) return;
  threadgroup float part[Q4TILE_SCALE_TG / 32];
  float m = 0.0f;
  for (uint sub = t; sub < d.subs; sub += Q4TILE_SCALE_TG) m = max(m, blockmax[row * d.subs + sub]);
  m = simd_max(m);
  if (sl == 0) part[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  float amax = 0.0f;
  for (uint i = 0; i < Q4TILE_SCALE_TG / 32; ++i) amax = max(amax, part[i]);
  float scale = 1.0f;
  if (amax > 0.0f && isfinite(amax)) {
    int e = 0;
    frexp(amax, e);              // amax = m * 2^e, m in [0.5, 1)
    scale = ldexp(1.0f, 14 - e); // amax * scale = m * 2^14, in [8192, 16384)
  }
  inv_scale[row] = 1.0f / scale; // exact: scale is a power of two
}

template <int ROWS>
kernel void q4tile_narrow(device const float *x          [[buffer(0)]],
                             device const float *inv_scale  [[buffer(1)]],
                             device half *xh                [[buffer(2)]],
                             device float *sums             [[buffer(3)]],
                             constant Q4TileDims &d         [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]) {
  uint sub = gid.x, row = gid.y;
  if (sub >= d.subs || row >= uint(ROWS)) return;
  float scale = 1.0f / inv_scale[row]; // exact again
  device const float *xb = x + ulong(row) * d.input_size + sub * 32;
  device half *hb = xh + ulong(row) * d.input_size + sub * 32;
  float s = 0.0f;
  for (uint k = 0; k < 32; ++k) {
    float v = xb[k];
    s += v;
    hb[k] = half(v * scale);
  }
  sums[sub * ROWS + row] = s;
}

// THE MATMUL. It reads the SHIPPED row-major Q4KS buffers through a STRIDED tensor view — strides
// {1, k} instead of a contiguous {1, 32} — so NO second copy of the weights exists. Row-major: nibble (row r, input c) at r*k + c, low-first, which is exactly the byte order
// of the island's u32 `codes` stream; scales/mins are [row][sub] bytes; `dd` is half2 {d, dmin} per
// (row, super-block) — the very buffers the shipped GEMV binds.
//
// 🔴 SUPERSEDED 2026-09-21 — THE LAYOUT PENALTY BELOW BELONGED TO THE UNSPLIT KERNEL. With the 8-way
// split (`q4rm_split`) the two layouts measure THE SAME: a standalone probe
// 8 8 <in> <out> [tile]`, interleaved, us per matmul, strided vs contiguous tiles —
// 5120->17408 301/279 vs 277/277; 17408->5120 294/308 vs 304/289; 5120->12288 201/201 vs 200/199;
// 5120->10240 180/170 vs 168/173; 5120->6144 114/106 vs 104/105; 6144->5120 106/106 vs 107/106.
// A tile-only weight layout would buy NOTHING at 8 rows. DO NOT RE-CHASE the layout for the
// decode unit. (The numbers that follow are kept: they are why the strided view was chosen.)
// MEASURED (mpp_q4_rowmajor_strided_probe): correct to 5e-5; 8 rows of 5120->17408 in 511 us against
// 371 us for contiguous tiles — ~40% slower, and it costs NO MEMORY. That trade is the whole point:
// the 27B on a 36 GB Mac has no room for a tile copy, and a tile-native b=1 GEMV measured 1.5-2.25x
// slower than the shipped one, so "tiles only" was never an option either.
template <int ROWS>
kernel void q4rm_mm(device half *xh                 [[buffer(0)]],
                    device uchar *codes             [[buffer(1)]],
                    device const uchar *scales      [[buffer(2)]],
                    device const char *mins         [[buffer(3)]],
                    device const half2 *dd          [[buffer(4)]],
                    device const float *sums        [[buffer(5)]],
                    device const float *inv_scale   [[buffer(6)]],
                    device float *y                 [[buffer(7)]],
                    constant Q4TileDims &d          [[buffer(8)]],
                    uint tile [[threadgroup_position_in_grid]]) {
  constexpr int TILEN = 256, GW = 32;
  if (tile >= d.output_size / TILEN) return;
  auto a = tensor(xh, dextents<int, 2>{int(d.input_size), ROWS}, array<int, 2>{1, int(d.input_size)});
  constexpr auto descriptor = matmul2d_descriptor(ROWS, TILEN, GW, false, true, false);
  matmul2d<descriptor, execution_simdgroups<8>> operation;

  ulong row0 = ulong(tile) * TILEN;
  device uchar *tile_w = codes + row0 * d.input_size / 2;
  auto a0 = a.slice<GW, ROWS>(0, 0);
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b_first(
      tile_w, dextents<int, 2>{GW, TILEN}, array<int, 2>{1, int(d.input_size)});
  auto b0 = b_first.slice<GW, TILEN>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  for (uint sub = 0; sub < d.subs; ++sub) {
    auto a_s = a.slice<GW, ROWS>(sub * GW, 0);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b(
        tile_w + ulong(sub) * GW / 2, dextents<int, 2>{GW, TILEN}, array<int, 2>{1, int(d.input_size)});
    auto b_s = b.slice<GW, TILEN>(0, 0);
    decltype(acc) part;
    operation.run(a_s, b_s, part);
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      auto idx = acc.get_multidimensional_index(i); // [0] = column in the tile, [1] = row
      ulong r = row0 + idx[0];
      half2 dm = dd[r * d.supers + sub / 8];
      float s = float(dm.x) * float(scales[r * d.subs + sub]);
      float lo = float(dm.y) * float(mins[r * d.subs + sub]);
      acc[i] += part[i] * inv_scale[idx[1]] * s + sums[sub * ROWS + idx[1]] * lo;
    }
  }
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto idx = acc.get_multidimensional_index(i);
    y[ulong(idx[1]) * d.output_size + row0 + idx[0]] = acc[i];
  }
}

// THE SPLIT DECODE MATMUL (2026-09-20). `q4rm_mm` gives each 256-row tile ONE threadgroup that walks
// all k/32 sub-blocks serially, so at 8-16 rows a matmul is LATENCY-bound on that loop while most of
// the GPU idles: one tile costs what the whole projection costs (5120->256 202 us, 17408->256 722 us
// against 718 us for the full 20-tile down projection). Here a tile's sub-blocks are cut into
// Q4RM_PARTS ranges, each its own threadgroup writing a PARTIAL output, and `q4rm_sum` adds them.
// MEASURED before it was built (unpublished probe mpp/probe_split.swift, strided layout, sum kernel
// included, us per matmul at 8 rows, split 1 -> 8): down 607 -> 294, gate/up 424 -> 279,
// attn q 292 -> 200, k/v 163 -> 27, gdn qkv 283 -> 170, z 174 -> 123 (111 at 4), out 200 -> 106;
// at 16 rows down 796 -> 390, gate/up 569 -> 394, qkv 389 -> 230, k/v 224 -> 35. 16 parts is no
// better than 8. The per-element arithmetic is `q4rm_mm`'s; only the ORDER of the sub-block sum
// changes (8 partial sums, then their sum), so results differ in the last float bits.
constant constexpr uint Q4RM_PARTS = 8; // k/32 is a multiple of 8 for every whole-super-block k
template <int ROWS>
kernel void q4rm_split(device half *xh                 [[buffer(0)]],
                       device uchar *codes             [[buffer(1)]],
                       device const uchar *scales      [[buffer(2)]],
                       device const char *mins         [[buffer(3)]],
                       device const half2 *dd          [[buffer(4)]],
                       device const float *sums        [[buffer(5)]],
                       device const float *inv_scale   [[buffer(6)]],
                       device float *yparts            [[buffer(7)]],
                       constant Q4TileDims &d          [[buffer(8)]],
                       uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr int TILEN = 256, GW = 32;
  uint tile = tg.x, part = tg.y;
  if (tile >= d.output_size / TILEN || part >= Q4RM_PARTS) return;
  device float *y = yparts + ulong(part) * ROWS * d.output_size;
  auto a = tensor(xh, dextents<int, 2>{int(d.input_size), ROWS}, array<int, 2>{1, int(d.input_size)});
  constexpr auto descriptor = matmul2d_descriptor(ROWS, TILEN, GW, false, true, false);
  matmul2d<descriptor, execution_simdgroups<8>> operation;

  ulong row0 = ulong(tile) * TILEN;
  device uchar *tile_w = codes + row0 * d.input_size / 2;
  auto a0 = a.slice<GW, ROWS>(0, 0);
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b_first(
      tile_w, dextents<int, 2>{GW, TILEN}, array<int, 2>{1, int(d.input_size)});
  auto b0 = b_first.slice<GW, TILEN>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  uint per = d.subs / Q4RM_PARTS;
  for (uint sub = part * per; sub < (part + 1) * per; ++sub) {
    auto a_s = a.slice<GW, ROWS>(sub * GW, 0);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b(
        tile_w + ulong(sub) * GW / 2, dextents<int, 2>{GW, TILEN}, array<int, 2>{1, int(d.input_size)});
    auto b_s = b.slice<GW, TILEN>(0, 0);
    decltype(acc) partial;
    operation.run(a_s, b_s, partial);
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      auto idx = acc.get_multidimensional_index(i); // [0] = column in the tile, [1] = row
      ulong r = row0 + idx[0];
      half2 dm = dd[r * d.supers + sub / 8];
      float s = float(dm.x) * float(scales[r * d.subs + sub]);
      float lo = float(dm.y) * float(mins[r * d.subs + sub]);
      acc[i] += partial[i] * inv_scale[idx[1]] * s + sums[sub * ROWS + idx[1]] * lo;
    }
  }
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto idx = acc.get_multidimensional_index(i);
    y[ulong(idx[1]) * d.output_size + row0 + idx[0]] = acc[i];
  }
}

// y[i] = sum over the Q4RM_PARTS partial outputs. One thread per output element of the unit.
template <int ROWS>
kernel void q4rm_sum(device const float *yparts [[buffer(0)]],
                     device float *y            [[buffer(1)]],
                     constant Q4TileDims &d     [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
  uint total = uint(ROWS) * d.output_size;
  if (i >= total) return;
  float s = yparts[i];
  for (uint q = 1; q < Q4RM_PARTS; ++q) s += yparts[ulong(q) * total + i];
  y[i] = s;
}


// The two units. The primitive takes multiples of 8 only; 8 serves 5..=8 live rows and 16 serves
// 9..=16. Explicit instantiation because ROWS feeds `matmul2d_descriptor`, which must be constexpr
// — a function constant cannot be a template argument.
#define Q4TILE_UNIT(R)                                                                              \
  template [[host_name("q4tile_blockmax_r" #R)]] kernel void q4tile_blockmax<R>(                     \
      device const float *, device float *, constant Q4TileDims &, uint2);                          \
  template [[host_name("q4tile_scale_r" #R)]] kernel void q4tile_scale<R>(                           \
      device const float *, device float *, constant Q4TileDims &, uint, uint, uint, uint);         \
  template [[host_name("q4tile_narrow_r" #R)]] kernel void q4tile_narrow<R>(                         \
      device const float *, device const float *, device half *, device float *,                    \
      constant Q4TileDims &, uint2);                                                                \
  template [[host_name("q4rm_mm_r" #R)]] kernel void q4rm_mm<R>(                                     \
      device half *, device uchar *, device const uchar *, device const char *,                     \
      device const half2 *, device const float *, device const float *, device float *,             \
      constant Q4TileDims &, uint);
Q4TILE_UNIT(8)
Q4TILE_UNIT(16)
// THE PREFILL UNIT (2026-09-19). Dispatched once PER 32-ROW TILE of a wide window, at buffer
// offsets, inside one concurrent encoder — so a 128-row window keeps 4 x (n/256) threadgroups in
// flight. MEASURED (unpublished probe probe_prefill.swift, 256 rows in flight, 5120->17408): 19.2 us/row
// on THIS strided no-copy layout against 17.4 for contiguous tiles — at saturation the layout
// penalty that costs ~1.5x at 8 rows is gone — and 28 / 21.6 / 19.2 us/row at 8 / 16 / 32-row
// units. One 32-row unit ALONE is no better per row than 16 (33 vs 30 us): the win is the
// row-tiles running side by side, not the unit.
Q4TILE_UNIT(32)
#define Q4RM_SPLIT_UNIT(R)                                                                          \
  template [[host_name("q4rm_split_r" #R)]] kernel void q4rm_split<R>(                               \
      device half *, device uchar *, device const uchar *, device const char *,                     \
      device const half2 *, device const float *, device const float *, device float *,             \
      constant Q4TileDims &, uint2);                                                                \
  template [[host_name("q4rm_sum_r" #R)]] kernel void q4rm_sum<R>(                                   \
      device const float *, device float *, constant Q4TileDims &, uint);
Q4RM_SPLIT_UNIT(8)
Q4RM_SPLIT_UNIT(16)
#undef Q4TILE_UNIT

// THE FUSED OPERAND PASS (2026-09-27). The three passes above, dispatched per 32-row tile of a
// prefill window, cost 0.73 s of a 14.6 s TTFT at 2,444 tokens (ARF_DUP_PF2_NARROW, in context).
// NOT the dispatch count: one dispatch per pass over the whole window measured level (TTFT 14.33 /
// 15.01 vs 14.39 / 14.87 s, interleaved, text and logprobs identical) — and a dependent stage costs
// ~1.2 us in context. It is the kernels: one thread per (row, sub-block) reading 32 floats, lanes
// 128 bytes apart. Here ONE threadgroup per row does all three: a coalesced float4 pass for the
// row's largest magnitude (max is exact and order-free — the same scale as blockmax -> scale),
// then each simdgroup stages 32 sub-blocks with coalesced loads (writing the halves as it goes)
// and each lane folds ITS sub-block's 32 values in order — the per-thread loop's exact sum. The
// operand is bit-identical to the three passes; the sums go to the tiled layout the prefill matmul
// reads, [tile][sub][TILE]. No blockmax buffer. DISPATCH: one threadgroup of Q4PREP_TG per row.
constant constexpr uint Q4PREP_SIMDS = 4;
constant constexpr uint Q4PREP_TG = Q4PREP_SIMDS * 32;
template <int TILE>
kernel void q4tile_prep_win(device const float *x          [[buffer(0)]],
                            device float *inv_scale        [[buffer(1)]],
                            device half *xh                [[buffer(2)]],
                            device float *sums             [[buffer(3)]],
                            constant Q4TileDims &d         [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint t   [[thread_index_in_threadgroup]],
                            uint sl  [[thread_index_in_simdgroup]],
                            uint sg  [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[Q4PREP_SIMDS];
  threadgroup float staged[Q4PREP_SIMDS * 32 * 33]; // 33: lane l reads row l of 32 without conflicts
  device const float *xr = x + ulong(row) * d.input_size;
  device const float4 *x4 = reinterpret_cast<device const float4 *>(xr);
  float m = 0.0f;
  for (uint i = t; i < d.input_size / 4; i += Q4PREP_TG) {
    float4 v = abs(x4[i]);
    m = max(m, max(max(v.x, v.y), max(v.z, v.w)));
  }
  m = simd_max(m);
  if (sl == 0) part[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float amax = 0.0f;
  for (uint i = 0; i < Q4PREP_SIMDS; ++i) amax = max(amax, part[i]);
  float scale = 1.0f;
  if (amax > 0.0f && isfinite(amax)) {
    int e = 0;
    frexp(amax, e);              // amax = m * 2^e, m in [0.5, 1)
    scale = ldexp(1.0f, 14 - e); // amax * scale = m * 2^14, in [8192, 16384)
  }
  const float inv = 1.0f / scale; // exact: scale is a power of two
  if (t == 0) inv_scale[row] = inv;
  scale = 1.0f / inv;             // as q4tile_narrow recomputes it
  threadgroup float *st = staged + sg * (32 * 33);
  device half *hr = xh + ulong(row) * d.input_size;
  const uint tile_base = (row / TILE) * d.subs * TILE + row % TILE;
  for (uint c = sg * 32; c < d.subs; c += Q4PREP_SIMDS * 32) {
    const uint nsub = min(32u, d.subs - c);
    for (uint j = 0; j < nsub; ++j) {
      const uint idx = (c + j) * 32 + sl; // element sl of sub-block c + j
      const float v = xr[idx];
      hr[idx] = half(v * scale);
      st[j * 33 + sl] = v;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (sl < nsub) {
      float s = 0.0f;
      for (uint k = 0; k < 32; ++k) s += st[sl * 33 + k];
      sums[tile_base + (c + sl) * TILE] = s;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
template [[host_name("q4tile_prep_win_t32")]] kernel void q4tile_prep_win<32>(
    device const float *, device float *, device half *, device float *, constant Q4TileDims &,
    uint, uint, uint, uint);

// =================================================================================================
// THE HAND-WRITTEN simdgroup_matrix DECODE MATMUL (`q4sg_*`), 2026-09-21.
//
// 1.31-1.48x over `q4rm_mm` above on EVERY decode shape, at the SAME Q4_K_S group-32 two-level
// scales — no format change, no second weight copy, no quality cost. Behind `ARF_SG_Q4=1` until
// the end-to-end A/B is recorded.
//
// WHY IT WINS, isolated by three diagnostic arms (measured 2026-09-21, and
// a standalone probe):
// it is THE PER-ELEMENT SCALE GATHER and nothing else. MPP's cooperative tensor scatters a lane's
// 8 accumulator elements across 8 DIFFERENT output columns, so each needs its own
// (d, dmin, scale, min) — 3.00 scalar loads per output, 119-136 us of a 415-594 us matmul (23-29%).
// This kernel's lane layout puts D[fm][fn] and D[fm][fn+1] in the SAME output column (different
// rows), so one gather serves both halves of the float2 — 1.50 per output, exactly the 2.0x
// measured. Caching `dd` (per 256-input super-block) across its 8 sub-blocks took it further,
// a prediction of the same theory that then measured: 1.22 -> 1.31x, 1.38 -> 1.48x.
//
// 🔴 THREE HYPOTHESES THAT MEASURED FALSE — DO NOT RE-CHASE:
//   1. "software-pipeline the weight load" (Splash's `load(g+1)` above `run_group(g)`). Prefetch is
//      monotonically SLOWER here: depth 0/1/2 = 316/396/453 us. The hardware already hides that
//      load and the staging registers spill. depth 0 with 2 chains is the fastest of everything
//      tried. This was my headline hypothesis and it is WRONG.
//   2. "MPP's per-operation.run() call overhead" — FLAT over a 4x change in call count
//      (160/80/40 calls: 281/283/280 us).
//   3. "the cooperative-tensor traversal / is_valid_element bookkeeping" — free (281 -> 280 us).
//      `get_multidimensional_index` is ~15 us; hoisting it measured 0.97x, i.e. slightly worse.
//
// This also retires the LAST argument for group-64 (⛔ rejected on this ledger, +0.71% perplexity):
// Splash's epilogue looks cheap because of its LANE LAYOUT, not its 64-wide groups. We get the
// same benefit at group-32 and keep the quality.
//
// THE 1024+q TRICK, the half analogue of Splash's bf16 `128+q`: in half, `0x6400 | q` is EXACTLY
// 1024+q for q in 0..15 (ulp 1.0 — verified numerically before this was written). So the MMA
// consumes (1024+q)*x and the epilogue subtracts 1024*sum(x) in fp32. One integer OR replaces the
// dequantise, and the activation keeps its full power-of-two-scaled range with no denormals.
// =================================================================================================

// The MMA contract, which fixes every index below:
//   simdgroup_multiply_accumulate(D, A, B, C) computes D[fm][fn] = sum_k A[fm][k] * B[k][fn],
//   and a lane's thread_elements() are M[fm][fn] and M[fm][fn+1].
// Take A = weights (fm -> output column, k -> input) and B = activations (k -> input, fn -> row).
// Then a lane's A-operand is two ADJACENT nibbles at input j*8+fn of output column col0+fm — and
// `fn` is always even, so that pair is exactly ONE BYTE, low nibble then high.
struct Q4SgLane { ushort fm; ushort fn; };
inline Q4SgLane q4sg_lane_map(uint lane) {
  const uint qid = lane >> 2;
  Q4SgLane l;
  l.fm = ushort((qid & 4) | ((lane >> 1) & 3));
  l.fn = ushort(((qid & 2) << 1) | ((lane & 1) << 1));
  return l;
}
template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2> &q4sg_te(thread simdgroup_matrix<T, 8, 8> &m) {
  return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}
// One 8x8x8 MMA on plain-register operands: c += a x b. The simdgroup_matrix objects live ONLY
// inside this call (the MLX "steel" pattern), so the persistent accumulator stays an ordinary
// float2 the compiler keeps in registers — a persistent simdgroup_matrix read through
// thread_elements() each group lands in thread memory instead.
template <typename T>
__attribute__((always_inline)) inline void q4sg_mma(thread float2 &c, vec<T, 2> a, vec<T, 2> b) {
  simdgroup_matrix<T, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, D;
  q4sg_te(A) = a;
  q4sg_te(B) = b;
  q4sg_te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = q4sg_te(D);
}

// The B-operand is x[input][row] — lane (fm,fn) wants input j*8+fm for rows fn and fn+1, two
// ADJACENT rows at one input. So the table is transposed relative to `q4tile_narrow`'s output:
// table[g][kk][m], kk = 0..31 the input within the sub-block, m = 0..7 the row. One 32-bit read.
// Written ONCE per matmul; the n/256 tiles all read it.
template <int ROWS>
kernel void q4sg_prepare(device const half *xh       [[buffer(0)]],
                         device half *table          [[buffer(1)]],
                         device float *sumh          [[buffer(3)]],
                         constant Q4TileDims &d      [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
  uint g = gid.x, row = gid.y;
  if (g >= d.subs || row >= uint(ROWS)) return;
  device const half *xb = xh + ulong(row) * d.input_size + g * 32;
  device half *t = table + ulong(g) * 32 * ROWS;
  // 🔴 THE BUG THIS FIXES, and it cost an evening: the 1024 offset MUST be cancelled against the
  // sum of the NARROWED values, because that is what the MMA accumulated. Cancelling it against
  // the f32 originals rescaled (`sums[..] / inv_scale`) carries a DIFFERENT rounding and leaves
  // ~1.5% error per sub-block — 113x worse (1.45e-2 vs 1.28e-4 on a 32-element model of it).
  // `sums` (of the originals) stays exactly as it is: the AFFINE term wants those, and keeping it
  // that way is why the affine term carries no narrowing error. Two sums, two different jobs.
  float s = 0.0f;
  for (uint k = 0; k < 32; ++k) { const half v = xb[k]; t[k * ROWS + row] = v; s += float(v); }
  sumh[g * ROWS + row] = s;
}

// CHAINS independent 8-column accumulator chains per simdgroup. 2 measured fastest of {1,2,4}:
// it has the smallest live register footprint, and this kernel is register-bound, not
// latency-bound (see the refuted hypotheses above).
constexpr constant uint Q4SG_CHAINS = 2;

// SPLIT-K (2026-09-22): `tg.y` = which of Q4RM_PARTS slices of the sub-block range this
// threadgroup walks; each writes its partial to `y + part * ROWS * n` (the `q4rm_split` layout),
// and the existing `q4rm_sum` reduces. WHY: at 5120->17408 the unsplit kernel is 68 threadgroups
// walking 160 sub-blocks each — far too little in flight for 40 cores; the shipped MPP kernel's
// ~25% came from exactly this split, and Splash's kernel splits K too (`persistent_groups`) —
// the one part of the port that was dropped. A grid with height 1 is the unsplit kernel.
template <int ROWS>
kernel void q4sg_mm(device const half *table        [[buffer(0)]],
                    device const uchar *codes       [[buffer(1)]],
                    device const uchar *scales      [[buffer(2)]],
                    device const char *mins         [[buffer(3)]],
                    device const half2 *dd          [[buffer(4)]],
                    device const float *sums        [[buffer(5)]],
                    device const float *inv_scale   [[buffer(6)]],
                    device float *y                 [[buffer(7)]],
                    device const float *sumh_in     [[buffer(9)]],
                    constant Q4TileDims &d          [[buffer(8)]],
                    uint2 tg2 [[threadgroup_position_in_grid]],
                    uint2 tgn [[threadgroups_per_grid]],
                    uint sgi  [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint CH = Q4SG_CHAINS;
  const uint tg = tg2.x, part = tg2.y, nparts = tgn.y;
  const uint per = d.subs / nparts;
  const uint g0 = part * per, g1 = part + 1 == nparts ? d.subs : g0 + per;
  y += ulong(part) * ROWS * d.output_size;
  // An 8x8 MMA covers EXACTLY 8 rows: `fn` takes {0,2,4,6} and each lane holds rows fn, fn+1.
  // So a 16-row unit is TWO row-blocks, each with its own B-operand and accumulators. (The first
  // version of this kernel had RB=1 and silently left rows 8..15 as whatever the buffer held —
  // 7e-1 error at 16 rows while 8 rows looked fine.)
  constexpr uint RB = ROWS / 8;
  const Q4SgLane l = q4sg_lane_map(lane);
  const uint fm = l.fm, fn = l.fn;
  const uint col0 = tg * 256 + sgi * (CH * 8) + fm;

  float2 acc[RB][CH];
  for (uint rb = 0; rb < RB; ++rb)
    for (uint ch = 0; ch < CH; ++ch) acc[rb][ch] = float2(0);
  half2 dmc[CH];   // per-256-input super-block d/dmin, re-read only when the super-block changes

  // Our SHIPPED row-major Q4_K_S: nibble (column c, input k) at c*K + k, low-first. One 32-input
  // sub-block of one column is 16 CONTIGUOUS bytes. The chains are columns 8 apart.
  const ulong colstep = ulong(d.input_size) / 2;
  device const uchar *wbase = codes + ulong(col0) * colstep;

  for (uint g = g0; g < g1; ++g) {
    uint w[CH][4];
    for (uint ch = 0; ch < CH; ++ch) {
      const uint4 v = *reinterpret_cast<device const uint4 *>(wbase + ch * 8 * colstep + ulong(g) * 16);
      w[ch][0] = v.x; w[ch][1] = v.y; w[ch][2] = v.z; w[ch][3] = v.w;
    }
    device const half *xt = table + ulong(g) * 32 * ROWS;
    float2 dot[RB][CH];
    for (uint rb = 0; rb < RB; ++rb)
      for (uint ch = 0; ch < CH; ++ch) dot[rb][ch] = float2(0);
    for (uint j = 0; j < 4; ++j) {                 // four 8-input fragments per 32-input sub-block
      const uint byteIdx = (j * 8 + fn) / 2;       // 0..15 within the 16-byte sub-block
      for (uint ch = 0; ch < CH; ++ch) {
        const uint word = w[ch][byteIdx / 4];
        const uint byte = (word >> (8 * (byteIdx & 3))) & 0xFFu;
        const uint pair = ((byte & 0xFu) | ((byte & 0xF0u) << 12)) | 0x64006400u;
        const half2 a = as_type<half2>(pair);
        // The A-operand (weights) is shared by every row-block; only B moves.
        for (uint rb = 0; rb < RB; ++rb) {
          const half2 b =
              *reinterpret_cast<device const half2 *>(xt + (j * 8 + fm) * ROWS + rb * 8 + fn);
          q4sg_mma<half>(dot[rb][ch], a, b);
        }
      }
    }
    // THE EPILOGUE, and the whole point: ONE gather per chain per sub-block (plus dd every 8th),
    // against MPP's three per accumulator element.
    if (g == g0 || (g & 7u) == 0u)
      for (uint ch = 0; ch < CH; ++ch) dmc[ch] = dd[ulong(col0 + ch * 8) * d.supers + g / 8];
    for (uint ch = 0; ch < CH; ++ch) {
      const ulong c = ulong(col0) + ch * 8;
      const ulong prm = c * d.subs + g;
      const float s = float(dmc[ch].x) * float(scales[prm]);
      const float lo = float(dmc[ch].y) * float(mins[prm]);
      for (uint rb = 0; rb < RB; ++rb) {
        const uint r = rb * 8 + fn;
        const float2 sum = float2(sums[g * ROWS + r], sums[g * ROWS + r + 1]);
        const float2 isc = float2(inv_scale[r], inv_scale[r + 1]);
        // The 1024-offset, cancelled against the sum of the NARROWED values (see q4sg_prepare):
        // sum(x)/inv_scale is NOT the same number and leaves ~1.5% per sub-block.
        const float2 sumh = float2(sumh_in[g * ROWS + r], sumh_in[g * ROWS + r + 1]);
        // `dot` is sum over k of (1024 + q)*xh, in NARROWED units, so subtract 1024*sum(xh) in
        // the same units and then undo the narrowing with inv_scale (exact: a power of two).
        // `sums` holds the f32 ORIGINALS (the affine term carries no narrowing error), so the
        // min term takes NO inv_scale — matching `q4rm_mm`'s epilogue above.
        const float2 q = fma(-1024.0f, sumh, dot[rb][ch]);
        acc[rb][ch] = fma(q * isc, s, acc[rb][ch]);
        acc[rb][ch] = fma(sum, lo, acc[rb][ch]);
      }
    }
  }
  for (uint ch = 0; ch < CH; ++ch) {
    const ulong c = ulong(col0) + ch * 8;
    for (uint rb = 0; rb < RB; ++rb) {
      const uint r = rb * 8 + fn;
      y[ulong(r) * d.output_size + c] = acc[rb][ch].x;
      y[ulong(r + 1) * d.output_size + c] = acc[rb][ch].y;
    }
  }
}

#define Q4SG_UNIT(R)                                                                                \
  template [[host_name("q4sg_prepare_r" #R)]] kernel void q4sg_prepare<R>(                          \
      device const half *, device half *, device float *, constant Q4TileDims &, uint2);            \
  template [[host_name("q4sg_mm_r" #R)]] kernel void q4sg_mm<R>(                                    \
      device const half *, device const uchar *, device const uchar *, device const char *,         \
      device const half2 *, device const float *, device const float *, device float *,             \
      device const float *, constant Q4TileDims &, uint2, uint2, uint, uint);
Q4SG_UNIT(8)
Q4SG_UNIT(16)
#undef Q4SG_UNIT

// =================================================================================================
// q4sg2 — THE SCALE GOES INTO THE MMA OPERAND (2026-09-23). Default; `ARF_SG_V1=1` = the kernel above.
//
// HOW IT WAS FOUND (a standalone probe measured
// 2026-09-23). The 8-row verify pass is ~90 ms GPU, 71 ms of it these matmuls at ~198 GB/s. Splash's
// OWN kernel, read verbatim from its source tree and timed on this box, is 1.35-1.44x faster on
// every verify shape. Bisected, keeping our group-32 format:
//   - weight LAYOUT (tiled like Splash vs our row-major)          1.01x  — not it
//   - Splash's 4-simdgroup / 64-column threadgroup shape          0.92x  — slower here
//   - epilogue REMOVED entirely (diagnostic, wrong output)         1.35x  — == Splash. IT IS THE EPILOGUE.
// Our two-level group-32 epilogue (u8 scale x f16 d, i8 min x f16 dmin, the 1024 cancellation
// against the narrowed sums) ran once per 32 inputs per chain; Splash's group-64 one per 64.
//
// THE FIX, format unchanged: q*sc (<= 15*255 = 3825) fits half with no overflow — exact to
// sc <= 136 — and goes straight into the MMA's A operand as ((1024+q) - 1024) * sc, the MMA
// accumulates a whole 256-input super-block, and d is applied ONCE per super-block. The min
// term is one float2 fma per sub-block (dmin once per super-block). No 1024 offset survives to
// be cancelled — so no `sumh`, and the error drops ~70x (3.8e-7 vs 2.7e-5 of rms, probe).
//
// AND THE LOADS. The v1 lane loads 16 bytes of a sub-block and uses 4 (the four lanes sharing a
// column load the same bytes): permuting them so a lane reads ONE contiguous uint was worth
// 1.06 -> 1.2x on top. The weights cannot move (the GEMV and prefill kernels read the same
// buffers), so the permutation moves to the ACTIVATIONS: the MMA's k is a dummy summation index,
// so slot s of fragment j may hold any input as long as A and B agree. Lane p = fn/2 reads the
// NATURAL bytes 4p..4p+3 and uses byte j in fragment j -> inputs 8p+2j+{0,1} in slots {fn,fn+1};
// `q4sg2_prepare` writes B row s of fragment j as input 8*(s>>1) + 2j + (s&1). (Splash does the
// same thing with its `klogical`.) The table is also in Splash's quad layout: ONE vec<half,8>
// load per lane per sub-block carries all four fragments.
//
// ⛔ MEASURED OUT 2026-09-25 — THE bf16 OPERAND (Splash 1.0.2's format; measured with a
// standalone probe and kernels `q4bf_prepare`/`q4bf_mm`, since removed). Its ONE prep pass is
// 3 us against the half path's ~10 us four-pass chain — but q*sc (<= 3825) is not exact in bf16's
// 8 mantissa bits, so the sub-block scale cannot ride in the operand and the epilogue returns once
// per 32 inputs: the matmul alone 281 / 278 / 174 / 103 us vs 181 / 195 / 111 / 71 (5120->17408,
// 17408->5120, 5120->10240, 6144->5120), 25-45% slower end to end, and 7-8x the error against an
// f64 reference (1.6-1.9e-3 vs 1.9-2.6e-4). Splash affords bf16 because ITS format carries one
// scale per 64 inputs. With group-32 two-level scales the half operand is the right one.
// Probe, all correct against a CPU reference, 8 rows, parts 8 (us, v1 -> v2):
//   5120->17408 220->183 | 17408->5120 236->188 | 6144->5120 86->71 | 5120->10240 135->113
//   5120->6144 85->71 | 5120->12288 160->133      = 1.19-1.33x
// Loads issued 4 sub-blocks at a time (X=4) — 1.03-1.05x over one at a time. Walking whole
// super-blocks with one uint4 of scales MEASURED SLOWER (0.92-1.04x: register pressure).
// MEASURED-OUT, DO NOT RE-CHASE (2026-09-23, sg_variant_probe on this kernel's own source):
//   - "v3": raw (1024+q) operand, scale applied per sub-block to the MMA result (2 float2 fmas
//     instead of 8 half ops): 1.21x — identical to this kernel.
//   - one uint of 4 sub-scales + one of 4 mins per chain per X, instead of byte loads: 1.20-1.24x
//     — identical. The last ~10% to the no-epilogue bound (1.33-1.40x, == Splash's kernel) is the
//     price of carrying group-32 two-level scales at all.
// =================================================================================================

// The table: [sub-block g][row-block rb][slot s 0..7][row-pair h 0..3][fragment j 0..3][2 rows], half.
template <int ROWS>
kernel void q4sg2_prepare(device const half *xh       [[buffer(0)]],
                          device half *table          [[buffer(1)]],
                          constant Q4TileDims &d      [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
  constexpr uint RB = ROWS / 8;
  const uint g = gid.x, row = gid.y;
  if (g >= d.subs || row >= uint(ROWS)) return;
  device const half *xb = xh + ulong(row) * d.input_size + g * 32;
  const uint rb = row / 8, rr = row % 8, h = rr / 2, rbit = rr % 2;
  device half *t = table + (ulong(g) * RB + rb) * 256;
  for (uint kk = 0; kk < 32; ++kk) {
    const uint a = kk / 8, j = (kk % 8) / 2, b = kk % 2;
    const uint slot = 2 * a + b;
    t[((slot * 4 + h) * 4 + j) * 2 + rbit] = xb[kk];
  }
}

constexpr constant uint Q4SG2_X = 4;  // sub-blocks whose loads are issued together
// ⛔ MEASURED-OUT 2026-09-25 — SOFTWARE-PIPELINED LOADS (the next X sub-blocks' loads issued before
// this X's math, two register buffers alternated statically). It is what took the group-64 kernel
// 56.2 -> 55.8 ms (G64_PF 2), but this kernel is register-bound: the DFlash draft at 1,024 context
// (examples/dflash2_block, 2 rounds each) went 6.38 / 6.43 -> 7.14 / 6.99 ms, its MLP matmuls
// 3.50 / 3.54 -> 4.10 / 3.96. DO NOT RE-CHASE load pipelining here without freeing registers first.

// Written INLINE, arrays in scope, no helper lambdas: the first port used lambdas taking the
// register arrays as parameters and measured NO faster than v1 in the engine (72.7 vs 71.4 ms per
// verify pass) while the probe's inline body was 1.2x — arrays passed by pointer land in thread
// memory. The HOST guarantees every part's sub-block range is a multiple of X (k % 1024 == 0 at
// 8 parts; true for every projection of the 27B), so there is no tail loop.
template <int ROWS>
kernel void q4sg2_mm(device const half *table        [[buffer(0)]],
                     device const uchar *codes       [[buffer(1)]],
                     device const uchar *scales      [[buffer(2)]],
                     device const char *mins         [[buffer(3)]],
                     device const half2 *dd          [[buffer(4)]],
                     device const float *sums        [[buffer(5)]],
                     device const float *inv_scale   [[buffer(6)]],
                     device float *y                 [[buffer(7)]],
                     constant Q4TileDims &d          [[buffer(8)]],
                     uint2 tg2 [[threadgroup_position_in_grid]],
                     uint2 tgn [[threadgroups_per_grid]],
                     uint sgi  [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint CH = Q4SG_CHAINS, RB = ROWS / 8, X = Q4SG2_X;
  const uint tg = tg2.x, part = tg2.y, nparts = tgn.y;
  const uint per = d.subs / nparts;
  const uint g0 = part * per, g1 = part + 1 == nparts ? d.subs : g0 + per;
  y += ulong(part) * ROWS * d.output_size;
  const Q4SgLane l = q4sg_lane_map(lane);
  const uint fm = l.fm, fn = l.fn;
  const uint col0 = tg * 256 + sgi * (CH * 8) + fm;
  const ulong colstep = ulong(d.input_size) / 2;
  device const uchar *wb = codes + ulong(col0) * colstep + (fn / 2) * 4;
  device const uchar *scb = scales + ulong(col0) * d.subs;
  device const char *mnb = mins + ulong(col0) * d.subs;

  // ONE accumulator per (row-block, chain), in ORIGINAL units: the narrowing's inv_scale is folded
  // into d at each super-block flush. A separate min-term accumulator cost registers, and this
  // kernel is register-bound: with it (plus scale and min in two arrays) the engine build ran
  // 0.93-1.06x of v1 while the probe's body — one accumulator, scale|min in one ushort — ran 1.2x
  // on the SAME buffers (sg_variant_probe, ENGINE-q4sg2 arm).
  float2 isc[RB];
  for (uint rb = 0; rb < RB; ++rb) isc[rb] = *reinterpret_cast<device const float2 *>(inv_scale + rb * 8 + fn);
  float2 acc[RB][CH], dot[RB][CH][2], mnacc[RB][CH];
  for (uint rb = 0; rb < RB; ++rb)
    for (uint ch = 0; ch < CH; ++ch) {
      acc[rb][ch] = float2(0);
      dot[rb][ch][0] = float2(0); dot[rb][ch][1] = float2(0); mnacc[rb][ch] = float2(0);
    }

  for (uint gg = g0; gg < g1; gg += X) {
    uint wv[X][CH]; ushort smx[X][CH]; vec<half, 8> xv[X][RB];
    for (uint u = 0; u < X; ++u) {
      for (uint ch = 0; ch < CH; ++ch) {
        wv[u][ch] = *reinterpret_cast<device const uint *>(wb + ch * 8 * colstep + ulong(gg + u) * 16);
        const ulong prm = ulong(ch * 8) * d.subs + gg + u;
        smx[u][ch] = ushort(scb[prm]) | ushort(ushort(uchar(mnb[prm])) << 8);
      }
      for (uint rb = 0; rb < RB; ++rb)
        xv[u][rb] = *reinterpret_cast<device const vec<half, 8> *>(
            table + ((ulong(gg + u) * RB + rb) * 8 + fm) * 32 + (fn / 2) * 8);
    }
    for (uint u = 0; u < X; ++u) {
      const uint g = gg + u;
      // ONE float2 per row-block per sub-block, hoisted out of the chain loop.
      float2 sm2[RB];
      for (uint rb = 0; rb < RB; ++rb)
        sm2[rb] = *reinterpret_cast<device const float2 *>(sums + g * ROWS + rb * 8 + fn);
      for (uint ch = 0; ch < CH; ++ch) {
        const half2 sc = half2(half(smx[u][ch] & 0xFF));
        for (uint j = 0; j < 4; ++j) {
          const uint byte = (wv[u][ch] >> (8 * j)) & 0xFFu;
          const uint pair = ((byte & 0xFu) | ((byte & 0xF0u) << 12)) | 0x64006400u;
          // (1024+q) - 1024 = q EXACTLY, then q*sc. NOT fma(1024+q, sc, -1024*sc): our
          // sub-scales are full u8 and -1024*255 overflows half to -inf -> inf - inf = NaN on
          // every output (the first build did exactly that; the probe had used scales <= 63).
          // q*sc <= 3825 never overflows; exact to sc <= 136, within half's ulp (2) above.
          const half2 a = (as_type<half2>(pair) - half2(1024.0h)) * sc;
          for (uint rb = 0; rb < RB; ++rb)
            q4sg_mma<half>(dot[rb][ch][j & 1], a, half2(xv[u][rb][2 * j], xv[u][rb][2 * j + 1]));
        }
        const float mn = float(char(smx[u][ch] >> 8));
        for (uint rb = 0; rb < RB; ++rb) mnacc[rb][ch] = fma(sm2[rb], mn, mnacc[rb][ch]);
      }
      if ((g & 7u) == 7u || g + 1 == g1) {       // end of a super-block, or of this part's range
        for (uint ch = 0; ch < CH; ++ch) {
          const half2 dm = dd[ulong(col0 + ch * 8) * d.supers + g / 8];
          for (uint rb = 0; rb < RB; ++rb) {
            // the MMA ran on NARROWED activations: x inv_scale here; the min term used originals
            acc[rb][ch] = fma(dot[rb][ch][0] + dot[rb][ch][1], float(dm.x) * isc[rb], acc[rb][ch]);
            acc[rb][ch] = fma(mnacc[rb][ch], float(dm.y), acc[rb][ch]);
            dot[rb][ch][0] = float2(0); dot[rb][ch][1] = float2(0); mnacc[rb][ch] = float2(0);
          }
        }
      }
    }
  }

  for (uint ch = 0; ch < CH; ++ch) {
    const ulong c = ulong(col0) + ch * 8;
    for (uint rb = 0; rb < RB; ++rb) {
      const uint r = rb * 8 + fn;
      y[ulong(r) * d.output_size + c] = acc[rb][ch].x;
      y[ulong(r + 1) * d.output_size + c] = acc[rb][ch].y;
    }
  }
}

#define Q4SG2_UNIT(R)                                                                               \
  template [[host_name("q4sg2_prepare_r" #R)]] kernel void q4sg2_prepare<R>(                        \
      device const half *, device half *, constant Q4TileDims &, uint2);                            \
  template [[host_name("q4sg2_mm_r" #R)]] kernel void q4sg2_mm<R>(                                  \
      device const half *, device const uchar *, device const uchar *, device const char *,         \
      device const half2 *, device const float *, device const float *, device float *,             \
      constant Q4TileDims &, uint2, uint2, uint, uint);
Q4SG2_UNIT(8)
Q4SG2_UNIT(16)
#undef Q4SG2_UNIT

// =================================================================================================
// PREFILL MATMUL v2 (`q4pf2_mm`, 2026-09-24): DEQUANTIZE ONCE INTO THREADGROUP MEMORY, NO EPILOGUE.
//
// It REPLACED `q4rm_pf` (removed 2026-09-24: 1.22-1.31x per shape, 10K prefill 79.3 -> 63.2 s,
// perplexity unchanged ). That kernel's own bisection still applies to any MPP
// kernel here, kept verbatim:
//   THE PREFILL MATMUL — `q4rm_mm` with two changes, and they are the whole difference (2026-09-19,
//   bisected in unpublished probes probe_prefill2/3.swift, 256 rows in flight, strided 5120->17408, M=32):
//       Splash-style body on OUR strided weights                   20.9 us/row
//       + our two-level u8/i8 x half2 scales                       18.2   (free)
//       + `is_valid_element` in the accumulate loop + per-element write   36.3
//       both = q4rm_mm as shipped                                  52.2   <- measured in-engine too
//       no validity branch, bulk `store` into a FLOAT tensor       19.7
//   The branch defeats `#pragma unroll` on the hot loop. It costs NOTHING at 8 or 16 rows (36.9 vs
//   35.0, 22.6 vs 23.0 us/row), so `q4rm_mm` — the decode path — is left exactly as it was.
//   Without the branch every capacity slot is accumulated, so the indices are CLAMPED: a slot that
//   is not a valid element must not index `dd`/`scales`/`sums` out of range. `store` writes only
//   the valid ones.
//
//
// `q4rm_pf` pays the group-32 affine epilogue every 32 inputs: MPP's cooperative layout puts a
// lane's accumulator elements in 8 different output columns, so each element gathers its own
// (d, sub-scale, dmin, min) from device memory 160-544 times per output. That epilogue is the
// measured gap to Splash on decode (2026-09-21) and prefill runs the same shape (measured
// 2026-09-24: 10K prefill 75.8 s vs Splash 52.9 s, matmuls 79% of ours).
//
// Here a threadgroup owns 32 output columns and ALL `ROWS` rows of the window. For each 256-input
// super-block, its 256 threads dequantize the 32 x 256 weight slice ONCE into threadgroup memory as
// half(d*sc*q + dmin*m) — one thread per (column, 32-input sub-block), one 16-byte load of codes —
// and one MPP op multiplies the window's narrowed activations (device) by it, accumulating in the
// cooperative tensor across the whole k. The only correction left is the narrowing's per-row
// inverse scale, applied once at the store. half keeps ~11 bits of a weight whose 4-bit
// quantization already costs ~6% — the rounding added here is two orders of magnitude smaller.
// The dequant is amortized over ROWS rows (128 MACs per dequantized weight at a full window).
// Grid: (n / 32, 1), 256 threads. Reads the SAME row-major buffers every other Q4 path binds.
template <int ROWS, int TN, int KB>
kernel void q4pf2_mm(device half *xh                 [[buffer(0)]],
                     device const uchar *codes       [[buffer(1)]],
                     device const uchar *scales      [[buffer(2)]],
                     device const char *mins         [[buffer(3)]],
                     device const half2 *dd          [[buffer(4)]],
                     device const float *inv_scale   [[buffer(6)]],
                     device float *y                 [[buffer(7)]],
                     constant Q4TileDims &d          [[buffer(8)]],
                     uint tile [[threadgroup_position_in_grid]],
                     uint tid  [[thread_index_in_threadgroup]]) {
  // TN output columns x KB inputs per chunk. The dequant is split into HALF sub-blocks (16
  // inputs = one 8-byte load) so any TN x KB can be served by 256 threads: TN * KB / 16 items.
  constexpr int SPC = KB / 32, HPC = KB / 16, ITEMS = TN * HPC;
  static_assert(ITEMS % 256 == 0, "whole items per thread");
  if (tile >= d.output_size / TN) return;
  threadgroup half wt[TN * KB];                                  // [column][input], TN*KB*2 bytes
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

  const uint chunks = d.input_size / KB;
  for (uint ck = 0; ck < chunks; ++ck) {
    for (uint item = tid; item < uint(ITEMS); item += 256u) {
      // (column, half sub-block) of this chunk
      const uint col = item / uint(HPC), h = item % uint(HPC);
      const ulong gcol = ulong(tile) * TN + col;
      const uint sub = ck * SPC + h / 2u;
      const half2 dm = dd[gcol * d.supers + sub / 8u];
      const float s = float(dm.x) * float(scales[gcol * d.subs + sub]);
      const float lo = float(dm.y) * float(mins[gcol * d.subs + sub]);
      const uint2 q = *reinterpret_cast<device const uint2 *>(
          codes + gcol * (d.input_size / 2) + ulong(sub) * 16u + (h % 2u) * 8u);
      threadgroup half *dst = wt + col * KB + h * 16u;
      const uint words[2] = {q.x, q.y};
      for (uint w = 0; w < 2; ++w) {
        const uint v = words[w];
        for (uint b = 0; b < 4; ++b) {
          const uint byte = (v >> (8u * b)) & 0xFFu;
          dst[w * 8u + b * 2u]      = half(fma(s, float(byte & 0xFu), lo));
          dst[w * 8u + b * 2u + 1u] = half(fma(s, float(byte >> 4), lo));
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
    const auto idx = acc.get_multidimensional_index(i);          // [0] column in tile, [1] row
    acc[i] *= inv_scale[min(uint(idx[1]), uint(ROWS - 1))];
  }
  acc.store(c.template slice<TN, ROWS>(int(tile) * TN, 0));
}
#define Q4PF2(R, TN, KB)                                                                            \
  template [[host_name("q4pf2_mm_r" #R "_n" #TN)]] kernel void q4pf2_mm<R, TN, KB>(                  \
      device half *, device const uchar *, device const uchar *, device const char *,               \
      device const half2 *, device const float *, device float *, constant Q4TileDims &, uint, uint);
Q4PF2(32, 64, 128)
Q4PF2(128, 64, 128)
// MEASURED-OUT 2026-09-24 (mpp_q4_parity, 128 rows, gpu us gate/up | down | attn q | gdn qkv):
//   64 x 128 (16 KB, kept) 2086 | 2328 | 1459 | 1207      64 x 64 (8 KB)   2137 | 2277 | 1597 | 1267
//   32 x 128 (8 KB)        2247 | 2329 | 1705 | 1364      128 x 32 (8 KB)  2771 | 3381 | 1913 | 1610
//   64 x 256 (32 KB)       2133 | 2281 | 1585 | 1272
// A smaller threadgroup footprint (more threadgroups resident to overlap one's dequant with
// another's MMA) buys nothing: gate/up already runs ~10.9 TFLOPS here. DO NOT RE-CHASE occupancy.
// 64 columns x 128 inputs beat 32 x 256 and 128 x 64 on every large shape (measured
// 2026-09-24); the other widths were deleted with it.
#undef Q4PF2
