// INDIRECT (MoE) DOWN GEMV with router-weight reduce — the down-projection half of the
// kernel_mul_mv_id MoE port. Like gemv_q4ks_id but: (a) the activation is the per-slot silu
// segment silu_all[slot*mi ..], (b) it folds the router weight wts[slot], and (c) it SUMS over
// the top_k routed slots into ONE output mlp_down[h_j] (the wgpu down_reduce semantics). One
// simdgroup per output hidden index h_j; the slot loop is inside (top_k is small). Reuses the
// EXACT dense Q4_K_S dot body — only the per-(slot,expert) weight offset + the wts fold differ.
//
//   mlp_down[h_j] = Σ_slot wts[slot] · Σ_c silu_all[slot*mi + c] · Wdown_{ids[slot]}[h_j, c]
//
// Dispatch: grid = (h, 1, 1), threadgroup = 32 lanes (one simdgroup per output h_j). The lane
// strides the mi/32 sub-blocks; simd_sum reduces. dims {m, k(=mi), n(=h), top_k}.
//
// Per-expert down packing (concatenated over num_experts): for expert e, output row (=hidden col)
// h_j, weight column wcol = e*n + h_j; codes uint4 per sub-block at wcol*(k/32); scales/mins u8/i8
// 4/word at wcol*(k/32); dd half2 {d,dmin} per super at pair wcol*(k/256). IDENTICAL to the WGSL
// matmul_vec_moe_q4ks_down_reduce_b layout, so output matches it within the 1e-3 bar.

#include <metal_stdlib>
using namespace metal;

struct IdDims { uint m; uint k; uint n; uint top_k; };

// ARF_GEMV_ID_NSG2 (function_constant 3, same index/flag as gemv_q4ks_id): simdgroups per
// threadgroup. Default 1 (32 threads, shipped path — the guard keeps the plain compile_msl
// byte-identical). NSG=2 (64 threads) splits the SLOT loop across simdgroups (slot = sgitg,
// sgitg+NSG, ...) — NOT the sub-block axis: down's k=mi=768 gives only 24 sub-blocks, so a
// second simdgroup on the sub-block stride would sit idle, while the top_k(=8) slot axis feeds
// both. Same effect as llama's N_SG=2: ~2× in-flight Q4_K weight loads per threadgroup.
// Below-tie f32 reorder (slot partials summed in a different order) — parity-gated, NOT bit-exact.
constant uint NSG_FC [[function_constant(3)]];
constant uint NSG = is_function_constant_defined(NSG_FC) ? NSG_FC : 1u;

kernel void gemv_q4ks_id_down(
        device const float4 *a       [[buffer(0)]],  // silu_all [top_k * mi] (per-slot segments)
        device const uint4  *codes   [[buffer(1)]],  // ALL experts' down codes
        device       float  *c       [[buffer(2)]],  // mlp_down [h]
        constant     IdDims &d       [[buffer(3)]],
        device const uint   *scales  [[buffer(4)]],
        device const uint   *ids     [[buffer(5)]],  // [top_k] physical expert per slot
        device const float  *wts     [[buffer(6)]],  // [top_k] router weight per slot
        device const uint   *mins    [[buffer(7)]],
        device const half2  *dd      [[buffer(8)]],
        uint  tgpig                  [[threadgroup_position_in_grid]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]]) {
    const uint subs = d.k / 32u;          // sub-blocks per weight row (k = mi)
    const uint slot_vecs = d.k / 4u;      // float4s per slot's silu segment (mi/4)
    const uint h_j = tgpig;               // output hidden index (one simdgroup per h_j)
    if (h_j >= d.n) return;

    float acc = 0.0f;                     // Σ_slot wts[slot] · (this lane's strided dot for slot)
    // NSG=1: simdgroup 0 walks all slots (byte-identical). NSG=2: each simdgroup takes a
    // disjoint slot stride; the partials are combined via threadgroup memory before the write.
    for (uint slot = sgitg; slot < d.top_k; slot += NSG) {
        const uint expert = ids[slot];
        const float rw = wts[slot];
        const uint wcol = expert * d.n + h_j;     // routed expert's down-weight column for h_j
        const uint sbase = slot * slot_vecs;      // this slot's silu segment (float4 units)
        float slot_acc = 0.0f;
        for (uint b = tiisg; b < subs; b += 32u) {
            const uint sblk = b / 8u;
            const uint byte_b = b & 3u;
            const uint base = wcol * subs;
            const uint ddbase = wcol * (subs / 8u);
            const float2 dp = float2(dd[ddbase + sblk]);
            const float dv  = dp.x;
            const float dmv = dp.y;
            const uint sword = scales[(base + b) >> 2];
            const uint mword = mins[(base + b) >> 2];
            const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
            const int  mraw = int(mword << (24u - 8u * byte_b)) >> 24;
            const float mi8 = float(mraw);
            const float s  = dv * su8;
            const float lo = dmv * mi8;

            const uint4 qv = codes[base + b];
            const uint abase = sbase + 8u * b;
            float4 av[8];
            for (short i = 0; i < 8; ++i) av[i] = a[abase + i];
            float a_sum = 0.0f;
            for (short i = 0; i < 8; ++i) a_sum += av[i].x + av[i].y + av[i].z + av[i].w;
            // Late-scale fold (see gemv_q4ks_id): CORR pre-scale + masked-in-place nibbles,
            // bit-for-bit equal to the shifted unpack, ~20 fewer shift ops per sub-block.
            float4 avc[8];
            {
                const float4 CORR = float4(1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f);
                for (short i = 0; i < 8; ++i) avc[i] = av[i] * CORR;
            }

            float code_sum = 0.0f;
            const uint4 w = qv;
            const uint hx = w.x >> 16, hy = w.y >> 16, hz = w.z >> 16, hw = w.w >> 16;
            code_sum += dot(avc[0], float4(float(w.x & 0xFu), float(w.x & 0xF0u), float(w.x & 0xF00u), float(w.x & 0xF000u)))
                      + dot(avc[1], float4(float(hx & 0xFu), float(hx & 0xF0u), float(hx & 0xF00u), float(hx & 0xF000u)));
            code_sum += dot(avc[2], float4(float(w.y & 0xFu), float(w.y & 0xF0u), float(w.y & 0xF00u), float(w.y & 0xF000u)))
                      + dot(avc[3], float4(float(hy & 0xFu), float(hy & 0xF0u), float(hy & 0xF00u), float(hy & 0xF000u)));
            code_sum += dot(avc[4], float4(float(w.z & 0xFu), float(w.z & 0xF0u), float(w.z & 0xF00u), float(w.z & 0xF000u)))
                      + dot(avc[5], float4(float(hz & 0xFu), float(hz & 0xF0u), float(hz & 0xF00u), float(hz & 0xF000u)));
            code_sum += dot(avc[6], float4(float(w.w & 0xFu), float(w.w & 0xF0u), float(w.w & 0xF00u), float(w.w & 0xF000u)))
                      + dot(avc[7], float4(float(hw & 0xFu), float(hw & 0xF0u), float(hw & 0xF00u), float(hw & 0xF000u)));

            slot_acc += s * code_sum + lo * a_sum;
        }
        acc += rw * slot_acc;             // router-weight this slot's partial
    }

    if (NSG == 1u) {
        const float total = simd_sum(acc);
        if (tiisg == 0u) c[h_j] = total;
    } else {
        // Each simdgroup holds a partial over its disjoint slot stride; simdgroup 0 combines.
        threadgroup float sgpart[8];      // up to 8 simdgroups
        const float part = simd_sum(acc);
        if (tiisg == 0u) sgpart[sgitg] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0u && tiisg == 0u) {
            float total = 0.0f;
            for (uint g = 0u; g < NSG; ++g) total += sgpart[g];
            c[h_j] = total;
        }
    }
}
