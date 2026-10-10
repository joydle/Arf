// INDIRECT (MoE) Q4_K_S decode GEMV — a faithful port of llama.cpp's kernel_mul_mv_id
// (ggml-metal.metal:10369): a thin EXPERT-INDIRECTION wrapper that reuses the EXACT dense
// gemv_q4ks body (matmul_vec_q4ks_msl.metal) with src0 redirected to the routed expert's
// weights. This is the whole MoE-decode secret: NO special slow MoE kernel — the same
// ~roofline dense GEMV, pointed at the expert chosen by the `ids` buffer, fanned out across
// the routed experts via the grid's Z dimension (one concurrent dispatch for all top_k).
//
// Dispatch (host side, mirrors ggml-metal-ops.cpp:2461): grid = (ceil(n/COLS), 1, top_k),
// threadgroup = 32 lanes. Z = the routed-expert SLOT (0..top_k-1) for THIS token (batch=1
// decode). The kernel reads ids[slot] → physical expert i02 → offsets codes/scales/mins/dd by
// i02 * per_expert_stride, runs the dense dot, and writes c[slot*n + col] (routed-slot-major,
// matching the wgpu batch_gu_b output layout so the downstream swiglu/down are unchanged).
//
// Per-expert packing (concatenated over num_experts), IDENTICAL to the WGSL MoE kernels:
//   codes:  uint4 per sub-block; expert e column col base = (e*n + col)*(k/32).
//   scales: u8 per sub-block, 4/word; same base. mins: i8 per sub-block, 4/word; same base.
//   dd:     half2 {d,dmin} per super-block; pair base = (e*n + col)*(k/256).
//
// Bit-exact to the dense gemv_q4ks per (expert, col) — only the e*n weight offset differs, so
// the output matches the wgpu matmul_vec_moe_q4ks_batch_gu_b kernel within the 1e-3 bar.

#include <metal_stdlib>
using namespace metal;

// {m, k(=hidden), n(=inter), top_k}. m=1 for decode (one token).
struct IdDims { uint m; uint k; uint n; uint top_k; };

// ARF_GEMV_ID_NSG2 (function_constant 3 — the dense kernel's NSG index): simdgroups per
// threadgroup. Default 1 (32 threads, the shipped path; the is_function_constant_defined guard
// keeps the plain compile_msl path byte-identical). NSG=2 (64 threads) splits the sub-block
// weight stream across TWO simdgroups on the SAME 2 columns — llama's kernel_mul_mv_id q4_K
// runs N_SG=2 for exactly this reason: one simdgroup can't keep enough Q4_K weight loads in
// flight (this kernel measures ~46-55% of the bandwidth wall); two simdgroups double the
// in-flight requests. At qwen3-coder dims (k=2048 → 64 sub-blocks) each simdgroup does exactly
// one 32-sub-block pass. Below-tie f32 reorder (different sub-block partition per lane) —
// parity-gated, NOT bit-exact. The dot-y late-scale fold is untouched (it is the advantage).
constant uint NSG_FC [[function_constant(3)]];
constant uint NSG = is_function_constant_defined(NSG_FC) ? NSG_FC : 1u;

kernel void gemv_q4ks_id(
        device const float4 *a       [[buffer(0)]],  // activation [k] (this token's normed hidden)
        device const uint4  *codes   [[buffer(1)]],  // ALL experts' codes, concatenated
        device       float  *c       [[buffer(2)]],  // [top_k * n] routed-slot-major output
        constant     IdDims &d       [[buffer(3)]],
        device const uint   *scales  [[buffer(4)]],  // u8 sub-scale, 4/word (all experts)
        device const uint   *ids     [[buffer(5)]],  // [top_k] physical expert id per slot
        device const uint   *mins    [[buffer(6)]],  // i8 sub-min, 4/word (all experts)
        device const half2  *dd      [[buffer(7)]],  // {d,dmin}/super (all experts)
        uint3  tgpig                 [[threadgroup_position_in_grid]],
        ushort tiisg                 [[thread_index_in_simdgroup]],
        ushort sgitg                 [[simdgroup_index_in_threadgroup]]) {
    const uint subs = d.k / 32u;
    const uint COLS = 2u;
    const uint slot = tgpig.z;                 // routed-expert SLOT for this token (grid Z)
    if (slot >= d.top_k) return;
    const uint col0 = tgpig.x * COLS;
    if (col0 >= d.n) return;

    // INDIRECTION: which physical expert did this slot route to? (kernel_mul_mv_id: i02=ids[idx])
    const uint expert = ids[slot];
    // Per-expert weight offset: expert's columns start at (expert*n) in sub-block / super units.
    const uint exp_col_base = expert * d.n;    // first weight column of this expert

    float acc[2] = {0.0f, 0.0f};

    // NSG=1: simdgroup 0 strides all sub-blocks (b += 32, byte-identical). NSG=2: each simdgroup
    // takes a disjoint stride (start sgitg*32 + tiisg, step NSG*32) → 64 lanes, ~2× in-flight loads.
    for (uint b = sgitg * 32u + tiisg; b < subs; b += NSG * 32u) {
        const uint abase = 8u * b;
        float4 av[8];
        for (short i = 0; i < 8; ++i) av[i] = a[abase + i];
        float a_sum = 0.0f;
        for (short i = 0; i < 8; ++i) a_sum += av[i].x + av[i].y + av[i].z + av[i].w;
        // Late-scale fold (the shipped conc16 `_b` lever, ported to the m=1 island): pre-scale
        // the activation ONCE per sub-block by CORR = (1, 1/16, 1/256, 1/4096) — exact powers
        // of two — then dot against nibbles MASKED IN PLACE (nib·16^lane ≤ 61440 < 2^24, exact
        // in fp32). dot(av·CORR, masked) == dot(av, shifted) bit-for-bit, and the CORR multiply
        // amortizes across the COLS=2 expert columns while ~20 shifts/column disappear.
        float4 avc[8];
        {
            const float4 CORR = float4(1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f);
            for (short i = 0; i < 8; ++i) avc[i] = av[i] * CORR;
        }

        const uint sblk = b / 8u;
        const uint byte_b = b & 3u;

        for (uint cc = 0u; cc < COLS; ++cc) {
            const uint col = col0 + cc;
            if (col >= d.n) break;
            const uint wcol = exp_col_base + col;   // ← routed expert's weight column (the redirect)
            const uint idx = wcol * subs + b;
            const uint4 qv = codes[idx];

            const uint sbase = wcol * subs;
            const uint ddbase = wcol * (subs / 8u);
            const float2 dp = float2(dd[ddbase + sblk]);
            const float dv  = dp.x;
            const float dmv = dp.y;
            const uint sword = scales[(sbase + b) >> 2];
            const uint mword = mins[(sbase + b) >> 2];
            const float su8 = float((sword >> (8u * byte_b)) & 0xFFu);
            const int  mraw = int(mword << (24u - 8u * byte_b)) >> 24;
            const float mi8 = float(mraw);
            const float s  = dv * su8;
            const float lo = dmv * mi8;

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

            acc[cc] += s * code_sum + lo * a_sum;
        }
    }

    // Write to this routed slot's output segment: c[slot*n + col].
    const uint obase = slot * d.n;
    if (NSG == 1u) {
        // Single simdgroup: reduce 32 lanes, lane 0 writes (the shipped path, unchanged).
        for (uint cc = 0u; cc < COLS; ++cc) {
            const uint col = col0 + cc;
            const float total = simd_sum(acc[cc]);
            if (tiisg == 0u && col < d.n) c[obase + col] = total;
        }
    } else {
        // NSG simdgroups each hold a partial over their disjoint sub-block stride. simd_sum within
        // each simdgroup → sgpart[sgitg][cc]; then simdgroup 0 sums the NSG partials and writes
        // (the dense gemv_q4ks NSG reduction, verbatim).
        threadgroup float sgpart[8][2];   // up to 8 simdgroups × COLS(=2)
        for (uint cc = 0u; cc < COLS; ++cc) {
            const float part = simd_sum(acc[cc]);
            if (tiisg == 0u) sgpart[sgitg][cc] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0u && tiisg == 0u) {
            for (uint cc = 0u; cc < COLS; ++cc) {
                const uint col = col0 + cc;
                if (col >= d.n) continue;
                float total = 0.0f;
                for (uint g = 0u; g < NSG; ++g) total += sgpart[g][cc];
                c[obase + col] = total;
            }
        }
    }
}
