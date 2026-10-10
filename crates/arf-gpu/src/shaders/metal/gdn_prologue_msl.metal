// MSL prologue helpers for the Qwen3.6-27B gated-delta-net (M3). Three tiny elementwise /
// per-head kernels that feed the delta-net recurrence, each a bit-exact port of the ggml op:
//
//   l2_norm  — ggml_l2_norm (ggml-cpu/ops.cpp ggml_compute_forward_l2_norm_f32, lines 4137-).
//              Per (head): y = x / fmax(sqrt(sum_i x_i^2), eps).  NOTE: max(sqrt,eps), NOT
//              sqrt(sum+eps), and NO learned gain (distinct from rmsnorm). Used on q_conv/k_conv
//              per head over head_k_dim.
//   softplus — ggml op_softplus (unary-ops.cpp:80): (x > 20) ? x : log(1+exp(x)). Used on
//              alpha = softplus(ssm_alpha@cur + ssm_dt) over [num_v_heads].
//   sigmoid  — ggml op_sigmoid (unary-ops.cpp:31): 1/(1+exp(-x)). Used on beta over [num_v_heads].
//
// All three are standalone (folding them into the recurrence kernel is an M6 perf option). They
// run on tiny vectors (num_v_heads = 48) or per-head (head_k_dim = 128 x num_*_heads). Parity is
// gated in examples/gated_delta_net_probe.rs against a CPU oracle of the SAME ggml math.

#include <metal_stdlib>
using namespace metal;

// ---- l2_norm: per-head L2 normalize over `dim` (head_k_dim) ----------------------------------
// Layout: x is [dim, n_heads] row-major-per-head (head h's vector at x[h*dim + 0..dim]). One
// threadgroup per head, `dim` threads cooperate (parallel sum-of-squares via threadgroup reduce),
// then each thread scales its element. Bit-exact to the ggml serial reduction is NOT guaranteed
// for a tree reduction, so we do the reduction SERIALLY in thread 0 (dim<=256, cheap) to match
// ggml's left-to-right f32 accumulation order. Bindings: 0 x[dim*n_heads] (READ+WRITE in place),
// 1 dims {dim, n_heads, eps_bits, _}. eps is passed as its f32 bit pattern in dims[2].
// L174 — `_p` becomes `seq_stride`: the per-sequence stride of the buffer this normalizes IN
// PLACE (conv_out, which is [n_seqs, conv_dim]). 0 == single stream == pre-L174 behaviour.
struct L2Dims { uint dim; uint n_heads; uint eps_bits; uint seq_stride; };

kernel void gdn_l2_norm(
        device       float *x   [[buffer(0)]],
        constant L2Dims    &d   [[buffer(1)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]]) {
    // L174 — SEQUENCE AXIS. This kernel is dispatched TWICE at BYTE OFFSETS (q half at 0, k half
    // at key_dim*4) and the host keeps those offsets; the sequence stride is applied here, so the
    // two compose. Already threadgroup-per-head, so no index reconstruction is needed — the
    // L172/L173 width trap does not apply to this one.
    const uint t    = uint(tpitg.x);
    const uint head = tgpig.x;
    const uint seq  = tgpig.y;
    if (head >= d.n_heads) return;
    const uint dim  = d.dim;
    const uint base = seq * d.seq_stride + head * dim;
    const float eps = as_type<float>(d.eps_bits);

    threadgroup float scale_sh;
    if (t == 0u) {
        // Serial f32 accumulation (matches ggml's ggml_float sum loop order; ggml_float is double
        // on CPU but the magnitudes here keep f32 within tol — see probe).
        float sum = 0.0f;
        for (uint i = 0u; i < dim; ++i) {
            float xi = x[base + i];
            sum += xi * xi;
        }
        scale_sh = 1.0f / fmax(sqrt(sum), eps);   // ggml: 1/fmaxf(sqrtf(sum), eps)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < dim) {
        x[base + t] = x[base + t] * scale_sh;
    }
}

// ---- softplus: in place over n elements ------------------------------------------------------
// Bindings: 0 x[n] (READ+WRITE), 1 dims {n,_,_,_}.
struct Elem1 { uint n; uint _p0; uint _p1; uint _p2; };

kernel void gdn_softplus(
        device       float *x  [[buffer(0)]],
        constant Elem1     &d  [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    const float v = x[gid];
    x[gid] = (v > 20.0f) ? v : log(1.0f + exp(v));   // ggml op_softplus
}

// ---- sigmoid: in place over n elements -------------------------------------------------------
// Bindings: 0 x[n] (READ+WRITE), 1 dims {n,_,_,_}.
kernel void gdn_sigmoid(
        device       float *x  [[buffer(0)]],
        constant Elem1     &d  [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    x[gid] = 1.0f / (1.0f + exp(-x[gid]));            // ggml op_sigmoid
}

// ---- gdn_sigmoid_out: out[i] = sigmoid(src[i]) over n elements (NON in-place) ----------------
// Same math as gdn_sigmoid but reads a separate source so the resident `beta_raw` (a shared
// buffer the wgpu projection wrote) is NOT mutated. Bindings: 0 out[n] (WRITE), 1 src[n] (READ),
// 2 dims {n,_,_,_}.
kernel void gdn_sigmoid_out(
        device       float *out [[buffer(0)]],
        device const float *src [[buffer(1)]],
        constant Elem1     &d   [[buffer(2)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    // L173 — SEQUENCE AXIS, using ggml's exact attribute set (ggml-metal.metal:1120-1126).
    //
    // Two constraints learned the hard way:
    //   L171 — a kernel may NOT declare both [[thread_position_in_grid]] and
    //          [[threadgroup_position_in_grid]]; it compiles, runs, and silently computes wrong
    //          values.
    //   L172 — reconstructing the linear index as `tgpig.x * 256 + lid` ALSO fails. The literal
    //          256 is the bug: Metal may clamp a pipeline's threadgroup width below what the host
    //          requested (maxTotalThreadsPerThreadgroup is per-PSO), so the real width must be
    //          READ, not assumed. ggml reads it via [[threads_per_threadgroup]] for exactly this
    //          reason.
    //
    // tgpig.y is the sequence; a 1-high grid gives 0 == pre-L173 behaviour.
    const uint gid = tgpig.x * uint(ntg.x) + uint(tpitg.x);
    const uint seq = tgpig.y;
    if (gid >= d.n) return;
    const uint i = seq * d.n + gid;
    out[i] = 1.0f / (1.0f + exp(-src[i]));            // ggml op_sigmoid
}

// ---- gdn_g_decay: g_decay[h] = softplus(alpha[h] + ssm_dt[h]) * ssm_a[h] over n=num_v_heads ---
// FUSED port of the host scalar the M10 path used to compute between the readback and the
// recurrence: `g_decay = softplus(ssm_alpha@cur + ssm_dt) * ssm_a`. ssm_alpha@cur is the
// `alpha` projection (a GEMV output, per v-head); ssm_dt + ssm_a are the static per-layer
// vectors. softplus = ggml op_softplus (>20 ? x : log(1+exp(x))) — bit-identical to the host
// loop it replaces. Computing this on the GPU lets the alpha projection stay resident (never
// read back to host), deleting the per-linear-layer Pass-A host round-trip.
// Bindings: 0 g_out[n] (WRITE), 1 alpha[n] (READ), 2 ssm_dt[n] (READ), 3 ssm_a[n] (READ),
//           4 dims {n,_,_,_}.
kernel void gdn_g_decay(
        device       float *g_out  [[buffer(0)]],
        device const float *alpha  [[buffer(1)]],
        device const float *ssm_dt [[buffer(2)]],
        device const float *ssm_a  [[buffer(3)]],
        constant Elem1     &d      [[buffer(4)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    // L173 — see gdn_sigmoid_out for the two constraints. g_out/alpha are PER-SEQUENCE
    // [n_seqs, n]; ssm_dt/ssm_a are STATIC per-layer vectors and MUST stay unoffset.
    const uint gid = tgpig.x * uint(ntg.x) + uint(tpitg.x);
    const uint seq = tgpig.y;
    if (gid >= d.n) return;
    const uint i = seq * d.n + gid;
    const float v  = alpha[i] + ssm_dt[gid];
    const float sp = (v > 20.0f) ? v : log(1.0f + exp(v));   // ggml op_softplus
    g_out[i] = sp * ssm_a[gid];
}

// ---- gdn_tile: split conv_out into q/k/v + TILE q,k across v-head groups (on the GPU) --------
// Replaces the HOST split+repeat that sat between the l2_norm command buffer and the recurrence
// command buffer (the seam that forced TWO island round-trips per linear layer). With this on
// the GPU, conv -> l2 -> tile -> recurrence all chain in ONE command buffer (ONE waitUntilCompleted).
//
// conv_out layout: [ q_conv(key_dim) | k_conv(key_dim) | v_conv(value_dim) ], key_dim = s*nkh,
// value_dim = s*nvh. The recurrence wants q,k repeated to [s*nvh]: v-head h reads k/q head
// `h % nkh` (ggml_repeat_4d TILING). v passes through unchanged.
//
// One thread per output index t in 0..s*nvh (== value_dim, since value_dim = s*nvh):
//   h = t / s ; i = t % s ; kh = h % nkh ; src = kh*s + i
//   q_rep[t] = conv_out[src] ; k_rep[t] = conv_out[key_dim + src] ; v[t] = conv_out[2*key_dim + t]
// Bindings: 0 conv_out[conv_dim] (READ), 1 q_rep[s*nvh] (WRITE), 2 k_rep[s*nvh] (WRITE),
//           3 v[value_dim] (WRITE), 4 dims {s, nkh, nvh, key_dim} u32x4.
struct TileDims { uint s; uint nkh; uint nvh; uint key_dim; };

kernel void gdn_tile(
        device const float *conv_out [[buffer(0)]],
        device       float *q_rep    [[buffer(1)]],
        device       float *k_rep    [[buffer(2)]],
        device       float *v_out    [[buffer(3)]],
        constant TileDims  &d        [[buffer(4)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    // L174 — SEQUENCE AXIS (ggml attribute set; see gdn_sigmoid_out for the three constraints).
    // conv_out is [n_seqs, conv_dim]; q_rep/k_rep/v_out are [n_seqs, s*nvh].
    const uint gid = tgpig.x * uint(ntg.x) + uint(tpitg.x);
    const uint seq = tgpig.y;
    const uint n = d.s * d.nvh;               // == value_dim, PER SEQUENCE
    if (gid >= n) return;
    const uint h   = gid / d.s;
    const uint i   = gid % d.s;
    const uint kh  = h % d.nkh;
    const uint conv_dim = 2u * d.key_dim + n;
    const uint cbase = seq * conv_dim;        // this seq's conv_out slab
    const uint obase = seq * n;               // this seq's q/k/v slab
    const uint src = kh * d.s + i;
    q_rep[obase + gid] = conv_out[cbase + src];
    k_rep[obase + gid] = conv_out[cbase + d.key_dim + src];
    v_out[obase + gid] = conv_out[cbase + 2u * d.key_dim + gid];
}
