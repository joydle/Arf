// WGSL gated-delta-net 1-token recurrence — B-PARALLEL (concurrency) twin of gated_delta_net.wgsl.
// Decodes B sequences in lockstep: each seq carries its OWN [S_v x S_v x nvh] state, the static
// weights (none here — q/k/v/g/beta are per-seq activations) are shared by the batched GEMM that
// produced them. The O(1)-per-token state is the advantage: it grows with B but NOT with context length.
//
// LAYOUT (b-outer, per the B1 contract): for batch b, head h, the state column j lives at
//   S[(b*nvh + h)*S_v*S_v + j*S_v + i].   q/k feeds at [(b*nvh+h)*S_k + ...], v/o at [(b*nvh+h)*S_v].
// b=0 slice is byte-identical to the single-stream kernel → "B=1 ≡ M11" parity gate.
//
// ONE WORKGROUP per (b, v-head): B*nvh groups. wg.x in [0, B*nvh): b = wg.x / nvh, head = wg.x % nvh.
// Distinct (b,head) groups touch disjoint state → fully independent, no cross-b hazard.
//
// Per-element math identical to the single-stream twin (same scalar serial f32 accumulation,
// same threadgroup q/k copies, same single barrier):
//   q = q*(1/sqrt(S_k)); gexp = exp(g_h); S[i][j]*=gexp; sk_j=sum_i S[i][j]*k[i];
//   d_j=(v[j]-sk_j)*beta_h; S[i][j]+=k[i]*d_j; o[j]=sum_i S[i][j]*q[i] (post-update).
//
// Bindings: 0 q[B*S_k*nvh] READ, 1 k[B*S_k*nvh] READ, 2 v[B*S_v*nvh] READ, 3 g[B*nvh] READ,
//           4 beta[B*nvh] READ, 5 S[B*S_v*S_v*nvh] READ+WRITE, 6 o[B*S_v*nvh] WRITE,
//           7 dims {S_v, S_k, nvh, B}.
// GRID: workgroups = B*nvh, threads/wg = S_v.

struct GdnDimsB { s_v: u32, s_k: u32, n_vheads: u32, batch: u32, };

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read>       g: array<f32>;
@group(0) @binding(4) var<storage, read>       beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> S: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;
@group(0) @binding(7) var<uniform>             d: GdnDimsB;

var<workgroup> ksh: array<f32, 256>;   // GDN_MAX_S (S_v == S_k == 128 for the beast)
var<workgroup> qsh: array<f32, 256>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let S_v = d.s_v;
    let S_k = d.s_k;          // == S_v for delta-net
    let nvh = d.n_vheads;
    let lane = wg.x;
    if (lane >= d.batch * nvh) { return; }
    let b = lane / nvh;        // which sequence
    let head = lane % nvh;     // which v-head within that sequence
    let j = lid.x;

    // b-outer strides: seq b's feeds/state start at b * (per-seq span).
    let qkbase = (b * nvh + head) * S_k;          // q,k row base for (b,head)
    let vbase = (b * nvh + head) * S_v;           // v / o row base
    let sbase = (b * nvh + head) * S_v * S_v;     // this (b,head)'s [S_v x S_v] state base

    let scale = 1.0 / sqrt(f32(S_k));

    // (a) cooperative load: thread j fills slot j of k,q (q pre-scaled here).
    if (j < S_k) {
        ksh[j] = k[qkbase + j];
        qsh[j] = q[qkbase + j] * scale;
    }
    workgroupBarrier();

    if (j >= S_v) { return; }

    let gexp = exp(g[b * nvh + head]);
    let beta_h = beta[b * nvh + head];

    // thread j owns COLUMN j: S[i][j] at flat index sbase + i + j*S_v.
    let col0 = sbase + j * S_v;       // S[0][j]

    var sk_j: f32 = 0.0;
    for (var i: u32 = 0u; i < S_v; i = i + 1u) {
        let s_dec = S[col0 + i] * gexp;
        S[col0 + i] = s_dec;
        sk_j = sk_j + s_dec * ksh[i];
    }
    let d_j = (v[vbase + j] - sk_j) * beta_h;

    var o_j: f32 = 0.0;
    for (var i: u32 = 0u; i < S_v; i = i + 1u) {
        let s_new = S[col0 + i] + ksh[i] * d_j;
        S[col0 + i] = s_new;
        o_j = o_j + s_new * qsh[i];
    }
    o[vbase + j] = o_j;
}
