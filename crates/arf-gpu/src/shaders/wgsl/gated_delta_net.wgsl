// WGSL gated-delta-net 1-token recurrence (M11 single-queue port). BIT-EXACT twin of
// gated_delta_net_msl.metal (port of llama.cpp build_delta_net_autoregressive, n_tokens==1).
// Lets the recurrence record onto the wgpu CommandPass — the whole linear layer on ONE queue,
// no Metal island, no cross-queue wait. Same scalar per-column serial f32 accumulation, same
// threadgroup q/k copies, same single barrier — Paris must stay identical.
//
// ONE WORKGROUP per v-head (nvh groups). S_v threads/group (== S_k == 128 for the beast), thread
// j owns COLUMN j of the [S_v x S_v] state S (flat S[i + j*S_v]).
//
// Per-element math (ggml ops):
//   q = q * (1/sqrt(S_k)) ; gexp = exp(g_h)
//   S[i][j] *= gexp ; sk_j = sum_i S[i][j]*k[i] ; d_j = (v[j]-sk_j)*beta_h
//   S[i][j] += k[i]*d_j ; o[j] = sum_i S[i][j]*q[i]   (post-update)
//
// Bindings: 0 q[S_k*nvh] READ, 1 k[S_k*nvh] READ, 2 v[S_v*nvh] READ, 3 g[nvh] READ (pre-exp),
//           4 beta[nvh] READ, 5 S[S_v*S_v*nvh] READ+WRITE, 6 o[S_v*nvh] WRITE,
//           7 dims {S_v, S_k, nvh, _}.
// GRID: workgroups = nvh, threads/wg = S_v.

struct GdnDims { s_v: u32, s_k: u32, n_vheads: u32, _p: u32, };

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read>       g: array<f32>;
@group(0) @binding(4) var<storage, read>       beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> S: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;
@group(0) @binding(7) var<uniform>             d: GdnDims;

const GDN_MAX_S: u32 = 256u;   // S_v == S_k == 128 for the beast; workgroup array cap.

var<workgroup> ksh: array<f32, 256>;   // GDN_MAX_S
var<workgroup> qsh: array<f32, 256>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let S_v = d.s_v;
    let S_k = d.s_k;          // == S_v for delta-net
    let head = wg.x;
    let j = lid.x;
    if (head >= d.n_vheads) { return; }

    let qkbase = head * S_k;          // q,k row base
    let vbase = head * S_v;           // v / o row base
    let sbase = head * S_v * S_v;     // this head's [S_v x S_v] state base

    let scale = 1.0 / sqrt(f32(S_k));

    // (a) cooperative load: thread j fills slot j of k,q (q pre-scaled here).
    if (j < S_k) {
        ksh[j] = k[qkbase + j];
        qsh[j] = q[qkbase + j] * scale;     // ggml_scale(q, 1/sqrt(S_k)) folded in
    }
    workgroupBarrier();

    if (j >= S_v) { return; }   // none for the beast (S_v == threads)

    let gexp = exp(g[head]);
    let beta_h = beta[head];

    // thread j owns COLUMN j: S[i][j] at flat index sbase + i + j*S_v.
    let col0 = sbase + j * S_v;       // S[0][j]

    // decay + sk_j = sum_i (S[i][j]*gexp) * k[i]   (decay folded into the read).
    var sk_j: f32 = 0.0;
    for (var i: u32 = 0u; i < S_v; i = i + 1u) {
        let s_dec = S[col0 + i] * gexp;    // S[i][j] after decay
        S[col0 + i] = s_dec;               // write decayed state back
        sk_j = sk_j + s_dec * ksh[i];
    }
    // d_j = (v[j] - sk_j) * beta_h
    let d_j = (v[vbase + j] - sk_j) * beta_h;

    // S[i][j] += k[i]*d_j ; o[j] = sum_i S[i][j]*q[i]  (post-update).
    var o_j: f32 = 0.0;
    for (var i: u32 = 0u; i < S_v; i = i + 1u) {
        let s_new = S[col0 + i] + ksh[i] * d_j;
        S[col0 + i] = s_new;
        o_j = o_j + s_new * qsh[i];
    }
    o[vbase + j] = o_j;
}
