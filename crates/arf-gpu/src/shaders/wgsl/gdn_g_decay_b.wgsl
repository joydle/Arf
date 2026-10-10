// WGSL fused g_decay — B-PARALLEL twin of gdn_g_decay.wgsl. g_out[b,h] = softplus(alpha[b,h] +
// ssm_dt[h]) * ssm_a[h] over B*n_per_seq elements. alpha/g_out are per-seq [B*n]; ssm_dt/ssm_a are
// SHARED static [n] → indexed by (i % n). b=0 slice ≡ single-stream. dims.n = n_per_seq (NOT B*n);
// dims.batch carries B so the grid covers B*n and the shared-weight wrap is i%n.
//
// Bindings: 0 g_out[B*n] (WRITE), 1 alpha[B*n] (READ), 2 ssm_dt[n] (SHARED, READ),
//           3 ssm_a[n] (SHARED, READ), 4 dims {n_per_seq, batch, _, _}.

struct GDecayDimsB { n: u32, batch: u32, _p1: u32, _p2: u32, };

@group(0) @binding(0) var<storage, read_write> g_out: array<f32>;
@group(0) @binding(1) var<storage, read>       alpha: array<f32>;
@group(0) @binding(2) var<storage, read>       ssm_dt: array<f32>;
@group(0) @binding(3) var<storage, read>       ssm_a: array<f32>;
@group(0) @binding(4) var<uniform>             d: GDecayDimsB;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = d.batch * d.n;
    let i = gid.x;
    if (i >= total) { return; }
    let h = i % d.n;                     // shared-weight index (ssm_dt/ssm_a are [n], reused per b)
    let v = alpha[i] + ssm_dt[h];
    var sp: f32;
    if (v > 20.0) { sp = v; } else { sp = log(1.0 + exp(v)); }
    g_out[i] = sp * ssm_a[h];
}
