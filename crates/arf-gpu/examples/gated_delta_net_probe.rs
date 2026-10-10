//! M3 GATE: standalone parity self-test of the `gated_delta_net` MSL recurrence kernel (the
//! Qwen3.6-27B gated-delta-net AUTOREGRESSIVE 1-token decode step) + the three prologue helpers
//! (`gdn_l2_norm`, `gdn_softplus`, `gdn_sigmoid`). NO MODEL — synthetic, deterministic buffers,
//! runs in milliseconds.
//!
//! It compares the GPU kernel against a CPU oracle of the EXACT ggml delta rule
//! (delta-net-base.cpp build_delta_net_autoregressive, lines 319-370):
//!   q *= 1/sqrt(S_k); S *= exp(g); sk[m]=Σ_i S[i][m]*k[i]; d[m]=(v[m]-sk[m])*beta;
//!   S[i][j] += k[i]*d[j];  o[j]=Σ_i S[i][j]*q[i].
//! and runs it for 3 SEQUENTIAL tokens to verify the recurrent STATE carries correctly.
//!
//! Run: cargo run --release -p arf-gpu --example gated_delta_net_probe

#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;

    // A SMALL but representative geometry (the beast is S_v=128, num_v_heads=48; we use a few
    // heads + the real S_v to keep the [128]-length f32 dots representative while staying tiny).
    let s_v: usize = 128; // == S_k for delta-net
    let s_k: usize = s_v;
    let n_heads: usize = 4; // a few v-heads (real model has 48; per-head independent)
    let n_tokens: usize = 3; // sequential tokens to exercise the carried recurrence

    // ---- deterministic synthetic generators (small ramps so f32 reductions stay well-behaved) --
    let gen = |salt: usize, len: usize, lo: f32, span: f32| -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = (((i
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503)))
                    >> 8)
                    & 0xffff) as f32
                    / 65535.0;
                lo + span * t
            })
            .collect()
    };

    // ---- CPU ORACLE helpers (EXACT ggml math) ----
    // l2_norm per head over dim: y = x / fmax(sqrt(Σ x^2), eps)  (no gain).
    let cpu_l2_norm = |x: &[f32], dim: usize, n_h: usize, eps: f32| -> Vec<f32> {
        let mut y = x.to_vec();
        for h in 0..n_h {
            let base = h * dim;
            let mut sum = 0.0f32;
            for i in 0..dim {
                let xi = x[base + i];
                sum += xi * xi;
            }
            let scale = 1.0 / sum.sqrt().max(eps);
            for i in 0..dim {
                y[base + i] = x[base + i] * scale;
            }
        }
        y
    };
    let cpu_softplus = |v: f32| -> f32 {
        if v > 20.0 {
            v
        } else {
            (1.0 + v.exp()).ln()
        }
    };
    let cpu_sigmoid = |v: f32| -> f32 { 1.0 / (1.0 + (-v).exp()) };

    // ---- CPU ORACLE: the autoregressive delta rule for one token (updates S in place) ----
    // Layout S[i + j*s_v]: i = row (ne0), j = col (ne1). q,k,v are per-head [s_v]. g,beta scalar.
    let scale = 1.0f32 / (s_k as f32).sqrt();
    let cpu_step = |state: &mut [f32],
                    o: &mut [f32],
                    q: &[f32],
                    k: &[f32],
                    v: &[f32],
                    g: &[f32],
                    beta: &[f32]| {
        for h in 0..n_heads {
            let sbase = h * s_v * s_v;
            let qkbase = h * s_k;
            let vbase = h * s_v;
            let gexp = g[h].exp();
            let beta_h = beta[h];
            // (1) decay: S[i][j] *= gexp
            for x in &mut state[sbase..sbase + s_v * s_v] {
                *x *= gexp;
            }
            // (2) sk[m] = Σ_i S[i][m]*k[i] ; d[m] = (v[m]-sk[m])*beta
            let mut d = vec![0.0f32; s_v];
            for m in 0..s_v {
                let col0 = sbase + m * s_v;
                let mut sk = 0.0f32;
                for i in 0..s_v {
                    sk += state[col0 + i] * k[qkbase + i];
                }
                d[m] = (v[vbase + m] - sk) * beta_h;
            }
            // (3) S[i][j] += k[i]*d[j] ; (4) o[j] = Σ_i S[i][j]*(q[i]*scale)
            for j in 0..s_v {
                let col0 = sbase + j * s_v;
                let d_j = d[j];
                let mut o_j = 0.0f32;
                for i in 0..s_v {
                    let s_new = state[col0 + i] + k[qkbase + i] * d_j;
                    state[col0 + i] = s_new;
                    o_j += s_new * (q[qkbase + i] * scale);
                }
                o[vbase + j] = o_j;
            }
        }
    };

    let ctx = GpuContext::new().expect("gpu context");
    let mut island = MetalIsland::new(&ctx).expect("metal island (need a Metal backend)");

    // ============================================================================
    // PART A — the three prologue helpers (l2_norm / softplus / sigmoid).
    // ============================================================================
    let eps = 1e-6f32;
    let mut helper_ok = true;
    let mut helper_max = 0.0f32;

    // l2_norm: per-head over head_k_dim (use s_k as dim, n_heads heads).
    {
        let x = gen(101, s_k * n_heads, -0.8, 1.6);
        let cpu = cpu_l2_norm(&x, s_k, n_heads, eps);
        let gpu = island
            .gdn_prologue_selftest(&ctx, "l2_norm", &x, s_k, n_heads, eps)
            .expect("l2_norm dispatch");
        let m = max_abs(&gpu, &cpu);
        helper_max = helper_max.max(m);
        helper_ok &= m < 1e-5;
        println!("  gdn_l2_norm  (dim {s_k} x {n_heads} heads) max_abs = {m:.3e}");
    }
    // softplus over [num_v_heads]-ish vector.
    {
        let x = gen(202, n_heads * 6, -5.0, 30.0); // span into the >20 branch too
        let cpu: Vec<f32> = x.iter().map(|&v| cpu_softplus(v)).collect();
        let gpu = island
            .gdn_prologue_selftest(&ctx, "softplus", &x, x.len(), 1, 0.0)
            .expect("softplus dispatch");
        let m = max_abs(&gpu, &cpu);
        helper_max = helper_max.max(m);
        helper_ok &= m < 1e-5;
        println!("  gdn_softplus (n {}) max_abs = {m:.3e}", x.len());
    }
    // sigmoid over [num_v_heads]-ish vector.
    {
        let x = gen(303, n_heads * 6, -12.0, 24.0);
        let cpu: Vec<f32> = x.iter().map(|&v| cpu_sigmoid(v)).collect();
        let gpu = island
            .gdn_prologue_selftest(&ctx, "sigmoid", &x, x.len(), 1, 0.0)
            .expect("sigmoid dispatch");
        let m = max_abs(&gpu, &cpu);
        helper_max = helper_max.max(m);
        helper_ok &= m < 1e-5;
        println!("  gdn_sigmoid  (n {}) max_abs = {m:.3e}", x.len());
    }

    // ============================================================================
    // PART B — the recurrence kernel, run for `n_tokens` SEQUENTIAL tokens to verify
    // the carried state. CPU oracle and GPU advance their OWN copy of S in lockstep.
    // ============================================================================
    println!("\n=== gated_delta_net M3 PARITY (synthetic, no model) ===");
    println!("  geometry: S_v = {s_v}, S_k = {s_k}, num_v_heads = {n_heads}, tokens = {n_tokens}");

    // Shared starting state (small so the f32 dots are representative, not blown up).
    let mut cpu_state = gen(7, s_v * s_v * n_heads, -0.05, 0.10);
    let mut gpu_state = cpu_state.clone();

    let mut worst_out = 0.0f32;
    let mut worst_state = 0.0f32;
    let mut last_out_gpu = vec![0.0f32; s_v * n_heads];
    let mut last_out_cpu = vec![0.0f32; s_v * n_heads];

    for t in 0..n_tokens {
        // Per-token deterministic q/k/v/g/beta. Note: in the real model q/k are L2-normed +
        // repeated and g/beta come from softplus*ssm_a / sigmoid; here we feed representative
        // values directly (the recurrence kernel takes them post-prologue).
        let q = gen(1000 + t, s_k * n_heads, -0.6, 1.2);
        let k = gen(2000 + t, s_k * n_heads, -0.6, 1.2);
        let v = gen(3000 + t, s_v * n_heads, -0.5, 1.0);
        // g = the GDA gate (negative-ish so exp(g) < 1, a decay); beta in (0,1).
        let g = gen(4000 + t, n_heads, -0.5, 0.4);
        let beta = gen(5000 + t, n_heads, 0.1, 0.8);

        // CPU oracle step.
        let mut cpu_o = vec![0.0f32; s_v * n_heads];
        cpu_step(&mut cpu_state, &mut cpu_o, &q, &k, &v, &g, &beta);

        // GPU step (advances gpu_state in place).
        let (gpu_o, new_gpu_state) = island
            .gated_delta_net_selftest(&ctx, &q, &k, &v, &g, &beta, &gpu_state, s_v, n_heads)
            .expect("gated_delta_net dispatch");
        gpu_state = new_gpu_state;

        let m_out = max_abs(&gpu_o, &cpu_o);
        let m_state = max_abs(&gpu_state, &cpu_state);
        worst_out = worst_out.max(m_out);
        worst_state = worst_state.max(m_state);
        println!("  token {t}: o max_abs = {m_out:.3e}   carried-state max_abs = {m_state:.3e}");
        last_out_gpu = gpu_o;
        last_out_cpu = cpu_o;
    }

    // Sample value sanity at the worst output index of the last token.
    let (am, _) = argmax_abs(&last_out_gpu, &last_out_cpu);
    println!(
        "  last-token sample @idx {am}: gpu {:.6} vs cpu {:.6}",
        last_out_gpu[am], last_out_cpu[am]
    );
    println!("  WORST over all tokens: o = {worst_out:.3e}, carried-state = {worst_state:.3e}");

    // ---- GATES ----
    // f32 reductions over [128]-length dots drift slightly more than the conv's; 1e-4 is the bar.
    let tol = 1e-4f32;
    assert!(helper_ok, "prologue helper parity {helper_max:.3e} >= 1e-5");
    assert!(
        worst_out < tol,
        "recurrence o parity {worst_out:.3e} >= {tol:.0e}"
    );
    assert!(
        worst_state < tol,
        "recurrence carried-state parity {worst_state:.3e} >= {tol:.0e}"
    );

    println!(
        "\nM3 GATE PASS: gated_delta_net GPU == CPU oracle (readout + carried state over {n_tokens} tokens), tol {tol:.0e}; prologue helpers tol 1e-5."
    );
}

#[cfg(target_os = "macos")]
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(target_os = "macos")]
fn argmax_abs(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut idx = 0usize;
    let mut m = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > m {
            m = d;
            idx = i;
        }
    }
    (idx, m)
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gated_delta_net_probe: the MSL gated_delta_net kernel is macOS-only (Metal island). Skipped.");
}
