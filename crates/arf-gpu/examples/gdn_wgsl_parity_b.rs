//! B-PARALLEL (concurrency) parity: prove the SIX B-parallel gated-delta-net WGSL twins decode B
//! independent sequences with ZERO cross-contamination. The gate is structural:
//!
//!   For each kernel, run a B=4 batch with DIFFERENT random inputs per sequence through the `_b`
//!   kernel, then run each sequence's inputs ALONE through the single-stream M11 kernel, and assert
//!   the b-th slice of the batched output == the standalone single-stream output (byte-for-byte for
//!   the elementwise kernels; within f32 tol for the recurrence). This proves (a) the `_b` kernels
//!   are correct and (b) seq b only ever touches its own b-strided state — no batch bleed.
//!
//! The recurrence test carries state across 3 tokens per sequence (the real decode pattern), with
//! each sequence's ssm_state advancing independently in its own b-outer slice.
//!
//! Run: cargo run --release -p arf-gpu --example gdn_wgsl_parity_b

use std::sync::Arc;

use arf_gpu::gpu::{CommandPass, GpuContext};
use arf_gpu::weights::build_kernels;

fn gen(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / (1u64 << 31) as f32;
            lo + (hi - lo) * (u % 1.0)
        })
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        m = m.max((a[i] - b[i]).abs());
    }
    m
}

fn main() {
    // Beast geometry.
    let s: usize = 128;
    let nkh: usize = 16;
    let nvh: usize = 48;
    let key_dim = s * nkh; // 2048
    let value_dim = s * nvh; // 6144
    let d_conv: usize = 4;
    let conv_dim = key_dim * 2 + value_dim; // 10240
    let eps = 1e-6f32;
    let batch: usize = 4;

    let ctx = Arc::new(GpuContext::new().expect("gpu context"));
    let k = build_kernels(&ctx, false);

    let mut all_ok = true;
    let mut report = |name: &str, m: f32, tol: f32| {
        let ok = m < tol;
        all_ok &= ok;
        println!(
            "  {name:<26} max_abs(batched_b vs single) = {m:.3e}  tol {tol:.0e}  {}",
            if ok { "PASS" } else { "FAIL" }
        );
    };

    // Helper: run the SINGLE-STREAM M11 kernel for one sequence and return its output, so the
    // B-parallel slice can be compared to "the same seq decoded alone". (Reuses the production
    // single-stream kernels, the M11 Paris oracle.)

    // ---- (1) gdn_sigmoid_out_b: out[B*n] = sigmoid(src), per-seq independent ----
    {
        let n = nvh;
        let src_all = gen(303, batch * n, -12.0, 24.0); // b-outer: seq b at [b*n..]
        let out_b = ctx.storage_init("pb.sig.out", &vec![0.0f32; batch * n]);
        let src_b = ctx.storage_init("pb.sig.src", &src_all);
        let dims = ctx.uniform_of("pb.sig.d", &[(batch * n) as u32, 0u32, 0, 0]);
        let mut cp = CommandPass::new(&ctx);
        cp.add_bound(
            &k.gdn_sigmoid_out_b,
            "sig_b",
            &[
                out_b.as_entire_binding(),
                src_b.as_entire_binding(),
                dims.as_entire_binding(),
            ],
            [((batch * n) as u32).div_ceil(256), 1, 1],
        );
        cp.submit();
        let gpu = ctx.read_f32(&out_b, batch * n);
        // single-stream reference: run each seq's src through the single-stream sigmoid kernel.
        let mut worst = 0.0f32;
        for b in 0..batch {
            let src = src_all[b * n..(b + 1) * n].to_vec();
            let so = ctx.storage_init("pb.sig.so", &vec![0.0f32; n]);
            let ss = ctx.storage_init("pb.sig.ss", &src);
            let sd = ctx.uniform_of("pb.sig.sd", &[n as u32, 0u32, 0, 0]);
            let mut cp = CommandPass::new(&ctx);
            cp.add_bound(
                &k.gdn_sigmoid_out,
                "sig",
                &[
                    so.as_entire_binding(),
                    ss.as_entire_binding(),
                    sd.as_entire_binding(),
                ],
                [(n as u32).div_ceil(256), 1, 1],
            );
            cp.submit();
            let single = ctx.read_f32(&so, n);
            worst = worst.max(max_abs(&gpu[b * n..(b + 1) * n], &single));
        }
        report("gdn_sigmoid_out_b", worst, 1e-7);
    }

    // ---- (2) gdn_g_decay_b: alpha per-seq [B*n], ssm_dt/ssm_a SHARED [n] ----
    {
        let n = nvh;
        let alpha_all = gen(404, batch * n, -5.0, 5.0);
        let dt = gen(405, n, -2.0, 2.0); // shared
        let a = gen(406, n, -1.5, 1.5); // shared
        let out_b = ctx.storage_init("pb.gd.out", &vec![0.0f32; batch * n]);
        let alpha_b = ctx.storage_init("pb.gd.alpha", &alpha_all);
        let dt_b = ctx.storage_init("pb.gd.dt", &dt);
        let a_b = ctx.storage_init("pb.gd.a", &a);
        let dims = ctx.uniform_of("pb.gd.d", &[n as u32, batch as u32, 0, 0]);
        let mut cp = CommandPass::new(&ctx);
        cp.add_bound(
            &k.gdn_g_decay_b,
            "gd_b",
            &[
                out_b.as_entire_binding(),
                alpha_b.as_entire_binding(),
                dt_b.as_entire_binding(),
                a_b.as_entire_binding(),
                dims.as_entire_binding(),
            ],
            [((batch * n) as u32).div_ceil(256), 1, 1],
        );
        cp.submit();
        let gpu = ctx.read_f32(&out_b, batch * n);
        let mut worst = 0.0f32;
        for b in 0..batch {
            let alpha = alpha_all[b * n..(b + 1) * n].to_vec();
            let so = ctx.storage_init("pb.gd.so", &vec![0.0f32; n]);
            let sa = ctx.storage_init("pb.gd.sa", &alpha);
            let sdt = ctx.storage_init("pb.gd.sdt", &dt);
            let saa = ctx.storage_init("pb.gd.saa", &a);
            let sd = ctx.uniform_of("pb.gd.sd", &[n as u32, 0u32, 0, 0]);
            let mut cp = CommandPass::new(&ctx);
            cp.add_bound(
                &k.gdn_g_decay,
                "gd",
                &[
                    so.as_entire_binding(),
                    sa.as_entire_binding(),
                    sdt.as_entire_binding(),
                    saa.as_entire_binding(),
                    sd.as_entire_binding(),
                ],
                [(n as u32).div_ceil(256), 1, 1],
            );
            cp.submit();
            let single = ctx.read_f32(&so, n);
            worst = worst.max(max_abs(&gpu[b * n..(b + 1) * n], &single));
        }
        report("gdn_g_decay_b", worst, 1e-7);
    }

    // ---- (3) ssm_conv1d_b: per-seq ring, SHARED kernel ----
    {
        let cs = d_conv - 1;
        let conv_in_all = gen(101, batch * conv_dim, -0.1, 0.2);
        let conv_kernel = gen(102, d_conv * conv_dim, -0.05, 0.05); // shared
        let conv_state_all = gen(103, batch * conv_dim * cs, -0.03, 0.06);
        let in_b = ctx.storage_init("pb.cv.in", &conv_in_all);
        let ker_b = ctx.storage_init("pb.cv.k", &conv_kernel);
        let state_b = ctx.storage_init("pb.cv.s", &conv_state_all);
        let out_b = ctx.storage_init("pb.cv.out", &vec![0.0f32; batch * conv_dim]);
        let dims = ctx.uniform_of(
            "pb.cv.d",
            &[conv_dim as u32, d_conv as u32, batch as u32, 0],
        );
        let mut cp = CommandPass::new(&ctx);
        cp.add_bound(
            &k.gdn_conv1d_b,
            "conv_b",
            &[
                in_b.as_entire_binding(),
                ker_b.as_entire_binding(),
                state_b.as_entire_binding(),
                out_b.as_entire_binding(),
                dims.as_entire_binding(),
            ],
            [((batch * conv_dim) as u32).div_ceil(256), 1, 1],
        );
        cp.submit();
        let gpu_out = ctx.read_f32(&out_b, batch * conv_dim);
        let gpu_state = ctx.read_f32(&state_b, batch * conv_dim * cs);
        let mut worst_o = 0.0f32;
        let mut worst_s = 0.0f32;
        for b in 0..batch {
            let cin = conv_in_all[b * conv_dim..(b + 1) * conv_dim].to_vec();
            let cst = conv_state_all[b * conv_dim * cs..(b + 1) * conv_dim * cs].to_vec();
            let sin = ctx.storage_init("pb.cv.sin", &cin);
            let sker = ctx.storage_init("pb.cv.sker", &conv_kernel);
            let sst = ctx.storage_init("pb.cv.sst", &cst);
            let sout = ctx.storage_init("pb.cv.sout", &vec![0.0f32; conv_dim]);
            let sd = ctx.uniform_of("pb.cv.sd", &[conv_dim as u32, d_conv as u32, 0, 0]);
            let mut cp = CommandPass::new(&ctx);
            cp.add_bound(
                &k.gdn_conv1d,
                "conv",
                &[
                    sin.as_entire_binding(),
                    sker.as_entire_binding(),
                    sst.as_entire_binding(),
                    sout.as_entire_binding(),
                    sd.as_entire_binding(),
                ],
                [(conv_dim as u32).div_ceil(256), 1, 1],
            );
            cp.submit();
            let single_out = ctx.read_f32(&sout, conv_dim);
            let single_state = ctx.read_f32(&sst, conv_dim * cs);
            worst_o = worst_o.max(max_abs(
                &gpu_out[b * conv_dim..(b + 1) * conv_dim],
                &single_out,
            ));
            worst_s = worst_s.max(max_abs(
                &gpu_state[b * conv_dim * cs..(b + 1) * conv_dim * cs],
                &single_state,
            ));
        }
        report("ssm_conv1d_b.out", worst_o, 1e-7);
        report("ssm_conv1d_b.ring", worst_s, 1e-7);
    }

    // ---- (4) gdn_l2_norm_b: per-(b,head) L2, in place ----
    {
        let dim = s; // 128
        let n_heads = nkh; // 16
        let x_all = gen(201, batch * dim * n_heads, -0.8, 1.6);
        let x_b = ctx.storage_init("pb.l2.x", &x_all);
        let dims = ctx.uniform_of(
            "pb.l2.d",
            &[dim as u32, n_heads as u32, eps.to_bits(), batch as u32],
        );
        let mut cp = CommandPass::new(&ctx);
        cp.add_bound(
            &k.gdn_l2_norm_b,
            "l2_b",
            &[x_b.as_entire_binding(), dims.as_entire_binding()],
            [(batch * n_heads) as u32, 1, 1],
        );
        cp.submit();
        let gpu = ctx.read_f32(&x_b, batch * dim * n_heads);
        let mut worst = 0.0f32;
        for b in 0..batch {
            let seg = x_all[b * dim * n_heads..(b + 1) * dim * n_heads].to_vec();
            let sx = ctx.storage_init("pb.l2.sx", &seg);
            let sd = ctx.uniform_of(
                "pb.l2.sd",
                &[dim as u32, n_heads as u32, eps.to_bits(), 0u32],
            );
            let mut cp = CommandPass::new(&ctx);
            cp.add_bound(
                &k.gdn_l2_norm,
                "l2",
                &[sx.as_entire_binding(), sd.as_entire_binding()],
                [n_heads as u32, 1, 1],
            );
            cp.submit();
            let single = ctx.read_f32(&sx, dim * n_heads);
            worst = worst.max(max_abs(
                &gpu[b * dim * n_heads..(b + 1) * dim * n_heads],
                &single,
            ));
        }
        report("gdn_l2_norm_b", worst, 1e-7);
    }

    // ---- (5) gdn_tile_b: per-seq split + repeat ----
    {
        let conv_out_all = gen(501, batch * conv_dim, -1.0, 1.0);
        let co_b = ctx.storage_init("pb.tl.co", &conv_out_all);
        let q_b = ctx.storage_init("pb.tl.q", &vec![0.0f32; batch * value_dim]);
        let k_b = ctx.storage_init("pb.tl.k", &vec![0.0f32; batch * value_dim]);
        let v_b = ctx.storage_init("pb.tl.v", &vec![0.0f32; batch * value_dim]);
        let dims = ctx.uniform_of(
            "pb.tl.d",
            &[s as u32, nkh as u32, nvh as u32, key_dim as u32],
        );
        let dims2 = ctx.uniform_of("pb.tl.d2", &[conv_dim as u32, batch as u32, 0, 0]);
        let mut cp = CommandPass::new(&ctx);
        cp.add_bound(
            &k.gdn_tile_b,
            "tile_b",
            &[
                co_b.as_entire_binding(),
                q_b.as_entire_binding(),
                k_b.as_entire_binding(),
                v_b.as_entire_binding(),
                dims.as_entire_binding(),
                dims2.as_entire_binding(),
            ],
            [((batch * value_dim) as u32).div_ceil(256), 1, 1],
        );
        cp.submit();
        let gq = ctx.read_f32(&q_b, batch * value_dim);
        let gk = ctx.read_f32(&k_b, batch * value_dim);
        let gv = ctx.read_f32(&v_b, batch * value_dim);
        let mut wq = 0.0f32;
        let mut wk = 0.0f32;
        let mut wv = 0.0f32;
        for b in 0..batch {
            let co = conv_out_all[b * conv_dim..(b + 1) * conv_dim].to_vec();
            let sco = ctx.storage_init("pb.tl.sco", &co);
            let sq = ctx.storage_init("pb.tl.sq", &vec![0.0f32; value_dim]);
            let sk = ctx.storage_init("pb.tl.sk", &vec![0.0f32; value_dim]);
            let sv = ctx.storage_init("pb.tl.sv", &vec![0.0f32; value_dim]);
            let sd = ctx.uniform_of(
                "pb.tl.sd",
                &[s as u32, nkh as u32, nvh as u32, key_dim as u32],
            );
            let mut cp = CommandPass::new(&ctx);
            cp.add_bound(
                &k.gdn_tile,
                "tile",
                &[
                    sco.as_entire_binding(),
                    sq.as_entire_binding(),
                    sk.as_entire_binding(),
                    sv.as_entire_binding(),
                    sd.as_entire_binding(),
                ],
                [(value_dim as u32).div_ceil(256), 1, 1],
            );
            cp.submit();
            wq = wq.max(max_abs(
                &gq[b * value_dim..(b + 1) * value_dim],
                &ctx.read_f32(&sq, value_dim),
            ));
            wk = wk.max(max_abs(
                &gk[b * value_dim..(b + 1) * value_dim],
                &ctx.read_f32(&sk, value_dim),
            ));
            wv = wv.max(max_abs(
                &gv[b * value_dim..(b + 1) * value_dim],
                &ctx.read_f32(&sv, value_dim),
            ));
        }
        report("gdn_tile_b.q", wq, 1e-7);
        report("gdn_tile_b.k", wk, 1e-7);
        report("gdn_tile_b.v", wv, 1e-7);
    }

    // ---- (6) gated_delta_net_b recurrence, 3 tokens, B independent state slices ----
    {
        let n_tokens = 3usize;
        // b-outer state: seq b at [b*s*s*nvh..]. Each seq gets DIFFERENT initial state + inputs.
        let mut state_all = Vec::with_capacity(batch * s * s * nvh);
        for b in 0..batch {
            state_all.extend(gen(7 + b as u64 * 97, s * s * nvh, -0.05, 0.10));
        }
        let gpu_state_b = ctx.storage_init("pb.rc.S", &state_all);
        // single-stream reference states (one per seq, advanced independently).
        let mut single_states: Vec<wgpu::Buffer> = (0..batch)
            .map(|b| {
                ctx.storage_init(
                    &format!("pb.rc.sS{b}"),
                    &state_all[b * s * s * nvh..(b + 1) * s * s * nvh],
                )
            })
            .collect();

        let mut worst_out = 0.0f32;
        let mut worst_state = 0.0f32;
        for t in 0..n_tokens {
            // per-(b,token) inputs, b-outer packed.
            let mut q_all = Vec::with_capacity(batch * s * nvh);
            let mut k_all = Vec::with_capacity(batch * s * nvh);
            let mut v_all = Vec::with_capacity(batch * s * nvh);
            let mut g_all = Vec::with_capacity(batch * nvh);
            let mut beta_all = Vec::with_capacity(batch * nvh);
            for b in 0..batch {
                let bs = (b * 100 + t) as u64;
                q_all.extend(gen(1000 + bs, s * nvh, -0.6, 1.2));
                k_all.extend(gen(2000 + bs, s * nvh, -0.6, 1.2));
                v_all.extend(gen(3000 + bs, s * nvh, -0.5, 1.0));
                g_all.extend(gen(4000 + bs, nvh, -0.5, 0.4));
                beta_all.extend(gen(5000 + bs, nvh, 0.1, 0.8));
            }
            // batched B-kernel step.
            let q_b = ctx.storage_init("pb.rc.q", &q_all);
            let k_b = ctx.storage_init("pb.rc.k", &k_all);
            let v_b = ctx.storage_init("pb.rc.v", &v_all);
            let g_b = ctx.storage_init("pb.rc.g", &g_all);
            let beta_b = ctx.storage_init("pb.rc.b", &beta_all);
            let o_b = ctx.storage_init("pb.rc.o", &vec![0.0f32; batch * s * nvh]);
            let dims = ctx.uniform_of("pb.rc.d", &[s as u32, s as u32, nvh as u32, batch as u32]);
            let mut cp = CommandPass::new(&ctx);
            cp.add_bound(
                &k.gdn_recurrence_b,
                "rec_b",
                &[
                    q_b.as_entire_binding(),
                    k_b.as_entire_binding(),
                    v_b.as_entire_binding(),
                    g_b.as_entire_binding(),
                    beta_b.as_entire_binding(),
                    gpu_state_b.as_entire_binding(),
                    o_b.as_entire_binding(),
                    dims.as_entire_binding(),
                ],
                [(batch * nvh) as u32, 1, 1],
            );
            cp.submit();
            let gpu_o = ctx.read_f32(&o_b, batch * s * nvh);
            let gpu_state = ctx.read_f32(&gpu_state_b, batch * s * s * nvh);

            // single-stream per-seq step (M11 kernel), each advancing its own state.
            for b in 0..batch {
                let q = q_all[b * s * nvh..(b + 1) * s * nvh].to_vec();
                let kk = k_all[b * s * nvh..(b + 1) * s * nvh].to_vec();
                let v = v_all[b * s * nvh..(b + 1) * s * nvh].to_vec();
                let g = g_all[b * nvh..(b + 1) * nvh].to_vec();
                let beta = beta_all[b * nvh..(b + 1) * nvh].to_vec();
                let sq = ctx.storage_init("pb.rc.sq", &q);
                let sk = ctx.storage_init("pb.rc.sk", &kk);
                let sv = ctx.storage_init("pb.rc.sv", &v);
                let sg = ctx.storage_init("pb.rc.sg", &g);
                let sbeta = ctx.storage_init("pb.rc.sbeta", &beta);
                let so = ctx.storage_init("pb.rc.so", &vec![0.0f32; s * nvh]);
                let sd = ctx.uniform_of("pb.rc.sd", &[s as u32, s as u32, nvh as u32, 0u32]);
                let mut cp = CommandPass::new(&ctx);
                cp.add_bound(
                    &k.gdn_recurrence,
                    "rec",
                    &[
                        sq.as_entire_binding(),
                        sk.as_entire_binding(),
                        sv.as_entire_binding(),
                        sg.as_entire_binding(),
                        sbeta.as_entire_binding(),
                        single_states[b].as_entire_binding(),
                        so.as_entire_binding(),
                        sd.as_entire_binding(),
                    ],
                    [nvh as u32, 1, 1],
                );
                cp.submit();
                let single_o = ctx.read_f32(&so, s * nvh);
                let single_s = ctx.read_f32(&single_states[b], s * s * nvh);
                worst_out =
                    worst_out.max(max_abs(&gpu_o[b * s * nvh..(b + 1) * s * nvh], &single_o));
                worst_state = worst_state.max(max_abs(
                    &gpu_state[b * s * s * nvh..(b + 1) * s * s * nvh],
                    &single_s,
                ));
            }
            let _ = &mut single_states;
            println!("    recurrence_b token {t}: o {worst_out:.3e}  state {worst_state:.3e}");
        }
        report("gdn_recurrence_b.out", worst_out, 1e-6);
        report("gdn_recurrence_b.state", worst_state, 1e-6);
    }

    println!(
        "\nB-PARALLEL WGSL PARITY: {}",
        if all_ok { "ALL PASS" } else { "SOME FAILED" }
    );
    assert!(all_ok, "B-parallel WGSL GDN kernel parity FAILED — a `_b` twin diverged or leaked across the batch");
}
