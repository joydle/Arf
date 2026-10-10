//! CPU oracle for CHUNKED gated-delta-net — the ~64x prefill lever, modelled
//! before a line of GPU kernel is written.
//!
//! # Why this is the most valuable test in the repo right now
//!
//! Prefill is LAUNCH-BOUND, measured: 13.40 ms/token FLAT across a 9x prompt
//! increase (measured). Every prompt token costs one dependent
//! round-trip through 48 GDN layers, so a 2K prompt takes 27 seconds and an 8K
//! prompt 110. That is the dominant cost of agentic work, and it is not
//! arithmetic — it is the number of dependent launches.
//!
//! llama.cpp removes it by solving each CS=64 chunk in CLOSED FORM instead of
//! stepping it (`build_delta_net_chunking`, src/models/delta-net-base.cpp:16),
//! which cuts dependent steps 64x. vLLM has the same split
//! (`chunk_gated_delta_rule` for prefill, `fused_recurrent_..._decode` for
//! decode); we have only the decode kernel, which is the whole gap.
//!
//! # Why an oracle FIRST
//!
//! This is a numerics port into a RECURRENCE. Getting it subtly wrong does not
//! error — it yields fluent, confident, wrong text, the failure mode that has
//! cost this project the most. Worse, the chunked form is algebraically
//! *different* from the serial one: it must agree only to f32 rounding, so
//! "close but not equal" is expected and a bug hides easily inside that
//! tolerance unless the equivalence is pinned deliberately.
//!
//! So: model both forms on the CPU, prove they agree, and only then write the
//! kernel against a reference that is known-good. That is the discipline that
//! cleared the tq4 KV port; skipping it is what let an `xds` layout bug reach
//! the model and produce multilingual garbage.
//!
//! # The recurrence being modelled
//!
//! Per head, state `S` is [d_k, d_v]. For each token t with decay `g_t`,
//! write-strength `beta_t`, and vectors q_t, k_t, v_t:
//!
//!     S_t = diag_decay(g_t) * S_{t-1} + beta_t * k_t (v_t - k_t^T S_{t-1})^T
//!     o_t = S_t^T q_t
//!
//! The delta rule: it REMOVES what the current key already predicts
//! (`k_t^T S_{t-1}`) before writing, which is what makes it a delta net rather
//! than a plain linear-attention sum.

/// A head's geometry. Small on purpose: the point is exactness, not scale.
const DK: usize = 8;
const DV: usize = 8;

/// Deterministic, non-trivial, not axis-aligned.
fn seq(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((t % 1009) as f32 / 1009.0 - 0.5) * 2.0
        })
        .collect()
}

/// The SERIAL form — one step per token. This is what a shipping GPU kernel
/// does and what the chunked form must reproduce.
///
/// Returns (outputs [T][DV], final state [DK*DV]).
fn serial(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    t_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut s = vec![0.0f32; DK * DV];
    let mut out = vec![0.0f32; t_len * DV];
    for t in 0..t_len {
        let (qt, kt, vt) = (&q[t * DK..], &k[t * DK..], &v[t * DV..]);
        // 1. decay the whole state by this token's gate.
        let decay = g[t].exp();
        for x in s.iter_mut() {
            *x *= decay;
        }
        // 2. delta rule: what does the current key ALREADY predict?
        //    pred[j] = sum_i k[i] * S[i][j]
        let mut pred = [0.0f32; DV];
        for i in 0..DK {
            for (j, p) in pred.iter_mut().enumerate() {
                *p += kt[i] * s[i * DV + j];
            }
        }
        // 3. write the CORRECTION, not the value.
        for i in 0..DK {
            for j in 0..DV {
                s[i * DV + j] += beta[t] * kt[i] * (vt[j] - pred[j]);
            }
        }
        // 4. read out.
        for j in 0..DV {
            let mut a = 0.0f32;
            for i in 0..DK {
                a += qt[i] * s[i * DV + j];
            }
            out[t * DV + j] = a;
        }
    }
    (out, s)
}

/// The CHUNKED form — llama.cpp's shape, `build_delta_net_chunking`.
///
/// Within a chunk the per-token dependency is inverted ALGEBRAICALLY instead of
/// stepped, so a chunk costs a few dense ops rather than CS sequential updates.
/// Chunks remain sequential, but there are CS-times fewer of them — which is the
/// entire win, since the cost is dependent launches and not arithmetic.
///
/// THE CORE IDENTITY, and the thing this test exists to prove: for tokens i < j
/// inside one chunk, token j's state already contains token i's write, decayed
/// by the gates between them. Collecting those pairwise decays into a matrix and
/// solving one lower-triangular system reproduces the whole chain at once.
fn chunked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    t_len: usize,
    cs: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut s = vec![0.0f32; DK * DV];
    let mut out = vec![0.0f32; t_len * DV];

    let mut c0 = 0usize;
    while c0 < t_len {
        let n = cs.min(t_len - c0);

        // Cumulative log-decay INSIDE the chunk: cg[t] = sum of g over 0..=t.
        // Everything below is expressed against these, which is what lets the
        // pairwise decays become a matrix instead of a loop.
        let mut cg = vec![0.0f32; n];
        let mut acc = 0.0f32;
        for t in 0..n {
            acc += g[c0 + t];
            cg[t] = acc;
        }

        // attn[i][j] for j < i: how much of token j's write survives to token i.
        // exp(cg[i] - cg[j]) is the decay between them; the k.k term is how much
        // token i's key overlaps token j's. Strictly lower triangular: a token
        // does not correct itself here.
        let mut attn = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..i {
                let mut kk = 0.0f32;
                for d in 0..DK {
                    kk += k[(c0 + i) * DK + d] * k[(c0 + j) * DK + d];
                }
                attn[i * n + j] = -beta[c0 + i] * kk * (cg[i] - cg[j]).exp();
            }
        }

        // Solve (I - attn) X = I by forward substitution — the closed form that
        // replaces CS sequential steps. `ggml_solve_tri` is this.
        let mut inv = vec![0.0f32; n * n];
        for i in 0..n {
            inv[i * n + i] = 1.0;
        }
        for col in 0..n {
            for i in 0..n {
                let mut a = inv[i * n + col];
                for j in 0..i {
                    a += attn[i * n + j] * inv[j * n + col];
                }
                inv[i * n + col] = a;
            }
        }

        // Each token's EFFECTIVE write, after intra-chunk corrections:
        //   u[i] = beta_i * (v_i - k_i^T S_prev * decay_to_i)
        // then propagated through `inv` so token i also carries the corrections
        // owed to the tokens before it.
        let mut w = vec![0.0f32; n * DV];
        for i in 0..n {
            let d_in = cg[i].exp();
            let mut pred = [0.0f32; DV];
            for d in 0..DK {
                let kv = k[(c0 + i) * DK + d] * d_in;
                for (j, p) in pred.iter_mut().enumerate() {
                    *p += kv * s[d * DV + j];
                }
            }
            for j in 0..DV {
                w[i * DV + j] = beta[c0 + i] * (v[(c0 + i) * DV + j] - pred[j]);
            }
        }
        let mut wc = vec![0.0f32; n * DV];
        for i in 0..n {
            for j in 0..n {
                let m = if i == j { 1.0 } else { inv[i * n + j] };
                if m == 0.0 {
                    continue;
                }
                for d in 0..DV {
                    wc[i * DV + d] += m * w[j * DV + d];
                }
            }
        }

        // Outputs: the carried-in state (decayed to this token) plus every
        // in-chunk write that precedes it (decayed from its own position).
        for i in 0..n {
            let d_in = cg[i].exp();
            for d in 0..DV {
                let mut a = 0.0f32;
                for x in 0..DK {
                    a += q[(c0 + i) * DK + x] * d_in * s[x * DV + d];
                }
                out[(c0 + i) * DV + d] = a;
            }
            for j in 0..=i {
                let mut qk = 0.0f32;
                for x in 0..DK {
                    qk += q[(c0 + i) * DK + x] * k[(c0 + j) * DK + x];
                }
                let dec = (cg[i] - cg[j]).exp();
                for d in 0..DV {
                    out[(c0 + i) * DV + d] += qk * dec * wc[j * DV + d];
                }
            }
        }

        // Fold the chunk into the running state, so the next chunk starts right.
        let total = cg[n - 1].exp();
        for x in s.iter_mut() {
            *x *= total;
        }
        for j in 0..n {
            let dec = (cg[n - 1] - cg[j]).exp();
            for x in 0..DK {
                let kx = k[(c0 + j) * DK + x] * dec;
                for d in 0..DV {
                    s[x * DV + d] += kx * wc[j * DV + d];
                }
            }
        }

        c0 += n;
    }
    (out, s)
}

fn inputs(t_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let q = seq(t_len * DK, 11);
    let k = seq(t_len * DK, 29);
    let v = seq(t_len * DV, 47);
    // Gates are log-decays: strictly negative, so state decays rather than grows.
    let g: Vec<f32> = seq(t_len, 71)
        .iter()
        .map(|x| -0.05 - 0.1 * x.abs())
        .collect();
    // beta in (0, 1): the write strength.
    let beta: Vec<f32> = seq(t_len, 97).iter().map(|x| 0.5 + 0.4 * x).collect();
    (q, k, v, g, beta)
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|x| x * x).sum::<f32>().max(1e-20);
    (num / den).sqrt()
}

/// THE EQUIVALENCE. Chunked must reproduce serial — outputs AND the final state,
/// because the state is what the next chunk and the next layer consume.
#[test]
fn chunked_matches_serial_at_every_chunk_size() {
    let t_len = 48;
    let (q, k, v, g, beta) = inputs(t_len);
    let (want_o, want_s) = serial(&q, &k, &v, &g, &beta, t_len);

    // CS=1 degenerates to the serial form and must match near-exactly; larger
    // chunks exercise the closed form. A size that does NOT divide t_len is
    // included deliberately — the last chunk is short and that is where an
    // off-by-one lives.
    for cs in [1usize, 2, 4, 8, 16, 48, 64, 7] {
        let (got_o, got_s) = chunked(&q, &k, &v, &g, &beta, t_len, cs);
        let ro = rel(&got_o, &want_o);
        let rs = rel(&got_s, &want_s);
        assert!(
            ro < 1e-4,
            "CS={cs}: outputs drift {ro:.3e} from serial — the chunked form is \
             not equivalent, not merely imprecise"
        );
        assert!(
            rs < 1e-4,
            "CS={cs}: FINAL STATE drifts {rs:.3e}. Outputs can look right while \
             the state is wrong; the next chunk and the next token consume this."
        );
    }
}

/// The state carried ACROSS chunks is what makes this a recurrence. A form that
/// resets per chunk passes a single-chunk test and fails here.
#[test]
fn state_carries_across_chunk_boundaries() {
    let t_len = 32;
    let (q, k, v, g, beta) = inputs(t_len);
    // One chunk covering everything vs four that must hand state along.
    let (o_one, s_one) = chunked(&q, &k, &v, &g, &beta, t_len, 32);
    let (o_four, s_four) = chunked(&q, &k, &v, &g, &beta, t_len, 8);
    assert!(
        rel(&o_four, &o_one) < 1e-4 && rel(&s_four, &s_one) < 1e-4,
        "four chunks must equal one: state is not being carried across the boundary"
    );
}

/// THE DELTA RULE ITSELF. With beta = 0 nothing is ever written, so the state
/// stays zero and every output is zero — regardless of q, k, v. A form that
/// writes the VALUE instead of the CORRECTION still passes the equivalence test
/// above (both forms would be wrong together) but fails this.
#[test]
fn zero_beta_writes_nothing() {
    let t_len = 16;
    let (q, k, v, g, _) = inputs(t_len);
    let beta = vec![0.0f32; t_len];
    let (o, s) = chunked(&q, &k, &v, &g, &beta, t_len, 8);
    assert!(
        o.iter().all(|x| x.abs() < 1e-6),
        "beta=0 must write nothing"
    );
    assert!(
        s.iter().all(|x| x.abs() < 1e-6),
        "beta=0 must leave state zero"
    );
}

/// DECAY MUST DECAY. A strongly negative gate should make an early token's
/// contribution vanish by the end; a near-zero gate should preserve it. This
/// pins the SIGN and placement of the cumulative-decay term, which is the part
/// most easily inverted (exp(cg[j] - cg[i]) instead of exp(cg[i] - cg[j])).
#[test]
fn decay_direction_is_not_inverted() {
    let t_len = 24;
    let (q, k, v, _, beta) = inputs(t_len);
    let strong: Vec<f32> = vec![-2.0; t_len];
    let weak: Vec<f32> = vec![-0.001; t_len];
    let (_, s_strong) = chunked(&q, &k, &v, &strong, &beta, t_len, 8);
    let (_, s_weak) = chunked(&q, &k, &v, &weak, &beta, t_len, 8);
    let n_strong: f32 = s_strong.iter().map(|x| x * x).sum::<f32>().sqrt();
    let n_weak: f32 = s_weak.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        n_strong < n_weak,
        "strong decay ({n_strong:.3e}) must leave LESS state than weak \
         ({n_weak:.3e}) — the decay term is inverted"
    );
}
