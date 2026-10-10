//! Mixture-of-experts feed-forward (Qwen3-MoE).
//!
//! A router (a bias-free `Linear`, `hidden → num_experts`) scores the experts for
//! each token; we softmax over all experts, take the `top_k` highest, optionally
//! renormalize those k weights to sum to 1 (Qwen3 `norm_topk_prob`), run each
//! selected expert's SwiGLU, and weight-sum. A shared expert (always-on, no router
//! weight) is added on top. This is the bit-exact CPU oracle for the GPU MoE block.
//!
//! Decision: softmax-over-all-then-select-top-k (vs top-k-then-softmax) —
//! tradeoff: matches HF Qwen3 (`Qwen3MoeSparseMoeBlock`: full softmax, topk,
//! then optional renorm) so our logits track the reference; the cheaper
//! top-k-then-softmax would diverge from the checkpoint's trained routing.

use crate::model::mlp::Mlp;
use crate::model::nn::Linear;
use crate::tensor::Tensor;

/// One mixture-of-experts block.
#[derive(Debug, Clone)]
pub struct MoeMlp {
    /// Router: `hidden → num_experts` logits.
    router: Linear,
    /// The routed experts (length `num_experts`), each a SwiGLU of `moe_intermediate`.
    experts: Vec<Mlp>,
    /// Always-on shared experts, summed unconditionally (Qwen3 has 0 or 1).
    shared: Vec<Mlp>,
    top_k: usize,
    /// Renormalize the selected top-k router weights to sum to 1.
    norm_topk: bool,
}

impl MoeMlp {
    pub fn new(
        router: Linear,
        experts: Vec<Mlp>,
        shared: Vec<Mlp>,
        top_k: usize,
        norm_topk: bool,
    ) -> Self {
        MoeMlp {
            router,
            experts,
            shared,
            top_k,
            norm_topk,
        }
    }

    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// `x`: `[tokens, hidden]` → `[tokens, hidden]`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (tokens, hidden) = x.dims2();
        let logits = self.router.forward(x); // [tokens, num_experts]
        let n_exp = self.experts.len();

        let mut out = vec![0.0f32; tokens * hidden];

        // Shared experts apply to every token regardless of routing.
        for sh in &self.shared {
            let sh_out = sh.forward(x);
            for (o, &s) in out.iter_mut().zip(sh_out.as_slice()) {
                *o += s;
            }
        }

        for t in 0..tokens {
            let row = &logits.as_slice()[t * n_exp..(t + 1) * n_exp];
            let (ids, weights) = top_k_softmax(row, self.top_k, self.norm_topk);

            // Each selected expert runs on this token's single row, scaled by its
            // router weight, and accumulated.
            let xt = Tensor::from_vec(x.row(t).to_vec(), vec![1, hidden]);
            for (&e, &w) in ids.iter().zip(&weights) {
                let y = self.experts[e].forward(&xt); // [1, hidden]
                let dst = &mut out[t * hidden..(t + 1) * hidden];
                for (d, &v) in dst.iter_mut().zip(y.as_slice()) {
                    *d += w * v;
                }
            }
        }

        Tensor::from_vec(out, vec![tokens, hidden])
    }
}

/// Full softmax over `logits`, pick the `top_k` largest, return their indices and
/// (optionally renormalized) weights. Ties broken by lower index (stable).
fn top_k_softmax(logits: &[f32], top_k: usize, norm_topk: bool) -> (Vec<usize>, Vec<f32>) {
    let n = logits.len();
    let k = top_k.min(n);

    // Softmax over all experts.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Top-k by probability (stable: ties keep the lower index, matching argsort).
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);

    let mut weights: Vec<f32> = idx.iter().map(|&i| probs[i]).collect();
    if norm_topk {
        let s: f32 = weights.iter().sum();
        if s > 0.0 {
            for w in &mut weights {
                *w /= s;
            }
        }
    }
    (idx, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Bf16Matrix;
    use std::sync::Arc;

    /// A Linear from an explicit `[out, in]` f32 weight (rounded to bf16).
    fn lin(weight: &[f32], out: usize, inn: usize) -> Linear {
        Linear::from_matrix(Arc::new(Bf16Matrix::from_f32(weight, out, inn)))
    }

    #[test]
    fn softmax_top_k_picks_largest_and_normalizes() {
        // logits favor experts 2 and 0; top_k=2, norm on -> weights sum to 1.
        let logits = [2.0f32, -1.0, 3.0, 0.0];
        let (ids, w) = top_k_softmax(&logits, 2, true);
        assert_eq!(ids, vec![2, 0]); // expert 2 highest, then 0
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(w[0] > w[1]); // expert 2 weighted more than expert 0
    }

    #[test]
    fn no_norm_keeps_raw_softmax_mass() {
        let logits = [2.0f32, -1.0, 3.0, 0.0];
        let (_ids, w) = top_k_softmax(&logits, 2, false);
        // Sum equals the softmax mass of the top-2, strictly < 1 (some mass on
        // experts 1,3).
        let s: f32 = w.iter().sum();
        assert!(s < 1.0 && s > 0.5);
    }

    #[test]
    fn ties_break_to_lower_index() {
        let logits = [1.0f32, 1.0, 1.0, 1.0];
        let (ids, _w) = top_k_softmax(&logits, 2, true);
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn shared_expert_adds_to_every_token() {
        // 2 experts, top_k=1, plus a shared expert. Router strongly prefers e0 for
        // both tokens. We verify the shared contribution is present by comparing a
        // build with/without it.
        let hidden = 4;
        let inter = 4;
        // Router weight [num_experts=2, hidden=4]: row0 large -> e0 always wins.
        let router_w = vec![
            10.0, 0.0, 0.0, 0.0, // e0 logit ~ 10*x0
            -10.0, 0.0, 0.0, 0.0, // e1 logit ~ -10*x0
        ];
        let router = lin(&router_w, 2, hidden);
        // Expert SwiGLU with simple weights (any deterministic values).
        let mk_expert = |seed: f32| {
            let gate = vec![seed; inter * hidden];
            let up = vec![seed * 0.5; inter * hidden];
            let down = vec![seed * 0.25; hidden * inter];
            Mlp::new(
                lin(&gate, inter, hidden),
                lin(&up, inter, hidden),
                lin(&down, hidden, inter),
            )
        };
        let experts = vec![mk_expert(0.3), mk_expert(0.1)];
        let shared_expert = mk_expert(0.2);

        let x = Tensor::from_vec(
            vec![1.0, 0.5, -0.5, 0.25, 0.8, -0.3, 0.2, 0.1],
            vec![2, hidden],
        );

        let without = MoeMlp::new(router.clone(), experts.clone(), vec![], 1, true).forward(&x);
        let with = MoeMlp::new(router, experts, vec![shared_expert.clone()], 1, true).forward(&x);

        let shared_only = shared_expert.forward(&x);
        // with == without + shared (the shared expert is added unconditionally).
        for ((a, b), s) in with
            .as_slice()
            .iter()
            .zip(without.as_slice())
            .zip(shared_only.as_slice())
        {
            assert!((a - (b + s)).abs() < 1e-4, "{a} != {b} + {s}");
        }
    }
}
