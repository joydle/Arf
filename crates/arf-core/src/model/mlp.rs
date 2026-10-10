//! The Llama gated feed-forward network: `down(silu(gate(x)) * up(x))`.

use crate::model::nn::Linear;
use crate::tensor::Tensor;

/// SwiGLU feed-forward block. All projections are bias-free, as in Llama.
#[derive(Debug, Clone)]
pub struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    pub fn new(gate: Linear, up: Linear, down: Linear) -> Self {
        Mlp { gate, up, down }
    }

    /// `x`: `[tokens, hidden]` → `[tokens, hidden]`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.gate.forward(x).silu();
        let up = self.up.forward(x);
        self.down.forward(&gate.mul(&up))
    }
}
