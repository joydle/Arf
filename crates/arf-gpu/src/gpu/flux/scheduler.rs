//! FLUX flow-match Euler scheduler (host-side, pure arithmetic — no GPU). Schnell denoises in a
//! few steps with no guidance. The schedule is deterministic: sigmas = linspace(1, 1/N, N) with
//! a resolution-dependent `shift` (mu) folded in, timesteps = sigma·1000. The Euler update is
//!     x_{i+1} = x_i + (sigma_{i+1} - sigma_i) · velocity_i
//! Kept a standalone struct so it's unit-tested against the diffusers reference in isolation from
//! the (heavy, already-parity-proven) DiT forward. Mirrors diffusers
//! `FlowMatchEulerDiscreteScheduler` + `calculate_shift` + the pipeline's `sigmas` line.

/// The flow-match Euler schedule for one generation: `sigmas[0..=steps]` (sigmas`steps`=0) and the
/// per-step `timesteps` (= sigma·1000) handed to the DiT.
#[derive(Debug, Clone)]
pub struct FlowMatchSchedule {
    pub sigmas: Vec<f32>,    // length steps+1, descending to 0
    pub timesteps: Vec<f32>, // length steps (sigma_i · 1000)
}

impl FlowMatchSchedule {
    /// Build the schnell schedule for `steps` inference steps. diffusers passes explicit
    /// `sigmas = linspace(1, 1/N, N)` to the scheduler, and with `use_dynamic_shifting=False`
    /// they are used AS-IS (the resolution `shift`/mu is computed but only applied when dynamic
    /// shifting is on — off for schnell). So sigmas = linspace(1,1/N,N) then a trailing 0, and
    /// timesteps = sigma·1000. `image_seq_len` is accepted for API symmetry / future dynamic
    /// shifting but does not alter the schnell schedule.
    pub fn schnell(steps: usize, _image_seq_len: usize) -> Self {
        let n = steps as f32;
        let mut sigmas: Vec<f32> = (0..steps)
            .map(|i| 1.0 - (i as f32) * (1.0 - 1.0 / n) / (n - 1.0).max(1.0))
            .collect();
        let timesteps: Vec<f32> = sigmas.iter().map(|&s| s * 1000.0).collect();
        sigmas.push(0.0); // terminal sigma
        FlowMatchSchedule { sigmas, timesteps }
    }

    pub fn steps(&self) -> usize {
        self.timesteps.len()
    }

    /// One Euler update in place: `x += (sigma_next - sigma) · velocity` (element-wise).
    pub fn step(&self, i: usize, x: &mut [f32], velocity: &[f32]) {
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        for (xi, &v) in x.iter_mut().zip(velocity.iter()) {
            *xi += dt * v;
        }
    }
}
