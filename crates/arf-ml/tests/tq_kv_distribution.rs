//! How much does tq4 attention actually lose — and does it depend on what the
//! K/V vectors LOOK like?
//!
//! WHY THIS EXISTS. A GPU attention-parity check can find a tq4 kernel exact
//! against the CPU codec and still see the CODEC itself move attention output
//! by a rel-L2 well past the 0.25 bar arf-core's own tq gate holds logits to
//! (tests/model.rs::tq_prefill_matches_f32_oracle_within_drift).
//!
//! Such a figure is usually measured on `sin`-noise vectors, which are ISOTROPIC: energy
//! spread evenly over all 256 coordinates. Real attention keys are not. They are
//! strongly anisotropic — a few dominant directions carry most of the energy —
//! and that anisotropy is the entire premise of TurboQuant: the random rotation
//! SPREADS a concentrated vector so its coordinates become near-Gaussian, which
//! is the density the Lloyd-Max codebook is fitted to.
//!
//! So "tq4 loses that much rel-L2" may be a statement about the test input rather than
//! about tq4. This test settles it by running the same codec over both regimes.
//! Whatever it finds is a number for the record either way: if isotropic input
//! is the pessimal case, such a gate's codec figure is a floor, not a verdict.

use arf_ml::turboquant::{Codebook, Rotation, RotationKind, TqKvBlock};

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|a| a * a).sum::<f32>().sqrt()
}

/// Deterministic LCG in [-1, 1). Not for crypto; for a reproducible sample.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u32 << 31) as f32) - 1.0
    }
    /// Box-Muller normal.
    fn normal(&mut self) -> f32 {
        let u1 = (self.next_f32() * 0.5 + 0.5).max(1e-7);
        let u2 = self.next_f32() * 0.5 + 0.5;
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Attention over an explicit K/V set, in f32. Returns the output vector.
fn attend(q: &[f32], keys: &[Vec<f32>], vals: &[Vec<f32>], d: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let mut s: Vec<f32> = keys
        .iter()
        .map(|k| q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale)
        .collect();
    let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in s.iter_mut() {
        *x = (*x - mx).exp();
        sum += *x;
    }
    let mut out = vec![0.0f32; d];
    for (w, v) in s.iter().zip(vals) {
        for i in 0..d {
            out[i] += w * v[i];
        }
    }
    for o in out.iter_mut() {
        *o /= sum;
    }
    out
}

fn round_trip(vs: &[Vec<f32>], d: usize, bits: u8) -> Vec<Vec<f32>> {
    let cb = Codebook::new(bits, d);
    let rot = Rotation::new(d, 0, RotationKind::Hadamard);
    let mut blk = TqKvBlock::zeros(vs.len(), d, bits);
    for (i, v) in vs.iter().enumerate() {
        blk.set_vector(i, v, &cb, &rot);
    }
    let flat = blk.to_f32(&cb, &rot);
    (0..vs.len())
        .map(|i| flat[i * d..(i + 1) * d].to_vec())
        .collect()
}

/// `(rel_l2, cosine)` of tq-`bits` attention against exact f32 attention.
fn drift(bits: u8, d: usize, n: usize, anisotropy: f32, seed: u64) -> (f32, f32) {
    let mut rng = Rng(seed);
    // A shared low-rank component makes the key set ANISOTROPIC: every key is a
    // mix of a few dominant directions plus its own noise. anisotropy=0 gives
    // isotropic Gaussian keys (the `sin`-noise regime); larger values
    // concentrate energy, which is what real attention keys do.
    let rank = 4;
    let basis: Vec<Vec<f32>> = (0..rank)
        .map(|_| (0..d).map(|_| rng.normal()).collect())
        .collect();
    let mk = |rng: &mut Rng| -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
        for b in &basis {
            let c = rng.normal() * anisotropy;
            for i in 0..d {
                v[i] += c * b[i];
            }
        }
        v
    };
    let keys: Vec<Vec<f32>> = (0..n).map(|_| mk(&mut rng)).collect();
    let vals: Vec<Vec<f32>> = (0..n).map(|_| mk(&mut rng)).collect();
    let q: Vec<f32> = (0..d).map(|_| rng.normal()).collect();

    let exact = attend(&q, &keys, &vals, d);
    let kq = round_trip(&keys, d, bits);
    let vq = round_trip(&vals, d, bits);
    let got = attend(&q, &kq, &vq, d);

    let diff: Vec<f32> = got.iter().zip(&exact).map(|(a, b)| a - b).collect();
    let rel = l2(&diff) / l2(&exact).max(1e-9);
    let cos =
        got.iter().zip(&exact).map(|(a, b)| a * b).sum::<f32>() / (l2(&got) * l2(&exact)).max(1e-9);
    (rel, cos)
}

/// MONOTONICITY, measured on the CODEC — per-vector round-trip cosine — and not
/// on attention output.
///
/// The first version of this test asserted monotonicity of attention-output
/// rel-L2 and FAILED: tq4 scored 1.005 against tq3's 0.759. That looked like a
/// codec defect and is not one. `diagnose_where_the_tq_error_comes_from` below
/// shows the codec is cleanly monotonic at every anisotropy level (cosine
/// 0.9828 at 3 bits -> 0.9953 at 4). What is NOT monotonic is what a softmax
/// does with the result, for a reason that is structural rather than numerical:
/// at high anisotropy the scores span ±23, so softmax is nearly a hard argmax.
/// Perturb the top scores by a fraction of a percent and a DIFFERENT key wins;
/// the output then jumps to a different value vector entirely. Which way it
/// jumps is chance, so a finer quantizer can land further from the f32 answer
/// than a coarser one on any single draw.
///
/// So: assert monotonicity where it must hold (the codec), and MEASURE the
/// attention-level behaviour separately instead of asserting a property it does
/// not have.
#[test]
fn tq_codec_error_decreases_with_bits() {
    let d = 256usize;
    let mut rng = Rng(12345);
    let keys: Vec<Vec<f32>> = (0..128)
        .map(|_| (0..d).map(|_| rng.normal()).collect())
        .collect();
    let mut prev = 0.0f32;
    for bits in [2u8, 3, 4] {
        let kq = round_trip(&keys, d, bits);
        let mean_cos = keys
            .iter()
            .zip(&kq)
            .map(|(a, b)| {
                a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (l2(a) * l2(b)).max(1e-9)
            })
            .sum::<f32>()
            / keys.len() as f32;
        assert!(
            mean_cos >= prev - 1e-4,
            "{bits}-bit round-trip cosine {mean_cos} is WORSE than the lower-bit {prev} — \
             more bits must never reconstruct worse"
        );
        prev = mean_cos;
    }
    // 4 bits must actually be good, not merely better than 2.
    assert!(prev > 0.99, "tq4 round-trip cosine {prev} is too low");
}

/// THE QUESTION A GPU PARITY GATE RAISED: is tq4's large attention rel-L2 a
/// property of tq4, or of the isotropic `sin`-noise the parity check feeds it?
///
/// ANSWER, measured: the opposite of the guess. Isotropic input is the EASY case
/// (rel-L2 ~0.17, cosine ~0.986); concentrated/anisotropic input is the hard one
/// (~0.79, cosine ~0.926) — even though the codec reconstructs both equally well
/// (per-vector cosine ~0.995 either way). The difference is entirely downstream:
/// concentrated keys produce a wide score range, a peaked softmax, and therefore
/// sensitivity to small score perturbations.
///
/// This is recorded as a MEASUREMENT, not a pass/fail on the codec, because the
/// direction is now known and the useful thing is the number. The assertion only
/// pins the finding itself, so a future change that reverses it gets noticed.
#[test]
fn tq4_attention_drift_is_larger_on_concentrated_keys() {
    let (d, n) = (256usize, 128);
    let (mut iso_sum, mut ani_sum) = (0.0, 0.0);
    let trials = 8;
    for t in 0..trials {
        let seed = 9001 + t * 7919;
        let (iso, iso_cos) = drift(4, d, n, 0.0, seed);
        let (ani, ani_cos) = drift(4, d, n, 3.0, seed);
        if t == 0 {
            println!("  tq4, head_dim {d}, {n} keys:");
            println!("    isotropic   rel-L2 {iso:.4}  cosine {iso_cos:.5}");
            println!("    anisotropic rel-L2 {ani:.4}  cosine {ani_cos:.5}");
        }
        iso_sum += iso;
        ani_sum += ani;
    }
    let (iso, ani) = (iso_sum / trials as f32, ani_sum / trials as f32);
    println!("  mean over {trials} trials: isotropic {iso:.4}, anisotropic {ani:.4}");
    assert!(
        ani > iso,
        "the recorded finding is that CONCENTRATED keys drift more ({ani} vs {iso}); \
         if that has reversed, the note above needs rewriting"
    );
}

/// DIAGNOSTIC for the two failures above: is the codec at fault, or the way this
/// test builds "anisotropic" vectors?
///
/// Reports the per-VECTOR round-trip cosine (the codec's own job) separately
/// from the attention-output drift (which composes the codec with a softmax).
/// If per-vector cosine is high while attention drift is large, the codec is
/// fine and the amplifier is the softmax: tiny score perturbations reweight an
/// almost-degenerate attention distribution.
#[test]
fn diagnose_where_the_tq_error_comes_from() {
    let d = 256usize;
    for &aniso in &[0.0f32, 1.0, 3.0] {
        let mut rng = Rng(4242);
        let rank = 4;
        let basis: Vec<Vec<f32>> = (0..rank)
            .map(|_| (0..d).map(|_| rng.normal()).collect())
            .collect();
        let mk = |rng: &mut Rng| -> Vec<f32> {
            let mut v: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
            for b in &basis {
                let c = rng.normal() * aniso;
                for i in 0..d {
                    v[i] += c * b[i];
                }
            }
            v
        };
        let keys: Vec<Vec<f32>> = (0..128).map(|_| mk(&mut rng)).collect();
        for bits in [3u8, 4] {
            let kq = round_trip(&keys, d, bits);
            let mut cos_min = f32::INFINITY;
            let mut cos_mean = 0.0;
            for (a, b) in keys.iter().zip(&kq) {
                let c =
                    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (l2(a) * l2(b)).max(1e-9);
                cos_min = cos_min.min(c);
                cos_mean += c;
            }
            cos_mean /= keys.len() as f32;
            // Score SPREAD: how distinguishable the keys are under this query.
            // A narrow spread means softmax is near-uniform and any perturbation
            // reorders it — the amplifier, if there is one.
            let q: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
            let scale = 1.0 / (d as f32).sqrt();
            let sc: Vec<f32> = keys
                .iter()
                .map(|k| q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale)
                .collect();
            let smin = sc.iter().cloned().fold(f32::INFINITY, f32::min);
            let smax = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!(
                "  aniso {aniso:.1} tq{bits}: per-vector cosine mean {cos_mean:.5} \
                 min {cos_min:.5} | key-norm mean {:.2} | score range [{smin:.2}, {smax:.2}]",
                keys.iter().map(|k| l2(k)).sum::<f32>() / keys.len() as f32
            );
        }
    }
}
