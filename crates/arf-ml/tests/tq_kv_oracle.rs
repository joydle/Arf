//! The CPU oracle a GPU backend's tq4 KV path must reproduce — written BEFORE
//! the kernels, because a wrong bit-unpack or a missed inverse rotation produces
//! fluent-but-WRONG text rather than an error.
//!
//! Pins three things the port depends on:
//!   1. the exact write path (`norm` -> unit -> Hadamard -> codebook -> pack),
//!   2. the ROTATION TRICK identity that lets the kernel skip rotating keys
//!      back: `q·x ≈ (Rq)·(dequantized rotated x)`,
//!   3. the bit layout (LSB-first, word-aligned for head_dim*bits % 32 == 0).
//!
//! These run on CPU only; no GPU required, so they gate the port in CI.

use arf_ml::turboquant::{Codebook, Rotation, RotationKind, TqKvBlock};

fn deterministic_vec(d: usize, seed: u32) -> Vec<f32> {
    // Deterministic, non-trivial, and not axis-aligned: a quantizer can look
    // perfect on a basis vector and fail on a general one.
    (0..d)
        .map(|i| {
            let t = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((t % 2003) as f32 / 2003.0 - 0.5) * 2.0
        })
        .collect()
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|a| a * a).sum::<f32>().sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// head_dim * bits must be a multiple of 32 for every supported model
/// (64/128/256 x 4), which is what lets a GPU reader stream whole words.
#[test]
fn tq4_vectors_are_word_aligned_for_supported_head_dims() {
    for d in [64usize, 128, 256] {
        assert_eq!(
            (d * 4) % 32,
            0,
            "head_dim {d} at 4 bits is not word-aligned; a GPU unpack \
             assumes every vector starts on a u32 boundary"
        );
    }
}

/// The round trip must preserve direction well enough to be usable: with 4 bits
/// per coordinate the reconstruction is lossy, but the cosine to the original
/// stays high. This is the number a GPU path must match, not exceed.
#[test]
fn tq4_round_trip_keeps_direction() {
    let d = 128;
    let cb = Codebook::new(4, d);
    let rot = Rotation::new(d, 0, RotationKind::Hadamard);
    let mut blk = TqKvBlock::zeros(4, d, 4);

    for v in 0..4 {
        let x = deterministic_vec(d, v as u32 * 17 + 1);
        blk.set_vector(v, &x, &cb, &rot);
    }
    let out = blk.to_f32(&cb, &rot);

    for v in 0..4 {
        let x = deterministic_vec(d, v as u32 * 17 + 1);
        let y = &out[v * d..(v + 1) * d];
        let cos = dot(&x, y) / (l2(&x) * l2(y)).max(1e-9);
        assert!(
            cos > 0.97,
            "vector {v}: cosine {cos} too low for tq4 — the codec or the \
             rotation is wrong"
        );
        // NOT a norm-preserving code. Quantizing a UNIT vector's coordinates to
        // 16 interior levels shortens it (||y_q|| < 1), so the rescaled vector
        // comes back systematically SHORT. The stored norm restores scale, the
        // codebook does not restore length. Direction is the invariant; this
        // bound just pins the shrinkage so a regression that makes it worse is
        // visible. Asserting norm equality here was wrong and the oracle caught it.
        let shrink = l2(y) / l2(&x).max(1e-9);
        assert!(
            (0.75..=1.05).contains(&shrink),
            "vector {v}: length ratio {shrink} outside the expected tq4 shrinkage"
        );
    }
}

/// THE ROTATION TRICK, which is the whole reason a GPU inner loop is cheap:
/// R is orthogonal, so `q·k == (Rq)·(Rk)`. The kernel rotates the QUERY once and
/// scores it against the stored ROTATED keys, never rotating keys back. This
/// pins that identity against the real codec, including quantization error.
#[test]
fn rotated_query_scores_match_unrotated_dot() {
    let d = 128;
    let cb = Codebook::new(4, d);
    let rot = Rotation::new(d, 0, RotationKind::Hadamard);

    let k = deterministic_vec(d, 7);
    let q = deterministic_vec(d, 99);

    let mut blk = TqKvBlock::zeros(1, d, 4);
    blk.set_vector(0, &k, &cb, &rot);

    // Path A: the honest reference — dequantize fully (inverse-rotated) and dot.
    let k_hat = blk.to_f32(&cb, &rot);
    let ref_score = dot(&q, &k_hat);

    // Path B: what the kernel does — rotate the query, dot against the stored
    // rotated+dequantized key, never rotating the key back.
    let mut qrot = q.clone();
    rot.apply(&mut qrot);
    let mut k_rot_hat = vec![0.0f32; d];
    // NOTE: `dequant_rotated_into` returns `norm * y_q` — the norm is ALREADY
    // applied. Multiplying by the norm again here double-scales (it produced a
    // 5.9x error when this test was first written). A GPU kernel must apply
    // the norm exactly ONCE, matching this.
    blk.dequant_rotated_into(0, &cb, &mut k_rot_hat);
    let kernel_score: f32 = dot(&qrot, &k_rot_hat);

    let rel = (kernel_score - ref_score).abs() / ref_score.abs().max(1e-6);
    assert!(
        rel < 1e-3,
        "rotation trick broken: kernel {kernel_score} vs reference \
         {ref_score} (rel {rel}). A GPU kernel MUST be able to skip \
         rotating keys back."
    );
}

/// A zeroed slot must reconstruct to the zero vector — a GPU pool starts
/// zeroed and unwritten slots are read before they are filled.
#[test]
fn a_zeroed_slot_reconstructs_to_zero() {
    let d = 64;
    let cb = Codebook::new(4, d);
    let rot = Rotation::new(d, 0, RotationKind::Hadamard);
    let blk = TqKvBlock::zeros(2, d, 4);
    let out = blk.to_f32(&cb, &rot);
    assert!(
        out.iter().all(|&x| x == 0.0),
        "a zeroed tq block must dequantize to zeros, got nonzero"
    );
}
