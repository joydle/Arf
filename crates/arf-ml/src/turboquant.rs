//! TurboQuant KV-cache quantization core: the optimal scalar codebook and the
//! random rotation it relies on.
//!
//! TurboQuant ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) compresses a
//! KV vector by (1) normalizing it to unit length (the norm is stored separately),
//! (2) applying a fixed random orthogonal rotation `R`, and (3) scalar-quantizing
//! each rotated coordinate with the Lloyd-Max-optimal quantizer for that
//! coordinate's distribution. The rotation is the trick: a random rotation maps
//! any unit vector to one whose coordinates are (for large `d`) i.i.d. from a
//! known density — the marginal of a uniform point on the unit sphere `S^{d-1}`,
//! `f(x) ∝ (1 − x²)^((d−3)/2)` on `[−1, 1]` (a symmetric Beta in disguise:
//! `(x+1)/2 ~ Beta((d−1)/2, (d−1)/2)`). Outliers get spread across all
//! coordinates, so one fixed per-coordinate quantizer is near-optimal — no
//! calibration, no training (the engine just needs the levels for its `head_dim`).
//!
//! This module is backend-agnostic CPU math: the codebook + rotation are the
//! oracle the GPU kernels are later checked against.

// This module is dense index-based linear algebra (matvec, Gram-Schmidt, the
// Hadamard butterfly, bit packing) where explicit `for i in 0..d` loops mirror
// the math and read more clearly than iterator-zip chains.
#![allow(clippy::needless_range_loop)]

/// Number of points on the `[-1, 1]` integration grid used to compute the
/// Lloyd-Max levels and the analytic MSE. Fine enough to resolve the density's
/// `~1/√d` concentration for `d` up to a few hundred; `f64` accumulation keeps
/// `(1−x²)^((d−3)/2)` from underflowing away the useful mass.
const GRID: usize = 16_384;

/// Lloyd-Max iterations. The sphere-coordinate density is log-concave and
/// symmetric, so Lloyd's algorithm converges to the global optimum quickly.
const LLOYD_ITERS: usize = 200;

/// The unnormalized sphere-coordinate density `f(x) ∝ (1 − x²)^((d−3)/2)` for a
/// single coordinate of a uniform unit vector in `R^d`. Returns `0` outside
/// `[-1, 1]`. `f64` throughout: the exponent reaches ~125 for `d = 256`, so the
/// tails underflow (correctly — the mass concentrates near 0).
fn sphere_coord_density(x: f64, d: usize) -> f64 {
    if !(-1.0..=1.0).contains(&x) {
        return 0.0;
    }
    let exponent = (d as f64 - 3.0) / 2.0;
    let base = 1.0 - x * x;
    if base <= 0.0 {
        // x = ±1: density is 0 for d > 3, and a point mass we ignore for d = 3.
        return 0.0;
    }
    base.powf(exponent)
}

/// `(grid_x[j], weight[j])`: the integration grid over `[-1, 1]` and the density
/// mass at each node (midpoint rule). Shared by the level fit and the MSE.
fn density_grid(d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::with_capacity(GRID);
    let mut ws = Vec::with_capacity(GRID);
    let step = 2.0 / GRID as f64;
    for j in 0..GRID {
        let x = -1.0 + (j as f64 + 0.5) * step; // cell midpoint
        xs.push(x);
        ws.push(sphere_coord_density(x, d) * step);
    }
    (xs, ws)
}

/// Compute the Lloyd-Max optimal `2^bits`-level scalar quantizer for the
/// `d`-dimensional sphere-coordinate density. Deterministic (grid-based, no RNG):
/// alternate (a) assign each grid node to its nearest level, (b) move each level
/// to the density-weighted centroid of its cell, until convergence. Symmetric
/// init + symmetric density ⇒ symmetric levels (no zero level for even counts —
/// optimal for a symmetric density).
fn lloyd_max_levels(bits: u8, d: usize) -> Vec<f32> {
    assert!((1..=8).contains(&bits), "bits must be in 1..=8");
    assert!(d >= 4, "sphere-coordinate density needs d >= 4");
    let n_levels = 1usize << bits;
    let (xs, ws) = density_grid(d);

    // Initialize levels spread over the effective support (std ≈ 1/√d).
    let sigma = 1.0 / (d as f64).sqrt();
    let mut levels: Vec<f64> = (0..n_levels)
        .map(|i| {
            // Uniform in [-3σ, 3σ]; clamped to [-1, 1].
            let t = (i as f64 + 0.5) / n_levels as f64; // (0,1)
            ((t - 0.5) * 6.0 * sigma).clamp(-1.0, 1.0)
        })
        .collect();

    for _ in 0..LLOYD_ITERS {
        let mut sum = vec![0.0f64; n_levels];
        let mut cnt = vec![0.0f64; n_levels];
        // Boundaries are implied by level midpoints; assign each node to the
        // nearest level via the sorted-level midpoint search.
        for (&x, &w) in xs.iter().zip(&ws) {
            if w == 0.0 {
                continue;
            }
            let idx = nearest_level(&levels, x);
            sum[idx] += w * x;
            cnt[idx] += w;
        }
        let mut moved = 0.0f64;
        for i in 0..n_levels {
            if cnt[i] > 0.0 {
                let new = sum[i] / cnt[i];
                moved += (new - levels[i]).abs();
                levels[i] = new;
            }
        }
        if moved < 1e-12 {
            break;
        }
    }

    levels.iter().map(|&l| l as f32).collect()
}

/// Index of the level nearest to `x` (levels sorted ascending). Branch-light
/// linear scan with early exit at the first midpoint `x` falls below — fine for
/// `n_levels ≤ 256` and used both in the fit loop and at quantize time.
#[inline]
fn nearest_level(levels: &[f64], x: f64) -> usize {
    for i in 0..levels.len() - 1 {
        let mid = 0.5 * (levels[i] + levels[i + 1]);
        if x < mid {
            return i;
        }
    }
    levels.len() - 1
}

/// A Lloyd-Max optimal scalar quantizer for the sphere-coordinate (Beta) density
/// of a given `(bits, head_dim)`. Quantizes a normalized rotated coordinate to a
/// `bits`-wide code and reconstructs it. Built once (cheaply) at model load.
#[derive(Debug, Clone)]
pub struct Codebook {
    levels: Vec<f32>, // length 2^bits, ascending
    bits: u8,
    d: usize,
}

impl Codebook {
    /// Fit the optimal `2^bits`-level codebook for `head_dim = d`.
    pub fn new(bits: u8, d: usize) -> Self {
        let levels = lloyd_max_levels(bits, d);
        Codebook { levels, bits, d }
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn head_dim(&self) -> usize {
        self.d
    }

    /// The reconstruction levels (length `2^bits`, ascending).
    pub fn levels(&self) -> &[f32] {
        &self.levels
    }

    /// Quantize a normalized coordinate to its nearest level's code.
    #[inline]
    pub fn quantize(&self, x: f32) -> u8 {
        // Reuse the f64 nearest-level search at f32 inputs (levels are f32).
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (i, &l) in self.levels.iter().enumerate() {
            let dd = (x - l).abs();
            if dd < best_d {
                best_d = dd;
                best = i;
            }
        }
        best as u8
    }

    /// Reconstruct the level for a code.
    #[inline]
    pub fn dequantize(&self, code: u8) -> f32 {
        self.levels[code as usize]
    }

    /// Analytic per-coordinate quantization MSE `∫ (x − q(x))² f(x) dx` under the
    /// sphere-coordinate density. The per-*vector* MSE for a unit vector is
    /// `head_dim ×` this (the paper's reported quantity).
    pub fn per_coord_mse(&self) -> f64 {
        let (xs, ws) = density_grid(self.d);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        let levels64: Vec<f64> = self.levels.iter().map(|&l| l as f64).collect();
        for (&x, &w) in xs.iter().zip(&ws) {
            if w == 0.0 {
                continue;
            }
            let q = levels64[nearest_level(&levels64, x)];
            num += w * (x - q) * (x - q);
            den += w;
        }
        num / den
    }
}

/// A packed set of TurboQuant-quantized vectors (one KV head-vector each): per
/// vector, a `bits`-wide code per coordinate plus a single bf16 norm. The codes
/// are the rotated **unit** vector's coordinates through the [`Codebook`]; the
/// norm restores the scale. Reconstruction (`to_f32`) inverse-rotates and rescales
/// — the CPU oracle the GPU pool is checked against.
///
/// Layout: codes are a flat `bits`-wide bitstream over `n_vectors · head_dim`
/// coordinates (LSB-first, straddling `u32` boundaries as needed); norms are one
/// bf16 per vector. For the supported models `head_dim ∈ {64,128,256}` so
/// `head_dim·bits` is a multiple of 32 and every vector is incidentally
/// word-aligned (which the GPU read relies on), but the packer does not require
/// it — `pack_code` clears-then-sets each code's exact bits, so a single-vector
/// overwrite (a decode step rewriting one slot) is correct even when vectors
/// share a word (tiny `head_dim`).
#[derive(Debug, Clone)]
pub struct TqKvBlock {
    codes: Vec<u32>, // bits-packed codes over [n_vectors * d] coordinates
    norms: Vec<u16>, // [n_vectors] bf16 norms
    n_vectors: usize,
    d: usize,
    bits: u8,
}

impl TqKvBlock {
    /// Allocate a zeroed block holding `n_vectors` vectors of dim `d`. All norms
    /// start 0 (a zeroed slot reconstructs to the zero vector).
    pub fn zeros(n_vectors: usize, d: usize, bits: u8) -> Self {
        assert!((1..=8).contains(&bits), "bits must be in 1..=8");
        let words = (n_vectors * d * bits as usize).div_ceil(32);
        TqKvBlock {
            codes: vec![0u32; words],
            norms: vec![0u16; n_vectors],
            n_vectors,
            d,
            bits,
        }
    }

    /// Quantize `n_vectors` contiguous `d`-vectors (`raw` is `[n_vectors * d]`).
    pub fn from_f32_vectors(
        raw: &[f32],
        n_vectors: usize,
        d: usize,
        bits: u8,
        cb: &Codebook,
        rot: &Rotation,
    ) -> Self {
        assert_eq!(raw.len(), n_vectors * d, "TqKvBlock shape mismatch");
        assert_eq!(cb.bits(), bits, "codebook bits mismatch");
        assert_eq!(rot.dim(), d, "rotation dim mismatch");
        let mut blk = Self::zeros(n_vectors, d, bits);
        for v in 0..n_vectors {
            blk.set_vector(v, &raw[v * d..(v + 1) * d], cb, rot);
        }
        blk
    }

    /// Quantize one `d`-vector into slot `v` (the decode-step write path). Rewrites
    /// exactly that vector's words (they are word-aligned), so an overwrite is
    /// clean — no stale bits from a previous occupant.
    pub fn set_vector(&mut self, v: usize, x: &[f32], cb: &Codebook, rot: &Rotation) {
        debug_assert_eq!(x.len(), self.d);
        let norm = x.iter().map(|a| a * a).sum::<f32>().sqrt();
        self.norms[v] = crate::dtype::f32_to_bf16(norm);
        // Normalize to unit, rotate, quantize each coordinate.
        let mut y = vec![0.0f32; self.d];
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for i in 0..self.d {
                y[i] = x[i] * inv;
            }
            rot.apply(&mut y);
        }
        // Pack codes (clear-then-set per code, so an overwrite leaves no stale bits).
        let base_bit = v * self.d * self.bits as usize;
        for i in 0..self.d {
            let code = if norm > 0.0 { cb.quantize(y[i]) } else { 0 } as u32;
            self.pack_code(base_bit + i * self.bits as usize, code);
        }
    }

    /// Clear the `bits` at `bitpos` and write `code` there (handles a `u32`
    /// straddle). Clear-then-set so the same slot can be overwritten in place.
    #[inline]
    fn pack_code(&mut self, bitpos: usize, code: u32) {
        let word = bitpos / 32;
        let off = bitpos % 32;
        let bits = self.bits as usize;
        let mask = (1u32 << bits) - 1;
        let masked = code & mask;
        self.codes[word] = (self.codes[word] & !(mask << off)) | (masked << off);
        if off + bits > 32 {
            let hi_mask = mask >> (32 - off);
            self.codes[word + 1] = (self.codes[word + 1] & !hi_mask) | (masked >> (32 - off));
        }
    }

    /// The code for coordinate `coord` of vector `v`.
    #[inline]
    pub fn code_at(&self, v: usize, coord: usize) -> u8 {
        let bitpos = v * self.d * self.bits as usize + coord * self.bits as usize;
        let word = bitpos / 32;
        let off = bitpos % 32;
        let bits = self.bits as usize;
        let mask = (1u32 << bits) - 1;
        let mut val = self.codes[word] >> off;
        if off + bits > 32 {
            val |= self.codes[word + 1] << (32 - off);
        }
        (val & mask) as u8
    }

    /// The stored (bf16-rounded) norm of vector `v`.
    #[inline]
    pub fn norm_at(&self, v: usize) -> f32 {
        crate::dtype::bf16_to_f32(self.norms[v])
    }

    pub fn n_vectors(&self) -> usize {
        self.n_vectors
    }
    pub fn head_dim(&self) -> usize {
        self.d
    }
    pub fn bits(&self) -> u8 {
        self.bits
    }
    /// Packed codes (for GPU upload).
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }
    /// bf16 norms (for GPU upload).
    pub fn norms(&self) -> &[u16] {
        &self.norms
    }

    /// Reconstruct vector `v` into `out` (`[d]`) in **rotated (R) space**, scaled
    /// by the norm but NOT inverse-rotated — i.e. `norm · ŷ_q`. This is what the
    /// attention score/output path consumes (it dots in R-space and inverse-rotates
    /// once at the end), avoiding a per-vector `Rᵀ`.
    pub fn dequant_rotated_into(&self, v: usize, cb: &Codebook, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.d);
        let norm = self.norm_at(v);
        for i in 0..self.d {
            out[i] = norm * cb.dequantize(self.code_at(v, i));
        }
    }

    /// Reconstruct all vectors to f32 (`[n_vectors * d]`), fully inverse-rotated
    /// and rescaled — the oracle. `out[v] ≈ x[v]`.
    pub fn to_f32(&self, cb: &Codebook, rot: &Rotation) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_vectors * self.d];
        let mut y = vec![0.0f32; self.d];
        for v in 0..self.n_vectors {
            self.dequant_rotated_into(v, cb, &mut y);
            rot.apply_transpose(&mut y); // norm · Rᵀ ŷ_q ≈ x
            out[v * self.d..(v + 1) * self.d].copy_from_slice(&y);
        }
        out
    }
}

/// Which rotation to use for a `head_dim`. `Hadamard` (fast, `O(d log d)`, no
/// stored matrix) when `head_dim` is a power of two — true for every supported
/// model (64/128/256); `Dense` (a stored `d×d` orthogonal matrix, `O(d²)`) is the
/// general oracle and the fallback for a non-power-of-two `head_dim`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationKind {
    Dense,
    Hadamard,
}

/// A fixed orthogonal rotation applied to each KV head-vector before quantization
/// (and its transpose to the attention output). Orthogonality is what makes the
/// quantization near-optimal AND lets attention dot in rotated space without ever
/// rotating keys back (`q·k = (Rq)·(Rk)`).
#[derive(Debug, Clone)]
pub enum Rotation {
    Dense(DenseRotation),
    Hadamard(Hadamard),
}

impl Rotation {
    /// Build the rotation for `head_dim`. `Hadamard` requires a power-of-two `d`;
    /// `Dense` works for any `d` (seeded, reproducible).
    pub fn new(d: usize, seed: u64, kind: RotationKind) -> Self {
        match kind {
            RotationKind::Dense => Rotation::Dense(DenseRotation::new(d, seed)),
            RotationKind::Hadamard => Rotation::Hadamard(Hadamard::new(d)),
        }
    }

    /// The default rotation for `head_dim`: Hadamard when it is a power of two
    /// (the fast path, all current models), else a seeded dense rotation.
    pub fn default_for(d: usize, seed: u64) -> Self {
        if d.is_power_of_two() {
            Rotation::Hadamard(Hadamard::new(d))
        } else {
            Rotation::Dense(DenseRotation::new(d, seed))
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Rotation::Dense(r) => r.dim(),
            Rotation::Hadamard(h) => h.dim(),
        }
    }

    /// Rotate `v` in place: `v ← R v`.
    #[inline]
    pub fn apply(&self, v: &mut [f32]) {
        match self {
            Rotation::Dense(r) => r.apply(v),
            Rotation::Hadamard(h) => h.apply(v),
        }
    }

    /// Inverse-rotate `v` in place: `v ← Rᵀ v` (= `R⁻¹ v`, since `R` is orthogonal).
    #[inline]
    pub fn apply_transpose(&self, v: &mut [f32]) {
        match self {
            Rotation::Dense(r) => r.apply_transpose(v),
            Rotation::Hadamard(h) => h.apply_transpose(v),
        }
    }
}

/// A dense `d×d` orthogonal matrix, seeded and reproducible. The oracle rotation:
/// correct for any `d`, `O(d²)` per apply. Generated by orthonormalizing a matrix
/// of seeded Gaussians (modified Gram-Schmidt — numerically stable).
#[derive(Debug, Clone)]
pub struct DenseRotation {
    r: Vec<f32>, // row-major [d, d]; row i is the i-th orthonormal basis vector
    d: usize,
}

impl DenseRotation {
    pub fn new(d: usize, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        // Standard-normal entries via Box-Muller from uniform (0,1).
        let mut gauss = || -> f64 {
            let u1: f64 = rng.random::<f64>().max(1e-12);
            let u2: f64 = rng.random::<f64>();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        let mut rows: Vec<Vec<f64>> = (0..d).map(|_| (0..d).map(|_| gauss()).collect()).collect();
        // Modified Gram-Schmidt: orthonormalize the rows.
        for i in 0..d {
            for j in 0..i {
                // project row i off the already-orthonormal row j
                let dot: f64 = (0..d).map(|k| rows[i][k] * rows[j][k]).sum();
                for k in 0..d {
                    rows[i][k] -= dot * rows[j][k];
                }
            }
            let norm: f64 = rows[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            let inv = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
            for k in 0..d {
                rows[i][k] *= inv;
            }
        }
        let r: Vec<f32> = rows
            .into_iter()
            .flat_map(|row| row.into_iter().map(|x| x as f32))
            .collect();
        DenseRotation { r, d }
    }

    pub fn dim(&self) -> usize {
        self.d
    }

    /// `v ← R v`: `out[i] = Σ_j R[i][j] v[j]`.
    pub fn apply(&self, v: &mut [f32]) {
        assert_eq!(v.len(), self.d, "DenseRotation::apply dim mismatch");
        let mut out = vec![0.0f32; self.d];
        for i in 0..self.d {
            let row = &self.r[i * self.d..(i + 1) * self.d];
            let mut acc = 0.0f32;
            for j in 0..self.d {
                acc += row[j] * v[j];
            }
            out[i] = acc;
        }
        v.copy_from_slice(&out);
    }

    /// `v ← Rᵀ v`: `out[i] = Σ_j R[j][i] v[j]`.
    pub fn apply_transpose(&self, v: &mut [f32]) {
        assert_eq!(
            v.len(),
            self.d,
            "DenseRotation::apply_transpose dim mismatch"
        );
        let mut out = vec![0.0f32; self.d];
        for j in 0..self.d {
            let row = &self.r[j * self.d..(j + 1) * self.d];
            let vj = v[j];
            for i in 0..self.d {
                out[i] += row[i] * vj;
            }
        }
        v.copy_from_slice(&out);
    }

    /// `R Rᵀ` as a flat `[d, d]` (for the orthogonality test): should be `I`.
    #[cfg(test)]
    fn gram(&self) -> Vec<f32> {
        let mut g = vec![0.0f32; self.d * self.d];
        for i in 0..self.d {
            for j in 0..self.d {
                let mut acc = 0.0f32;
                for k in 0..self.d {
                    acc += self.r[i * self.d + k] * self.r[j * self.d + k];
                }
                g[i * self.d + j] = acc;
            }
        }
        g
    }
}

/// A normalized Walsh-Hadamard rotation: `H = (1/√d) · WHT`. Symmetric AND
/// orthogonal (`H = Hᵀ`, `HᵀH = I`), so `apply == apply_transpose` and the
/// in-place butterfly costs `O(d log d)` with NO stored matrix — the perf path.
/// Requires `d` to be a power of two.
#[derive(Debug, Clone)]
pub struct Hadamard {
    d: usize,
    scale: f32,
}

impl Hadamard {
    pub fn new(d: usize) -> Self {
        assert!(
            d.is_power_of_two(),
            "Hadamard rotation requires power-of-two dim, got {d}"
        );
        Hadamard {
            d,
            scale: 1.0 / (d as f32).sqrt(),
        }
    }

    pub fn dim(&self) -> usize {
        self.d
    }

    /// In-place normalized Walsh-Hadamard transform. Self-inverse up to the
    /// normalization we already fold in, so this is both `apply` and its transpose.
    pub fn apply(&self, v: &mut [f32]) {
        assert_eq!(v.len(), self.d, "Hadamard dim mismatch");
        let mut h = 1usize;
        while h < self.d {
            let mut i = 0usize;
            while i < self.d {
                for j in i..i + h {
                    let x = v[j];
                    let y = v[j + h];
                    v[j] = x + y;
                    v[j + h] = x - y;
                }
                i += h * 2;
            }
            h *= 2;
        }
        for x in v.iter_mut() {
            *x *= self.scale;
        }
    }

    /// `H = Hᵀ`, so the transpose is the same transform.
    #[inline]
    pub fn apply_transpose(&self, v: &mut [f32]) {
        self.apply(v);
    }
}

/// A fixed Quantized Johnson-Lindenstrauss projection (TurboQuant Algorithm 2):
/// the **1-bit residual** stage that de-biases inner-product estimates left over
/// after the scalar quantizer.
///
/// After the MSE quantizer reconstructs `ŷ_q ≈ k̂` (the unit rotated key), the
/// residual `r = k̂ − ŷ_q` still carries inner-product signal. QJL stores only
/// `sign(S·r)` (one bit per projection row) plus `‖r‖`, and recovers an
/// **unbiased** estimate of `⟨q, r⟩` via
/// `⟨q,r⟩ ≈ ‖r‖ · √(π/2) · (1/m) · Σ_i sign(⟨s_i, r⟩)·⟨s_i, q⟩`.
/// The estimator is exact-in-expectation for a standard-Gaussian `S` (the
/// `E[sign(g)·g] = √(2/π)` identity); we use a seeded Gaussian `S` for that
/// reason — Rademacher (`±1`) rows are only asymptotically unbiased and would
/// need an empirically fitted constant.
///
/// Default OFF in the engine (the MSE path is the shipping default); this is the
/// research-complete arm with its own unbiasedness guarantee.
#[derive(Debug, Clone)]
pub struct QjlProjection {
    s: Vec<f32>, // [m, d] row-major, standard-normal entries
    m: usize,
    d: usize,
}

impl QjlProjection {
    /// `m` projection rows over `d`-vectors, seeded.
    pub fn new(m: usize, d: usize, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let s: Vec<f32> = (0..m * d)
            .map(|_| {
                let u1: f64 = rng.random::<f64>().max(1e-12);
                let u2: f64 = rng.random::<f64>();
                ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
            })
            .collect();
        QjlProjection { s, m, d }
    }

    pub fn rows(&self) -> usize {
        self.m
    }
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Packed sign bits of `S·r` (`m` bits, LSB-first). Bit set ⇒ `⟨s_i, r⟩ ≥ 0`.
    pub fn sign_sketch(&self, r: &[f32]) -> Vec<u32> {
        assert_eq!(r.len(), self.d, "QJL sketch dim mismatch");
        let words = self.m.div_ceil(32);
        let mut out = vec![0u32; words];
        for i in 0..self.m {
            let row = &self.s[i * self.d..(i + 1) * self.d];
            let mut acc = 0.0f32;
            for j in 0..self.d {
                acc += row[j] * r[j];
            }
            if acc >= 0.0 {
                out[i / 32] |= 1u32 << (i % 32);
            }
        }
        out
    }

    /// `S·q` (length `m`) — computed once per query and reused across all keys.
    pub fn project_query(&self, q: &[f32]) -> Vec<f32> {
        assert_eq!(q.len(), self.d, "QJL project dim mismatch");
        let mut out = vec![0.0f32; self.m];
        for i in 0..self.m {
            let row = &self.s[i * self.d..(i + 1) * self.d];
            let mut acc = 0.0f32;
            for j in 0..self.d {
                acc += row[j] * q[j];
            }
            out[i] = acc;
        }
        out
    }

    /// Unbiased estimate of `⟨q, r⟩` from the sign sketch, `‖r‖`, and a
    /// precomputed `S·q` (`sq`).
    pub fn estimate_inner_with(&self, sketch: &[u32], r_norm: f32, sq: &[f32]) -> f32 {
        debug_assert_eq!(sq.len(), self.m);
        let mut acc = 0.0f32;
        for i in 0..self.m {
            let bit = (sketch[i / 32] >> (i % 32)) & 1;
            let sign = if bit == 1 { 1.0 } else { -1.0 };
            acc += sign * sq[i];
        }
        const SQRT_PI_OVER_2: f32 = 1.253_314_1; // √(π/2)
        r_norm * SQRT_PI_OVER_2 * acc / self.m as f32
    }

    /// Convenience: project `q` then estimate `⟨q, r⟩`.
    pub fn estimate_inner(&self, sketch: &[u32], r_norm: f32, q: &[f32]) -> f32 {
        let sq = self.project_query(q);
        self.estimate_inner_with(sketch, r_norm, &sq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn rand_vec(d: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..d)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 40) as f32 / (1u64 << 23) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn levels_are_sorted_and_symmetric() {
        for bits in [2u8, 3, 4] {
            let cb = Codebook::new(bits, 128);
            let lv = cb.levels();
            assert_eq!(lv.len(), 1 << bits);
            // Ascending.
            for w in lv.windows(2) {
                assert!(w[0] < w[1], "levels not ascending: {lv:?}");
            }
            // Symmetric about 0 (even level count, symmetric density).
            for i in 0..lv.len() / 2 {
                let lo = lv[i];
                let hi = lv[lv.len() - 1 - i];
                assert!((lo + hi).abs() < 1e-3, "levels not symmetric: {lo} vs {hi}");
            }
        }
    }

    #[test]
    fn mse_within_paper_bound() {
        // Paper (unit vectors): vector MSE ≤ ~0.009 @4-bit, ~0.117 @2-bit. The
        // per-coordinate MSE × head_dim is that quantity. Allow 1.5× slack for the
        // grid discretization + the finite-d vs asymptotic distribution.
        let d = 128usize;
        let bound = |bits: u8| -> f64 {
            match bits {
                4 => 0.009,
                3 => 0.036,
                2 => 0.117,
                _ => unreachable!(),
            }
        };
        for bits in [2u8, 3, 4] {
            let cb = Codebook::new(bits, d);
            let vec_mse = cb.per_coord_mse() * d as f64;
            assert!(
                vec_mse <= bound(bits) * 1.5,
                "{bits}-bit vector MSE {vec_mse} over bound {}",
                bound(bits)
            );
        }
    }

    #[test]
    fn more_bits_lower_mse() {
        let d = 128usize;
        let m2 = Codebook::new(2, d).per_coord_mse();
        let m3 = Codebook::new(3, d).per_coord_mse();
        let m4 = Codebook::new(4, d).per_coord_mse();
        assert!(m3 < m2, "3-bit ({m3}) should beat 2-bit ({m2})");
        assert!(m4 < m3, "4-bit ({m4}) should beat 3-bit ({m3})");
    }

    #[test]
    fn quantize_dequantize_is_nearest_level() {
        let cb = Codebook::new(4, 64);
        // A coordinate exactly on a level round-trips to that level.
        for (code, &l) in cb.levels().iter().enumerate() {
            assert_eq!(cb.quantize(l), code as u8, "level {l} should self-encode");
            assert_eq!(cb.dequantize(code as u8), l);
        }
        // A value between two levels picks the closer one.
        let lv = cb.levels();
        let just_above = lv[0] + (lv[1] - lv[0]) * 0.49;
        assert_eq!(cb.quantize(just_above), 0);
        let just_below = lv[1] - (lv[1] - lv[0]) * 0.49;
        assert_eq!(cb.quantize(just_below), 1);
    }

    #[test]
    fn levels_concentrate_with_dimension() {
        // Higher d → coordinates concentrate (std ≈ 1/√d) → smaller level spread.
        let spread = |d: usize| {
            let cb = Codebook::new(4, d);
            let lv = cb.levels();
            lv[lv.len() - 1] - lv[0]
        };
        assert!(
            spread(256) < spread(64),
            "d=256 spread {} should be < d=64 spread {}",
            spread(256),
            spread(64)
        );
    }

    #[test]
    fn dense_rotation_is_orthogonal() {
        let r = DenseRotation::new(64, 7);
        // R Rᵀ = I.
        let g = r.gram();
        for i in 0..64 {
            for j in 0..64 {
                let want = if i == j { 1.0 } else { 0.0 };
                let got = g[i * 64 + j];
                assert!((got - want).abs() < 1e-4, "gram[{i}][{j}] = {got}");
            }
        }
    }

    #[test]
    fn dense_rotation_preserves_norm_and_inverts() {
        let r = DenseRotation::new(48, 11);
        let x = rand_vec(48, 3);
        let n0 = l2(&x);
        let mut y = x.clone();
        r.apply(&mut y);
        assert!(
            (l2(&y) - n0).abs() < 1e-3,
            "norm changed: {} vs {n0}",
            l2(&y)
        );
        r.apply_transpose(&mut y); // Rᵀ R x = x
        for (a, b) in y.iter().zip(&x) {
            assert!((a - b).abs() < 1e-3, "RᵀR not identity: {a} vs {b}");
        }
    }

    #[test]
    fn dense_rotation_preserves_inner_products() {
        // The trick attention relies on: (Rq)·(Rk) == q·k.
        let r = DenseRotation::new(64, 5);
        let (q, k) = (rand_vec(64, 1), rand_vec(64, 2));
        let (mut rq, mut rk) = (q.clone(), k.clone());
        r.apply(&mut rq);
        r.apply(&mut rk);
        assert!(
            (dot(&rq, &rk) - dot(&q, &k)).abs() < 1e-3,
            "inner product not preserved: {} vs {}",
            dot(&rq, &rk),
            dot(&q, &k)
        );
    }

    #[test]
    fn hadamard_is_orthonormal_and_self_inverse() {
        let h = Hadamard::new(128);
        let x = rand_vec(128, 9);
        let n0 = l2(&x);
        let mut y = x.clone();
        h.apply(&mut y);
        assert!(
            (l2(&y) - n0).abs() < 1e-3,
            "norm changed: {} vs {n0}",
            l2(&y)
        );
        h.apply_transpose(&mut y); // H Hᵀ = H H = I
        for (a, b) in y.iter().zip(&x) {
            assert!((a - b).abs() < 1e-4, "HH not identity: {a} vs {b}");
        }
    }

    #[test]
    fn hadamard_preserves_inner_products() {
        let h = Hadamard::new(64);
        let (q, k) = (rand_vec(64, 1), rand_vec(64, 2));
        let (mut rq, mut rk) = (q.clone(), k.clone());
        h.apply(&mut rq);
        h.apply(&mut rk);
        assert!(
            (dot(&rq, &rk) - dot(&q, &k)).abs() < 1e-3,
            "Hadamard inner product not preserved: {} vs {}",
            dot(&rq, &rk),
            dot(&q, &k)
        );
    }

    #[test]
    fn rotation_enum_dispatches() {
        let had = Rotation::default_for(64, 0); // pow2 → Hadamard
        assert!(matches!(had, Rotation::Hadamard(_)));
        let dense = Rotation::default_for(48, 0); // non-pow2 → Dense
        assert!(matches!(dense, Rotation::Dense(_)));
        // Round-trips through the enum too.
        let mut v = rand_vec(64, 4);
        let orig = v.clone();
        had.apply(&mut v);
        had.apply_transpose(&mut v);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    #[should_panic(expected = "power-of-two")]
    fn hadamard_rejects_non_pow2() {
        let _ = Hadamard::new(48);
    }

    // Gaussian vectors: normalized, they are exactly uniform on the sphere, so
    // they match the codebook's target distribution.
    fn gaussian_vectors(n: usize, d: usize, seed: u64) -> Vec<f32> {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * d)
            .map(|_| {
                let u1: f64 = rng.random::<f64>().max(1e-12);
                let u2: f64 = rng.random::<f64>();
                ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
            })
            .collect()
    }

    #[test]
    fn tqkv_roundtrip_within_mse_bound() {
        let (n, d) = (256usize, 64usize);
        let raw = gaussian_vectors(n, d, 42);
        let rot = Rotation::default_for(d, 0);
        let bound = |bits: u8| match bits {
            4 => 0.009,
            3 => 0.036,
            2 => 0.117,
            _ => unreachable!(),
        };
        for bits in [2u8, 3, 4] {
            let cb = Codebook::new(bits, d);
            let blk = TqKvBlock::from_f32_vectors(&raw, n, d, bits, &cb, &rot);
            let recon = blk.to_f32(&cb, &rot);
            let mut err = 0.0f64;
            let mut sig = 0.0f64;
            for i in 0..raw.len() {
                let e = (recon[i] - raw[i]) as f64;
                err += e * e;
                sig += (raw[i] as f64) * (raw[i] as f64);
            }
            let rel = err / sig;
            assert!(
                rel <= bound(bits) * 2.0,
                "{bits}-bit relative reconstruction MSE {rel} over bound {}",
                bound(bits)
            );
        }
    }

    #[test]
    fn tqkv_more_bits_better_reconstruction() {
        let (n, d) = (128usize, 128usize);
        let raw = gaussian_vectors(n, d, 7);
        let rot = Rotation::default_for(d, 0);
        let rel = |bits: u8| {
            let cb = Codebook::new(bits, d);
            let blk = TqKvBlock::from_f32_vectors(&raw, n, d, bits, &cb, &rot);
            let recon = blk.to_f32(&cb, &rot);
            let mut err = 0.0f64;
            let mut sig = 0.0f64;
            for i in 0..raw.len() {
                let e = (recon[i] - raw[i]) as f64;
                err += e * e;
                sig += (raw[i] as f64) * (raw[i] as f64);
            }
            err / sig
        };
        assert!(rel(4) < rel(3) && rel(3) < rel(2));
    }

    #[test]
    fn tqkv_zero_vector_reconstructs_zero() {
        let (n, d) = (4usize, 64usize);
        let raw = vec![0.0f32; n * d];
        let cb = Codebook::new(4, d);
        let rot = Rotation::default_for(d, 0);
        let blk = TqKvBlock::from_f32_vectors(&raw, n, d, 4, &cb, &rot);
        assert!(blk.to_f32(&cb, &rot).iter().all(|&x| x == 0.0));
        assert_eq!(blk.norm_at(0), 0.0);
    }

    #[test]
    fn tqkv_set_vector_overwrite_is_clean() {
        // Writing a slot twice leaves no stale bits from the first occupant.
        let d = 64usize;
        let cb = Codebook::new(3, d); // 3-bit: the straddle-prone width
        let rot = Rotation::default_for(d, 0);
        let first = gaussian_vectors(1, d, 1);
        let second = gaussian_vectors(1, d, 2);
        let mut blk = TqKvBlock::zeros(2, d, 3);
        blk.set_vector(0, &first, &cb, &rot);
        blk.set_vector(0, &second, &cb, &rot); // overwrite
                                               // Slot 0 must now match a fresh block built from `second` alone.
        let fresh = TqKvBlock::from_f32_vectors(&second, 1, d, 3, &cb, &rot);
        for i in 0..d {
            assert_eq!(blk.code_at(0, i), fresh.code_at(0, i), "stale bit at {i}");
        }
        // And slot 1 (never written) stays zero.
        assert_eq!(blk.norm_at(1), 0.0);
    }

    #[test]
    fn tqkv_is_memory_compressed() {
        let (n, d) = (100usize, 128usize);
        let f32_bytes = n * d * 4;
        for bits in [2u8, 3, 4] {
            let blk = TqKvBlock::zeros(n, d, bits);
            let packed = blk.codes().len() * 4 + blk.norms().len() * 2;
            // Codes alone should be ~ bits/32 of f32; allow the small norm overhead.
            let ratio = f32_bytes as f64 / packed as f64;
            let expected = 32.0 / bits as f64;
            assert!(
                ratio > expected * 0.9,
                "{bits}-bit compression {ratio:.1}x below expected ~{expected:.1}x"
            );
        }
    }

    #[test]
    fn qjl_sketch_packs_one_bit_per_row() {
        let proj = QjlProjection::new(70, 64, 1);
        let r = gaussian_vectors(1, 64, 3);
        let sketch = proj.sign_sketch(&r);
        assert_eq!(sketch.len(), 70usize.div_ceil(32)); // 3 words for 70 bits
    }

    #[test]
    fn qjl_estimates_inner_product_unbiased() {
        // Over many random (q, r) pairs, the QJL estimate of ⟨q, r⟩ is unbiased
        // and tracks the true value (a large m drives the per-estimate variance
        // down so the mean signed error is tiny).
        let d = 64usize;
        let proj = QjlProjection::new(4096, d, 1234);
        let pairs = 200usize;
        let qs = gaussian_vectors(pairs, d, 11);
        let rs = gaussian_vectors(pairs, d, 22);
        let mut sum_signed = 0.0f64;
        let mut sum_abs_true = 0.0f64;
        let mut sum_abs_err = 0.0f64;
        for p in 0..pairs {
            let q = &qs[p * d..(p + 1) * d];
            let r = &rs[p * d..(p + 1) * d];
            let r_norm = l2(r);
            let sketch = proj.sign_sketch(r);
            let est = proj.estimate_inner(&sketch, r_norm, q);
            let truth = dot(q, r);
            sum_signed += (est - truth) as f64;
            sum_abs_true += truth.abs() as f64;
            sum_abs_err += (est - truth).abs() as f64;
        }
        let mean_signed = sum_signed / pairs as f64;
        let mean_abs_true = sum_abs_true / pairs as f64;
        // Unbiased: mean signed error ≪ typical magnitude.
        assert!(
            mean_signed.abs() < 0.1 * mean_abs_true,
            "QJL biased: mean signed err {mean_signed} vs scale {mean_abs_true}"
        );
        // And the average estimate is in the right ballpark of the truth.
        let mean_abs_err = sum_abs_err / pairs as f64;
        assert!(
            mean_abs_err < 0.6 * mean_abs_true,
            "QJL too noisy: mean abs err {mean_abs_err} vs scale {mean_abs_true}"
        );
    }

    #[test]
    fn qjl_residual_reduces_inner_product_bias() {
        // The point of QJL: ⟨q', k̂⟩ ≈ ⟨q', ŷ_q⟩ + QJL(residual) is *less biased*
        // than the MSE-only ⟨q', ŷ_q⟩. Quantization shrinks reconstructed norms,
        // so ⟨q', ŷ_q⟩ systematically under-estimates ⟨q', k̂⟩; adding the QJL
        // residual term should shrink that systematic gap.
        let d = 128usize;
        let bits = 3u8;
        let cb = Codebook::new(bits, d);
        let rot = Rotation::default_for(d, 0);
        let proj = QjlProjection::new(2048, d, 7);
        let n = 300usize;
        let ks = gaussian_vectors(n, d, 100);
        let qs = gaussian_vectors(n, d, 200);

        let mut mse_only_bias = 0.0f64;
        let mut with_qjl_bias = 0.0f64;
        let mut yq = vec![0.0f32; d];
        for p in 0..n {
            let k = &ks[p * d..(p + 1) * d];
            let q = &qs[p * d..(p + 1) * d];
            // Rotate q the same way keys are rotated; dot in R-space (the trick).
            let mut qr = q.to_vec();
            rot.apply(&mut qr);
            // Unit rotated key k̂ and its quantized reconstruction ŷ_q.
            let kn = l2(k);
            let mut khat = k.to_vec();
            for x in khat.iter_mut() {
                *x /= kn;
            }
            rot.apply(&mut khat); // k̂ = R(k/‖k‖)
            for i in 0..d {
                yq[i] = cb.dequantize(cb.quantize(khat[i]));
            }
            // residual r = k̂ − ŷ_q.
            let mut r = vec![0.0f32; d];
            for i in 0..d {
                r[i] = khat[i] - yq[i];
            }
            let r_norm = l2(&r);
            let sketch = proj.sign_sketch(&r);

            let truth = dot(&qr, &khat); // exact ⟨q', k̂⟩
            let mse_only = dot(&qr, &yq);
            let qjl_term = if r_norm > 0.0 {
                proj.estimate_inner(&sketch, r_norm, &qr)
            } else {
                0.0
            };
            mse_only_bias += (mse_only - truth) as f64;
            with_qjl_bias += (mse_only + qjl_term - truth) as f64;
        }
        let mse_only_bias = (mse_only_bias / n as f64).abs();
        let with_qjl_bias = (with_qjl_bias / n as f64).abs();
        assert!(
            with_qjl_bias < mse_only_bias,
            "QJL did not reduce bias: with {with_qjl_bias} vs mse-only {mse_only_bias}"
        );
    }

    #[test]
    fn tqkv_dequant_rotated_then_inverse_matches_to_f32() {
        // dequant_rotated_into + Rᵀ must equal to_f32 (the two consumer paths agree).
        let (n, d) = (3usize, 64usize);
        let raw = gaussian_vectors(n, d, 5);
        let cb = Codebook::new(4, d);
        let rot = Rotation::default_for(d, 0);
        let blk = TqKvBlock::from_f32_vectors(&raw, n, d, 4, &cb, &rot);
        let full = blk.to_f32(&cb, &rot);
        let mut y = vec![0.0f32; d];
        for v in 0..n {
            blk.dequant_rotated_into(v, &cb, &mut y);
            rot.apply_transpose(&mut y);
            for i in 0..d {
                assert!((y[i] - full[v * d + i]).abs() < 1e-6);
            }
        }
    }
}
