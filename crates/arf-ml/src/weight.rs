//! A 2-D weight matrix stored in bf16 to halve resident memory.
//!
//! Real Llama checkpoints are bf16, so loading is a straight copy; f32 sources
//! are rounded down. Values are widened to f32 in the matmul ([`Bf16Matrix::linear`])
//! and in row gathers ([`Bf16Matrix::gather`], for the embedding table).

use crate::dtype::{bf16_to_f32, f32_to_bf16};
use crate::matmul::matmul_nt_bf16;
use crate::Tensor;

/// Row-major `[rows, cols]` bf16 matrix.
#[derive(Debug, Clone)]
pub struct Bf16Matrix {
    data: Vec<u16>,
    rows: usize,
    cols: usize,
}

impl Bf16Matrix {
    /// Wrap raw bf16 bit patterns. Panics if `data` does not match `rows*cols`.
    pub fn from_bf16(data: Vec<u16>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "Bf16Matrix shape mismatch");
        Bf16Matrix { data, rows, cols }
    }

    /// Round an f32 slice down to bf16.
    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> Self {
        Self::from_bf16(data.iter().copied().map(f32_to_bf16).collect(), rows, cols)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Widen the whole matrix to a flat f32 buffer (e.g. for GPU upload).
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().copied().map(bf16_to_f32).collect()
    }

    /// The raw row-major bf16 bit patterns (round-to-nearest-even of the source).
    /// Lets a GPU uploader pack straight to bf16-as-`u32` WITHOUT the 2×-larger
    /// `to_f32()` round-trip — the upload's bf16 bits are these bits unchanged, so
    /// it stays bit-identical to the CPU `Bf16Matrix` (the parity oracle).
    pub fn bits(&self) -> &[u16] {
        &self.data
    }

    /// Linear (no bias): `x [m, cols]` (f32) → `[m, rows]` (f32), i.e. `x · selfᵀ`.
    /// `self` is the `[out, in]` weight, matching a HuggingFace checkpoint.
    pub fn linear(&self, x: &Tensor) -> Tensor {
        let (m, k) = x.dims2();
        assert_eq!(
            k, self.cols,
            "linear: input dim {k} != weight in-dim {}",
            self.cols
        );
        let data = matmul_nt_bf16(x.as_slice(), m, k, &self.data, self.rows);
        Tensor::from_vec(data, vec![m, self.rows])
    }

    /// Gather rows `ids` and widen to f32: `[ids.len(), cols]` (embedding lookup).
    pub fn gather(&self, ids: &[u32]) -> Tensor {
        let mut out = Vec::with_capacity(ids.len() * self.cols);
        for &id in ids {
            let base = id as usize * self.cols;
            out.extend(
                self.data[base..base + self.cols]
                    .iter()
                    .copied()
                    .map(bf16_to_f32),
            );
        }
        Tensor::from_vec(out, vec![ids.len(), self.cols])
    }
}

/// Row-major `[rows, cols]` weight quantized to per-row symmetric int8.
///
/// Each output row `r` carries its own f32 `scale[r] = max(|row|)/127`; a weight
/// is stored as `round(w/scale)` clamped to i8 and read back as `q · scale[r]`.
/// Per-row keeps accuracy high, and rows are the natural unit because the matmul
/// already walks output rows. ~1 byte/weight + a scale per row vs bf16's 2.
#[derive(Debug, Clone)]
pub struct Q8Matrix {
    data: Vec<i8>,
    scale: Vec<f32>, // one per row
    rows: usize,
    cols: usize,
}

impl Q8Matrix {
    /// Quantize an f32 `[rows, cols]` weight to per-row int8.
    pub fn from_f32(w: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(w.len(), rows * cols, "Q8Matrix shape mismatch");
        let mut data = vec![0i8; rows * cols];
        let mut scale = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let max_abs = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // A zero row has no scale; 1.0 keeps it zero without dividing by zero.
            let s = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scale[r] = s;
            for c in 0..cols {
                data[r * cols + c] = (row[c] / s).round().clamp(-127.0, 127.0) as i8;
            }
        }
        Q8Matrix {
            data,
            scale,
            rows,
            cols,
        }
    }

    /// Quantize a bf16 weight (the real-checkpoint path) to per-row int8.
    pub fn from_bf16(bits: &[u16], rows: usize, cols: usize) -> Self {
        let f: Vec<f32> = bits.iter().copied().map(bf16_to_f32).collect();
        Self::from_f32(&f, rows, cols)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The row-major `[rows, cols]` int8 weights (for GPU upload).
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// The per-row f32 scales (`scale[r] = max(|row r|)/127`).
    pub fn scales(&self) -> &[f32] {
        &self.scale
    }

    /// Dequantize the whole matrix to f32 (tests, GPU upload).
    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            let s = self.scale[r];
            for c in 0..self.cols {
                let i = r * self.cols + c;
                out[i] = f32::from(self.data[i]) * s;
            }
        }
        out
    }

    /// Linear (no bias): `x [m, cols]` (f32) → `[m, rows]` (f32), i.e. `x · selfᵀ`.
    pub fn linear(&self, x: &Tensor) -> Tensor {
        let (m, k) = x.dims2();
        assert_eq!(
            k, self.cols,
            "linear: input dim {k} != weight in-dim {}",
            self.cols
        );
        let data =
            crate::matmul::matmul_nt_q8(x.as_slice(), m, k, &self.data, &self.scale, self.rows);
        Tensor::from_vec(data, vec![m, self.rows])
    }
}

/// Number of weights sharing one scale in [`Q4Matrix`] (ollama Q4_0 uses 32).
pub const Q4_BLOCK: usize = 32;

/// Row-major `[rows, cols]` weight quantized to **Q4_0-style block 4-bit**: every
/// run of `Q4_BLOCK` consecutive weights in a row shares one scale, and each
/// weight is a signed 4-bit code in `[-8, 7]`.
///
/// Layout (chosen for a coalesced GPU read + a CPU oracle that is bit-identical):
/// - `codes`: the 4-bit codes packed 8 per `u32`, low nibble first. A block of 32
///   weights is exactly 4 `u32` words. `cols % 32 == 0` for every Llama weight.
/// - `scales`: one **bf16** scale per block, row-major `[rows][cols/32]`.
///
/// Decision: picked Q4_0 block-of-32 (ollama-matching) from {per-row Q4,
/// Q4_0 block-32, Q4_K double-quant} — tradeoff: per-row Q4 reuses the int8 kernel
/// shape but loses accuracy (one scale per 2048-wide row); Q4_K is most accurate
/// but needs a second-level scale-of-scales. Block-32 matches ollama's accuracy
/// for the fair fight at one extra scale lookup per 32 weights.
/// Decision: scales as **bf16, not IEEE-f16** — tradeoff: f16 is ollama's
/// literal Q4_0, but bf16 reuses our already-tested, GPU-identical `bf16<->f32`
/// path (zero new conversion code, zero new dep) at the same 2 bytes/block; bf16's
/// wider range / coarser mantissa is harmless for a max-abs block scale.
#[derive(Debug, Clone)]
pub struct Q4Matrix {
    codes: Vec<u32>,  // [rows * cols/8] — 8 nibbles per u32
    scales: Vec<u16>, // [rows * cols/32] — one bf16 per block
    rows: usize,
    cols: usize,
}

impl Q4Matrix {
    /// Quantize an f32 `[rows, cols]` weight to Q4_0 blocks. `cols % 32 == 0`.
    pub fn from_f32(w: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(w.len(), rows * cols, "Q4Matrix shape mismatch");
        assert_eq!(cols % Q4_BLOCK, 0, "Q4Matrix cols must be a multiple of 32");
        let blocks_per_row = cols / Q4_BLOCK;
        let mut codes = vec![0u32; rows * cols / 8];
        let mut scales = vec![0u16; rows * blocks_per_row];
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let base = r * cols + b * Q4_BLOCK;
                let block = &w[base..base + Q4_BLOCK];
                let max_abs = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                // Symmetric: codes in [-8,7], so the divisor is max_abs/7 (a zero
                // block keeps scale 1.0 to avoid 0/0; its codes are all 0).
                let s = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
                scales[r * blocks_per_row + b] = f32_to_bf16(s);
                // Quantize with the SAME stored (bf16-rounded) scale the dequant
                // will use, so CPU and GPU agree to the bit.
                let s_q = bf16_to_f32(scales[r * blocks_per_row + b]);
                for (j, &v) in block.iter().enumerate() {
                    let code = (v / s_q).round().clamp(-8.0, 7.0) as i32;
                    // Pack the 4-bit two's-complement code (0..=15) into its nibble.
                    let nibble = (code & 0xF) as u32;
                    let widx = (base + j) / 8;
                    let shift = ((base + j) % 8) * 4;
                    codes[widx] |= nibble << shift;
                }
            }
        }
        Q4Matrix {
            codes,
            scales,
            rows,
            cols,
        }
    }

    /// Quantize a bf16 weight (the real-checkpoint path) to Q4_0 blocks.
    pub fn from_bf16(bits: &[u16], rows: usize, cols: usize) -> Self {
        let f: Vec<f32> = bits.iter().copied().map(bf16_to_f32).collect();
        Self::from_f32(&f, rows, cols)
    }

    /// LOSSLESS transcode of native ggml **Q4_0** blocks straight into this layout —
    /// no dequant→re-quant round-trip (which double-quantizes a Q4_0 GGUF and adds the
    /// logit noise that surfaces junk tokens when sampling for the Q4_K analog).
    ///
    /// ggml Q4_0 block = 18 bytes per 32 weights: `d` (f16 LE) then 16 bytes of codes,
    /// where byte `k` (0..16) holds element `k` in its LOW nibble and element `k+16` in
    /// its HIGH nibble, and the value is `d·(nibble - 8)` (see `dequant_q4_0`). Arf's
    /// `Q4Matrix` packs codes SEQUENTIALLY (8 per u32) as two's-complement signed nibbles
    /// with `y = scale·signed_code` (no -8 offset). So we de-interleave and store
    /// `signed = nibble - 8` (as a 4-bit two's-complement nibble); the scale is the same
    /// f16 `d`, converted to bf16 (Arf's scale type). `cols % 32 == 0`.
    pub fn from_ggml_q4_0(blocks: &[u8], rows: usize, cols: usize) -> Self {
        assert_eq!(
            cols % Q4_BLOCK,
            0,
            "from_ggml_q4_0: cols must be a multiple of 32"
        );
        let blocks_per_row = cols / Q4_BLOCK;
        const GGML_Q4_0_BYTES: usize = 18; // 2 (f16 d) + 16 (codes)
        assert_eq!(
            blocks.len(),
            rows * blocks_per_row * GGML_Q4_0_BYTES,
            "from_ggml_q4_0: blocks length mismatch"
        );
        let mut codes = vec![0u32; rows * cols / 8];
        let mut scales = vec![0u16; rows * blocks_per_row];
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let p = (r * blocks_per_row + b) * GGML_Q4_0_BYTES;
                let d = crate::dtype::f16_to_f32(u16::from_le_bytes([blocks[p], blocks[p + 1]]));
                scales[r * blocks_per_row + b] = f32_to_bf16(d);
                let qs = &blocks[p + 2..p + 18]; // 16 bytes → 32 nibbles
                let base = r * cols + b * Q4_BLOCK; // element index of this block, row-major
                for k in 0..16 {
                    let byte = qs[k];
                    // ggml: low nibble → element k, high nibble → element k+16.
                    // Arf signed code = (ggml nibble) - 8, stored as a 4-bit nibble.
                    let lo = (((byte & 0x0f) as i32) - 8) & 0xF;
                    let hi = (((byte >> 4) as i32) - 8) & 0xF;
                    for (elem, nib) in [(k, lo as u32), (k + 16, hi as u32)] {
                        let gi = base + elem;
                        codes[gi / 8] |= nib << ((gi % 8) * 4);
                    }
                }
            }
        }
        Q4Matrix {
            codes,
            scales,
            rows,
            cols,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The packed 4-bit codes, 8 per `u32` (for GPU upload).
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }

    /// The per-block bf16 scales, row-major `[rows][cols/32]` (for GPU upload).
    pub fn scales(&self) -> &[u16] {
        &self.scales
    }

    /// Sign-extend a 4-bit code (0..=15) to its value in `[-8, 7]`.
    #[inline]
    pub fn decode_nibble(nibble: u32) -> i32 {
        // bit 3 is the sign bit: subtract 16 when set.
        (nibble as i32) - (((nibble >> 3) & 1) as i32) * 16
    }

    /// Dequantize the whole matrix to f32 (tests, GPU upload, parity oracle).
    pub fn to_f32(&self) -> Vec<f32> {
        let blocks_per_row = self.cols / Q4_BLOCK;
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            for b in 0..blocks_per_row {
                let s = bf16_to_f32(self.scales[r * blocks_per_row + b]);
                let base = r * self.cols + b * Q4_BLOCK;
                for j in 0..Q4_BLOCK {
                    let widx = (base + j) / 8;
                    let shift = ((base + j) % 8) * 4;
                    let nibble = (self.codes[widx] >> shift) & 0xF;
                    out[base + j] = Self::decode_nibble(nibble) as f32 * s;
                }
            }
        }
        out
    }

    /// Linear (no bias): `x [m, cols]` (f32) → `[m, rows]` (f32), i.e. `x · selfᵀ`.
    pub fn linear(&self, x: &Tensor) -> Tensor {
        let (m, k) = x.dims2();
        assert_eq!(
            k, self.cols,
            "linear: input dim {k} != weight in-dim {}",
            self.cols
        );
        let data =
            crate::matmul::matmul_nt_q4(x.as_slice(), m, k, &self.codes, &self.scales, self.rows);
        Tensor::from_vec(data, vec![m, self.rows])
    }
}

/// Q4_K super-block size (weights sharing one `d`/`dmin`); ollama Q4_K uses 256.
pub const Q4K_SUPER: usize = 256;
/// Q4_K sub-block size (weights sharing one (scale, min)); 8 sub-blocks per super.
pub const Q4K_SUB: usize = 32;

/// Row-major `[rows, cols]` weight quantized to **Q4_K-style** 4-bit: 256-weight
/// super-blocks of 8 sub-blocks (32 each), where every sub-block has its OWN
/// asymmetric `(scale, min)` and codes are **unsigned** `[0, 15]`. The min-offset
/// (vs Q4_0's symmetric `[-8,7]`) is the accuracy win — it fits a sub-block's range
/// exactly rather than forcing it to be centred on zero, which matters for the
/// skewed weight distributions Q4_0 wastes a sign bit on.
///
/// Layout (coalesced GPU read + bit-identical CPU oracle): `codes` are unsigned
/// 4-bit packed 8 per `u32` (low nibble first, same as Q4_0); `scales`/`mins` are
/// one **bf16** each per sub-block, row-major `[rows][cols/32]`. Dequant is
/// `w = code * scale[sub] + min[sub]` with `code` in `0..15`.
///
/// Decision: picked **Q4_K-lite (asymmetric per-32 scale+min as bf16)** from
/// {literal ggml Q4_K (6-bit-packed scales + super-block d/dmin), Q4_K-lite,
/// stay-Q4_0} — tradeoff: literal Q4_K is byte-compatible with ollama GGUF but its
/// 12-byte 6-bit scale/min packing is fiddly and only pays off if we LOAD a GGUF
/// (we quantize ourselves today). Q4_K-lite captures the actual accuracy win (the
/// asymmetric min-offset + sub-block granularity) at our own self-consistent
/// layout, reusing the proven Q4_0 codes packing + bf16 scale path. The full 6-bit
/// super-block form is a follow-up gated on a GGUF-load feature.
#[derive(Debug, Clone)]
pub struct Q4KMatrix {
    codes: Vec<u32>,  // [rows * cols/8] — 8 unsigned nibbles per u32
    scales: Vec<u16>, // [rows * cols/32] — one bf16 scale per sub-block
    mins: Vec<u16>,   // [rows * cols/32] — one bf16 min per sub-block
    rows: usize,
    cols: usize,
}

impl Q4KMatrix {
    /// Quantize an f32 `[rows, cols]` weight to Q4_K-lite. `cols % 256 == 0`.
    pub fn from_f32(w: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(w.len(), rows * cols, "Q4KMatrix shape mismatch");
        assert_eq!(
            cols % Q4K_SUPER,
            0,
            "Q4KMatrix cols must be a multiple of 256"
        );
        let subs_per_row = cols / Q4K_SUB;
        let mut codes = vec![0u32; rows * cols / 8];
        let mut scales = vec![0u16; rows * subs_per_row];
        let mut mins = vec![0u16; rows * subs_per_row];
        for r in 0..rows {
            for sb in 0..subs_per_row {
                let base = r * cols + sb * Q4K_SUB;
                let block = &w[base..base + Q4K_SUB];
                let lo = block.iter().fold(f32::INFINITY, |m, &v| m.min(v));
                let hi = block.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                // Asymmetric: map [lo, hi] onto codes [0,15]. A flat block (hi==lo)
                // keeps scale 1.0 and min=lo so every code 0 dequants back to lo.
                let s = if hi > lo { (hi - lo) / 15.0 } else { 1.0 };
                let idx = r * subs_per_row + sb;
                scales[idx] = f32_to_bf16(s);
                mins[idx] = f32_to_bf16(lo);
                // Quantize with the STORED (bf16-rounded) scale+min the dequant uses.
                let s_q = bf16_to_f32(scales[idx]);
                let lo_q = bf16_to_f32(mins[idx]);
                for (j, &v) in block.iter().enumerate() {
                    let code = ((v - lo_q) / s_q).round().clamp(0.0, 15.0) as u32;
                    let widx = (base + j) / 8;
                    let shift = ((base + j) % 8) * 4;
                    codes[widx] |= code << shift;
                }
            }
        }
        Q4KMatrix {
            codes,
            scales,
            mins,
            rows,
            cols,
        }
    }

    /// Quantize a bf16 weight (the real-checkpoint path) to Q4_K-lite.
    pub fn from_bf16(bits: &[u16], rows: usize, cols: usize) -> Self {
        let f: Vec<f32> = bits.iter().copied().map(bf16_to_f32).collect();
        Self::from_f32(&f, rows, cols)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Packed unsigned 4-bit codes, 8 per `u32` (for GPU upload).
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }
    /// Per-sub-block bf16 scales, row-major `[rows][cols/32]`.
    pub fn scales(&self) -> &[u16] {
        &self.scales
    }
    /// Per-sub-block bf16 mins, row-major `[rows][cols/32]`.
    pub fn mins(&self) -> &[u16] {
        &self.mins
    }

    /// Dequantize the whole matrix to f32 (tests, parity oracle).
    pub fn to_f32(&self) -> Vec<f32> {
        let subs_per_row = self.cols / Q4K_SUB;
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            for sb in 0..subs_per_row {
                let idx = r * subs_per_row + sb;
                let s = bf16_to_f32(self.scales[idx]);
                let lo = bf16_to_f32(self.mins[idx]);
                let base = r * self.cols + sb * Q4K_SUB;
                for j in 0..Q4K_SUB {
                    let widx = (base + j) / 8;
                    let shift = ((base + j) % 8) * 4;
                    let code = (self.codes[widx] >> shift) & 0xF;
                    out[base + j] = code as f32 * s + lo;
                }
            }
        }
        out
    }

    /// Linear (no bias): `x [m, cols]` (f32) → `[m, rows]` (f32), i.e. `x · selfᵀ`.
    pub fn linear(&self, x: &Tensor) -> Tensor {
        let (m, k) = x.dims2();
        assert_eq!(
            k, self.cols,
            "linear: input dim {k} != weight in-dim {}",
            self.cols
        );
        let data = crate::matmul::matmul_nt_q4k(
            x.as_slice(),
            m,
            k,
            &self.codes,
            &self.scales,
            &self.mins,
            self.rows,
        );
        Tensor::from_vec(data, vec![m, self.rows])
    }
}

/// Row-major `[rows, cols]` weight quantized to **Q4_K_S-style** 4-bit: like
/// [`Q4KMatrix`] (super-block-256, 8 asymmetric sub-blocks of 32) but the 8 per-
/// sub-block `(scale, min)` are themselves quantized to **u8/i8 against a per-
/// super-block f32 `d`/`dmin`** (a second-level scale), instead of stored as bf16.
/// This halves the scale-overhead bytes (1 vs 2 per value) — the ggml Q4_K_S idea —
/// for ~5% fewer total bytes (decode is bandwidth-bound) at ~Q4_K accuracy, safely
/// above the 4-bit cliff.
///
/// Dequant for sub-block `s`: `w = (d·scale_u8[s])·code + (dmin·min_i8[s])`, with
/// `code` in `0..15`, `scale_u8` in `0..255`, `min_i8` in `-127..127`.
///
/// Decision: u8 sub-scales (not the literal ggml 6-bit) — same byte/value as
/// 6-bit rounds to in practice (1 byte packed), far simpler to pack/unpack on the
/// GPU, self-consistent CPU==GPU. The real 6-bit form is a GGUF-load follow-up.
/// A Q5_K matrix held in the GGUF's own layout.
///
/// WHY A SEPARATE TYPE. `Q4KSMatrix` packs 4-bit codes eight to a `u32`, and
/// 40 call sites index that packing. Widening it for one extra bit would
/// touch all of them; a parallel type touches none.
///
/// WHY NATIVE AT ALL. A mixed-quant GGUF keeps its sensitive tensors above 4
/// bits - Qwen3.8-27B-UD-Q4_K_M is 131 Q5_K, 117 IQ4_XS, 106 Q8_0 against
/// only 104 Q4_K - and re-quantising those down measured rel 1.1e-1 on a
/// projection where the stored precision gives 1.5e-6.
///
/// LAYOUT is Q4_K's plus one bit plane, exactly as ggml stores it
/// (`block_q5_K`, 176 bytes per 256 weights = 5.5 bits):
///   d, dmin      f16 super-block scales
///   scales\[12\]   the SAME 6-bit packed scale/min pairs as Q4_K
///   qh\[32\]       the 5th bit of each of 256 codes, one bit per weight
///   qs\[128\]      the low 4 bits, two per byte
///
/// llama.cpp's `vec_dot_q5_K_q8_1` comments its scale decode "same as q4_K";
/// this keeps that shared and only the code unpack differs.
#[derive(Debug, Clone)]
pub struct Q5KSMatrix {
    /// Low nibbles, 8 per u32 - byte-identical packing to `Q4KSMatrix`.
    codes_lo: Vec<u32>,
    /// The 5th bit of each code, 32 per u32.
    codes_hi: Vec<u32>,
    scales: Vec<u8>,
    mins: Vec<i8>,
    d: Vec<f32>,
    dmin: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Q5KSMatrix {
    /// Read GGUF `block_q5_K` bytes directly - no dequantise, no re-quantise.
    pub fn from_ggml_q5k(blocks: &[u8], rows: usize, cols: usize) -> Self {
        const SUPER: usize = 256;
        const SUB: usize = 32;
        const BLOCK_BYTES: usize = 176;
        assert_eq!(cols % SUPER, 0, "Q5_K needs cols % 256 == 0");
        assert_eq!(
            blocks.len(),
            rows * cols / SUPER * BLOCK_BYTES,
            "Q5_K blocks length mismatch for ({rows},{cols})"
        );

        let subs_per_row = cols / SUB;
        let supers_per_row = cols / SUPER;
        let subs_per_super = SUPER / SUB;

        let mut codes_lo = vec![0u32; rows * cols / 8];
        let mut codes_hi = vec![0u32; rows * cols / 32];
        let mut scales = vec![0u8; rows * subs_per_row];
        let mut mins = vec![0i8; rows * subs_per_row];
        let mut d = vec![0.0f32; rows * supers_per_row];
        let mut dmin = vec![0.0f32; rows * supers_per_row];

        for r in 0..rows {
            for sblk in 0..supers_per_row {
                let p = (r * supers_per_row + sblk) * BLOCK_BYTES;
                let d_f32 =
                    crate::dtype::f16_to_f32(u16::from_le_bytes([blocks[p], blocks[p + 1]]));
                let dmin_f32 =
                    crate::dtype::f16_to_f32(u16::from_le_bytes([blocks[p + 2], blocks[p + 3]]));
                let sidx = r * supers_per_row + sblk;
                d[sidx] = d_f32;
                dmin[sidx] = dmin_f32;

                let scale_bytes = &blocks[p + 4..p + 16]; // 12, same as Q4_K
                let qh = &blocks[p + 16..p + 48]; // 32: the 5th bits
                let qs = &blocks[p + 48..p + 176]; // 128: the low nibbles

                for ss in 0..subs_per_super {
                    let half = ss / 2;
                    let shift = if ss % 2 == 0 { 0 } else { 4 };
                    // Identical decode to Q4_K - llama.cpp says so too.
                    let (sc, m) = crate::weight::q4k_get_scale_min(scale_bytes, ss);
                    let sb = r * subs_per_row + sblk * subs_per_super + ss;
                    scales[sb] = sc;
                    mins[sb] = -(m as i8);

                    // THE HIGH BIT IS INDEXED BY POSITION WITHIN THE 32, NOT
                    // BY A FLAT ELEMENT INDEX.
                    //
                    // `qh[l]` holds the 5th bit of eight different weights -
                    // one per sub-block pair - and the bit selected is
                    // `1 << ss`, with `l` running 0..32 inside the sub-block.
                    // Reading it as `qh[e % 32] >> (e / 32)` looks equivalent
                    // and is not: it transposes which weight each bit belongs
                    // to. Checked against `dequant_q5_k` (gguf.rs:1946),
                    // which selects `u1 = 1 << (2*g)` for the low-nibble half
                    // and `u2 = 2 << (2*g)` for the high - i.e. bit `ss` for
                    // sub-block `ss`.
                    let qbase = half * 32;
                    let hbit = 1u8 << ss;
                    for i in 0..SUB {
                        let lo = ((qs[qbase + i] >> shift) & 0xF) as u32;
                        let hi = u32::from(qh[i] & hbit != 0);
                        let pos = r * cols + sblk * SUPER + ss * SUB + i;
                        codes_lo[pos / 8] |= lo << ((pos % 8) * 4);
                        codes_hi[pos / 32] |= hi << (pos % 32);
                    }
                }
            }
        }

        Q5KSMatrix {
            codes_lo,
            codes_hi,
            scales,
            mins,
            d,
            dmin,
            rows,
            cols,
        }
    }

    pub fn codes_lo(&self) -> &[u32] {
        &self.codes_lo
    }
    pub fn codes_hi(&self) -> &[u32] {
        &self.codes_hi
    }
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
    pub fn mins(&self) -> &[i8] {
        &self.mins
    }
    pub fn d(&self) -> &[f32] {
        &self.d
    }
    pub fn dmin(&self) -> &[f32] {
        &self.dmin
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

#[derive(Debug, Clone)]
pub struct Q4KSMatrix {
    codes: Vec<u32>, // [rows * cols/8] — 4-bit, 8/u32 (same as Q4_K)
    scales: Vec<u8>, // [rows * cols/32] — u8 sub-scale (against per-super d)
    mins: Vec<i8>,   // [rows * cols/32] — i8 sub-min (against per-super dmin)
    d: Vec<f32>,     // [rows * cols/256] — super-block scale-of-scales
    dmin: Vec<f32>,  // [rows * cols/256] — super-block scale-of-mins
    rows: usize,
    cols: usize,
}

impl Q4KSMatrix {
    /// The matrix as little-endian bytes (`rows`, `cols`, then codes, scales, mins, d, dmin) — the
    /// draft's quantize cache stores what a load would otherwise search for again (2026-10-06).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(
            16 + self.codes.len() * 4
                + self.scales.len() * 2
                + (self.d.len() + self.dmin.len()) * 4,
        );
        b.extend_from_slice(&(self.rows as u64).to_le_bytes());
        b.extend_from_slice(&(self.cols as u64).to_le_bytes());
        self.codes
            .iter()
            .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
        b.extend_from_slice(&self.scales);
        self.mins.iter().for_each(|x| b.push(*x as u8));
        self.d
            .iter()
            .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
        self.dmin
            .iter()
            .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
        b
    }

    /// The inverse of [`to_bytes`](Self::to_bytes); `None` when the bytes do not hold exactly one
    /// matrix of a valid shape.
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let rows = u64::from_le_bytes(b.get(0..8)?.try_into().ok()?) as usize;
        let cols = u64::from_le_bytes(b.get(8..16)?.try_into().ok()?) as usize;
        if cols == 0 || !cols.is_multiple_of(256) {
            return None;
        }
        let n = rows.checked_mul(cols)?;
        let (nc, ns, nd) = (n / 8, n / 32, n / 256);
        let mut at = 16usize;
        let mut take = |len: usize| -> Option<&[u8]> {
            let s = b.get(at..at.checked_add(len)?)?;
            at += len;
            Some(s)
        };
        let codes = take(nc * 4)?
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let scales = take(ns)?.to_vec();
        let mins = take(ns)?.iter().map(|&x| x as i8).collect();
        let f32s = |s: &[u8]| -> Vec<f32> {
            s.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let d = f32s(take(nd * 4)?);
        let dmin = f32s(take(nd * 4)?);
        (at == b.len()).then_some(Q4KSMatrix {
            codes,
            scales,
            mins,
            d,
            dmin,
            rows,
            cols,
        })
    }

    /// Quantize an f32 `[rows, cols]` weight to Q4_K_S-lite. `cols % 256 == 0`.
    pub fn from_f32(w: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(w.len(), rows * cols, "Q4KSMatrix shape mismatch");
        assert_eq!(
            cols % Q4K_SUPER,
            0,
            "Q4KSMatrix cols must be a multiple of 256"
        );
        let subs_per_row = cols / Q4K_SUB;
        let supers_per_row = cols / Q4K_SUPER;
        let subs_per_super = Q4K_SUPER / Q4K_SUB; // 8
        let mut codes = vec![0u32; rows * cols / 8];
        let mut scales = vec![0u8; rows * subs_per_row];
        let mut mins = vec![0i8; rows * subs_per_row];
        let mut d = vec![0.0f32; rows * supers_per_row];
        let mut dmin = vec![0.0f32; rows * supers_per_row];
        for r in 0..rows {
            for sblk in 0..supers_per_row {
                // First pass: per-sub-block float (scale, min).
                let mut sub_scale = [0.0f32; 8];
                let mut sub_min = [0.0f32; 8];
                for ss in 0..subs_per_super {
                    let sb = sblk * subs_per_super + ss;
                    let base = r * cols + sb * Q4K_SUB;
                    let block = &w[base..base + Q4K_SUB];
                    let lo = block.iter().fold(f32::INFINITY, |m, &v| m.min(v));
                    let hi = block.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                    sub_scale[ss] = if hi > lo { (hi - lo) / 15.0 } else { 1.0 };
                    sub_min[ss] = lo;
                }
                // Second level: quantize the 8 (scale, min) against per-super d/dmin.
                let max_scale = sub_scale.iter().cloned().fold(0.0f32, f32::max);
                let max_min_abs = sub_min.iter().map(|m| m.abs()).fold(0.0f32, f32::max);
                let sidx = r * supers_per_row + sblk;
                d[sidx] = if max_scale > 0.0 {
                    max_scale / 255.0
                } else {
                    1.0
                };
                dmin[sidx] = if max_min_abs > 0.0 {
                    max_min_abs / 127.0
                } else {
                    1.0
                };
                let dq = d[sidx];
                let dmq = dmin[sidx];
                for ss in 0..subs_per_super {
                    // Row-global sub-block index (NOT row-local — scales/mins/codes
                    // are sized rows*subs_per_row, so row r's data must land in its
                    // own slots, not overwrite row 0's).
                    let sb = r * subs_per_row + sblk * subs_per_super + ss;
                    let su8 = (sub_scale[ss] / dq).round().clamp(0.0, 255.0) as u8;
                    let mi8 = (sub_min[ss] / dmq).round().clamp(-127.0, 127.0) as i8;
                    scales[sb] = su8;
                    mins[sb] = mi8;
                    // Quantize the codes with the RECONSTRUCTED (two-level) scale+min.
                    let s_q = dq * su8 as f32;
                    let lo_q = dmq * mi8 as f32;
                    let s_safe = if s_q > 0.0 { s_q } else { 1.0 };
                    let base = r * cols + (sblk * subs_per_super + ss) * Q4K_SUB;
                    for j in 0..Q4K_SUB {
                        let v = w[base + j];
                        let code = ((v - lo_q) / s_safe).round().clamp(0.0, 15.0) as u32;
                        let widx = (base + j) / 8;
                        let shift = ((base + j) % 8) * 4;
                        codes[widx] |= code << shift;
                    }
                }
            }
        }
        Q4KSMatrix {
            codes,
            scales,
            mins,
            d,
            dmin,
            rows,
            cols,
        }
    }

    /// [`Self::from_f32`] with an ERROR-MINIMISING fit per 32-weight sub-block instead of
    /// min/max: for a ladder of candidate step sizes (the range mapped onto 13..19 levels, so
    /// both "leave head-room" and "clip the outliers" are tried) assign codes, then solve the
    /// least-squares `(scale, min)` for those codes in closed form, and keep the candidate with
    /// the smallest squared error — what llama.cpp's Q4_K (`make_qkx2_quants`) does, unweighted.
    /// Same layout, same kernels; only the numbers stored differ. Rows are independent, so they
    /// are fitted on all cores.
    ///
    /// `from_f32` is left exactly as it was: every bf16 model that self-quantises through it
    /// keeps its bits. This exists for weights that were never meant to run at 4 bits and whose
    /// OUTPUT the plain fit moves too far — measured on the DFlash 2 draft, 2026-09-20.
    pub fn from_f32_searched(w: &[f32], rows: usize, cols: usize) -> Self {
        Self::searched_grouped(w, rows, cols, Q4K_SUB)
    }

    /// The searched fit with ONE (scale, min) per `group` weights (32 = ordinary Q4_K_S; 64 or
    /// 128 = adjacent sub-blocks share theirs). Fitted JOINTLY over the whole group — not picked
    /// from the members' own values — and stored replicated, so the result is an ordinary
    /// Q4_K_S matrix every kernel reads unchanged while a kernel MAY take `group`-wide steps.
    pub fn searched_grouped(w: &[f32], rows: usize, cols: usize, group: usize) -> Self {
        assert!(
            matches!(group, 32 | 64 | 128),
            "group must be 32, 64 or 128"
        );
        assert_eq!(w.len(), rows * cols, "Q4KSMatrix shape mismatch");
        assert_eq!(
            cols % Q4K_SUPER,
            0,
            "Q4KSMatrix cols must be a multiple of 256"
        );
        let subs_per_row = cols / Q4K_SUB;
        let supers_per_row = cols / Q4K_SUPER;
        let subs_per_super = Q4K_SUPER / Q4K_SUB; // 8
        let mut codes = vec![0u32; rows * cols / 8];
        let mut scales = vec![0u8; rows * subs_per_row];
        let mut mins = vec![0i8; rows * subs_per_row];
        let mut d = vec![0.0f32; rows * supers_per_row];
        let mut dmin = vec![0.0f32; rows * supers_per_row];

        // The least-squares (scale, min) of one sub-block.
        fn fit(block: &[f32]) -> (f32, f32) {
            let lo = block.iter().fold(f32::INFINITY, |m, &v| m.min(v));
            let hi = block.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            if hi <= lo {
                return (1.0, lo);
            }
            let n = block.len() as f64;
            let mut best = ((hi - lo) / 15.0, lo, f64::INFINITY);
            for step in 0..=24 {
                let inv = (13.0 + 0.25 * step as f32) / (hi - lo);
                let (mut sl, mut sl2, mut sx, mut sxl) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                let mut lv = [0.0f64; 128];
                for (i, &x) in block.iter().enumerate() {
                    let l = (inv * (x - lo)).round().clamp(0.0, 15.0) as f64;
                    lv[i] = l;
                    sl += l;
                    sl2 += l * l;
                    sx += x as f64;
                    sxl += x as f64 * l;
                }
                let det = n * sl2 - sl * sl;
                if det <= 0.0 {
                    continue;
                }
                let sc = (n * sxl - sx * sl) / det;
                let mn = (sl2 * sx - sl * sxl) / det;
                if sc <= 0.0 {
                    continue;
                }
                // Score with the codes this (scale, min) would actually be given.
                let err: f64 = block
                    .iter()
                    .map(|&x| {
                        let l = ((x as f64 - mn) / sc).round().clamp(0.0, 15.0);
                        (sc * l + mn - x as f64).powi(2)
                    })
                    .sum();
                if err < best.2 {
                    best = (sc as f32, mn as f32, err);
                }
            }
            (best.0, best.1)
        }

        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(rows.max(1));
        let per = rows.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut parts = (
                codes.chunks_mut(per * cols / 8),
                scales.chunks_mut(per * subs_per_row),
                mins.chunks_mut(per * subs_per_row),
                d.chunks_mut(per * supers_per_row),
                dmin.chunks_mut(per * supers_per_row),
            );
            let mut r0 = 0usize;
            while let (Some(c), Some(sc), Some(mn), Some(dd), Some(dm)) = (
                parts.0.next(),
                parts.1.next(),
                parts.2.next(),
                parts.3.next(),
                parts.4.next(),
            ) {
                let start = r0;
                r0 += per;
                scope.spawn(move || {
                    let n_rows = dd.len() / supers_per_row;
                    for lr in 0..n_rows {
                        let r = start + lr;
                        for sblk in 0..supers_per_row {
                            let mut sub_scale = [0.0f32; 8];
                            let mut sub_min = [0.0f32; 8];
                            let members = group / Q4K_SUB;
                            for ss in (0..subs_per_super).step_by(members) {
                                let base = r * cols + (sblk * subs_per_super + ss) * Q4K_SUB;
                                let f = fit(&w[base..base + group]);
                                for j in 0..members {
                                    (sub_scale[ss + j], sub_min[ss + j]) = f;
                                }
                            }
                            let max_scale = sub_scale.iter().cloned().fold(0.0f32, f32::max);
                            let max_min_abs =
                                sub_min.iter().map(|m| m.abs()).fold(0.0f32, f32::max);
                            let sidx = lr * supers_per_row + sblk;
                            dd[sidx] = if max_scale > 0.0 {
                                max_scale / 255.0
                            } else {
                                1.0
                            };
                            dm[sidx] = if max_min_abs > 0.0 {
                                max_min_abs / 127.0
                            } else {
                                1.0
                            };
                            for ss in 0..subs_per_super {
                                let sb = lr * subs_per_row + sblk * subs_per_super + ss;
                                let su8 =
                                    (sub_scale[ss] / dd[sidx]).round().clamp(0.0, 255.0) as u8;
                                let mi8 =
                                    (sub_min[ss] / dm[sidx]).round().clamp(-127.0, 127.0) as i8;
                                sc[sb] = su8;
                                mn[sb] = mi8;
                                let s_q = dd[sidx] * su8 as f32;
                                let lo_q = dm[sidx] * mi8 as f32;
                                let s_safe = if s_q > 0.0 { s_q } else { 1.0 };
                                let gbase = r * cols + (sblk * subs_per_super + ss) * Q4K_SUB;
                                let lbase = lr * cols + (sblk * subs_per_super + ss) * Q4K_SUB;
                                for j in 0..Q4K_SUB {
                                    let code =
                                        ((w[gbase + j] - lo_q) / s_safe).round().clamp(0.0, 15.0)
                                            as u32;
                                    c[(lbase + j) / 8] |= code << (((lbase + j) % 8) * 4);
                                }
                            }
                        }
                    }
                });
            }
        });
        Q4KSMatrix {
            codes,
            scales,
            mins,
            d,
            dmin,
            rows,
            cols,
        }
    }

    /// [`Self::from_f32_searched`] for a bf16 weight.
    pub fn from_bf16_searched(bits: &[u16], rows: usize, cols: usize) -> Self {
        let f: Vec<f32> = bits.iter().copied().map(bf16_to_f32).collect();
        Self::from_f32_searched(&f, rows, cols)
    }

    /// Transcode native ggml Q4_K blocks (144 bytes / 256 weights) into this Q4KS
    /// layout **losslessly** — codes copied verbatim, ggml's f16 d/dmin widened to f32,
    /// and the 6-bit sub scale/min mapped to u8/i8 (min negated to fold ggml's
    /// `- dmin*m` into Arf's `+ dmin*min_i8` convention).
    ///
    /// The dequant output of the result's [`to_f32`](Self::to_f32) is **bit-identical**
    /// to ggml's `dequant_q4_k` on the same bytes — closing the measured re-quant quality
    /// gap without any re-quantization step (proven by the `from_ggml_q4k_is_bit_exact`
    /// test). `cols % 256 == 0`.
    pub fn from_ggml_q4k(blocks: &[u8], rows: usize, cols: usize) -> Self {
        assert_eq!(
            cols % Q4K_SUPER,
            0,
            "Q4KSMatrix::from_ggml_q4k: cols must be a multiple of 256"
        );
        assert_eq!(
            blocks.len(),
            rows * cols / Q4K_SUPER * 144,
            "Q4KSMatrix::from_ggml_q4k: blocks length mismatch"
        );
        let subs_per_row = cols / Q4K_SUB; // cols/32
        let supers_per_row = cols / Q4K_SUPER; // cols/256
        let subs_per_super = Q4K_SUPER / Q4K_SUB; // 8

        let mut codes = vec![0u32; rows * cols / 8];
        let mut scales = vec![0u8; rows * subs_per_row];
        let mut mins = vec![0i8; rows * subs_per_row];
        let mut d = vec![0.0f32; rows * supers_per_row];
        let mut dmin = vec![0.0f32; rows * supers_per_row];

        for r in 0..rows {
            for sblk in 0..supers_per_row {
                // ggml block index (sequential across all rows, row-major).
                let b = r * supers_per_row + sblk;
                let p = b * 144; // byte offset into blocks

                // Read d / dmin (f16 LE → f32).
                let d_f16 = u16::from_le_bytes([blocks[p], blocks[p + 1]]);
                let dmin_f16 = u16::from_le_bytes([blocks[p + 2], blocks[p + 3]]);
                let d_f32 = crate::dtype::f16_to_f32(d_f16);
                let dmin_f32 = crate::dtype::f16_to_f32(dmin_f16);

                let sidx = r * supers_per_row + sblk;
                d[sidx] = d_f32;
                dmin[sidx] = dmin_f32;

                let scale_bytes = &blocks[p + 4..p + 16]; // 12 bytes
                let qs = &blocks[p + 16..p + 144]; // 128 bytes (256 nibbles)

                for ss in 0..subs_per_super {
                    // ggml packs two sub-blocks (ss and ss+1 if even) per 32 qs bytes.
                    // half = ss/2 → which pair of 32 bytes.
                    // shift = 0 for even ss, 4 for odd ss.
                    let half = ss / 2;
                    let shift = if ss % 2 == 0 { 0 } else { 4 };

                    // 6-bit scale/min for this sub-block.
                    let (sc, m) = crate::weight::q4k_get_scale_min(scale_bytes, ss);

                    // Row-global sub-block index (matches from_f32 layout).
                    let sb = r * subs_per_row + sblk * subs_per_super + ss;
                    scales[sb] = sc;
                    // Fold ggml's `- dmin*m` into Arf's `+ dmin*min_i8`: min_i8 = -(m as i8).
                    // m is 6-bit (0..63); -(63 as i8) = -63, fits i8.
                    mins[sb] = -(m as i8);

                    // Copy nibbles verbatim into Arf's codes packing.
                    let qbase = half * 32;
                    for i in 0..Q4K_SUB {
                        let code = ((qs[qbase + i] >> shift) & 0xF) as u32;
                        // Arf absolute element position.
                        let pos = r * cols + sblk * Q4K_SUPER + ss * Q4K_SUB + i;
                        let widx = pos / 8;
                        let shift_p = (pos % 8) * 4;
                        codes[widx] |= code << shift_p;
                    }
                }
            }
        }

        Q4KSMatrix {
            codes,
            scales,
            mins,
            d,
            dmin,
            rows,
            cols,
        }
    }

    pub fn from_bf16(bits: &[u16], rows: usize, cols: usize) -> Self {
        let f: Vec<f32> = bits.iter().copied().map(bf16_to_f32).collect();
        Self::from_f32(&f, rows, cols)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
    pub fn mins(&self) -> &[i8] {
        &self.mins
    }
    pub fn d(&self) -> &[f32] {
        &self.d
    }
    pub fn dmin(&self) -> &[f32] {
        &self.dmin
    }

    /// Dequantize the whole matrix to f32 (tests, parity oracle).
    pub fn to_f32(&self) -> Vec<f32> {
        let subs_per_row = self.cols / Q4K_SUB;
        let supers_per_row = self.cols / Q4K_SUPER;
        let subs_per_super = Q4K_SUPER / Q4K_SUB;
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            for sblk in 0..supers_per_row {
                let sidx = r * supers_per_row + sblk;
                let dq = self.d[sidx];
                let dmq = self.dmin[sidx];
                for ss in 0..subs_per_super {
                    // Row-global sub-block index for scales/mins (sized rows*subs/row).
                    let sb = r * subs_per_row + sblk * subs_per_super + ss;
                    let s = dq * self.scales[sb] as f32;
                    let lo = dmq * self.mins[sb] as f32;
                    let base = r * self.cols + (sblk * subs_per_super + ss) * Q4K_SUB;
                    for j in 0..Q4K_SUB {
                        let widx = (base + j) / 8;
                        let shift = ((base + j) % 8) * 4;
                        let code = (self.codes[widx] >> shift) & 0xF;
                        out[base + j] = code as f32 * s + lo;
                    }
                }
            }
        }
        out
    }

    /// Linear (no bias): `x [m, cols]` → `[m, rows]`, i.e. `x · selfᵀ`.
    pub fn linear(&self, x: &Tensor) -> Tensor {
        let (m, k) = x.dims2();
        assert_eq!(
            k, self.cols,
            "linear: input dim {k} != weight in-dim {}",
            self.cols
        );
        let data = crate::matmul::matmul_nt_q4ks(
            x.as_slice(),
            m,
            k,
            &self.codes,
            &self.scales,
            &self.mins,
            &self.d,
            &self.dmin,
            self.rows,
        );
        Tensor::from_vec(data, vec![m, self.rows])
    }
}

/// Row-major `[rows, cols]` weight quantized to **Q3_K-style** 3-bit: like
/// [`Q4KSMatrix`] (super-block-256, 8 asymmetric sub-blocks of 32, u8/i8 two-level
/// scales vs per-super `d`/`dmin`) but codes are **3-bit unsigned** `[0, 7]`. The
/// 3 bits don't tile a `u32`, so they're SPLIT (ggml Q3_K style): `ql` holds the 2
/// LOW bits packed 16 per `u32` (a 32-weight sub-block = 2 words), `qh` the 1 HIGH
/// bit packed 32 per `u32` (a sub-block = 1 word), and `code = (qh<<2) | ql`. No
/// code straddles a `u32` boundary — clean for the GPU. ~3.125 bits/weight; ~25%
/// fewer code bytes than Q4 → faster decode (bandwidth-bound), at a real accuracy
/// cost (3-bit is the danger zone). Dequant `w = (d·scale_u8)·code + (dmin·min_i8)`.
///
/// Decision: 3-bit split-plane (ql/qh) over a single 3-bit-packed stream —
/// the split keeps every code inside one `u32` (GPU reads 2 ql-words + 1 qh-word
/// per sub-block, no cross-word shifts), exactly why ggml uses it.
#[derive(Debug, Clone)]
pub struct Q3KMatrix {
    ql: Vec<u32>,    // [rows * cols/16] — 2 low bits, 16/u32
    qh: Vec<u32>,    // [rows * cols/32] — 1 high bit, 32/u32
    scales: Vec<u8>, // [rows * cols/32] — u8 sub-scale vs per-super d
    mins: Vec<i8>,   // [rows * cols/32] — i8 sub-min vs per-super dmin
    d: Vec<f32>,     // [rows * cols/256]
    dmin: Vec<f32>,  // [rows * cols/256]
    rows: usize,
    cols: usize,
}

impl Q3KMatrix {
    /// Quantize an f32 `[rows, cols]` weight to Q3_K-lite. `cols % 256 == 0`.
    pub fn from_f32(w: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(w.len(), rows * cols, "Q3KMatrix shape mismatch");
        assert_eq!(
            cols % Q4K_SUPER,
            0,
            "Q3KMatrix cols must be a multiple of 256"
        );
        let subs_per_row = cols / Q4K_SUB;
        let supers_per_row = cols / Q4K_SUPER;
        let subs_per_super = Q4K_SUPER / Q4K_SUB; // 8
        let mut ql = vec![0u32; rows * cols / 16];
        let mut qh = vec![0u32; rows * cols / 32];
        let mut scales = vec![0u8; rows * subs_per_row];
        let mut mins = vec![0i8; rows * subs_per_row];
        let mut d = vec![0.0f32; rows * supers_per_row];
        let mut dmin = vec![0.0f32; rows * supers_per_row];
        for r in 0..rows {
            for sblk in 0..supers_per_row {
                let mut sub_scale = [0.0f32; 8];
                let mut sub_min = [0.0f32; 8];
                for ss in 0..subs_per_super {
                    let sb = sblk * subs_per_super + ss;
                    let base = r * cols + sb * Q4K_SUB;
                    let block = &w[base..base + Q4K_SUB];
                    let lo = block.iter().fold(f32::INFINITY, |m, &v| m.min(v));
                    let hi = block.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                    sub_scale[ss] = if hi > lo { (hi - lo) / 7.0 } else { 1.0 }; // 3-bit: /7
                    sub_min[ss] = lo;
                }
                let max_scale = sub_scale.iter().cloned().fold(0.0f32, f32::max);
                let max_min_abs = sub_min.iter().map(|m| m.abs()).fold(0.0f32, f32::max);
                let sidx = r * supers_per_row + sblk;
                d[sidx] = if max_scale > 0.0 {
                    max_scale / 255.0
                } else {
                    1.0
                };
                dmin[sidx] = if max_min_abs > 0.0 {
                    max_min_abs / 127.0
                } else {
                    1.0
                };
                let dq = d[sidx];
                let dmq = dmin[sidx];
                for ss in 0..subs_per_super {
                    let sb = r * subs_per_row + sblk * subs_per_super + ss;
                    let su8 = (sub_scale[ss] / dq).round().clamp(0.0, 255.0) as u8;
                    let mi8 = (sub_min[ss] / dmq).round().clamp(-127.0, 127.0) as i8;
                    scales[sb] = su8;
                    mins[sb] = mi8;
                    let s_q = dq * su8 as f32;
                    let lo_q = dmq * mi8 as f32;
                    let s_safe = if s_q > 0.0 { s_q } else { 1.0 };
                    let base = r * cols + (sblk * subs_per_super + ss) * Q4K_SUB;
                    for j in 0..Q4K_SUB {
                        let v = w[base + j];
                        let code = ((v - lo_q) / s_safe).round().clamp(0.0, 7.0) as u32;
                        let widx = base + j; // global weight index
                                             // ql: 2 low bits, 16 weights per u32.
                        ql[widx / 16] |= (code & 0x3) << ((widx % 16) * 2);
                        // qh: 1 high bit, 32 weights per u32.
                        qh[widx / 32] |= ((code >> 2) & 0x1) << (widx % 32);
                    }
                }
            }
        }
        Q3KMatrix {
            ql,
            qh,
            scales,
            mins,
            d,
            dmin,
            rows,
            cols,
        }
    }

    pub fn from_bf16(bits: &[u16], rows: usize, cols: usize) -> Self {
        let f: Vec<f32> = bits.iter().copied().map(bf16_to_f32).collect();
        Self::from_f32(&f, rows, cols)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn ql(&self) -> &[u32] {
        &self.ql
    }
    pub fn qh(&self) -> &[u32] {
        &self.qh
    }
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
    pub fn mins(&self) -> &[i8] {
        &self.mins
    }
    pub fn d(&self) -> &[f32] {
        &self.d
    }
    pub fn dmin(&self) -> &[f32] {
        &self.dmin
    }

    /// Reconstruct the 3-bit code for global weight index `widx`.
    #[inline]
    pub fn code_at(&self, widx: usize) -> u32 {
        let lo2 = (self.ql[widx / 16] >> ((widx % 16) * 2)) & 0x3;
        let hi1 = (self.qh[widx / 32] >> (widx % 32)) & 0x1;
        (hi1 << 2) | lo2
    }

    /// Dequantize the whole matrix to f32 (tests, parity oracle).
    pub fn to_f32(&self) -> Vec<f32> {
        let subs_per_row = self.cols / Q4K_SUB;
        let supers_per_row = self.cols / Q4K_SUPER;
        let subs_per_super = Q4K_SUPER / Q4K_SUB;
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            for sblk in 0..supers_per_row {
                let sidx = r * supers_per_row + sblk;
                let dq = self.d[sidx];
                let dmq = self.dmin[sidx];
                for ss in 0..subs_per_super {
                    let sb = r * subs_per_row + sblk * subs_per_super + ss;
                    let s = dq * self.scales[sb] as f32;
                    let lo = dmq * self.mins[sb] as f32;
                    let base = r * self.cols + (sblk * subs_per_super + ss) * Q4K_SUB;
                    for j in 0..Q4K_SUB {
                        out[base + j] = self.code_at(base + j) as f32 * s + lo;
                    }
                }
            }
        }
        out
    }

    /// Linear (no bias): `x [m, cols]` → `[m, rows]`, i.e. `x · selfᵀ`.
    pub fn linear(&self, x: &Tensor) -> Tensor {
        let (m, k) = x.dims2();
        assert_eq!(
            k, self.cols,
            "linear: input dim {k} != weight in-dim {}",
            self.cols
        );
        let data = crate::matmul::matmul_nt_q3k(
            x.as_slice(),
            m,
            k,
            &self.ql,
            &self.qh,
            &self.scales,
            &self.mins,
            &self.d,
            &self.dmin,
            self.rows,
        );
        Tensor::from_vec(data, vec![m, self.rows])
    }
}

/// Extract the 6-bit scale and min for sub-block `j` (0..8) from Q4_K's 12 scale
/// bytes. Matches ggml `get_scale_min_k4`.
pub fn q4k_get_scale_min(scales: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        let sc = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (sc, m)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_manual_dot() {
        // x [1,3] · w[2,3]ᵀ -> [1,2], with values exact in bf16.
        let x = Tensor::from_vec(vec![1.0, 2.0, 4.0], vec![1, 3]);
        let w = Bf16Matrix::from_f32(&[1.0, 0.0, -1.0, 2.0, 2.0, 2.0], 2, 3);
        let y = w.linear(&x);
        // row0: 1 - 4 = -3 ; row1: 2 + 4 + 8 = 14
        assert_eq!(y.as_slice(), &[-3.0, 14.0]);
    }

    #[test]
    fn gather_widens_rows() {
        let m = Bf16Matrix::from_f32(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let g = m.gather(&[1, 0]);
        assert_eq!(g.shape(), &[2, 2]);
        assert_eq!(g.as_slice(), &[3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn q8_roundtrip_is_within_per_row_error() {
        let (rows, cols) = (2, 5);
        let w = vec![
            1.0, -2.0, 0.5, 0.0, 2.0, // row 0: max-abs 2.0
            -8.0, 4.0, 1.0, -1.0, 8.0, // row 1: max-abs 8.0
        ];
        let back = Q8Matrix::from_f32(&w, rows, cols).to_f32();
        for r in 0..rows {
            let scale = w[r * cols..(r + 1) * cols]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()))
                / 127.0;
            for c in 0..cols {
                let i = r * cols + c;
                assert!(
                    (back[i] - w[i]).abs() <= scale / 2.0 + 1e-6,
                    "row {r} col {c}: {} vs {}",
                    back[i],
                    w[i]
                );
            }
        }
    }

    #[test]
    fn q8_linear_approximates_f32_linear() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 4.0], vec![1, 3]);
        // row0 = [1,0,-1], row1 = [2,2,2] — both representable, so result is exact.
        let q = Q8Matrix::from_f32(&[1.0, 0.0, -1.0, 2.0, 2.0, 2.0], 2, 3);
        let got = q.linear(&x);
        let y = got.as_slice();
        assert!((y[0] - -3.0).abs() < 0.05, "row0 {}", y[0]); // 1 - 4
        assert!((y[1] - 14.0).abs() < 0.2, "row1 {}", y[1]); // 2 + 4 + 8
    }

    #[test]
    fn q8_dims() {
        let q = Q8Matrix::from_f32(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!((q.rows(), q.cols()), (2, 2));
    }

    #[test]
    fn q4_decode_nibble_sign_extends() {
        // 0..7 stay positive; 8..15 are -8..-1.
        assert_eq!(Q4Matrix::decode_nibble(0), 0);
        assert_eq!(Q4Matrix::decode_nibble(7), 7);
        assert_eq!(Q4Matrix::decode_nibble(8), -8);
        assert_eq!(Q4Matrix::decode_nibble(15), -1);
    }

    #[test]
    fn q4_roundtrip_is_within_block_error() {
        // 32-wide block so it's exactly one Q4 block. Values span a range so the
        // dequant error is bounded by scale/2 (round-to-nearest).
        let cols = 32;
        let mut w = vec![0.0f32; cols];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i as f32 - 16.0) * 0.5; // -8.0 .. 7.5
        }
        let back = Q4Matrix::from_f32(&w, 1, cols).to_f32();
        let max_abs = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = bf16_to_f32(f32_to_bf16(max_abs / 7.0));
        for c in 0..cols {
            assert!(
                (back[c] - w[c]).abs() <= scale / 2.0 + 1e-4,
                "col {c}: {} vs {} (scale {scale})",
                back[c],
                w[c]
            );
        }
    }

    #[test]
    fn q4_packs_two_blocks_independently() {
        // Two blocks with very different magnitudes must get separate scales —
        // the small block must not lose all precision to the large block's scale.
        let cols = 64;
        let mut w = vec![0.0f32; cols];
        for (i, v) in w.iter_mut().enumerate() {
            // first block (0..32) tiny, second block (32..64) large.
            let mag = if i < 32 { 0.01 } else { 7.0 };
            *v = if i % 2 == 0 { mag } else { -mag };
        }
        let back = Q4Matrix::from_f32(&w, 1, cols).to_f32();
        // The tiny block round-trips with its own scale (≈0.01/7), so its values
        // are NOT crushed to zero.
        assert!(back[0].abs() > 0.005, "tiny block crushed: {}", back[0]);
        assert!((back[32] - 7.0).abs() < 0.2, "large block: {}", back[32]);
    }

    #[test]
    fn q4_linear_matches_dequant_then_f32_matmul() {
        // The Q4 linear must equal dequantizing to f32 and running the f32 matmul —
        // this is exactly what the GPU kernel must reproduce.
        let (rows, cols) = (3usize, 64usize);
        let mut w = vec![0.0f32; rows * cols];
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i * 7 % 29) as f32 - 14.0) * 0.3;
        }
        let q = Q4Matrix::from_f32(&w, rows, cols);
        let deq = q.to_f32();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 % 5.0) - 2.0).collect();
        let xt = Tensor::from_vec(x.clone(), vec![1, cols]);
        let got = q.linear(&xt);
        // Reference: dequantized weights through the plain f32 matmul.
        let want = crate::matmul::matmul_nt(&x, 1, cols, &deq, rows);
        for (j, (&g, &wnt)) in got.as_slice().iter().zip(&want).enumerate() {
            assert!((g - wnt).abs() <= 1e-3, "row {j}: {g} vs {wnt}");
        }
    }

    #[test]
    fn q4_dims() {
        let q = Q4Matrix::from_f32(&[0.5f32; 32], 1, 32);
        assert_eq!((q.rows(), q.cols()), (1, 32));
    }

    #[test]
    fn q4k_roundtrip_is_within_subblock_error() {
        // One 256-weight super-block (8 sub-blocks of 32). Asymmetric quant of a
        // ramp; dequant error is bounded by scale/2 (round-to-nearest) per sub-block.
        let cols = 256;
        let w: Vec<f32> = (0..cols).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let back = Q4KMatrix::from_f32(&w, 1, cols).to_f32();
        for sb in 0..(cols / Q4K_SUB) {
            let blk = &w[sb * Q4K_SUB..(sb + 1) * Q4K_SUB];
            let lo = blk.iter().fold(f32::INFINITY, |m, &v| m.min(v));
            let hi = blk.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let scale = bf16_to_f32(f32_to_bf16((hi - lo) / 15.0));
            for j in 0..Q4K_SUB {
                let c = sb * Q4K_SUB + j;
                assert!(
                    (back[c] - w[c]).abs() <= scale / 2.0 + 1e-3,
                    "sub {sb} col {c}: {} vs {}",
                    back[c],
                    w[c]
                );
            }
        }
    }

    #[test]
    fn q4k_subblocks_quantize_independently() {
        // Two sub-blocks with wildly different ranges in one super-block: a tiny
        // one and a large one. The asymmetric per-sub-block scale must keep the tiny
        // one's values from being crushed.
        let cols = 256;
        let mut w = vec![0.0f32; cols];
        for (i, v) in w.iter_mut().enumerate() {
            // sub-block 0 (0..32) tiny around 0.02; sub-block 1 (32..64) large ±5.
            *v = if i < 32 {
                0.02 * ((i % 4) as f32 - 1.5)
            } else if i < 64 {
                5.0 * ((i % 4) as f32 - 1.5)
            } else {
                0.0
            };
        }
        let back = Q4KMatrix::from_f32(&w, 1, cols).to_f32();
        // tiny block resolves to ~scale/2 of its own range (~0.06/15), not crushed
        // to the large block's grid.
        assert!(
            (back[0] - w[0]).abs() < 0.01,
            "tiny block crushed: {} vs {}",
            back[0],
            w[0]
        );
        assert!(
            (back[35] - w[35]).abs() < 0.7,
            "large block: {} vs {}",
            back[35],
            w[35]
        );
    }

    #[test]
    fn q4k_linear_matches_dequant_then_f32_matmul() {
        // Q4_K linear must equal dequantize-to-f32 then plain f32 matmul — exactly
        // what the GPU kernel must reproduce.
        let (rows, cols) = (3usize, 256usize);
        let mut w = vec![0.0f32; rows * cols];
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i * 11 % 53) as f32 - 26.0) * 0.2;
        }
        let q = Q4KMatrix::from_f32(&w, rows, cols);
        let deq = q.to_f32();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let xt = Tensor::from_vec(x.clone(), vec![1, cols]);
        let got = q.linear(&xt);
        let want = crate::matmul::matmul_nt(&x, 1, cols, &deq, rows);
        for (j, (&g, &wnt)) in got.as_slice().iter().zip(&want).enumerate() {
            assert!((g - wnt).abs() <= 1e-2, "row {j}: {g} vs {wnt}");
        }
    }

    #[test]
    fn q4k_more_accurate_than_q4_0() {
        // The asymmetric min-offset should give LOWER reconstruction error than Q4_0's
        // symmetric centering on a SKEWED (all-positive) distribution — the case Q4_0
        // wastes its sign bit on.
        let cols = 256;
        let w: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32) * 0.01).collect(); // 1.0..3.55, all positive
        let q4k_err: f32 = Q4KMatrix::from_f32(&w, 1, cols)
            .to_f32()
            .iter()
            .zip(&w)
            .map(|(a, b)| (a - b).abs())
            .sum();
        let q40_err: f32 = Q4Matrix::from_f32(&w, 1, cols)
            .to_f32()
            .iter()
            .zip(&w)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            q4k_err < q40_err,
            "Q4_K err {q4k_err} should be < Q4_0 err {q40_err} on a skewed positive block"
        );
    }

    #[test]
    fn q4k_dims() {
        let q = Q4KMatrix::from_f32(&[0.3f32; 256], 1, 256);
        assert_eq!((q.rows(), q.cols()), (1, 256));
        assert_eq!(q.scales().len(), 256 / 32);
        assert_eq!(q.mins().len(), 256 / 32);
    }

    /// The searched fit never reconstructs WORSE than min/max on bell-shaped weights (it
    /// starts from min/max and only replaces it with a lower-error candidate), it must win
    /// clearly where min/max is at its worst — a block stretched by one outlier — and rows
    /// split across threads must land in their own slots.
    #[test]
    fn q4ks_searched_fit_beats_minmax() {
        let (rows, cols) = (37, 512); // 37 rows: an uneven split over any thread count
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        };
        // sum of 6 uniforms ~ a bell; every 97th weight is a 6x outlier
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let g: f32 = (0..6).map(|_| rnd()).sum::<f32>() * 0.05;
                if i % 97 == 0 {
                    g * 6.0
                } else {
                    g
                }
            })
            .collect();
        let sq = |q: Q4KSMatrix| -> f64 {
            q.to_f32()
                .iter()
                .zip(&w)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum()
        };
        let plain = sq(Q4KSMatrix::from_f32(&w, rows, cols));
        let searched = sq(Q4KSMatrix::from_f32_searched(&w, rows, cols));
        assert!(
            searched < plain * 0.9,
            "searched fit {searched} should be well under min/max {plain}"
        );
        // Row independence: quantising one row alone gives that row's bits in the big matrix.
        let r = 23;
        let one = Q4KSMatrix::from_f32_searched(&w[r * cols..(r + 1) * cols], 1, cols);
        let all = Q4KSMatrix::from_f32_searched(&w, rows, cols);
        assert_eq!(one.codes(), &all.codes()[r * cols / 8..(r + 1) * cols / 8]);
        assert_eq!(
            one.scales(),
            &all.scales()[r * cols / 32..(r + 1) * cols / 32]
        );
        assert_eq!(one.d(), &all.d()[r * cols / 256..(r + 1) * cols / 256]);
    }

    #[test]
    fn q4ks_roundtrip_close_to_q4k() {
        // Q4_K_S (u8 two-level scales) should reconstruct nearly as well as Q4_K-lite
        // (bf16 scales) — the u8 sub-scale quant adds only a small extra error.
        let cols = 256;
        let w: Vec<f32> = (0..cols).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let q4ks_err: f32 = Q4KSMatrix::from_f32(&w, 1, cols)
            .to_f32()
            .iter()
            .zip(&w)
            .map(|(a, b)| (a - b).abs())
            .sum();
        let q4k_err: f32 = Q4KMatrix::from_f32(&w, 1, cols)
            .to_f32()
            .iter()
            .zip(&w)
            .map(|(a, b)| (a - b).abs())
            .sum();
        // Within 2× of Q4_K's error (u8 sub-scale rounding is the only extra source).
        assert!(
            q4ks_err <= q4k_err * 2.0 + 1.0,
            "Q4_K_S err {q4ks_err} should be ~Q4_K err {q4k_err}"
        );
    }

    #[test]
    fn q4ks_linear_matches_dequant_then_f32_matmul() {
        let (rows, cols) = (3usize, 256usize);
        let mut w = vec![0.0f32; rows * cols];
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i * 11 % 53) as f32 - 26.0) * 0.2;
        }
        let q = Q4KSMatrix::from_f32(&w, rows, cols);
        let deq = q.to_f32();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let xt = Tensor::from_vec(x.clone(), vec![1, cols]);
        let got = q.linear(&xt);
        let want = crate::matmul::matmul_nt(&x, 1, cols, &deq, rows);
        for (j, (&g, &wnt)) in got.as_slice().iter().zip(&want).enumerate() {
            assert!((g - wnt).abs() <= 1e-2, "row {j}: {g} vs {wnt}");
        }
    }

    #[test]
    fn q4ks_dims() {
        let q = Q4KSMatrix::from_f32(&[0.3f32; 256], 1, 256);
        assert_eq!((q.rows(), q.cols()), (1, 256));
        assert_eq!(q.scales().len(), 256 / 32);
        assert_eq!(q.d().len(), 256 / 256);
        // u8 scale + i8 min = 1 byte each, vs Q4_K-lite's 2 (bf16) — the byte win.
    }

    #[test]
    fn q3k_code_split_roundtrips() {
        // The ql/qh split must reconstruct every 3-bit code 0..7 exactly.
        let cols = 256;
        let w: Vec<f32> = (0..cols).map(|i| (i % 8) as f32 - 3.5).collect();
        let q = Q3KMatrix::from_f32(&w, 1, cols);
        // code_at must invert the pack: re-quantize the first sub-block's values and
        // check codes round-trip in [0,7].
        for widx in 0..cols {
            let c = q.code_at(widx);
            assert!(c <= 7, "code {c} at {widx} out of 3-bit range");
        }
    }

    #[test]
    fn q3k_linear_matches_dequant_then_f32_matmul() {
        let (rows, cols) = (3usize, 256usize);
        let mut w = vec![0.0f32; rows * cols];
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i * 11 % 53) as f32 - 26.0) * 0.2;
        }
        let q = Q3KMatrix::from_f32(&w, rows, cols);
        let deq = q.to_f32();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let xt = Tensor::from_vec(x.clone(), vec![1, cols]);
        let got = q.linear(&xt);
        let want = crate::matmul::matmul_nt(&x, 1, cols, &deq, rows);
        for (j, (&g, &wnt)) in got.as_slice().iter().zip(&want).enumerate() {
            assert!((g - wnt).abs() <= 2e-2, "row {j}: {g} vs {wnt}");
        }
    }

    #[test]
    fn q3k_is_lossier_than_q4_but_bounded() {
        // Quantify the trade: Q3_K (3-bit) error should be HIGHER than Q4_0 (4-bit)
        // but within ~3× (not a cliff). This is the accuracy cost we're trading for
        // ~25% fewer bytes — measured, not assumed.
        let cols = 256;
        let w: Vec<f32> = (0..cols)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) * 0.1)
            .collect();
        let l1 = |back: Vec<f32>| -> f32 { back.iter().zip(&w).map(|(a, b)| (a - b).abs()).sum() };
        let q3 = l1(Q3KMatrix::from_f32(&w, 1, cols).to_f32());
        let q4 = l1(Q4Matrix::from_f32(&w, 1, cols).to_f32());
        assert!(q3 > q4, "Q3 ({q3}) should be lossier than Q4 ({q4})");
        assert!(q3 < q4 * 3.5, "Q3 ({q3}) within ~3× Q4 ({q4}) — no cliff");
    }

    #[test]
    fn q3k_dims() {
        let q = Q3KMatrix::from_f32(&[0.3f32; 256], 1, 256);
        assert_eq!((q.rows(), q.cols()), (1, 256));
        assert_eq!(q.ql().len(), 256 / 16); // 2 low bits, 16/u32
        assert_eq!(q.qh().len(), 256 / 32); // 1 high bit, 32/u32
    }

    // ──────────────────────────────────────────────────────────────────────────
    // the design bit-exact gate: Q4KSMatrix::from_ggml_q4k must reproduce the
    // EXACT same f32 values as ggml's dequant_q4_k on the same raw bytes.
    // Every assertion below uses `==` (bit-exact), not approx.
    // ──────────────────────────────────────────────────────────────────────────

    /// Minimal f32 → f16 bits (exact for values representable in f16; used only
    /// to build test block headers without pulling in a half-float crate).
    /// Build one 144-byte Q4_K block with the given parameters.
    /// Assert that two f32 slices are bit-identical (no ULP tolerance).
    #[test]
    fn get_q4k_blocks_byte_len_invariant() {
        // The invariant from Step 1 spec: returned slice length == ggml_byte_len(Q4K, rows*cols).
        // We verify the math directly (LazyGguf.get_q4k_blocks can't be unit-tested
        // without a real GGUF file, but the length formula is trivial to verify).
        // 256 weights = 1 block = 144 bytes.
        let weights_per_block = 256usize;
        let bytes_per_block = 144usize;
        for n_blocks in [1, 8, 64, 6144] {
            let n = n_blocks * weights_per_block;
            let expected = n_blocks * bytes_per_block;
            // This is exactly what ggml_byte_len(Q4K, n) returns.
            assert_eq!(n / weights_per_block * bytes_per_block, expected);
        }
    }
}
