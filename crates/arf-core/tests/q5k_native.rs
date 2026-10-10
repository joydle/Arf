//! Does `Q5KSMatrix::from_ggml_q5k` reconstruct the same values as the
//! reference dequantizer?
//!
//! The high bit is the part that is easy to get wrong: `qh[l]` holds the 5th
//! bit of EIGHT different weights - one per sub-block - and the bit selected
//! is `1 << sub_block`, with `l` running inside the 32. A flat-index reading
//! (`qh[e % 32] >> (e / 32)`) looks equivalent and transposes which weight
//! each bit belongs to. The first version of the reader had exactly that bug
//! and it would have been silent: plausible magnitudes, wrong weights.
//!
//! Both sides read the same synthetic block bytes, so a mismatch is the
//! unpack and nothing else.

/// Rebuild the f32 values from Arf's unpacked form, the way the GEMV will.
fn reconstruct(m: &arf_ml::weight::Q5KSMatrix, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let pos = r * cols + c;
            let lo = (m.codes_lo()[pos / 8] >> ((pos % 8) * 4)) & 0xF;
            let hi = (m.codes_hi()[pos / 32] >> (pos % 32)) & 1;
            let code = (lo | (hi << 4)) as f32;
            let sub = r * (cols / 32) + c / 32;
            let sup = r * (cols / 256) + c / 256;
            out[pos] =
                code * (m.d()[sup] * m.scales()[sub] as f32) + m.dmin()[sup] * m.mins()[sub] as f32;
        }
    }
    out
}

#[test]
fn q5k_native_unpack_matches_the_reference_dequantizer() {
    const ROWS: usize = 2;
    const COLS: usize = 512; // two superblocks per row
    const BLOCK: usize = 176;
    let nblocks = ROWS * COLS / 256;

    // Deterministic block bytes. Every field is exercised: non-trivial d and
    // dmin, all 12 scale bytes, and a qh pattern where different sub-blocks
    // disagree - which is what catches a transposed high bit.
    let mut bytes = vec![0u8; nblocks * BLOCK];
    for b in 0..nblocks {
        let p = b * BLOCK;
        // d = 0.05, dmin = -0.01 as f16
        bytes[p..p + 2].copy_from_slice(&half_bits(0.05).to_le_bytes());
        bytes[p + 2..p + 4].copy_from_slice(&half_bits(-0.01).to_le_bytes());
        for i in 0..12 {
            bytes[p + 4 + i] = ((b * 13 + i * 7) % 256) as u8;
        }
        for i in 0..32 {
            bytes[p + 16 + i] = ((b * 31 + i * 11) % 256) as u8; // qh
        }
        for i in 0..128 {
            bytes[p + 48 + i] = ((b * 17 + i * 3) % 256) as u8; // qs
        }
    }

    let want = arf_core::model::gguf::dequant_q5_k_for_test(&bytes, ROWS * COLS)
        .expect("reference dequant");
    let m = arf_ml::weight::Q5KSMatrix::from_ggml_q5k(&bytes, ROWS, COLS);
    let got = reconstruct(&m, ROWS, COLS);

    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..want.len() {
        let d = (want[i] - got[i]).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst < 1e-5,
        "Q5_K unpack differs from the reference: worst {worst:.3e} at {at} \
         (want {}, got {})",
        want[at],
        got[at]
    );
}

/// f32 -> f16 bits, enough for the two scale fields above.
fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = ((b >> 13) & 0x3FF) as u16;
    if exp <= 0 {
        sign
    } else {
        sign | ((exp as u16) << 10) | mant
    }
}
