//! bfloat16 ↔ f32 conversion.
//!
//! bf16 is the top 16 bits of an f32 (same exponent range, 8-bit mantissa), so
//! widening is a shift and narrowing is a round of the low 16 bits. We store
//! weights as bf16 to halve memory and widen them to f32 in the matmul.

/// Widen a bf16 bit pattern to `f32`.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Widen 8 bf16 bit patterns to an `f32x8`. bf16 is the top 16 bits of an f32,
/// so the widen is `(u32(bits) << 16)` reinterpreted as f32 — the shift runs
/// lane-parallel, in safe Rust (no `unsafe`, no per-lane closure call).
#[inline]
pub fn widen8(bits: [u16; 8]) -> wide::f32x8 {
    use wide::u32x8;
    let raw = u32x8::from(std::array::from_fn::<u32, 8, _>(|i| u32::from(bits[i])));
    let shifted: u32x8 = raw << 16;
    let words = shifted.to_array();
    wide::f32x8::from(std::array::from_fn::<f32, 8, _>(|i| {
        f32::from_bits(words[i])
    }))
}

/// Narrow an `f32` to bf16 bits, round-to-nearest-even, NaN-preserving.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7FC0; // quiet NaN
    }
    let bits = x.to_bits();
    // Round to nearest even: add the rounding bias derived from the kept LSB.
    let round_bias = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + round_bias) >> 16) as u16
}

/// IEEE half (f16) → f32. Public so self-contained GPU modules (FLUX T5/DiT) can read
/// per-block Q8_0 scales when splitting raw ggml blocks into aligned GPU buffers.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let f = match exp {
        0 => {
            if mant == 0 {
                (sign as u32) << 31
            } else {
                // subnormal → normalize
                let mut e: i32 = -14;
                let mut m = mant as u32;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
            }
        }
        0x1f => ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13),
        _ => ((sign as u32) << 31) | ((exp as u32 + 112) << 23) | ((mant as u32) << 13),
    };
    f32::from_bits(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_values_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, 2.0, 0.5, -0.25, 256.0] {
            assert_eq!(
                bf16_to_f32(f32_to_bf16(v)),
                v,
                "{v} should be exact in bf16"
            );
        }
    }

    #[test]
    fn rounding_is_within_bf16_resolution() {
        // bf16 has 8 mantissa bits → relative error <= 2^-8.
        for &v in &[3.3711f32, -2.512, 123.456, 0.001234] {
            let r = bf16_to_f32(f32_to_bf16(v));
            assert!((r - v).abs() <= v.abs() * 2f32.powi(-8) + f32::EPSILON);
        }
    }

    #[test]
    fn nan_stays_nan() {
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn widen8_matches_scalar() {
        let bits: [u16; 8] = [
            0x3F80, 0x4000, 0xBF80, 0x3F00, 0x0000, 0x4380, 0xBE80, 0x4040,
        ];
        let got = widen8(bits).to_array();
        for (i, &b) in bits.iter().enumerate() {
            assert_eq!(got[i], bf16_to_f32(b), "lane {i}");
        }
        assert_eq!(got, [1.0, 2.0, -1.0, 0.5, 0.0, 256.0, -0.25, 3.0]);
    }
}
