//! PCM audio input (2026-09-27): WAV decoding, channel mixdown, and sample-rate conversion, in
//! pure Rust with no new dependency. The Qwen3-Omni audio front end (`qwen_audio`) is the first
//! user; nothing here is model-specific.
//!
//! What each step reproduces, read in the reference source before it was written:
//! - **decode**: qwen-omni-utils 0.0.9 loads audio with `librosa.load(.., sr=16000)`
//!   (`audio_process.py:84-91`), which in librosa 1.0.0 reads through soundfile as float32
//!   (`librosa/core/audio.py:171-198`). libsndfile's integer -> float scale is `x / 2^(bits-1)`
//!   (16-bit: `x / 32768`), 8-bit WAV is unsigned (`(x - 128) / 128`), float WAV is passed through.
//! - **mixdown**: `librosa.to_mono` is the mean over channels (`audio.py:705-740`, `norm=True`),
//!   in float32, BEFORE resampling (`audio.py:159-163`).
//! - **resample**: librosa's default `res_type="soxr_hq"` (`audio.py:1158-1167`), then
//!   `fix_length` to `int(ceil(n * target / orig))` samples (`audio.py:1121-1123,1172-1173`). A
//!   same-rate input is returned untouched (`audio.py:1118-1119`).
//!
//! **The resampler is NOT soxr.** soxr HQ is a multi-stage FFT-based design; porting it is out of
//! scope. [`resample`] is a single-stage Kaiser-windowed sinc whose three parameters were FITTED to
//! soxr 1.1.0 HQ's measured response (2026-09-27, 16 kHz output: flat to 7.4 kHz, -2.9 dB at
//! 7.6 kHz, -24 dB at 7.8 kHz, below -130 dB from 8.0 kHz): cutoff 0.957 of the lower Nyquist,
//! transition 0.08, 120 dB. How the fit went (worst max-abs vs soxr, python prototype, 0.3 s
//! clips of tones up to 7.7 kHz and uniform noise at 8/22.05/44.1/48 kHz): 0.95 / 0.075 / 120 dB
//! -> 3.5e-2, 0.95 / 0.07 / 130 -> 3.5e-2, 0.96 / 0.08 / 120 -> 1.5e-2, 0.955 / 0.08 / 120 ->
//! 9.0e-3; then a finer grid on three of those clips: 0.957 / 0.08 / 120 -> 1.1e-3 (chosen),
//! 0.9565 -> 1.8e-3, at 100 dB -> 2.6e-3, at 140 dB -> 3.3e-3. On the `testdata/qwen_audio`
//! fixtures this Rust port lands at 1.2e-4 (8 kHz up) to 7.1e-3 (22.05 kHz half-scale white
//! noise, whose energy fills the 7.4-8 kHz transition band where the two filter shapes differ);
//! the test gate is the plan's 2e-2. The transformers `load_audio` path with torchcodec installed
//! resamples with FFmpeg's swr instead, so "the reference resampler" is itself backend-dependent;
//! Arf follows the model card's (qwen-omni-utils -> librosa -> soxr).

use crate::{ArfError, Result};

/// Interleaved PCM as decoded: `samples.len() == frames * channels`, float in [-1, 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Pcm {
    pub sample_rate: u32,
    pub channels: usize,
    /// Interleaved samples (frame-major).
    pub samples: Vec<f32>,
}

impl Pcm {
    /// Frames (samples per channel).
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1)
    }
}

/// Is this a RIFF/RF64 WAVE file? (A cheap sniff: the caller hands anything else to ffmpeg.)
pub fn is_wav(bytes: &[u8]) -> bool {
    bytes.len() >= 12
        && (&bytes[0..4] == b"RIFF" || &bytes[0..4] == b"RF64")
        && &bytes[8..12] == b"WAVE"
}

fn le16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Decode a WAV file: PCM 8/16/24/32-bit and IEEE float 32/64-bit, plain or
/// `WAVE_FORMAT_EXTENSIBLE`, any channel count. A `data` chunk whose size is `0xFFFFFFFF` or runs
/// past the end (what ffmpeg writes to a pipe, and RF64) is read to the end of the buffer; a
/// trailing partial frame is dropped. Compressed WAV payloads (ADPCM, mu-law, ...) are refused
/// with an error naming the format tag, so a caller can route them to ffmpeg.
pub fn decode_wav(bytes: &[u8]) -> Result<Pcm> {
    if !is_wav(bytes) {
        return Err(ArfError::other("not a RIFF/WAVE file"));
    }
    let mut at = 12usize;
    let mut fmt: Option<(u16, usize, u32, usize)> = None; // (tag, channels, rate, bits)
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = le32(bytes, at + 4) as usize;
        let body = at + 8;
        if id == b"fmt " {
            if size < 16 || body + 16 > bytes.len() {
                return Err(ArfError::other("WAV fmt chunk is truncated"));
            }
            let mut tag = le16(bytes, body);
            let channels = le16(bytes, body + 2) as usize;
            let rate = le32(bytes, body + 4);
            let bits = le16(bytes, body + 14) as usize;
            if tag == 0xFFFE {
                // WAVE_FORMAT_EXTENSIBLE: cbSize(2) validBits(2) mask(4) then the subformat GUID,
                // whose first two bytes are the real format tag.
                if size < 40 || body + 26 > bytes.len() {
                    return Err(ArfError::other("WAV extensible fmt chunk is truncated"));
                }
                tag = le16(bytes, body + 24);
            }
            fmt = Some((tag, channels, rate, bits));
        } else if id == b"data" {
            let (tag, channels, rate, bits) =
                fmt.ok_or_else(|| ArfError::other("WAV data chunk before its fmt chunk"))?;
            let end = if size == 0xFFFF_FFFF || body.saturating_add(size) > bytes.len() {
                bytes.len()
            } else {
                body + size
            };
            return decode_samples(&bytes[body..end], tag, channels, rate, bits);
        }
        // Chunks are word-aligned: an odd size carries one pad byte.
        at = body.saturating_add(size).saturating_add(size & 1);
    }
    Err(ArfError::other("WAV file has no data chunk"))
}

fn decode_samples(data: &[u8], tag: u16, channels: usize, rate: u32, bits: usize) -> Result<Pcm> {
    if channels == 0 || rate == 0 {
        return Err(ArfError::other(format!(
            "WAV header: {channels} channels at {rate} Hz"
        )));
    }
    let width = bits.div_ceil(8);
    let supported = matches!((tag, bits), (1, 8 | 16 | 24 | 32) | (3, 32 | 64));
    if !supported {
        return Err(ArfError::other(format!(
            "WAV format tag {tag:#06x} at {bits} bits is not decoded natively"
        )));
    }
    let frame = width * channels;
    let n = data.len() / frame * channels;
    let mut samples = Vec::with_capacity(n);
    for c in data.chunks_exact(width).take(n) {
        let v = match (tag, bits) {
            (1, 8) => (c[0] as f32 - 128.0) / 128.0,
            (1, 16) => i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0,
            (1, 24) => {
                let x = i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8;
                x as f32 / 8_388_608.0
            }
            (1, 32) => i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0,
            (3, 32) => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            (3, 64) => f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32,
            _ => unreachable!("checked above"),
        };
        samples.push(v);
    }
    Ok(Pcm {
        sample_rate: rate,
        channels,
        samples,
    })
}

/// `librosa.to_mono`: the float32 mean over channels, summed in channel order.
pub fn to_mono(pcm: &Pcm) -> Vec<f32> {
    let ch = pcm.channels.max(1);
    if ch == 1 {
        return pcm.samples.clone();
    }
    pcm.samples
        .chunks_exact(ch)
        .map(|f| f.iter().fold(0.0f32, |a, &b| a + b) / ch as f32)
        .collect()
}

/// Resampler parameters, fitted to soxr HQ (see the module doc).
const RS_CUTOFF: f64 = 0.957;
const RS_TRANSITION: f64 = 0.08;
const RS_ATTEN_DB: f64 = 120.0;
/// Phases above this are computed per output sample instead of tabulated (44100 -> 16000 has
/// 160 phases, 37800 -> 16000 has 80; an odd rate such as 44056 -> 16000 has 2000).
const RS_MAX_TABLE_PHASES: usize = 1024;

fn bessel_i0(x: f64) -> f64 {
    // Power series; converges fast for the beta ~12 used here (terms fall below 1e-17 by k ~ 40).
    let (mut sum, mut term) = (1.0f64, 1.0f64);
    let q = x * x / 4.0;
    for k in 1..200 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// The windowed-sinc kernel, in input-sample units.
struct Kernel {
    /// Cutoff over the input rate (cycles per input sample).
    fc: f64,
    half_width: usize,
    beta: f64,
    i0_beta: f64,
}

impl Kernel {
    fn new(from: u32, to: u32) -> Kernel {
        let fin = from as f64;
        let nyq = from.min(to) as f64 / 2.0;
        let fc = RS_CUTOFF * nyq / fin;
        let df = RS_TRANSITION * nyq / fin;
        let beta = 0.1102 * (RS_ATTEN_DB - 8.7);
        // Kaiser's length estimate: N = (A - 7.95) / (2.285 * 2*pi*df).
        let taps = (RS_ATTEN_DB - 7.95) / (2.285 * 2.0 * std::f64::consts::PI * df);
        Kernel {
            fc,
            half_width: (taps / 2.0).ceil() as usize,
            beta,
            i0_beta: bessel_i0(beta),
        }
    }

    fn at(&self, tau: f64) -> f64 {
        let hw = self.half_width as f64;
        let r = tau / hw;
        let w = bessel_i0(self.beta * (1.0 - r * r).max(0.0).sqrt()) / self.i0_beta;
        let x = 2.0 * self.fc * tau;
        let sinc = if x == 0.0 {
            1.0
        } else {
            let px = std::f64::consts::PI * x;
            px.sin() / px
        };
        2.0 * self.fc * sinc * w
    }

    /// The `2 * half_width` taps for fractional position `frac` in [0, 1): tap `j` multiplies
    /// input sample `floor(t) - half_width + 1 + j`.
    fn taps(&self, frac: f64, out: &mut [f64]) {
        let hw = self.half_width as f64;
        for (j, o) in out.iter_mut().enumerate() {
            *o = self.at(frac + hw - 1.0 - j as f64);
        }
    }
}

/// Resample mono `x` from `from` Hz to `to` Hz: exactly `int(ceil(len * to / from))` samples (the
/// length librosa's `fix_length` enforces), zeros assumed outside the input, accumulated in f64.
/// See the module doc for what this does and does not match.
pub fn resample(x: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == 0 || to == 0 {
        return Err(ArfError::other(format!("resample {from} Hz -> {to} Hz")));
    }
    if from == to {
        return Ok(x.to_vec());
    }
    // librosa: n_samples = int(np.ceil(y.shape[-1] * (float(target_sr) / orig_sr))) -- the float
    // expression, kept verbatim so an off-by-one from float rounding matches too.
    let n_out = (x.len() as f64 * (to as f64 / from as f64)).ceil() as usize;
    let g = gcd(from as u64, to as u64);
    let (up, down) = (to as u64 / g, from as u64 / g); // t_n = n * down / up input samples
    let k = Kernel::new(from, to);
    let width = 2 * k.half_width;
    let table: Option<Vec<f64>> = (up as usize <= RS_MAX_TABLE_PHASES).then(|| {
        let mut t = vec![0.0f64; up as usize * width];
        for (p, row) in t.chunks_exact_mut(width).enumerate() {
            k.taps(p as f64 / up as f64, row);
        }
        t
    });
    let mut scratch = vec![0.0f64; width];
    let mut out = Vec::with_capacity(n_out);
    for n in 0..n_out as u64 {
        let pos = n * down;
        let (i0, phase) = ((pos / up) as i64, (pos % up) as usize);
        let taps: &[f64] = match &table {
            Some(t) => &t[phase * width..(phase + 1) * width],
            None => {
                k.taps(phase as f64 / up as f64, &mut scratch);
                &scratch
            }
        };
        let first = i0 - k.half_width as i64 + 1;
        let lo = (-first).max(0) as usize;
        let hi = ((x.len() as i64 - first).max(0) as usize).min(width);
        let mut acc = 0.0f64;
        for j in lo..hi {
            acc += taps[j] * x[(first + j as i64) as usize] as f64;
        }
        out.push(acc as f32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_bytes(
        tag: u16,
        channels: u16,
        rate: u32,
        bits: u16,
        data: &[u8],
        extensible: bool,
    ) -> Vec<u8> {
        let mut fmt = Vec::new();
        let block = channels * bits.div_ceil(8);
        fmt.extend_from_slice(&(if extensible { 0xFFFE } else { tag }).to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * block as u32).to_le_bytes());
        fmt.extend_from_slice(&block.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        if extensible {
            fmt.extend_from_slice(&22u16.to_le_bytes());
            fmt.extend_from_slice(&bits.to_le_bytes());
            fmt.extend_from_slice(&0u32.to_le_bytes());
            fmt.extend_from_slice(&tag.to_le_bytes());
            fmt.extend_from_slice(&[0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
        }
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        // An odd-sized unknown chunk first: exercises the word-alignment pad byte.
        b.extend_from_slice(b"junk");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&[1, 2, 3, 0]);
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        b.extend_from_slice(&fmt);
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(data);
        b
    }

    #[test]
    fn wav_integer_and_float_formats_decode_to_libsndfile_scale() {
        let d16: Vec<u8> = [-32768i16, -1, 0, 16384, 32767]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let p = decode_wav(&wav_bytes(1, 1, 16000, 16, &d16, false)).unwrap();
        assert_eq!(p.sample_rate, 16000);
        assert_eq!(
            p.samples,
            vec![-1.0, -1.0 / 32768.0, 0.0, 0.5, 32767.0 / 32768.0]
        );

        let p = decode_wav(&wav_bytes(1, 1, 8000, 8, &[0, 128, 192, 255], false)).unwrap();
        assert_eq!(p.samples, vec![-1.0, 0.0, 0.5, 127.0 / 128.0]);

        // 24-bit: -2^23, -1, 2^22 (little-endian 3 bytes each), stereo.
        let d24 = [0, 0, 0x80, 0xFF, 0xFF, 0xFF, 0, 0, 0x40, 0, 0, 0];
        let p = decode_wav(&wav_bytes(1, 2, 44100, 24, &d24, true)).unwrap();
        assert_eq!((p.channels, p.frames()), (2, 2));
        assert_eq!(p.samples, vec![-1.0, -1.0 / 8_388_608.0, 0.5, 0.0]);

        let d32: Vec<u8> = [i32::MIN, 1 << 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let p = decode_wav(&wav_bytes(1, 1, 16000, 32, &d32, false)).unwrap();
        assert_eq!(p.samples, vec![-1.0, 0.5]);

        let df: Vec<u8> = [0.25f32, -0.75]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let p = decode_wav(&wav_bytes(3, 1, 48000, 32, &df, true)).unwrap();
        assert_eq!(p.samples, vec![0.25, -0.75]);
        let dd: Vec<u8> = [0.125f64].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            decode_wav(&wav_bytes(3, 1, 48000, 64, &dd, false))
                .unwrap()
                .samples,
            vec![0.125]
        );
    }

    #[test]
    fn wav_streamed_size_partial_frames_and_refusals() {
        // ffmpeg writing to a pipe: data size 0xFFFFFFFF -> read to the end; the trailing odd
        // byte of a 16-bit stereo stream is a partial frame and is dropped.
        let d: Vec<u8> = [100i16, -100, 200, -200]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut b = wav_bytes(1, 2, 16000, 16, &d, false);
        let n = b.len();
        b[n - d.len() - 4..n - d.len()].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        b.extend_from_slice(&[7, 7, 7]);
        let p = decode_wav(&b).unwrap();
        assert_eq!(p.frames(), 2);
        assert_eq!(p.samples[3], -200.0 / 32768.0);

        let e = decode_wav(&wav_bytes(2, 1, 16000, 4, &[0; 8], false)).unwrap_err();
        assert!(e.to_string().contains("0x0002"), "{e}");
        assert!(decode_wav(b"RIFF\0\0\0\0WAVE").is_err());
        assert!(decode_wav(b"OggS not a wav").is_err());
    }

    #[test]
    fn mono_is_the_float32_channel_mean() {
        let p = Pcm {
            sample_rate: 16000,
            channels: 3,
            samples: vec![0.1, 0.2, 0.4, -1.0, 1.0, 0.5],
        };
        let m = to_mono(&p);
        assert_eq!(
            m,
            vec![(0.1f32 + 0.2 + 0.4) / 3.0, (-1.0f32 + 1.0 + 0.5) / 3.0]
        );
    }

    #[test]
    fn resample_lengths_identity_and_passband() {
        let x: Vec<f32> = (0..14417).map(|i| (i as f32 * 0.01).sin()).collect();
        assert_eq!(resample(&x, 16000, 16000).unwrap(), x);
        assert_eq!(resample(&x[..13230], 44100, 16000).unwrap().len(), 4800);
        assert_eq!(resample(&x[..14417], 48000, 16000).unwrap().len(), 4806);
        assert_eq!(resample(&x[..2403], 8000, 16000).unwrap().len(), 4806);
        // The table-free path (2000 phases) keeps the length rule.
        let odd = resample(&x, 44056, 16000).unwrap();
        assert_eq!(odd.len(), 5236); // ceil(14417 * 16000 / 44056) = ceil(5235.93)

        // A 1 kHz tone keeps unit gain; a 9 kHz tone (above the 8 kHz Nyquist) is removed.
        for (f, want) in [(1000.0f64, 1.0f64), (9000.0, 0.0)] {
            for from in [44100u32, 48000, 44056] {
                let s: Vec<f32> = (0..from as usize / 4)
                    .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / from as f64).sin() as f32)
                    .collect();
                let y = resample(&s, from, 16000).unwrap();
                let mid = &y[800..3200];
                let amp = (2.0 * mid.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()
                    / mid.len() as f64)
                    .sqrt();
                assert!((amp - want).abs() < 1e-4, "{f} Hz from {from}: gain {amp}");
            }
        }
    }
}
