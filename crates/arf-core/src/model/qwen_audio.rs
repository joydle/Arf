//! Qwen3-Omni AUDIO front end (M1 of the port, 2026-09-27): 16 kHz mono PCM ->
//! the `[128, T]` log-mel the audio encoder reads, plus the frame mask and the audio-token count
//! the processor derives from it. Pure CPU, no dependency; decoding a file is [`super::pcm`]'s job
//! and the server's (`arf_serve::qwen_audio`, which shells out to `ffmpeg` for compressed formats).
//!
//! Every parameter below was read in the reference, not assumed. The reference is transformers
//! 5.17.0's `WhisperFeatureExtractor`, which Qwen3-Omni's `preprocessor_config.json` names
//! (`"feature_extractor_type": "WhisperFeatureExtractor"`, Qwen/Qwen3-Omni-30B-A3B-Instruct @
//! 26291f79, copied to `testdata/qwen_audio/`), called with the processor's audio kwargs.
//! `W:` = `models/whisper/feature_extraction_whisper.py`, `A:` = `audio_utils.py`,
//! `P:` = `models/qwen3_omni_moe/processing_qwen3_omni_moe.py`, `S:` =
//! `feature_extraction_sequence_utils.py`.
//!
//! | parameter | value | source |
//! |---|---|---|
//! | sample rate | 16000; any other rate is an ERROR, not resampled | config; `P:100`; `W:265-271` |
//! | n_fft / window | 400 / Hann, **periodic** (`np.hanning(401)[:-1]`) | config; `W:120`; `A:843-890` |
//! | hop | 160 | config; `W:122` |
//! | centering | `center=True`, **reflect** pad of 200 each side (numpy `reflect`: no edge repeat) | `A:914-915,1051-1054` |
//! | spectrum | `rfft` in float64, stored as **complex64**, then `abs(.., float64) ** 2` | `A:1057,1064,1086,1091`; `W:123` |
//! | mel bank | 128 filters, 0-8000 Hz, **Slaney** scale, **Slaney** area norm, over `linspace(0, 8000, 201)` | config; `W:95-103`; `A:546-579,582-615,639-658,736-827` |
//! | floor / log | `max(1e-10, mel)`, **log10** (float64), then cast to float32 | `A:920,1096,1101-1102,1113` |
//! | last frame | **dropped**: `L // 160` frames from `1 + L // 160` | `W:128`; `A:1061` |
//! | dynamic range | `max(x, max(x) - 8)`, the max over the WHOLE clip | `W:129` |
//! | scale | `(x + 4) / 4` | `W:130` |
//! | dither | 0.0 | config; `W:124` |
//! | padding | `padding=True` (longest in the batch), `truncation=False`; so no 30 s cut and no chunking | `P:98-104,160`; `S:199-201` |
//! | pad value | 0.0, on the right | config `padding_value`, `padding_side` |
//! | mask | `attention_mask[:, ::160]`, minus its last column when the padded length % 160 != 0 | `W:332-341`; `S:268-276` |
//! | tokens | `_get_feat_extract_output_lengths(mask.sum(-1), n_window=50)` | `P:99,108-116,168` |
//! | dtype | float32 out | `A:926,1113`; `W:327-330` |
//!
//! Which reference path: without torch (this machine), `W:321-324` takes the numpy path
//! `W:105-133`, and that is what the fixtures were generated with. With torch installed the
//! extractor runs `W:135-168` instead (`torch.stft`, float32 throughout; same window, centering,
//! filters, log and clamps). HF's own docstring puts the two within 1e-5 of each other (`W:107-108,
//! 137-138`); that is NOT measured here (torch is not installed). This file follows the numpy
//! path, including its complex64 rounding of the FFT output, and computes everything else in f64
//! up to the float32 cast, exactly where the reference casts.
//!
//! Traps, each checked against the fixtures:
//! - **Whole-clip normalisation**: the `max - 8` floor uses the max over the entire clip, so a
//!   clip cannot be featurised incrementally (plan §1).
//! - **Batches change features.** The processor pads every clip in a call to the longest with
//!   zeros BEFORE the STFT, and the mask of a shorter clip counts `ceil(L / 160)` frames, one more
//!   than the same clip alone (`L // 160`) when `L % 160 != 0`: that extra frame straddles the
//!   zero padding. [`log_mel_batch`] reproduces this; [`log_mel`] is a batch of one.
//! - **Short clips**: under 160 samples the reference produces 0 frames and then fails on an
//!   empty `max` (`W:129`); here that is an error. Between 160 and 200 samples numpy's `reflect`
//!   pad is wider than the clip and reflects repeatedly (torch's refuses); this follows numpy.
//! - Non-finite samples would propagate NaN through the reference; here they are an error.

use std::sync::OnceLock;

use crate::{ArfError, Result};

/// The only rate the extractor accepts.
pub const SAMPLE_RATE: u32 = 16_000;
/// FFT size and window length (25 ms).
pub const N_FFT: usize = 400;
/// Hop between frames (10 ms).
pub const HOP: usize = 160;
/// Mel filters (`feature_size`).
pub const N_MELS: usize = 128;
/// One-sided FFT bins.
pub const N_FREQ: usize = N_FFT / 2 + 1;
/// The encoder's `n_window` (processor default `P:99`): chunks of `2 * N_WINDOW` mel frames.
pub const N_WINDOW: usize = 50;

/// A log-mel spectrogram, `[n_mels, n_frames]` row-major (the reference's `[1, 128, T]` layout).
#[derive(Debug, Clone, PartialEq)]
pub struct LogMel {
    pub n_mels: usize,
    pub n_frames: usize,
    pub data: Vec<f32>,
}

impl LogMel {
    /// Value at mel bin `m`, frame `t`.
    pub fn at(&self, m: usize, t: usize) -> f32 {
        self.data[m * self.n_frames + t]
    }
}

/// [`log_mel_batch`]'s output: one `[128, T]` per clip, all with the same `T` (the longest clip's
/// `L // 160`), and each clip's valid-frame count (`feature_attention_mask.sum(-1)`; the mask is
/// that many ones followed by zeros).
#[derive(Debug, Clone, PartialEq)]
pub struct MelBatch {
    pub features: Vec<LogMel>,
    pub valid_frames: Vec<usize>,
}

/// Audio tokens the encoder emits for `mel_frames` valid frames (`P:108-116`, the same formula as
/// the model's `_get_feat_extract_output_lengths`): 13 per full 100-frame chunk, plus the tail's
/// three stride-2 convolutions.
pub fn audio_token_count(mel_frames: usize) -> usize {
    let chunk = N_WINDOW * 2;
    let leave = (mel_frames % chunk) as i64;
    let feat = (leave - 1).div_euclid(2) + 1;
    let out = ((feat - 1).div_euclid(2) + 1 - 1).div_euclid(2) + 1;
    (out + (mel_frames / chunk) as i64 * 13) as usize
}

/// A complex number (f64), for the FFT.
#[derive(Debug, Clone, Copy, Default)]
struct C {
    re: f64,
    im: f64,
}

/// Mixed-radix Cooley-Tukey FFT (kissfft's recursion with a generic radix-p butterfly). 400 =
/// 4 * 4 * 5 * 5, so a transform is ~7.2k complex multiply-adds against 80k for the direct DFT.
struct Fft {
    n: usize,
    twiddles: Vec<C>,
    /// (radix p, remaining length m = n_stage / p) per stage.
    factors: Vec<(usize, usize)>,
}

impl Fft {
    fn new(n: usize) -> Fft {
        let twiddles = (0..n)
            .map(|i| {
                let a = -2.0 * std::f64::consts::PI * i as f64 / n as f64;
                C {
                    re: a.cos(),
                    im: a.sin(),
                }
            })
            .collect();
        let mut factors = Vec::new();
        let mut rest = n;
        let mut p = 4;
        while rest > 1 {
            while !rest.is_multiple_of(p) {
                p = match p {
                    4 => 2,
                    2 => 3,
                    _ => p + 2,
                };
                if p * p > rest {
                    p = rest;
                }
            }
            rest /= p;
            factors.push((p, rest));
        }
        Fft {
            n,
            twiddles,
            factors,
        }
    }

    fn forward(&self, input: &[C], out: &mut [C]) {
        debug_assert_eq!(input.len(), self.n);
        self.work(out, input, 0, 1, 0);
    }

    fn work(&self, out: &mut [C], input: &[C], off: usize, fstride: usize, stage: usize) {
        let (p, m) = self.factors[stage];
        if m == 1 {
            for (j, o) in out.iter_mut().take(p).enumerate() {
                *o = input[off + j * fstride];
            }
        } else {
            for j in 0..p {
                self.work(
                    &mut out[j * m..(j + 1) * m],
                    input,
                    off + j * fstride,
                    fstride * p,
                    stage + 1,
                );
            }
        }
        let mut scratch = [C::default(); 8];
        let scratch = &mut scratch[..p.min(8)];
        let mut big = Vec::new();
        let scratch: &mut [C] = if p <= 8 {
            scratch
        } else {
            big.resize(p, C::default());
            &mut big
        };
        for u in 0..m {
            for q1 in 0..p {
                scratch[q1] = out[u + q1 * m];
            }
            for q1 in 0..p {
                let k = u + q1 * m;
                let mut acc = scratch[0];
                let mut tw = 0usize;
                for s in scratch.iter().skip(1) {
                    tw = (tw + fstride * k) % self.n;
                    let t = self.twiddles[tw];
                    acc.re += s.re * t.re - s.im * t.im;
                    acc.im += s.re * t.im + s.im * t.re;
                }
                out[k] = acc;
            }
        }
    }
}

/// `A:546-579` with `mel_scale="slaney"`, scalar form.
fn hz_to_mel_slaney(f: f64) -> f64 {
    let (min_log_hz, min_log_mel) = (1000.0, 15.0);
    let logstep = 27.0 / 6.4f64.ln();
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() * logstep
    } else {
        3.0 * f / 200.0
    }
}

/// `A:582-615` with `mel_scale="slaney"`.
fn mel_to_hz_slaney(m: f64) -> f64 {
    let (min_log_hz, min_log_mel) = (1000.0, 15.0);
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        200.0 * m / 3.0
    }
}

/// The reference filter bank, `[N_FREQ][N_MELS]` (`A:736-827` as `W:95-103` calls it), in f64.
pub fn mel_filter_bank() -> Vec<f64> {
    let (fmin, fmax) = (0.0f64, 8000.0f64);
    let n = N_MELS + 2;
    // np.linspace: start + i * ((stop - start) / (num - 1)), the last point set to stop exactly.
    let (mmin, mmax) = (hz_to_mel_slaney(fmin), hz_to_mel_slaney(fmax));
    let step = (mmax - mmin) / (n - 1) as f64;
    let filter_hz: Vec<f64> = (0..n)
        .map(|i| {
            let m = if i == n - 1 {
                mmax
            } else {
                i as f64 * step + mmin
            };
            mel_to_hz_slaney(m)
        })
        .collect();
    // fft_freqs = np.linspace(0, sampling_rate // 2, N_FREQ): exactly i * 40 Hz.
    let fft_step = (SAMPLE_RATE / 2) as f64 / (N_FREQ - 1) as f64;
    let diff: Vec<f64> = filter_hz.windows(2).map(|w| w[1] - w[0]).collect();
    let mut fb = vec![0.0f64; N_FREQ * N_MELS];
    for i in 0..N_FREQ {
        let f = if i == N_FREQ - 1 {
            (SAMPLE_RATE / 2) as f64
        } else {
            i as f64 * fft_step
        };
        for j in 0..N_MELS {
            let down = -(filter_hz[j] - f) / diff[j];
            let up = (filter_hz[j + 2] - f) / diff[j + 1];
            let tri = down.min(up).max(0.0);
            let enorm = 2.0 / (filter_hz[j + 2] - filter_hz[j]);
            fb[i * N_MELS + j] = tri * enorm;
        }
    }
    fb
}

/// numpy `reflect` padding index: the periodic extension (period `2 (n - 1)`) of the mirror
/// without edge repeat, which is what `np.pad(.., mode="reflect")` produces even when the pad is
/// wider than the array (it reflects repeatedly).
fn reflect_index(i: i64, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as i64 - 1);
    let m = i.rem_euclid(period);
    if m < n as i64 {
        m as usize
    } else {
        (period - m) as usize
    }
}

/// The front end: window, filter bank (sparse per mel), FFT plan. Built once ([`log_mel`] keeps
/// one in a `OnceLock`).
pub struct MelFrontEnd {
    window: Vec<f64>,
    /// Per mel: first nonzero FFT bin and the weights from there.
    filters: Vec<(usize, Vec<f64>)>,
    fft: Fft,
}

impl Default for MelFrontEnd {
    fn default() -> Self {
        Self::new()
    }
}

impl MelFrontEnd {
    pub fn new() -> MelFrontEnd {
        // np.hanning(M): 0.5 + 0.5 * cos(pi * n / (M - 1)) for n = arange(1 - M, M, 2); periodic
        // takes M = N_FFT + 1 and drops the last point (A:876-890).
        let m = (N_FFT + 1) as f64;
        let window = (0..N_FFT)
            .map(|i| {
                let n = (1.0 - m) + 2.0 * i as f64;
                0.5 + 0.5 * (std::f64::consts::PI * n / (m - 1.0)).cos()
            })
            .collect();
        let fb = mel_filter_bank();
        let filters = (0..N_MELS)
            .map(|j| {
                let col: Vec<f64> = (0..N_FREQ).map(|i| fb[i * N_MELS + j]).collect();
                let lo = col.iter().position(|&w| w != 0.0).unwrap_or(0);
                let hi = col.iter().rposition(|&w| w != 0.0).map_or(lo, |h| h + 1);
                (lo, col[lo..hi].to_vec())
            })
            .collect();
        MelFrontEnd {
            window,
            filters,
            fft: Fft::new(N_FFT),
        }
    }

    /// `log10(max(1e-10, mel))` as float32 for the first `n_frames` frames of `wave` (already at
    /// its batch length), `[N_MELS][n_frames]`. The reference computes one more frame and drops
    /// it (`W:128`); it is never computed here.
    fn log10_mel(&self, wave: &[f32], n_frames: usize) -> Vec<f32> {
        let half = (N_FFT / 2) as i64;
        let mut buf = vec![C::default(); N_FFT];
        let mut spec = vec![C::default(); N_FFT];
        let mut power = [0.0f64; N_FREQ];
        let mut out = vec![0.0f32; N_MELS * n_frames];
        for t in 0..n_frames {
            let start = (t * HOP) as i64 - half;
            for (j, b) in buf.iter_mut().enumerate() {
                let s = wave[reflect_index(start + j as i64, wave.len())] as f64;
                *b = C {
                    re: s * self.window[j],
                    im: 0.0,
                };
            }
            self.fft.forward(&buf, &mut spec);
            for (k, p) in power.iter_mut().enumerate() {
                // The reference stores the rfft in a complex64 array (A:1064,1086).
                let (re, im) = (spec[k].re as f32 as f64, spec[k].im as f32 as f64);
                *p = re * re + im * im;
            }
            for (j, (lo, w)) in self.filters.iter().enumerate() {
                let mel: f64 = w.iter().zip(&power[*lo..]).map(|(a, b)| a * b).sum();
                out[j * n_frames + t] = mel.max(1e-10).log10() as f32;
            }
        }
        out
    }

    /// One clip at its own length (a batch of one). See [`log_mel`].
    pub fn log_mel(&self, pcm: &[f32], sr: u32) -> Result<LogMel> {
        let mut b = self.log_mel_batch(&[pcm], sr)?;
        Ok(b.features.pop().expect("one clip"))
    }

    /// Several clips as ONE processor call: see [`log_mel_batch`].
    pub fn log_mel_batch(&self, clips: &[&[f32]], sr: u32) -> Result<MelBatch> {
        if sr != SAMPLE_RATE {
            return Err(ArfError::other(format!(
                "qwen audio: the feature extractor takes {SAMPLE_RATE} Hz audio, got {sr} Hz \
                 (resample first)"
            )));
        }
        if clips.is_empty() {
            return Err(ArfError::other("qwen audio: no clips"));
        }
        let longest = clips.iter().map(|c| c.len()).max().unwrap_or(0);
        if longest < HOP {
            return Err(ArfError::other(format!(
                "qwen audio: a clip needs at least {HOP} samples (10 ms at 16 kHz) to make one \
                 frame, the longest has {longest}"
            )));
        }
        if let Some(i) = clips.iter().position(|c| c.iter().any(|v| !v.is_finite())) {
            return Err(ArfError::other(format!(
                "qwen audio: clip {i} has a non-finite sample"
            )));
        }
        let n_frames = longest / HOP;
        let mut features = Vec::with_capacity(clips.len());
        let mut valid_frames = Vec::with_capacity(clips.len());
        let mut padded = Vec::new();
        for clip in clips {
            let wave: &[f32] = if clip.len() == longest {
                clip
            } else {
                padded.clear();
                padded.extend_from_slice(clip);
                padded.resize(longest, 0.0);
                &padded
            };
            let mut x = self.log10_mel(wave, n_frames);
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let floor = max - 8.0;
            for v in &mut x {
                *v = (v.max(floor) + 4.0) / 4.0;
            }
            features.push(LogMel {
                n_mels: N_MELS,
                n_frames,
                data: x,
            });
            // Mask: ones for samples < len, sampled every HOP -> ceil(len / HOP) ones, over
            // ceil(longest / HOP) columns minus the last when longest % HOP != 0 (= n_frames).
            valid_frames.push(clip.len().div_ceil(HOP).min(n_frames));
        }
        Ok(MelBatch {
            features,
            valid_frames,
        })
    }
}

fn front_end() -> &'static MelFrontEnd {
    static FE: OnceLock<MelFrontEnd> = OnceLock::new();
    FE.get_or_init(MelFrontEnd::new)
}

/// The Qwen3-Omni log-mel of one 16 kHz mono clip: `[128, L / 160]`, what the processor's
/// `input_features[0]` holds for a single-audio call. Every frame is valid.
pub fn log_mel(pcm: &[f32], sr: u32) -> Result<LogMel> {
    front_end().log_mel(pcm, sr)
}

/// Several clips as ONE processor call (`audio=[a, b, ..]`): each is zero-padded to the longest
/// before the STFT, every output has the longest clip's frame count, and `valid_frames[i]` is
/// `min(ceil(len_i / 160), T)`. For clips that are not the longest this is not what [`log_mel`]
/// gives the same clip alone; see the module doc.
pub fn log_mel_batch(clips: &[&[f32]], sr: u32) -> Result<MelBatch> {
    front_end().log_mel_batch(clips, sr)
}

/// Decoded PCM -> the extractor's input, the way the model card's loader does it
/// (`librosa.load(sr=16000)`): mean over channels, then resample to 16 kHz. See [`super::pcm`]
/// for what the resampler does and does not match.
pub fn pcm_to_16k_mono(pcm: &super::pcm::Pcm) -> Result<Vec<f32>> {
    let mono = super::pcm::to_mono(pcm);
    super::pcm::resample(&mono, pcm.sample_rate, SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::super::pcm;
    use super::*;
    use serde_json::Value;
    use std::path::PathBuf;

    #[test]
    fn fft_matches_the_direct_dft() {
        let fft = Fft::new(N_FFT);
        assert_eq!(fft.factors, vec![(4, 100), (4, 25), (5, 5), (5, 1)]);
        let x: Vec<C> = (0..N_FFT)
            .map(|i| C {
                re: ((i * 37 % 101) as f64 - 50.0) / 50.0,
                im: 0.0,
            })
            .collect();
        let mut y = vec![C::default(); N_FFT];
        fft.forward(&x, &mut y);
        let mut worst = 0.0f64;
        for (k, yk) in y.iter().enumerate() {
            let (mut re, mut im) = (0.0, 0.0);
            for (n, xn) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * ((k * n) % N_FFT) as f64 / N_FFT as f64;
                re += xn.re * a.cos();
                im += xn.re * a.sin();
            }
            worst = worst.max((re - yk.re).abs()).max((im - yk.im).abs());
        }
        assert!(worst < 1e-10, "fft vs dft max-abs {worst}");
        // A non-400 size with a radix above 5 still transforms correctly (7 * 3 = 21).
        let f21 = Fft::new(21);
        let x: Vec<C> = (0..21)
            .map(|i| C {
                re: i as f64,
                im: 0.0,
            })
            .collect();
        let mut y = vec![C::default(); 21];
        f21.forward(&x, &mut y);
        assert!((y[0].re - 210.0).abs() < 1e-9);
    }

    #[test]
    fn filter_bank_is_slaney_and_every_filter_is_nonzero() {
        let fb = mel_filter_bank();
        for j in 0..N_MELS {
            assert!(
                (0..N_FREQ).any(|i| fb[i * N_MELS + j] > 0.0),
                "mel {j} is empty"
            );
        }
        // Slaney scale is linear below 1 kHz: 3 mels per 200 Hz.
        assert_eq!(hz_to_mel_slaney(200.0), 3.0);
        assert!((mel_to_hz_slaney(hz_to_mel_slaney(5000.0)) - 5000.0).abs() < 1e-9);
        // The DC bin feeds nothing (filter 0 starts at 0 Hz with weight 0).
        assert!((0..N_MELS).all(|j| fb[j] == 0.0));
    }

    #[test]
    fn reflect_padding_is_numpys() {
        // Printed by numpy 2.5.3: np.pad(np.arange(4), 6, mode="reflect") and
        // np.pad(np.arange(3), 7, mode="reflect") -- pads wider than the array.
        assert_eq!(
            (-6..10).map(|i| reflect_index(i, 4)).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3]
        );
        assert_eq!(
            (-7..10).map(|i| reflect_index(i, 3)).collect::<Vec<_>>(),
            vec![1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1]
        );
        assert_eq!(reflect_index(-3, 1), 0);
    }

    #[test]
    fn token_counts_match_the_processor_formula() {
        // P:108-116 by hand: 100 frames -> 13; 300 -> 39 (the plan's 3.0077 s check); 1 -> 1.
        assert_eq!(audio_token_count(100), 13);
        assert_eq!(audio_token_count(300), 39);
        assert_eq!(audio_token_count(1), 1);
        assert_eq!(audio_token_count(150), 13 + 7);
        assert_eq!(audio_token_count(3100), 31 * 13);
        // 0 leftover frames: (0 - 1) // 2 + 1 = 0 in Python floor division -> +1 from the last
        // stage: ((0 - 1) // 2 + 1 - 1) // 2 + 1 = 0. 200 frames -> 26.
        assert_eq!(audio_token_count(200), 26);
    }

    #[test]
    fn wrong_rate_short_and_nonfinite_clips_are_errors() {
        let x = vec![0.1f32; 16000];
        assert!(log_mel(&x, 44100)
            .unwrap_err()
            .to_string()
            .contains("16000 Hz"));
        assert!(log_mel(&x[..159], 16000).is_err());
        assert_eq!(log_mel(&x[..160], 16000).unwrap().n_frames, 1);
        let mut y = x.clone();
        y[5] = f32::NAN;
        assert!(log_mel(&y, 16000).is_err());
    }

    // ---- parity against the reference fixtures (testdata/qwen_audio, made by gen_ref.py) ----

    fn dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/qwen_audio")
    }

    fn manifest() -> Value {
        let s = std::fs::read_to_string(dir().join("manifest.json")).expect("manifest.json");
        serde_json::from_str(&s).unwrap()
    }

    fn read_f32(name: &str) -> Vec<f32> {
        std::fs::read(dir().join(name))
            .expect(name)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn splitmix_i16(seed: u64, n: usize) -> Vec<i32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 48) as u16 as i16) as i32
            })
            .collect()
    }

    fn tones_i16(n: usize, sr: f64, tones: &[(f64, f64)]) -> Vec<i32> {
        (0..n)
            .map(|i| {
                let mut v = 0.0f64;
                for &(f, a) in tones {
                    v += a * (2.0 * std::f64::consts::PI * f * i as f64 / sr).sin();
                }
                ((v * 32768.0).round_ties_even() as i64).clamp(-32768, 32767) as i32
            })
            .collect()
    }

    fn fnv(x: &[i32]) -> String {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for &v in x {
            for b in (v as i16).to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        format!("{h:016x}")
    }

    /// Rebuild one generated input from its manifest entry, checking the generator's checksum.
    fn input(m: &Value) -> Vec<i32> {
        let n = m["n"].as_u64().unwrap() as usize;
        let tones = || -> Vec<(f64, f64)> {
            m["tones"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| (t[0].as_f64().unwrap(), t[1].as_f64().unwrap()))
                .collect()
        };
        let shift = m["shift"].as_u64().unwrap_or(0) as u32;
        let x = match m["kind"].as_str().unwrap() {
            "zeros" => vec![0; n],
            "tones" => tones_i16(n, m["sr"].as_f64().unwrap(), &tones()),
            "noise" => splitmix_i16(m["seed"].as_u64().unwrap(), n)
                .into_iter()
                .map(|v| v >> shift)
                .collect(),
            "tones+noise" => tones_i16(n, m["sr"].as_f64().unwrap(), &tones())
                .into_iter()
                .zip(splitmix_i16(m["seed"].as_u64().unwrap(), n))
                .map(|(a, b)| (a + (b >> shift)).clamp(-32768, 32767))
                .collect(),
            k => panic!("unknown input kind {k}"),
        };
        assert_eq!(
            fnv(&x),
            m["fnv"].as_str().unwrap(),
            "input {m} does not match the generator's"
        );
        x
    }

    fn to_f32(x: &[i32]) -> Vec<f32> {
        x.iter().map(|&v| v as f32 / 32768.0).collect()
    }

    /// Comparison of one clip against its fixture: (max-abs over the stored frames,
    /// rel = ||a-b|| / ||b|| over them, |sum diff| / count and |sumsq diff| / sumsq over ALL).
    fn compare(got: &LogMel, clip: &Value) -> (f64, f64, f64, f64) {
        let frames: Vec<usize> = clip["frames"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let want = read_f32(clip["file"].as_str().unwrap());
        assert_eq!(want.len(), frames.len() * N_MELS);
        let (mut max_abs, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
        for (k, &t) in frames.iter().enumerate() {
            for m in 0..N_MELS {
                let (a, b) = (got.at(m, t) as f64, want[k * N_MELS + m] as f64);
                max_abs = max_abs.max((a - b).abs());
                num += (a - b) * (a - b);
                den += b * b;
            }
        }
        let sum: f64 = got.data.iter().map(|&v| v as f64).sum();
        let sumsq: f64 = got.data.iter().map(|&v| (v as f64).powi(2)).sum();
        let mn = got.data.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let mx = got.data.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        max_abs = max_abs
            .max((mn - clip["min"].as_f64().unwrap()).abs())
            .max((mx - clip["max"].as_f64().unwrap()).abs());
        let n = got.data.len() as f64;
        (
            max_abs,
            (num / den.max(1e-30)).sqrt(),
            (sum - clip["sum"].as_f64().unwrap()).abs() / n,
            (sumsq - clip["sumsq"].as_f64().unwrap()).abs() / clip["sumsq"].as_f64().unwrap(),
        )
    }

    /// The M1 gate (plan §6): max-abs <= 1e-4 on `[128, T]`, exact frame counts and masks. The
    /// reference path is float64 up to one float32 cast of log10(mel) (A:1113), and so is this
    /// one, so the expected error is float32 rounding of values of magnitude <= ~2.5 (ulp ~2.4e-7)
    /// plus summation-order differences in the FFT and the mel dot, far below 1e-4. The rel gate
    /// (1e-5) and the whole-array mean |sum diff| gate (1e-5 per element) cover every frame, not
    /// only the stored ones.
    #[test]
    fn log_mel_matches_the_whisper_extractor_reference() {
        let man = manifest();
        let mut report = Vec::new();
        for case in man["mel"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let clips: Vec<Vec<f32>> = if let Some(wav) = case.get("wav") {
                // (e) stereo: decode the stdlib-written WAV, check it IS the generated input,
                // then mix down as librosa does.
                let bytes = std::fs::read(dir().join(wav.as_str().unwrap())).unwrap();
                let p = pcm::decode_wav(&bytes).unwrap();
                assert_eq!((p.sample_rate, p.channels), (16000, 2));
                let im = &case["inputs"][0];
                let n = im["n"].as_u64().unwrap() as usize;
                let left = tones_i16(n, 16000.0, &[(440.0, 0.4)]);
                let right: Vec<i32> = splitmix_i16(2, n).into_iter().map(|v| v >> 1).collect();
                assert_eq!(fnv(&left), im["left_fnv"].as_str().unwrap());
                assert_eq!(fnv(&right), im["right_fnv"].as_str().unwrap());
                for i in 0..n {
                    assert_eq!(p.samples[2 * i], left[i] as f32 / 32768.0);
                    assert_eq!(p.samples[2 * i + 1], right[i] as f32 / 32768.0);
                }
                vec![pcm_to_16k_mono(&p).unwrap()]
            } else {
                case["inputs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|m| to_f32(&input(m)))
                    .collect()
            };
            let refs: Vec<&[f32]> = clips.iter().map(|c| c.as_slice()).collect();
            let batch = log_mel_batch(&refs, 16000).unwrap();
            let want_frames = case["n_frames"].as_u64().unwrap() as usize;
            let want_valid: Vec<usize> = case["mask_sums"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(batch.valid_frames, want_valid, "{name}: mask sums");
            for (i, clip) in case["clips"].as_array().unwrap().iter().enumerate() {
                let got = &batch.features[i];
                assert_eq!(
                    (got.n_mels, got.n_frames),
                    (N_MELS, want_frames),
                    "{name}[{i}]: shape"
                );
                if let Some(mask) = clip["mask"].as_array() {
                    let ones = mask.iter().take_while(|v| v.as_u64() == Some(1)).count();
                    assert!(
                        mask[ones..].iter().all(|v| v.as_u64() == Some(0)),
                        "{name}: mask not a prefix"
                    );
                    assert_eq!(ones, want_valid[i]);
                }
                let (max_abs, rel, mean_sum, sumsq) = compare(got, clip);
                report.push(format!(
                    "{name}[{i}] T={want_frames} valid={} max_abs={max_abs:.3e} rel={rel:.3e} \
                     mean|dsum|={mean_sum:.1e} dsumsq/sumsq={sumsq:.1e}",
                    want_valid[i]
                ));
                assert!(max_abs <= 1e-4, "{name}[{i}]: max-abs {max_abs:.3e} > 1e-4");
                assert!(rel <= 1e-5, "{name}[{i}]: rel {rel:.3e} > 1e-5");
                assert!(
                    mean_sum <= 1e-5,
                    "{name}[{i}]: whole-array mean diff {mean_sum:.3e}"
                );
                assert!(
                    sumsq <= 1e-5,
                    "{name}[{i}]: whole-array sumsq diff {sumsq:.3e}"
                );
            }
            // A single clip through the batch-of-one API is the same thing.
            if clips.len() == 1 {
                assert_eq!(log_mel(&clips[0], 16000).unwrap(), batch.features[0]);
            }
        }
        // Silence: every value is (log10(1e-10) + 4) / 4 = -1.5 exactly.
        let s = log_mel(&vec![0.0; 8000], 16000).unwrap();
        assert!(s.data.iter().all(|&v| v == -1.5));
        for line in &report {
            eprintln!("[qwen-audio parity] {line}");
        }
    }

    /// The input-side gate (plan §6 M1: max-abs <= 2e-2 against the reference's 16 kHz PCM, which
    /// is soxr HQ through librosa). Also reports what the resampler difference does to the mel.
    #[test]
    fn decode_and_resample_are_within_the_plan_gate_of_librosa_soxr() {
        let man = manifest();
        for case in man["resample"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let sr = case["sr"].as_u64().unwrap() as u32;
            let chans: Vec<Vec<i32>> = case["channels"]
                .as_array()
                .unwrap()
                .iter()
                .map(input)
                .collect();
            let n = chans[0].len();
            let mut inter = Vec::with_capacity(n * chans.len());
            for i in 0..n {
                for c in &chans {
                    inter.push(c[i] as f32 / 32768.0);
                }
            }
            let p = pcm::Pcm {
                sample_rate: sr,
                channels: chans.len(),
                samples: inter,
            };
            let got = pcm_to_16k_mono(&p).unwrap();
            let want = read_f32(case["file"].as_str().unwrap());
            assert_eq!(
                got.len(),
                case["n_out"].as_u64().unwrap() as usize,
                "{name}: length"
            );
            assert_eq!(got.len(), want.len());
            let max_abs = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let (gm, wm) = (
                log_mel(&got, 16000).unwrap(),
                log_mel(&want, 16000).unwrap(),
            );
            let (mut num, mut den, mut mel_max) = (0.0f64, 0.0f64, 0.0f64);
            for (a, b) in gm.data.iter().zip(&wm.data) {
                let d = (*a - *b) as f64;
                num += d * d;
                den += (*b as f64).powi(2);
                mel_max = mel_max.max(d.abs());
            }
            let mel_rel = (num / den).sqrt();
            eprintln!(
                "[qwen-audio resample] {name}: {sr} Hz x{} -> {} samples, pcm max-abs \
                 {max_abs:.3e}, mel rel {mel_rel:.3e} max-abs {mel_max:.3e}",
                chans.len(),
                got.len()
            );
            assert!(max_abs <= 2e-2, "{name}: pcm max-abs {max_abs:.3e} > 2e-2");
            assert!(mel_rel <= 1e-2, "{name}: mel rel {mel_rel:.3e} > 1e-2");
        }
    }
}
