//! Qwen3-Omni AUDIO input for the HTTP layer (2026-09-27, M1 of the port): an
//! audio reference or payload -> 16 kHz mono f32 PCM, ready for
//! `arf_core::model::qwen_audio::log_mel`. M3 (the same day) wires it into `/v1/chat/completions`
//! (`input_audio` / `audio_url` parts, `arf-serve --audio-tower`); that half is the second part
//! of this file, from [`AudioTower`] on.
//!
//! - WAV (PCM 8/16/24/32-bit, float 32/64, any channel count) decodes in pure Rust
//!   (`arf_core::model::pcm::decode_wav`).
//! - Anything else (mp3, m4a/aac, ogg/opus, flac, a WAV with a compressed payload, a video's audio
//!   track) shells out to `ffmpeg` from PATH, as `qwen_video` does, and asks it for float WAV at the
//!   file's own rate and channel count. Mixdown and resampling are then Arf's, identical for both
//!   routes, in librosa's order (mean over channels, then resample; see `arf_core::model::pcm` for
//!   how the resampler relates to the reference's soxr). A server without ffmpeg answers a
//!   compressed clip with an error naming the gap; WAV still works.
//! - Same safety rules as `qwen_video`: ffmpeg runs with `-protocol_whitelist file` on
//!   `file:<abs path>`, so a playlist-like input cannot make the server fetch a URL and a path
//!   cannot be read as an option or a protocol; a data URL's bytes go to a private temp file
//!   (some containers need a seekable input) that is removed on drop; file paths need
//!   `ARF_IMAGE_FILES=1`, the gate image and video files already use.
//!
//! UNVERIFIED: compressed decoders differ between ffmpeg and the reference's libsndfile/mpg123
//! (gapless trimming of mp3 encoder delay in particular). Only lossless FLAC is gated here, where
//! the two routes must agree bit for bit.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use arf_core::model::pcm::{decode_wav, is_wav, Pcm};
use arf_core::model::qwen_audio::{
    audio_token_count, log_mel_batch, pcm_to_16k_mono, LogMel, MelBatch, SAMPLE_RATE,
};
use arf_core::model::qwen_audio_encoder::AudioEncoderConfig;
use arf_core::scheduler::ImagePrompt;
use arf_core::Tokenizer;

use crate::http::ChatMessage;

/// Load one audio reference: a `data:audio/...;base64,...` URL, (with `ARF_IMAGE_FILES=1`) a
/// `file://` URL / absolute path, or (with `ARF_MEDIA_URLS=1`, `remote_media`) an `http(s)://` URL.
/// Returns 16 kHz mono PCM.
pub fn load_audio(url: &str) -> Result<Vec<f32>, String> {
    if crate::remote_media::is_remote(url) {
        return decode_audio_bytes(&crate::remote_media::fetch(url, "audio")?);
    }
    if url.starts_with("data:") {
        let b64 = url
            .find("base64,")
            .map(|i| &url[i + "base64,".len()..])
            .ok_or("audio data URL must be base64-encoded")?;
        return decode_audio_base64(b64);
    }
    let p = url.strip_prefix("file://").unwrap_or(url);
    if !p.starts_with('/') {
        return Err(
            "audio url must be a data: URL, an http(s) URL, a file:// URL or an absolute path"
                .into(),
        );
    }
    if std::env::var_os("ARF_IMAGE_FILES").is_none() {
        return Err(
            "audio file paths are disabled (set ARF_IMAGE_FILES=1 on the server to allow)".into(),
        );
    }
    let path = Path::new(p);
    if !path.is_file() {
        return Err(format!("audio file {p}: not found"));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("audio file {p}: {e}"))?;
    if is_wav(&bytes) {
        if let Ok(pcm) = decode_wav(&bytes) {
            return to_16k(&pcm);
        }
    }
    to_16k(&ffmpeg_decode("ffmpeg", path)?)
}

/// OpenAI-style `input_audio.data`: bare base64 of the file's bytes (the `format` field is a hint
/// only; the bytes are sniffed).
pub fn decode_audio_base64(b64: &str) -> Result<Vec<f32>, String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("audio base64 decode: {e}"))?;
    decode_audio_bytes(&bytes)
}

/// Encoded audio bytes -> 16 kHz mono PCM: WAV natively, everything else through ffmpeg.
pub fn decode_audio_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    decode_audio_bytes_with(bytes, "ffmpeg")
}

fn decode_audio_bytes_with(bytes: &[u8], ffmpeg: &str) -> Result<Vec<f32>, String> {
    if bytes.is_empty() {
        return Err("audio payload is empty".into());
    }
    if is_wav(bytes) {
        // A WAV whose payload is not plain PCM/float (ADPCM, mu-law, ...) falls through to ffmpeg.
        if let Ok(pcm) = decode_wav(bytes) {
            return to_16k(&pcm);
        }
    }
    let file = TempAudio::write(bytes)?;
    to_16k(&ffmpeg_decode(ffmpeg, &file.path)?)
}

fn to_16k(pcm: &Pcm) -> Result<Vec<f32>, String> {
    if pcm.frames() == 0 {
        return Err("audio has no samples".into());
    }
    pcm_to_16k_mono(pcm).map_err(|e| e.to_string())
}

/// A data payload on disk for ffmpeg, removed on drop.
struct TempAudio {
    path: PathBuf,
}

impl TempAudio {
    fn write(bytes: &[u8]) -> Result<TempAudio, String> {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("arf-audio-{}-{n}.bin", std::process::id()));
        std::fs::write(&path, bytes).map_err(|e| format!("audio temp file: {e}"))?;
        Ok(TempAudio { path })
    }
}

impl Drop for TempAudio {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn spawn_err(name: &str, e: std::io::Error) -> String {
    if e.kind() == std::io::ErrorKind::NotFound {
        format!(
            "compressed audio input needs `{name}` on the server's PATH (install ffmpeg, or send \
             WAV)"
        )
    } else {
        format!("running {name}: {e}")
    }
}

/// The first audio stream of `path` as float WAV at its native rate and channel count, decoded by
/// ffmpeg (`-protocol_whitelist file`, input spelled `file:<abs path>`).
fn ffmpeg_decode(ffmpeg: &str, path: &Path) -> Result<Pcm, String> {
    let mut child = Command::new(ffmpeg)
        .stdin(Stdio::null())
        .args([
            "-nostdin",
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-i",
        ])
        .arg(format!("file:{}", path.display()))
        .args([
            "-vn",
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_f32le",
            "-f",
            "wav",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| spawn_err(ffmpeg, e))?;
    // Drain stderr on its own thread so a chatty ffmpeg cannot block on a full pipe.
    let mut err_pipe = child.stderr.take().expect("piped");
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err_pipe.read_to_string(&mut s);
        s
    });
    let mut wav = Vec::new();
    let read = child.stdout.take().expect("piped").read_to_end(&mut wav);
    let status = child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
    let stderr = err_thread.join().unwrap_or_default();
    if !status.success() || read.is_err() {
        return Err(format!(
            "ffmpeg could not decode the audio: {}",
            stderr.trim()
        ));
    }
    decode_wav(&wav).map_err(|e| format!("ffmpeg output: {e}"))
}

// ---------------------------------------------------------------------------------------------
// M3 (2026-09-27): audio in a chat prompt.
//
// How a clip reaches the Thinker, each step read in the reference (transformers 5.17,
// `processing_qwen3_omni_moe.py` = P:, `modeling_qwen3_omni_moe.py` = M:):
// 1. The model's chat template renders an audio content part as
//    `<|audio_start|><|audio_pad|><|audio_end|>` in content-part order (the GGUF's
//    `tokenizer.chat_template`, read 2026-09-27). Arf's chat path flattens text parts into
//    `content`, so the parser records where each clip sat (`ChatMessage::audio_at`); here a
//    private-use SENTINEL goes at those offsets before the template renders, and each is replaced
//    by the expanded block after it.
// 2. The processor featurises every clip of the request in ONE call (padded to the longest,
//    `P:98-104`) and repeats `<|audio_pad|>` once per encoder token (`P:239-240`), i.e.
//    `audio_token_count(valid_frames)` -- 13 per second of audio.
// 3. The encoder runs that batch as one call; its `[sum tokens, 2048]` rows are
//    `masked_scatter`ed over the pad rows in order (`M:2121-2127`).
// 4. Rope: an audio span's positions are `arange(n)` on all three M-RoPE axes (`M:356-365`), so a
//    text + audio prompt's positions are the plain cumulative ones and the rows rotate exactly as
//    text does. Hence `ImagePrompt { mrope: None, causal: true }`: an embedding override on an
//    otherwise ordinary text prompt, attended causally like the rest of the Thinker's input.
// ---------------------------------------------------------------------------------------------

/// Stand-in for one audio part inside message content while the template renders (private-use
/// code points, as `qwen_image::IMAGE_SENTINEL`, but a different string: the two are replaced
/// by different code).
pub const AUDIO_SENTINEL: &str = "\u{E000}arf-audio\u{E001}";
pub const AUDIO_START: &str = "<|audio_start|>";
pub const AUDIO_PAD: &str = "<|audio_pad|>";
pub const AUDIO_END: &str = "<|audio_end|>";

/// Audio tokens per second of 16 kHz audio (13 per 100-frame chunk), for the size guard only; the
/// real count is [`audio_token_count`] of the mel.
const TOKENS_PER_SECOND: usize = 13;

/// The server's Qwen3-Omni audio encoder (`--audio-tower`, the `thinker.audio_tower.*`
/// safetensors): on Metal, `arf_gpu`'s `QwenAudio` (GPU encoder, `ARF_AUDIO_CPU=1` for the CPU
/// one, and a GPU failure falls back to the CPU); elsewhere the CPU encoder.
pub struct AudioTower {
    pub cfg: AudioEncoderConfig,
    #[cfg(all(feature = "wgpu", target_os = "macos"))]
    inner: arf_gpu::gpu::metal::qwen_audio::QwenAudio,
    #[cfg(not(all(feature = "wgpu", target_os = "macos")))]
    inner: arf_core::model::qwen_audio_encoder::QwenAudioEncoder,
}

impl AudioTower {
    pub fn load(path: &Path) -> Result<Self, String> {
        #[cfg(all(feature = "wgpu", target_os = "macos"))]
        let inner = arf_gpu::gpu::metal::qwen_audio::QwenAudio::load(path)?;
        #[cfg(not(all(feature = "wgpu", target_os = "macos")))]
        let inner = arf_core::model::qwen_audio_encoder::QwenAudioEncoder::load(path)
            .map_err(|e| e.to_string())?;
        Ok(AudioTower {
            cfg: inner.cfg.clone(),
            inner,
        })
    }

    /// "GPU" or "CPU": which encoder serves requests (before any fallback).
    pub fn backend(&self) -> &'static str {
        #[cfg(all(feature = "wgpu", target_os = "macos"))]
        return self.inner.backend();
        #[cfg(not(all(feature = "wgpu", target_os = "macos")))]
        "CPU"
    }

    /// Width of the rows it emits (the Thinker's hidden size, 2048).
    pub fn output_dim(&self) -> usize {
        self.cfg.output_dim
    }

    /// One processor batch as one encoder call: `[sum tokens, output_dim]`.
    pub fn encode_batch(&self, clips: &[(&LogMel, usize)]) -> Result<Vec<f32>, String> {
        #[cfg(all(feature = "wgpu", target_os = "macos"))]
        return self.inner.encode_batch(clips);
        #[cfg(not(all(feature = "wgpu", target_os = "macos")))]
        self.inner
            .encode_batch(clips, None)
            .map_err(|e| e.to_string())
    }
}

/// Every audio clip of one request, featurised as ONE processor call, and the encoder tokens each
/// clip becomes (its `<|audio_pad|>` run length), in prompt order.
#[derive(Debug, Clone)]
pub struct StagedAudio {
    pub mels: MelBatch,
    pub tokens: Vec<usize>,
}

impl StagedAudio {
    /// Featurise decoded 16 kHz clips together (`log_mel_batch`: padded to the longest, as the
    /// processor does for `audio=[a, b, ..]`), refusing more audio than `max_tokens` of context
    /// could hold before spending the mel on it.
    pub fn from_pcm(clips: &[Vec<f32>], max_tokens: usize) -> Result<Self, String> {
        let samples: usize = clips.iter().map(Vec::len).sum();
        let est = samples.div_ceil(SAMPLE_RATE as usize) * TOKENS_PER_SECOND;
        if est > max_tokens {
            return Err(format!(
                "{:.1} s of audio is ~{est} prompt tokens, more than the {max_tokens}-token context",
                samples as f64 / SAMPLE_RATE as f64
            ));
        }
        let refs: Vec<&[f32]> = clips.iter().map(Vec::as_slice).collect();
        let mels = log_mel_batch(&refs, SAMPLE_RATE).map_err(|e| e.to_string())?;
        let tokens = mels
            .valid_frames
            .iter()
            .map(|&f| audio_token_count(f))
            .collect();
        Ok(StagedAudio { mels, tokens })
    }

    /// Total encoder rows (= `<|audio_pad|>` tokens) of the request.
    pub fn total_tokens(&self) -> usize {
        self.tokens.iter().sum()
    }
}

/// The prompt text that replaces one clip's sentinel: `<|audio_start|>` + `n` x `<|audio_pad|>`
/// + `<|audio_end|>` (the template's block with the processor's pad expansion).
pub fn audio_block(n: usize) -> String {
    format!("{AUDIO_START}{}{AUDIO_END}", AUDIO_PAD.repeat(n))
}

/// Messages with a sentinel at each audio part's position (`audio_at`), and every audio
/// reference of the request in prompt order (still undecoded: decoding may run ffmpeg, so the
/// caller does it off the async runtime with [`decode_all`]).
pub fn stage_messages(messages: &[ChatMessage]) -> (Vec<ChatMessage>, Vec<String>) {
    let mut out = Vec::with_capacity(messages.len());
    let mut refs = Vec::new();
    for m in messages {
        if m.audios.is_empty() {
            out.push(m.clone());
            continue;
        }
        let mut content = String::with_capacity(m.content.len() + 24 * m.audios.len());
        let mut last = 0usize;
        for (i, url) in m.audios.iter().enumerate() {
            // Without an offset the clip goes first: the template's order for [audio, text].
            let at = m.audio_at.get(i).copied().unwrap_or(0).min(m.content.len());
            let at = if m.content.is_char_boundary(at) {
                at
            } else {
                last
            };
            let at = at.max(last);
            content.push_str(&m.content[last..at]);
            content.push_str(AUDIO_SENTINEL);
            last = at;
            refs.push(url.clone());
        }
        content.push_str(&m.content[last..]);
        out.push(ChatMessage {
            content,
            audios: Vec::new(),
            audio_at: Vec::new(),
            ..m.clone()
        });
    }
    (out, refs)
}

/// Decode every staged reference to 16 kHz mono PCM, in order ([`load_audio`]; blocking).
pub fn decode_all(refs: &[String]) -> Result<Vec<Vec<f32>>, String> {
    refs.iter()
        .enumerate()
        .map(|(i, r)| load_audio(r).map_err(|e| format!("audio {i}: {e}")))
        .collect()
}

/// Replace the k-th audio sentinel of the rendered prompt with clip k's block.
pub fn expand_sentinels(prompt: &str, tokens: &[usize]) -> Result<String, String> {
    let parts: Vec<&str> = prompt.split(AUDIO_SENTINEL).collect();
    if parts.len() != tokens.len() + 1 {
        return Err(format!(
            "the chat template kept {} of {} audio parts (a system message cannot hold one)",
            parts.len() - 1,
            tokens.len()
        ));
    }
    let mut s = String::with_capacity(prompt.len() + tokens.iter().sum::<usize>() * 16);
    for (i, part) in parts.iter().enumerate() {
        s.push_str(part);
        if let Some(&n) = tokens.get(i) {
            s.push_str(&audio_block(n));
        }
    }
    Ok(s)
}

/// The start of each clip's `<|audio_pad|>` run in the tokenized prompt, in order. Errors unless
/// the runs match the clips one for one with the expected lengths (a pad split or merged by the
/// tokenizer, or typed by the client, must not shift the splice).
pub fn locate(ids: &[u32], pad_id: u32, tokens: &[usize]) -> Result<Vec<usize>, String> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == pad_id {
            let s = i;
            while i < ids.len() && ids[i] == pad_id {
                i += 1;
            }
            runs.push((s, i - s));
        } else {
            i += 1;
        }
    }
    if runs.len() != tokens.len() {
        return Err(format!(
            "{} <|audio_pad|> runs in the prompt for {} audio parts",
            runs.len(),
            tokens.len()
        ));
    }
    runs.iter()
        .zip(tokens)
        .map(|(&(start, len), &want)| {
            if len != want {
                return Err(format!(
                    "audio pad run of {len} tokens at {start}, expected {want}"
                ));
            }
            Ok(start)
        })
        .collect()
}

/// The request's [`ImagePrompt`] from already-encoded rows (`[sum tokens, hidden]`, clip order)
/// and each clip's run start ([`locate`]): positions are the pad rows in order, no M-RoPE layout,
/// causal attention. Pure: this is what the CPU tests check.
pub fn audio_prompt(
    embeds: Vec<f32>,
    hidden: usize,
    starts: &[usize],
    tokens: &[usize],
) -> Result<ImagePrompt, String> {
    let total: usize = tokens.iter().sum();
    if starts.len() != tokens.len() {
        return Err(format!(
            "{} pad runs for {} clips",
            starts.len(),
            tokens.len()
        ));
    }
    if embeds.len() != total * hidden {
        return Err(format!(
            "audio encoder returned {} values for {total} tokens x {hidden}",
            embeds.len()
        ));
    }
    let positions = starts
        .iter()
        .zip(tokens)
        .flat_map(|(&s, &n)| s..s + n)
        .collect();
    Ok(ImagePrompt {
        embeds,
        hidden,
        positions,
        mrope: None,
        causal: true,
    })
}

/// Encode the request's audio (GPU, or the CPU encoder -- call it off the async runtime) and
/// assemble its [`ImagePrompt`]. Logs one line per request: the proof the encoder ran.
pub fn build_audio_prompt(
    tower: &AudioTower,
    staged: &StagedAudio,
    starts: &[usize],
) -> Result<ImagePrompt, String> {
    let t0 = std::time::Instant::now();
    let clips: Vec<(&LogMel, usize)> = staged
        .mels
        .features
        .iter()
        .zip(&staged.mels.valid_frames)
        .map(|(m, &v)| (m, v))
        .collect();
    let embeds = tower.encode_batch(&clips)?;
    eprintln!(
        "[qwen-audio] encoded {} clip(s), {} frames -> {} tokens in {:.2}s ({})",
        clips.len(),
        staged.mels.valid_frames.iter().sum::<usize>(),
        staged.total_tokens(),
        t0.elapsed().as_secs_f64(),
        tower.backend()
    );
    audio_prompt(embeds, tower.output_dim(), starts, &staged.tokens)
}

/// The `<|audio_pad|>` id, or an error naming the tokenizer's gap.
pub fn pad_id(tok: &Tokenizer) -> Result<u32, String> {
    tok.token_to_id(AUDIO_PAD)
        .ok_or_else(|| "the tokenizer has no <|audio_pad|> token".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn have_ffmpeg() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    /// A 16-bit PCM WAV of interleaved `samples`.
    fn wav_i16(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        b.extend_from_slice(&(rate * 2 * channels as u32).to_le_bytes());
        b.extend_from_slice(&(2 * channels).to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(&data);
        b
    }

    fn stereo_clip(rate: u32, frames: usize) -> Vec<i16> {
        (0..frames)
            .flat_map(|i| {
                let t = i as f64 / rate as f64;
                let l = (0.4 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 32767.0) as i16;
                let r = (0.3 * (2.0 * std::f64::consts::PI * 1300.0 * t).sin() * 32767.0) as i16;
                [l, r]
            })
            .collect()
    }

    #[test]
    fn wav_decodes_without_ffmpeg_and_refusals_are_clear() {
        let wav = wav_i16(16000, 2, &stereo_clip(16000, 1600));
        // No ffmpeg needed for WAV: a bogus tool name is never spawned.
        let pcm = decode_audio_bytes_with(&wav, "arf-no-such-ffmpeg").unwrap();
        assert_eq!(pcm.len(), 1600);
        // A compressed payload names the missing tool.
        let e =
            decode_audio_bytes_with(b"ID3\x04not really an mp3", "arf-no-such-ffmpeg").unwrap_err();
        assert!(
            e.contains("needs `arf-no-such-ffmpeg` on the server's PATH"),
            "{e}"
        );
        assert!(decode_audio_bytes(&[]).is_err());
        for u in ["a.wav", "file://relative.wav"] {
            let e = load_audio(u).unwrap_err();
            assert!(e.contains("absolute path"), "{u}: {e}");
        }
        if std::env::var_os("ARF_MEDIA_URLS").is_none() {
            let e = load_audio("https://example.com/a.mp3").unwrap_err();
            assert!(e.contains("ARF_MEDIA_URLS=1"), "{e}");
        }
        if std::env::var_os("ARF_IMAGE_FILES").is_none() {
            assert!(load_audio("/etc/hosts").unwrap_err().contains("disabled"));
        }
        assert!(load_audio("data:audio/wav,AAAA").is_err());
    }

    /// Through the real ffmpeg (CPU only): a 44.1 kHz stereo WAV and the SAME samples as lossless
    /// FLAC must come out identical, bit for bit -- the FLAC through ffmpeg -> float WAV pipe ->
    /// Arf mixdown/resample, the WAV natively. ffmpeg's s16 -> float conversion is x / 32768,
    /// the same as the WAV decoder's. Skips loudly without ffmpeg.
    #[test]
    fn ffmpeg_flac_route_equals_the_native_wav_route() {
        if !have_ffmpeg() {
            eprintln!("SKIP: ffmpeg not on PATH -- this run verified NOTHING");
            return;
        }
        let wav = wav_i16(44100, 2, &stereo_clip(44100, 22050));
        let dir = std::env::temp_dir().join(format!("arf-audio-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("clip.wav");
        let flac = dir.join("clip.flac");
        std::fs::write(&src, &wav).unwrap();
        let st = Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-i"])
            .arg(&src)
            .args(["-c:a", "flac"])
            .arg(&flac)
            .status()
            .unwrap();
        assert!(st.success(), "ffmpeg could not make the FLAC");
        let flac_bytes = std::fs::read(&flac).unwrap();
        assert!(!is_wav(&flac_bytes));

        let native = decode_audio_bytes(&wav).unwrap();
        let via_ffmpeg = decode_audio_bytes(&flac_bytes).unwrap();
        assert_eq!(
            native.len(),
            8000,
            "0.5 s at 44.1 kHz -> 8000 samples at 16 kHz"
        );
        assert_eq!(via_ffmpeg, native, "FLAC via ffmpeg != WAV natively");

        use base64::Engine;
        let url = format!(
            "data:audio/flac;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&flac_bytes)
        );
        assert_eq!(load_audio(&url).unwrap(), native);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- M3: prompt layout (CPU only; nothing here touches a GPU) ----

    fn omni_gguf() -> Option<PathBuf> {
        let p = std::env::var("ARF_QWEN_OMNI_GGUF")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../models/qwen3-omni/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"
                ))
            });
        if p.is_file() {
            Some(p)
        } else {
            eprintln!(
                "SKIP: {} not present -- this run verified NOTHING about the real tokenizer",
                p.display()
            );
            None
        }
    }

    fn tone(secs: f64) -> Vec<f32> {
        let n = (secs * 16000.0).round() as usize;
        (0..n)
            .map(|i| (0.3 * (2.0 * std::f64::consts::PI * 300.0 * i as f64 / 16000.0).sin()) as f32)
            .collect()
    }

    /// 13 tokens per full second plus the tail's three stride-2 convs (`P:108-116`), checked
    /// by hand: 1.30 s = 130 frames -> 13 + [30 -> 15 -> 8 -> 4] = 17.
    #[test]
    fn token_counts_follow_the_processor_formula() {
        let s = StagedAudio::from_pcm(&[tone(1.30)], 4096).unwrap();
        assert_eq!(s.mels.valid_frames, vec![130]);
        assert_eq!(s.tokens, vec![17]);
        let s = StagedAudio::from_pcm(&[tone(2.0), tone(0.5)], 4096).unwrap();
        // a batch pads to the longest; the shorter clip's mask counts ceil(len / 160) frames
        assert_eq!(s.mels.valid_frames, vec![200, 50]);
        assert_eq!(s.tokens, vec![26, 7]);
        assert_eq!(s.total_tokens(), 33);
        // the size guard refuses before the mel
        let e = StagedAudio::from_pcm(&[tone(10.0)], 100).unwrap_err();
        assert!(e.contains("more than the 100-token context"), "{e}");
    }

    #[test]
    fn sentinels_expand_in_place_and_runs_are_located() {
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "What is said?\nAnd then?".into(),
            audios: vec!["data:audio/wav;base64,AA".into(), "b".into()],
            audio_at: vec![0, 13],
            ..Default::default()
        }];
        let (staged, refs) = stage_messages(&msgs);
        assert_eq!(refs, vec!["data:audio/wav;base64,AA", "b"]);
        assert!(staged[0].audios.is_empty());
        assert_eq!(
            staged[0].content,
            format!("{AUDIO_SENTINEL}What is said?{AUDIO_SENTINEL}\nAnd then?")
        );
        let p = expand_sentinels(&staged[0].content, &[2, 1]).unwrap();
        assert_eq!(
            p,
            format!(
                "{AUDIO_START}{AUDIO_PAD}{AUDIO_PAD}{AUDIO_END}What is said?\
                 {AUDIO_START}{AUDIO_PAD}{AUDIO_END}\nAnd then?"
            )
        );
        assert!(expand_sentinels(&staged[0].content, &[2]).is_err());
        // located on ids: 7 = pad
        let ids = [1, 9, 7, 7, 8, 5, 9, 7, 8];
        assert_eq!(locate(&ids, 7, &[2, 1]).unwrap(), vec![2, 7]);
        assert!(locate(&ids, 7, &[2, 2]).is_err());
        assert!(locate(&ids, 7, &[2]).is_err());
        // rows = the pads, in clip order; causal, no layout
        let ip = audio_prompt(vec![0.0; 3 * 4], 4, &[2, 7], &[2, 1]).unwrap();
        assert_eq!(ip.positions, vec![2, 3, 7]);
        assert!(ip.causal && ip.mrope.is_none());
        assert!(audio_prompt(vec![0.0; 5], 4, &[2, 7], &[2, 1]).is_err());
    }

    /// The whole text half of a spoken-question request against the REAL Omni tokenizer (the GGUF
    /// header): parse `input_audio`, stage, render (ChatML, the path a text + audio request
    /// takes), expand, tokenize, locate. The run sits between `<|audio_start|>` 151669 and
    /// `<|audio_end|>` 151670, is made of `<|audio_pad|>` 151675, and is exactly the processor's
    /// count long; the text after it survives.
    #[test]
    fn a_spoken_question_lays_out_on_the_real_omni_tokenizer() {
        let Some(gguf) = omni_gguf() else { return };
        let tok = Tokenizer::from_gguf_path(&gguf).unwrap();
        assert_eq!(pad_id(&tok).unwrap(), 151_675);
        assert_eq!(tok.token_to_id(AUDIO_START), Some(151_669));
        assert_eq!(tok.token_to_id(AUDIO_END), Some(151_670));

        use base64::Engine;
        let pcm = tone(3.0);
        let wav = {
            let s: Vec<i16> = pcm.iter().map(|x| (x * 32767.0) as i16).collect();
            wav_i16(16000, 1, &s)
        };
        let body = serde_json::json!({"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {
                    "data": base64::engine::general_purpose::STANDARD.encode(&wav),
                    "format": "wav"}},
                {"type": "text", "text": "Answer the question."}
            ]}
        ]});
        let req: crate::http::ChatRequest = serde_json::from_value(body).unwrap();
        assert_eq!(req.messages[1].audios.len(), 1);
        assert_eq!(req.messages[1].audio_at, vec![0]);
        let (msgs, refs) = stage_messages(&req.messages);
        let staged = StagedAudio::from_pcm(&decode_all(&refs).unwrap(), 65_536).unwrap();
        assert_eq!(
            staged.tokens,
            vec![39],
            "3.0 s -> 39 tokens (the plan's own check)"
        );
        let rendered = crate::chat::apply_chat_template_with(&tok, &msgs, None);
        let prompt = expand_sentinels(&rendered, &staged.tokens).unwrap();
        assert!(prompt.starts_with(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\
             <|audio_start|><|audio_pad|>"
        ));
        assert!(prompt.ends_with(
            "<|audio_pad|><|audio_end|>Answer the question.<|im_end|>\n<|im_start|>assistant\n"
        ));
        let ids = crate::chat::encode_prompt(&tok, &prompt).unwrap();
        let starts = locate(&ids, 151_675, &staged.tokens).unwrap();
        let (s, n) = (starts[0], staged.tokens[0]);
        assert_eq!(ids[s - 1], 151_669);
        assert_eq!(ids[s + n], 151_670);
        assert!(ids[s..s + n].iter().all(|&t| t == 151_675));
        assert_eq!(ids.iter().filter(|&&t| t == 151_675).count(), n);
        assert_eq!(
            tok.decode(&ids[s + n + 1..], false).unwrap(),
            "Answer the question.<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// The CPU half of the request path with the REAL tower: speech_q.wav (the M0 spoken
    /// question) -> decode -> mel -> the CPU encoder (never the GPU: this calls
    /// `QwenAudioEncoder` directly, not `AudioTower`) -> the request's rows. Checks shapes and
    /// the row count against the pad count; numeric agreement is the encoder's own gate.
    #[test]
    #[ignore = "loads the 1.3 GB audio tower on the CPU: cargo test --release -p arf-serve qwen_audio -- --ignored"]
    fn speech_clip_encodes_on_the_cpu_into_the_pad_rows() {
        let root = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/qwen3-omni-ref"
        ));
        let tower = std::env::var("ARF_QWEN_OMNI_AUDIO_TOWER")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("audio_tower.safetensors"));
        let clip = tower.with_file_name("speech_q.wav");
        if !tower.is_file() || !clip.is_file() {
            eprintln!(
                "SKIP: {} / {} missing -- verified NOTHING",
                tower.display(),
                clip.display()
            );
            return;
        }
        let enc = arf_core::model::qwen_audio_encoder::QwenAudioEncoder::load(&tower).unwrap();
        let pcm = load_audio_bytes(&std::fs::read(&clip).unwrap());
        let staged = StagedAudio::from_pcm(&[pcm], 65_536).unwrap();
        let n = staged.tokens[0];
        let rows = enc
            .encode(&staged.mels.features[0], staged.mels.valid_frames[0])
            .unwrap();
        assert_eq!(rows.len(), n * 2048);
        assert!(rows.iter().all(|v| v.is_finite()));
        let ip = audio_prompt(rows, 2048, &[17], &staged.tokens).unwrap();
        assert_eq!(ip.positions, (17..17 + n).collect::<Vec<_>>());
        eprintln!(
            "speech_q.wav: {} frames -> {n} rows",
            staged.mels.valid_frames[0]
        );
    }

    fn load_audio_bytes(b: &[u8]) -> Vec<f32> {
        decode_audio_bytes(b).unwrap()
    }
}
