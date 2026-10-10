//! FLUX image-generation actor + HTTP handler, kept self-contained (mirrors the LLM actor
//! pattern in `actor.rs`): the `FluxPipeline` is `!Sync` and ~30 GB resident, so it lives on a
//! dedicated thread reached over an mpsc channel. Loaded LAZILY on the first request (so a
//! text-only server pays nothing). Exposes:
//!   POST /v1/images/generations  — prompt → PNG (OpenAI-images-shaped: { data: [{ b64_json }] })
//!   GET  /flux/trace             — the last generation's per-step denoise trace (for the HUD)
//!
//! All compute is GPU-side inside FluxPipeline; this module only tokenizes (CLIP BPE + T5 Unigram)
//! and PNG-encodes the readback.

use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};

use arf_core::Tokenizer;
use base64::Engine;
use serde::{Deserialize, Serialize};

/// One image request handed to the FLUX actor thread; the reply carries the PNG + trace.
struct FluxJob {
    prompt: String,
    steps: usize,
    size: usize,
    seed: u64,
    reply: std::sync::mpsc::Sender<FluxOutput>,
}

#[derive(Clone)]
struct FluxOutput {
    png: Vec<u8>,
    trace: Vec<StepTraceJson>,
    load_ms: f32,
    gen_ms: f32,
}

/// Serializable per-step denoise trace for `/flux/trace` (the timeline scrubber).
#[derive(Clone, Serialize)]
pub struct StepTraceJson {
    pub step: usize,
    pub sigma: f32,
    pub latent_l2: f32,
    pub dit_ms: f32,
}

/// The handle the HTTP handlers use: a job channel + the cached last trace.
pub struct FluxHandle {
    tx: Sender<FluxJob>,
    last_trace: Arc<Mutex<Vec<StepTraceJson>>>,
}

static FLUX: OnceLock<FluxHandle> = OnceLock::new();

/// Get-or-spawn the FLUX actor. The pipeline loads on the actor thread on the FIRST job (so spawn
/// is cheap; the heavy load is deferred to first use). `models_dir` is the flux-schnell dir.
fn handle(models_dir: std::path::PathBuf) -> &'static FluxHandle {
    FLUX.get_or_init(|| {
        let (tx, rx): (Sender<FluxJob>, Receiver<FluxJob>) = std::sync::mpsc::channel();
        let last_trace = Arc::new(Mutex::new(Vec::new()));
        let lt = last_trace.clone();
        std::thread::Builder::new()
            .name("flux-actor".into())
            .spawn(move || flux_actor_loop(models_dir, rx, lt))
            .expect("spawn flux actor");
        FluxHandle { tx, last_trace }
    })
}

/// The actor thread: load the pipeline once (on first job), then serve jobs serially.
fn flux_actor_loop(
    models_dir: std::path::PathBuf,
    rx: Receiver<FluxJob>,
    last_trace: Arc<Mutex<Vec<StepTraceJson>>>,
) {
    use arf_gpu::gpu::flux::{FluxPipeline, GenConfig};
    use arf_gpu::gpu::GpuContext;

    let clip_tok = Tokenizer::from_file(models_dir.join("tokenizers/clip_tokenizer.json")).ok();
    let t5_tok = Tokenizer::from_file(models_dir.join("tokenizers/t5_tokenizer.json")).ok();
    let ctx = Arc::new(GpuContext::new().expect("flux gpu ctx"));
    let mut pipe: Option<(FluxPipeline, usize)> = None; // (pipeline, size it was built for)

    while let Ok(job) = rx.recv() {
        let t_load = std::time::Instant::now();
        // (re)build the pipeline if the requested size changed (the DiT grid is size-specific).
        let rebuild = pipe.as_ref().map(|(_, s)| *s != job.size).unwrap_or(true);
        if rebuild {
            let gen = GenConfig {
                height: job.size,
                width: job.size,
                steps: job.steps,
                seed: job.seed,
                t5_len: 256,
            };
            match FluxPipeline::load(&ctx, &models_dir, &gen) {
                Ok(p) => pipe = Some((p, job.size)),
                Err(e) => {
                    eprintln!("flux load failed: {e}");
                    let _ = job.reply.send(FluxOutput {
                        png: Vec::new(),
                        trace: Vec::new(),
                        load_ms: 0.0,
                        gen_ms: 0.0,
                    });
                    continue;
                }
            }
        }
        let load_ms = t_load.elapsed().as_secs_f32() * 1000.0;
        let (pipeline, _) = pipe.as_ref().unwrap();

        let (ct, eos) = clip_tokens(clip_tok.as_ref(), &job.prompt);
        let tt = t5_tokens(t5_tok.as_ref(), &job.prompt, 256);
        let gen = GenConfig {
            height: job.size,
            width: job.size,
            steps: job.steps,
            seed: job.seed,
            t5_len: 256,
        };
        let t_gen = std::time::Instant::now();
        let res = pipeline.generate(&ct, eos, &tt, &gen);
        let gen_ms = t_gen.elapsed().as_secs_f32() * 1000.0;

        let trace: Vec<StepTraceJson> = res
            .trace
            .iter()
            .map(|s| StepTraceJson {
                step: s.step,
                sigma: s.sigma,
                latent_l2: s.latent_l2,
                dit_ms: s.dit_ms,
            })
            .collect();
        *last_trace.lock().unwrap() = trace.clone();
        let png = encode_png(&res.rgb, res.height, res.width);
        let _ = job.reply.send(FluxOutput {
            png,
            trace,
            load_ms,
            gen_ms,
        });
    }
}

/// CLIP-L tokens: BOS(49406) + text + EOS(49407), padded to 77. Falls back to a tiny fixed
/// sequence if the tokenizer is missing (keeps the endpoint responsive for smoke tests).
fn clip_tokens(tok: Option<&Tokenizer>, prompt: &str) -> (Vec<u32>, usize) {
    let mut ids = vec![49406u32];
    if let Some(t) = tok {
        if let Ok(enc) = t.encode(prompt, false) {
            ids.extend(enc.iter().take(75));
        }
    }
    ids.push(49407);
    let eos = ids.len() - 1;
    ids.resize(77, 49407);
    (ids, eos)
}

fn t5_tokens(tok: Option<&Tokenizer>, prompt: &str, len: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    if let Some(t) = tok {
        if let Ok(enc) = t.encode(prompt, false) {
            ids.extend(enc.iter().take(len - 1));
        }
    }
    ids.push(1);
    ids.resize(len, 0);
    ids
}

/// RGB `[3,H,W]` (~[-1,1], channel-major) → PNG bytes.
fn encode_png(rgb: &[f32], h: usize, w: usize) -> Vec<u8> {
    let plane = h * w;
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let px = |c: usize| {
                (((rgb[c * plane + y * w + x] * 0.5 + 0.5).clamp(0.0, 1.0)) * 255.0).round() as u8
            };
            img.put_pixel(x as u32, y as u32, image::Rgb([px(0), px(1), px(2)]));
        }
    }
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .expect("png encode");
    buf.into_inner()
}

// ---- HTTP layer ----

#[derive(Deserialize)]
pub struct ImageRequest {
    pub prompt: String,
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// The image's side in pixels: a number (`512`), or the OpenAI form `"512x512"` (square only).
    #[serde(default = "default_size", deserialize_with = "size_px")]
    pub size: usize,
    #[serde(default)]
    pub seed: u64,
}
/// `size` as a number of pixels, from `512`, `"512"` or `"512x512"`. The OpenAI images API sends
/// the last; it was refused with 422 (2026-10-07). A non-square size is an error that says so.
fn size_px<'de, D: serde::Deserializer<'de>>(d: D) -> Result<usize, D::Error> {
    use serde::de::Error;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Size {
        Px(usize),
        Text(String),
    }
    match Size::deserialize(d)? {
        Size::Px(n) => Ok(n),
        Size::Text(s) => {
            let (w, h) = s.split_once(['x', 'X']).unwrap_or((&s, &s));
            let parse = |v: &str| v.trim().parse::<usize>().ok();
            match (parse(w), parse(h)) {
                (Some(w), Some(h)) if w == h => Ok(w),
                (Some(_), Some(_)) => Err(D::Error::custom(format!(
                    "size {s:?}: only square images are generated (e.g. \"512x512\")"
                ))),
                _ => Err(D::Error::custom(format!(
                    "size {s:?}: expected a number of pixels or \"<n>x<n>\""
                ))),
            }
        }
    }
}

fn default_steps() -> usize {
    4
}
fn default_size() -> usize {
    512
}

#[derive(Serialize)]
pub struct ImageData {
    pub b64_json: String,
}
#[derive(Serialize)]
pub struct ImageResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
    pub trace: Vec<StepTraceJson>,
    pub load_ms: f32,
    pub gen_ms: f32,
}

/// `POST /v1/images/generations` — prompt → PNG (base64). Blocks until the image is generated
/// (FLUX is slow; the UI shows the trace as it polls /flux/trace meanwhile).
pub async fn images_generations(
    prompt: String,
    steps: usize,
    size: usize,
    seed: u64,
    models_dir: std::path::PathBuf,
) -> Result<ImageResponse, String> {
    let h = handle(models_dir);
    let (reply_tx, reply_rx) = std::sync::mpsc::channel();
    h.tx.send(FluxJob {
        prompt,
        steps,
        size,
        seed,
        reply: reply_tx,
    })
    .map_err(|_| "flux actor unavailable".to_string())?;
    // run the blocking recv on a tokio blocking thread so the reactor isn't stalled.
    let out = tokio::task::spawn_blocking(move || reply_rx.recv())
        .await
        .map_err(|e| e.to_string())?
        .map_err(|_| "flux generation failed".to_string())?;
    if out.png.is_empty() {
        return Err("flux pipeline load/generation failed (see server log)".into());
    }
    let b64 = base64::engine::general_purpose::STANDARD.encode(&out.png);
    Ok(ImageResponse {
        created: 0,
        data: vec![ImageData { b64_json: b64 }],
        trace: out.trace,
        load_ms: out.load_ms,
        gen_ms: out.gen_ms,
    })
}

/// `GET /flux/trace` — the last generation's per-step denoise trace (for the timeline scrubber).
pub fn last_trace(models_dir: std::path::PathBuf) -> Vec<StepTraceJson> {
    handle(models_dir).last_trace.lock().unwrap().clone()
}

#[cfg(test)]
mod request_tests {
    use super::ImageRequest;

    /// `size` is a number of pixels, or the OpenAI images form; a non-square size is refused.
    #[test]
    fn size_takes_a_number_or_the_openai_form() {
        let size = |v: serde_json::Value| {
            serde_json::from_value::<ImageRequest>(serde_json::json!({"prompt": "x", "size": v}))
                .map(|r| r.size)
        };
        assert_eq!(size(serde_json::json!(512)).unwrap(), 512);
        assert_eq!(size(serde_json::json!("512x512")).unwrap(), 512);
        assert_eq!(size(serde_json::json!("768")).unwrap(), 768);
        let e = size(serde_json::json!("1024x512")).unwrap_err().to_string();
        assert!(e.contains("only square"), "{e}");
        assert!(size(serde_json::json!("big")).is_err());
        let d: ImageRequest = serde_json::from_value(serde_json::json!({"prompt": "x"})).unwrap();
        assert_eq!((d.size, d.steps), (512, 4));
    }
}
