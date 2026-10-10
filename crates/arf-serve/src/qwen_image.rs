//! Qwen3.8 image input for the HTTP layer: load the image, lay out its `<|image_pad|>` run in the
//! rendered prompt, and encode it (GPU, or the CPU reference) into an [`ImagePrompt`] with the
//! M-RoPE layout.
//!
//! Only reachable when `--mmproj` names a `qwen3vl_merger` projector AND a request carries an
//! image; a text request never enters this module.
//!
//! How an image reaches the prompt: the model's own template renders every image content part
//! as `<|vision_start|><|image_pad|><|vision_end|>` IN PLACE (content-part order), and the HF
//! processor then repeats `<|image_pad|>` once per merged token (`grid_h * grid_w`). Arf's
//! chat path flattens text parts into `content`, so the parser records where each image sat
//! (`ChatMessage::image_at`); here a private-use SENTINEL is put at those offsets before the
//! template renders, and each sentinel is replaced by the expanded block after it — so the image
//! lands where the client put it, inside whichever template renders the turn.
//!
//! VIDEO (2026-09-27) takes the same road: a `video_url` / `video` part is one more sentinel, and
//! its block is, per frame pair, `<{t:.1f} seconds><|vision_start|>` + `<|video_pad|>` x (pair's
//! merged tokens) + `<|vision_end|>` — the HF processor's replacement for the template's
//! `<|vision_start|><|video_pad|><|vision_end|>` (`arf_core::model::qwen_video` has the sources).
//! Each pair is one pad run, one M-RoPE grid and one encoder call ([`QwenVision::encode_pair`]).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arf_core::model::mrope::{mrope_layout, ImageGrid};
use arf_core::model::qwen_video::{timestamp_text, QwenVideo};
use arf_core::model::qwen_vision::{preprocess, QwenImage, QwenVisionConfig, QwenVisionEncoder};
use arf_core::scheduler::ImagePrompt;
use arf_core::Tokenizer;

use crate::http::ChatMessage;

/// Stand-in for one image inside message content while the template renders. Private-use code
/// points: no tokenizer merges them with neighbours, and no client text contains them by chance.
pub const IMAGE_SENTINEL: &str = "\u{E000}arf-image\u{E001}";
pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const VIDEO_PAD: &str = "<|video_pad|>";

/// One visual item of a request, in prompt order.
#[derive(Debug, Clone)]
pub enum QwenMedia {
    Image(QwenImage),
    Video(QwenVideo),
}

impl QwenMedia {
    /// The prompt text that replaces this item's sentinel.
    pub fn block(&self) -> String {
        match self {
            QwenMedia::Image(img) => {
                format!(
                    "{VISION_START}{}{VISION_END}",
                    IMAGE_PAD.repeat(img.n_tokens())
                )
            }
            QwenMedia::Video(v) => {
                let pads = VIDEO_PAD.repeat(v.tokens_per_group());
                v.timestamps
                    .iter()
                    .map(|&t| format!("{}{VISION_START}{pads}{VISION_END}", timestamp_text(t)))
                    .collect()
            }
        }
    }

    /// Its pad runs in prompt order: `(is_video, (rows, cols))` — one per image, one per frame pair.
    fn runs(&self) -> Vec<(bool, (usize, usize))> {
        match self {
            QwenMedia::Image(img) => vec![(false, img.token_grid())],
            QwenMedia::Video(v) => vec![(true, v.token_grid()); v.groups()],
        }
    }
}

/// The server's Qwen3.8 vision encoder: the GPU port (`arf_gpu::gpu::metal::qwen_vision`) when it
/// is compiled in and comes up, the CPU encoder otherwise.
///
/// - `ARF_VISION_CPU=1` (read at load) selects the CPU encoder and never builds the GPU one.
/// - With the GPU up, the CPU encoder's 0.93 GB of weights are DROPPED after the upload (the GPU
///   holds its own copy); a GPU failure on a request logs one line and falls back to the CPU,
///   which is reloaded from the mmproj the first time that happens (~1.7 s) and kept after.
pub struct QwenVision {
    pub cfg: QwenVisionConfig,
    path: PathBuf,
    #[cfg(all(feature = "wgpu", target_os = "macos"))]
    gpu: Option<arf_gpu::gpu::metal::qwen_vision::QwenVisionGpu>,
    cpu: Mutex<Option<Arc<QwenVisionEncoder>>>,
}

impl QwenVision {
    /// Load the mmproj's encoder and, unless `ARF_VISION_CPU` is set, bring up the GPU one. A GPU
    /// that does not come up is one log line and the CPU encoder, not an error.
    pub fn load(path: &Path) -> Result<Self, String> {
        let enc = QwenVisionEncoder::load(path).map_err(|e| e.to_string())?;
        let cfg = enc.cfg.clone();
        let cpu_only = std::env::var_os("ARF_VISION_CPU").is_some();
        #[cfg(all(feature = "wgpu", target_os = "macos"))]
        let gpu = if cpu_only {
            None
        } else {
            match arf_gpu::gpu::metal::qwen_vision::QwenVisionGpu::new(&enc) {
                Ok(g) => {
                    eprintln!(
                        "[qwen-vision] GPU encoder on {} ({:.2} GB of weights)",
                        g.device_name(),
                        g.weight_bytes() as f64 / 1e9
                    );
                    Some(g)
                }
                Err(e) => {
                    eprintln!("[qwen-vision] GPU encoder unavailable ({e}); using the CPU encoder");
                    None
                }
            }
        };
        #[cfg(all(feature = "wgpu", target_os = "macos"))]
        let keep_cpu = gpu.is_none();
        #[cfg(not(all(feature = "wgpu", target_os = "macos")))]
        let keep_cpu = {
            let _ = cpu_only;
            true
        };
        Ok(QwenVision {
            cfg,
            path: path.to_path_buf(),
            #[cfg(all(feature = "wgpu", target_os = "macos"))]
            gpu,
            cpu: Mutex::new(keep_cpu.then(|| Arc::new(enc))),
        })
    }

    /// "GPU" or "CPU": which encoder serves images (before any fallback).
    pub fn backend(&self) -> &'static str {
        #[cfg(all(feature = "wgpu", target_os = "macos"))]
        if self.gpu.is_some() {
            return "GPU";
        }
        "CPU"
    }

    /// Encode one preprocessed image: `[tokens, proj_dim]` and which encoder produced it.
    pub fn encode(&self, img: &QwenImage) -> Result<(Vec<f32>, &'static str), String> {
        #[cfg(all(feature = "wgpu", target_os = "macos"))]
        if let Some(g) = &self.gpu {
            match g.encode(img) {
                Ok(v) => return Ok((v, "GPU")),
                Err(e) => eprintln!(
                    "[qwen-vision] GPU encode of {}x{} failed ({e}); falling back to the CPU encoder",
                    img.width, img.height
                ),
            }
        }
        Ok((self.cpu()?.encode(img), "CPU"))
    }

    /// Encode one video frame pair (frames 2k, 2k+1): `[tokens, proj_dim]` and the encoder used.
    /// Same GPU-then-CPU fallback as [`Self::encode`].
    pub fn encode_pair(
        &self,
        a: &QwenImage,
        b: &QwenImage,
    ) -> Result<(Vec<f32>, &'static str), String> {
        #[cfg(all(feature = "wgpu", target_os = "macos"))]
        if let Some(g) = &self.gpu {
            match g.encode_frame_pair(a, b) {
                Ok(v) => return Ok((v, "GPU")),
                Err(e) => eprintln!(
                    "[qwen-vision] GPU encode of a {}x{} frame pair failed ({e}); falling back to \
                     the CPU encoder",
                    a.width, a.height
                ),
            }
        }
        Ok((self.cpu()?.encode_frame_pair(a, b), "CPU"))
    }

    /// The CPU encoder, loaded from the mmproj on first use when the GPU one runs.
    fn cpu(&self) -> Result<Arc<QwenVisionEncoder>, String> {
        let mut g = self
            .cpu
            .lock()
            .map_err(|_| "qwen vision: CPU encoder lock poisoned")?;
        if let Some(c) = g.as_ref() {
            return Ok(c.clone());
        }
        let enc = Arc::new(
            QwenVisionEncoder::load(&self.path)
                .map_err(|e| format!("qwen vision: reloading the CPU encoder: {e}"))?,
        );
        *g = Some(enc.clone());
        Ok(enc)
    }
}

/// Decode an image reference into RGB8: a `data:` URL / bare base64 (as the Gemma path), or —
/// only with `ARF_IMAGE_FILES=1` — a local `file://` URL or absolute path. File reads are OFF by
/// default: they let any HTTP client read files the server can. An `http(s)://` URL is fetched only
/// with `ARF_MEDIA_URLS=1` (`remote_media`, bounded and public addresses only).
pub fn load_image(url: &str) -> Result<(Vec<u8>, usize, usize), String> {
    if crate::remote_media::is_remote(url) {
        let bytes = crate::remote_media::fetch(url, "image")?;
        let img = image::load_from_memory(&bytes).map_err(|e| format!("image decode: {e}"))?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        return Ok((rgb.into_raw(), w, h));
    }
    let path = url
        .strip_prefix("file://")
        .or_else(|| url.starts_with('/').then_some(url));
    if let Some(p) = path {
        if std::env::var_os("ARF_IMAGE_FILES").is_none() {
            return Err(
                "image file paths are disabled (set ARF_IMAGE_FILES=1 on the server to allow)"
                    .into(),
            );
        }
        let bytes = std::fs::read(p).map_err(|e| format!("image file {p}: {e}"))?;
        let img = image::load_from_memory(&bytes).map_err(|e| format!("image decode: {e}"))?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        return Ok((rgb.into_raw(), w, h));
    }
    crate::chat::decode_image_data_url(url)
}

/// Messages with a sentinel at each image's (or video's) position (`image_at`), and every item of
/// the request decoded + preprocessed, in order. Messages without images are cloned unchanged.
pub fn stage_messages(
    enc: &QwenVision,
    messages: &[ChatMessage],
) -> Result<(Vec<ChatMessage>, Vec<QwenMedia>), String> {
    let mut out = Vec::with_capacity(messages.len());
    let mut images = Vec::new();
    for m in messages {
        if m.images.is_empty() {
            out.push(m.clone());
            continue;
        }
        let mut content = String::with_capacity(m.content.len() + 32 * m.images.len());
        let mut last = 0usize;
        for (i, url) in m.images.iter().enumerate() {
            // A message built without offsets (anthropic flattening, internal) puts images first,
            // the HF template's order for an image part followed by text.
            let at = m.image_at.get(i).copied().unwrap_or(0).min(m.content.len());
            let at = if m.content.is_char_boundary(at) {
                at
            } else {
                last
            };
            let at = at.max(last);
            content.push_str(&m.content[last..at]);
            content.push_str(IMAGE_SENTINEL);
            last = at;
            if let Some(v) = m.video_at(i) {
                let video = crate::qwen_video::load_video(&enc.cfg, v)
                    .map_err(|e| format!("video: {e}"))?;
                images.push(QwenMedia::Video(video));
            } else {
                let (rgb, w, h) = load_image(url)?;
                images.push(QwenMedia::Image(preprocess(&enc.cfg, &rgb, w, h)));
            }
        }
        content.push_str(&m.content[last..]);
        out.push(ChatMessage {
            content,
            images: Vec::new(),
            image_at: Vec::new(),
            videos: Vec::new(),
            ..m.clone()
        });
    }
    Ok((out, images))
}

/// Replace the k-th sentinel of the rendered prompt with item k's block ([`QwenMedia::block`]):
/// `<|vision_start|>` + `grid_h*grid_w` x `<|image_pad|>` + `<|vision_end|>` for an image.
pub fn expand_sentinels(prompt: &str, images: &[QwenMedia]) -> Result<String, String> {
    let parts: Vec<&str> = prompt.split(IMAGE_SENTINEL).collect();
    if parts.len() != images.len() + 1 {
        return Err(format!(
            "the chat template kept {} of {} images (a system message cannot hold one)",
            parts.len() - 1,
            images.len()
        ));
    }
    let mut s = String::with_capacity(prompt.len());
    for (i, part) in parts.iter().enumerate() {
        s.push_str(part);
        if let Some(item) = images.get(i) {
            s.push_str(&item.block());
        }
    }
    Ok(s)
}

/// Find each image's `<|image_pad|>` run — and each video frame pair's `<|video_pad|>` run — in the
/// tokenized prompt and return their grids, in prompt order (one per image, one per frame pair:
/// the M-RoPE layout treats a pair exactly as an image). Errors if the runs do not match the
/// items one for one (a pad token split or merged by the tokenizer, or typed by the client).
/// `video_pad_id` may be `None` when the request has no video.
pub fn locate(
    ids: &[u32],
    pad_id: u32,
    video_pad_id: Option<u32>,
    images: &[QwenMedia],
) -> Result<Vec<ImageGrid>, String> {
    let want: Vec<(bool, (usize, usize))> = images.iter().flat_map(QwenMedia::runs).collect();
    let has_video = want.iter().any(|w| w.0);
    if has_video && video_pad_id.is_none() {
        return Err("the tokenizer has no <|video_pad|> token".into());
    }
    // Without a video, `<|video_pad|>` is ordinary text — exactly as before video existed.
    let video_pad_id = video_pad_id.filter(|_| has_video);
    let is_pad = |t: u32| t == pad_id || Some(t) == video_pad_id;
    let mut runs: Vec<(bool, usize, usize)> = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if is_pad(ids[i]) {
            let (s, id) = (i, ids[i]);
            while i < ids.len() && ids[i] == id {
                i += 1;
            }
            runs.push((id != pad_id, s, i - s));
        } else {
            i += 1;
        }
    }
    if runs.len() != want.len() {
        return Err(format!(
            "{} vision pad runs in the prompt for {} images / video frame pairs",
            runs.len(),
            want.len()
        ));
    }
    runs.iter()
        .zip(&want)
        .map(|(&(video, start, len), &(want_video, (gh, gw)))| {
            let kind = if want_video { "video" } else { "image" };
            if video != want_video {
                return Err(format!("a {kind} pad run was expected at token {start}"));
            }
            if len != gh * gw {
                return Err(format!(
                    "{kind} pad run of {len} tokens, expected {}",
                    gh * gw
                ));
            }
            Ok(ImageGrid {
                start,
                grid_h: gh,
                grid_w: gw,
            })
        })
        .collect()
}

/// Encode every image and video frame pair (GPU, or seconds per image on the CPU — call it off
/// the async runtime) and assemble the request's [`ImagePrompt`]: embeddings in pad order, their
/// prompt positions, and the M-RoPE layout of the whole prompt. `grids` is [`locate`]'s: one per
/// image and one per frame pair.
pub fn build_image_prompt(
    enc: &QwenVision,
    images: &[QwenMedia],
    grids: &[ImageGrid],
    prompt_len: usize,
) -> Result<ImagePrompt, String> {
    let hidden = enc.cfg.proj_dim;
    let total: usize = grids.iter().map(|g| g.tokens()).sum();
    let mut embeds = Vec::with_capacity(total * hidden);
    let mut positions = Vec::with_capacity(total);
    // One encoder call per grid: an image, or one frame pair of a video.
    let mut calls: Vec<(&QwenImage, Option<&QwenImage>)> = Vec::with_capacity(grids.len());
    for item in images {
        match item {
            QwenMedia::Image(img) => calls.push((img, None)),
            QwenMedia::Video(v) => {
                calls.extend((0..v.groups()).map(|k| {
                    let (a, b) = v.pair(k);
                    (a, Some(b))
                }));
            }
        }
    }
    if calls.len() != grids.len() {
        return Err(format!(
            "qwen vision: {} encoder calls for {} pad runs",
            calls.len(),
            grids.len()
        ));
    }
    for (&(img, second), g) in calls.iter().zip(grids) {
        let t0 = std::time::Instant::now();
        let (e, backend) = match second {
            None => enc.encode(img)?,
            Some(b) => enc.encode_pair(img, b)?,
        };
        if std::env::var_os("ARF_VISION_TIMING").is_some() {
            eprintln!(
                "[qwen-vision] encoded {}x{} {}({} tokens) in {:.2}s ({backend})",
                img.width,
                img.height,
                if second.is_some() { "frame pair " } else { "" },
                g.tokens(),
                t0.elapsed().as_secs_f64()
            );
        }
        if e.len() != g.tokens() * hidden {
            return Err(format!(
                "qwen vision: {} values for {} tokens x {hidden}",
                e.len(),
                g.tokens()
            ));
        }
        embeds.extend_from_slice(&e);
        positions.extend(g.start..g.start + g.tokens());
    }
    Ok(ImagePrompt {
        embeds,
        hidden,
        positions,
        mrope: Some(mrope_layout(prompt_len, grids)),
        causal: true,
    })
}

/// Image requests are served GREEDY: the Metal path that splices images and rotates M-RoPE is
/// the greedy island record (`forward_mrope_batch`); the logits paths (sampling, logprobs,
/// penalties) do not carry either yet. Said once in the log, not hidden.
pub fn force_greedy(params: &mut arf_core::SamplingParams) {
    static SAID: std::sync::Once = std::sync::Once::new();
    if params.temperature > 0.0 || (params.repetition_penalty - 1.0).abs() > f32::EPSILON {
        SAID.call_once(|| {
            eprintln!(
                "[qwen-vision] image requests are served greedy (temperature, repetition \
                 penalty and logprobs are ignored for them; logged once)"
            )
        });
    }
    params.temperature = 0.0;
    params.repetition_penalty = 1.0;
}

/// The `<|image_pad|>` id, or an error naming the tokenizer's gap.
pub fn pad_id(tok: &Tokenizer) -> Result<u32, String> {
    tok.token_to_id(IMAGE_PAD)
        .ok_or_else(|| "the tokenizer has no <|image_pad|> token".to_string())
}

/// The `<|video_pad|>` id, when the tokenizer has one ([`locate`] errors only if a video needs it).
pub fn video_pad_id(tok: &Tokenizer) -> Option<u32> {
    tok.token_to_id(VIDEO_PAD)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(gw: usize, gh: usize) -> QwenImage {
        QwenImage {
            pixels: Vec::new(),
            width: gw * 16,
            height: gh * 16,
            grid_w: gw,
            grid_h: gh,
        }
    }

    fn media(v: &[QwenImage]) -> Vec<QwenMedia> {
        v.iter().cloned().map(QwenMedia::Image).collect()
    }

    fn video(gw: usize, gh: usize, timestamps: &[f64]) -> QwenMedia {
        let frames = vec![img(gw, gh); 2 * timestamps.len()];
        QwenMedia::Video(QwenVideo::new(frames, timestamps.to_vec()).unwrap())
    }

    #[test]
    fn sentinels_expand_in_order() {
        let p = format!("<|im_start|>user\n{IMAGE_SENTINEL}What?{IMAGE_SENTINEL}<|im_end|>");
        let s = expand_sentinels(&p, &media(&[img(4, 2), img(2, 2)])).unwrap();
        let want = format!(
            "<|im_start|>user\n{VISION_START}{}{VISION_END}What?{VISION_START}{}{VISION_END}<|im_end|>",
            IMAGE_PAD.repeat(2),
            IMAGE_PAD
        );
        assert_eq!(s, want);
        assert!(expand_sentinels(&p, &media(&[img(2, 2)])).is_err());
    }

    /// A video's block: per frame pair, the timestamp text then a bracketed `<|video_pad|>` run
    /// (HF 4.57.1 `Qwen3VLProcessor.__call__`, which replaces the template's whole
    /// `<|vision_start|><|video_pad|><|vision_end|>`), here next to an image.
    #[test]
    fn a_video_expands_to_timestamped_frame_pairs() {
        let p = format!("<|im_start|>user\n{IMAGE_SENTINEL}{IMAGE_SENTINEL}Describe.<|im_end|>");
        let items = vec![QwenMedia::Image(img(2, 2)), video(4, 2, &[0.25, 1.25])];
        let s = expand_sentinels(&p, &items).unwrap();
        let pair = format!("{VISION_START}{}{VISION_END}", VIDEO_PAD.repeat(2));
        let want = format!(
            "<|im_start|>user\n{VISION_START}{IMAGE_PAD}{VISION_END}\
             <0.2 seconds>{pair}<1.2 seconds>{pair}Describe.<|im_end|>"
        );
        assert_eq!(s, want);
    }

    #[test]
    fn locate_finds_runs_and_checks_counts() {
        let pad = 7;
        let ids = [1, 2, 7, 7, 7, 7, 3, 7, 4];
        let g = locate(&ids, pad, None, &media(&[img(4, 4), img(2, 2)])).unwrap();
        assert_eq!(
            g,
            vec![
                ImageGrid {
                    start: 2,
                    grid_h: 2,
                    grid_w: 2
                },
                ImageGrid {
                    start: 7,
                    grid_h: 1,
                    grid_w: 1
                }
            ]
        );
        assert!(locate(&ids, pad, None, &media(&[img(4, 4)])).is_err());
        assert!(locate(&ids, pad, None, &media(&[img(2, 2), img(2, 2)])).is_err());
    }

    /// Image + a two-pair video: three runs, three grids, the video's two tagged `<|video_pad|>`.
    #[test]
    fn locate_splits_a_video_into_one_grid_per_frame_pair() {
        let (ip, vp) = (7, 8);
        //         0  1  2  3  4  5  6  7  8  9 10 11 12
        let ids = [1, 7, 7, 7, 7, 5, 8, 8, 6, 5, 8, 8, 6];
        let items = vec![QwenMedia::Image(img(4, 4)), video(4, 2, &[0.0, 1.0])];
        let g = locate(&ids, ip, Some(vp), &items).unwrap();
        let grid = |start, grid_h, grid_w| ImageGrid {
            start,
            grid_h,
            grid_w,
        };
        assert_eq!(g, vec![grid(1, 2, 2), grid(6, 1, 2), grid(10, 1, 2)]);
        // no video pad id in the tokenizer
        assert!(locate(&ids, ip, None, &items).is_err());
        // an image-only request ignores video pad ids (a literal one is just text), as before
        let img_only = [1, 7, 7, 7, 7, 8, 8, 6];
        let g = locate(&img_only, ip, Some(vp), &items[..1]).unwrap();
        assert_eq!(g, vec![grid(1, 2, 2)]);
        // a video run where an image was expected
        let swapped = vec![video(4, 2, &[0.0, 1.0]), QwenMedia::Image(img(4, 4))];
        assert!(locate(&ids, ip, Some(vp), &swapped).is_err());
        // a video with one pair too many
        assert!(locate(
            &ids,
            ip,
            Some(vp),
            &[items[0].clone(), video(4, 2, &[0.0, 1.0, 2.0])]
        )
        .is_err());
    }

    /// The real Qwen3.8 tokenizer keeps the vision brackets and every pad atomic, so a rendered
    /// image block tokenizes to `<|vision_start|>`, exactly `grid_h*grid_w` pad ids, `<|vision_end|>`
    /// — the property `locate` depends on. Needs the GGUF (`ARF_QWEN_GGUF`, reads only its
    /// tokenizer metadata); skips loudly without it.
    #[test]
    fn real_tokenizer_keeps_the_image_block_atomic() {
        let Some(path) = std::env::var_os("ARF_QWEN_GGUF") else {
            eprintln!("SKIP: set ARF_QWEN_GGUF to a Qwen3.8 GGUF -- this run verified NOTHING");
            return;
        };
        let tok = Tokenizer::from_gguf_path(std::path::Path::new(&path)).expect("tokenizer");
        let msgs = [ChatMessage {
            role: "user".into(),
            content: "What is in this picture?".into(),
            images: vec!["unused".into()],
            image_at: vec![0],
            ..Default::default()
        }];
        // stage by hand (no encoder needed): sentinel at offset 0
        let staged = vec![ChatMessage {
            content: format!("{IMAGE_SENTINEL}{}", msgs[0].content),
            images: Vec::new(),
            image_at: Vec::new(),
            ..msgs[0].clone()
        }];
        let prompt = crate::chat::apply_chat_template_with(&tok, &staged, None);
        let im = img(6, 4); // 2 x 3 merged tokens
        let prompt = expand_sentinels(&prompt, &media(std::slice::from_ref(&im))).unwrap();
        let ids = crate::chat::encode_prompt(&tok, &prompt).unwrap();
        let pad = pad_id(&tok).unwrap();
        assert_eq!(pad, 248056);
        let g = locate(&ids, pad, None, &media(std::slice::from_ref(&im))).unwrap();
        assert_eq!((g[0].grid_h, g[0].grid_w), (2, 3));
        assert_eq!(ids[g[0].start - 1], tok.token_to_id(VISION_START).unwrap());
        assert_eq!(ids[g[0].start + 6], tok.token_to_id(VISION_END).unwrap());
        eprintln!("prompt head: {:?}", &prompt[..prompt.len().min(200)]);
    }

    /// The same for a VIDEO block with the real tokenizer: `<|video_pad|>` is one token, every
    /// frame pair tokenizes to timestamp text, `<|vision_start|>`, exactly rows*cols pads,
    /// `<|vision_end|>`. Needs `ARF_QWEN_GGUF` (tokenizer metadata only); skips loudly without it.
    #[test]
    fn real_tokenizer_keeps_the_video_block_atomic() {
        let Some(path) = std::env::var_os("ARF_QWEN_GGUF") else {
            eprintln!("SKIP: set ARF_QWEN_GGUF to a Qwen3.8 GGUF -- this run verified NOTHING");
            return;
        };
        let tok = Tokenizer::from_gguf_path(std::path::Path::new(&path)).expect("tokenizer");
        let staged = vec![ChatMessage {
            role: "user".into(),
            content: format!("{IMAGE_SENTINEL}What happens in this video?"),
            ..Default::default()
        }];
        let prompt = crate::chat::apply_chat_template_with(&tok, &staged, None);
        let items = vec![video(6, 4, &[0.25, 1.25, 2.25])]; // 3 pairs of 2 x 3 tokens
        let prompt = expand_sentinels(&prompt, &items).unwrap();
        let ids = crate::chat::encode_prompt(&tok, &prompt).unwrap();
        let vp = video_pad_id(&tok).expect("<|video_pad|> in the vocabulary");
        let g = locate(&ids, pad_id(&tok).unwrap(), Some(vp), &items).unwrap();
        assert_eq!(g.len(), 3);
        let (vs, ve) = (
            tok.token_to_id(VISION_START).unwrap(),
            tok.token_to_id(VISION_END).unwrap(),
        );
        for gr in &g {
            assert_eq!((gr.grid_h, gr.grid_w), (2, 3));
            assert_eq!(ids[gr.start - 1], vs);
            assert_eq!(ids[gr.start + 6], ve);
        }
        // the text between two pairs is exactly the timestamp's tokens
        let between = &ids[g[0].start + 7..g[1].start - 1];
        assert_eq!(tok.decode(between, false).unwrap(), "<1.2 seconds>");
        eprintln!(
            "video_pad {vp}; timestamp '<1.2 seconds>' = {} tokens; prompt head: {:?}",
            between.len(),
            &prompt[..prompt.len().min(160)]
        );
    }

    #[test]
    fn file_paths_are_off_by_default() {
        if std::env::var_os("ARF_IMAGE_FILES").is_none() {
            assert!(load_image("/etc/hosts").unwrap_err().contains("disabled"));
        }
    }

    /// The whole video request path short of the language model, on the CPU encoder with the real
    /// mmproj and tokenizer: a `video_url` data URL (a generated 2 s, 10 fps, 320x240 clip) goes
    /// through the parser, ffmpeg, the sampler (4 frames = 2 pairs at 320x256 = 8x10 tokens), the
    /// template, the tokenizer, `locate` and the encoder into an `ImagePrompt`. Refuses to run
    /// without `ARF_VISION_CPU=1` so it can never bring up the GPU encoder. Run with
    /// `ARF_VISION_CPU=1 ARF_QWEN_MMPROJ=.. ARF_QWEN_GGUF=.. cargo test --release -p arf-serve
    /// real_video_request -- --ignored --nocapture`.
    #[test]
    #[ignore = "real mmproj + tokenizer, CPU encoder; see the doc comment"]
    fn real_video_request_builds_its_image_prompt_on_the_cpu() {
        let (Some(mm), Some(gguf)) = (
            std::env::var_os("ARF_QWEN_MMPROJ"),
            std::env::var_os("ARF_QWEN_GGUF"),
        ) else {
            eprintln!("SKIP: set ARF_QWEN_MMPROJ and ARF_QWEN_GGUF -- this run verified NOTHING");
            return;
        };
        if std::env::var_os("ARF_VISION_CPU").is_none() {
            eprintln!(
                "SKIP: needs ARF_VISION_CPU=1 (CPU encoder only) -- this run verified NOTHING"
            );
            return;
        }
        let dir = std::env::temp_dir().join(format!("arf-video-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("clip.mkv");
        let ok = std::process::Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-f", "lavfi", "-i"])
            .arg("testsrc=size=320x240:rate=10:duration=2")
            .args(["-c:v", "ffv1"])
            .arg(&clip)
            .status()
            .is_ok_and(|s| s.success());
        if !ok {
            eprintln!("SKIP: ffmpeg could not make the clip -- this run verified NOTHING");
            return;
        }
        use base64::Engine;
        let url = format!(
            "data:video/x-matroska;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(std::fs::read(&clip).unwrap())
        );
        let msg: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": "What changes over time?"},
                        {"type": "video_url", "video_url": {"url": url}}]
        }))
        .unwrap();
        let tok = Tokenizer::from_gguf_path(std::path::Path::new(&gguf)).expect("tokenizer");
        let enc = QwenVision::load(std::path::Path::new(&mm)).expect("mmproj");
        assert_eq!(enc.backend(), "CPU");
        let t0 = std::time::Instant::now();
        let (msgs, media) = stage_messages(&enc, std::slice::from_ref(&msg)).unwrap();
        let prompt = crate::chat::apply_chat_template_with(&tok, &msgs, None);
        let prompt = expand_sentinels(&prompt, &media).unwrap();
        let ids = crate::chat::encode_prompt(&tok, &prompt).unwrap();
        let grids = locate(&ids, pad_id(&tok).unwrap(), video_pad_id(&tok), &media).unwrap();
        assert_eq!(grids.len(), 2, "4 sampled frames = 2 pairs");
        for g in &grids {
            assert_eq!((g.grid_h, g.grid_w), (8, 10), "320x240 -> 320x256");
        }
        let ip = build_image_prompt(&enc, &media, &grids, ids.len()).unwrap();
        eprintln!(
            "video prompt: {} tokens, {} video tokens, encode {:.2}s (CPU); head {:?}",
            ids.len(),
            ip.positions.len(),
            t0.elapsed().as_secs_f64(),
            &prompt[..prompt.len().min(90)]
        );
        assert!(prompt.contains("What changes over time?"));
        assert!(prompt.contains("<0.3 seconds><|vision_start|>")); // (0 + 6/10) / 2
        assert!(prompt.contains("<1.6 seconds><|vision_start|>")); // (13/10 + 19/10) / 2
        assert_eq!(ip.hidden, 5120);
        assert_eq!(ip.positions.len(), 160);
        assert_eq!(ip.embeds.len(), 160 * 5120);
        assert!(ip.embeds.iter().all(|v| v.is_finite()));
        let want: Vec<usize> = grids.iter().flat_map(|g| g.start..g.start + 80).collect();
        assert_eq!(ip.positions, want);
        let m = ip.mrope.as_ref().expect("M-RoPE layout");
        assert_eq!(
            m.delta as usize,
            160 - 2 * 10,
            "each pair advances max(8, 10)"
        );
        // The clip moves, so its two pairs do not encode alike.
        let (p0, p1) = ip.embeds.split_at(80 * 5120);
        assert!(p0.iter().zip(p1).any(|(a, b)| (a - b).abs() > 1e-3));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
