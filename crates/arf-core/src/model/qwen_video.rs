//! Qwen3.8 VIDEO input (2026-09-27): which frames to take, what size to resize them to, and how the
//! prompt names each frame pair. Pure functions only — decoding a file is the server's job
//! (`arf_serve::qwen_video`, which shells out to `ffmpeg`), and encoding is the image encoder's
//! (`QwenVisionEncoder::encode_frame_pair`).
//!
//! Every rule below was read in the reference source before it was written, and the unit tests
//! pin values printed by a verbatim numpy transcription of those functions (not by this file):
//! - transformers 5.17.0 `models/qwen3_vl/video_processing_qwen3_vl.py`: `sample_frames` (fps 2,
//!   min 4, max 768 frames, `np.linspace(..).round()`), `smart_resize` (the TOTAL pixel budget
//!   over all frames, Python `round` = ties-to-even), `resize` with `cap_pixels_per_frame=True`
//!   (the qwen-vl-utils behaviour, 768 tokens a frame at most, which HF says becomes the default in
//!   5.22), and `patchify` (an odd frame count repeats the LAST frame).
//! - transformers `processing_qwen3_vl.py`: `_calculate_timestamps` (indices / fps, padded to a
//!   pair by repeating the last index, averaged per pair) and the placeholder
//!   `<{t:.1f} seconds><|vision_start|><|video_pad|>*N<|vision_end|>` per frame pair. In 4.57.1 it
//!   replaces the template's whole `<|vision_start|><|video_pad|><|vision_end|>`; 5.17's generic
//!   `get_text_with_replacements` substitutes only the pad token, which with this template would
//!   nest one `<|vision_start|>` inside another. Arf follows 4.57.1's explicit rule.
//! - the model's own `video_preprocessor_config.json` (mlx-community/Qwen3.8-27B-4bit snapshot in
//!   the local HF cache): `size = {shortest_edge 4096, longest_edge 25165824}`, patch 16,
//!   temporal patch 2, merge 2, mean = std = 0.5. fps / min / max frames are not in it, so they are
//!   the class defaults.
//! - qwen-vl-utils 0.0.14 `fetch_video` for a LIST of frames: padded to an even count by repeating
//!   the last frame, indices `0..n`, `sample_fps` 2.0 unless given.
//!
//! What a frame pair IS to the model (transformers `modeling_qwen3_vl.py`, `vision_utils.py`):
//! - the Conv3d patch embed sees frames `(2k, 2k+1)` through its two temporal taps — llama.cpp's
//!   converter splits it into `v.patch_embd.weight` (tap 0) and `v.patch_embd.weight.1` (tap 1),
//!   and `clip_graph_qwen2vl::build_inp_with_temporal_merge` feeds frame 0 to the first, frame 1
//!   to the second;
//! - ViT attention is per frame pair (`get_vision_cu_seqlens`, `merge_temporal=False`: one segment
//!   of `h*w` per temporal index), the 2-D rope and the learned position table repeat per pair —
//!   so a pair is encoded EXACTLY as an image is, with a different patch embedding;
//! - the LM's M-RoPE splits `video_grid_thw` into `t` grids of `t = 1`
//!   (`get_rope_index`: "timestamps are used to separate videos"), so each pair is placed exactly
//!   like an image: `(p0, p0 + row, p0 + col)` and the text advances by `max(rows, cols)`.
//!   `mrope::mrope_layout` with one `ImageGrid` per pair is therefore the video layout, unchanged.
//!
//! Where Arf deviates, on purpose: the total pixel budget is capped by `ARF_QWEN_VIDEO_MAX_TOKENS`
//! (default 2048 LM tokens for the whole video, vs the model's 25165824 px ~ 12288) — the same
//! kind of cap the image path has — and frames are resized with the image path's Pillow-exact
//! bicubic (HF's video processor uses torchvision's). NOT MEASURED: whether either changes text.

use crate::model::qwen_vision::QwenImage;
use crate::{ArfError, Result};

/// Frames per temporal patch (the Conv3d's temporal kernel).
pub const TEMPORAL_PATCH: usize = 2;

/// The video budget: HF `Qwen3VLVideoProcessor` defaults plus the model's own
/// `video_preprocessor_config.json`, with Arf's cap folded into `max_pixels`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoBudget {
    /// Frames sampled per second of video (HF `fps = 2`).
    pub fps: f64,
    pub min_frames: usize,
    pub max_frames: usize,
    /// `size.shortest_edge`: minimum pixels over ALL frames (4096).
    pub min_pixels: usize,
    /// `size.longest_edge`: maximum pixels over ALL frames (25165824 in the model's config).
    pub max_pixels: usize,
    /// `max_video_tokens`: the per-frame cap of `cap_pixels_per_frame`, in 32x32 cells (768).
    pub max_frame_tokens: usize,
}

impl VideoBudget {
    /// The reference numbers, no Arf cap.
    pub fn reference() -> Self {
        VideoBudget {
            fps: 2.0,
            min_frames: 4,
            max_frames: 768,
            min_pixels: 4096,
            max_pixels: 25_165_824,
            max_frame_tokens: 768,
        }
    }

    /// The reference with the total capped at `max_tokens` LM tokens: a pair of `h x w` frames
    /// costs `h*w / 1024` tokens, so `max_tokens` tokens is `max_tokens * 2 * 32 * 32` pixels over
    /// all frames. Approximate for an odd frame count (the budget counts `n` frames, the encoder
    /// runs `ceil(n/2)` pairs), as in the reference.
    pub fn capped(max_tokens: usize, align: usize) -> Self {
        let r = Self::reference();
        let cap = max_tokens.max(1) * TEMPORAL_PATCH * align * align;
        VideoBudget {
            max_pixels: r.max_pixels.min(cap),
            ..r
        }
    }

    /// [`Self::capped`] at `ARF_QWEN_VIDEO_MAX_TOKENS` (default 2048).
    pub fn from_env(align: usize) -> Self {
        let t = std::env::var("ARF_QWEN_VIDEO_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048usize);
        Self::capped(t, align)
    }
}

/// Which frames of a `total`-frame video at `src_fps` to keep — HF `sample_frames`:
/// `n = int(total / src_fps * fps)`, clamped to `[min_frames, max_frames]` and to `total`, then
/// `round(linspace(0, total - 1, n))` with numpy's ties-to-even rounding.
pub fn sample_frame_indices(
    total: usize,
    src_fps: f64,
    fps: f64,
    min_frames: usize,
    max_frames: usize,
) -> Vec<usize> {
    if total == 0 {
        return Vec::new();
    }
    let src_fps = if src_fps > 0.0 { src_fps } else { 24.0 }; // HF's own fallback
    let n = (total as f64 / src_fps * fps) as usize; // Python int(): truncation
    let n = n.max(min_frames).min(max_frames).min(total);
    linspace_round(total - 1, n)
}

/// `np.linspace(0, last, n).round().astype(int)`: numpy computes `i * (last / (n - 1))` and pins
/// the final value to `last`.
fn linspace_round(last: usize, n: usize) -> Vec<usize> {
    match n {
        0 => Vec::new(),
        1 => vec![0],
        _ => {
            let step = last as f64 / (n - 1) as f64;
            (0..n)
                .map(|i| {
                    if i == n - 1 {
                        last
                    } else {
                        (i as f64 * step).round_ties_even() as usize
                    }
                })
                .collect()
        }
    }
}

/// HF's video `smart_resize` (5.17), returning `(width, height)`: sides rounded (ties-to-even) to
/// multiples of `factor`, then scaled so that `t_bar * h * w` fits `[min_pixels, max_pixels]`, where
/// `t_bar` is `num_frames` rounded to the temporal factor but the scale uses `num_frames` itself.
/// A side under `factor` is first scaled up (keeping the aspect). Errors where the reference raises.
pub fn smart_resize_video(
    num_frames: usize,
    width: usize,
    height: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if num_frames < TEMPORAL_PATCH {
        return Err(ArfError::model_load(format!(
            "qwen video: {num_frames} frame(s); a video needs at least {TEMPORAL_PATCH} (send one frame as an image)"
        )));
    }
    if width == 0 || height == 0 {
        return Err(ArfError::model_load("qwen video: empty frame"));
    }
    let f = factor as f64;
    let (mut h, mut w) = (height as f64, width as f64);
    if height < factor || width < factor {
        let scale = (f / h).max(f / w);
        h = (h * scale).trunc();
        w = (w * scale).trunc();
    }
    if h.max(w) / h.min(w) > 200.0 {
        return Err(ArfError::model_load(format!(
            "qwen video: aspect ratio {:.0} is over 200",
            h.max(w) / h.min(w)
        )));
    }
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    let t_bar =
        (num_frames as f64 / TEMPORAL_PATCH as f64).round_ties_even() * TEMPORAL_PATCH as f64;
    let n = num_frames as f64;
    if t_bar * h_bar * w_bar > max_pixels as f64 {
        let beta = ((n * h * w) / max_pixels as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if t_bar * h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (n * h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((w_bar as usize, h_bar as usize))
}

/// The size every frame of an `num_frames`-frame `width x height` video is resized to — HF
/// `Qwen3VLVideoProcessor.resize` with `cap_pixels_per_frame=True`: each frame gets
/// `max(min(max_frame_tokens * factor^2, max_pixels // n), int(min_pixels * 1.05))` pixels, and the
/// video `smart_resize` runs against that times `n`.
pub fn video_frame_size(
    b: &VideoBudget,
    num_frames: usize,
    width: usize,
    height: usize,
    factor: usize,
) -> Result<(usize, usize)> {
    let frame_cap = b.max_frame_tokens * factor * factor;
    let per_frame = frame_cap
        .min(b.max_pixels / num_frames.max(1))
        .max((b.min_pixels as f64 * 1.05) as usize);
    smart_resize_video(
        num_frames,
        width,
        height,
        factor,
        b.min_pixels,
        per_frame * num_frames,
    )
}

/// One timestamp per frame pair — HF `_calculate_timestamps`: `index / fps` per frame, the index
/// list padded to a pair by repeating the last index, each pair averaged.
pub fn group_timestamps(indices: &[usize], src_fps: f64) -> Vec<f64> {
    let mut idx = indices.to_vec();
    if let Some(&last) = idx.last() {
        while !idx.len().is_multiple_of(TEMPORAL_PATCH) {
            idx.push(last);
        }
    }
    let ts: Vec<f64> = idx.iter().map(|&i| i as f64 / src_fps).collect();
    ts.chunks(TEMPORAL_PATCH)
        .map(|p| (p[0] + p[TEMPORAL_PATCH - 1]) / 2.0)
        .collect()
}

/// The text ahead of each frame pair: `<{t:.1f} seconds>` (Python's `.1f` and Rust's `{:.1}` both
/// round the exact binary value half-to-even; `timestamp_text_matches_python` pins it).
pub fn timestamp_text(t: f64) -> String {
    format!("<{t:.1} seconds>")
}

/// A preprocessed video: an EVEN number of same-size frames (an odd count has its last frame
/// repeated, as HF `patchify` does) and one timestamp per frame pair.
#[derive(Debug, Clone)]
pub struct QwenVideo {
    pub frames: Vec<QwenImage>,
    pub timestamps: Vec<f64>,
}

impl QwenVideo {
    /// Build from preprocessed frames (same size, at least 2) and per-pair timestamps; pads an odd
    /// frame count with the last frame.
    pub fn new(mut frames: Vec<QwenImage>, timestamps: Vec<f64>) -> Result<Self> {
        if frames.len() < TEMPORAL_PATCH {
            return Err(ArfError::model_load(format!(
                "qwen video: {} frame(s); need at least {TEMPORAL_PATCH}",
                frames.len()
            )));
        }
        let (w, h) = (frames[0].width, frames[0].height);
        if let Some(f) = frames.iter().find(|f| (f.width, f.height) != (w, h)) {
            return Err(ArfError::model_load(format!(
                "qwen video: frames of {}x{} and {}x{} in one video",
                w, h, f.width, f.height
            )));
        }
        if !frames.len().is_multiple_of(TEMPORAL_PATCH) {
            let last = frames.last().expect("non-empty").clone();
            frames.push(last);
        }
        if timestamps.len() != frames.len() / TEMPORAL_PATCH {
            return Err(ArfError::model_load(format!(
                "qwen video: {} timestamps for {} frame pairs",
                timestamps.len(),
                frames.len() / TEMPORAL_PATCH
            )));
        }
        Ok(QwenVideo { frames, timestamps })
    }

    /// Number of frame pairs (the temporal grid `t`).
    pub fn groups(&self) -> usize {
        self.frames.len() / TEMPORAL_PATCH
    }

    /// The pair `k`: `(frame 2k, frame 2k+1)`.
    pub fn pair(&self, k: usize) -> (&QwenImage, &QwenImage) {
        (&self.frames[2 * k], &self.frames[2 * k + 1])
    }

    /// Merged-token grid `(rows, cols)` of every pair.
    pub fn token_grid(&self) -> (usize, usize) {
        self.frames[0].token_grid()
    }

    /// `<|video_pad|>` tokens per pair.
    pub fn tokens_per_group(&self) -> usize {
        self.frames[0].n_tokens()
    }

    /// All `<|video_pad|>` tokens of the video.
    pub fn n_tokens(&self) -> usize {
        self.groups() * self.tokens_per_group()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected value below was printed by a numpy transcription of the transformers 5.17.0
    // functions named in the module docs (copied verbatim, torch swapped for numpy), 2026-09-27.

    #[test]
    fn frame_sampling_matches_hf_sample_frames() {
        let s = |total, src, fps| sample_frame_indices(total, src, fps, 4, 768);
        // 10 s at 30 fps, 2 fps -> 20 frames.
        let a = s(300, 30.0, 2.0);
        assert_eq!(a.len(), 20);
        assert_eq!(&a[..6], &[0, 16, 31, 47, 63, 79]);
        assert_eq!(&a[16..], &[252, 268, 283, 299]);
        assert_eq!(s(61, 29.97, 2.0), vec![0, 20, 40, 60]);
        // 0.4 s: int(0.8) = 0 frames -> min_frames 4.
        assert_eq!(s(10, 25.0, 2.0), vec![0, 3, 6, 9]);
        // Shorter than min_frames: every frame.
        assert_eq!(s(3, 30.0, 2.0), vec![0, 1, 2]);
        // linspace(0, 6, 5) = 0, 1.5, 3, 4.5, 6 -> ties to EVEN: 2 and 4, not 2 and 5.
        assert_eq!(s(7, 7.0, 5.0), vec![0, 2, 3, 4, 6]);
        // Long video: capped at max_frames.
        let l = s(100_000, 30.0, 2.0);
        assert_eq!(l.len(), 768);
        assert_eq!(&l[..6], &[0, 130, 261, 391, 522, 652]);
        assert_eq!(&l[764..], &[99608, 99738, 99869, 99999]);
        assert_eq!(
            s(250, 25.0, 1.0),
            vec![0, 28, 55, 83, 111, 138, 166, 194, 221, 249]
        );
        assert!(s(0, 30.0, 2.0).is_empty());
    }

    #[test]
    fn frame_size_matches_hf_resize_with_the_per_frame_cap() {
        let r = VideoBudget::reference();
        let arf = VideoBudget::capped(2048, 32);
        assert_eq!(arf.max_pixels, 2048 * 2 * 1024);
        // (n, w, h) -> reference (w, h), Arf-capped (w, h)
        let cases = [
            ((20, 1920, 1080), (1152, 640), (608, 320)),
            ((4, 640, 480), (640, 480), (640, 480)),
            ((768, 1280, 720), (224, 128), (96, 32)),
            ((3, 100, 50), (96, 64), (96, 64)),
            ((2, 700, 20), (1120, 32), (1120, 32)), // a side under 32 is scaled up first
            ((8, 80, 48), (64, 64), (64, 64)),      // 2.5 and 1.5 both round to 2 (ties-to-even)
            ((2, 64, 64), (64, 64), (64, 64)),
            ((40, 1280, 720), (1056, 576), (416, 224)),
            ((5, 333, 777), (320, 768), (320, 768)),
        ];
        for ((n, w, h), want_ref, want_arf) in cases {
            assert_eq!(
                video_frame_size(&r, n, w, h, 32).unwrap(),
                want_ref,
                "ref {n} {w}x{h}"
            );
            assert_eq!(
                video_frame_size(&arf, n, w, h, 32).unwrap(),
                want_arf,
                "arf {n} {w}x{h}"
            );
        }
        // Raw smart_resize, no budget: Python round(1.5) = round(2.5) = 2.
        let big = 1usize << 40;
        assert_eq!(smart_resize_video(4, 80, 48, 32, 0, big).unwrap(), (64, 64));
        assert_eq!(smart_resize_video(4, 48, 80, 32, 0, big).unwrap(), (64, 64));
        assert_eq!(
            smart_resize_video(4, 144, 112, 32, 0, big).unwrap(),
            (128, 128)
        );
        assert!(smart_resize_video(1, 64, 64, 32, 0, big).is_err());
        assert!(smart_resize_video(4, 64, 64 * 201, 32, 0, big).is_err());
    }

    #[test]
    fn timestamps_match_hf_calculate_timestamps() {
        let t = group_timestamps(&[0, 20, 40, 60], 29.97);
        assert_eq!(t, vec![0.333_667_000_333_667_03, 1.668_335_001_668_335_1]);
        // odd: the last index is repeated
        assert_eq!(
            group_timestamps(&[0, 1, 2], 30.0),
            vec![0.016_666_666_666_666_666, 0.066_666_666_666_666_67]
        );
        assert_eq!(group_timestamps(&[0, 1, 2, 3], 2.0), vec![0.25, 1.25]);
        let txt: Vec<String> = group_timestamps(&[0, 20, 40, 60], 29.97)
            .into_iter()
            .map(timestamp_text)
            .collect();
        assert_eq!(txt, ["<0.3 seconds>", "<1.7 seconds>"]);
    }

    /// Python `f"{t:.1f}"` on the values the processor produces, ties included.
    #[test]
    fn timestamp_text_matches_python() {
        for (t, want) in [
            (0.25, "<0.2 seconds>"),
            (0.75, "<0.8 seconds>"),
            (1.25, "<1.2 seconds>"),
            (2.25, "<2.2 seconds>"),
            (0.05, "<0.1 seconds>"),
            (0.15, "<0.1 seconds>"),
            (12.35, "<12.3 seconds>"),
            (3.45, "<3.5 seconds>"),
            (0.3336, "<0.3 seconds>"),
            (99.95, "<100.0 seconds>"),
            (0.016_666_666_666_666_666, "<0.0 seconds>"),
            (0.125, "<0.1 seconds>"),
        ] {
            assert_eq!(timestamp_text(t), want, "{t}");
        }
    }

    fn frame(gw: usize, gh: usize, v: f32) -> QwenImage {
        QwenImage {
            pixels: vec![v; 3 * gw * 16 * gh * 16],
            width: gw * 16,
            height: gh * 16,
            grid_w: gw,
            grid_h: gh,
        }
    }

    #[test]
    fn odd_frame_counts_repeat_the_last_frame() {
        let v = QwenVideo::new(
            vec![frame(4, 2, 0.0), frame(4, 2, 1.0), frame(4, 2, 2.0)],
            vec![0.0, 1.0],
        )
        .unwrap();
        assert_eq!(v.frames.len(), 4);
        assert_eq!(v.groups(), 2);
        assert_eq!(v.frames[3].pixels[0], 2.0);
        assert_eq!(v.token_grid(), (1, 2));
        assert_eq!(v.tokens_per_group(), 2);
        assert_eq!(v.n_tokens(), 4);
        assert!(QwenVideo::new(vec![frame(4, 2, 0.0)], vec![0.0]).is_err());
        assert!(QwenVideo::new(vec![frame(4, 2, 0.0), frame(2, 2, 0.0)], vec![0.0]).is_err());
        assert!(QwenVideo::new(vec![frame(4, 2, 0.0), frame(4, 2, 0.0)], vec![]).is_err());
    }
}
