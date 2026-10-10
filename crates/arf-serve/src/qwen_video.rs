//! Qwen3.8 VIDEO input for the HTTP layer (2026-09-27): turn a `video_url` / `video` content part
//! into a preprocessed [`QwenVideo`]. The rules (which frames, what size, what timestamps) are
//! `arf_core::model::qwen_video`'s, each checked against the reference source there; this file is
//! only I/O.
//!
//! Decoding shells out to `ffmpeg` / `ffprobe` from PATH (no new Rust dependency, the approach
//! llama.cpp's `mtmd-helper.cpp` takes too). A server without them answers a video request with a
//! 400 naming the gap; nothing else changes. Only reachable when `--mmproj` names a
//! `qwen3vl_merger` projector AND a request carries a video.
//!
//! Sources:
//! - `data:video/...;base64,...` — written to a private temp file (an MP4's `moov` atom may sit at
//!   the end, so ffmpeg needs a seekable input), removed when done;
//! - `file:///abs/path` or `/abs/path` — only with `ARF_IMAGE_FILES=1`, the same gate as image
//!   files (a file read on behalf of any HTTP client);
//! - `http(s)://` — only with `ARF_MEDIA_URLS=1` (`remote_media`: bounded, public addresses only,
//!   no redirects), written to a private temp file like a data URL;
//! - a LIST of image frames (`{"type": "video", "video": [..]}`, the qwen-vl-utils / DashScope
//!   form), each an image reference as `image_url` takes it, assumed sampled at `fps` (default 2).
//!
//! ffmpeg is run with `-protocol_whitelist file` and the input spelled `file:<abs path>`, so a
//! playlist inside a "video" cannot make the server fetch a URL, and a path cannot be read as an
//! option or a protocol.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use arf_core::model::qwen_video::{
    group_timestamps, sample_frame_indices, video_frame_size, QwenVideo, VideoBudget,
    TEMPORAL_PATCH,
};
use arf_core::model::qwen_vision::{preprocess_to, QwenImage, QwenVisionConfig};

/// Where a video's frames come from.
#[derive(Debug, Clone, PartialEq)]
pub enum VideoSource {
    /// A `data:` URL, `file://` URL or absolute path to a video file.
    Url(String),
    /// Already-sampled frames, each an image reference (data URL, or a gated file path).
    Frames(Vec<String>),
}

/// One `video_url` / `video` content part.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoInput {
    pub source: VideoSource,
    /// For a file: frames to sample per second (default 2, the processor's). For a frame list:
    /// the rate the frames were sampled at (default 2, qwen-vl-utils' `sample_fps`).
    pub fps: Option<f64>,
    /// Cap on sampled frames (default 768, the processor's `max_frames`).
    pub max_frames: Option<usize>,
}

impl VideoInput {
    /// A short name for logs and for the `images` slot the video occupies in `ChatMessage`.
    pub fn label(&self) -> String {
        match &self.source {
            VideoSource::Url(u) if u.starts_with("data:") => "video:data-url".into(),
            VideoSource::Url(u) => format!("video:{u}"),
            VideoSource::Frames(f) => format!("video:{} frames", f.len()),
        }
    }
}

/// Decode and preprocess one video part: `[frames]` at the video budget's size, timestamps per
/// frame pair.
pub fn load_video(cfg: &QwenVisionConfig, v: &VideoInput) -> Result<QwenVideo, String> {
    let mut budget = VideoBudget::from_env(cfg.align());
    if let Some(f) = v.fps {
        if !(f.is_finite() && f > 0.0) {
            return Err(format!("video fps must be positive, got {f}"));
        }
        budget.fps = f;
    }
    if let Some(m) = v.max_frames {
        budget.max_frames = m.max(TEMPORAL_PATCH);
    }
    match &v.source {
        VideoSource::Url(url) => load_video_url(cfg, &budget, url),
        VideoSource::Frames(frames) => load_frame_list(cfg, &budget, frames),
    }
}

/// A frame list, qwen-vl-utils style: padded to an even count by repeating the last frame, frame
/// `i` at `i / fps` seconds, all frames resized to the size the first one's shape gets.
fn load_frame_list(
    cfg: &QwenVisionConfig,
    b: &VideoBudget,
    urls: &[String],
) -> Result<QwenVideo, String> {
    if urls.len() < TEMPORAL_PATCH {
        return Err(format!(
            "a video needs at least {TEMPORAL_PATCH} frames ({} given); send one frame as an image",
            urls.len()
        ));
    }
    if urls.len() > b.max_frames {
        return Err(format!(
            "{} frames is more than max_frames {}",
            urls.len(),
            b.max_frames
        ));
    }
    let padded = urls.len().next_multiple_of(TEMPORAL_PATCH);
    let mut size = None;
    let mut frames = Vec::with_capacity(padded);
    for u in urls {
        let (rgb, w, h) = crate::qwen_image::load_image(u)?;
        let (tw, th) = match size {
            Some(s) => s,
            None => {
                let s =
                    video_frame_size(b, padded, w, h, cfg.align()).map_err(|e| e.to_string())?;
                size = Some(s);
                s
            }
        };
        frames.push(preprocess_to(cfg, &rgb, w, h, tw, th));
    }
    let indices: Vec<usize> = (0..padded).collect();
    let ts = group_timestamps(&indices, b.fps);
    QwenVideo::new(frames, ts).map_err(|e| e.to_string())
}

/// A video file or data URL through ffprobe + ffmpeg.
fn load_video_url(cfg: &QwenVisionConfig, b: &VideoBudget, url: &str) -> Result<QwenVideo, String> {
    let file = VideoFile::resolve(url)?;
    let info = probe(file.path())?;
    let indices = sample_frame_indices(info.frames, info.fps, b.fps, b.min_frames, b.max_frames);
    if indices.len() < TEMPORAL_PATCH {
        return Err(format!(
            "the video has {} decodable frame(s); send one frame as an image",
            info.frames
        ));
    }
    let (tw, th) = video_frame_size(b, indices.len(), info.width, info.height, cfg.align())
        .map_err(|e| e.to_string())?;
    let rgb_frames = extract_frames(file.path(), &indices, info.width, info.height)?;
    let frames: Vec<QwenImage> = rgb_frames
        .iter()
        .map(|rgb| preprocess_to(cfg, rgb, info.width, info.height, tw, th))
        .collect();
    let ts = group_timestamps(&indices, info.fps);
    if std::env::var_os("ARF_VISION_TIMING").is_some() {
        eprintln!(
            "[qwen-video] {}x{} {} frames at {:.3} fps -> {} sampled at {tw}x{th}",
            info.width,
            info.height,
            info.frames,
            info.fps,
            indices.len()
        );
    }
    QwenVideo::new(frames, ts).map_err(|e| e.to_string())
}

/// A video on disk: the client's file, or a temp file holding a data URL's bytes (removed on drop).
struct VideoFile {
    path: PathBuf,
    temp: bool,
}

impl VideoFile {
    fn resolve(url: &str) -> Result<VideoFile, String> {
        if crate::remote_media::is_remote(url) {
            return VideoFile::temp(&crate::remote_media::fetch(url, "video")?);
        }
        if url.starts_with("data:") {
            use base64::Engine;
            let b64 = url
                .find("base64,")
                .map(|i| &url[i + "base64,".len()..])
                .ok_or("video data URL must be base64-encoded")?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64.trim())
                .map_err(|e| format!("video base64 decode: {e}"))?;
            return VideoFile::temp(&bytes);
        }
        let p = url.strip_prefix("file://").unwrap_or(url);
        if !p.starts_with('/') {
            return Err(
                "video url must be a data: URL, an http(s) URL, a file:// URL or an absolute \
                 path"
                    .into(),
            );
        }
        if std::env::var_os("ARF_IMAGE_FILES").is_none() {
            return Err(
                "video file paths are disabled (set ARF_IMAGE_FILES=1 on the server to allow)"
                    .into(),
            );
        }
        let path = PathBuf::from(p);
        if !path.is_file() {
            return Err(format!("video file {p}: not found"));
        }
        Ok(VideoFile { path, temp: false })
    }

    /// A private temp file holding `bytes` (a data URL's or a fetched URL's), removed on drop.
    fn temp(bytes: &[u8]) -> Result<VideoFile, String> {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("arf-video-{}-{n}.bin", std::process::id()));
        std::fs::write(&path, bytes).map_err(|e| format!("video temp file: {e}"))?;
        Ok(VideoFile { path, temp: true })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for VideoFile {
    fn drop(&mut self) {
        if self.temp {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// What ffprobe says about the first video stream (display orientation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    /// Decoded frames (`-count_frames`: every frame decoded once, the count the reference's
    /// readers take from the decoded tensor).
    pub frames: usize,
    pub fps: f64,
}

fn tool(name: &str) -> Command {
    let mut c = Command::new(name);
    c.stdin(Stdio::null());
    c
}

fn spawn_err(name: &str, e: std::io::Error) -> String {
    if e.kind() == std::io::ErrorKind::NotFound {
        format!("video input needs `{name}` on the server's PATH (install ffmpeg)")
    } else {
        format!("running {name}: {e}")
    }
}

/// `file:` + the absolute path: ffmpeg never reads it as an option or another protocol.
fn ff_input(path: &Path) -> String {
    format!("file:{}", path.display())
}

/// ffprobe the first video stream: size (swapped for a 90/270-degree display rotation, which
/// ffmpeg applies when decoding), decoded frame count, frame rate.
pub fn probe(path: &Path) -> Result<VideoInfo, String> {
    let out = tool("ffprobe")
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames:stream_side_data=rotation",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(ff_input(path))
        .output()
        .map_err(|e| spawn_err("ffprobe", e))?;
    if !out.status.success() {
        return Err(format!(
            "ffprobe could not read the video: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    parse_probe(&String::from_utf8_lossy(&out.stdout))
}

/// Parse ffprobe's `key=value` lines (split out for the unit test).
pub fn parse_probe(text: &str) -> Result<VideoInfo, String> {
    let (mut w, mut h, mut frames) = (0usize, 0usize, 0usize);
    let (mut avg, mut r) = (0f64, 0f64);
    let mut rotation = 0i64;
    let rate = |s: &str| -> f64 {
        match s.split_once('/') {
            Some((a, b)) => match (a.parse::<f64>(), b.parse::<f64>()) {
                (Ok(a), Ok(b)) if b > 0.0 => a / b,
                _ => 0.0,
            },
            None => s.parse().unwrap_or(0.0),
        }
    };
    for line in text.lines() {
        let Some((k, v)) = line.trim().split_once('=') else {
            continue;
        };
        match k {
            "width" => w = v.parse().unwrap_or(0),
            "height" => h = v.parse().unwrap_or(0),
            "nb_read_frames" => frames = v.parse().unwrap_or(0),
            "avg_frame_rate" => avg = rate(v),
            "r_frame_rate" => r = rate(v),
            "rotation" => rotation = v.parse::<f64>().map(|x| x as i64).unwrap_or(0),
            _ => {}
        }
    }
    if w == 0 || h == 0 || frames == 0 {
        return Err("ffprobe found no decodable video stream".into());
    }
    if rotation.rem_euclid(180) == 90 {
        std::mem::swap(&mut w, &mut h);
    }
    let fps = if avg > 0.0 { avg } else { r };
    if fps <= 0.0 {
        return Err("ffprobe reported no frame rate".into());
    }
    Ok(VideoInfo {
        width: w,
        height: h,
        frames,
        fps,
    })
}

/// Decode exactly the frames `indices` (decode order, 0-based, may repeat) as RGB8 `[h, w, 3]`,
/// through ffmpeg's `select` filter. Errors unless every requested frame arrives at `w x h`.
pub fn extract_frames(
    path: &Path,
    indices: &[usize],
    w: usize,
    h: usize,
) -> Result<Vec<Vec<u8>>, String> {
    let mut uniq: Vec<usize> = indices.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    let select = uniq
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");
    let mut child = tool("ffmpeg")
        .args([
            "-nostdin",
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-i",
        ])
        .arg(ff_input(path))
        .args(["-vf", &format!("select='{select}'")])
        .args([
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| spawn_err("ffmpeg", e))?;
    // Drain stderr on its own thread so a chatty ffmpeg cannot block on a full pipe.
    let mut err_pipe = child.stderr.take().expect("piped");
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err_pipe.read_to_string(&mut s);
        s
    });
    let mut out = child.stdout.take().expect("piped");
    let frame_bytes = w * h * 3;
    let mut got: Vec<Vec<u8>> = Vec::with_capacity(uniq.len());
    let mut read_err = None;
    for _ in 0..uniq.len() {
        let mut buf = vec![0u8; frame_bytes];
        if let Err(e) = out.read_exact(&mut buf) {
            read_err = Some(e);
            break;
        }
        got.push(buf);
    }
    // Anything past the last requested frame means the size or count assumption was wrong.
    let mut extra = [0u8; 1];
    let trailing = read_err.is_none() && out.read(&mut extra).map(|n| n > 0).unwrap_or(false);
    drop(out);
    let status = child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
    let stderr = err_thread.join().unwrap_or_default();
    if !status.success() || read_err.is_some() || trailing {
        return Err(format!(
            "ffmpeg returned {} of {} frames at {w}x{h}{}: {}",
            got.len(),
            uniq.len(),
            if trailing { " (and more bytes)" } else { "" },
            stderr.trim()
        ));
    }
    Ok(indices
        .iter()
        .map(|i| got[uniq.binary_search(i).expect("from uniq")].clone())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_output_parses_with_rotation() {
        let a =
            "width=320\nheight=240\nr_frame_rate=10/1\navg_frame_rate=10/1\nnb_read_frames=20\n";
        assert_eq!(
            parse_probe(a).unwrap(),
            VideoInfo {
                width: 320,
                height: 240,
                frames: 20,
                fps: 10.0
            }
        );
        let r = format!("{a}rotation=-90\n");
        let i = parse_probe(&r).unwrap();
        assert_eq!((i.width, i.height), (240, 320));
        let ntsc =
            "width=64\nheight=64\nr_frame_rate=30000/1001\navg_frame_rate=0/0\nnb_read_frames=3\n";
        assert!((parse_probe(ntsc).unwrap().fps - 29.97).abs() < 1e-3);
        assert!(parse_probe("width=0\nheight=0\n").is_err());
    }

    #[test]
    fn remote_and_relative_urls_are_refused() {
        for u in ["v.mp4", "file://relative.mp4"] {
            let e = VideoFile::resolve(u).err().expect("refused");
            assert!(e.contains("absolute path"), "{u}: {e}");
        }
        if std::env::var_os("ARF_MEDIA_URLS").is_none() {
            let e = VideoFile::resolve("https://example.com/v.mp4")
                .err()
                .expect("off");
            assert!(e.contains("ARF_MEDIA_URLS=1"), "{e}");
        }
        if std::env::var_os("ARF_IMAGE_FILES").is_none() {
            let e = VideoFile::resolve("/etc/hosts").err().expect("gated");
            assert!(e.contains("disabled"), "{e}");
        }
        assert!(VideoFile::resolve("data:video/mp4,AAAA").is_err());
    }

    fn have_ffmpeg() -> bool {
        ["ffmpeg", "ffprobe"].iter().all(|t| {
            Command::new(t)
                .arg("-version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        })
    }

    /// End to end through the real ffmpeg (CPU only): a generated 2 s, 10 fps, 64x48 clip whose
    /// frame k is a flat gray of level 10k, as a data URL. The sampler must pick HF's indices
    /// (20 frames at 10 fps, 2 fps -> 4 frames: 0, 6, 13, 19) and the decoded frames must be
    /// THOSE frames (their gray level says which). Skips loudly without ffmpeg.
    #[test]
    fn ffmpeg_decodes_the_sampled_frames_of_a_data_url() {
        if !have_ffmpeg() {
            eprintln!("SKIP: ffmpeg/ffprobe not on PATH -- this run verified NOTHING");
            return;
        }
        let dir = std::env::temp_dir().join(format!("arf-video-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("gray.mkv");
        // geq: every pixel of frame N is 10*N (lossless ffv1 in rgb24, so levels survive).
        let st = Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-f", "lavfi", "-i"])
            .arg("color=c=black:s=64x48:r=10:d=2,format=rgb24,geq=r='10*N':g='10*N':b='10*N'")
            .args(["-c:v", "ffv1", "-pix_fmt", "rgb24"])
            .arg(&clip)
            .status()
            .unwrap();
        assert!(st.success(), "ffmpeg could not make the test clip");
        use base64::Engine;
        let url = format!(
            "data:video/x-matroska;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(std::fs::read(&clip).unwrap())
        );
        let file = VideoFile::resolve(&url).unwrap();
        let tmp = file.path().to_path_buf();
        let info = probe(file.path()).unwrap();
        assert_eq!(
            info,
            VideoInfo {
                width: 64,
                height: 48,
                frames: 20,
                fps: 10.0
            }
        );
        let idx = sample_frame_indices(info.frames, info.fps, 2.0, 4, 768);
        assert_eq!(idx, vec![0, 6, 13, 19]);
        let frames = extract_frames(file.path(), &[0, 6, 13, 19, 6], 64, 48).unwrap();
        assert_eq!(frames.len(), 5);
        for (f, want) in frames.iter().zip([0u8, 60, 130, 190, 60]) {
            assert_eq!(f.len(), 64 * 48 * 3);
            assert!(
                f.iter().all(|&v| v == want),
                "frame level {} != {want}",
                f[0]
            );
        }
        drop(file);
        assert!(!tmp.exists(), "the data URL's temp file must be removed");

        // Through the whole loader with a tiny config (patch 4 -> 8-pixel cells).
        let enc = arf_core::model::qwen_vision::synthetic::tiny();
        let v = VideoInput {
            source: VideoSource::Url(url),
            fps: None,
            max_frames: None,
        };
        let video = load_video(&enc.cfg, &v).unwrap();
        assert_eq!(video.frames.len(), 4);
        assert_eq!(video.groups(), 2);
        // timestamps: (0 + 0.6) / 2 = 0.3, (1.3 + 1.9) / 2 = 1.6
        assert_eq!(video.timestamps.len(), 2);
        assert!((video.timestamps[0] - 0.3).abs() < 1e-12);
        assert!((video.timestamps[1] - 1.6).abs() < 1e-12);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
