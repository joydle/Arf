//! Transcode cache + mmap zero-copy weight loading (macOS island path).
//!
//! ## The problem
//! Cold start for qwen3-coder-30B Q4_K_S was ~150-195s: ~150s to copy+transcode+upload
//! ~19 GB of weights (GGUF Q4_K → our Q4_K_S layout, a lossless REPACK that changes byte
//! order so it is NOT a free re-view), then a ~45s first-decode page-in stall. llama.cpp
//! loads in SECONDS by mmap'ing the GGUF and letting the OS demand-page it into unified
//! memory, zero copy. This module closes that gap with two compounding wins:
//!
//! ## WIN 1 — transcode cache (kills the ~150s on the 2nd+ run)
//! On the FIRST load we transcode the GGUF → q4ks and write every weight region (codes /
//! scales / mins / dd for every dense + MoE-expert + norm buffer) into a SIDECAR cache
//! file, each region page-aligned, indexed by its buffer label. On subsequent loads, if
//! the cache matches (GGUF identity + layout version), we `mmap` the cache and wrap each
//! region DIRECTLY as an `MTLBuffer` via `newBufferWithBytesNoCopy` — zero transcode, zero
//! copy → load in seconds.
//!
//! ## WIN 2 — no-copy on the FIRST load too (kills the 45s page-in + halves the copy)
//! Even on the cache-building run we transcode straight INTO the cache file's mmap'd region
//! and wrap THAT region no-copy, instead of Vec-then-memcpy-into-a-fresh-MTLBuffer. This
//! removes the second copy AND makes the weights file-backed + demand-paged (like llama),
//! so the OS can prefetch/overlap the page-in instead of faulting a fresh 19 GB anonymous
//! alloc all at once on the first dispatch.
//!
//! ## Zero-copy contract (Apple Silicon)
//! `newBufferWithBytesNoCopy` requires the host pointer page-aligned and the length a
//! page multiple on macOS (16 KB pages on Apple Silicon). Each cache region is padded up
//! to a page boundary so the wrap is legal. The mmap that backs every wrapped MTLBuffer
//! MUST outlive those buffers (they alias it) — we hand the `Arc<Mmap>` to the caller to
//! store on the `GpuModel`. We pass a `None` deallocator: lifetime is owned by the Arc,
//! NOT by Metal releasing the buffer.
//!
//! ## Keying
//! The cache is keyed by the GGUF's `(len, mtime)` + this module's `LAYOUT_VERSION` + the
//! quant/path env flags that change WHAT gets stored. A full sha256 over 19 GB would itself
//! cost seconds (defeating the point); `(len, mtime)` is the same fast identity llama.cpp's
//! own mmap relies on and is robust for an immutable blob. Gate: `ARF_WEIGHT_CACHE` is
//! default-ON; `ARF_NO_WEIGHT_CACHE` disables both wins (pure transcode+copy path).

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapMut};

/// Apple Silicon system page size. `newBufferWithBytesNoCopy` needs the wrapped pointer
/// page-aligned and the length a page multiple; cache regions are padded to this.
const PAGE: usize = 16 * 1024;

/// Bump when the transcode layout, packing, or region set changes so a stale cache is
/// rejected instead of silently miscompiling weights. Tied to the q4ks repack + the
/// scales/mins 4-per-u32 packing + the [d,dmin] interleave this loader emits.
///
/// 2 (2026-09-24): the RoPE tables left the cache (they were served stale — see `region_for`);
/// every sidecar written before is rebuilt once so none of them keeps one.
const LAYOUT_VERSION: u32 = 2;

/// Magic + header for the sidecar cache file. Layout:
///   [u64 magic][u32 version][u32 _pad][u64 gguf_len][u64 gguf_mtime_ns]
///   [u64 flags_hash][u64 n_entries]
///   then n_entries × index records (written at the END, offset stored in header), then
///   the page-aligned data blob. The index is appended last (after all data) so the
///   builder streams data forward without seeking.
const MAGIC: u64 = 0x5052_4d52_5751_3453; // "PRMRWQ4S"

/// One cached buffer region: label → (data offset, real byte length). The region in the
/// file is padded to a page multiple after `len` so the NEXT region is also page-aligned.
#[derive(Clone)]
struct Entry {
    offset: u64,
    len: u64,
}

/// Per-load cache mode held in a thread-local for the duration of `build_gpu_inner`.
enum Mode {
    /// Cache HIT: the sidecar is mmap'd; `index` maps label → region in `mmap`.
    Read {
        mmap: Arc<Mmap>,
        index: HashMap<String, Entry>,
    },
    /// Cache MISS / build: append each region (page-aligned) to `file`, recording the
    /// index. `mmap` is the writable mapping we transcode into so the SAME bytes can be
    /// wrapped no-copy on this very run (WIN 2). `offset` is the next write offset.
    Build {
        /// The `.building` temp file being written this run (what `mmap`/`file` map).
        path: PathBuf,
        /// The final `<gguf>.arf-q4ks-cache` path; `finish` renames `path` → here on success.
        final_path: PathBuf,
        file: File,
        mmap: MmapMut,
        index: Vec<(String, Entry)>,
        offset: u64,
        gguf_len: u64,
        gguf_mtime_ns: u64,
        flags_hash: u64,
        /// Swap-used (MB) sampled when this build started — the L109 guard in `finish` refuses
        /// to promote a sidecar built while the box was paging (see `swap_used_mb`).
        swap_at_build_start: u64,
    },
}

thread_local! {
    static CACHE: RefCell<Option<Mode>> = const { RefCell::new(None) };
    /// The GGUF path the current `build_gpu_inner` should key its cache on, set by the
    /// GGUF load entry points just before they call `build_gpu_*` (the public build
    /// signatures stay unchanged). `None` for safetensors / non-GGUF loads (no cache).
    static GGUF_PATH: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Record the GGUF path for the imminent `build_gpu_inner` cache session (called by the
/// `load_gguf_gpu_*` entry points). Cleared by `take_gguf_path` inside the build.
pub fn set_gguf_path(path: &Path) {
    GGUF_PATH.with(|p| *p.borrow_mut() = Some(path.to_path_buf()));
}

/// Take the recorded GGUF path (consumes it) for the current build, if any.
pub fn take_gguf_path() -> Option<PathBuf> {
    GGUF_PATH.with(|p| p.borrow_mut().take())
}

/// A weight cache region the caller can wrap no-copy: a raw pointer into a long-lived
/// mmap + its real length. The pointer is page-aligned and the mmap stays alive as long
/// as the keep-alive ([`CacheKeep`]) does (the caller stores it on the GpuModel).
pub struct CacheRegion {
    pub ptr: *mut std::ffi::c_void,
    pub len: usize,
}

/// The long-lived backing for every no-copy MTLBuffer wrapped this load. MUST be stored
/// on the `GpuModel` and live as long as the model — the MTLBuffers alias its pages
/// (dropping it while a buffer is live = use-after-free / GPU fault). Holds the read-only
/// `Mmap` on a cache hit, or the writable build `MmapMut` (the pages the no-copy buffers
/// were wrapped over) on a build run. `Send + Sync`: both mmap types are, and the bytes are
/// read-only after load.
pub enum CacheKeep {
    Read(Arc<Mmap>),
    Build(MmapMut),
}
// SAFETY: after load the mapping is only read (by the GPU); no &mut aliasing of the bytes.
unsafe impl Send for CacheKeep {}
unsafe impl Sync for CacheKeep {}

/// Swap currently in use, in MB (macOS `vm.swapusage`). Used by the L109 build guard: a weight
/// cache written while the box is paging can be silently corrupt, and `memory_pressure` does NOT
/// see swap death (L28) — `vm.swapusage` is the honest signal. Returns 0 when unavailable, which
/// makes the guard a no-op rather than a false alarm.
fn swap_used_mb() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "vm.swapusage"])
            .output()
        else {
            return 0;
        };
        // "total = 7168.00M  used = 5603.81M  free = 1564.19M  (encrypted)"
        let s = String::from_utf8_lossy(&out.stdout);
        let Some(rest) = s.split("used =").nth(1) else {
            return 0;
        };
        rest.trim()
            .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.')
            .split('M')
            .next()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .map(|v| v as u64)
            .unwrap_or(0)
    }
    #[cfg(not(target_os = "macos"))]
    0
}

/// Fast file identity: (length, mtime in ns). Cheap (a `stat`), unlike sha256 over 19 GB.
fn gguf_identity(path: &Path) -> Option<(u64, u64)> {
    let md = std::fs::metadata(path).ok()?;
    let len = md.len();
    let mtime = md.modified().ok()?;
    let ns = mtime.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos() as u64;
    Some((len, ns))
}

/// Hash the env flags that change WHAT gets transcoded/stored, so a cache built under one
/// configuration is not reused under another (e.g. a different QUANT or megakernel set).
fn flags_hash() -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    LAYOUT_VERSION.hash(&mut h);
    // Only env vars that change the CACHED BYTES (quantization / layout / packing) belong here.
    // A var that merely gates runtime KERNEL DISPATCH must NOT be here, or the cache thrashes
    // (rebuilds ~2min) whenever that lever's env state differs run-to-run — the load-slowness
    // regression of 2026-06-28. Removed ARF_BATCH_MEGA (a default-on B-row compile/dispatch gate
    // that does NOT change dense weight bytes — verified: it only selects which kernel runs at
    // decode). QUANT/MSL_GEMV (ddh packing)/MOE_MM_F32/Q3K_FFN DO change stored bytes → keep.
    for k in [
        "QUANT",
        "ARF_MSL_GEMV",
        "ARF_MOE_MM_F32",
        "ARF_Q3K_FFN",
        "ARF_G64_3BIT",
    ] {
        std::env::var(k).unwrap_or_default().hash(&mut h);
    }
    h.finish()
}

/// Cache file path for a GGUF: `<gguf>.arf-q4ks-cache`. Sits next to the blob so it is
/// trivially discoverable + cleanable. (Ollama blobs live in a content-addressed dir, so a
/// sibling sidecar is the natural place.)
fn cache_path_for(gguf: &Path) -> PathBuf {
    let mut s = gguf.as_os_str().to_os_string();
    s.push(".arf-q4ks-cache");
    PathBuf::from(s)
}

/// The TEMP path a build writes to before it is atomically promoted. A build truncates+fills
/// THIS file, never the live `<gguf>.arf-q4ks-cache`; `finish` renames it over the final
/// path only once the header is written. So a killed/timed-out load leaves this `.building`
/// orphan but NEVER destroys a previously-good cache (the all-afternoon fragility of 2026-07-08:
/// a `timeout`-killed rebuild used to truncate the good cache in place → a fresh ~3-min rebuild
/// every time). Same-filesystem rename is atomic AND inode-stable, so the `MmapMut` the no-copy
/// MTLBuffers alias stays valid across the promotion (the mapping references the inode, not the name).
fn build_path_for(gguf: &Path) -> PathBuf {
    let mut s = gguf.as_os_str().to_os_string();
    s.push(".arf-q4ks-cache.building");
    PathBuf::from(s)
}

/// Begin a cache session for this load. Tries to open a matching sidecar (→ `Read`); on
/// miss, creates a fresh build file pre-sized to `est_bytes` (→ `Build`). Returns `true` if
/// a session was started (the caller MUST call `finish` to get the keep-alive), `false` when
/// the cache is disabled or the GGUF identity is unavailable (legacy memcpy path).
///
/// `est_bytes` is an UPPER BOUND on the total transcoded weight bytes (used to pre-size the
/// build file's mmap). Over-estimating is harmless (the file is truncated to the real size
/// in `finish`); under-estimating leaves the overflowing regions uncached but still correct
/// (they fall back to the memcpy path), so callers pass a generous bound.
pub fn begin(gguf: &Path, est_bytes: u64) -> bool {
    if std::env::var_os("ARF_NO_WEIGHT_CACHE").is_some() {
        return false;
    }
    // Default-ON: only skip when explicitly disabled above.
    let Some((gguf_len, gguf_mtime_ns)) = gguf_identity(gguf) else {
        return false;
    };
    let fh = flags_hash();
    let path = cache_path_for(gguf);

    // --- Try READ (cache hit) ---
    if let Some((mmap, index)) = try_open_read(&path, gguf_len, gguf_mtime_ns, fh) {
        eprintln!(
            "[weight-cache] HIT {} ({} regions, mmap zero-copy)",
            path.display(),
            index.len()
        );
        CACHE.with(|c| {
            *c.borrow_mut() = Some(Mode::Read {
                mmap: Arc::new(mmap),
                index,
            })
        });
        return true;
    }

    // --- Build (cache miss): write to a `.building` TEMP file, promote on finish. ---
    // Never touch the live final cache (`path`) here — if this build is killed/timed-out, the
    // previously-good cache survives intact and next load HITs it (the 2026-07-08 fragility fix).
    let build_path = build_path_for(gguf);
    // A `.building` left by a prior killed build is stale garbage — remove it before we start.
    let _ = std::fs::remove_file(&build_path);
    // Reserve room for header + index + a generous page-padded data blob.
    let reserve = est_bytes + (est_bytes / 64) + (16 * 1024 * 1024); // ~1.5% pad + 16 MB
    let file = match OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&build_path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "[weight-cache] build disabled (open {} failed: {e})",
                build_path.display()
            );
            return false;
        }
    };
    if let Err(e) = file.set_len(reserve) {
        eprintln!("[weight-cache] build disabled (set_len failed: {e})");
        return false;
    }
    let mmap = match unsafe { MmapMut::map_mut(&file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[weight-cache] build disabled (mmap_mut failed: {e})");
            return false;
        }
    };
    // Data starts after a reserved page (header is rewritten on finish; index appended).
    let data_start = PAGE as u64;
    // L327 — say WHY it missed. A MISS costs ~20 minutes and 22 GB of rebuild, and the three
    // causes need completely different fixes: a stale blob (rebuild is correct), a flags
    // mismatch (set the env var and the existing cache is reused), or a truncated/absent file.
    // Without this line they are indistinguishable, and the flags case looks like corruption —
    // it cost a full A/B session to tell them apart by hand (the env `QUANT` is in the hash, so
    // passing only `--quant` on the CLI misses every time).
    if let Ok(f) = std::fs::File::open(&path) {
        use std::io::Read as _;
        let mut hdr = [0u8; 48];
        if (&f).read_exact(&mut hdr).is_ok() {
            let rd = |o: usize| u64::from_le_bytes(hdr[o..o + 8].try_into().unwrap());
            let (m, gl, gm, cfh) = (rd(0), rd(16), rd(24), rd(32));
            let ver = u32::from_le_bytes(hdr[8..12].try_into().unwrap());
            let why = if m != MAGIC {
                "not a cache file (bad magic)".to_string()
            } else if ver != LAYOUT_VERSION {
                format!("layout version {ver} != {LAYOUT_VERSION} (this build changed the format)")
            } else if gl != gguf_len {
                format!("GGUF length {gl} != {gguf_len} (the blob changed)")
            } else if gm != gguf_mtime_ns {
                format!(
                    "GGUF mtime {gm} != {gguf_mtime_ns} (the blob was rewritten or re-downloaded)"
                )
            } else if cfh != fh {
                format!(
                    "FLAGS differ: cache {cfh} vs this run {fh}. The hash covers the ENV vars \
                     QUANT / ARF_MSL_GEMV / ARF_MOE_MM_F32 / ARF_Q3K_FFN / ARF_G64_3BIT — a CLI flag \
                     alone does not set them. Export the same values the cache was built with \
                     and this file is reused as-is."
                )
            } else {
                "header matches — the index or data is unreadable (truncated?)".to_string()
            };
            eprintln!("[weight-cache] why: {why}");
        }
    }
    eprintln!(
        "[weight-cache] MISS {} — building (transcode→mmap, no-copy on this run too; promotes atomically on finish)",
        path.display()
    );
    CACHE.with(|c| {
        *c.borrow_mut() = Some(Mode::Build {
            path: build_path,
            final_path: path,
            file,
            mmap,
            index: Vec::new(),
            offset: data_start,
            gguf_len,
            gguf_mtime_ns,
            flags_hash: fh,
            swap_at_build_start: swap_used_mb(),
        })
    });
    true
}

/// Try to open + validate a sidecar cache for the given GGUF identity. Returns the mmap +
/// the parsed label→Entry index on a full match, else None (stale/corrupt/missing).
fn try_open_read(
    path: &Path,
    gguf_len: u64,
    gguf_mtime_ns: u64,
    fh: u64,
) -> Option<(Mmap, HashMap<String, Entry>)> {
    let mut f = File::open(path).ok()?;
    let mut hdr = [0u8; 64];
    f.read_exact(&mut hdr).ok()?;
    let rd_u64 = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap());
    let rd_u32 = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
    if rd_u64(&hdr[0..8]) != MAGIC {
        return None;
    }
    if rd_u32(&hdr[8..12]) != LAYOUT_VERSION {
        return None;
    }
    if rd_u64(&hdr[16..24]) != gguf_len
        || rd_u64(&hdr[24..32]) != gguf_mtime_ns
        || rd_u64(&hdr[32..40]) != fh
    {
        return None;
    }
    let n = rd_u64(&hdr[40..48]);
    let index_off = rd_u64(&hdr[48..56]);
    // Parse the index records (label_len:u32, label bytes, offset:u64, len:u64).
    f.seek(SeekFrom::Start(index_off)).ok()?;
    // Read the index ONLY — bounded by its entry count (label <= 1 KiB). `read_to_end` here read
    // the builder's reserved slack as well: 7.2 GB into RAM on every Qwen3.8 load, 12.0 GB on
    // qwen3-coder (2026-09-24), to parse a 345 KB index.
    let mut idx_buf = Vec::new();
    (&mut f)
        .take(n.saturating_mul(4 + 1024 + 16))
        .read_to_end(&mut idx_buf)
        .ok()?;
    let mut p = 0usize;
    let mut index = HashMap::with_capacity(n as usize);
    for _ in 0..n {
        if p + 4 > idx_buf.len() {
            return None;
        }
        let ll = rd_u32(&idx_buf[p..p + 4]) as usize;
        p += 4;
        if p + ll + 16 > idx_buf.len() {
            return None;
        }
        let label = String::from_utf8(idx_buf[p..p + ll].to_vec()).ok()?;
        p += ll;
        let offset = rd_u64(&idx_buf[p..p + 8]);
        p += 8;
        let len = rd_u64(&idx_buf[p..p + 8]);
        p += 8;
        index.insert(label, Entry { offset, len });
    }
    // TRIM THE BUILD RESERVATION. The builder sizes the file from an estimate and cannot
    // truncate at finish (its live mapping backs the running model's buffers), so every
    // sidecar carried slack past the index: 7.2 GB (Qwen3.8), 12.0 GB (qwen3-coder), 1.6 GB
    // (gemma-4-12b) — 20.8 GB of a 97%-full disk. Nothing maps the file yet, and the header
    // already records where the index is, so cutting after it changes no byte a reader uses.
    let keep = (index_off + p as u64).div_ceil(PAGE as u64) * PAGE as u64;
    if f.metadata().map_or(0, |m| m.len()) > keep {
        if let Ok(w) = std::fs::OpenOptions::new().write(true).open(path) {
            if w.set_len(keep).is_ok() {
                eprintln!(
                    "[weight-cache] trimmed the build reservation: sidecar is now {:.2} GB",
                    keep as f64 / 1e9
                );
            }
        }
    }
    // Map the whole file read-only; the wrapped MTLBuffers alias these pages for the
    // model's lifetime (the caller keeps the returned Arc<Mmap>).
    let mmap = unsafe { Mmap::map(&f) }.ok()?;
    Some((mmap, index))
}

/// True iff a cache session is active in READ (hit) mode — i.e. every region can be served
/// without transcoding. Callers use this to SKIP the expensive GGUF→q4ks transcode entirely
/// on a hit (the transcode result is already in the cache), the real WIN-1 speedup.
pub fn is_read_hit() -> bool {
    CACHE.with(|c| matches!(c.borrow().as_ref(), Some(Mode::Read { .. })))
}

/// Look up a cached region by `label` WITHOUT supplying bytes — the read-hit fast path that
/// lets a caller wrap the region no-copy and skip transcoding. Returns None in build mode
/// (the bytes don't exist yet → caller must transcode + `region_for`), on a miss, or off.
pub fn lookup(label: &str) -> Option<CacheRegion> {
    CACHE.with(|c| {
        let g = c.borrow();
        match g.as_ref()? {
            Mode::Read { mmap, index } => {
                let e = index.get(label)?;
                let (off, len) = (e.offset as usize, e.len as usize);
                if off + len > mmap.len() {
                    return None;
                }
                // SAFETY: mmap outlives every MTLBuffer (Arc kept on the model); region was
                // written page-aligned at build time.
                let ptr = unsafe { mmap.as_ptr().add(off) } as *mut std::ffi::c_void;
                Some(CacheRegion { ptr, len })
            }
            Mode::Build { .. } => None,
        }
    })
}

/// Look up or store a weight region by `label`, returning a page-aligned pointer into a
/// long-lived mapping that the caller wraps no-copy as an `MTLBuffer`.
///
/// - Read mode: `bytes` is ignored (the cached region is authoritative); returns the
///   mmap'd region for `label`, or None on a miss (caller falls back to transcode+copy).
/// - Build mode: copies `bytes` into the next page-aligned slot of the build mmap, records
///   the index, and returns THAT region (so the SAME bytes are wrapped no-copy now — WIN 2).
/// - Off: returns None (caller uses the legacy memcpy-into-fresh-MTLBuffer path).
///
/// SAFETY: the returned pointer is valid only while the cache mmap is alive. In read mode
/// the caller holds the `Arc<Mmap>` from `begin`; in build mode the caller holds the Arc
/// returned by `finish` (which re-maps the finished file read-only). Wrapping before
/// `finish` is fine because `finish` keeps the underlying file + pages stable.
pub fn region_for(label: &str, bytes: &[u8]) -> Option<CacheRegion> {
    CACHE.with(|c| {
        let mut g = c.borrow_mut();
        match g.as_mut()? {
            Mode::Read { mmap, index } => {
                let e = index.get(label)?;
                let off = e.offset as usize;
                let len = e.len as usize;
                if off + len > mmap.len() {
                    return None;
                }
                // The caller has just COMPUTED `bytes` — a cached region of another length is a
                // different buffer under the same label (a load-time table sized by context or
                // by code, not by the GGUF), and must not be served. MEASURED 2026-09-24: the
                // qwen35 RoPE tables were cached at the context of whichever run built the
                // sidecar (10,464 positions, 128 pairs) and every later load got THOSE — the
                // served model ran at perplexity 7.04 where a fresh load reads 6.06, and a
                // longer context read past the end of the table.
                if len != bytes.len() {
                    eprintln!(
                        "[weight-cache] {label}: cached {len} bytes, this load built {} — not \
                         serving the cached region",
                        bytes.len()
                    );
                    return None;
                }
                // SAFETY: `mmap` outlives every MTLBuffer (Arc kept on the model). The
                // region was written page-aligned at build time, so `ptr` is page-aligned.
                let ptr = unsafe { mmap.as_ptr().add(off) } as *mut std::ffi::c_void;
                Some(CacheRegion { ptr, len })
            }
            Mode::Build {
                mmap,
                index,
                offset,
                ..
            } => {
                // THE SAME LABEL WITH THE SAME BYTES AGAIN: the region already written. MEASURED
                // 2026-10-06 on the 27B: 960 labels were written twice — 2.26 GB of a 19.08 GB
                // sidecar — and a read served only the later copy of each (the index is a map).
                // A label written again with OTHER bytes still gets its own region, and the later
                // one wins on a read, as before.
                if let Some((_, e)) = index.iter().rev().find(|(l, _)| l == label) {
                    let (o, l) = (e.offset as usize, e.len as usize);
                    if l == bytes.len() && mmap[o..o + l] == *bytes {
                        let ptr = unsafe { mmap.as_ptr().add(o) } as *mut std::ffi::c_void;
                        return Some(CacheRegion { ptr, len: l });
                    }
                }
                let off = *offset as usize;
                let len = bytes.len();
                let end = off + len;
                if end > mmap.len() {
                    // Build file under-sized (bad est_bytes); abort caching for this region.
                    eprintln!(
                        "[weight-cache] build overflow at {label} (off={off} len={len} cap={}); region uncached",
                        mmap.len()
                    );
                    return None;
                }
                // Transcode result copied ONCE into the file-backed region; this is the
                // region we then wrap no-copy (replaces the old copy-into-fresh-MTLBuffer).
                mmap[off..end].copy_from_slice(bytes);
                index.push((
                    label.to_string(),
                    Entry {
                        offset: off as u64,
                        len: len as u64,
                    },
                ));
                let ptr = unsafe { mmap.as_ptr().add(off) } as *mut std::ffi::c_void;
                // Advance to the next page boundary so the next region is page-aligned.
                let padded = len.div_ceil(PAGE) * PAGE;
                *offset = (off + padded) as u64;
                Some(CacheRegion { ptr, len })
            }
        }
    })
}

/// Finalize the cache session and return the long-lived keep-alive the caller stores on the
/// `GpuModel` (the no-copy MTLBuffers alias its pages). On a build run: append the index +
/// header into the build mapping, flush it to disk (so the sidecar is reusable next load),
/// then hand back the SAME `MmapMut` as `CacheKeep::Build` — the exact pages the buffers
/// were wrapped over, so there is NO unmap window and NO pointer invalidation. On a read
/// run: returns the `Mmap` as `CacheKeep::Read`. Off: returns None (no buffers were wrapped
/// no-copy, so nothing needs keeping alive).
pub fn finish() -> Option<CacheKeep> {
    CACHE.with(|c| {
        let mode = c.borrow_mut().take()?;
        match mode {
            Mode::Read { mmap, .. } => Some(CacheKeep::Read(mmap)),
            Mode::Build {
                path,
                final_path,
                file,
                mut mmap,
                index,
                offset,
                gguf_len,
                gguf_mtime_ns,
                flags_hash,
                swap_at_build_start,
            } => {
                // Index starts at the current (page-aligned) write offset.
                let index_off = offset;
                // Serialize the index after the data blob.
                let mut idx_bytes: Vec<u8> = Vec::new();
                for (label, e) in &index {
                    let lb = label.as_bytes();
                    idx_bytes.extend_from_slice(&(lb.len() as u32).to_le_bytes());
                    idx_bytes.extend_from_slice(lb);
                    idx_bytes.extend_from_slice(&e.offset.to_le_bytes());
                    idx_bytes.extend_from_slice(&e.len.to_le_bytes());
                }
                let idx_end = index_off as usize + idx_bytes.len();
                let reusable = if idx_end <= mmap.len() {
                    mmap[index_off as usize..idx_end].copy_from_slice(&idx_bytes);
                    let mut hdr = [0u8; 64];
                    hdr[0..8].copy_from_slice(&MAGIC.to_le_bytes());
                    hdr[8..12].copy_from_slice(&LAYOUT_VERSION.to_le_bytes());
                    hdr[16..24].copy_from_slice(&gguf_len.to_le_bytes());
                    hdr[24..32].copy_from_slice(&gguf_mtime_ns.to_le_bytes());
                    hdr[32..40].copy_from_slice(&flags_hash.to_le_bytes());
                    hdr[40..48].copy_from_slice(&(index.len() as u64).to_le_bytes());
                    hdr[48..56].copy_from_slice(&index_off.to_le_bytes());
                    mmap[0..64].copy_from_slice(&hdr);
                    // Flush ONLY the written prefix [0, idx_end) so the sidecar is valid next
                    // load — flushing the whole 40+ GB reserved (mostly-sparse) mapping would
                    // be a needless msync over clean/unwritten pages. Pages stay mapped (we
                    // keep `mmap` as the keep-alive), so flushing does not unmap.
                    let _ = mmap.flush_range(0, idx_end);
                    true
                } else {
                    eprintln!("[weight-cache] finish: index would overflow reserved file; sidecar NOT finalized (this run is correct, next load re-builds)");
                    false
                };
                // The file is over-sized by our reservation. We can't truncate while the
                // `MmapMut` is live (it maps the reserved length), and we must NOT drop the
                // map (the MTLBuffers alias it). So leave the file at its reserved size on
                // disk — the header records the real index offset, so a reader ignores the
                // slack. Keep the rw `file` handle dropped (mapping owns the pages).
                drop(file);
                // L109 SWAP GUARD — a sidecar built while the box is paging can be silently
                // WRONG, and because the cache key is only (gguf len, mtime) + flags_hash, a
                // corrupt-but-well-formed sidecar HITs forever and poisons every later load.
                // Measured 2026-08-10: the sidecar written during the L108 swap-death session
                // made EVERY binary (including pristine HEAD) emit `!!!!` for any prompt; no
                // perf number could see it. Refuse to promote when swap grew materially during
                // the build — this run stays correct (buffers alias the `.building` pages), we
                // just decline to hand the next load a suspect cache. Override:
                // ARF_CACHE_NO_SWAP_GUARD=1.
                let reusable = if reusable
                    && std::env::var_os("ARF_CACHE_NO_SWAP_GUARD").is_none()
                {
                    let grew_mb = swap_used_mb().saturating_sub(swap_at_build_start);
                    if grew_mb > 512 {
                        eprintln!(
                            "[weight-cache] ⚠️  NOT promoting: swap grew {grew_mb} MB during the \
build — a cache built under memory pressure can be silently corrupt (L109). This run is \
correct; next load rebuilds. Free memory (or power-cycle) and rebuild, or set \
ARF_CACHE_NO_SWAP_GUARD=1 to promote anyway."
                        );
                        false
                    } else {
                        true
                    }
                } else {
                    reusable
                };
                if reusable {
                    // ATOMIC PROMOTION: the `.building` file now has a valid, flushed header, so
                    // rename it over the final cache path. Same-filesystem rename is atomic (a
                    // reader sees either the old cache or the new one, never a half) AND leaves the
                    // inode unchanged — so `mmap` (aliased by the live no-copy MTLBuffers) stays
                    // valid across the promotion (it references the inode, not the name). If the
                    // rename fails, THIS run is still correct (buffers alias the `.building` pages);
                    // only the NEXT load's HIT is lost (it'll rebuild), so this is non-fatal.
                    match std::fs::rename(&path, &final_path) {
                        Ok(()) => eprintln!(
                            "[weight-cache] BUILT {} ({} regions, {:.2} GB data) — promoted atomically, reusable next load",
                            final_path.display(),
                            index.len(),
                            index_off as f64 / 1e9
                        ),
                        Err(e) => eprintln!(
                            "[weight-cache] BUILT {} ({} regions) but promote (rename → {}) failed: {e} — this run correct, next load re-builds",
                            path.display(),
                            index.len(),
                            final_path.display()
                        ),
                    }
                }
                // Hand back the SAME mapping the no-copy buffers were wrapped over.
                Some(CacheKeep::Build(mmap))
            }
        }
    })
}

/// Clear any active cache session without finalizing (error path). Drops the build mmap;
/// the partial file is left and will be rejected on next load (header not written).
pub fn abort() {
    CACHE.with(|c| {
        *c.borrow_mut() = None;
    });
}

#[cfg(test)]
mod duplicate_region_tests {
    use super::*;

    /// A build writes a label's bytes once: the same label with the same bytes again is the
    /// region already written; with other bytes it is a new region.
    #[test]
    fn a_label_written_twice_with_the_same_bytes_is_one_region() {
        if std::env::var_os("ARF_NO_WEIGHT_CACHE").is_some() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("arf-weight-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = dir.join("model.gguf");
        std::fs::write(&gguf, b"not a model").unwrap();
        assert!(begin(&gguf, 16 * PAGE as u64));
        let next = || {
            CACHE.with(|c| match c.borrow().as_ref() {
                Some(Mode::Build { offset, .. }) => *offset,
                _ => panic!("a build session"),
            })
        };
        let a = region_for("w", &[7u8; 100]).unwrap();
        let after_one = next();
        let again = region_for("w", &[7u8; 100]).unwrap();
        assert_eq!((again.ptr, again.len), (a.ptr, a.len));
        assert_eq!(next(), after_one, "nothing written the second time");
        let other = region_for("w", &[9u8; 100]).unwrap();
        assert_ne!(
            other.ptr, a.ptr,
            "other bytes under the label: a new region"
        );
        assert!(next() > after_one);
        abort();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
mod l109_swap_guard_tests {
    /// The guard is only as good as this parse: a wrong number silently disables it (0) or
    /// blocks every promotion. Pin the real `sysctl -n vm.swapusage` shape.
    #[test]
    fn parses_real_sysctl_swapusage_shape() {
        // Exactly what `sysctl -n vm.swapusage` prints on macOS 26.
        let s = "total = 7168.00M  used = 5603.81M  free = 1564.19M  (encrypted)";
        let parsed = s
            .split("used =")
            .nth(1)
            .map(|rest| {
                rest.trim()
                    .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.')
                    .split('M')
                    .next()
                    .and_then(|v| v.trim().parse::<f64>().ok())
                    .map(|v| v as u64)
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        assert_eq!(parsed, 5603, "must read the USED field, not total/free");
    }

    #[test]
    fn zero_swap_box_parses_as_zero_not_garbage() {
        let s = "total = 0.00M  used = 0.00M  free = 0.00M";
        let parsed = s
            .split("used =")
            .nth(1)
            .map(|rest| {
                rest.trim()
                    .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.')
                    .split('M')
                    .next()
                    .and_then(|v| v.trim().parse::<f64>().ok())
                    .map(|v| v as u64)
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        assert_eq!(parsed, 0);
    }
}
