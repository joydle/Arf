//! AFTER A GPU DRIVER PANIC (2026-10-09, #60): serve without changing GPU memory mappings.
//!
//! macOS panicked twice in its GPU driver on one M4 Max (`IOGPUFamily`, "Kernel data abort", the
//! same place both times) with `arf-serve` the panicked task; the second time 8 s after the
//! grow-on-use KV cache had grown. A kernel panic stops every process at once — no handler of
//! ours can run — so what a server can do is learn from one: macOS keeps the report in
//! `/Library/Logs/DiagnosticReports`, readable by every user. At start, a report from the last
//! [`SAFE_DAYS`] days naming `arf-serve` in the GPU driver turns on SAFE MEMORY: the grow-on-use KV
//! cache maps its first `PREMAP_KV_SLOTS` (98,304) slots and the recurrent bank its first rows at the
//! first step — the load-time warm-up, GPU idle (`ARF_SPARSE_PREMAP`) — so an agent's usual working
//! set (a ~47K-token first message beside a ~31K-token safety check) never changes a mapping while
//! the server serves; past it, growth still happens, behind `idle_for_mapping`. The context and the
//! block pool stay what they would be. The cost: ~3.3 GB of KV (the 27B) and ~0.45 GB of bank
//! resident from the start. `ARF_SPARSE_AFTER_PANIC=1` keeps grow-on-use memory regardless.
//!
//! MEASURED OUT 2026-10-09 (that day's first release test): safe memory at a 65,536-token context. Its block
//! pool (4,096 blocks) held one of the agent's two prompts, not both; each evicted the other, every
//! request read from 0 ("cached 0" on an identical repeat: 569 s against 30), and the release
//! failed.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// How long after such a panic a server keeps its working set mapped at load.
pub const SAFE_DAYS: u64 = 14;

const REPORTS: &str = "/Library/Logs/DiagnosticReports";

/// The most recent panic report in `dir` modified after `since` that names `arf-serve` as the
/// panicked task in Apple's GPU driver.
pub fn find(dir: &Path, since: SystemTime) -> Option<(SystemTime, PathBuf)> {
    let mut found: Option<(SystemTime, PathBuf)> = None;
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !(name.starts_with("panic") && name.ends_with(".panic")) {
            continue;
        }
        let Some(when) = e.metadata().ok().and_then(|m| m.modified().ok()) else {
            continue;
        };
        if when < since || found.as_ref().is_some_and(|(t, _)| *t >= when) {
            continue;
        }
        // The panicked task and the backtrace's extensions sit in the report's first part.
        let mut head = vec![0u8; 256 << 10];
        let n = std::fs::File::open(&p)
            .and_then(|mut f| std::io::Read::read(&mut f, &mut head))
            .unwrap_or(0);
        let text = String::from_utf8_lossy(&head[..n]);
        let ours = text
            .split("Panicked task")
            .nth(1)
            .and_then(|rest| rest.split(['\n', '\\']).next())
            .is_some_and(|line| line.contains("arf-serve"));
        if ours && text.contains("IOGPUFamily") {
            found = Some((when, p));
        }
    }
    found
}

/// Turn on safe memory when a recent GPU panic named this server (see the module doc). Call before
/// anything reads `ARF_NO_SPARSE_KV` / `ARF_NO_SPARSE_GDN`. Returns the line to log, if it did.
pub fn apply() -> Option<String> {
    if std::env::var_os("ARF_SPARSE_AFTER_PANIC").is_some_and(|v| v != "0") {
        return None;
    }
    let since = SystemTime::now().checked_sub(Duration::from_secs(SAFE_DAYS * 86_400))?;
    let (when, path) = find(Path::new(REPORTS), since)?;
    // Grow-on-use buffers: the KV pool mapped whole at the first step (the load-time warm-up) and
    // never after, the recurrent bank's first rows likewise (`gdn_ensure_rows`). NOT a smaller
    // bank (`ARF_GDN_ROWS=5` decoded at 6.9 tok/s against ~30, 2026-10-09). NOT the committed pool
    // (`ARF_NO_SPARSE_KV`): measured 2026-10-09 on the 27B, a committed 131K pool decoded at 1.4
    // tok/s and read prompts at ~85 tok/s — its buffers did not stay resident.
    for (k, v) in [("ARF_SPARSE_PREMAP", "1")] {
        if std::env::var_os(k).is_none() {
            std::env::set_var(k, v);
        }
    }
    let days_left = SAFE_DAYS.saturating_sub(
        SystemTime::now()
            .duration_since(when)
            .map_or(0, |d| d.as_secs() / 86_400),
    );
    Some(format!(
        "[safety] after a GPU driver crash report naming arf-serve ({}): this server maps its first \
         98,304 KV slots and the recurrent bank's first rows at load, so an agent's usual working set \
         changes no GPU memory mapping while it serves (#60), for {days_left} more day(s); \
         ARF_SPARSE_AFTER_PANIC=1 keeps plain grow-on-use memory",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("a panic report")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(dir: &Path, name: &str, task: &str, kext: &str) {
        let body = format!(
            "{{\"bug_type\":\"210\"}}\n{{\"panicString\":\"panic(cpu 9): Kernel data abort.\\nPanicked task 0xfffffe: 49353 pages, 20 threads: pid 95098: {task}\\nPanicked thread: 0x1\\nKernel Extensions in backtrace:\\n com.apple.iokit.{kext}(130.13)\\n\"}}"
        );
        std::fs::write(dir.join(name), body).unwrap();
    }

    #[test]
    fn only_a_gpu_panic_of_this_server_counts() {
        let dir = std::env::temp_dir().join(format!("arf-gpu-panic-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let long_ago = SystemTime::now() - Duration::from_secs(3600);
        report(&dir, "panic-full-a.panic", "kernel_task", "IOGPUFamily");
        report(&dir, "panic-full-b.panic", "arf-serve", "IOUSBFamily");
        report(&dir, "other.ips", "arf-serve", "IOGPUFamily");
        assert_eq!(
            find(&dir, long_ago),
            None,
            "another task, another driver, not a panic report"
        );
        report(&dir, "panic-full-c.panic", "arf-serve", "IOGPUFamily");
        let (_, p) = find(&dir, long_ago).expect("this server in the GPU driver");
        assert!(p.ends_with("panic-full-c.panic"));
        assert_eq!(
            find(&dir, SystemTime::now() + Duration::from_secs(60)),
            None,
            "older than the window"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
