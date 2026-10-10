//! Resident ratatui dashboard for `arf-serve`: reads the shared `Metrics`
//! atomics each tick into a small history ring and renders sparklines + gauges.
//!
//! The dashboard runs on a DEDICATED OS thread (never tokio, never the actor).
//! It only ever reads the lock-free atomics in [`crate::metrics::Metrics`], so it
//! cannot perturb the hot decode path. It is spawned from `main` only when stdout
//! is a TTY and `--no-tty` was not passed; under systemd/Docker the plain
//! `eprintln!` logging is used instead and this module is never entered.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline},
    Frame,
};

/// A fixed-capacity ring of recent samples for one sparkline series.
pub struct Ring {
    buf: Vec<u64>,
    cap: usize,
    head: usize,
    len: usize,
}

impl Ring {
    pub fn new(cap: usize) -> Self {
        // `push`/`series`/`last` index modulo `cap`; a zero-cap ring would
        // divide by zero on first use. Callers use a fixed positive cap.
        debug_assert!(cap > 0, "Ring capacity must be > 0");
        Self {
            buf: vec![0; cap],
            cap,
            head: 0,
            len: 0,
        }
    }
    pub fn push(&mut self, v: u64) {
        self.buf[self.head] = v;
        self.head = (self.head + 1) % self.cap;
        self.len = (self.len + 1).min(self.cap);
    }
    /// Oldest→newest slice for rendering.
    pub fn series(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let idx = (self.head + self.cap - self.len + i) % self.cap;
            out.push(self.buf[idx]);
        }
        out
    }
    pub fn last(&self) -> u64 {
        if self.len == 0 {
            0
        } else {
            self.buf[(self.head + self.cap - 1) % self.cap]
        }
    }
}

/// tok/s between two cumulative token counts over `dt_secs`.
pub fn rate(prev: u64, now: u64, dt_secs: f64) -> u64 {
    if dt_secs <= 0.0 {
        return 0;
    }
    (now.saturating_sub(prev) as f64 / dt_secs).round() as u64
}

/// Prefix-cache hit ratio as 0–100 percent.
pub fn prefix_hit_pct(reused: u64, total_prompt: u64) -> u16 {
    if total_prompt == 0 {
        return 0;
    }
    ((reused as f64 / total_prompt as f64) * 100.0)
        .round()
        .min(100.0) as u16
}

/// Everything the dashboard needs for one frame.
pub struct DashSnapshot {
    pub model_id: String,
    pub endpoint: String,
    pub uptime: String,
    pub tok_series: Vec<u64>,
    pub tok_now: u64,
    pub tok_peak: u64,
    pub req_series: Vec<u64>,
    pub req_active: usize,
    pub req_served: u64,
    pub kv_used: usize,
    pub kv_total: usize,
    pub batch_running: usize,
    pub batch_max: usize,
    pub prefix_hit: u16,
    /// GPU-frame breakdown for the kernel panel: `(name, pct_of_frame, opt_roofline_pct)`,
    /// sorted by descending pct. Empty when profiling data isn't wired (the panel then
    /// shows a hint). Populated from `gpu::profile::snapshot()` on the live path.
    pub kernels: Vec<(String, u16, Option<u16>)>,
}

// Game-HUD neon palette.
const ACCENT: Color = Color::Rgb(70, 179, 255); // electric blue
const CYAN: Color = Color::Rgb(64, 245, 230); // neon cyan
const VIOLET: Color = Color::Rgb(164, 123, 255); // violet
const MAGENTA: Color = Color::Rgb(255, 92, 200); // hot magenta
const GOOD: Color = Color::Rgb(57, 230, 160); // green
const WARN: Color = Color::Rgb(255, 207, 92); // amber
#[allow(dead_code)]
const HOT: Color = Color::Rgb(255, 93, 108); // red (gradient endpoint, see heat())
const DIM: Color = Color::Rgb(95, 116, 136); // muted
const FG: Color = Color::Rgb(207, 232, 255); // foreground
const BORDER: Color = Color::Rgb(38, 58, 78); // panel border (brighter than before)

/// Smooth green→amber→red gradient over a 0..1 load ratio (no hard steps).
fn heat(r: f64) -> Color {
    let r = r.clamp(0.0, 1.0);
    // green (57,230,160) → amber (255,207,92) → red (255,93,108)
    let (a, b, t) = if r < 0.6 {
        ((57.0, 230.0, 160.0), (255.0, 207.0, 92.0), r / 0.6)
    } else {
        ((255.0, 207.0, 92.0), (255.0, 93.0, 108.0), (r - 0.6) / 0.4)
    };
    let lerp = |x: f64, y: f64| (x + (y - x) * t) as u8;
    Color::Rgb(lerp(a.0, b.0), lerp(a.1, b.1), lerp(a.2, b.2))
}

/// Throughput tier color — the hero number shifts hue as the swarm scales:
/// dim → blue → cyan → green → magenta as aggregate tok/s climbs. The visual
/// reward for concurrency (the advantage), straight on the biggest number on screen.
fn tput_color(tok: u64) -> Color {
    match tok {
        0..=120 => ACCENT,
        121..=280 => CYAN,
        281..=480 => GOOD,
        _ => MAGENTA,
    }
}

/// A vivid per-kernel color (stable by name) for the GPU-frame breakdown bars.
fn kernel_color(name: &str) -> Color {
    match name {
        "matmul" => ACCENT,
        "attention" | "attention_batched" => VIOLET,
        "swiglu" => MAGENTA,
        n if n.contains("norm") => CYAN,
        n if n.contains("rope") => Color::Rgb(120, 200, 120),
        n if n.contains("kv") => Color::Rgb(90, 140, 200),
        _ => DIM,
    }
}

/// A 5-row block-glyph rendering of one number — the big "hero" tok/s readout.
fn big_number(n: u64) -> Vec<String> {
    // 3-wide, 5-tall block digits.
    const D: [[&str; 5]; 10] = [
        ["███", "█ █", "█ █", "█ █", "███"], // 0
        [" █ ", "██ ", " █ ", " █ ", "███"], // 1
        ["███", "  █", "███", "█  ", "███"], // 2
        ["███", "  █", "███", "  █", "███"], // 3
        ["█ █", "█ █", "███", "  █", "  █"], // 4
        ["███", "█  ", "███", "  █", "███"], // 5
        ["███", "█  ", "███", "█ █", "███"], // 6
        ["███", "  █", "  █", "  █", "  █"], // 7
        ["███", "█ █", "███", "█ █", "███"], // 8
        ["███", "█ █", "███", "  █", "███"], // 9
    ];
    let s = n.to_string();
    (0..5)
        .map(|row| {
            s.chars()
                .map(|c| D[c.to_digit(10).unwrap_or(0) as usize][row])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Draw one dashboard frame — the game-HUD.
pub fn draw(f: &mut Frame, s: &DashSnapshot) {
    let area = f.area();
    // Outer rows: title (1) · hero throughput (8) · gauges+sparkline (rest) · footer (1).
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(9),
            Constraint::Min(6),
            Constraint::Length(1),
        ])
        .split(area);

    // ---- title bar ----
    let title = Line::from(vec![
        // The name is split so its last letter takes the accent colour. (Until 2026-10-07 it was
        // the project's old name, split the same way, which is why no search of the source found
        // it; the test below reads the rendered bar.)
        Span::styled("  Ar", Style::default().fg(FG).add_modifier(Modifier::BOLD)),
        Span::styled(
            "f",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ● LIVE  ", Style::default().fg(GOOD)),
        Span::styled(format!("{}  ", s.model_id), Style::default().fg(FG)),
        Span::styled(format!("{}  ", s.endpoint), Style::default().fg(DIM)),
        Span::styled(format!("up {}", s.uptime), Style::default().fg(DIM)),
    ]);
    f.render_widget(Paragraph::new(title), rows[0]);

    // ---- hero: big tok/s + throughput sparkline side by side ----
    let hero = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(38), Constraint::Min(20)])
        .split(rows[1]);

    let hero_col = tput_color(s.tok_now);
    let mut hero_lines = vec![Line::from(Span::styled(
        " AGGREGATE THROUGHPUT",
        Style::default().fg(DIM).add_modifier(Modifier::BOLD),
    ))];
    for l in big_number(s.tok_now) {
        hero_lines.push(Line::from(Span::styled(
            format!(" {l}"),
            Style::default().fg(hero_col).add_modifier(Modifier::BOLD),
        )));
    }
    hero_lines.push(Line::from(vec![
        Span::styled(" tok/s", Style::default().fg(FG)),
        Span::styled(
            format!("   peak {}   served {}", s.tok_peak, s.req_served),
            Style::default().fg(DIM),
        ),
    ]));
    f.render_widget(Paragraph::new(hero_lines).block(panel(" hero ")), hero[0]);

    let tput = Sparkline::default()
        .block(panel(format!(
            " throughput · {} active agents ",
            s.req_active
        )))
        .style(Style::default().fg(ACCENT))
        .data(&s.tok_series);
    f.render_widget(tput, hero[1]);

    // ---- middle: three gauges (KV · batch · prefix) stacked with a req sparkline ----
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(rows[2]);

    let gauges = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(mid[0]);

    let kv_ratio = ratio(s.kv_used, s.kv_total);
    f.render_widget(
        Gauge::default()
            .block(panel(" KV cache "))
            .gauge_style(Style::default().fg(heat(kv_ratio)))
            .ratio(kv_ratio)
            .label(format!(
                "{}/{} blk · {:.0}%",
                s.kv_used,
                s.kv_total,
                kv_ratio * 100.0
            )),
        gauges[0],
    );
    let b_ratio = ratio(s.batch_running, s.batch_max);
    f.render_widget(
        Gauge::default()
            .block(panel(" batch occupancy "))
            .gauge_style(Style::default().fg(heat(b_ratio)))
            .ratio(b_ratio)
            .label(format!("{}/{} slots", s.batch_running, s.batch_max)),
        gauges[1],
    );
    // batch "slot" glyphs — one ◧ per running seq, ◍ per free slot (the swarm at a glance).
    slot_strip(f, gauges[2], s.batch_running, s.batch_max);

    // right: prefix gauge + GPU-frame kernel breakdown (vivid colored bars)
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(mid[1]);
    let p_ratio = (s.prefix_hit as f64 / 100.0).min(1.0);
    f.render_widget(
        Gauge::default()
            .block(panel(" prefix cache hit "))
            .gauge_style(Style::default().fg(VIOLET))
            .ratio(p_ratio)
            .label(format!("{}%", s.prefix_hit)),
        right[0],
    );
    kernel_panel(f, right[1], &s.kernels);

    // ---- footer hint ----
    let foot = Line::from(vec![
        Span::styled(
            "  continuous batching · paged KV · prefix cache · coop GEMM",
            Style::default().fg(DIM),
        ),
        Span::styled("        press ", Style::default().fg(DIM)),
        Span::styled("q", Style::default().fg(WARN).add_modifier(Modifier::BOLD)),
        Span::styled(" / ", Style::default().fg(DIM)),
        Span::styled(
            "Ctrl-C",
            Style::default().fg(WARN).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" to quit", Style::default().fg(DIM)),
    ]);
    f.render_widget(Paragraph::new(foot).alignment(Alignment::Left), rows[3]);
}

/// GPU-frame breakdown panel: per-kernel colored bars (the "where the time goes" view).
fn kernel_panel(f: &mut Frame, area: Rect, kernels: &[(String, u16, Option<u16>)]) {
    let inner_w = area.width.saturating_sub(2) as usize;
    let bar_w = inner_w.saturating_sub(20).max(4); // leave room for name + pct
    let mut lines: Vec<Line> = Vec::new();
    if kernels.is_empty() {
        lines.push(Line::from(Span::styled(
            "  run with --profile for the GPU frame",
            Style::default().fg(DIM),
        )));
    }
    for (name, pct, roof) in kernels.iter().take(area.height.saturating_sub(2) as usize) {
        let filled = ((*pct as usize * bar_w) / 100).min(bar_w);
        let col = kernel_color(name);
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_w - filled);
        let roof_s = roof.map(|r| format!(" {r:>2}%roof")).unwrap_or_default();
        lines.push(Line::from(vec![
            Span::styled(format!(" {:<10}", trunc(name, 10)), Style::default().fg(FG)),
            Span::styled(bar, Style::default().fg(col)),
            Span::styled(
                format!(" {pct:>2}%"),
                Style::default().fg(FG).add_modifier(Modifier::BOLD),
            ),
            Span::styled(roof_s, Style::default().fg(DIM)),
        ]));
    }
    f.render_widget(
        Paragraph::new(lines).block(panel(" GPU frame · where time goes ")),
        area,
    );
}

fn trunc(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n - 1])
    }
}

/// A bordered HUD panel block with a dim title.
fn panel(title: impl Into<String>) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .title(Span::styled(title.into(), Style::default().fg(DIM)))
}

fn ratio(used: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (used as f64 / total as f64).min(1.0)
    }
}

/// Render `running` filled slots + free slots as glyphs — the batch swarm at a glance.
fn slot_strip(f: &mut Frame, area: Rect, running: usize, max: usize) {
    if max == 0 || area.height == 0 {
        return;
    }
    let mut spans = vec![Span::styled("  ", Style::default())];
    for i in 0..max {
        let (g, c) = if i < running {
            ("◧ ", GOOD)
        } else {
            ("◌ ", DIM)
        };
        spans.push(Span::styled(g, Style::default().fg(c)));
    }
    f.render_widget(
        Paragraph::new(Line::from(spans)).block(panel(" slots ")),
        area,
    );
}

/// RAII guard: enters raw mode + alternate screen on `enter()`, restores both on
/// drop.
///
/// A crate-local equivalent of arf-cli's `TermGuard` (that one is not on this
/// crate's dependency path). The guard restores the terminal on every normal exit
/// path via `Drop`. Note: `Drop` does NOT run on a SIGINT/Ctrl-C (the process is
/// torn down without unwinding), so a Ctrl-C may leave the terminal in the
/// alternate screen — acceptable for v1; a SIGINT handler is a future nicety.
pub struct TermGuard {
    active: bool,
    real: bool,
}

/// LOG LINES UNDER THE DASHBOARD (#57): while the dashboard owns the terminal, the server's
/// `eprintln!` lines (`[start] request ...`, `[done] ...`) were printed over its panels. This
/// points stderr (fd 2) at a log file for as long as it lives and points it back when dropped —
/// declared before [`TermGuard`], so the terminal is restored first and later lines (a stop, a
/// shutdown) appear in it again. `None` when stderr is not a terminal (nothing to protect) or the
/// file cannot be opened (the lines stay where they were).
#[cfg(unix)]
pub struct StderrToFile {
    saved: std::os::raw::c_int,
}

#[cfg(unix)]
extern "C" {
    fn dup(fd: std::os::raw::c_int) -> std::os::raw::c_int;
    fn dup2(src: std::os::raw::c_int, dst: std::os::raw::c_int) -> std::os::raw::c_int;
    fn close(fd: std::os::raw::c_int) -> std::os::raw::c_int;
}

#[cfg(unix)]
impl StderrToFile {
    pub fn open(path: &std::path::Path) -> Option<Self> {
        use std::io::IsTerminal as _;
        use std::os::fd::AsRawFd as _;
        if !std::io::stderr().is_terminal() {
            return None;
        }
        std::fs::create_dir_all(path.parent()?).ok()?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()?;
        // SAFETY: plain descriptor calls on fd 2 and a file this function owns; `saved` is a
        // fresh descriptor closed in `drop`.
        let saved = unsafe { dup(2) };
        if saved < 0 {
            return None;
        }
        if unsafe { dup2(file.as_raw_fd(), 2) } < 0 {
            unsafe { close(saved) };
            return None;
        }
        Some(StderrToFile { saved })
    }
}

#[cfg(unix)]
impl Drop for StderrToFile {
    fn drop(&mut self) {
        // SAFETY: `saved` is the descriptor `open` duplicated from the original stderr.
        unsafe {
            dup2(self.saved, 2);
            close(self.saved);
        }
    }
}

impl TermGuard {
    /// Enter raw mode + alternate screen. Returns an inert guard on failure so the
    /// caller can fall back without unwinding.
    pub fn enter() -> Self {
        use crossterm::{
            execute,
            terminal::{enable_raw_mode, EnterAlternateScreen},
        };
        use std::io::stdout;
        // Once raw mode is on, mark `real` so a later alt-screen failure still gets
        // cleaned up on drop (restore disables raw mode).
        if enable_raw_mode().is_err() {
            return TermGuard {
                active: false,
                real: false,
            };
        }
        let alt_ok = execute!(stdout(), EnterAlternateScreen).is_ok();
        TermGuard {
            active: alt_ok,
            real: true,
        }
    }

    pub fn active(&self) -> bool {
        self.active
    }

    fn restore(&mut self) {
        if self.real {
            use crossterm::{
                execute,
                terminal::{disable_raw_mode, LeaveAlternateScreen},
            };
            use std::io::{stdout, Write};
            let _ = execute!(stdout(), LeaveAlternateScreen);
            let _ = disable_raw_mode();
            let _ = stdout().flush();
        }
        self.active = false;
        self.real = false;
    }
}

impl Drop for TermGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ring_keeps_last_n_in_order() {
        let mut r = Ring::new(3);
        for v in [1, 2, 3, 4] {
            r.push(v);
        }
        assert_eq!(r.series(), vec![2, 3, 4]);
        assert_eq!(r.last(), 4);
    }
    #[test]
    fn rate_and_prefix_pct() {
        assert_eq!(rate(100, 220, 2.0), 60);
        assert_eq!(rate(5, 5, 1.0), 0);
        assert_eq!(prefix_hit_pct(37, 100), 37);
        assert_eq!(prefix_hit_pct(0, 0), 0);
    }
    #[test]
    fn draws_into_test_backend() {
        use ratatui::{backend::TestBackend, Terminal};
        let backend = TestBackend::new(80, 16);
        let mut term = Terminal::new(backend).unwrap();
        let snap = DashSnapshot {
            model_id: "gemma-3-4b-it".into(),
            endpoint: "127.0.0.1:8080".into(),
            uptime: "4m".into(),
            tok_series: vec![1, 2, 3, 4, 5],
            tok_now: 142,
            tok_peak: 188,
            req_series: vec![1, 1, 2, 3],
            req_active: 3,
            req_served: 1204,
            kv_used: 5,
            kv_total: 8,
            batch_running: 5,
            batch_max: 8,
            prefix_hit: 37,
            kernels: vec![],
        };
        term.draw(|f| draw(f, &snap)).unwrap();
        let buf = term.backend().buffer().clone();
        let text: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("LIVE"));
        // The name as rendered: a search of the source cannot see across the two spans.
        assert!(text.contains("  Arf  "), "the title names Arf");
        // The old name is assembled here, so this line is not itself a hit for the leak scan.
        let old_name = ["PRI", "MER"].concat();
        assert!(
            !text.to_ascii_uppercase().contains(&old_name),
            "and not the old name"
        );
        assert!(text.contains("gemma-3-4b-it"));
        assert!(text.contains("KV cache"));
        assert!(text.contains("batch"));
    }
}
