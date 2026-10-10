//! Single source of CLI styling: color-mode resolution + an accent palette.
//!
//! Nothing else in the CLI hard-codes an escape code — callers ask `Ui` how to
//! paint. When color is off (piped output, `NO_COLOR`, `--no-color`, dumb term)
//! every paint function returns the text unchanged, so scripts get clean output.

/// Whether styled ANSI output is enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    On,
    Off,
}

/// Resolve the effective color mode.
///
/// Precedence: an explicit `--no-color` (or `NO_COLOR` set to any value) forces
/// Off; `TERM=dumb` forces Off; otherwise On iff `is_tty` is true.
pub fn resolve(
    no_color_flag: bool,
    no_color_env: Option<&str>,
    term: Option<&str>,
    is_tty: bool,
) -> ColorMode {
    if no_color_flag || no_color_env.is_some() || term == Some("dumb") || !is_tty {
        ColorMode::Off
    } else {
        ColorMode::On
    }
}

/// A cheap, copyable styling handle threaded into renderers.
#[derive(Clone, Copy, Debug)]
pub struct Ui {
    mode: ColorMode,
}

/// The one accent (cyan-green) and the supporting greys, as truecolor SGR codes.
const ACCENT: &str = "\x1b[38;2;55;224;176m"; // #37e0b0
const DIM: &str = "\x1b[38;2;124;136;150m"; // muted grey
const FAINT: &str = "\x1b[38;2;77;87;101m"; // fainter grey
const AMBER: &str = "\x1b[38;2;224;178;90m";
const RED: &str = "\x1b[38;2;224;106;106m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

impl Ui {
    /// Build from an already-resolved mode.
    pub fn new(mode: ColorMode) -> Self {
        Self { mode }
    }

    /// Resolve from the live environment (the normal entry point).
    pub fn from_env(no_color_flag: bool) -> Self {
        use std::io::IsTerminal;
        let no_color_env = std::env::var("NO_COLOR").ok();
        let term = std::env::var("TERM").ok();
        let is_tty = std::io::stdout().is_terminal();
        Self::new(resolve(
            no_color_flag,
            no_color_env.as_deref(),
            term.as_deref(),
            is_tty,
        ))
    }

    /// True when styled output is enabled.
    pub fn color(self) -> bool {
        self.mode == ColorMode::On
    }

    fn wrap(self, code: &str, s: &str) -> String {
        if self.color() {
            format!("{code}{s}{RESET}")
        } else {
            s.to_string()
        }
    }

    pub fn accent(self, s: &str) -> String {
        self.wrap(ACCENT, s)
    }
    pub fn dim(self, s: &str) -> String {
        self.wrap(DIM, s)
    }
    pub fn faint(self, s: &str) -> String {
        self.wrap(FAINT, s)
    }
    pub fn amber(self, s: &str) -> String {
        self.wrap(AMBER, s)
    }
    pub fn red(self, s: &str) -> String {
        self.wrap(RED, s)
    }
    pub fn bold(self, s: &str) -> String {
        self.wrap(BOLD, s)
    }
}

/// Human-readable byte size, e.g. `2.0 GB`, `33.4 MB`, `96 B`. The single copy
/// for the whole CLI (previously duplicated in list.rs and pull.rs).
pub fn human_size(bytes: u64) -> String {
    const K: f64 = 1024.0;
    let b = bytes as f64;
    if b >= K * K * K {
        format!("{:.1} GB", b / (K * K * K))
    } else if b >= K * K {
        format!("{:.1} MB", b / (K * K))
    } else if b >= K {
        format!("{:.1} KB", b / K)
    } else {
        format!("{bytes} B")
    }
}

impl Ui {
    /// Render a model-state badge. Colored when on; plain text when off (so the
    /// width is identical and `Table` alignment is unaffected — color codes are
    /// zero-width). Unknown states render dim.
    pub fn badge(self, state: &str) -> String {
        match state {
            "ready" => self.accent(state),
            "downloading" => self.amber(state),
            "incomplete" => self.red(state),
            other => self.dim(other),
        }
    }
}

/// A minimal left-aligned column table. Column widths are computed from the
/// **visible** (un-styled) cell text so ANSI codes never throw off alignment;
/// callers therefore pass plain strings and let the renderer paint.
pub struct Table {
    ui: Ui,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    widths: Vec<usize>,
}

impl Table {
    pub fn new(ui: Ui, headers: &[&str]) -> Self {
        let headers: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
        let widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
        Self {
            ui,
            headers,
            rows: Vec::new(),
            widths,
        }
    }

    /// Add a row of cells. Cells may contain ANSI styling; widths track the
    /// **visible** (un-styled) text so colored cells still align.
    pub fn row(&mut self, cells: Vec<String>) {
        for (i, c) in cells.iter().enumerate() {
            if i < self.widths.len() {
                self.widths[i] = self.widths[i].max(visible_width(c));
            }
        }
        self.rows.push(cells);
    }

    /// Render the table to a string: a faint header line then the rows, each
    /// column padded to its width + two trailing spaces between columns.
    pub fn render(&self) -> String {
        let mut out = String::new();
        // header
        let mut hline = String::from("  ");
        for (i, h) in self.headers.iter().enumerate() {
            hline.push_str(&pad(h, self.widths.get(i).copied().unwrap_or(0)));
            hline.push_str("  ");
        }
        out.push_str(&self.ui.faint(hline.trim_end()));
        out.push('\n');
        // rows
        for cells in &self.rows {
            let mut line = String::from("  ");
            for (i, c) in cells.iter().enumerate() {
                let w = self.widths.get(i).copied().unwrap_or(0);
                line.push_str(&pad(c, w));
                line.push_str("  ");
            }
            out.push_str(line.trim_end());
            out.push('\n');
        }
        out
    }
}

/// Left-pad-to-width by visible char count. ANSI styling in `s` is zero-width,
/// so any trailing spaces land after the styled text (including its reset).
fn pad(s: &str, w: usize) -> String {
    let n = visible_width(s);
    if n >= w {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(w - n))
    }
}

/// Drop ANSI SGR escape sequences (`\x1b[ ... m`), keeping only printable text.
/// Deliberately minimal — it handles only the `m`-terminated CSI sequences this
/// CLI emits, not a general terminal parser (YAGNI).
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip until (and including) the terminating 'm'.
            for e in chars.by_ref() {
                if e == 'm' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Visible (printable) width of a string, ignoring ANSI styling.
fn visible_width(s: &str) -> usize {
    strip_ansi(s).chars().count()
}

/// RAII guard: enters raw mode + alternate screen on `enter()`, restores both on
/// drop. `inert()` builds a guard that does no terminal I/O (for tests / when
/// color is off). Disarm before a clean manual restore to avoid double work.
pub struct TermGuard {
    active: bool,
    real: bool,
}

impl TermGuard {
    /// Enter raw mode + alternate screen. Returns an inert guard on failure so
    /// the caller can fall back to plain output without unwinding.
    pub fn enter() -> Self {
        use crossterm::{
            execute,
            terminal::{enable_raw_mode, EnterAlternateScreen},
        };
        use std::io::stdout;
        // Once raw mode is on, mark the guard `real` so a later alt-screen failure
        // still gets cleaned up on drop (restore disables raw mode) — the guard's
        // "restore on every exit path" contract must hold even on partial setup.
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

    /// A guard that performs no terminal I/O (tests, color-off path). Part of the
    /// guard's public API; today only the unit test constructs one.
    #[allow(dead_code)]
    pub fn inert() -> Self {
        TermGuard {
            active: true,
            real: false,
        }
    }

    pub fn active(&self) -> bool {
        self.active
    }

    /// Stop tracking so `drop` does nothing (e.g. you restored manually). Part of
    /// the guard's public API; today only the unit test exercises it.
    #[allow(dead_code)]
    pub fn disarm(&mut self) {
        // Manual restore claimed: clear both flags so drop is a no-op and reports
        // inactive.
        self.active = false;
        self.real = false;
    }

    fn restore(&mut self) {
        // Key off `real`: any guard that actually changed terminal state must undo
        // it on drop, even if alt-screen entry failed after raw mode was enabled
        // (active == false but real == true).
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
    fn off_when_not_tty() {
        assert_eq!(resolve(false, None, Some("xterm"), false), ColorMode::Off);
    }
    #[test]
    fn off_when_no_color_env_or_flag_or_dumb() {
        assert_eq!(resolve(true, None, Some("xterm"), true), ColorMode::Off);
        assert_eq!(
            resolve(false, Some(""), Some("xterm"), true),
            ColorMode::Off
        );
        assert_eq!(resolve(false, None, Some("dumb"), true), ColorMode::Off);
    }
    #[test]
    fn on_when_tty_and_clean() {
        assert_eq!(
            resolve(false, None, Some("xterm-256color"), true),
            ColorMode::On
        );
    }
    #[test]
    fn paints_with_color_on() {
        let ui = Ui::new(ColorMode::On);
        let s = ui.accent("hi");
        assert!(s.starts_with("\x1b["));
        assert!(s.ends_with("\x1b[0m"));
        assert!(s.contains("hi"));
    }
    #[test]
    fn passthrough_with_color_off() {
        let ui = Ui::new(ColorMode::Off);
        assert_eq!(ui.accent("hi"), "hi");
        assert_eq!(ui.bold("x"), "x");
    }

    #[test]
    fn human_size_buckets() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(2 * 1024), "2.0 KB");
        assert_eq!(human_size(5 * 1024 * 1024), "5.0 MB");
        assert_eq!(human_size(3 * 1024 * 1024 * 1024), "3.0 GB");
    }

    #[test]
    fn badge_text_is_plain_when_off() {
        let ui = Ui::new(ColorMode::Off);
        assert_eq!(ui.badge("ready"), "ready");
        assert_eq!(ui.badge("downloading"), "downloading");
        assert_eq!(ui.badge("incomplete"), "incomplete");
    }

    #[test]
    fn visible_width_ignores_ansi() {
        assert_eq!(visible_width("hi"), 2);
        assert_eq!(visible_width("\x1b[38;2;55;224;176mhi\x1b[0m"), 2);
        assert_eq!(visible_width(""), 0);
    }

    #[test]
    fn table_aligns_colored_cells() {
        // A colored cell must align the same as its plain text.
        let on = Ui::new(ColorMode::On);
        let mut t = Table::new(on, &["NAME", "STATE"]);
        t.row(vec!["gemma3".into(), on.badge("ready")]);
        t.row(vec!["llama3.2:1b".into(), on.badge("incomplete")]);
        let out = t.render();
        // The visible layout: 'gemma3' padded to width of 'llama3.2:1b' (11),
        // so a plain-text scan with ANSI stripped shows aligned columns.
        let stripped: String = strip_ansi(&out);
        // Each data line should have STATE starting at the same visible column.
        let lines: Vec<&str> = stripped
            .lines()
            .filter(|l| l.contains("ready") || l.contains("incomplete"))
            .collect();
        assert_eq!(lines.len(), 2);
        let col_ready = lines[0].find("ready").unwrap();
        let col_inc = lines[1].find("incomplete").unwrap();
        assert_eq!(col_ready, col_inc);
    }

    #[test]
    fn term_guard_disarm_is_idempotent() {
        let mut g = TermGuard::inert(); // test constructor: no real terminal calls
        assert!(g.active());
        g.disarm();
        assert!(!g.active());
        g.disarm(); // no panic / no double work
        assert!(!g.active());
    }

    #[test]
    fn table_aligns_columns_plain() {
        let ui = Ui::new(ColorMode::Off);
        let mut t = Table::new(ui, &["NAME", "SIZE"]);
        t.row(vec!["gemma3".into(), "8.1 GB".into()]);
        t.row(vec!["llama3.2:1b".into(), "1.2 GB".into()]);
        let out = t.render();
        // Header present, second column left-aligned under the widest name.
        assert!(out.contains("NAME"));
        assert!(out.contains("gemma3     ")); // padded to 'llama3.2:1b' width
        assert!(out.contains("llama3.2:1b"));
    }
}
