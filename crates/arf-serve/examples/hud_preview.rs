//! Live preview of the `arf serve` HUD with ANIMATED fake data — so you can watch
//! the real ratatui dashboard (the same `dashboard::draw`) without a server or model.
//!
//!   cargo run --release -p arf-serve --example hud_preview
//!
//! Quit: q / Esc / Ctrl-C (the same clean teardown the live dashboard uses).
//! It simulates an agent swarm warming up: throughput ramps, sequences join the batch,
//! KV fills, the prefix cache warms — then it breathes.

use arf_serve::dashboard::{self, DashSnapshot, Ring};
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::stdout;
use std::time::{Duration, Instant};

fn main() {
    // Terminal setup (same shape as the real dashboard).
    if enable_raw_mode().is_err() {
        eprintln!("not a TTY — run this in a terminal");
        return;
    }
    let _ = execute!(stdout(), EnterAlternateScreen);
    let mut term = Terminal::new(CrosstermBackend::new(stdout())).expect("terminal");

    let cap = 60usize;
    let mut tok = Ring::new(cap);
    let mut req = Ring::new(cap);
    let mut peak = 0u64;
    let started = Instant::now();
    // Deterministic-ish wander (no rand dep): a couple of LCG counters.
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 33) as f64 / (1u64 << 31) as f64 // 0..1
    };

    let mut running = 0usize;
    let max = 8usize;
    let kv_total = 512usize;
    let mut served = 0u64;
    let mut frame = 0u64;

    loop {
        // Quit on q / Esc / Ctrl-C.
        if let Ok(true) = event::poll(Duration::from_millis(120)) {
            if let Ok(Event::Key(k)) = event::read() {
                let quit = matches!(k.code, KeyCode::Char('q') | KeyCode::Esc)
                    || (k.code == KeyCode::Char('c')
                        && k.modifiers.contains(KeyModifiers::CONTROL));
                if quit {
                    break;
                }
            }
        }
        frame += 1;

        // Swarm warms up: agents join over the first ~3s, then it breathes 5..8.
        if running < max && rng() < 0.25 {
            running += 1;
        }
        if running > 5 && rng() < 0.06 {
            running -= 1;
            served += 1 + (rng() * 3.0) as u64;
        }

        // Throughput scales with concurrency (the advantage) + jitter + a slow swell.
        let base = 70.0 * running as f64; // ~70 tok/s/seq amortized
        let swell = (frame as f64 / 9.0).sin() * 40.0;
        let agg = (base + swell + rng() * 30.0).max(0.0) as u64;
        tok.push(agg);
        peak = peak.max(agg);
        req.push(running as u64);

        let kv_used = ((running as f64 / max as f64) * kv_total as f64 * 0.78
            + (frame as f64 / 13.0).sin() * 30.0)
            .clamp(0.0, kv_total as f64) as usize;
        let prefix =
            ((running as f64 * 9.0) + (frame as f64 / 7.0).sin() * 8.0).clamp(0.0, 92.0) as u16;

        let snap = DashSnapshot {
            model_id: "gemma-4-12b-q4k".into(),
            endpoint: ":8080".into(),
            uptime: fmt_uptime(started.elapsed()),
            tok_series: tok.series(),
            tok_now: agg,
            tok_peak: peak,
            req_series: req.series(),
            req_active: running,
            req_served: served,
            kv_used,
            kv_total,
            batch_running: running,
            batch_max: max,
            prefix_hit: prefix,
            // Demo GPU-frame breakdown (the real path fills this from gpu::profile).
            kernels: vec![
                ("matmul".into(), 77, Some(25)),
                ("attention".into(), 6, None),
                ("post_norm".into(), 4, None),
                ("in_norm".into(), 3, None),
                ("swiglu".into(), 2, None),
                ("rope_qk".into(), 2, None),
                ("kv_scatter".into(), 2, None),
            ],
        };

        if term.draw(|f| dashboard::draw(f, &snap)).is_err() {
            break;
        }
        std::thread::sleep(Duration::from_millis(130));
    }

    // Restore terminal.
    let _ = execute!(stdout(), LeaveAlternateScreen);
    let _ = disable_raw_mode();
    println!("HUD preview closed cleanly.");
}

fn fmt_uptime(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}", (s / 60) % 60, s % 60)
}
