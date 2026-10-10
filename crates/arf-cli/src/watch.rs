//! A compact ANSI live view for `generate --watch`: streaming text, a KV-cache
//! fill bar, and decode tok/s. Pure `render` is unit-tested; the driver does IO.

use std::io::{IsTerminal, Write};
use std::time::Instant;

use arf_core::engine::{LlmEngine, StepInfo};
use arf_core::sampling::SamplingParams;
use arf_core::Tokenizer;

/// A snapshot to render. Borrows the decoded text so rendering allocates little.
pub struct Frame<'a> {
    pub header: &'a str,
    pub text: &'a str,
    pub kv_used: usize,
    pub kv_total: usize,
    pub prefill_tokens: usize,
    pub tok_per_sec: f64,
    pub context_len: usize,
}

/// A `width`-cell `█`/`░` bar for `used/total` (no divide-by-zero).
fn kv_bar(used: usize, total: usize, width: usize) -> String {
    let frac = if total == 0 {
        0.0
    } else {
        used as f64 / total as f64
    };
    let filled = ((frac * width as f64) as usize).min(width);
    "█".repeat(filled) + &"░".repeat(width - filled)
}

/// Render one frame to a string (pure; the driver prints it).
fn render(f: &Frame) -> String {
    let bar = kv_bar(f.kv_used, f.kv_total, 20);
    format!(
        "{header}\n\n  {text}▮\n\n  ┌─ kv-cache ──────────────────────┐\n  │ {bar}  {used}/{total} blk │\n  └─────────────────────────────────┘\n  prefill {pre} tok · decode {tps:.2} tok/s · ctx {ctx}\n",
        header = f.header,
        text = f.text,
        bar = bar,
        used = f.kv_used,
        total = f.kv_total,
        pre = f.prefill_tokens,
        tps = f.tok_per_sec,
        ctx = f.context_len,
    )
}

/// Drive a single-prompt generation with the live view. Falls back to plain
/// incremental printing when stdout is not a TTY.
pub fn watch_generate(
    engine: &mut LlmEngine,
    tokenizer: &Tokenizer,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
    header: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let is_tty = std::io::stdout().is_terminal();
    let prefill_tokens = prompt_ids.len();
    let prompt_text = tokenizer.decode(&prompt_ids, true)?;

    // Accumulate generated ids so we can decode incrementally (the prompt text
    // is fixed; we re-decode the growing generated suffix and print the delta).
    let mut gen_ids: Vec<u32> = Vec::new();
    let mut shown = prompt_text.clone();
    let start = Instant::now();

    // A tokenizer error inside the callback is stored and surfaced after the run.
    let mut decode_err: Option<String> = None;

    engine.generate_streaming(prompt_ids, params, &mut |info: StepInfo| {
        gen_ids.push(info.token);
        let gen_text = match tokenizer.decode(&gen_ids, true) {
            Ok(t) => t,
            Err(e) => {
                decode_err = Some(e.to_string());
                return;
            }
        };
        let full = format!("{prompt_text}{gen_text}");
        let elapsed = start.elapsed().as_secs_f64().max(1e-6);
        let tps = gen_ids.len() as f64 / elapsed;

        if is_tty {
            // Clear screen + home, then redraw the whole frame.
            print!("\x1b[2J\x1b[H");
            let frame = Frame {
                header,
                text: &full,
                kv_used: info.kv_blocks_used,
                kv_total: info.kv_blocks_total,
                prefill_tokens,
                tok_per_sec: tps,
                context_len: info.context_len,
            };
            print!("{}", render(&frame));
            let _ = std::io::stdout().flush();
        } else {
            // Non-TTY: print only the newly added text (no escape codes).
            if full.len() > shown.len() {
                print!("{}", &full[shown.len()..]);
                let _ = std::io::stdout().flush();
            }
            shown = full;
        }
    })?;

    if let Some(e) = decode_err {
        return Err(format!("tokenizer decode failed mid-stream: {e}").into());
    }
    if !is_tty {
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Frame<'static> {
        Frame {
            header: "arf · llama-3.2-1b · cpu",
            text: "The capital of France is Paris",
            kv_used: 41,
            kv_total: 96,
            prefill_tokens: 38,
            tok_per_sec: 0.51,
            context_len: 49,
        }
    }

    #[test]
    fn render_contains_text_and_occupancy() {
        let s = render(&frame());
        assert!(s.contains("The capital of France is Paris"));
        assert!(s.contains("41/96"));
        assert!(s.contains("0.5")); // tok/s, one decimal visible
        assert!(s.contains("ctx 49"));
    }

    #[test]
    fn kv_bar_fill_is_proportional() {
        // 41/96 ≈ 0.427 → 8 of 20 cells filled.
        let bar = kv_bar(41, 96, 20);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 8);
        assert_eq!(bar.chars().count(), 20);
    }

    #[test]
    fn kv_bar_handles_zero_total() {
        // No divide-by-zero; an empty pool renders all-empty.
        let bar = kv_bar(0, 0, 20);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 20);
    }
}
