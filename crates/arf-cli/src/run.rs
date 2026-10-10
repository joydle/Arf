//! `arf run <model>` — an interactive streaming chat REPL in the terminal.
//! Styled prints only (no alternate screen) so scrollback and SSH work.

use crate::ui::Ui;
use std::error::Error;
use std::io::{self, Write};

/// A REPL line is either a slash-command or a user prompt.
#[derive(Debug, PartialEq)]
pub(crate) enum Line {
    Exit,
    Clear,
    /// Show or hide the model's reasoning (the HTTP chat only).
    Think,
    Prompt(String),
    Empty,
}

/// Classify a raw input line.
pub(crate) fn parse_line(raw: &str) -> Line {
    match raw.trim() {
        "" => Line::Empty,
        "/exit" | "/quit" => Line::Exit,
        "/clear" => Line::Clear,
        "/think" => Line::Think,
        other => Line::Prompt(other.to_string()),
    }
}

/// Per-turn generation stats shown in the footer line.
pub struct Stats {
    pub tokens: usize,
    pub tok_per_s: f64,
    pub ttft_ms: u64,
}

/// Run the REPL. `gen` streams/produces a reply for one prompt, returning
/// (reply_text, stats). Injecting `gen` keeps the loop independent of the engine.
pub fn repl<G>(ui: Ui, model_label: &str, mut gen: G) -> Result<(), Box<dyn Error>>
where
    G: FnMut(&str) -> Result<(String, Stats), Box<dyn Error>>,
{
    println!(
        "{} · type {} to quit, {} to reset",
        ui.dim(model_label),
        ui.accent("/exit"),
        ui.accent("/clear")
    );
    let stdin = io::stdin();
    loop {
        print!("\n{} ", ui.accent("›"));
        io::stdout().flush()?;
        let mut buf = String::new();
        if stdin.read_line(&mut buf)? == 0 {
            break; // EOF (Ctrl-D)
        }
        match parse_line(&buf) {
            Line::Exit => break,
            Line::Empty | Line::Think => continue,
            Line::Clear => {
                println!("{}", ui.faint("— context cleared —"));
                continue;
            }
            Line::Prompt(p) => {
                let (text, st) = gen(&p)?;
                println!("{text}");
                println!(
                    "{}",
                    ui.faint(&format!(
                        "{} tok · {:.0} tok/s · ttft {}ms",
                        st.tokens, st.tok_per_s, st.ttft_ms
                    ))
                );
            }
        }
    }
    Ok(())
}

/// What the HTTP chat sends besides the conversation.
pub struct ChatOpts {
    /// `max_tokens` per reply; `None` leaves it to the server.
    pub max_tokens: Option<usize>,
    /// `false` asks the model not to reason first (`enable_thinking: false`).
    pub think: bool,
}

/// `arf run`: chat with a running `arf-serve` over `/v1/chat/completions`, streaming. The
/// conversation is kept and resent each turn, so the server's prefix cache makes later turns start
/// fast. Reasoning prints dimmed (`/think` hides it and shows a counter instead).
pub fn chat_http(
    ui: Ui,
    server: &crate::daemon::Server,
    label: &str,
    opts: &ChatOpts,
) -> Result<(), Box<dyn Error>> {
    println!(
        "{} · {} to quit, {} to start over, {} to show or hide reasoning",
        ui.dim(label),
        ui.accent("/exit"),
        ui.accent("/clear"),
        ui.accent("/think")
    );
    let mut history: Vec<serde_json::Value> = Vec::new();
    let mut show_thinking = true;
    let stdin = io::stdin();
    loop {
        print!("\n{} ", ui.accent("›"));
        io::stdout().flush()?;
        let mut buf = String::new();
        if stdin.read_line(&mut buf)? == 0 {
            break;
        }
        match parse_line(&buf) {
            Line::Exit => break,
            Line::Empty => continue,
            Line::Clear => {
                history.clear();
                println!("{}", ui.faint("— conversation cleared —"));
            }
            Line::Think => {
                show_thinking = !show_thinking;
                println!(
                    "{}",
                    ui.faint(if show_thinking {
                        "— reasoning shown —"
                    } else {
                        "— reasoning hidden —"
                    })
                );
            }
            Line::Prompt(p) => {
                history.push(serde_json::json!({"role": "user", "content": p}));
                match turn(ui, server, &history, opts, show_thinking) {
                    Ok(reply) => {
                        history.push(serde_json::json!({"role": "assistant", "content": reply}))
                    }
                    Err(e) => {
                        history.pop();
                        println!("{} {e}", ui.red("error:"));
                    }
                }
            }
        }
    }
    Ok(())
}

/// One streamed reply; returns its answer text (reasoning is not kept in the history).
fn turn(
    ui: Ui,
    server: &crate::daemon::Server,
    history: &[serde_json::Value],
    opts: &ChatOpts,
    show_thinking: bool,
) -> Result<String, Box<dyn Error>> {
    let mut body = serde_json::json!({
        "model": server.model_id,
        "messages": history,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if let Some(n) = opts.max_tokens {
        body["max_tokens"] = n.into();
    }
    if !opts.think {
        body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
    }
    let t0 = std::time::Instant::now();
    let resp = ureq::post(&format!(
        "http://127.0.0.1:{}/v1/chat/completions",
        server.port
    ))
    .set("Content-Type", "application/json")
    .send_string(&body.to_string())
    .map_err(|e| format!("the server did not answer: {e}"))?;
    let reader = io::BufReader::new(resp.into_reader());
    let (mut answer, mut thinking_pieces) = (String::new(), 0usize);
    let (mut first, mut last) = (None, None);
    let (mut in_thinking, mut usage) = (false, serde_json::Value::Null);
    for line in io::BufRead::lines(reader) {
        let line = line?;
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };
        if !v["usage"].is_null() {
            usage = v["usage"].clone();
        }
        let delta = &v["choices"][0]["delta"];
        let reasoning = delta["reasoning_content"].as_str().unwrap_or("");
        let content = delta["content"].as_str().unwrap_or("");
        if reasoning.is_empty() && content.is_empty() {
            continue;
        }
        let now = std::time::Instant::now();
        first.get_or_insert(now);
        last = Some(now);
        if !reasoning.is_empty() {
            thinking_pieces += 1;
            if show_thinking {
                print!("{}", ui.faint(reasoning));
            } else {
                print!("\r{}", ui.faint(&format!("thinking … {thinking_pieces}")));
            }
            in_thinking = true;
        }
        if !content.is_empty() {
            if in_thinking {
                print!("{}", if show_thinking { "\n\n" } else { "\r\x1b[2K" });
                in_thinking = false;
            }
            print!("{content}");
            answer.push_str(content);
        }
        io::stdout().flush()?;
    }
    println!();
    let tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    let ttft = first.map_or(0.0, |f| (f - t0).as_secs_f64());
    let decode = match (first, last) {
        (Some(f), Some(l)) if tokens > 1 && l > f => (tokens - 1) as f64 / (l - f).as_secs_f64(),
        _ => 0.0,
    };
    let mut footer = format!("{tokens} tok · {decode:.1} tok/s · first token {ttft:.1}s");
    if cached > 0 {
        footer.push_str(&format!(" · {cached} prompt tokens reused"));
    }
    println!("{}", ui.faint(&footer));
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_slash_commands_and_prompts() {
        assert_eq!(parse_line("  /exit "), Line::Exit);
        assert_eq!(parse_line("/quit"), Line::Exit);
        assert_eq!(parse_line("/clear"), Line::Clear);
        assert_eq!(parse_line(" /think"), Line::Think);
        assert_eq!(parse_line(""), Line::Empty);
        assert_eq!(parse_line("hello"), Line::Prompt("hello".into()));
    }
}
