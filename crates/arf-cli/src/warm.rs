//! `arf warm-claude` (internal): read Claude Code's safety-check prompts before the first session
//! needs them.
//!
//! In auto mode Claude Code checks each tool call with the same model, through a ~30,500-token
//! prompt in two stages (Claude Code 2.1.294, captured 2026-10-08). Their system text is the same
//! for every project, but a server that has never read it takes ~170 s on an M4 Max, past the
//! check's 60 s: the first tool calls of a session come back "temporarily unavailable" while it is
//! read. `arf launch claude` starts this beside the agent. It runs the installed `claude` once,
//! headless, in an empty folder against a stub endpoint here (no model, a few seconds of CPU) and
//! keeps the two check requests it sends; then it sends them to the server as BACKGROUND work
//! (`x-arf-background: 1`): read only when nothing else has work, and a real check that arrives
//! first waits for it rather than reading the same tokens again. The states they leave are saved
//! to disk with the others (`prefix_disk`), so the next start loads them. Nothing of Claude Code's
//! is stored here: the prompts come from the version installed, each time.
//!
//! `ARF_NO_WARM=1` turns it off; so does a permission mode other than auto (no check is made).

use std::error::Error;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The task the headless agent is given, and the commands the stub has it run. Which form of the
/// check Claude Code sends depends on the command (2.1.294: `python3 test_x.py` drew the
/// `<severity>` form, `python3 hello.py` and `sleep 5 && python3 test_calc.py` the `<block>` one;
/// a `python3 -c` one-liner ran unchecked), so it runs one of each kind.
const TASK: &str =
    "Run `python3 test_hello.py`, then `python3 hello.py`, then `sleep 1 && python3 \
test_hello.py` with the Bash tool, then reply DONE.";
const COMMANDS: [&str; 3] = [
    "python3 test_hello.py",
    "python3 hello.py",
    "sleep 1 && python3 test_hello.py",
];
/// The longest the headless capture may take.
const CAPTURE_LIMIT: Duration = Duration::from_secs(90);

/// Does the user's setup run Claude Code in auto mode? A `--permission-mode` among the agent's
/// arguments decides; else `permissions.defaultMode` in `~/.claude/settings.json`; else yes (the
/// default the checks were captured under).
pub fn auto_mode(args: &[String]) -> bool {
    if args.iter().any(|a| a == "--dangerously-skip-permissions") {
        return false;
    }
    if let Some(i) = args.iter().position(|a| a == "--permission-mode") {
        return args.get(i + 1).is_some_and(|m| m == "auto");
    }
    if let Some(m) = args
        .iter()
        .find_map(|a| a.strip_prefix("--permission-mode="))
    {
        return m == "auto";
    }
    let Some(dir) = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".claude")))
    else {
        return true;
    };
    let settings = dir.join("settings.json");
    std::fs::read_to_string(settings)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v["permissions"]["defaultMode"].as_str().map(str::to_string))
        .is_none_or(|m| m == "auto")
}

/// What `arf launch claude` sets for Claude Code, and the capture with it: the check's system
/// text depends on it (under this list Claude Code 2.1.294 sends the `<block>` form; the agent
/// benchmark's environment drew a `<severity>` form first, 2026-10-08).
pub fn claude_env(base: &str, model_id: &str) -> [(&'static str, String); 10] {
    let id = || model_id.to_string();
    [
        ("ANTHROPIC_BASE_URL", base.to_string()),
        ("ANTHROPIC_AUTH_TOKEN", "arf".to_string()),
        ("ANTHROPIC_MODEL", id()),
        ("ANTHROPIC_SMALL_FAST_MODEL", id()),
        ("ANTHROPIC_DEFAULT_OPUS_MODEL", id()),
        ("ANTHROPIC_DEFAULT_SONNET_MODEL", id()),
        ("ANTHROPIC_DEFAULT_HAIKU_MODEL", id()),
        ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1".to_string()),
        ("CLAUDE_CODE_DISABLE_TERMINAL_TITLE", "1".to_string()),
        ("CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION", "false".to_string()),
    ]
}

/// The entries of [`claude_env`] a user's own setting overrides in `arf launch`.
pub const USER_MAY_SET: [&str; 2] = [
    "CLAUDE_CODE_DISABLE_TERMINAL_TITLE",
    "CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION",
];

/// The `autoMode` block of the user's Claude Code settings (`$CLAUDE_CONFIG_DIR/settings.json`,
/// else `~/.claude/settings.json`), if any.
fn user_auto_mode() -> Option<serde_json::Value> {
    let dir = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".claude")))?;
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("settings.json")).ok()?).ok()?;
    v.get("autoMode").cloned()
}

/// Start `arf warm-claude` detached from this terminal, unless `ARF_NO_WARM` is set.
pub fn spawn(port: u16, model_id: &str, claude: &Path) {
    if std::env::var_os("ARF_NO_WARM").is_some() {
        return;
    }
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.args([
        "warm-claude",
        "--port",
        &port.to_string(),
        "--model-id",
        model_id,
        "--claude",
    ])
    .arg(claude)
    .stdin(std::process::Stdio::null())
    .stdout(std::process::Stdio::null())
    .stderr(std::process::Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let _ = cmd.spawn();
}

/// Which request a captured body is.
#[derive(Debug, PartialEq)]
enum Kind {
    /// The agent's own turn (it carries tools).
    Agent,
    /// A stage of the safety check: no tools, a system text of ~139K characters (2.1.294).
    Check,
    /// Anything else (a title, a summary).
    Other,
}

fn system_text(body: &serde_json::Value) -> String {
    match &body["system"] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b["text"].as_str())
            .collect::<String>(),
        _ => String::new(),
    }
}

fn kind(body: &serde_json::Value) -> Kind {
    if body["tools"].as_array().is_some_and(|t| !t.is_empty()) {
        Kind::Agent
    } else if system_text(body).len() >= 20_000 {
        Kind::Check
    } else {
        Kind::Other
    }
}

/// Does this check request ask for a `<severity>` (the first stage)? Its closing instruction
/// says so; the second stage's system text mentions severities too, so that is not looked at.
fn asks_severity(body: &serde_json::Value) -> bool {
    body["messages"]
        .as_array()
        .and_then(|m| m.last())
        .is_some_and(|m| {
            let text = match &m["content"] {
                serde_json::Value::String(s) => s.clone(),
                c => c.to_string(),
            };
            text.contains("<severity>")
        })
}

/// Every distinct check prompt among the captured requests, in the order first sent. Claude Code
/// sends one or two stages a tool call, with system texts that differ only in their closing
/// output format (139,238 and 140,508 characters seen in 2.1.294, not always both): each is
/// read once, and the later ones resume from the first's checkpoints.
fn checks(seen: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut out: Vec<serde_json::Value> = Vec::new();
    for b in seen.iter().filter(|b| kind(b) == Kind::Check) {
        if !out.iter().any(|o| system_text(o) == system_text(b)) {
            out.push(b.clone());
        }
    }
    out
}

/// The stub's reply to one captured request: the agent runs one Bash command and then ends, the
/// check's first stage asks for the second (a high severity), and the second allows it.
fn reply(body: &serde_json::Value) -> (Vec<serde_json::Value>, &'static str) {
    let text = |t: &str| vec![serde_json::json!({"type": "text", "text": t})];
    match kind(body) {
        Kind::Agent => {
            let ran = body["messages"].as_array().map_or(0, |m| {
                m.iter()
                    .filter_map(|m| m["content"].as_array())
                    .flatten()
                    .filter(|b| b["type"] == "tool_result")
                    .count()
            });
            match COMMANDS.get(ran) {
                Some(command) => (
                    vec![
                        serde_json::json!({"type": "tool_use", "id": format!("toolu_warm{ran}"),
                        "name": "Bash", "input": {"command": command, "description": "Run hello.py"}}),
                    ],
                    "tool_use",
                ),
                None => (text("DONE"), "end_turn"),
            }
        }
        Kind::Check if asks_severity(body) => (text("<severity>90</severity>"), "end_turn"),
        Kind::Check => (text("<block>no</block>"), "end_turn"),
        Kind::Other => (text("ok"), "end_turn"),
    }
}

/// Answer one HTTP/1.1 request on `conn` (then close it), recording a `/v1/messages` body.
fn serve_one(mut conn: TcpStream, seen: &Mutex<Vec<serde_json::Value>>) -> std::io::Result<()> {
    conn.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut reader = BufReader::new(conn.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut raw = vec![0u8; len];
    reader.read_exact(&mut raw)?;
    let respond = |conn: &mut TcpStream, ctype: &str, body: &str| {
        write!(
            conn,
            "HTTP/1.1 200 OK\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    };
    if method != "POST" || !path.starts_with("/v1/messages") {
        return respond(
            &mut conn,
            "application/json",
            r#"{"data":[],"input_tokens":1}"#,
        );
    }
    let body: serde_json::Value = serde_json::from_slice(&raw).unwrap_or_default();
    let (content, stop) = reply(&body);
    let stream = body["stream"].as_bool().unwrap_or(false);
    let model = body["model"].clone();
    seen.lock().unwrap().push(body);
    let usage = serde_json::json!({"input_tokens": 1, "output_tokens": 1});
    let message = |content: serde_json::Value, stop: serde_json::Value| {
        serde_json::json!({"id": "msg_warm", "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": stop, "stop_sequence": null, "usage": usage})
    };
    if !stream {
        let m = message(serde_json::json!(content), serde_json::json!(stop));
        return respond(&mut conn, "application/json", &m.to_string());
    }
    let mut sse = String::new();
    let mut ev =
        |t: &str, d: serde_json::Value| sse.push_str(&format!("event: {t}\ndata: {d}\n\n"));
    ev(
        "message_start",
        serde_json::json!({"type": "message_start", "message": message(serde_json::json!([]), serde_json::Value::Null)}),
    );
    for (i, b) in content.iter().enumerate() {
        let (start, delta) = if b["type"] == "text" {
            (
                serde_json::json!({"type": "text", "text": ""}),
                serde_json::json!({"type": "text_delta", "text": b["text"]}),
            )
        } else {
            let mut start = b.clone();
            start["input"] = serde_json::json!({});
            (
                start,
                serde_json::json!({"type": "input_json_delta", "partial_json": b["input"].to_string()}),
            )
        };
        ev(
            "content_block_start",
            serde_json::json!({"type": "content_block_start", "index": i, "content_block": start}),
        );
        ev(
            "content_block_delta",
            serde_json::json!({"type": "content_block_delta", "index": i, "delta": delta}),
        );
        ev(
            "content_block_stop",
            serde_json::json!({"type": "content_block_stop", "index": i}),
        );
    }
    ev(
        "message_delta",
        serde_json::json!({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": null},
            "usage": {"output_tokens": 1}}),
    );
    ev("message_stop", serde_json::json!({"type": "message_stop"}));
    respond(&mut conn, "text/event-stream", &sse)
}

/// Run `claude` headless against a stub and return the check requests it sent ([`checks`]).
fn capture(claude: &Path, model_id: &str) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    {
        let seen = seen.clone();
        std::thread::spawn(move || {
            for conn in listener.incoming().flatten() {
                let seen = seen.clone();
                std::thread::spawn(move || {
                    let _ = serve_one(conn, &seen);
                });
            }
        });
    }
    let dir = std::env::temp_dir().join(format!("arf-warm-{}", std::process::id()));
    let (work, config) = (dir.join("work"), dir.join("config"));
    std::fs::create_dir_all(&work)?;
    std::fs::create_dir_all(&config)?;
    // The user's auto-mode settings shape the check's system text (an `autoMode.environment`
    // describing their clouds and repositories put a real session's check ~9,000 tokens from the
    // default one, 2026-10-08), so they are copied — that key only: hooks, plugins and the rest of
    // the user's settings do not run here.
    if let Some(auto) = user_auto_mode() {
        std::fs::write(
            config.join("settings.json"),
            serde_json::json!({ "autoMode": auto }).to_string(),
        )?;
    }
    let mut child = std::process::Command::new(claude)
        // The task first: `--allowedTools` takes every value after it.
        .args(["-p", TASK, "--model", model_id, "--strict-mcp-config"])
        .args(["--allowedTools", "Bash(python3:*)"])
        .current_dir(&work)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", &config)
        .env("CLAUDE_CONFIG_DIR", &config)
        .env("PWD", &work)
        // The check names the user (`$USER`) in its session context.
        .env("USER", std::env::var_os("USER").unwrap_or_default())
        .env("LOGNAME", std::env::var_os("LOGNAME").unwrap_or_default())
        .env("TERM", "dumb")
        .envs(claude_env(&format!("http://127.0.0.1:{port}"), model_id))
        .env("DISABLE_TELEMETRY", "1")
        .env("DISABLE_ERROR_REPORTING", "1")
        .env("DISABLE_AUTOUPDATER", "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    let start = Instant::now();
    while start.elapsed() < CAPTURE_LIMIT && child.try_wait()?.is_none() {
        std::thread::sleep(Duration::from_millis(200));
    }
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
    let found = checks(&seen.lock().unwrap());
    if found.is_empty() {
        return Err("Claude Code sent no safety check (not in auto mode?)".into());
    }
    Ok(found)
}

/// `arf warm-claude`: capture the checks, then have the server on `port` read them in the
/// background, one after the other (the second resumes from the first's last checkpoint).
///
/// One at a time per port: every `arf launch claude` starts one, and in the 2026-10-08 release test
/// the launches against one server stacked their reads up. A second exits while the first lives.
pub fn claude(port: u16, model_id: &str, claude: &Path) -> Result<(), Box<dyn Error>> {
    let lock = crate::daemon::arf_home()
        .join("run")
        .join(format!("warm-{port}.pid"));
    let holder = std::fs::read_to_string(&lock)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok());
    if holder.is_some_and(crate::daemon::alive) {
        return Ok(());
    }
    if let Some(dir) = lock.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&lock, std::process::id().to_string())?;
    let mut log = Log::open(port);
    let done = send(port, model_id, claude, &mut log);
    if let Err(e) = &done {
        log.line(&format!("stopped: {e}"));
    }
    let _ = std::fs::remove_file(&lock);
    done
}

/// What a warm-up did, in `~/.arf/logs/warm-PORT.log` (rewritten by each): it runs detached
/// with its output closed, and a warm-up that silently did nothing looks exactly like one that
/// worked until the first tool call is slow.
struct Log {
    t0: Instant,
    file: Option<std::fs::File>,
}

impl Log {
    fn open(port: u16) -> Self {
        let dir = crate::daemon::arf_home().join("logs");
        let _ = std::fs::create_dir_all(&dir);
        let file = std::fs::File::create(dir.join(format!("warm-{port}.log"))).ok();
        Log {
            t0: Instant::now(),
            file,
        }
    }
    fn line(&mut self, text: &str) {
        if let Some(f) = &mut self.file {
            let _ = writeln!(f, "[{:7.1} s] {text}", self.t0.elapsed().as_secs_f64());
        }
    }
}

fn send(port: u16, model_id: &str, claude: &Path, log: &mut Log) -> Result<(), Box<dyn Error>> {
    let checks = capture(claude, model_id)?;
    log.line(&format!(
        "captured {} check prompt(s): {} characters of system text",
        checks.len(),
        checks
            .iter()
            .map(|b| system_text(b).len().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    for (i, mut body) in checks.into_iter().enumerate() {
        body["max_tokens"] = serde_json::json!(1);
        body["stream"] = serde_json::json!(false);
        // A background read waits for the GPU to be free; it is not given up on.
        let t = Instant::now();
        ureq::post(&format!("http://127.0.0.1:{port}/v1/messages"))
            .set("content-type", "application/json")
            .set("anthropic-version", "2023-06-01")
            .set("x-arf-background", "1")
            .timeout(Duration::from_secs(3600))
            .send_string(&body.to_string())?;
        log.line(&format!(
            "check {} read by the server in {:.1} s",
            i + 1,
            t.elapsed().as_secs_f64()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_are_told_apart() {
        let long = "x".repeat(30_000);
        let msg = |t: &str| serde_json::json!([{"role": "user", "content": t}]);
        let agent = serde_json::json!({"tools": [{"name": "Bash"}], "messages": []});
        let s1 = serde_json::json!({"system": [{"text": format!("{long} <severity> <block>yes</block>")}],
            "messages": msg("Respond with <severity>N</severity> ONLY.")});
        let s2 = serde_json::json!({"system": format!("{long} <severity> Output <block>"),
            "messages": msg("Review the classification process.")});
        let side = serde_json::json!({"system": "Write a title", "messages": []});
        assert_eq!(kind(&agent), Kind::Agent);
        assert_eq!(kind(&s1), Kind::Check);
        assert_eq!(kind(&side), Kind::Other);
        assert_eq!(reply(&s1).0[0]["text"], "<severity>90</severity>");
        assert_eq!(reply(&s2).0[0]["text"], "<block>no</block>");
        assert_eq!(reply(&agent).1, "tool_use");
        let ran = serde_json::json!({"tools": [{"name": "Bash"}], "messages": [{"role": "user",
            "content": [{"type": "tool_result"}, {"type": "tool_result"}, {"type": "tool_result"}]}]});
        assert_eq!(reply(&ran).0[0]["text"], "DONE");
        // A retry of the first stage is not the second.
        let seen = vec![side, s1.clone(), s1.clone(), agent, s2.clone(), s1.clone()];
        assert_eq!(checks(&seen), vec![s1.clone(), s2]);
        assert_eq!(checks(&[s1.clone(), s1.clone()]), vec![s1]);
    }

    #[test]
    fn a_permission_mode_other_than_auto_turns_it_off() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(!auto_mode(&a(&["--permission-mode", "default"])));
        assert!(!auto_mode(&a(&["--permission-mode=plan"])));
        assert!(!auto_mode(&a(&["--dangerously-skip-permissions"])));
        assert!(auto_mode(&a(&["--permission-mode", "auto"])));
    }
}
