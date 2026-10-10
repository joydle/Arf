//! The background `arf-serve` that `arf run` and `arf launch` talk to.
//!
//! `arf run <model>` used to load the model inside the CLI on a simple path: bf16 weights (a 27B
//! GGUF does not fit a 36 GB Mac that way), no chat template, no history, no draft. Everything the
//! engine is good at lives in `arf-serve`. So the CLI starts one in the background — its own process
//! group, so a Ctrl-C in the chat does not take it down — and talks to it over HTTP; the next `arf
//! run` finds it warm. `arf stop` ends it.
//!
//! State: `~/.arf/run/serve-<port>.json` names the pid and the model of the server this CLI started;
//! its log is `~/.arf/logs/serve-<port>.log`. A server someone else started on the port is used only
//! if it already serves the model asked for.

use std::error::Error;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::ui::Ui;

/// The port `arf run` / `arf launch` use: `ARF_PORT`, else arf-serve's default.
pub fn port() -> u16 {
    std::env::var("ARF_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080)
}

pub(crate) fn arf_home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".arf")
}

fn state_file(port: u16) -> PathBuf {
    arf_home().join("run").join(format!("serve-{port}.json"))
}

pub fn log_file(port: u16) -> PathBuf {
    arf_home().join("logs").join(format!("serve-{port}.log"))
}

/// A server this CLI started: its pid, the model directory it serves and the architecture it was
/// given (`arf run --arch`), so a restart of it starts the same thing.
#[derive(Debug, PartialEq)]
struct Started {
    pid: u32,
    model: PathBuf,
    arch: Option<String>,
}

fn read_state(port: u16) -> Option<Started> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(state_file(port)).ok()?).ok()?;
    Some(Started {
        pid: v["pid"].as_u64()? as u32,
        model: PathBuf::from(v["model"].as_str()?),
        arch: v["arch"].as_str().map(str::to_string),
    })
}

fn write_state(port: u16, s: &Started) -> std::io::Result<()> {
    let path = state_file(port);
    std::fs::create_dir_all(path.parent().expect("has a parent"))?;
    let v = serde_json::json!({"pid": s.pid, "model": s.model.to_string_lossy(), "arch": s.arch});
    std::fs::write(path, v.to_string())
}

pub(crate) fn alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// The model ids the server on `port` lists, or `None` when nothing answers there.
pub fn probe(port: u16) -> Option<Vec<String>> {
    Some(models(port)?.into_iter().map(|(id, _)| id).collect())
}

/// `(id, context window)` per model the server on `port` lists (`/v1/models`, `max_model_len`).
fn models(port: u16) -> Option<Vec<(String, Option<u64>)>> {
    let resp = ureq::get(&format!("http://127.0.0.1:{port}/v1/models"))
        .timeout(Duration::from_secs(2))
        .call()
        .ok()?;
    let v: serde_json::Value = serde_json::from_str(&resp.into_string().ok()?).ok()?;
    Some(
        v["data"]
            .as_array()?
            .iter()
            .filter_map(|m| Some((m["id"].as_str()?.to_string(), m["max_model_len"].as_u64())))
            .collect(),
    )
}

/// A running server: its port, the model id to put in requests, and its context window.
pub struct Server {
    pub port: u16,
    pub model_id: String,
    pub context: Option<u64>,
}

impl Server {
    fn on(port: u16) -> Server {
        let (model_id, context) = models(port)
            .and_then(|m| m.into_iter().next())
            .unwrap_or_else(|| ("local".into(), None));
        Server {
            port,
            model_id,
            context,
        }
    }
}

/// Make sure a server for `model` (a resolved local model directory or file) answers on the port:
/// reuse the one this CLI started for the same model, or start one and wait until it is ready.
/// `arch`: an architecture for a model whose header and folder do not name one (`arf run
/// --arch`); `None` lets the server read it from the model.
pub fn ensure(model: &Path, arch: Option<&str>, ui: Ui) -> Result<Server, Box<dyn Error>> {
    let port = port();
    let canon = model.canonicalize().unwrap_or_else(|_| model.to_path_buf());
    let ours = read_state(port).filter(|s| alive(s.pid));
    if let Some(ids) = probe(port) {
        match &ours {
            Some(s) if s.model == canon && (arch.is_none() || s.arch.as_deref() == arch) => {
                return Ok(Server::on(port))
            }
            Some(s) => {
                let running = s
                    .model
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let interactive = std::io::IsTerminal::is_terminal(&std::io::stdin())
                    && std::io::IsTerminal::is_terminal(&std::io::stderr());
                if !interactive {
                    return Err(format!(
                        "the background server on port {port} serves {running} — stop it first: arf stop"
                    )
                    .into());
                }
                eprint!(
                    "{running} is running in the background. Switch to {}? [Y/n] ",
                    canon
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default()
                );
                std::io::stderr().flush()?;
                let mut answer = String::new();
                std::io::stdin().read_line(&mut answer)?;
                if !matches!(
                    answer.trim().to_ascii_lowercase().as_str(),
                    "" | "y" | "yes"
                ) {
                    return Err("kept the running model".into());
                }
                // Both models rarely fit at once, so the running one has to stop first — but when
                // the new one then fails to load, bring the old one back rather than leave the user
                // with nothing (a teammate's `arf run gemma3:4b` did exactly that, 2026-10-06).
                let (previous, previous_arch) = (s.model.clone(), s.arch.clone());
                stop(ui)?;
                return match start(&canon, arch, port, ui) {
                    Ok(server) => Ok(server),
                    Err(e) => {
                        eprintln!("{e}");
                        eprintln!(
                            "{} {} did not start — restarting {running}",
                            ui.accent("●"),
                            canon
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default()
                        );
                        match start(&previous, previous_arch.as_deref(), port, ui) {
                            Ok(_) => {
                                Err(format!("{running} is serving again on port {port}").into())
                            }
                            Err(e2) => {
                                Err(format!("{running} did not restart either: {e2}").into())
                            }
                        }
                    }
                };
            }
            None => {
                return Err(format!(
                    "port {port} is taken by a server arf did not start (it lists {ids:?})\n  \
                     → stop it, or use another port:  ARF_PORT={} arf …",
                    port + 1
                )
                .into())
            }
        }
    }
    start(&canon, arch, port, ui)
}

fn start(model: &Path, arch: Option<&str>, port: u16, ui: Ui) -> Result<Server, Box<dyn Error>> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let serve = crate::serve_shim::locate_serve(exe_dir.as_deref())
        .ok_or("could not find the arf-serve binary next to arf")?;
    let log = log_file(port);
    std::fs::create_dir_all(log.parent().expect("has a parent"))?;
    let out = std::fs::File::create(&log)?;
    let mut cmd = std::process::Command::new(&serve);
    // The background server re-reads the agent prompt it served last (`anchor_replay`) as soon as
    // it is up: an agent's first message sends ~20K tokens of instructions and
    // tools, minutes of prefill, and the replay starts that while the user is still typing. Off
    // for a bare `arf-serve` (a replay queues ahead of unrelated requests); `ARF_ANCHOR_REPLAY=0`
    // turns it off here too.
    if std::env::var_os("ARF_ANCHOR_REPLAY").is_none() {
        cmd.env("ARF_ANCHOR_REPLAY", "1");
    }
    cmd.arg("--model").arg(model);
    if let Some(a) = arch {
        cmd.args(["--arch", a]);
    }
    cmd.args(["--port", &port.to_string()])
        .stdin(std::process::Stdio::null())
        .stdout(out.try_clone()?)
        .stderr(out);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0); // a Ctrl-C in the chat must not reach the server
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not start {}: {e}", serve.display()))?;
    write_state(
        port,
        &Started {
            pid: child.id(),
            model: model.to_path_buf(),
            arch: arch.map(str::to_string),
        },
    )?;
    let name = model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let t0 = Instant::now();
    let tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
    loop {
        if let Some(status) = child.try_wait()? {
            let _ = std::fs::remove_file(state_file(port));
            return Err(format!(
                "arf-serve stopped while loading {name} ({status}). Its log, {}:\n{}",
                log.display(),
                tail(&log, 12)
            )
            .into());
        }
        if probe(port).is_some() {
            spawn_watch(port, child.id());
            if tty {
                eprint!("\r\x1b[2K");
            }
            eprintln!(
                "{} {} ready in {:.0}s {}",
                ui.accent("●"),
                name,
                t0.elapsed().as_secs_f64(),
                ui.faint(&format!(
                    "· keeps running in the background (arf stop) · log {}",
                    log.display()
                ))
            );
            return Ok(Server::on(port));
        }
        if t0.elapsed() > Duration::from_secs(1800) {
            return Err(format!(
                "arf-serve did not become ready in 30 min ({})",
                log.display()
            )
            .into());
        }
        if tty {
            let phase = phase(&log);
            eprint!(
                "\r\x1b[2K{} starting {} … {:.0}s {}",
                ui.accent("◐"),
                name,
                t0.elapsed().as_secs_f64(),
                ui.faint(&phase)
            );
            let _ = std::io::stderr().flush();
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Start the watcher of a background server that is up (`watch`), detached from this terminal.
/// `ARF_NO_WATCH=1` leaves the server unwatched.
fn spawn_watch(port: u16, pid: u32) {
    if std::env::var_os("ARF_NO_WATCH").is_some() {
        return;
    }
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.args([
        "watch",
        "--port",
        &port.to_string(),
        "--pid",
        &pid.to_string(),
    ])
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

/// Restarts allowed in [`RESTART_WINDOW`] before the watcher gives up: a server that crashes on
/// every start (memory, a bad file) is not restarted for ever.
const MAX_RESTARTS: usize = 5;
const RESTART_WINDOW: Duration = Duration::from_secs(600);

/// `arf watch` (internal): keep the background server on `port` running. RESILIENCE
/// (2026-10-08): an agent session lost its server for good when `arf-serve` ended — a model-thread
/// panic, the memory killer, a GPU fault — and saw "connection refused" until someone ran an `arf`
/// command. The watcher polls `pid`; when it ends while the state file still names it (a stop
/// removes the file first, and so does a clean shutdown), it starts the same model again on the
/// same port. The saved prompt states load from disk, so a restart costs the start, not the
/// session. Returns when the server is stopped, replaced, or restarted (the new one has its own
/// watcher), or after [`MAX_RESTARTS`] in [`RESTART_WINDOW`].
pub fn watch(port: u16, pid: u32) -> Result<(), Box<dyn Error>> {
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let Some(s) = read_state(port).filter(|s| s.pid == pid) else {
            return Ok(()); // stopped on purpose, or another server took the port
        };
        if alive(pid) {
            continue;
        }
        let log = log_file(port);
        // The note goes at the end of the ended server's log, kept as `serve-PORT.crashed.log`:
        // the restarted server writes the live log from its own file offset, over anything
        // appended to it. `arf status` reports the restarts.
        let crashed = log.with_extension("crashed.log");
        let _ = std::fs::copy(&log, &crashed);
        let note = |line: &str| {
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&crashed) {
                let _ = writeln!(f, "{line}");
            }
        };
        let history = arf_home()
            .join("run")
            .join(format!("serve-{port}.restarts"));
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let mut recent: Vec<u64> = std::fs::read_to_string(&history)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| l.trim().parse().ok())
            .filter(|&t: &u64| now.saturating_sub(t) < RESTART_WINDOW.as_secs())
            .collect();
        if recent.len() >= MAX_RESTARTS {
            note(&format!(
                "[watch] arf-serve (pid {pid}) ended without a stop; it already restarted \
                 {MAX_RESTARTS} times in {} min, so it is left stopped",
                RESTART_WINDOW.as_secs() / 60
            ));
            let _ = std::fs::remove_file(state_file(port));
            return Ok(());
        }
        recent.push(now);
        let _ = std::fs::write(
            &history,
            recent.iter().map(|t| format!("{t}\n")).collect::<String>(),
        );
        note(&format!(
            "[watch] arf-serve (pid {pid}) ended without a stop; starting it again ({} of \
             {MAX_RESTARTS} in {} min)",
            recent.len(),
            RESTART_WINDOW.as_secs() / 60,
        ));
        let _ = std::fs::remove_file(state_file(port));
        start(&s.model, s.arch.as_deref(), port, Ui::from_env(true))?;
        return Ok(());
    }
}

/// What the server is doing, from its log: the last line naming a loading step, shortened.
fn phase(log: &Path) -> String {
    let Ok(f) = std::fs::File::open(log) else {
        return String::new();
    };
    let mut last = String::new();
    for line in std::io::BufReader::new(f).lines().map_while(Result::ok) {
        let l = line.to_ascii_lowercase();
        if l.contains("loading")
            || l.contains("draft attached")
            || l.contains("warm-up")
            || l.contains("transcod")
        {
            last = line;
        }
    }
    let mut s: String = last.chars().take(60).collect();
    if last.chars().count() > 60 {
        s.push('…');
    }
    s
}

fn tail(log: &Path, n: usize) -> String {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// `arf status`: what the background server is doing (`GET /v1/arf/status`).
pub fn status(ui: Ui) -> Result<(), Box<dyn Error>> {
    let port = port();
    let Some(s) = read_state(port).filter(|s| alive(s.pid)) else {
        println!(
            "{}",
            ui.dim(&format!("no background server on port {port}"))
        );
        return Ok(());
    };
    let url = format!("http://127.0.0.1:{port}/v1/arf/status");
    let reply = ureq::get(&url)
        .timeout(Duration::from_secs(3))
        .call()
        .map_err(|e| e.to_string())
        .and_then(|r| r.into_string().map_err(|e| e.to_string()));
    let v: serde_json::Value = match reply {
        Ok(body) => serde_json::from_str(&body)?,
        Err(e) => {
            println!(
                "{} {} is starting or busy loading ({e}); its log: {}",
                ui.accent("●"),
                s.model.display(),
                log_file(port).display()
            );
            return Ok(());
        }
    };
    let (running, waiting) = (
        v["running"].as_u64().unwrap_or(0),
        v["waiting"].as_u64().unwrap_or(0),
    );
    println!(
        "{} {} on :{port} · {running} running · {waiting} waiting",
        ui.accent("●"),
        ui.bold(v["model"].as_str().unwrap_or("?"))
    );
    if let Some(gb) = v["footprint_gb"].as_f64() {
        println!("  memory: {gb:.1} GB of its own, plus the weights paged from the model file");
    }
    for p in v["prefill"].as_array().into_iter().flatten() {
        let (done, total) = (
            p["done"].as_u64().unwrap_or(0),
            p["total"].as_u64().unwrap_or(1),
        );
        let (rate, eta) = (
            p["tok_per_s"].as_f64().unwrap_or(0.0),
            p["eta_s"].as_f64().unwrap_or(0.0),
        );
        let left = if rate <= 0.0 {
            "measuring".to_string()
        } else if eta >= 90.0 {
            format!("~{:.1} min left", eta / 60.0)
        } else {
            format!("~{eta:.0} s left")
        };
        println!(
            "  {}: {done} / {total} tokens ({:.0}%) · {rate:.0} tok/s · {left}",
            if p["replay"].as_bool().unwrap_or(false) {
                "re-reading the last agent prompt (background)"
            } else {
                "reading a prompt"
            },
            100.0 * done as f64 / total.max(1) as f64
        );
    }
    if v["low_power"].as_bool().unwrap_or(false) {
        println!(
            "  {} Low Power Mode is on: the GPU runs slower (System Settings > Battery)",
            ui.accent("⚠")
        );
    }
    if matches!(v["thermal"].as_str(), Some("serious" | "critical")) {
        println!(
            "  {} the Mac is hot: the GPU may be throttled",
            ui.accent("⚠")
        );
    }
    if running == 0 && waiting == 0 {
        println!("  {}", ui.dim("idle"));
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let restarts = std::fs::read_to_string(
        arf_home()
            .join("run")
            .join(format!("serve-{port}.restarts")),
    )
    .unwrap_or_default()
    .lines()
    .filter_map(|l| l.trim().parse::<u64>().ok())
    .filter(|&t| now.saturating_sub(t) < 3600)
    .count();
    if restarts > 0 {
        println!(
            "  {} the server ended without a stop {restarts} time(s) in the last hour and was \
             started again; the log of the last one is {}",
            ui.accent("⚠"),
            log_file(port).with_extension("crashed.log").display()
        );
    }
    Ok(())
}

/// `arf stop`: end the background server this CLI started on the port.
pub fn stop(ui: Ui) -> Result<(), Box<dyn Error>> {
    let port = port();
    let Some(s) = read_state(port).filter(|s| alive(s.pid)) else {
        let _ = std::fs::remove_file(state_file(port));
        println!(
            "{}",
            ui.dim(&format!("no background server on port {port}"))
        );
        return Ok(());
    };
    // The state file goes FIRST: its watcher (`watch`) restarts a server that ends while the file
    // still names it, and this one is being stopped on purpose.
    let _ = std::fs::remove_file(state_file(port));
    std::process::Command::new("kill")
        .args(["-INT", &s.pid.to_string()])
        .status()?;
    let t0 = Instant::now();
    while alive(s.pid) && t0.elapsed() < Duration::from_secs(30) {
        std::thread::sleep(Duration::from_millis(200));
    }
    if alive(s.pid) {
        std::process::Command::new("kill")
            .args(["-KILL", &s.pid.to_string()])
            .status()?;
    }
    let _ = std::fs::remove_file(state_file(port));
    println!(
        "{} stopped the server for {}",
        ui.accent("●"),
        s.model.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_round_trips_and_a_dead_pid_is_not_alive() {
        let s = Started {
            pid: 4_000_000, // above any real pid on macOS
            model: PathBuf::from("/tmp/m"),
            arch: None,
        };
        assert!(!alive(s.pid));
        let v = serde_json::json!({"pid": s.pid, "model": "/tmp/m"});
        let back: serde_json::Value = serde_json::from_str(&v.to_string()).unwrap();
        assert_eq!(back["pid"].as_u64(), Some(4_000_000));
        assert!(alive(std::process::id()));
    }

    #[test]
    fn phase_picks_the_last_loading_step() {
        let p = std::env::temp_dir().join(format!("arf_phase_{}.log", std::process::id()));
        std::fs::write(
            &p,
            "HF_TOKEN: unset\nloading weights from x\nblock draft attached: d (12.8 s)\nother\n",
        )
        .unwrap();
        assert_eq!(phase(&p), "block draft attached: d (12.8 s)");
        let _ = std::fs::remove_file(&p);
    }
}
