//! Cross-engine correctness verifier — "two engines may diverge, but not by much".
//!
//! The wrong bar is *bit-identical output across engines*: different GEMM kernels,
//! reduction orders, and quant schemes give slightly different logits, so at a
//! near-tie position the greedy argmax legitimately flips and the sequences part
//! ways. That is EXPECTED, not a bug. The right questions are (a) how long do two
//! engines agree under greedy decoding, and (b) when they first differ, is it a
//! benign near-tie or a real discrepancy?
//!
//! This driver answers both against any two OpenAI `/v1/completions` servers
//! (Arf, ollama, vLLM, …) on a shared prompt set, greedy (temperature 0):
//!   1. GREEDY COMMON PREFIX — tokens (or chars, if an engine hides token ids) that
//!      match before the first divergence. Long prefix = healthy.
//!   2. DIVERGENCE CLASSIFICATION — at the first differing token, if both engines
//!      expose `logprobs`, compare each engine's logprob for ITS pick vs the OTHER
//!      engine's pick. A small gap (both near-tied) ⇒ BENIGN; a large gap ⇒ CONCERNING.
//!   3. DETERMINISM — `--self-check` runs one engine twice; greedy must be identical.
//!
//! Logit-level teacher-forced KL/top-k agreement is the gold standard but needs the
//! server to expose per-position logits (an Arf feature TODO).
//!
//! Usage:
//!   cargo run -p arf-serve --release --example verify -- \
//!     --a http://127.0.0.1:8080 --b http://127.0.0.1:11434 --b-model llama3.2:1b \
//!     --max-tokens 64
//!   cargo run -p arf-serve --release --example verify -- \
//!     --a http://127.0.0.1:8080 --self-check --max-tokens 64

use serde_json::Value;

struct Args {
    a: String,
    a_model: String,
    b: String,
    b_model: String,
    max_tokens: usize,
    self_check: bool,
    /// |Δlogprob| below which a token flip counts as a benign near-tie (nats).
    tie_eps: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        a: "http://127.0.0.1:8080".into(),
        a_model: String::new(),
        b: "http://127.0.0.1:11434".into(),
        b_model: String::new(),
        max_tokens: 64,
        self_check: false,
        tie_eps: 0.75,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag.as_str() == "--self-check" {
            a.self_check = true;
            continue;
        }
        let val = it.next().unwrap_or_default();
        match flag.as_str() {
            "--a" => a.a = val,
            "--a-model" => a.a_model = val,
            "--b" => a.b = val,
            "--b-model" => a.b_model = val,
            "--max-tokens" => a.max_tokens = val.parse().unwrap(),
            "--tie-eps" => a.tie_eps = val.parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

/// The shared prompt set: a mix of factual (low-entropy, should agree long), open
/// generation (high-entropy, expected to diverge early but benignly), and a couple
/// of code/format prompts (structure-sensitive).
fn prompts() -> Vec<&'static str> {
    vec![
        "The capital of France is",
        "Water boils at a temperature of",
        "1, 2, 3, 4,",
        "The first three prime numbers are",
        "Once upon a time, in a quiet village,",
        "Explain in one sentence why the sky is blue:",
        "def fibonacci(n):",
        "The opposite of hot is",
    ]
}

/// One greedy completion with logprobs requested. Returns (decoded text, tokens,
/// token_logprobs, top_logprobs-per-position). Token/logprob fields are empty when
/// the server does not expose them — the caller falls back to text comparison.
async fn complete(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: usize,
) -> Result<(String, Vec<String>, Vec<f64>, Vec<Vec<(String, f64)>>), String> {
    let mut body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": false,
        "temperature": 0.0,
        "logprobs": 5,
    });
    if !model.is_empty() {
        body["model"] = Value::String(model.to_string());
    }
    let resp = client
        .post(format!("{url}/v1/completions"))
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let v: Value = resp.json().await.map_err(|e| e.to_string())?;
    let choice = &v["choices"][0];
    let text = choice["text"].as_str().unwrap_or("").to_string();

    let lp = &choice["logprobs"];
    let tokens: Vec<String> = lp["tokens"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|t| t.as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default();
    let token_logprobs: Vec<f64> = lp["token_logprobs"]
        .as_array()
        .map(|a| a.iter().map(|t| t.as_f64().unwrap_or(f64::NAN)).collect())
        .unwrap_or_default();
    let top_logprobs: Vec<Vec<(String, f64)>> = lp["top_logprobs"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|m| {
                    m.as_object()
                        .map(|o| {
                            o.iter()
                                .map(|(k, val)| (k.clone(), val.as_f64().unwrap_or(f64::NAN)))
                                .collect()
                        })
                        .unwrap_or_default()
                })
                .collect()
        })
        .unwrap_or_default();
    Ok((text, tokens, token_logprobs, top_logprobs))
}

/// Longest common prefix length over chars.
fn common_prefix_chars(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// Completion-only text: some servers (ollama's `/v1/completions`) ECHO the prompt
/// back in `text`, others (Arf, OpenAI default) return the continuation only.
/// Strip a leading prompt echo, then leading whitespace, so the comparison is over
/// the generated continuation on both sides.
fn completion_only<'t>(text: &'t str, prompt: &str) -> &'t str {
    let t = text.strip_prefix(prompt).unwrap_or(text);
    t.trim_start()
}

/// Longest common prefix length over token strings.
fn common_prefix_tokens(a: &[String], b: &[String]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[tokio::main]
async fn main() {
    let args = parse_args();
    let client = reqwest::Client::new();

    if args.self_check {
        println!(
            "== DETERMINISM self-check ({}): greedy must be identical run-to-run ==",
            args.a
        );
        let mut all_ok = true;
        for p in prompts() {
            let r1 = complete(&client, &args.a, &args.a_model, p, args.max_tokens).await;
            let r2 = complete(&client, &args.a, &args.a_model, p, args.max_tokens).await;
            match (r1, r2) {
                (Ok((t1, ..)), Ok((t2, ..))) => {
                    let ok = t1 == t2;
                    all_ok &= ok;
                    println!(
                        "  [{}] {:?}",
                        if ok { "OK " } else { "DIFF" },
                        &p[..p.len().min(40)]
                    );
                    if !ok {
                        println!("      run1: {:?}", &t1[..t1.len().min(60)]);
                        println!("      run2: {:?}", &t2[..t2.len().min(60)]);
                    }
                }
                _ => {
                    all_ok = false;
                    println!("  [ERR ] {p:?}");
                }
            }
        }
        println!(
            "\n{}",
            if all_ok {
                "DETERMINISM OK — greedy is reproducible."
            } else {
                "NON-DETERMINISTIC — investigate sampling/race."
            }
        );
        return;
    }

    println!("== CROSS-ENGINE greedy agreement ==");
    println!(
        "   A = {} ({})",
        args.a,
        if args.a_model.is_empty() {
            "resident"
        } else {
            &args.a_model
        }
    );
    println!(
        "   B = {} ({})",
        args.b,
        if args.b_model.is_empty() {
            "resident"
        } else {
            &args.b_model
        }
    );
    println!(
        "   greedy (temp=0), {} tokens/prompt, tie-eps={} nats\n",
        args.max_tokens, args.tie_eps
    );

    let mut exact = 0usize;
    let mut n = 0usize;
    let mut prefix_frac_sum = 0.0f64;
    let mut concerning = 0usize;

    for p in prompts() {
        let ra = complete(&client, &args.a, &args.a_model, p, args.max_tokens).await;
        let rb = complete(&client, &args.b, &args.b_model, p, args.max_tokens).await;
        let ((ta, toka, lpa, topa), (tb, tokb, _lpb, topb)) = match (ra, rb) {
            (Ok(x), Ok(y)) => (x, y),
            _ => {
                println!("[ERR ] {p:?} — one engine failed");
                continue;
            }
        };
        n += 1;

        // Normalize away prompt echo (ollama) so we compare continuations.
        let ca = completion_only(&ta, p);
        let cb = completion_only(&tb, p);

        // Prefer token-level comparison when BOTH expose token ids; else chars.
        let have_tokens = !toka.is_empty() && !tokb.is_empty();
        let (prefix, total, unit) = if have_tokens {
            let cp = common_prefix_tokens(&toka, &tokb);
            (cp, toka.len().min(tokb.len()).max(1), "tok")
        } else {
            let cp = common_prefix_chars(ca, cb);
            (
                cp,
                ca.chars().count().min(cb.chars().count()).max(1),
                "char",
            )
        };
        let frac = prefix as f64 / total as f64;
        prefix_frac_sum += frac;
        let is_exact = ca == cb;
        if is_exact {
            exact += 1;
        }

        // Divergence classification (only when both expose logprobs at the flip).
        let mut verdict = String::new();
        if have_tokens && !is_exact && prefix < toka.len() && prefix < tokb.len() {
            let a_pick = &toka[prefix];
            let b_pick = &tokb[prefix];
            // A's logprob for its own pick, and for B's pick (from A's top list).
            let a_self = lpa.get(prefix).copied().unwrap_or(f64::NAN);
            let a_for_b = topa
                .get(prefix)
                .and_then(|m| m.iter().find(|(k, _)| k == b_pick).map(|(_, v)| *v));
            // B's logprob for its own pick, and for A's pick.
            let b_self = topb
                .get(prefix)
                .and_then(|m| m.iter().find(|(k, _)| k == b_pick).map(|(_, v)| *v))
                .unwrap_or(f64::NAN);
            let b_for_a = topb
                .get(prefix)
                .and_then(|m| m.iter().find(|(k, _)| k == a_pick).map(|(_, v)| *v));
            // Gap = how much each engine "regrets" the other's pick. If the other's
            // pick is in-list and within tie_eps, it was a near-tie ⇒ benign.
            let gap_a = a_for_b.map(|x| a_self - x);
            let gap_b = b_for_a.map(|x| b_self - x);
            let benign = matches!(gap_a, Some(g) if g <= args.tie_eps)
                || matches!(gap_b, Some(g) if g <= args.tie_eps);
            if benign {
                verdict = format!(
                    "BENIGN near-tie (A:{a_pick:?} vs B:{b_pick:?}, ΔlpA={}, ΔlpB={})",
                    gap_a.map(|g| format!("{g:.2}")).unwrap_or("n/a".into()),
                    gap_b.map(|g| format!("{g:.2}")).unwrap_or("n/a".into()),
                );
            } else {
                concerning += 1;
                verdict = format!(
                    "CONCERNING (A:{a_pick:?} vs B:{b_pick:?}, ΔlpA={}, ΔlpB={} — other pick {}",
                    gap_a.map(|g| format!("{g:.2}")).unwrap_or("n/a".into()),
                    gap_b.map(|g| format!("{g:.2}")).unwrap_or("n/a".into()),
                    if gap_a.is_none() && gap_b.is_none() {
                        "not in top-5)"
                    } else {
                        "far behind)"
                    },
                );
            }
        } else if !have_tokens && !is_exact {
            let snip = |s: &str| -> String {
                let from: String = s.chars().skip(prefix).take(36).collect();
                from.replace('\n', "⏎")
            };
            verdict = format!(
                "diverge@{prefix}: A…{:?} | B…{:?}  (text-only; no cross-engine logprobs)",
                snip(ca),
                snip(cb),
            );
        } else if is_exact {
            verdict = "exact match".into();
        }

        println!(
            "[{:>5.0}%] {:<3} prefix {}/{} {}  {:?}\n         {}",
            frac * 100.0,
            unit,
            prefix,
            total,
            if is_exact { "==" } else { "≠ " },
            &p[..p.len().min(46)],
            verdict,
        );
    }

    let denom = n.max(1) as f64;
    println!("\n== SUMMARY ==");
    println!("  prompts            : {n}");
    println!("  exact greedy match : {exact}/{n}");
    println!(
        "  mean common prefix : {:.0}% of the shorter output",
        100.0 * prefix_frac_sum / denom
    );
    println!("  concerning diffs   : {concerning}  (high-confidence disagreements — real-bug candidates)");
    println!(
        "\n  Interpretation: divergence at near-ties is EXPECTED across engines (fp/kernel/quant\n  noise flips a coin-flip token). {} ⇐ this is the bar, not bit-identical output.",
        if concerning == 0 {
            "Zero CONCERNING diffs here means the model math agrees"
        } else {
            "CONCERNING diffs mean an engine is computing materially different logits — investigate"
        }
    );
}
