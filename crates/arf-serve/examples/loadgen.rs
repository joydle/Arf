//! Concurrency-sweep load generator — the the design acceptance gate.
//!
//! Drives `POST /v1/completions` (stream:true) at fixed concurrency levels and
//! reports aggregate output tok/s, mean TTFT, and p50/p99 inter-token gaps.
//! Run the same sweep against Arf, ollama, and vLLM on the SAME box at a
//! MATCHED KV budget. Machine discipline: AC power, Low Power
//! Mode off, no other load (the project's rules).
//!
//! Usage:
//!   cargo run -p arf-serve --release --example loadgen -- \
//!     --url http://127.0.0.1:8080 --levels 1,4,16,32 --requests 32 --max-tokens 64

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;

struct Args {
    url: String,
    levels: Vec<usize>,
    requests: usize,
    max_tokens: usize,
    prompt: String,
    /// A long shared preamble prepended to EVERY request (e.g. a system prompt /
    /// few-shot block). Each request still gets a unique tail, so outputs differ
    /// but the prefix is identical — the workload automatic prefix caching wins.
    /// `--shared-prefix-words N` synthesizes one of N repeated words.
    shared_prefix: String,
    /// OpenAI `model` field. Required by ollama's /v1 endpoint; Arf ignores
    /// it (defaults to the resident model). Empty = omit the field.
    model: String,
}

fn parse_args() -> Args {
    let mut a = Args {
        url: "http://127.0.0.1:8080".into(),
        levels: vec![1, 4, 16, 32],
        requests: 32,
        max_tokens: 64,
        prompt: "Write a long, detailed explanation of how rivers form.".into(),
        shared_prefix: String::new(),
        model: String::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let val = it.next().unwrap_or_default();
        match flag.as_str() {
            "--url" => a.url = val,
            "--levels" => a.levels = val.split(',').map(|s| s.parse().unwrap()).collect(),
            "--requests" => a.requests = val.parse().unwrap(),
            "--max-tokens" => a.max_tokens = val.parse().unwrap(),
            "--prompt" => a.prompt = val,
            "--shared-prefix" => a.shared_prefix = val,
            "--shared-prefix-words" => {
                let n: usize = val.parse().unwrap();
                a.shared_prefix = std::iter::repeat_n("context", n)
                    .collect::<Vec<_>>()
                    .join(" ");
            }
            "--model" => a.model = val,
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

/// One request: returns (ttft, inter-token gaps, token count).
async fn one_request(
    client: &reqwest::Client,
    url: &str,
    prompt: &str,
    max_tokens: usize,
    model: &str,
) -> Result<(Duration, Vec<Duration>, usize), reqwest::Error> {
    let mut body = serde_json::json!({
        "prompt": prompt, "max_tokens": max_tokens, "stream": true
    });
    if !model.is_empty() {
        body["model"] = serde_json::Value::String(model.to_string());
    }
    let start = Instant::now();
    let resp = client
        .post(format!("{url}/v1/completions"))
        .json(&body)
        .send()
        .await?;
    let mut stream = resp.bytes_stream();
    let mut ttft = None;
    let mut gaps = Vec::new();
    let mut last = start;
    let mut tokens = 0usize;
    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        buf.push_str(&String::from_utf8_lossy(&chunk?));
        // SSE events are separated by blank lines; each data line is one chunk.
        while let Some(pos) = buf.find("\n\n") {
            let event: String = buf.drain(..pos + 2).collect();
            for line in event.lines() {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
                    continue;
                };
                let text = v["choices"][0]["text"].as_str().unwrap_or("");
                let is_finish = !v["choices"][0]["finish_reason"].is_null();
                if !text.is_empty() && !is_finish {
                    let now = Instant::now();
                    match ttft {
                        None => ttft = Some(now - start),
                        Some(_) => gaps.push(now - last),
                    }
                    last = now;
                    tokens += 1;
                }
            }
        }
    }
    Ok((ttft.unwrap_or_else(|| start.elapsed()), gaps, tokens))
}

fn pct(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    sorted[((sorted.len() - 1) as f64 * p) as usize]
}

#[tokio::main]
async fn main() {
    let args = Arc::new(parse_args());
    let client = reqwest::Client::new();
    println!("level | reqs | out tok/s | mean TTFT | p50 gap | p99 gap");
    for &level in &args.levels {
        let wall = Instant::now();
        let mut handles = Vec::new();
        let sem = Arc::new(tokio::sync::Semaphore::new(level));
        for i in 0..args.requests {
            let (c, a, s) = (client.clone(), args.clone(), sem.clone());
            handles.push(tokio::spawn(async move {
                let _permit = s.acquire().await.unwrap();
                // [shared preamble][per-request unique tail]: the prefix is reused
                // across requests, the tail makes each request distinct.
                let prompt = if a.shared_prefix.is_empty() {
                    a.prompt.clone()
                } else {
                    format!("{} Question {i}: {}", a.shared_prefix, a.prompt)
                };
                one_request(&c, &a.url, &prompt, a.max_tokens, &a.model).await
            }));
        }
        let mut ttfts = Vec::new();
        let mut gaps = Vec::new();
        let mut total_tokens = 0usize;
        for h in handles {
            if let Ok(Ok((t, g, n))) = h.await {
                ttfts.push(t);
                gaps.extend(g);
                total_tokens += n;
            }
        }
        let secs = wall.elapsed().as_secs_f64();
        gaps.sort_unstable();
        let mean_ttft = ttfts.iter().sum::<Duration>() / ttfts.len().max(1) as u32;
        println!(
            "{level:>5} | {:>4} | {:>9.1} | {:>9.0?} | {:>7.0?} | {:>7.0?}",
            args.requests,
            total_tokens as f64 / secs,
            mean_ttft,
            pct(&gaps, 0.50),
            pct(&gaps, 0.99),
        );
    }
}
