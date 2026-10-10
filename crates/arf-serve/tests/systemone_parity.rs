//! System One parity gate (issue #11): Arf's `/v1/systemone` pipeline against llama.cpp's own
//! server, on the same GGUF and the same requests.
//!
//! The reference was recorded by `scripts/probes/systemone_parity.py record` from llama.cpp's
//! `llama-server` at 050439614 (System One landed in a4cb4c61f), Metal build, one slot, on the tiny openjev
//! test model `ggml-org/tinyopenjev-for-testing-gguf` (`tinyopenjev-for-testing-Q8_0.gguf`,
//! ~45 MB). That model is licensed **CC-BY-NC-4.0** (non-commercial): download it yourself, it
//! is never committed here.
//!
//! Per case this checks: the rendered prompt equals the one llama.cpp rendered (read back from
//! its verbose log), the token ids are identical to llama.cpp's `/tokenize` of it, the token
//! count equals llama.cpp's `usage.input_tokens`, and every probability, `score`, `noul` and
//! `confidence` is within 1e-3 of llama.cpp's. The forward pass is Arf's CPU reference
//! (`forward_cpu_reference`): the GPU path serves only the 27B geometry. Invalid requests must
//! fail with llama.cpp's status and message.
//!
//! Run (ignored by default, needs the model):
//!
//! ```sh
//! ARF_TEST_TINYOPENJEV_GGUF=/path/to/tinyopenjev-for-testing-Q8_0.gguf \
//!     cargo test -p arf-serve --test systemone_parity -- --ignored --nocapture
//! ```
#![cfg(feature = "wgpu")]

use std::path::{Path, PathBuf};

use arf_core::config::ModelConfig;
use arf_core::model::gguf::LazyGguf;
use arf_core::Tokenizer;
use arf_gpu::gpu::ssm_qwen35::{forward_cpu_reference, read_hparams};
use arf_serve::http::systemone::{answer, probabilities, DecisionModel};

const TOL: f64 = 1e-3;

fn probes_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/probes")
}

fn load_json(name: &str) -> serde_json::Value {
    let p = probes_dir().join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p:?}: {e}")))
        .unwrap()
}

/// The largest difference over the numeric leaves of a reference answer and the same paths in
/// ours; any other leaf (a `choice`, a `legend` entry) that differs is pushed to `diffs`.
fn max_delta(
    reference: &serde_json::Value,
    ours: &serde_json::Value,
    path: &str,
    diffs: &mut Vec<String>,
) -> f64 {
    match reference {
        serde_json::Value::Number(r) => match ours.as_f64() {
            Some(o) => (r.as_f64().unwrap() - o).abs(),
            None => {
                diffs.push(format!("{path}: missing"));
                f64::INFINITY
            }
        },
        serde_json::Value::Object(m) => m
            .iter()
            .map(|(k, v)| max_delta(v, &ours[k], &format!("{path}.{k}"), diffs))
            .fold(0.0, f64::max),
        _ => {
            if reference != ours {
                diffs.push(format!("{path}: {ours} vs llama.cpp {reference}"));
            }
            0.0
        }
    }
}

#[test]
#[ignore = "needs the tiny openjev GGUF (CC-BY-NC-4.0): set ARF_TEST_TINYOPENJEV_GGUF"]
fn systemone_parity_tiny() {
    let path = std::env::var_os("ARF_TEST_TINYOPENJEV_GGUF")
        .expect("set ARF_TEST_TINYOPENJEV_GGUF to tinyopenjev-for-testing-Q8_0.gguf");
    let path = Path::new(&path);
    let g = LazyGguf::open_raw(path).expect("open GGUF");
    let tok = Tokenizer::from_gguf_path(path).expect("tokenizer");
    let model = DecisionModel::from_gguf(&g, &tok)
        .expect("decision model")
        .expect("the GGUF has decision metadata");
    let hp = read_hparams(&g, &ModelConfig::qwen35_27b()).expect("hparams");

    let requests = load_json("systemone_requests.json");
    let reference = load_json("systemone_reference.json");
    let mut failures = Vec::new();
    println!(
        "{:<24} {:>6} {:>7} {:>10}  checks",
        "case", "status", "tokens", "max|dp|"
    );
    let ordered: Vec<minijinja::Value> = serde_json::from_str(
        &std::fs::read_to_string(probes_dir().join("systemone_requests.json")).unwrap(),
    )
    .unwrap();
    let cases = requests
        .as_array()
        .unwrap()
        .iter()
        .zip(reference.as_array().unwrap());
    for (idx, (case, r)) in cases.enumerate() {
        let name = case["name"].as_str().unwrap();
        assert_eq!(r["name"], name, "request and reference files out of step");
        let status = r["status"].as_u64().unwrap();
        let body = match case.get("raw_body") {
            Some(raw) => raw.as_str().unwrap().as_bytes().to_vec(),
            // `serde_json::Value` sorts object keys; the request goes out in the file's order,
            // as the probe sends it (re-read order-preserving).
            None => serde_json::to_vec(&ordered[idx].get_attr("request").unwrap()).unwrap(),
        };
        let parsed = model.parse(&body);
        if status != 200 {
            // 400: the same message as llama.cpp (a JSON syntax error's text is the parser's own);
            // 501: images, which validated and are not supported yet.
            let ok = match (&parsed, status) {
                (Err(e), 400) => {
                    e.status.as_u16() == 400
                        && (name == "err_malformed_json"
                            || e.message == r["response"]["error"]["message"])
                }
                (Ok(p), 501) => p.n_images > 0,
                _ => false,
            };
            println!(
                "{name:<24} {status:>6} {:>7} {:>10}  {}",
                "-",
                "-",
                if ok { "ok" } else { "FAIL" }
            );
            if !ok {
                failures.push(format!("{name}: expected {status}, got {parsed:?}"));
            }
            continue;
        }
        let req = parsed.unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let mut n_tokens = 0;
        let mut answers = serde_json::Map::new();
        let mut notes = Vec::new();
        for (i, q) in req.questions.iter().enumerate() {
            let prompt = model.render(&req.state, q).unwrap();
            if prompt != r["prompts"][i].as_str().unwrap() {
                notes.push(format!("prompt[{i}] text differs"));
                failures.push(format!(
                    "{name}: prompt {i} differs\n--- ours\n{prompt}\n--- llama.cpp\n{}",
                    r["prompts"][i].as_str().unwrap()
                ));
            }
            let ids = tok.encode(&prompt, false).unwrap();
            let want: Vec<u32> = r["prompt_tokens"][i]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap() as u32)
                .collect();
            if ids != want {
                notes.push(format!("prompt[{i}] tokens differ"));
                failures.push(format!("{name}: prompt {i} token ids differ"));
            }
            n_tokens += ids.len();
            let logits = forward_cpu_reference(&g, &hp, &ids).expect("forward");
            let scores: Vec<f32> = model
                .labels_for(q)
                .iter()
                .map(|&t| logits[t as usize])
                .collect();
            let probs = probabilities(&scores, model.temperature(q));
            let a: serde_json::Value = serde_json::to_value(answer(q, &probs)).unwrap();
            answers.insert(q.id.clone(), a);
        }
        let ref_answers = &r["response"]["answers"];
        let order: Vec<&String> = ref_answers.as_object().unwrap().keys().collect();
        if order != answers.keys().collect::<Vec<_>>() {
            failures.push(format!("{name}: answer order differs"));
        }
        let ours = serde_json::Value::Object(answers);
        let mut diffs = Vec::new();
        let dp = max_delta(ref_answers, &ours, name, &mut diffs);
        for d in diffs {
            notes.push(d.clone());
            failures.push(d);
        }
        let want_tokens = r["response"]["usage"]["input_tokens"].as_u64().unwrap() as usize;
        if n_tokens != want_tokens {
            notes.push(format!("{n_tokens} tokens vs {want_tokens}"));
            failures.push(format!(
                "{name}: {n_tokens} tokens, llama.cpp {want_tokens}"
            ));
        }
        if dp > TOL {
            failures.push(format!("{name}: max |dp| {dp:.2e} > {TOL}"));
        }
        let verdict = if notes.is_empty() && dp <= TOL {
            "ok".to_string()
        } else {
            notes.join("; ")
        };
        println!("{name:<24} {status:>6} {n_tokens:>7} {dp:>10.2e}  {verdict}");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
