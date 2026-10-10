//! Is the tokenizer Arf BUILDS from a GGUF canonical? Encode the same text with it and with a
//! reference `tokenizer.json` for the same model, and report the first divergence.
//!
//! Found 2026-09-20 through prompt-lookup speculation: drafts copied from the PROMPT were
//! rejected at single tokens where the text was identical (prompt `13` = "." where the model
//! emits a merged token) — i.e. the prompt's segmentation is not the one the model generates.
//! A non-canonical prompt tokenization costs every request a little quality, not just speculation.
//!
//! Run: cargo run --release -p arf-core --example tok_canonical_check -- <model.gguf> <tokenizer.json>
use tokenizers::Tokenizer;

const SAMPLES: &[&str] = &[
    "    let raw_text = std::fs::read_to_string(path).map_err(ConfigError::Io)?;",
    "        let line = line.trim();\n        if line.is_empty() || line.starts_with('#') { continue; }",
    "fn parse_config(path: &str) -> Result<Config, ConfigError> {",
    "\"port\" => cfg.port = value.trim().parse().map_err(|_| ConfigError::Value(line_no))?,",
    "The harbour master, Oswin Trelawney, kept a ledger of every boat that entered after dusk.",
];

fn main() {
    let mut a = std::env::args().skip(1);
    let gguf = a.next().expect("model.gguf");
    let reference = a.next().expect("tokenizer.json");
    let ours = arf_core::tokenizer::Tokenizer::from_gguf_path(&gguf).expect("gguf tokenizer");
    let theirs = Tokenizer::from_file(reference).expect("reference tokenizer");
    // Any further arguments are FILES, compared whole — a corpus check, not five lines.
    let files: Vec<String> = a.collect();
    let owned: Vec<String> = files
        .iter()
        .map(|f| std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{f}: {e}")))
        .collect();
    let mut bad = 0;
    let mut total = (0usize, 0usize);
    for s in SAMPLES
        .iter()
        .copied()
        .chain(owned.iter().map(|s| s.as_str()))
    {
        let x = ours.encode(s, false).expect("encode");
        let y = theirs.encode(s, false).expect("encode").get_ids().to_vec();
        total = (total.0 + x.len(), total.1 + y.len());
        if x == y {
            println!(
                "SAME    ({:6} tokens) {:?}",
                x.len(),
                s.chars().take(48).collect::<String>()
            );
            continue;
        }
        bad += 1;
        let i = x
            .iter()
            .zip(&y)
            .position(|(p, q)| p != q)
            .unwrap_or(x.len().min(y.len()));
        println!(
            "DIFFERS ours {} vs reference {} tokens, first at #{i}: {:?}",
            x.len(),
            y.len(),
            s.chars().take(48).collect::<String>()
        );
        println!(
            "   ours      {:?}",
            &x[i.saturating_sub(2)..(i + 5).min(x.len())]
        );
        println!(
            "   reference {:?} = {:?}",
            &y[i.saturating_sub(2)..(i + 5).min(y.len())],
            y[i.saturating_sub(2)..(i + 5).min(y.len())]
                .iter()
                .map(|&t| theirs.id_to_token(t).unwrap_or_default())
                .collect::<Vec<_>>()
        );
    }
    println!(
        "{bad} of {} texts tokenize differently | tokens ours {} vs reference {}",
        SAMPLES.len() + owned.len(),
        total.0,
        total.1
    );
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
