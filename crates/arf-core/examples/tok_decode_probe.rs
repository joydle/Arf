//! TEMP probe: compare batched vs per-token decode for the gemma tokenizer.
use tokenizers::Tokenizer;

fn main() {
    // L264 — ARF_TOKENIZER=<path/to/tokenizer.json>, defaulting to a repo-relative model.
    let path = std::env::var("ARF_TOKENIZER")
        .unwrap_or_else(|_| "models/gemma-4-31b-it/tokenizer.json".to_string());
    let t = Tokenizer::from_file(path).expect("load");
    let samples = [
        "</title>",
        "background-color",
        "<!DOCTYPE html>",
        "--bg:",
        "</div></div></div>",
        "align-items",
    ];
    for s in samples {
        let enc = t.encode(s, false).expect("encode");
        let ids = enc.get_ids().to_vec();
        let toks = enc.get_tokens().to_vec();
        let batched = t.decode(&ids, true).unwrap_or_default();
        let per: String = ids
            .iter()
            .map(|&i| t.decode(&[i], true).unwrap_or_default())
            .collect();
        println!("INPUT  {:?}", s);
        println!("  ids    {:?}", ids);
        println!("  tokens {:?}", toks);
        println!("  batched   {:?}", batched);
        println!(
            "  pertoken  {:?}{}",
            per,
            if per != batched {
                "   <<< MISMATCH"
            } else {
                ""
            }
        );
    }
}
