//! Dump a GGUF's token vocabulary as JSON `{token_id: text}` — so log analysers (which run in
//! python and have no GGUF reader) can turn token IDs back into text.
//! usage: dump_vocab <model.gguf> [out.json]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let path = a
        .next()
        .ok_or("usage: dump_vocab <model.gguf> [out.json]")?;
    let out = a.next().unwrap_or_else(|| "vocab.json".into());
    let meta = arf_core::model::gguf::read_tokenizer_meta(std::path::Path::new(&path))?;
    let mut s = String::from("{");
    for (i, t) in meta.tokens.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\"{i}\":{}", serde_json::to_string(t)?));
    }
    s.push('}');
    std::fs::write(&out, s)?;
    eprintln!("{} tokens -> {out}", meta.tokens.len());
    Ok(())
}
