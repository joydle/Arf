//! Chat prompt construction shared by every front end.
//!
//! The serve daemon and the CLI must emit **byte-identical** prompts for the same conversation —
//! a difference of one character changes what the model sees and makes the two paths quietly
//! disagree. They used to hold separate copies of this, each with a comment admitting the
//! duplication ("Duplicated rather than shared because the CLI does not depend on arf-serve").
//! Both depend on `arf-core`, so it lives here and the compiler keeps them in step.

/// Today's UTC date as `YYYY-MM-DD`.
///
/// Howard Hinnant's civil-from-days, rather than pulling a date crate for one format string.
/// A **runtime** value on purpose: baking in a build-time constant would make model output depend
/// on when the binary happened to be compiled.
pub fn today_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let z = secs.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = era * 400 + yoe + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Muse Glimmer's default system turn, WITHOUT the `<|start|>system<|message|>` … `<|eot|>`
/// wrapper — callers that need the wrapper add it.
///
/// Every line is load-bearing. The `Current date` line is rendered from the real clock to match
/// llama.cpp's `strftime_now` branch, so the two engines produce identical prompts for identical
/// messages; the recipients line is what selects the model's answer channel.
pub fn muse_glimmer_system_body() -> String {
    format!(
        "You are a helpful AI assistant.\n\
         Knowledge cutoff: 2026-01-04.\n\
         Current date: {}.\n\n\
         Reasoning strength: high.\n\n\
         # Valid recipients: \"self\", \"user\".",
        today_utc()
    )
}

/// The same system turn, wrapped for direct concatenation into a prompt.
pub fn muse_glimmer_system() -> String {
    format!(
        "<|start|>system<|message|>{}<|eot|>",
        muse_glimmer_system_body()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn today_utc_is_a_plausible_iso_date() {
        let d = today_utc();
        assert_eq!(d.len(), 10, "{d}");
        let (y, rest) = d.split_at(4);
        let year: i32 = y.parse().expect("year");
        assert!((2020..2100).contains(&year), "{d}");
        assert!(rest.starts_with('-') && rest[3..4] == *"-", "{d}");
    }

    /// The wrapped form must be exactly the body inside the markers — this is the invariant that
    /// used to be two separate string literals in two crates.
    #[test]
    fn wrapped_system_turn_contains_the_body_verbatim() {
        let body = muse_glimmer_system_body();
        let wrapped = muse_glimmer_system();
        assert_eq!(wrapped, format!("<|start|>system<|message|>{body}<|eot|>"));
    }
}
