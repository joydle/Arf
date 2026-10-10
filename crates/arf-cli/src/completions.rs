//! `arf completions <shell>` — print a shell completion script via clap_complete.

use clap::CommandFactory;
use clap_complete::Shell;

/// Generate the completion script for `shell` into a String (testable).
pub fn generate_to_string<C: CommandFactory>(shell: Shell, bin: &str) -> String {
    let mut cmd = C::command();
    let mut buf: Vec<u8> = Vec::new();
    clap_complete::generate(shell, &mut cmd, bin, &mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

/// A short install hint (as shell comments) prepended to the generated script.
pub fn install_hint(shell: Shell) -> String {
    match shell {
        Shell::Zsh => "# arf zsh completions. Install:\n#   arf completions zsh > \"${fpath[1]}/_arf\"\n#   # or, quick: add  source <(arf completions zsh)  to ~/.zshrc\n".to_string(),
        Shell::Bash => "# arf bash completions. Install:\n#   arf completions bash > /usr/local/etc/bash_completion.d/arf\n#   # or, quick: add  source <(arf completions bash)  to ~/.bashrc\n".to_string(),
        Shell::Fish => "# arf fish completions. Install:\n#   arf completions fish > ~/.config/fish/completions/arf.fish\n".to_string(),
        other => format!("# arf completions for {other:?}. Pipe to your shell's completion dir.\n"),
    }
}

/// A defensive, per-shell snippet appended after the generated clap script so
/// `arf pull <TAB>` completes from BOTH local model names (`arf model-names`)
/// AND live HuggingFace search (`arf hf-search <word>`). clap_complete's static
/// scripts don't do dynamic shell-callouts, so we add our own.
///
/// Every callout is best-effort: stderr is silenced and failures fall through to
/// "no suggestions" (the `hf-search`/`model-names` commands themselves exit 0 on
/// every failure path). Returns an empty string for shells we don't special-case.
pub fn dynamic_pull_snippet(shell: Shell) -> String {
    match shell {
        // zsh: clap registers the `_arf` completer. We ADD model candidates only
        // when completing `pull`'s argument, then defer to clap for everything else,
        // so other subcommands keep clap's flag/subcommand completion.
        Shell::Zsh => "\
\n# arf: augment `arf pull <TAB>` with local + HuggingFace model names.
# Wraps clap's `_arf` so other subcommands keep their normal completion.
_arf_with_models() {
  if [[ \"$words[2]\" == \"pull\" && $CURRENT -ge 3 ]]; then
    local -a models
    models=(${(f)\"$(arf model-names 2>/dev/null)\"} ${(f)\"$(arf hf-search \"$words[CURRENT]\" 2>/dev/null)\"})
    compadd -- $models
  fi
  _arf \"$@\"
}
compdef _arf_with_models arf
"
        .to_string(),
        // bash: clap defines `_arf`. We wrap it: add model candidates when the
        // first word after `arf` is `pull`, then call clap's `_arf` so flags
        // and other subcommands still complete normally (no clobbering).
        Shell::Bash => "\
\n# arf: augment `arf pull <TAB>` with local + HuggingFace model names.
# Wraps clap's `_arf` so other subcommands keep their normal completion.
_arf_with_models() {
  local cur=\"${COMP_WORDS[COMP_CWORD]}\"
  if [[ \"${COMP_WORDS[1]}\" == \"pull\" && ${COMP_CWORD} -ge 2 ]]; then
    local opts
    opts=\"$(arf model-names 2>/dev/null; arf hf-search \"$cur\" 2>/dev/null)\"
    COMPREPLY=( $(compgen -W \"$opts\" -- \"$cur\") )
    [[ ${#COMPREPLY[@]} -gt 0 ]] && return 0
  fi
  _arf 2>/dev/null
}
complete -F _arf_with_models arf 2>/dev/null || true
"
        .to_string(),
        Shell::Fish => "\
\n# arf: dynamic completion for `arf pull`
complete -c arf -n '__fish_seen_subcommand_from pull' -f -a '(arf model-names 2>/dev/null; arf hf-search (commandline -ct) 2>/dev/null)'
"
        .to_string(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cli; // the top-level clap Parser

    #[test]
    fn zsh_script_is_non_empty_and_names_binary() {
        let s = generate_to_string::<Cli>(Shell::Zsh, "arf");
        assert!(s.len() > 100);
        assert!(s.contains("arf"));
    }
    #[test]
    fn bash_script_generates() {
        let s = generate_to_string::<Cli>(Shell::Bash, "arf");
        assert!(s.contains("arf"));
    }
    #[test]
    fn install_hint_is_commented_and_shell_specific() {
        let z = install_hint(Shell::Zsh);
        assert!(z.lines().all(|l| l.is_empty() || l.starts_with('#')));
        assert!(z.contains("_arf") || z.contains("zshrc"));
        assert!(install_hint(Shell::Fish).contains("fish"));
    }

    #[test]
    fn dynamic_snippet_is_shell_specific() {
        assert!(dynamic_pull_snippet(Shell::Zsh).contains("hf-search"));
        assert!(dynamic_pull_snippet(Shell::Bash).contains("hf-search"));
        assert!(dynamic_pull_snippet(Shell::Fish).contains("hf-search"));
        assert!(dynamic_pull_snippet(Shell::Elvish).is_empty());
    }
}
