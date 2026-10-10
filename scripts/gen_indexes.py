#!/usr/bin/env python3
"""Regenerate docs/EXAMPLES.md and docs/SCRIPTS.md from the tree.

Both files are indexes of things that exist on disk. Hand-maintaining them means they drift the
moment someone adds a probe, so they are generated: run this after adding an example or script.
`scripts/check_generated_docs.sh` fails CI if the committed copy is stale.

The CATEGORIES are hand-assigned below (a name -> section table); the DESCRIPTIONS come from each
file's first doc line unless a one-liner is given here. A tracked file that no table names still
appears, under "Unsorted", so a new script or example is never silently missing from the index —
when you see one there, give it a section.
"""
import os, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


# `ARF_INDEX_EXCLUDE`, when set, names a file (relative to the repository root, or absolute) that
# lists paths to leave out of the generated indexes: one per line, `#` comments allowed, a trailing
# `/` covers a directory. Unset, nothing is skipped. Set to a file that does not exist, this fails
# rather than silently skipping nothing.
def _load_excluded(root):
    name = os.environ.get("ARF_INDEX_EXCLUDE")
    if not name:
        return []
    f = root / name
    if not f.is_file():
        sys.exit(f"ARF_INDEX_EXCLUDE={name}: no such file under {root}")
    return [l.strip() for l in f.read_text().split("\n") if l.strip() and not l.startswith("#")]


def excluded(rel):
    return any(rel == p or (p.endswith("/") and rel.startswith(p)) for p in EXCLUDED)


EXCLUDED = _load_excluded(ROOT)


def tracked(pattern):
    out = subprocess.run(["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True)
    return sorted(f for f in out.stdout.split() if not excluded(f))


def first_doc_line(path, markers):
    for line in (ROOT / path).read_text(errors="ignore").splitlines()[:8]:
        for m in markers:
            if line.startswith(m):
                text = line[len(m):].strip()
                if text and not text.startswith(("-*-", "!")):
                    return text
    return ""


def script_doc_line(path):
    """First line of a script's header comment or docstring (`#`, `//` or `\"\"\"`), shebang and a
    bare opening `\"\"\"` skipped."""
    lines = (ROOT / path).read_text(errors="ignore").splitlines()[:10]
    in_doc = False
    for line in lines:
        s = line.strip()
        if s.startswith("#!") or s.startswith("# -*-"):
            continue
        if in_doc:
            if s:
                return s.removesuffix('"""').strip()
            continue
        if s.startswith('"""'):
            rest = s[3:].strip()
            if rest:
                return rest.removesuffix('"""').strip()
            in_doc = True
            continue
        for m in ("#", "//"):
            if s.startswith(m) and s[len(m):].strip():
                return s[len(m):].strip()
    return ""


def cell(d, width):
    d = d.replace("|", "\\|").rstrip()
    return (d[:width - 3].rstrip() + "…" if len(d) > width else d) or "—"


# ------------------------------------------------------------------------------------------------
# EXAMPLES
# ------------------------------------------------------------------------------------------------

# The handful a user or contributor should actually run, with a one-line reason each. Everything
# else is listed further down, classified, for the reader who is chasing a specific question.
EXAMPLES_START = [
    ("hud_preview", "arf-serve", "No model needed. The live `arf serve` dashboard on animated fake data."),
    ("gguf_probe", "arf-gpu", "Does this GGUF parse? Tensor count and finite/magnitude checks before a full load."),
    ("dump_vocab", "arf-core", "A GGUF's vocabulary as JSON `{id: text}`, for log analysers."),
    ("tok_canonical_check", "arf-core", "Is the tokenizer Arf builds from a GGUF the canonical one? Encodes the same text both ways."),
    ("loadgen", "arf-serve", "Concurrency sweep against any OpenAI-compatible server: aggregate tok/s, TTFT, p50/p99 gaps."),
    ("verify", "arf-serve", "Cross-engine correctness: how long two servers agree under greedy, and whether a split is a near-tie."),
    ("serve_loop_bench", "arf-gpu", "The serving loop without HTTP; the harness behind the concurrency numbers and the PGO workload."),
    ("batched_mega_parity", "arf-gpu", "`make parity-conc`: the batched megakernel against its oracle. Must be all PASS before a perf hunt."),
    ("ppl_windows", "arf-gpu", "Teacher-forced perplexity through the hybrid serving path; the gate for any change to weight numbers."),
    ("spec_accept", "arf-core", "Speculative-decoding acceptance on real weights."),
    ("qwen_vision_gpu_parity", "arf-gpu", "GPU vision encoder against the CPU reference on the same image."),
    ("gated_delta_net_probe", "arf-gpu", "No model needed. Parity self-test of the gated-delta-net recurrence kernel."),
    ("ssm_conv1d_probe", "arf-gpu", "No model needed. Parity self-test of the `ssm_conv1d` kernel."),
    ("dflash_chain_sampled_probe", "arf-gpu", "No model needed. Gate for the sampled GPU draft chain on synthetic inputs."),
    ("pso_limits_probe", "arf-gpu", "No model needed. Compile-only: each attention kernel's pipeline launch limits on this GPU."),
    ("flux_generate", "arf-gpu", "FLUX end to end: prompt to image, all on the GPU."),
]

# Classification of every public example. "gate" = a parity/correctness check that proves a kernel
# or a load path against an oracle (wired into make/scripts, or a stage gate of a port plan);
# "tool" = a lasting instrument or benchmark; "probe" = a one-off investigation, kept because its
# answer is recorded in a code comment and re-running it is how that answer is checked.
EXAMPLE_CLASS = {
    # gates
    "batched_mega_parity": "gate", "mega_moe_gate": "gate", "coop_q4ks_probe": "gate",
    "gemv_id_probe": "gate", "mm_q4ks_probe": "gate", "mpp_q4_parity": "gate",
    "gdn_wgsl_parity_b": "gate", "coop_gemma_match": "gate", "gated_delta_net_probe": "gate",
    "ssm_conv1d_probe": "gate", "dflash_chain_sampled_probe": "gate", "g64_pfs_probe": "gate",
    "dflash2_load": "gate", "dflash2_taps": "gate", "dflash2_context": "gate",
    "qwen35_load_probe": "gate", "qwen35_forward_probe": "gate", "qwen35_generate": "gate",
    "qwen_vision_gpu_parity": "gate", "flux_text_parity": "gate", "flux_dit_parity": "gate",
    "flux_vae_parity": "gate", "flux_sched_parity": "gate", "flux_dit_q4ks_ab": "gate",
    "flux_t5_q8_ab": "gate", "t5_q8_split_check": "gate", "tok_canonical_check": "gate",
    "ppl_windows": "gate",
    # tools and benchmarks
    "serve_loop_bench": "tool", "loadgen": "tool", "verify": "tool", "hud_preview": "tool",
    "gguf_probe": "tool", "dump_vocab": "tool", "dump_tensor_f32": "tool", "compare_f32": "tool",
    "spec_accept": "tool", "flux_generate": "tool", "gemma_ab": "tool", "coop_batch_bench": "tool",
    "gpu_kv_quant_bench": "tool", "pso_limits_probe": "tool",
    # one-off probes and investigations
    "cpu_repro": "probe", "dflash_probe": "probe", "flux_dit_quant_probe": "probe",
    "qwen_vision_time": "probe", "tok_decode_probe": "probe", "agentic_proof": "probe",
    "batched_gemma4_diff": "probe", "brow_microbench": "probe", "coop_bench": "probe",
    "coop_nt_probe": "probe", "coop_probe": "probe", "dflash2_block": "probe",
    "dflash2_precision": "probe", "flux_dit_load_probe": "probe", "flux_nan_probe": "probe",
    "flux_t5_load_probe": "probe", "flux_text_smoke": "probe", "gdn_l2_offset_probe": "probe",
    "gemv_msl_ab": "probe", "gemv_q4_dims_bench": "probe", "gemv_roofline_probe": "probe",
    "gpu_prefill_timing": "probe", "gpu_spec_timing": "probe", "group64_price": "probe",
    "int_dot_capability_probe": "probe", "mega_perf_breakdown": "probe",
    "metal_concurrent_probe": "probe", "qwen35_bpar_generate": "probe", "qwen35_meta_probe": "probe",
    "verify_matmul_share": "probe", "vision_discriminate": "probe",
    "vision_gpu_probe": "probe", "vision_sum_probe": "probe",
}

EXAMPLE_SECTIONS = [
    ("gate", "Parity and correctness gates",
     "Each proves a kernel, a load path or a port stage against an oracle (a CPU reference, a "
     "Python reference, or the previous path). Run the relevant one after touching what it covers."),
    ("tool", "Tools and benchmarks", "Lasting instruments. Speed numbers from these follow the "
     "interleaving rules in [PERFORMANCE.md](PERFORMANCE.md)."),
    ("probe", "One-off probes and investigations",
     "Each answered one question. They stay because re-running the probe is how its answer, recorded "
     "in a code comment, is checked. "
     "Not maintained as tools; expect hard-coded model paths."),
    (None, "Unsorted", "Not yet classified in `scripts/gen_indexes.py`: give it a section."),
]


def examples_md():
    rows = {}
    for f in tracked("crates/*/examples/*.rs"):
        p = pathlib.Path(f)
        rows[p.stem] = (f.split("/")[1], first_doc_line(f, ["//!"]))

    out = ["# Examples — the probes, gates and benches\n",
           "<!-- Generated by scripts/gen_indexes.py. Do not edit by hand. -->\n",
           "These are the tools the engine was built with: parity gates that prove a kernel against a",
           "CPU oracle, capability probes that answer a GO/NO-GO question about the hardware, and benches",
           "that produced the numbers in [PERFORMANCE.md](PERFORMANCE.md).\n",
           "They are examples rather than tests because most need a real multi-gigabyte model file, which",
           "CI does not have. All of them compile in CI (`clippy --all-targets`), so they cannot silently",
           "rot.\n", "```sh", "cargo run --release -p <crate> --example <name> -- [args]", "```",
           "\n## Start here\n",
           "The ones a user or contributor is most likely to need. Each file's header comment gives "
           "its arguments.\n",
           "| example | crate | why you would run it |", "|---|---|---|"]
    for n, c, why in EXAMPLES_START:
        if n in rows:
            out.append(f"| `{n}` | {c} | {cell(why, 120)} |")

    for key, title, blurb in EXAMPLE_SECTIONS:
        group = sorted(n for n in rows if EXAMPLE_CLASS.get(n) == key)
        if not group:
            continue
        out += [f"\n## {title} ({len(group)})\n", blurb + "\n",
                "| example | crate | what it does |", "|---|---|---|"]
        for n in group:
            c, d = rows[n]
            out.append(f"| `{n}` | {c} | {cell(d, 96)} |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------
# SCRIPTS
# ------------------------------------------------------------------------------------------------

# Paths are relative to scripts/. A trailing "/" makes one row of a whole directory.
SCRIPT_SECTIONS = [
    ("Gates: repository hygiene (no model needed)",
     "Cheap, offline. `check_generated_docs.sh` and `check_no_local_paths.sh` fail on drift; "
     "`env_surface_guard.sh` is the ratchet on the number of `ARF_*` flags; `leak_scan.sh` runs before every push.",
     ["check_generated_docs.sh", "check_no_local_paths.sh", "env_surface_guard.sh", "leak_scan.sh"]),
    ("Gates: correctness (need a model)",
     "Text before speed (rule 4): run the one covering what you changed before quoting a speed "
     "number. Each file's header gives its arguments and the model it expects.",
     ["two_model_gate.sh", "smoke_all_models.sh", "refactor_gate.sh", "spec_identity_check.py",
      "conc_prompt_fidelity.py", "conc_sanity.py", "mpp_e2e_parity.py", "self_consistency.py",
      "prefill_recall.py", "prefix_cache_hybrid.py", "ctx_cross_check.py"]),
    ("Gates: make targets",
     "`make gate` / `make gate-quick` run `nightly_gate.sh`; `make repro-conc` runs `repro_conc.sh`.",
     ["nightly_gate.sh", "repro_conc.sh"]),
    ("Benchmarks",
     "A number from any of these is quotable only with its arms interleaved in one session on a "
     "quiet box (see [PERFORMANCE.md](PERFORMANCE.md)). `bench_engines.py` produced the README's "
     "comparison table; `speed_bench.py` runs the SPEED-Bench coding workload on every engine installed.",
     ["bench_engines.py", "speed_bench.py", "bench_me.sh", "honest_bench.sh", "spec_ab.sh",
      "kernel_ab.sh", "ab_bench.py", "mpp_ladder.py", "real_conc_baseline.sh", "soak_conc64.sh",
      "agent_bench/"]),
    ("Machine and instrument checks",
     "Run before believing any number (rule 8).",
     ["machine_health.sh", "verify_instrument.sh", "profile.sh"]),
    ("Log analysers and file probes", "",
     ["spec_miss_report.py", "routing_histogram.py"]),
    ("Probes (`scripts/probes/`)",
     "Request sets and checkers for a feature's parity with a reference server. Each header says how "
     "to run it (from the repository root).",
     ["probes/"]),
    ("Reference implementations",
     "Independent references the engine is checked against: written from the upstream model code, "
     "not from Arf's Rust.",
     ["ref/qwen3_omni/", "ref/trellis2/", "qwen_vision_ref.py", "dump_hf_logits.py",
      "compare_logits.py", "hf_perplexity.py", "mlx_ppl.py"]),
    ("Building model files", "",
     ["get_qwen38.sh", "g64/"]),
    ("Doc generators", "Run after adding a flag, a shader, an example or a script.",
     ["gen_env_inventory.py", "gen_indexes.py", "gen_kernel_map.py"]),
    ("Shared library", "", ["lib/"]),
]

SCRIPT_DOC = {
    "conc_sanity.py": "N distinct prompts sent concurrently to the 27B; prints each answer so a "
                      "crossed or garbled row is visible. `conc_sanity.py <arf-serve> 4,8 [flags]`.",
    "mpp_ladder.py": "Interleaved OFF/ON speed ladder (p50 step) for a switch, via `ab_bench.py`. "
                     "MODEL/ARCH/EXTRA from MPP_* env.",
    "mlx_ppl.py": "Teacher-forced perplexity of an MLX model over a text, in ppl_windows' protocol.",
    "self_consistency.py": "THE CONTROL: the shipped path against itself (same binary, flags, 16 "
                           "concurrent prompts, two daemon starts).",
    "compare_logits.py": "Compare two JSONL logit dumps: top-1 agreement, top-5 overlap, KL, max |Δlogit|.",
    "dump_hf_logits.py": "Reference logit dumper (HF transformers, f32), same schema as `arf dump-logits`.",
    "agent_bench/": "What a user of a coding agent feels: wall time from prompt to PASSING tests, "
                    "per engine x agent x task (`run.py`, tasks in `tasks/`).",
    "ref/qwen3_omni/": "Qwen3-Omni audio encoder reference (`audio_encoder_ref.py`, "
                       "`fetch_audio_tower.py`).",
    "ref/trellis2/": "TRELLIS.2 reference harness (`run_ref.py`, `compare.py`, `shim.py`, "
                     "`fetch_weights.sh`).",
    "g64/": "Our group-64 quantisation: `quantize_g64.py` (8-bit GGUF -> group-64 4-bit) and "
            "`mlx_to_q4_1_gguf.py` (MLX affine 4-bit -> hybrid GGUF). Used by `get_qwen38.sh`.",
    "lib/": "`arf_env.sh`: the one definition of how a benchmark launches arf-serve, plus the RAM "
            "guard.",
}



def scripts_md():
    files = [f[len("scripts/"):] for f in tracked("scripts/")]
    claimed = set()

    def unit_files(unit):
        if unit.endswith("/"):
            return [f for f in files if f.startswith(unit)]
        return [unit] if unit in files else []

    def doc_for(unit):
        if unit in SCRIPT_DOC:
            return SCRIPT_DOC[unit]
        d = script_doc_line("scripts/" + unit)
        return re.sub(rf"^{re.escape(pathlib.Path(unit).name)}\s*[—-]\s*", "", d)

    out = ["# Scripts — gates, harnesses and probes\n",
           "<!-- Generated by scripts/gen_indexes.py. Do not edit by hand. -->\n",
           "Every performance claim in this repo was produced by one of these, and the gates encode the",
           "house rules: interleave A/B arms in one session, check the machine before trusting a number,",
           "verify correctness before measuring speed.\n",
           "Start with `machine_health.sh` (is the box healthy right now) and `bench_me.sh` (this Mac's",
           "numbers against [PERFORMANCE.md](PERFORMANCE.md)). Run everything from the",
           "repository root. Paths below are relative to `scripts/`."]

    for title, blurb, units in SCRIPT_SECTIONS:
        body = []
        for u in units:
            fs = unit_files(u)
            if not fs:
                continue
            claimed.update(fs)
            if u == "probes/":
                for f in fs:
                    body.append((f, doc_for(f)))
            else:
                body.append((u, doc_for(u)))
        if not body:
            continue
        out += [f"\n## {title}\n"] + ([blurb + "\n"] if blurb else [])
        out += ["| script | what it does |", "|---|---|"]
        for n, d in body:
            out.append(f"| `{n}` | {cell(d, 104)} |")

    rest = [f for f in files if f not in claimed]
    if rest:
        out += ["\n## Unsorted\n", "Not yet given a section in `scripts/gen_indexes.py`.\n",
                "| script | what it does |", "|---|---|"]
        for f in rest:
            out.append(f"| `{f}` | {cell(doc_for(f), 104)} |")
    return "\n".join(out) + "\n"


def main():
    check = "--check" in sys.argv
    stale = []
    for rel, body in [("docs/EXAMPLES.md", examples_md()), ("docs/SCRIPTS.md", scripts_md())]:
        path = ROOT / rel
        if check:
            if not path.exists() or path.read_text() != body:
                stale.append(rel)
        else:
            path.write_text(body)
            print(f"wrote {rel}")
    if check and stale:
        print("stale generated docs: " + ", ".join(stale))
        print("run: python3 scripts/gen_indexes.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
