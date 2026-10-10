---
name: Bug report
about: Something produces the wrong output, crashes, or fails to load
labels: bug
---

**What happened, and what you expected instead.**

**Command** (the exact command line, including every `ARF_*` variable you set)

```sh

```

**Environment**

- Hardware (chip and memory, e.g. M4 Max 36 GB; or GPU model for the wgpu path):
- macOS version (`sw_vers -productVersion`), or OS and version:
- Model and quantization (name + HuggingFace repo or GGUF file, e.g. Qwen3.8-27B Q4_K_M):
- Arf version or commit (`arf --version`, or `git rev-parse --short HEAD`):
- Rust toolchain if you built from source (`rustc --version`):

**Output**

<!-- The error or panic with its backtrace (RUST_BACKTRACE=1), or for wrong text: the prompt and
     what came back. Say whether it is wrong at batch size 1, at batch size 2 or more, or both. -->

```

```

<!-- If this is a SPEED report, use the "Performance report" template instead. A short run
     measures model load and page-in, not decode: the same binary reads 2.7 tok/s at 32 tokens
     and 13.6 at 512. -->
