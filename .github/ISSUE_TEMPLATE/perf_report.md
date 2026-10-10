---
name: Performance report
about: A measured speed regression or gap, with a command anyone can re-run
labels: performance
---

<!-- Please read docs/PERFORMANCE.md first (how numbers are taken here). A number without its workload and command is a rumour,
     not a report: we cannot act on it. Reports missing the fields below will be asked for them. -->

**Summary** — what is slower than what, by how much.

**Reproducible command** (required — exact command lines for every arm, every `ARF_*` variable,
prompt or prompt file, `max_tokens`, temperature, and concurrency)

```sh

```

**Evidence** (required)

| date | commit | machine | model + quant | workload (prompt, tokens, concurrency) | arm | tok/s | runs |
|------|--------|---------|---------------|----------------------------------------|-----|-------|------|
|      |        |         |               |                                        |  A  |       |      |
|      |        |         |               |                                        |  B  |       |      |

- [ ] Both arms were run **interleaved in one session** (A, B, A, B ...), not hours apart
- [ ] Load average and swap (`sysctl vm.swapusage`) were checked before the runs; values:
- [ ] The output text was checked and is **correct** in every arm (a fast wrong answer is not a result)
- [ ] The log shows the feature or path being measured actually ran
- [ ] For a comparison with another engine: both were measured on the same machine, same model
      weights and the same prompts (thinking vs plain text changes acceptance and speed a lot)

**Environment**

- Hardware (chip and memory):
- macOS version:
- Arf commit:

**Anything else** — e.g. what you already tried.
