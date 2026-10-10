---
name: Model support
about: A model that does not load, or loads and produces garbage
labels: model-support
---

**Model**

- Name and link (HuggingFace repo or GGUF URL):
- Architecture as reported by the loader, if it got that far:
- Quantization:

**Where it fails**

- [ ] Won't load at all
- [ ] Loads, but `--arch` is rejected
- [ ] Loads and runs, but the text is wrong

**Output**

```
```

<!-- Supported today: Qwen3-Coder-30B, muse-glimmer-30b, Qwen3.5/3.6/3.8-27B, Gemma 4 (12B/31B),
     Gemma 3 4B, Llama 3.2 1B. The full list with aliases is in arf_core::config::ARCHITECTURES;
     `arf serve --arch nonsense` prints it. -->
