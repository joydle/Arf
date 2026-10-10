# Documentation

An index of the documentation in this repository.

## Start here

| page | what it is |
|---|---|
| [`../README.md`](../README.md) | what Arf is, what works, where it wins and loses, how to use it |
| [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) | how a single token flows through the system |
| [`ENGINE.md`](ENGINE.md) | **the complete technical account** |
| [`PLATFORMS.md`](PLATFORMS.md) | what runs where — and which of it has actually been run |
| [`PERFORMANCE.md`](PERFORMANCE.md) | measured results, each with its date, machine and method |

## Reference

| page | what it is |
|---|---|
| [`KERNELS.md`](KERNELS.md) · [`KERNEL_MAP.md`](KERNEL_MAP.md) | the GPU kernels and what each costs · a generated index of every shader |
| [`ENV_SUPPORTED.md`](ENV_SUPPORTED.md) · [`ENV_INVENTORY.md`](ENV_INVENTORY.md) | the environment variables you may set · a generated census of every one the code reads |
| [`SAFETY.md`](SAFETY.md) | what the `unsafe` is, and the invariants that keep it sound |
| [`EXAMPLES.md`](EXAMPLES.md) · [`SCRIPTS.md`](SCRIPTS.md) | the probes, parity gates and benches · the scripts (both generated) |
| [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | notes for the current release |
| [`RELEASING.md`](RELEASING.md) | how a release is cut: measurements, notes, version, tag, the Homebrew formula |

See also [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

An NVIDIA backend is developed separately and is not part of this repository.

## Conventions

- **A number comes with its date, machine and method**, or it is marked unverified.
- **Claims are marked** *measured*, *projected* (arithmetic from a measured number) or *assumed*.
- **Corrections are public.** A wrong claim is corrected where it was published, and the correction says
  what it replaced.
- **Generated pages are not hand-edited** — `scripts/check_generated_docs.sh` fails CI if one drifts.
