# Security

## Reporting a vulnerability

Report privately through GitHub's [Report a vulnerability](../../security/advisories/new) form
rather than opening a public issue. Include what you did, what happened, and the smallest input
that reproduces it.

Expect an acknowledgement within a few days. This is a small project, not a vendor with an on-call
rotation — that is stated plainly so you can calibrate.

## What the threat model actually is

Two things here are worth an attacker's attention, and they are not the same shape.

**Model files are parsed input.** `arf` reads GGUF and safetensors files, which carry
attacker-controllable lengths, offsets and shapes. A malformed file reaching an unchecked index is
the most likely real vulnerability in this codebase. Loader bugs of that kind are in scope and
worth reporting.

**Only load model files you trust.** The loaders validate what they read, but treat a GGUF from an
unknown source the way you would treat any untrusted binary format.

## The server is a development daemon

`arf-serve` binds `127.0.0.1` by default, has **no authentication**, and sets a permissive CORS
policy. That is a deliberate choice for a local tool, not an oversight.

If you expose it beyond localhost, put it behind something that authenticates. Reports that amount
to "the server has no auth" describe documented behaviour; reports that it leaks data *despite*
being on localhost, or that a request can escape its process, are real and wanted.

## The `unsafe` code

The `unsafe` code, essentially all of it in `arf-gpu`, is catalogued in
[`docs/SAFETY.md`](docs/SAFETY.md), which states the four invariants it rests on: a mapped pointer only
outlives its buffer, host reads are fenced against the GPU, no-copy wrapping is page-aligned, and
indices are bounds-checked before pointer arithmetic.

A concrete way to violate one of those — a path where a length is not checked, a read that is not
fenced — is exactly the kind of report that is useful. "There is a lot of unsafe" is not.

## Out of scope

- Denial of service by asking for a very large generation. It is a local inference engine; it will
  use the resources you give it.
- Resource exhaustion from loading a model larger than available memory.
- Findings from a scanner with no demonstrated impact on this codebase.
