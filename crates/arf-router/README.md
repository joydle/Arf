# arf-router

One endpoint in front of N `arf-serve` machines.

A single M4 Max runs one model well. To serve from several Macs, the router sits in front of
them, load-balances OpenAI-compatible requests across them, and keeps the
prefix-cache win that makes Arf fast in the first place.

It is a pure HTTP proxy: no engine dependency, no GPU, no model. It builds and runs anywhere,
which is the point — the replicas are the machines with the hardware.

```sh
arf-router --replica http://mac-1:8080 --replica http://mac-2:8080 --port 8080
```

Then point any OpenAI client at the router instead of at a single machine.

## What it does

- **Least-outstanding-requests.** Load is measured by requests the router itself has in flight
  per replica, not by polling the replica for a number that is already stale.
- **Prefix affinity (HRW).** Requests sharing a prompt prefix are steered to the same replica so
  its KV cache still holds that prefix. This compounds the shared-prefix win — the reason a
  coding agent's second request is much faster than its first — instead of throwing it away at
  the load balancer.
- **Model pools.** Tag a replica with `--replica qwen3.8=http://host:8080` and it only receives
  requests for that model; untagged replicas serve anything. Several replicas sharing a tag form a
  pool. Several Macs serving different models sit behind the same
  endpoint.
- **Health polling.** `/healthz` per replica on an interval; the live list is swapped
  atomically, so a replica going down does not stall in-flight work on the others.
- **Discovery.** `--replica` for a static list, or `--discovery-url` to poll a JSON endpoint.
  Consul, Kubernetes and Nomad all work by pointing that at an adapter.

## What it deliberately does not do

**No consensus, no gossip, no shared state.** The router is stateless; whatever runs your
machines is the authority on which replicas exist. That is why it can be restarted, duplicated,
or put behind another load balancer without coordination.

**No queueing or admission control.** Each `arf-serve` already has a continuous-batching
scheduler that knows its own KV pressure. A second queue in front of it would make worse
decisions with less information.

## Flags

| flag | default | what it does |
|---|---|---|
| `--replica [model=]URL` | — | A backend. Repeat for each. Tag it (`qwen3.8=http://h:8080`) to restrict it to one model; untagged serves any. |
| `--discovery-url <URL>` | — | Poll this JSON endpoint for the replica list instead. |
| `--discovery-interval-secs` | 5 | How often to re-poll discovery. |
| `--health-interval-secs` | 2 | How often to poll each replica's `/healthz`. |
| `--no-affinity` | off | Disable prefix affinity; pure least-outstanding. |
| `--port` | 8080 | Listen port. |
| `--host` | 0.0.0.0 | Listen address. |

## Status

Written 2026-06, 38 tests, unchanged since — it does one job and has not needed to change.
It is an **optional** component: a single machine needs nothing in front of it.
