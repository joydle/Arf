//! Background health-check loop.
//!
//! Polls each replica's `/healthz` endpoint every `interval`. A 200 response
//! marks the replica healthy; anything else (timeout, non-200, connect error)
//! marks it unhealthy and drops it from the routing pool until it recovers.
//!
//! Transitions are logged at INFO level so ops can see a replica going in/out
//! of rotation without enabling DEBUG.
//!
//! The initial probe fires *immediately* (before the first `interval` sleep)
//! so the router doesn't 503 every request during the first health-check window.
//!
//! ## Dynamic membership
//!
//! The replica list is now behind an `ArcSwap`.  The health loop calls
//! `state.load_replicas()` on EVERY tick (not once at startup) so it always
//! operates on the CURRENT live list — replicas added by the discovery refresh
//! are probed on the very next tick, and departed replicas are naturally absent
//! (the Arc is simply not in the loaded snapshot).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use tracing::{info, warn};

use crate::state::FleetState;

/// Spawn the health-check background task and return the `JoinHandle`.
///
/// `state` is cloned into the task (cheap Arc clone).
/// `interval` — how often to poll each replica.
/// `timeout` — per-replica request timeout.
pub fn spawn(
    state: Arc<FleetState>,
    interval: Duration,
    timeout: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // Do one immediate probe so the router is live from the first request.
        probe_all(&state, timeout).await;

        loop {
            tokio::time::sleep(interval).await;
            probe_all(&state, timeout).await;
        }
    })
}

/// Check every replica in the CURRENT live list once, updating `healthy` and
/// logging transitions.
///
/// Reads the live list via `state.load_replicas()` each call so it stays
/// correct after a discovery refresh (new replicas are included, departed ones
/// are absent).
async fn probe_all(state: &FleetState, timeout: Duration) {
    // Load the current snapshot once per tick.  `arc_swap::Guard` is cheap —
    // it just bumps a reference count on the inner Arc for the duration of
    // this function, then releases on drop.
    let replicas = state.load_replicas();

    for replica in replicas.iter() {
        let url = format!("{}/healthz", replica.url);
        let was_healthy = replica.healthy.load(Ordering::Relaxed);

        let ok = match state.client.get(&url).timeout(timeout).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        };

        if ok != was_healthy {
            if ok {
                info!(replica = %replica.url, "replica UP — entering rotation");
            } else {
                warn!(replica = %replica.url, "replica DOWN — removed from rotation");
            }
        }

        replica.healthy.store(ok, Ordering::Relaxed);
    }
}
