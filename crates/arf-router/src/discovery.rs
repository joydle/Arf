//! Dynamic membership discovery for the fleet router.
//!
//! ## Architecture
//!
//! The router is stateless; the orchestrator is the authority on membership.
//! This module provides a pluggable `Discovery` trait that the router
//! CONSUMES — it does NOT implement Raft, gossip, or any consensus.
//!
//! ## Impls
//!
//! ### StaticDiscovery
//! Returns the seed `--replica` list forever.  Identical to the static default.
//!
//! ### EndpointDiscovery
//! GETs a user-supplied URL on each refresh tick and parses:
//! ```json
//! [
//!   {"url": "http://10.0.0.1:8080"},
//!   {"url": "http://10.0.0.2:8080", "models": ["llama-3.2-1b"]}
//! ]
//! ```
//! This is the GENERIC schema this router understands.  Consul/k8s/Nomad
//! integration = "point the router at an endpoint (or tiny sidecar) that
//! emits this shape".  No Consul-specific parsing is in this crate;
//! a 5-line adapter/jq transform bridges the gap.
//!
//! ## Refresh loop
//! `spawn(state, discovery, interval)` polls `resolve()` on the given interval,
//! diffs against the current list, builds a new snapshot REUSING existing Arcs
//! (via `FleetState::refresh_replicas`), and logs additions/removals.
//! New replicas start `healthy=false`; the health loop flips them.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use tracing::{info, warn};

use crate::state::{FleetState, ReplicaSpec};

// ---------------------------------------------------------------------------
// Discovery trait
// ---------------------------------------------------------------------------

/// A source of truth for the desired replica set.
///
/// Implementors are called on a timer; the router CONSUMES the result and
/// applies it to `FleetState` (no consensus, no gossip).
#[async_trait]
pub trait Discovery: Send + Sync {
    /// Return the current desired replica set.
    async fn resolve(&self) -> Result<Vec<ReplicaSpec>>;
}

// ---------------------------------------------------------------------------
// StaticDiscovery
// ---------------------------------------------------------------------------

/// Returns the seed replica list verbatim — never changes.
/// This is the default: `--discovery static` (the static default).
pub struct StaticDiscovery(pub Vec<ReplicaSpec>);

#[async_trait]
impl Discovery for StaticDiscovery {
    async fn resolve(&self) -> Result<Vec<ReplicaSpec>> {
        Ok(self.0.clone())
    }
}

// ---------------------------------------------------------------------------
// EndpointDiscovery
// ---------------------------------------------------------------------------

/// Wires a JSON HTTP endpoint into the discovery seam.
///
/// ## Generic schema (the one this router speaks)
///
/// ```json
/// [
///   {"url": "http://host:port"},
///   {"url": "http://host:port", "models": ["model-id"]}
/// ]
/// ```
///
/// ## Consul/k8s/Nomad integration
///
/// These orchestrators return their OWN JSON schema from their service-catalog
/// APIs.  Rather than building Consul-specific or k8s-specific parsers here
/// (which would bloat this pure-proxy crate with cloud-provider SDKs), the
/// integration point is:
///
/// > Point `--discovery-url` at an endpoint that emits the generic schema.
///
/// A sidecar adapter can be as simple as:
/// ```sh
/// # Consul: translate /v1/health/service/<svc> → generic schema
/// jq '[.[] | {url: ("http://" + .Service.Address + ":" + (.Service.Port|tostring)),
///             models: [.Service.Meta.model // empty]}]' <consul-response>
/// ```
/// This keeps the router zero-dependency on specific orchestrators and lets
/// Consul/k8s/Nomad "just work" with a translator shim.
pub struct EndpointDiscovery {
    url: String,
    client: reqwest::Client,
}

impl EndpointDiscovery {
    pub fn new(url: String, client: reqwest::Client) -> Self {
        Self { url, client }
    }
}

/// Wire format for the generic discovery endpoint.
#[derive(Deserialize)]
struct DiscoveryEntry {
    url: String,
    #[serde(default)]
    models: Vec<String>,
}

#[async_trait]
impl Discovery for EndpointDiscovery {
    async fn resolve(&self) -> Result<Vec<ReplicaSpec>> {
        let entries: Vec<DiscoveryEntry> = self
            .client
            .get(&self.url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let specs = entries
            .into_iter()
            .map(|e| ReplicaSpec {
                url: e.url,
                models: e.models,
            })
            .collect();
        Ok(specs)
    }
}

// ---------------------------------------------------------------------------
// Refresh loop
// ---------------------------------------------------------------------------

/// Spawn the discovery refresh background task.
///
/// On each tick:
/// 1. Call `discovery.resolve()`.
/// 2. Log additions and removals (INFO level).
/// 3. Call `FleetState::refresh_replicas` which REUSES existing `Arc<Replica>`
///    for URLs that are still present (preserves in_flight/healthy) and starts
///    new replicas at `in_flight=0`, `healthy=false`.
pub fn spawn(
    state: Arc<FleetState>,
    discovery: Arc<dyn Discovery>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // First tick fires immediately so the list is current before the router
        // starts serving requests (same eager-probe philosophy as health.rs).
        refresh_once(&state, &*discovery).await;

        loop {
            tokio::time::sleep(interval).await;
            refresh_once(&state, &*discovery).await;
        }
    })
}

async fn refresh_once(state: &FleetState, discovery: &dyn Discovery) {
    match discovery.resolve().await {
        Err(e) => {
            warn!(error = %e, "discovery resolve() failed — keeping current replica list");
        }
        Ok(new_specs) => {
            let current = state.replicas.load();
            let current_urls: std::collections::HashSet<&str> =
                current.iter().map(|r| r.url.as_str()).collect();
            let new_urls: std::collections::HashSet<&str> =
                new_specs.iter().map(|s| s.url.as_str()).collect();

            // Log additions.
            for url in new_urls.difference(&current_urls) {
                info!(url = %url, "discovery: replica ADDED");
            }
            // Log removals.
            for url in current_urls.difference(&new_urls) {
                info!(url = %url, "discovery: replica REMOVED");
            }

            state.refresh_replicas(new_specs);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- StaticDiscovery ----

    #[tokio::test]
    async fn static_discovery_returns_seed() {
        let specs = vec![
            ReplicaSpec {
                url: "http://a:8080".into(),
                models: vec!["llama".into()],
            },
            ReplicaSpec {
                url: "http://b:8080".into(),
                models: vec![],
            },
        ];
        let d = StaticDiscovery(specs.clone());
        let got = d.resolve().await.unwrap();
        assert_eq!(got, specs);
    }

    #[tokio::test]
    async fn static_discovery_idempotent() {
        let specs = vec![ReplicaSpec {
            url: "http://a:8080".into(),
            models: vec![],
        }];
        let d = StaticDiscovery(specs.clone());
        let first = d.resolve().await.unwrap();
        let second = d.resolve().await.unwrap();
        assert_eq!(first, second);
    }

    // ---- EndpointDiscovery JSON parsing ----

    // We test the parsing logic directly by constructing an EndpointDiscovery
    // against a tiny in-process HTTP server (using axum + a tokio listener).
    // This avoids any live network calls while still exercising the real
    // reqwest + serde_json path.

    async fn serve_json(body: &'static str) -> String {
        use axum::http::header;
        use axum::response::IntoResponse;
        use axum::routing::get;
        use axum::Router;

        let body_str = body;
        let app = Router::new().route(
            "/discovery",
            get(move || async move {
                ([(header::CONTENT_TYPE, "application/json")], body_str).into_response()
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://127.0.0.1:{}/discovery", addr.port())
    }

    #[tokio::test]
    async fn endpoint_discovery_parses_generic_schema() {
        let url =
            serve_json(r#"[{"url":"http://a:8080"},{"url":"http://b:8080","models":["llama"]}]"#)
                .await;

        let d = EndpointDiscovery::new(url, reqwest::Client::new());
        let specs = d.resolve().await.unwrap();

        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].url, "http://a:8080");
        assert!(specs[0].models.is_empty());
        assert_eq!(specs[1].url, "http://b:8080");
        assert_eq!(specs[1].models, vec!["llama"]);
    }

    #[tokio::test]
    async fn endpoint_discovery_empty_array() {
        let url = serve_json(r#"[]"#).await;
        let d = EndpointDiscovery::new(url, reqwest::Client::new());
        let specs = d.resolve().await.unwrap();
        assert!(specs.is_empty());
    }
}
