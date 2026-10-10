//! `arf-router` — stateless reactor-based fleet router for Arf.
//!
//! Sits in front of N `arf-serve` replica machines and load-balances OpenAI
//! requests across them.  Pure HTTP proxy; zero engine/GPU deps.
//!
//! ## Design
//!
//! - Load signal: router-tracked in-flight count (AtomicUsize per replica).
//! - Routing: least-outstanding-requests across healthy replicas, scoped to
//!   the model pool (replicas tagged for the requested model id).
//! - Affinity: HRW (highest-random-weight) prefix-affinity within the model pool.
//! - InFlightGuard: holds `Arc<Replica>` — correct under dynamic list churn.
//! - Discovery: pluggable trait (Static = --replica list; EndpointDiscovery =
//!   polls a JSON URL).  Consul/k8s/Nomad = point at a generic-schema adapter.
//! - Health: `/healthz` poll reads the live ArcSwap'd list each tick.
//! - Stateless: the orchestrator is the authority; no Raft/gossip here.

mod affinity;
mod discovery;
mod health;
mod proxy;
mod state;

use std::sync::Arc;
use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use tracing::info;

use discovery::{Discovery, EndpointDiscovery, StaticDiscovery};
use state::{FleetState, ReplicaSpec};

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "arf-router",
    about = "Stateless fleet router — load-balances OpenAI requests across N arf-serve replicas",
    version
)]
struct Args {
    /// Replica base URLs (repeatable). Format: [model=]url.
    ///
    /// Examples:
    ///   --replica http://10.0.0.1:8080          (serves any model)
    ///   --replica llama-3.2-1b=http://h:8080    (serves only llama-3.2-1b)
    ///
    /// Multiple replicas may share a model tag to form a pool.
    /// When --discovery-url is set these become the seed/initial set.
    #[arg(long = "replica")]
    replicas: Vec<String>,

    /// Port for the router to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// How often (seconds) to poll each replica's /healthz.
    #[arg(long = "health-interval-secs", default_value_t = 2)]
    health_interval_secs: u64,

    /// Disable prefix-affinity routing. When set, the router uses pure
    /// least-outstanding-requests. Use for A/B comparison.
    #[arg(long = "no-affinity", default_value_t = false)]
    no_affinity: bool,

    /// URL of a discovery endpoint returning:
    ///   [{"url": "http://host:port"}, {"url": "...", "models": ["model-id"]}]
    ///
    /// When set, the router polls this URL every --discovery-interval-secs and
    /// updates the replica list dynamically.  --replica becomes the seed set.
    ///
    /// Consul/k8s/Nomad integration: point this at an endpoint (or a tiny
    /// sidecar adapter) that translates the orchestrator's service-catalog
    /// response into the generic schema above.
    #[arg(long = "discovery-url")]
    discovery_url: Option<String>,

    /// How often (seconds) to refresh the discovery endpoint.
    #[arg(long = "discovery-interval-secs", default_value_t = 5)]
    discovery_interval_secs: u64,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // Parse --replica [model=]url entries.
    let specs: Vec<ReplicaSpec> = args
        .replicas
        .iter()
        .map(|s| ReplicaSpec::parse(s))
        .collect();

    info!(
        replicas = ?args.replicas,
        discovery_url = ?args.discovery_url,
        discovery_interval_secs = args.discovery_interval_secs,
        port = args.port,
        health_interval_secs = args.health_interval_secs,
        affinity = !args.no_affinity,
        "arf-router starting"
    );

    // Build the shared reqwest client.
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(64)
        .build()
        .expect("failed to build reqwest client");

    // Require at least one replica when using static discovery.
    if args.discovery_url.is_none() && specs.is_empty() {
        eprintln!("error: at least one --replica is required when --discovery-url is not set");
        std::process::exit(1);
    }

    let state = Arc::new(FleetState::new(
        specs.clone(),
        client.clone(),
        !args.no_affinity,
    ));

    // Spawn the discovery refresh loop.
    let disc: Arc<dyn Discovery> = if let Some(url) = args.discovery_url {
        Arc::new(EndpointDiscovery::new(url, client.clone()))
    } else {
        Arc::new(StaticDiscovery(specs))
    };

    discovery::spawn(
        Arc::clone(&state),
        disc,
        Duration::from_secs(args.discovery_interval_secs),
    );

    // Spawn the health-check loop (reads live ArcSwap list each tick).
    health::spawn(
        Arc::clone(&state),
        Duration::from_secs(args.health_interval_secs),
        Duration::from_secs(1),
    );

    // Build the axum router.
    let app = Router::new()
        .route("/v1/chat/completions", post(proxy::chat_completions))
        .route("/v1/completions", post(proxy::completions))
        .route("/v1/models", get(proxy::models))
        .route("/healthz", get(proxy::healthz))
        .with_state(Arc::clone(&state));

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {addr}: {e}"));

    info!(addr = %addr, "router listening");

    axum::serve(listener, app).await.expect("axum serve failed");
}
