//! Proxy handlers: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
//! and the router's own `/healthz`.
//!
//! ## Dynamic-membership changes
//!
//! ### Model-id sharding
//! `proxy_completions` now extracts the request `model` field and passes it to
//! `pick_replica_affinity(key, model)`. The picker narrows candidates to
//! replicas tagged for that model (or all healthy if model is absent/unknown).
//!
//! ### InFlightGuard → holds `Arc<Replica>`
//! Guards now wrap `Arc<Replica>` directly (not a Vec index). This is
//! correct under list churn: the guard always decrements the EXACT replica
//! it incremented, even if the live list has been refreshed since.
//!
//! ### /v1/models from tags
//! When the router has model tags (at least one tagged replica), it synthesises
//! the response from the union of known model ids — zero upstream calls, always
//! accurate. When no tags are configured (back-compat), it proxies one healthy
//! replica as before.
//!
//! ## Routing / retry rule
//!
//! For the two completions endpoints:
//!
//! 1. Pick a replica (now model-scoped):
//!    - Attempt 0: affinity key if enabled.
//!    - Attempt > 0: LOR (drop affinity after a failure — the affine replica failed).
//! 2. Wrap in `InFlightGuard` (Drop decrements).
//! 3. Forward body verbatim to `{replica_url}{path}`.
//! 4. Pre-token failure: mark unhealthy, drop guard, retry.
//! 5. Post-first-byte: terminate stream, mark unhealthy, no retry.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::json;
use tracing::{debug, warn};

use crate::affinity::{extract_affinity_key, extract_model};
use crate::state::{FleetState, InFlightGuard};

// ---------------------------------------------------------------------------
// Shared helper: forward a completions request (streaming or non-streaming)
// ---------------------------------------------------------------------------

fn copy_content_type(src: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    if let Some(ct) = src.get(header::CONTENT_TYPE) {
        out.insert(header::CONTENT_TYPE, ct.clone());
    }
    out
}

/// Attempt to forward a completions request. Retries up to `replicas.len()`
/// times on pre-token failures.
async fn proxy_completions(
    state: Arc<FleetState>,
    path: &str,
    headers: HeaderMap,
    body_bytes: Bytes,
) -> Response {
    // Compute affinity key + model ONCE before the retry loop.
    let affinity_key: Option<u64> = if state.affinity_enabled {
        extract_affinity_key(path, &body_bytes)
    } else {
        None
    };
    let model: Option<String> = extract_model(&body_bytes);
    let model_ref: Option<&str> = model.as_deref();

    // Max attempts = current replica count (snapshot; good enough for a bound).
    let max_attempts = state.replicas.load().len().max(1);

    for attempt in 0..max_attempts {
        // --- 1. Pick a replica (model-pool-scoped) -------------------------
        let key_for_attempt = if attempt == 0 { affinity_key } else { None };

        let Some(replica) = state.pick_replica_affinity(key_for_attempt, model_ref) else {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(json!({ "error": "no healthy replicas available" })),
            )
                .into_response();
        };

        // --- 2. Acquire the in-flight guard --------------------------------
        // Guard holds Arc<Replica> directly — correct under list churn.
        let replica_url_str = format!("{}{}", replica.url, path);
        let guard = InFlightGuard::acquire(Arc::clone(&replica));

        debug!(
            attempt,
            replica = %replica.url,
            model = ?model_ref,
            "dispatching request"
        );

        // --- 3. Forward the request body -----------------------------------
        let req = state
            .client
            .post(&replica_url_str)
            .headers(copy_content_type(&headers))
            .body(body_bytes.clone());

        let upstream = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                warn!(replica = %replica.url, error = %e, "replica send() failed — marking unhealthy");
                replica.healthy.store(false, Ordering::Relaxed);
                drop(guard);
                continue;
            }
        };

        let status = upstream.status();

        // --- 4a. Pre-token: 5xx before any body bytes ----------------------
        if status.is_server_error() {
            warn!(
                replica = %replica.url,
                status = %status,
                "replica returned 5xx (pre-token) — marking unhealthy"
            );
            replica.healthy.store(false, Ordering::Relaxed);
            drop(guard);
            continue;
        }

        // --- 5. Build the passthrough response -----------------------------
        let content_type = upstream
            .headers()
            .get(header::CONTENT_TYPE)
            .cloned()
            .unwrap_or_else(|| HeaderValue::from_static("application/json"));

        let upstream_status =
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        let byte_stream = upstream.bytes_stream();
        let replica_url_for_log = replica.url.clone();
        let replica_for_err = Arc::clone(&replica);

        let forwarding_stream = async_stream::stream! {
            let _guard = guard; // guard lives until this stream is dropped
            tokio::pin!(byte_stream);
            let mut first_byte_sent = false;

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        first_byte_sent = true;
                        yield Ok::<Bytes, std::io::Error>(chunk);
                    }
                    Err(e) => {
                        if first_byte_sent {
                            warn!(
                                replica = %replica_url_for_log,
                                error = %e,
                                "mid-stream error (post-first-byte) — terminating client stream"
                            );
                        } else {
                            warn!(
                                replica = %replica_url_for_log,
                                error = %e,
                                "stream error before first byte — marking unhealthy"
                            );
                        }
                        replica_for_err.healthy.store(false, Ordering::Relaxed);
                        return;
                    }
                }
            }
        };

        let body = Body::from_stream(forwarding_stream);
        let response = Response::builder()
            .status(upstream_status)
            .header(header::CONTENT_TYPE, content_type)
            .body(body)
            .unwrap_or_else(|_| {
                (StatusCode::INTERNAL_SERVER_ERROR, "response build error").into_response()
            });
        return response;
    }

    (
        StatusCode::SERVICE_UNAVAILABLE,
        axum::Json(json!({ "error": "all replicas failed or are unhealthy" })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// axum handlers
// ---------------------------------------------------------------------------

/// `POST /v1/chat/completions`
pub async fn chat_completions(State(state): State<Arc<FleetState>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, 4 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(json!({"error": format!("failed to read request body: {e}")})),
            )
                .into_response();
        }
    };
    proxy_completions(state, "/v1/chat/completions", parts.headers, body_bytes).await
}

/// `POST /v1/completions`
pub async fn completions(State(state): State<Arc<FleetState>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, 4 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(json!({"error": format!("failed to read request body: {e}")})),
            )
                .into_response();
        }
    };
    proxy_completions(state, "/v1/completions", parts.headers, body_bytes).await
}

/// `GET /v1/models`
///
/// ## Behaviour under dynamic membership
///
/// If the router has at least one *tagged* replica (i.e. a replica with a
/// non-empty `models` list), it synthesises the OpenAI `/v1/models` response
/// from the union of all known model ids — no upstream call needed, zero
/// latency, always accurate.
///
/// If no tags are configured (back-compat / all replicas serve any model),
/// it proxies one healthy replica as before.
pub async fn models(State(state): State<Arc<FleetState>>) -> Response {
    let known = state.known_models();
    if !known.is_empty() {
        // Synthesise from tags.
        let data: Vec<serde_json::Value> = known
            .into_iter()
            .map(|id| {
                json!({
                    "id": id,
                    "object": "model",
                    "owned_by": "arf-router"
                })
            })
            .collect();
        // axum::Json sets the JSON content-type and is itself IntoResponse.
        return (
            StatusCode::OK,
            axum::Json(json!({ "object": "list", "data": data })),
        )
            .into_response();
    }

    // Back-compat: no tags → proxy one healthy replica.
    let Some(replica) = state.pick_replica(None) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(json!({"error": "no healthy replicas available"})),
        )
            .into_response();
    };

    let url = format!("{}/v1/models", replica.url);
    match state.client.get(&url).send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let body = resp.bytes().await.unwrap_or_default();
            (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            axum::Json(json!({"error": format!("failed to contact replica: {e}")})),
        )
            .into_response(),
    }
}

/// `GET /healthz` — the router's own health endpoint.
pub async fn healthz(State(state): State<Arc<FleetState>>) -> Response {
    let healthy = state.healthy_count();
    let total = state.replicas.load().len();
    let status = if healthy > 0 {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        axum::Json(json!({
            "status": if healthy > 0 { "ok" } else { "degraded" },
            "healthy_replicas": healthy,
            "total": total,
        })),
    )
        .into_response()
}
