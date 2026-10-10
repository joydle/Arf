//! Fleet state: replica registry, in-flight tracking, health flags.
//!
//! ## Design
//!
//! ### Model-id sharding
//! Each `Replica` now carries a `models` list — the model ids it serves.
//! Empty = serves any model (back-compat default). The picker filters the
//! candidate set to replicas whose `models` contains the request model (or
//! all healthy replicas if the request omits `model` or no tagged replica
//! matches it — graceful back-compat).
//!
//! ### Dynamic membership (ArcSwap)
//! The live replica list is stored behind an `ArcSwap<Vec<Arc<Replica>>>`.
//! * **Hot path** (every request): single lock-free Acquire load — cheaper
//!   than `RwLock::read()` under contention.
//! * **Refresh path** (discovery loop): builds a new `Vec<Arc<Replica>>`
//!   REUSING the existing `Arc<Replica>` for each URL that is still present
//!   (match-by-URL) then calls `ArcSwap::store`. In-flight counts and health
//!   flags survive across refreshes for surviving URLs; departed replicas are
//!   simply dropped (the Arc lives until all guards holding it are dropped);
//!   new replicas start at `in_flight=0`, `healthy=false`.
//!
//! ### InFlightGuard → holds `Arc<Replica>`
//! The guard now holds `Arc<Replica>` instead of `(Arc<FleetState>, index)`.
//! This is the correctness-critical change for dynamic membership: an index
//! into the Vec would become invalid after a refresh, decrementing the wrong
//! replica. Holding the Arc means the decrement always targets the correct
//! replica regardless of whether it is still in the live list.
//!
//! ### Picker API
//! `pick_replica` / `pick_replica_affinity` are replaced by model-pool-aware
//! equivalents that accept `model: Option<&str>`. If model is `None` OR no
//! replica matches → candidates = all healthy (back-compat).
//!
//! HRW scoring hashes `(key, replica.url)` — the replica's stable identity —
//! so affinity is stable within a model pool across list refreshes AND across
//! pool reorderings (discovery endpoints don't guarantee list order; keying on
//! url, not position, prevents a no-op reorder from thrashing KV caches).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use reqwest::Client;

/// If the affinity-preferred replica's in-flight count exceeds the
/// least-loaded healthy replica's count by more than this value, affinity
/// is abandoned and least-outstanding routing is used instead.
///
/// Threshold rationale: `preferred.in_flight > min_in_flight * 2 + 4`
const AFFINITY_LOAD_OVERRIDE_THRESHOLD_MULTIPLIER: usize = 2;
const AFFINITY_LOAD_OVERRIDE_THRESHOLD_SLACK: usize = 4;

// ---------------------------------------------------------------------------
// ReplicaSpec (config / discovery output)
// ---------------------------------------------------------------------------

/// Parsed descriptor produced by `--replica [model=]url` flags and
/// `Discovery::resolve`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicaSpec {
    /// Base URL, e.g. `"http://10.0.0.1:8080"`. No trailing slash.
    pub url: String,
    /// Model ids this replica serves. Empty = serves any model (back-compat).
    pub models: Vec<String>,
}

impl ReplicaSpec {
    /// Parse `[model=]url` as given on the CLI.
    ///
    /// Rule: if a `=` appears before the first `://`, the prefix is the model
    /// tag and the suffix is the URL. Otherwise the whole string is the URL.
    ///
    /// Examples:
    /// - `"http://h:8080"` → `models=[]`, `url="http://h:8080"`
    /// - `"llama-3.2-1b=http://h:8080"` → `models=["llama-3.2-1b"]`, url=...
    pub fn parse(s: &str) -> Self {
        // Look for '=' before the first '://' to avoid splitting a URL scheme.
        let eq_pos = s.find('=');
        let scheme_pos = s.find("://");

        match (eq_pos, scheme_pos) {
            (Some(eq), Some(scheme)) if eq < scheme => {
                let model = s[..eq].to_string();
                let url = s[eq + 1..].to_string();
                ReplicaSpec {
                    url,
                    models: vec![model],
                }
            }
            _ => ReplicaSpec {
                url: s.to_string(),
                models: vec![],
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Replica
// ---------------------------------------------------------------------------

pub struct Replica {
    /// Base URL, e.g. `"http://10.0.0.1:8080"`. No trailing slash.
    pub url: String,
    /// Model ids this replica serves. Empty = serves any model.
    pub models: Vec<String>,
    /// Router-tracked outstanding requests on this replica.
    pub in_flight: AtomicUsize,
    /// Set to `true` by the health-check loop when `/healthz` returns 200.
    pub healthy: AtomicBool,
}

impl Replica {
    pub fn new(spec: ReplicaSpec) -> Self {
        Self {
            url: spec.url,
            models: spec.models,
            in_flight: AtomicUsize::new(0),
            healthy: AtomicBool::new(false), // starts false; health loop flips it
        }
    }

    /// Returns true if this replica serves the given model (or serves any model).
    pub fn serves_model(&self, model: &str) -> bool {
        self.models.is_empty() || self.models.iter().any(|m| m == model)
    }
}

// ---------------------------------------------------------------------------
// Fleet state
// ---------------------------------------------------------------------------

pub struct FleetState {
    /// Live replica list. Lock-free reads on the hot path; the discovery
    /// refresh loop calls `store` with a new snapshot. Each `Arc<Replica>` is
    /// reused across refreshes (match by URL) so in-flight counts survive.
    pub replicas: ArcSwap<Vec<Arc<Replica>>>,
    /// One shared connection-pooled client for all outbound proxy calls.
    pub client: Client,
    /// Whether prefix-affinity routing is enabled.
    pub affinity_enabled: bool,
}

impl FleetState {
    pub fn new(specs: Vec<ReplicaSpec>, client: Client, affinity_enabled: bool) -> Self {
        let replicas: Vec<Arc<Replica>> = specs
            .into_iter()
            .map(|s| Arc::new(Replica::new(s)))
            .collect();
        Self {
            replicas: ArcSwap::from_pointee(replicas),
            client,
            affinity_enabled,
        }
    }

    /// Load the current replica list (lock-free Acquire load).
    #[inline]
    pub fn load_replicas(&self) -> arc_swap::Guard<Arc<Vec<Arc<Replica>>>> {
        self.replicas.load()
    }

    /// Replace the replica list.  Called ONLY by the discovery refresh loop.
    ///
    /// Reuses existing `Arc<Replica>` for URLs that are still present so that
    /// in-flight counts and healthy flags are preserved across refreshes.
    /// New URLs start at `in_flight=0`, `healthy=false`.
    pub fn refresh_replicas(&self, new_specs: Vec<ReplicaSpec>) {
        let current = self.replicas.load();
        let new_list: Vec<Arc<Replica>> = new_specs
            .into_iter()
            .map(|spec| {
                // Reuse existing Arc if URL still present — preserves in_flight/healthy.
                if let Some(existing) = current.iter().find(|r| r.url == spec.url) {
                    Arc::clone(existing)
                } else {
                    Arc::new(Replica::new(spec))
                }
            })
            .collect();
        self.replicas.store(Arc::new(new_list));
    }

    // ---------------------------------------------------------------------------
    // Pickers (model-pool-scoped)
    // ---------------------------------------------------------------------------

    /// Return the candidate set for a model selection.
    ///
    /// Candidate set = healthy replicas where `models.is_empty() || models.contains(model)`.
    /// If `model` is None OR the filtered set is empty → all healthy replicas
    /// (back-compat / graceful degradation).
    ///
    /// Returns `(candidates, is_full_pool)` where `is_full_pool` is true when
    /// we fell back to all-healthy (used only for logging; not exposed to callers).
    fn candidate_set<'a>(
        replicas: &'a [Arc<Replica>],
        model: Option<&str>,
    ) -> Vec<&'a Arc<Replica>> {
        if let Some(m) = model {
            // First try the model-specific pool.
            let pool: Vec<&Arc<Replica>> = replicas
                .iter()
                .filter(|r| r.healthy.load(Ordering::Relaxed) && r.serves_model(m))
                .collect();
            if !pool.is_empty() {
                return pool;
            }
        }
        // Fallback: all healthy replicas (model=None, or unknown model).
        replicas
            .iter()
            .filter(|r| r.healthy.load(Ordering::Relaxed))
            .collect()
    }

    /// Least-outstanding-requests pick, scoped to the model pool.
    pub fn pick_replica(&self, model: Option<&str>) -> Option<Arc<Replica>> {
        let replicas = self.replicas.load();
        let candidates = Self::candidate_set(&replicas, model);

        candidates
            .into_iter()
            .min_by_key(|r| r.in_flight.load(Ordering::Relaxed))
            .cloned()
    }

    /// Affinity-aware pick (HRW over the model pool), scoped to the model pool.
    ///
    /// If `key` is `None` → identical to `pick_replica(model)` (pure LOR).
    ///
    /// HRW is applied over the candidate subset (not global indices) so:
    /// * Affinity is stable within a model pool.
    /// * A pool refresh only remaps prefixes whose home was in the changed
    ///   portion of the pool — stable-under-add/remove within the pool.
    pub fn pick_replica_affinity(
        &self,
        key: Option<u64>,
        model: Option<&str>,
    ) -> Option<Arc<Replica>> {
        let Some(k) = key else {
            return self.pick_replica(model);
        };

        let replicas = self.replicas.load();
        let candidates = Self::candidate_set(&replicas, model);
        if candidates.is_empty() {
            return None;
        }

        // Single pass: HRW preferred + min_in_flight for load-override check.
        // HRW scores over (key, replica.url) — the replica's STABLE IDENTITY, not
        // its candidate position. A discovery refresh that merely reorders an
        // unchanged pool (Consul/k8s endpoints don't guarantee list order) must
        // NOT remap affinity — keying on url gives proper rendezvous stability so
        // a given prefix stays on its home replica across refreshes (the whole
        // point of R2: don't thrash KV caches when membership hasn't changed).
        let mut hrw_candidate: Option<&Arc<Replica>> = None;
        let mut hrw_score: u64 = 0;
        let mut min_in_flight = usize::MAX;
        let mut lor_candidate: Option<&Arc<Replica>> = None;

        for r in candidates.iter() {
            let in_flight = r.in_flight.load(Ordering::Relaxed);

            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            r.url.hash(&mut h);
            let score = h.finish();

            if hrw_candidate.is_none() || score > hrw_score {
                hrw_score = score;
                hrw_candidate = Some(r);
            }

            if in_flight < min_in_flight {
                min_in_flight = in_flight;
                lor_candidate = Some(r);
            }
        }

        let preferred = hrw_candidate?;
        let preferred_in_flight = preferred.in_flight.load(Ordering::Relaxed);
        let overload_threshold = min_in_flight
            .saturating_mul(AFFINITY_LOAD_OVERRIDE_THRESHOLD_MULTIPLIER)
            .saturating_add(AFFINITY_LOAD_OVERRIDE_THRESHOLD_SLACK);

        if preferred_in_flight > overload_threshold {
            lor_candidate.cloned()
        } else {
            Some(Arc::clone(preferred))
        }
    }

    /// Number of currently healthy replicas (across all pools).
    pub fn healthy_count(&self) -> usize {
        self.replicas
            .load()
            .iter()
            .filter(|r| r.healthy.load(Ordering::Relaxed))
            .count()
    }

    /// Distinct model ids across all replicas (union of all tags).
    /// Replicas with empty `models` (serves-any) are not included in the tag
    /// list (they contribute to the catch-all pool but not to the named set).
    pub fn known_models(&self) -> Vec<String> {
        let replicas = self.replicas.load();
        let mut models: Vec<String> = replicas
            .iter()
            .flat_map(|r| r.models.iter().cloned())
            .collect();
        models.sort_unstable();
        models.dedup();
        models
    }
}

// ---------------------------------------------------------------------------
// In-flight Drop guard  (holds Arc<Replica>, NOT a Vec index)
// ---------------------------------------------------------------------------

/// Holds an `Arc<Replica>` and decrements its `in_flight` counter on `Drop`.
///
/// ## Why `Arc<Replica>` instead of `(Arc<FleetState>, index)`
///
/// The replica list is dynamic: a discovery refresh replaces the `Vec` stored
/// in `FleetState::replicas`.  If the guard held a `(state, idx)` pair, the
/// Drop impl would index into `state.replicas.load()[idx]` which is WRONG
/// after a refresh — the slot at `idx` may now be a different replica.
///
/// Holding the `Arc<Replica>` directly means the decrement always targets the
/// exact replica that was incremented, regardless of list mutations.  The Arc
/// also keeps the `Replica` alive until the last guard is dropped, so a
/// departed replica's in-flight count can still drain correctly.
///
/// # Invariant
/// One `InFlightGuard` per incremented `in_flight`. Never clone or copy.
pub struct InFlightGuard {
    pub replica: Arc<Replica>,
}

impl InFlightGuard {
    /// Increment the counter for `replica` and return the guard.
    pub fn acquire(replica: Arc<Replica>) -> Self {
        replica.in_flight.fetch_add(1, Ordering::Relaxed);
        Self { replica }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        let prev = self.replica.in_flight.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(
            prev > 0,
            "in_flight underflow on replica {}",
            self.replica.url
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(n: usize) -> Arc<FleetState> {
        let s: Vec<ReplicaSpec> = (0..n)
            .map(|i| ReplicaSpec {
                url: format!("http://127.0.0.1:{}", 8000 + i),
                models: vec![],
            })
            .collect();
        let client = Client::new();
        Arc::new(FleetState::new(s, client, true))
    }

    fn mark_healthy(state: &FleetState, idx: usize) {
        state.replicas.load()[idx]
            .healthy
            .store(true, Ordering::Relaxed);
    }

    fn replica_at(state: &FleetState, idx: usize) -> Arc<Replica> {
        Arc::clone(&state.replicas.load()[idx])
    }

    // ---- ReplicaSpec::parse ----

    #[test]
    fn parse_bare_url() {
        let s = ReplicaSpec::parse("http://h:8080");
        assert_eq!(s.url, "http://h:8080");
        assert!(s.models.is_empty());
    }

    #[test]
    fn parse_model_eq_url() {
        let s = ReplicaSpec::parse("llama-3.2-1b=http://h:8080");
        assert_eq!(s.url, "http://h:8080");
        assert_eq!(s.models, vec!["llama-3.2-1b"]);
    }

    #[test]
    fn parse_url_with_eq_in_scheme_not_split() {
        // The `=` in a query param appears AFTER `://` — must not be split.
        let s = ReplicaSpec::parse("http://h:8080/path?foo=bar");
        assert_eq!(s.url, "http://h:8080/path?foo=bar");
        assert!(s.models.is_empty());
    }

    // ---- Basic pick_replica (no model filter) ----

    #[test]
    fn pick_none_when_all_unhealthy() {
        let s = make_state(3);
        assert!(s.pick_replica(None).is_none());
    }

    #[test]
    fn pick_only_healthy() {
        let s = make_state(3);
        mark_healthy(&s, 1);
        let r = s.pick_replica(None).unwrap();
        assert_eq!(r.url, "http://127.0.0.1:8001");
    }

    #[test]
    fn pick_lowest_in_flight() {
        let s = make_state(3);
        mark_healthy(&s, 0);
        mark_healthy(&s, 1);
        mark_healthy(&s, 2);
        replica_at(&s, 0).in_flight.store(5, Ordering::Relaxed);
        replica_at(&s, 1).in_flight.store(3, Ordering::Relaxed);
        replica_at(&s, 2).in_flight.store(1, Ordering::Relaxed);
        let r = s.pick_replica(None).unwrap();
        assert_eq!(r.url, "http://127.0.0.1:8002");
    }

    #[test]
    fn pick_skips_unhealthy() {
        let s = make_state(3);
        replica_at(&s, 0).in_flight.store(0, Ordering::Relaxed);
        mark_healthy(&s, 1);
        replica_at(&s, 1).in_flight.store(5, Ordering::Relaxed);
        let r = s.pick_replica(None).unwrap();
        assert_eq!(r.url, "http://127.0.0.1:8001");
    }

    // ---- Model-id sharding ----

    #[test]
    fn sharding_picks_tagged_replica_for_model() {
        let specs = vec![
            ReplicaSpec {
                url: "http://a:8080".into(),
                models: vec!["llama".into()],
            },
            ReplicaSpec {
                url: "http://b:8080".into(),
                models: vec!["mistral".into()],
            },
        ];
        let state = Arc::new(FleetState::new(specs, Client::new(), true));
        // Mark both healthy.
        state.replicas.load()[0]
            .healthy
            .store(true, Ordering::Relaxed);
        state.replicas.load()[1]
            .healthy
            .store(true, Ordering::Relaxed);

        let r = state.pick_replica(Some("llama")).unwrap();
        assert_eq!(
            r.url, "http://a:8080",
            "llama request must route to replica a"
        );

        let r2 = state.pick_replica(Some("mistral")).unwrap();
        assert_eq!(
            r2.url, "http://b:8080",
            "mistral request must route to replica b"
        );
    }

    #[test]
    fn sharding_unknown_model_falls_back_to_all_healthy() {
        let specs = vec![
            ReplicaSpec {
                url: "http://a:8080".into(),
                models: vec!["llama".into()],
            },
            ReplicaSpec {
                url: "http://b:8080".into(),
                models: vec!["mistral".into()],
            },
        ];
        let state = Arc::new(FleetState::new(specs, Client::new(), true));
        state.replicas.load()[0]
            .healthy
            .store(true, Ordering::Relaxed);
        state.replicas.load()[1]
            .healthy
            .store(true, Ordering::Relaxed);

        // "unknown" model → no match → fallback to all healthy
        let r = state.pick_replica(Some("unknown-model"));
        assert!(r.is_some(), "unknown model must not return None");
    }

    #[test]
    fn sharding_none_model_falls_back_to_all_healthy() {
        let specs = vec![ReplicaSpec {
            url: "http://a:8080".into(),
            models: vec!["llama".into()],
        }];
        let state = Arc::new(FleetState::new(specs, Client::new(), true));
        state.replicas.load()[0]
            .healthy
            .store(true, Ordering::Relaxed);

        // model=None → all healthy regardless of tags
        let r = state.pick_replica(None);
        assert!(r.is_some());
    }

    #[test]
    fn sharding_untagged_replica_serves_any_model() {
        let specs = vec![
            ReplicaSpec {
                url: "http://a:8080".into(),
                models: vec![],
            }, // serves any
        ];
        let state = Arc::new(FleetState::new(specs, Client::new(), true));
        state.replicas.load()[0]
            .healthy
            .store(true, Ordering::Relaxed);

        let r = state.pick_replica(Some("some-model")).unwrap();
        assert_eq!(r.url, "http://a:8080");
    }

    // ---- Affinity within a pool ----

    #[test]
    fn affinity_stable_across_pool_reorder() {
        // The fleet-scale invariant: a discovery refresh that merely
        // REORDERS an unchanged pool must NOT remap affinity (HRW keys on url,
        // not candidate position) — else every 5s refresh thrashes KV caches.
        let a = ReplicaSpec {
            url: "http://a:8080".into(),
            models: vec![],
        };
        let b = ReplicaSpec {
            url: "http://b:8080".into(),
            models: vec![],
        };
        let c = ReplicaSpec {
            url: "http://d:8080".into(),
            models: vec![],
        };

        let state = Arc::new(FleetState::new(
            vec![a.clone(), b.clone(), c.clone()],
            Client::new(),
            true,
        ));
        for r in state.replicas.load().iter() {
            r.healthy.store(true, Ordering::Relaxed);
        }
        let key = Some(0xDEAD_BEEFu64);
        let before = state.pick_replica_affinity(key, None).unwrap().url.clone();

        // Refresh with the SAME members in a DIFFERENT order (what an unordered
        // Consul/k8s endpoint returns). All still healthy after (same Arcs reused).
        state.refresh_replicas(vec![c, a, b]);
        for r in state.replicas.load().iter() {
            r.healthy.store(true, Ordering::Relaxed);
        }
        let after = state.pick_replica_affinity(key, None).unwrap().url.clone();
        assert_eq!(
            before, after,
            "reordering an unchanged pool must not remap affinity (key on url, not position)"
        );
    }

    #[test]
    fn guard_drains_count_on_departed_replica() {
        // The other guard case: a replica departs the list
        // while a request is still in flight. The guard holds the Arc<Replica>, so
        // its Drop still decrements the (now-orphaned) replica's count correctly —
        // no leak, no wrong-replica decrement.
        let state = make_state(2);
        let departing = replica_at(&state, 0);
        let guard = InFlightGuard::acquire(Arc::clone(&departing));
        assert_eq!(departing.in_flight.load(Ordering::Relaxed), 1);

        // Refresh: replica 0 departs (only replica 1's URL survives).
        let survivor_url = state.replicas.load()[1].url.clone();
        state.refresh_replicas(vec![ReplicaSpec {
            url: survivor_url,
            models: vec![],
        }]);
        // The departed replica is gone from the live list...
        assert!(state
            .replicas
            .load()
            .iter()
            .all(|r| !Arc::ptr_eq(r, &departing)));
        // ...but the guard still holds it; dropping decrements the orphan, no panic.
        assert_eq!(departing.in_flight.load(Ordering::Relaxed), 1);
        drop(guard);
        assert_eq!(
            departing.in_flight.load(Ordering::Relaxed),
            0,
            "guard must drain the departed replica's count via the held Arc"
        );
    }

    #[test]
    fn affinity_same_key_same_model_same_replica() {
        let specs = vec![
            ReplicaSpec {
                url: "http://a:8080".into(),
                models: vec!["llama".into()],
            },
            ReplicaSpec {
                url: "http://b:8080".into(),
                models: vec!["llama".into()],
            },
            ReplicaSpec {
                url: "http://c:8080".into(),
                models: vec!["mistral".into()],
            },
        ];
        let state = Arc::new(FleetState::new(specs, Client::new(), true));
        for r in state.replicas.load().iter() {
            r.healthy.store(true, Ordering::Relaxed);
        }

        let key = Some(42u64);
        let first = state.pick_replica_affinity(key, Some("llama")).unwrap();
        // Must always route to the same replica within the llama pool.
        for _ in 0..10 {
            let r = state.pick_replica_affinity(key, Some("llama")).unwrap();
            assert_eq!(
                r.url, first.url,
                "same key + same model must route to same replica"
            );
        }
        // Must not route to the mistral replica.
        assert_ne!(
            first.url, "http://c:8080",
            "affinity must stay within the model pool"
        );
    }

    // ---- in-flight survives refresh ----

    #[test]
    fn in_flight_survives_refresh() {
        let initial = vec![
            ReplicaSpec {
                url: "http://a:8080".into(),
                models: vec![],
            },
            ReplicaSpec {
                url: "http://b:8080".into(),
                models: vec![],
            },
        ];
        let state = Arc::new(FleetState::new(initial, Client::new(), true));

        // Grab Arc to replica a and increment its in-flight.
        let replica_a = Arc::clone(&state.replicas.load()[0]);
        replica_a.in_flight.store(7, Ordering::Relaxed);

        // Refresh: a stays, b departs, c is new.
        state.refresh_replicas(vec![
            ReplicaSpec {
                url: "http://a:8080".into(),
                models: vec![],
            },
            ReplicaSpec {
                url: "http://c:8080".into(),
                models: vec![],
            },
        ]);

        let after = state.replicas.load();
        // a must be the SAME Arc (in_flight preserved).
        let new_a = after
            .iter()
            .find(|r| r.url == "http://a:8080")
            .expect("a must survive");
        assert!(
            Arc::ptr_eq(new_a, &replica_a),
            "refresh must reuse Arc for surviving URL"
        );
        assert_eq!(
            new_a.in_flight.load(Ordering::Relaxed),
            7,
            "in_flight must be preserved"
        );

        // c must be fresh (in_flight=0, healthy=false).
        let new_c = after
            .iter()
            .find(|r| r.url == "http://c:8080")
            .expect("c must be added");
        assert_eq!(
            new_c.in_flight.load(Ordering::Relaxed),
            0,
            "new replica starts at 0 in_flight"
        );
        assert!(
            !new_c.healthy.load(Ordering::Relaxed),
            "new replica starts unhealthy"
        );

        // b must not be present.
        assert!(
            !after.iter().any(|r| r.url == "http://b:8080"),
            "departed b must be gone"
        );
    }

    // ---- InFlightGuard holds Arc<Replica> ----

    #[test]
    fn guard_holds_arc_replica_survives_list_churn() {
        let specs = vec![ReplicaSpec {
            url: "http://a:8080".into(),
            models: vec![],
        }];
        let state = Arc::new(FleetState::new(specs, Client::new(), true));
        let replica_a = Arc::clone(&state.replicas.load()[0]);
        replica_a.healthy.store(true, Ordering::Relaxed);

        // Acquire guard.
        let guard = InFlightGuard::acquire(Arc::clone(&replica_a));
        assert_eq!(replica_a.in_flight.load(Ordering::Relaxed), 1);

        // Refresh replicas (same URL, so same Arc is reused — but even if it
        // weren't, the guard still holds the original Arc).
        state.refresh_replicas(vec![ReplicaSpec {
            url: "http://a:8080".into(),
            models: vec![],
        }]);

        // Drop guard → the ORIGINAL replica's in_flight decrements correctly.
        drop(guard);
        assert_eq!(
            replica_a.in_flight.load(Ordering::Relaxed),
            0,
            "guard decrement targets correct replica after churn"
        );
    }

    #[test]
    fn guard_increments_decrements() {
        let s = make_state(1);
        mark_healthy(&s, 0);
        let r = replica_at(&s, 0);
        assert_eq!(r.in_flight.load(Ordering::Relaxed), 0);
        let g = InFlightGuard::acquire(Arc::clone(&r));
        assert_eq!(r.in_flight.load(Ordering::Relaxed), 1);
        drop(g);
        assert_eq!(r.in_flight.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn multiple_guards_all_decrement() {
        let s = make_state(1);
        mark_healthy(&s, 0);
        let r = replica_at(&s, 0);
        let g1 = InFlightGuard::acquire(Arc::clone(&r));
        let g2 = InFlightGuard::acquire(Arc::clone(&r));
        let g3 = InFlightGuard::acquire(Arc::clone(&r));
        assert_eq!(r.in_flight.load(Ordering::Relaxed), 3);
        drop(g1);
        assert_eq!(r.in_flight.load(Ordering::Relaxed), 2);
        drop(g2);
        assert_eq!(r.in_flight.load(Ordering::Relaxed), 1);
        drop(g3);
        assert_eq!(r.in_flight.load(Ordering::Relaxed), 0);
    }

    // ---- known_models ----

    #[test]
    fn known_models_union() {
        let specs = vec![
            ReplicaSpec {
                url: "http://a:8080".into(),
                models: vec!["llama".into(), "phi".into()],
            },
            ReplicaSpec {
                url: "http://b:8080".into(),
                models: vec!["llama".into()],
            },
            ReplicaSpec {
                url: "http://c:8080".into(),
                models: vec![],
            }, // serves any, not listed
        ];
        let state = FleetState::new(specs, Client::new(), true);
        let mut models = state.known_models();
        models.sort();
        assert_eq!(models, vec!["llama", "phi"]);
    }

    // ---- Affinity back-compat ----

    #[test]
    fn affinity_none_key_equals_pick_replica() {
        let s = make_state(3);
        mark_healthy(&s, 0);
        mark_healthy(&s, 2);
        replica_at(&s, 0).in_flight.store(5, Ordering::Relaxed);
        replica_at(&s, 2).in_flight.store(1, Ordering::Relaxed);

        let via_affinity = s.pick_replica_affinity(None, None).map(|r| r.url.clone());
        let via_plain = s.pick_replica(None).map(|r| r.url.clone());
        assert_eq!(via_affinity, via_plain, "key=None must equal pick_replica");
    }

    #[test]
    fn affinity_returns_none_when_all_unhealthy() {
        let s = make_state(3);
        assert!(s.pick_replica_affinity(Some(12345), None).is_none());
        assert!(s.pick_replica_affinity(None, None).is_none());
    }

    #[test]
    fn affinity_falls_back_to_lor_when_preferred_overloaded() {
        let s = make_state(3);
        mark_healthy(&s, 0);
        mark_healthy(&s, 1);
        mark_healthy(&s, 2);

        let key = Some(99u64);
        let preferred = s.pick_replica_affinity(key, None).expect("should pick");
        preferred.in_flight.store(100, Ordering::Relaxed);

        let fallback = s
            .pick_replica_affinity(key, None)
            .expect("should still pick");
        assert_ne!(
            fallback.url, preferred.url,
            "overloaded preferred must be abandoned"
        );
        assert_eq!(
            fallback.in_flight.load(Ordering::Relaxed),
            0,
            "fallback must be LOR"
        );
    }

    #[test]
    fn affinity_different_keys_spread() {
        let s = make_state(4);
        for i in 0..4 {
            mark_healthy(&s, i);
        }
        let picked: std::collections::HashSet<String> = (0u64..64)
            .filter_map(|k| {
                s.pick_replica_affinity(Some(k), None)
                    .map(|r| r.url.clone())
            })
            .collect();
        assert!(
            picked.len() > 1,
            "64 keys should spread across replicas; got {:?}",
            picked
        );
    }

    #[test]
    fn healthy_count() {
        let s = make_state(4);
        assert_eq!(s.healthy_count(), 0);
        mark_healthy(&s, 0);
        mark_healthy(&s, 2);
        assert_eq!(s.healthy_count(), 2);
    }
}
