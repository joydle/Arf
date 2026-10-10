//! Request lifecycle and continuous-batching scheduling.

pub mod block_manager;
pub mod request;
#[allow(clippy::module_inception)]
pub mod scheduler;

pub use block_manager::BlockManager;
pub use request::{FinishReason, ImagePrompt, Request, Sequence, SequenceStatus, StopCheck};
pub use scheduler::{
    junction_snapshots_enabled, session_start_snapshots_enabled, tools_anchor_snapshots_enabled,
    BatchPlan, RequestOutput, Scheduler, SeqPlan, ANCHOR_MIN_TOKENS, JUNCTION_MIN_EXTRA_TOKENS,
    JUNCTION_MIN_TOKENS, SNAPSHOT_TAIL_TOKENS,
};
