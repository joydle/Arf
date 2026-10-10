//! Opt-in Postgres-backed chat persistence (feature = "postgres").
//!
//! Zero-DB by default: [`init_pool`] reads `DATABASE_URL` from the environment and
//! returns `None` when it is unset OR the connection fails (it logs a warning and
//! NEVER panics), so the server runs identically with no database. When a pool is
//! present, [`run_migrations`] creates the `conversations` / `messages` tables IF
//! NOT EXISTS, and the CRUD helpers below back the `/conversations` HTTP routes.
//!
//! All queries are RUNTIME `sqlx::query` / `query_as` (NOT the compile-time `query!`
//! macro), so this crate builds with no database reachable. Timestamps are rendered
//! to TEXT in SQL via `to_char(...)`, so there is no chrono dependency.

use serde::Serialize;
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;

/// ISO-8601-ish timestamp format applied in SQL via `to_char`. Keeps timestamps as
/// plain TEXT in the JSON the API returns, avoiding a chrono / time dependency.
const TS_FMT: &str = "YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"";

/// A conversation row (no messages). Returned by the list endpoint.
#[derive(Debug, Serialize)]
pub struct Conversation {
    pub id: String,
    pub title: Option<String>,
    pub model: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// One chat message within a conversation.
#[derive(Debug, Serialize)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub idx: i32,
    pub created_at: String,
}

/// A conversation together with its ordered messages (the detail endpoint).
#[derive(Debug, Serialize)]
pub struct ConversationDetail {
    #[serde(flatten)]
    pub conversation: Conversation,
    pub messages: Vec<Message>,
}

/// Read `DATABASE_URL` and connect. Returns `None` (with a warning) when the var is
/// unset or the connection cannot be established — never panics, so the daemon keeps
/// serving inference with persistence simply disabled.
pub async fn init_pool() -> Option<PgPool> {
    let url = match std::env::var("DATABASE_URL") {
        Ok(u) if !u.trim().is_empty() => u,
        _ => {
            eprintln!(
                "persistence: DATABASE_URL not set — chat persistence DISABLED (zero-DB default)"
            );
            return None;
        }
    };
    match PgPoolOptions::new().max_connections(5).connect(&url).await {
        Ok(pool) => {
            eprintln!("persistence: connected to Postgres — chat persistence ENABLED");
            Some(pool)
        }
        Err(e) => {
            eprintln!(
                "persistence: failed to connect to DATABASE_URL ({e}) — chat persistence DISABLED"
            );
            None
        }
    }
}

/// Create the `conversations` and `messages` tables IF NOT EXISTS. Idempotent.
pub async fn run_migrations(pool: &PgPool) -> Result<(), sqlx::Error> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS conversations (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            title text,
            model text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS messages (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role text NOT NULL,
            content text NOT NULL,
            idx int NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )",
    )
    .execute(pool)
    .await?;

    Ok(())
}

/// List conversations, most-recently-updated first.
pub async fn list_conversations(pool: &PgPool) -> Result<Vec<Conversation>, sqlx::Error> {
    let sql = format!(
        "SELECT id::text AS id, title, model,
                to_char(created_at, '{TS_FMT}') AS created_at,
                to_char(updated_at, '{TS_FMT}') AS updated_at
         FROM conversations
         ORDER BY updated_at DESC"
    );
    let rows = sqlx::query(&sql).fetch_all(pool).await?;
    Ok(rows.into_iter().map(row_to_conversation).collect())
}

/// Create a new conversation and return it.
pub async fn create_conversation(
    pool: &PgPool,
    title: Option<&str>,
    model: Option<&str>,
) -> Result<Conversation, sqlx::Error> {
    let sql = format!(
        "INSERT INTO conversations (title, model)
         VALUES ($1, $2)
         RETURNING id::text AS id, title, model,
                   to_char(created_at, '{TS_FMT}') AS created_at,
                   to_char(updated_at, '{TS_FMT}') AS updated_at"
    );
    let row = sqlx::query(&sql)
        .bind(title)
        .bind(model)
        .fetch_one(pool)
        .await?;
    Ok(row_to_conversation(row))
}

/// Fetch one conversation with its ordered messages. `None` if the id is unknown.
pub async fn get_conversation_with_messages(
    pool: &PgPool,
    id: &str,
) -> Result<Option<ConversationDetail>, sqlx::Error> {
    let uuid = match uuid::Uuid::parse_str(id) {
        Ok(u) => u,
        Err(_) => return Ok(None), // not a uuid → no such conversation
    };

    let conv_sql = format!(
        "SELECT id::text AS id, title, model,
                to_char(created_at, '{TS_FMT}') AS created_at,
                to_char(updated_at, '{TS_FMT}') AS updated_at
         FROM conversations
         WHERE id = $1"
    );
    let conv_row = sqlx::query(&conv_sql)
        .bind(uuid)
        .fetch_optional(pool)
        .await?;
    let conversation = match conv_row {
        Some(r) => row_to_conversation(r),
        None => return Ok(None),
    };

    let msg_sql = format!(
        "SELECT id::text AS id, role, content, idx,
                to_char(created_at, '{TS_FMT}') AS created_at
         FROM messages
         WHERE conversation_id = $1
         ORDER BY idx ASC, created_at ASC"
    );
    let msg_rows = sqlx::query(&msg_sql).bind(uuid).fetch_all(pool).await?;
    let messages = msg_rows.into_iter().map(row_to_message).collect();

    Ok(Some(ConversationDetail {
        conversation,
        messages,
    }))
}

/// Append a message to a conversation and bump its `updated_at`. The message `idx`
/// is the next integer after the current max for that conversation (0-based).
/// Returns `None` if the conversation id is unknown.
pub async fn add_message(
    pool: &PgPool,
    conversation_id: &str,
    role: &str,
    content: &str,
) -> Result<Option<Message>, sqlx::Error> {
    let uuid = match uuid::Uuid::parse_str(conversation_id) {
        Ok(u) => u,
        Err(_) => return Ok(None),
    };

    // Reject unknown conversations up-front so we return 404, not a FK error.
    let exists = sqlx::query("SELECT 1 FROM conversations WHERE id = $1")
        .bind(uuid)
        .fetch_optional(pool)
        .await?;
    if exists.is_none() {
        return Ok(None);
    }

    let next_idx: i32 = sqlx::query(
        "SELECT COALESCE(MAX(idx) + 1, 0) AS next_idx FROM messages WHERE conversation_id = $1",
    )
    .bind(uuid)
    .fetch_one(pool)
    .await?
    .try_get("next_idx")?;

    let insert_sql = format!(
        "INSERT INTO messages (conversation_id, role, content, idx)
         VALUES ($1, $2, $3, $4)
         RETURNING id::text AS id, role, content, idx,
                   to_char(created_at, '{TS_FMT}') AS created_at"
    );
    let row = sqlx::query(&insert_sql)
        .bind(uuid)
        .bind(role)
        .bind(content)
        .bind(next_idx)
        .fetch_one(pool)
        .await?;

    sqlx::query("UPDATE conversations SET updated_at = now() WHERE id = $1")
        .bind(uuid)
        .execute(pool)
        .await?;

    Ok(Some(row_to_message(row)))
}

/// Delete a conversation (its messages cascade). Returns `true` if a row was removed.
pub async fn delete_conversation(pool: &PgPool, id: &str) -> Result<bool, sqlx::Error> {
    let uuid = match uuid::Uuid::parse_str(id) {
        Ok(u) => u,
        Err(_) => return Ok(false),
    };
    let result = sqlx::query("DELETE FROM conversations WHERE id = $1")
        .bind(uuid)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// ---------------------------------------------------------------------------
// Row → struct mappers (shared so SELECT column aliases live in one place).
// ---------------------------------------------------------------------------

fn row_to_conversation(row: sqlx::postgres::PgRow) -> Conversation {
    Conversation {
        id: row.get("id"),
        title: row.get("title"),
        model: row.get("model"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    }
}

fn row_to_message(row: sqlx::postgres::PgRow) -> Message {
    Message {
        id: row.get("id"),
        role: row.get("role"),
        content: row.get("content"),
        idx: row.get("idx"),
        created_at: row.get("created_at"),
    }
}
