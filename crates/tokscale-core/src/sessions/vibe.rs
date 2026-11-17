//! Mistral Vibe session parser
//!
//! Mistral Vibe (Mistral AI's terminal coding agent) stores session logs under
//! `~/.vibe/logs/session/` (or `$VIBE_HOME/logs/session/`).
//!
//! Directory layout (Vibe 2.x):
//!   ~/.vibe/logs/session/session_YYYYMMDD_HHMMSS_<short_id>/
//!     ├── meta.json
//!     └── messages.jsonl
//!
//! Legacy Vibe 1.x flat format:
//!   ~/.vibe/logs/session/session_*.json
//!
//! Usage statistics are recorded in `meta.json` (or the top-level `metadata`
//! key in legacy session files) under `stats`:
//! - `session_prompt_tokens`: overall prompt tokens (includes cached prompt tokens)
//! - `session_completion_tokens`: completion / output tokens
//! - `session_cached_tokens`: cached prompt tokens
//!
//! Cached tokens are a *subset* of the prompt tokens, not an addition to them —
//! upstream types the field that way and its own cost function subtracts the
//! cached count before charging the input rate — so the breakdown splits them
//! out instead of summing. Usage is session-granular: a Vibe transcript carries
//! no per-message model, timestamp, or token counts, so one message per session
//! is the finest row the format supports.
//!
//! Unified session store (Vibe 3.x, `store_format`
//! `mistral.vibe.unified-session-store/v1` in the session's `CURRENT` file):
//!   ~/.vibe/logs/session/unified/<session_id>/
//!     ├── meta.json            (no `stats`; model, times, workspace, title)
//!     ├── CURRENT
//!     ├── generations/<seq>/   (snapshots; no token counters)
//!     └── journal/<first_sequence>.jsonl
//!
//! Each journal row is one recovery record. A model call is an `action_result`
//! record whose `payload.result.type` is `completion_succeeded` (or
//! `completion_failed`) and whose `payload.result.result.usage` holds that
//! call's own counts. The same result is echoed into the following
//! `core_input` record, so only `action_result` rows are counted. See
//! [`parse_unified_journal`] for the usage semantics and what a compacted
//! session loses.

use super::utils::{file_modified_timestamp_ms, parse_timestamp_value};
use super::{normalize_workspace_key, workspace_label_from_key, UnifiedMessage};
use crate::{provider_identity, TokenBreakdown};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Alias of the model Vibe resolves an unpinned session to when the session
/// snapshot does not carry a `config.models` table to read it from. Upstream's
/// own default (`DEFAULT_ACTIVE_MODEL_CONFIG` in `vibe/core/config/vibe_schema.py`)
/// publishes this alias; `name` on the same record is the CLI-routed wire id
/// (`mistral-vibe-cli-latest`), which no pricing dataset carries.
const DEFAULT_MODEL_ALIAS: &str = "mistral-medium-latest";

/// Resolve the session's model from a Vibe metadata record.
///
/// `config.active_model` is the pinned alias, but it is **empty by default**:
/// upstream documents `""` as `UNPINNED_ACTIVE_MODEL`, the sentinel for "follow
/// the configured default" rather than missing data, and a session stays
/// unpinned until the user runs `/model`. A blank value therefore has to fall
/// through instead of being accepted as a model id — an empty `model_id` leaves
/// the Model column blank and, because no pricing row can match it, prices the
/// whole session at zero.
///
/// Resolution order mirrors upstream's `get_active_model`: the pinned alias,
/// then the session's own `config.models` table (single-entry for a default
/// install), then the published default alias.
fn resolve_model(meta: &Value) -> String {
    let pinned = meta
        .pointer("/config/active_model")
        .and_then(Value::as_str)
        .or_else(|| meta.get("active_model").and_then(Value::as_str))
        .or_else(|| meta.get("model").and_then(Value::as_str))
        .map(str::trim)
        .filter(|model| !model.is_empty());
    if let Some(model) = pinned {
        return model.to_string();
    }

    // An unpinned session still names the models it was configured with. Prefer
    // the record's own `alias` over its map key: both identify the same model,
    // but the alias is what the CLI displays and what upstream pins on `/model`.
    if let Some(models) = meta.pointer("/config/models").and_then(Value::as_object) {
        let mut aliases: Vec<&str> = models
            .iter()
            .map(|(key, entry)| {
                entry
                    .get("alias")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|alias| !alias.is_empty())
                    .unwrap_or(key.as_str())
            })
            .filter(|alias| !alias.is_empty())
            .collect();
        if aliases.len() == 1 {
            return aliases[0].to_string();
        }
        // More than one configured model and nothing pinned: the transcript
        // cannot say which one ran, so pick deterministically rather than
        // letting map iteration order decide.
        if aliases.len() > 1 {
            aliases.sort_unstable();
            if let Some(default) = aliases
                .iter()
                .find(|alias| **alias == DEFAULT_MODEL_ALIAS)
                .or_else(|| aliases.first())
            {
                return (*default).to_string();
            }
        }
    }

    DEFAULT_MODEL_ALIAS.to_string()
}

/// Returns the session directory when `path` is a unified-store journal
/// segment: `<root>/unified/<session_id>/journal/<sequence>.jsonl`.
pub(crate) fn unified_journal_session_dir(path: &Path) -> Option<&Path> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
        return None;
    }
    let journal_dir = path.parent()?;
    if journal_dir.file_name()? != "journal" {
        return None;
    }
    let session_dir = journal_dir.parent()?;
    if session_dir.parent()?.file_name()? != "unified" {
        return None;
    }
    Some(session_dir)
}

fn non_negative(usage: &Value, key: &str) -> i64 {
    usage.get(key).and_then(Value::as_i64).unwrap_or(0).max(0)
}

/// Split one unified-store `usage` record into additive buckets.
///
/// Every recorded call satisfies `total_tokens == input_tokens + output_tokens`,
/// so `cached_input_tokens` and `cache_write_tokens` are subsets of
/// `input_tokens` and `reasoning_tokens` is a subset of `output_tokens`. Vibe's
/// own stats agree: it reports `session_prompt_tokens = input_tokens` and
/// `session_cached_tokens = cached_input_tokens`. `TokenBreakdown` buckets are
/// additive, so each subset is split out (and clamped, so a malformed row cannot
/// drive a bucket negative) rather than counted on top.
fn unified_usage_tokens(usage: &Value) -> TokenBreakdown {
    let input = non_negative(usage, "input_tokens");
    let output = non_negative(usage, "output_tokens");
    let cache_read = non_negative(usage, "cached_input_tokens").min(input);
    let cache_write = non_negative(usage, "cache_write_tokens").min(input - cache_read);
    let reasoning = non_negative(usage, "reasoning_tokens").min(output);
    TokenBreakdown {
        input: input - cache_read - cache_write,
        output: output - reasoning,
        cache_read,
        cache_write,
        reasoning,
        ..Default::default()
    }
}

/// Parse one journal segment of a Vibe 3.x unified session.
///
/// Emits one message per recorded model call, keyed `vibe:<session>:<action>`
/// so a call that appears in two segments is counted once. The journal carries
/// no per-call timestamp or model, so those come from the sibling `meta.json`
/// exactly as for the 2.x layout.
///
/// Scanning the segments (not `meta.json`) is deliberate: the journal is the
/// file that grows with each call, so the per-file parse cache refreshes when
/// usage changes.
///
/// When Vibe compacts a long session it drops the journal segments before the
/// new snapshot, and neither the snapshots nor `meta.json` keep a cumulative
/// token count, so calls made before a compaction cannot be recovered. For such
/// a session the parsed usage is a lower bound.
fn parse_unified_journal(path: &Path, session_dir: &Path) -> Vec<UnifiedMessage> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };

    let meta = std::fs::read_to_string(session_dir.join("meta.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| Value::Object(Default::default()));

    let session_id = meta
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .or_else(|| {
            session_dir
                .file_name()
                .and_then(|name| name.to_str())
                .map(String::from)
        })
        .unwrap_or_else(|| "unknown".to_string());

    let timestamp = meta
        .get("end_time")
        .and_then(parse_timestamp_value)
        .or_else(|| meta.get("start_time").and_then(parse_timestamp_value))
        .filter(|&ts| ts > 0)
        .unwrap_or_else(|| file_modified_timestamp_ms(path));

    let model = resolve_model(&meta);
    let inferred = provider_identity::inferred_provider_from_model(&model).unwrap_or("mistral");
    let canonical_prov = provider_identity::canonical_provider(inferred);
    let provider = canonical_prov.as_deref().unwrap_or(inferred);

    let workspace_key = meta
        .get("origin_directory")
        .and_then(Value::as_str)
        .or_else(|| {
            meta.pointer("/environment/working_directory")
                .and_then(Value::as_str)
        })
        .map(str::trim)
        .filter(|dir| !dir.is_empty())
        .and_then(normalize_workspace_key);
    let workspace_label = workspace_key.as_deref().and_then(workspace_label_from_key);
    let title = meta
        .get("title")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|title| !title.is_empty());

    let mut messages = Vec::new();
    for line in BufReader::new(file).lines() {
        let Ok(line) = line else {
            break;
        };
        // Tool calls carry the whole model input; skip them unparsed. A live
        // session may also end on a partially written line, which fails to
        // parse below and is skipped.
        if !line.contains("\"action_result\"") || !line.contains("\"usage\"") {
            continue;
        }
        let Ok(record) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if record.get("type").and_then(Value::as_str) != Some("action_result") {
            continue;
        }
        let Some(payload) = record.get("payload") else {
            continue;
        };
        let Some(result) = payload.get("result") else {
            continue;
        };
        let is_completion = result
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind.starts_with("completion_"));
        if !is_completion {
            continue;
        }
        let Some(usage) = result.pointer("/result/usage").filter(|u| u.is_object()) else {
            continue;
        };
        let tokens = unified_usage_tokens(usage);
        if tokens.total() == 0 {
            continue;
        }

        let call_id = payload
            .get("action_id")
            .or_else(|| result.get("action_id"))
            .and_then(Value::as_str)
            .map(String::from)
            .or_else(|| {
                record
                    .get("sequence")
                    .and_then(Value::as_i64)
                    .map(|seq| format!("seq-{seq}"))
            });
        let Some(call_id) = call_id else {
            continue;
        };

        let mut message = UnifiedMessage::new_with_dedup(
            "vibe",
            model.as_str(),
            provider,
            session_id.clone(),
            timestamp,
            tokens,
            0.0,
            Some(format!("vibe:{session_id}:{call_id}")),
        );
        if workspace_key.is_some() {
            message.set_workspace(workspace_key.clone(), workspace_label.clone());
        }
        if let Some(title) = title {
            message.session_title = Some(title.into());
        }
        messages.push(message);
    }
    messages
}

pub fn parse_vibe_file(path: &Path) -> Vec<UnifiedMessage> {
    if let Some(session_dir) = unified_journal_session_dir(path) {
        return parse_unified_journal(path, session_dir);
    }

    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    if content.trim().is_empty() {
        return Vec::new();
    }
    let Ok(root) = serde_json::from_str::<Value>(&content) else {
        return Vec::new();
    };

    // Vibe 1.x flat files wrap metadata under a "metadata" key;
    // Vibe 2.x meta.json has metadata at the root.
    let meta = if let Some(sub_meta) = root.get("metadata").filter(|v| v.is_object()) {
        sub_meta
    } else {
        &root
    };

    let Some(stats) = meta.get("stats").filter(|v| v.is_object()) else {
        return Vec::new();
    };

    let prompt_tokens = stats
        .get("session_prompt_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let completion_tokens = stats
        .get("session_completion_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let cached_tokens = stats
        .get("session_cached_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);

    let input = prompt_tokens.saturating_sub(cached_tokens).max(0);
    let output = completion_tokens.max(0);
    let cache_read = cached_tokens.max(0);

    let tokens = TokenBreakdown {
        input,
        output,
        cache_read,
        ..Default::default()
    };

    if tokens.total() == 0 {
        return Vec::new();
    }

    let timestamp = meta
        .get("end_time")
        .and_then(parse_timestamp_value)
        .or_else(|| meta.get("start_time").and_then(parse_timestamp_value))
        .filter(|&ts| ts > 0)
        .unwrap_or_else(|| file_modified_timestamp_ms(path));

    let session_id = meta
        .get("session_id")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(String::from)
        .unwrap_or_else(|| {
            if path.file_name().and_then(|f| f.to_str()) == Some("meta.json") {
                path.parent()
                    .and_then(|p| p.file_name())
                    .and_then(|f| f.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            } else {
                path.file_stem()
                    .and_then(|f| f.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            }
        });

    let model = resolve_model(meta);
    let model = model.as_str();
    let inferred = provider_identity::inferred_provider_from_model(model).unwrap_or("mistral");
    let canonical_prov = provider_identity::canonical_provider(inferred);
    let provider = canonical_prov.as_deref().unwrap_or(inferred);

    let dedup_key = format!("vibe:{session_id}");

    let mut message = UnifiedMessage::new_with_dedup(
        "vibe",
        model,
        provider,
        session_id,
        timestamp,
        tokens,
        0.0,
        Some(dedup_key),
    );

    let origin_dir = meta
        .get("origin_directory")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .or_else(|| {
            meta.pointer("/environment/working_directory")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|d| !d.is_empty())
        });

    if let Some(dir) = origin_dir {
        let workspace_key = normalize_workspace_key(dir);
        let workspace_label = workspace_key.as_deref().and_then(workspace_label_from_key);
        if workspace_key.is_some() {
            message.set_workspace(workspace_key, workspace_label);
        }
    }

    if let Some(title) = meta.get("title").and_then(Value::as_str) {
        let trimmed = title.trim();
        if !trimmed.is_empty() {
            message.session_title = Some(trimmed.into());
        }
    }

    if let Some(steps) = stats
        .get("steps")
        .and_then(Value::as_i64)
        .filter(|&s| s > 0)
    {
        message.message_count = steps.min(i32::MAX as i64) as i32;
    } else if let Some(total_msgs) = meta
        .get("total_messages")
        .and_then(Value::as_i64)
        .filter(|&m| m > 0)
    {
        message.message_count = total_msgs.min(i32::MAX as i64) as i32;
    }

    vec![message]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parse_vibe_v2_metadata() {
        let dir = tempdir().unwrap();
        let session_dir = dir.path().join("session_20261010_080000_abc123");
        std::fs::create_dir_all(&session_dir).unwrap();
        let meta_file = session_dir.join("meta.json");

        let json = r#"{
            "session_id": "019e1e27-6f8e-73cb-b0eb-78da684e5111",
            "start_time": "2026-10-10T08:00:00Z",
            "end_time": "2026-10-10T08:05:00Z",
            "title": "Refactor parser",
            "origin_directory": "/home/ubuntu/repo",
            "environment": {
                "working_directory": "/home/ubuntu/repo"
            },
            "config": {
                "active_model": "codestral-latest"
            },
            "stats": {
                "steps": 3,
                "session_prompt_tokens": 15000,
                "session_completion_tokens": 1200,
                "session_cached_tokens": 5000
            }
        }"#;
        std::fs::write(&meta_file, json).unwrap();

        let msgs = parse_vibe_file(&meta_file);
        assert_eq!(msgs.len(), 1);
        let msg = &msgs[0];
        assert_eq!(msg.client, "vibe");
        assert_eq!(msg.session_id, "019e1e27-6f8e-73cb-b0eb-78da684e5111");
        assert_eq!(msg.model_id, "codestral-latest");
        assert_eq!(msg.provider_id, "mistralai");
        assert_eq!(msg.tokens.input, 10000); // 15000 - 5000
        assert_eq!(msg.tokens.output, 1200);
        assert_eq!(msg.tokens.cache_read, 5000);
        assert_eq!(msg.message_count, 3);
        assert_eq!(msg.session_title.as_deref(), Some("Refactor parser"));
        assert_eq!(msg.workspace_key.as_deref(), Some("/home/ubuntu/repo"));
    }

    #[test]
    fn test_parse_vibe_v1_legacy_format() {
        let dir = tempdir().unwrap();
        let session_file = dir.path().join("session_20260901_010203.json");

        let json = r#"{
            "metadata": {
                "session_id": "legacy-session-1",
                "start_time": "2026-09-01T01:02:03Z",
                "active_model": "mistral-large-latest",
                "stats": {
                    "steps": 1,
                    "session_prompt_tokens": 8000,
                    "session_completion_tokens": 500,
                    "session_cached_tokens": 0
                }
            },
            "messages": []
        }"#;
        std::fs::write(&session_file, json).unwrap();

        let msgs = parse_vibe_file(&session_file);
        assert_eq!(msgs.len(), 1);
        let msg = &msgs[0];
        assert_eq!(msg.client, "vibe");
        assert_eq!(msg.session_id, "legacy-session-1");
        assert_eq!(msg.model_id, "mistral-large-latest");
        assert_eq!(msg.tokens.input, 8000);
        assert_eq!(msg.tokens.output, 500);
        assert_eq!(msg.tokens.cache_read, 0);
    }

    #[test]
    fn test_parse_vibe_skips_zero_usage() {
        let dir = tempdir().unwrap();
        let meta_file = dir.path().join("meta.json");

        let json = r#"{
            "session_id": "zero-session",
            "stats": {
                "session_prompt_tokens": 0,
                "session_completion_tokens": 0,
                "session_cached_tokens": 0
            }
        }"#;
        std::fs::write(&meta_file, json).unwrap();

        let msgs = parse_vibe_file(&meta_file);
        assert!(msgs.is_empty());
    }
}
