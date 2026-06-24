"""Memory utilities — namespaces, episode persistence, context loading.

Two namespaces per user_id (down from three — no L1 explicit channel):

  ("memories", "episodic", user_id)   — every turn auto-persisted
  ("memories", "semantic", user_id)   — distilled stable preferences

There are NO memory tools the supervisor LLM can call. Memory is purely
automatic:
  - WRITE: persist_episode (graph node) at end of every turn
  - READ:  load_semantic_context (once per thread) + load_episodic_context
           (every turn), assembled into the supervisor's "working memory"
  - CONSOLIDATE: distill_user (in distiller.py), triggered at session end

This keeps the per-turn LLM call cheap — no extra tool the supervisor has to
consider, no on-demand vector search.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from langgraph.store.base import BaseStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(user_id: str) -> str:
    # Namespace segments can't contain '.', so replace.
    return user_id.replace(".", "-")


def episodic_ns(user_id: str) -> tuple:
    return ("memories", "episodic", _safe(user_id))


def semantic_ns(user_id: str) -> tuple:
    return ("memories", "semantic", _safe(user_id))


def meta_ns(user_id: str) -> tuple:
    return ("meta", _safe(user_id))


# ---------------------------------------------------------------------------
# Programmatic episode persistence — runs at end of every turn
# ---------------------------------------------------------------------------

def persist_episode(
    store: BaseStore,
    *,
    user_id: str,
    thread_id: str,
    query: str,
    answer: str,
    subagents_used: list[str],
) -> None:
    episode_id = f"{thread_id}-{uuid.uuid4().hex[:8]}"
    store.put(
        episodic_ns(user_id),
        episode_id,
        {
            "query": query,
            "answer": answer,
            "subagents_used": sorted(set(subagents_used)),
            "thread_id": thread_id,
            "occurred_at": _now_iso(),
            "distilled_at": None,
        },
    )


# ---------------------------------------------------------------------------
# Inspection helpers — used by the Streamlit sidebar
# ---------------------------------------------------------------------------

def list_known_users(store: BaseStore) -> list[str]:
    """All user_ids that have at least one episodic memory."""
    namespaces = store.list_namespaces(prefix=("memories", "episodic")) or []
    return sorted({ns[-1] for ns in namespaces})


def get_semantic_memories(store: BaseStore, user_id: str) -> list[dict]:
    """All distilled semantic preferences for a user, flattened for display."""
    items = store.search(semantic_ns(user_id), limit=100) or []
    return [{"key": it.key, **it.value} for it in items]


def get_session_summaries(store: BaseStore, user_id: str) -> list[dict]:
    """Group this user's episodic memories by thread_id."""
    items = store.search(episodic_ns(user_id), limit=500) or []
    sessions: dict[str, dict] = {}
    for it in items:
        v = it.value
        thread_id = v.get("thread_id", "unknown")
        ts = v.get("occurred_at", "")
        if thread_id not in sessions:
            sessions[thread_id] = {
                "thread_id": thread_id,
                "turn_count": 0,
                "first_at": ts,
                "last_at": ts,
            }
        s = sessions[thread_id]
        s["turn_count"] += 1
        if ts:
            if not s["first_at"] or ts < s["first_at"]:
                s["first_at"] = ts
            if not s["last_at"] or ts > s["last_at"]:
                s["last_at"] = ts
    return sorted(sessions.values(), key=lambda x: x["last_at"], reverse=True)


# ---------------------------------------------------------------------------
# Context loading — assembles the supervisor's "working memory" (MemAlign term)
#
# Split into two functions because the two reads have different lifetimes:
#   - semantic is query-INdependent  -> load once per thread, then cache
#   - episodic is query-DEPENDENT    -> reload every turn (similarity to query)
# ---------------------------------------------------------------------------

def load_semantic_context(store: BaseStore, *, user_id: str) -> str:
    """KNOWN_USER_PREFERENCES block — ALL semantic prefs, no vector search.

    Query-independent (same result regardless of the current question), so the
    caller loads this once at the start of a thread and reuses the cached value.
    """
    semantic = store.search(semantic_ns(user_id), limit=20) or []
    if not semantic:
        return ""

    parts = ["KNOWN_USER_PREFERENCES:"]
    for item in semantic:
        val = item.value
        display = val.get(
            "value",
            json.dumps({k: v for k, v in val.items() if k != "updated_at"}),
        )
        conf = val.get("confidence")
        stated = " · user-stated" if val.get("stated_by_user") else ""
        if isinstance(conf, (int, float)):
            suffix = f"  (conf {conf:.2f}{stated})"
        else:
            suffix = f"  ({stated.strip(' ·')})" if stated else ""
        parts.append(f"  - {item.key}: {display}{suffix}")
    return "\n".join(parts)


def load_episodic_context(store: BaseStore, *, user_id: str, query: str) -> str:
    """RELATED_PAST_INTERACTIONS block — top-3 episodes by pgvector similarity.

    Query-dependent, so the caller reloads this every turn.
    """
    episodic = store.search(episodic_ns(user_id), query=query, limit=3) or []
    if not episodic:
        return ""

    parts = ["RELATED_PAST_INTERACTIONS:"]
    for item in episodic:
        v = item.value
        when = (v.get("occurred_at") or "?")[:10]
        used = ", ".join(v.get("subagents_used") or []) or "no sub-agents"
        parts.append(f"  - ({when}) \"{v.get('query', '')}\"  → used: {used}")
    return "\n".join(parts)
