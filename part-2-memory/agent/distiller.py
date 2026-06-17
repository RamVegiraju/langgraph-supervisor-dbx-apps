"""L2 — distill recent episodic memories into stable semantic preferences.

A separate LLM pass: read existing semantic + new episodic for a user, ask an
LLM to propose minimal updates, apply them, mark episodes as distilled.

Called opportunistically from the Streamlit app (at session start if stale),
and would be wired to a Databricks Job in production.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Literal

from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field, ValidationError

from agent.memory import episodic_ns, meta_ns, semantic_ns, _now_iso

logger = logging.getLogger(__name__)


DISTILLATION_SYSTEM = (
    "You are a memory distiller for a personal trip-planning assistant. "
    "Given a user's existing semantic preferences and their recent episodic "
    "interactions, propose minimal updates capturing what the assistant should "
    "remember about this user.\n\n"
    "TWO kinds of signal to capture:\n"
    "  1. EXPLICIT statements where the user directly said 'remember X', "
    "'I prefer Y', 'my X is Y', 'don't forget Z', etc. "
    "Mark these with HIGH confidence (0.9+) and stated_by_user: true.\n"
    "  2. INFERRED patterns where the user has REPEATEDLY exhibited a behavior "
    "across multiple interactions (e.g. always asking about Tokyo, always "
    "converting to JPY, repeatedly requesting Celsius). "
    "Lower confidence (0.5-0.8) and stated_by_user: false.\n\n"
    "Skip true one-off behavior (a single mention isn't a pattern). "
    "Prefer updating existing entries (raising confidence) over adding new ones "
    "for the same idea. Use delete only when a preference has been clearly "
    "contradicted by recent behavior.\n\n"
    "Output ONLY a JSON object with this exact shape, no prose, no markdown fences:\n"
    '{\n'
    '  "updates": [\n'
    '    {"op": "add"|"update"|"delete", "key": "snake_case_id", '
    '"value": {"value": "...", "stated_by_user": true|false}, '
    '"confidence": 0.0-1.0}\n'
    '  ]\n'
    '}'
)


class MemoryUpdate(BaseModel):
    op: Literal["add", "update", "delete"]
    key: str
    value: dict = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DistillationOutput(BaseModel):
    updates: list[MemoryUpdate] = Field(default_factory=list)


def _extract_json_text(content) -> str:
    """Pull plain text out of Responses-API content (list of blocks) or string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def _parse_distillation(text: str) -> DistillationOutput:
    """Strip code fences, parse JSON, validate."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: find the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    return DistillationOutput.model_validate(data)


def _format_items(items, *, label: str) -> str:
    if not items:
        return f"({label}: none)"
    lines = []
    for it in items:
        lines.append(f"- {it.key}: {json.dumps(it.value)}")
    return "\n".join(lines)


def distill_user(
    store: BaseStore,
    *,
    user_id: str,
    llm: ChatDatabricks,
    min_episodes: int = 3,
) -> int:
    """Run one distillation pass for a user. Returns the number of updates applied."""
    from agent.memory import _safe  # local to avoid cycles

    existing = list(store.search(semantic_ns(user_id), limit=50) or [])
    all_episodes = list(store.search(episodic_ns(user_id), limit=50) or [])
    new_episodes = [e for e in all_episodes if not e.value.get("distilled_at")]

    if len(new_episodes) < min_episodes:
        return 0

    user_msg = (
        f"EXISTING_PREFERENCES:\n{_format_items(existing, label='existing')}\n\n"
        f"RECENT_EPISODES (since last distillation):\n"
        f"{_format_items(new_episodes, label='episodes')}"
    )

    raw = llm.invoke([SystemMessage(DISTILLATION_SYSTEM), HumanMessage(user_msg)])
    try:
        output = _parse_distillation(_extract_json_text(raw.content))
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Could not parse distillation output: %s", exc)
        return 0

    now = _now_iso()
    n_applied = 0
    for upd in output.updates:
        if upd.op in ("add", "update"):
            payload = {
                **upd.value,
                "confidence": upd.confidence,
                "updated_at": now,
            }
            store.put(semantic_ns(user_id), upd.key, payload)
            n_applied += 1
        elif upd.op == "delete":
            try:
                store.delete(semantic_ns(user_id), upd.key)
                n_applied += 1
            except Exception:
                pass

    for e in new_episodes:
        store.put(episodic_ns(user_id), e.key, {**e.value, "distilled_at": now})

    store.put(meta_ns(user_id), "last_distill", {
        "at": now,
        "episodes_processed": len(new_episodes),
        "updates_applied": n_applied,
    })
    return n_applied


