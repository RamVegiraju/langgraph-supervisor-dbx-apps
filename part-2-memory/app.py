"""Streamlit UI for the LangGraph supervisor with Lakebase-backed memory.

Memory layers in play:
  - short-term: LangGraph CheckpointSaver, per (thread_id, user_id)
  - long-term explicit: L1 tools the supervisor LLM calls deliberately
  - long-term episodic: auto-persisted at end of every turn
  - long-term semantic: distilled from episodic by an opportunistic background pass

Everything is keyed on a `user_id` taken from a sidebar input.
"""

import atexit
import contextlib
import json
import os
import time
import uuid
from collections import defaultdict
from typing import Optional, Tuple

# Sync trace export for local dev so the span tree is readable instantly.
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

import mlflow
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

mlflow.langchain.autolog()
mlflow.set_experiment(
    os.environ.get("MLFLOW_EXPERIMENT_NAME", "langgraph-supervisor-memory")
)

from agent import build_agent  # noqa: E402
from agent.distiller import distill_user  # noqa: E402
from agent.memory import (  # noqa: E402
    get_semantic_memories,
    get_session_summaries,
    list_known_users,
)


st.set_page_config(page_title="Trip Helper Supervisor", layout="wide")
st.title("🧭 Trip Helper Supervisor (with memory)")
st.caption(
    "Supervisor + weather/finance sub-agents on Databricks GPT-5.5. "
    "Short-term memory in Lakebase checkpoint, long-term in Lakebase store."
)


# ---------------------------------------------------------------------------
# Lakebase resources — opened once at startup, kept alive for the process
# ---------------------------------------------------------------------------

@st.cache_resource
def _lakebase_resources() -> Tuple[Optional[object], Optional[object]]:
    """Open CheckpointSaver + DatabricksStore once. Returns (checkpointer, store)."""
    project = os.environ.get("LAKEBASE_AUTOSCALING_PROJECT")
    branch = os.environ.get("LAKEBASE_AUTOSCALING_BRANCH")
    embedding_endpoint = os.environ.get(
        "DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
    )

    if not (project and branch):
        return None, None

    from databricks_langchain import CheckpointSaver, DatabricksStore

    stack = contextlib.ExitStack()
    atexit.register(stack.close)
    checkpointer = stack.enter_context(
        CheckpointSaver(project=project, branch=branch)
    )
    checkpointer.setup()  # idempotent — creates checkpoint tables if absent
    # Sync DatabricksStore doesn't implement context-manager protocol — instantiate
    # directly and call setup() once to create its tables.
    store = DatabricksStore(
        project=project,
        branch=branch,
        embedding_endpoint=embedding_endpoint,
        embedding_dims=1024,
    )
    store.setup()
    return checkpointer, store


@st.cache_resource
def _agent():
    checkpointer, store = _lakebase_resources()
    return build_agent(checkpointer=checkpointer, store=store)


@st.cache_resource
def _distill_llm():
    from databricks_langchain import ChatDatabricks
    return ChatDatabricks(
        endpoint=os.environ.get("MODEL_ENDPOINT", "databricks-gpt-5-5"),
        use_responses_api=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: user_id and thread state
# ---------------------------------------------------------------------------

DEFAULT_USER = "ramvegdev@gmail-com"
ADD_NEW_SENTINEL = "+ add new user…"


def _switch_user(new_user: str) -> None:
    if new_user == st.session_state.get("user_id"):
        return
    st.session_state.user_id = new_user
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []


with st.sidebar:
    _, store = _lakebase_resources()

    st.markdown("### Session")

    known = list_known_users(store) if store is not None else []
    current = st.session_state.get("user_id", DEFAULT_USER)
    options = sorted(set(known) | {current}) + [ADD_NEW_SENTINEL]

    chosen = st.selectbox(
        "Logged in as", options, index=options.index(current)
    )
    if chosen == ADD_NEW_SENTINEL:
        new_user = st.text_input(
            "New user_id", placeholder="alice@example.com",
            help="Dots get replaced with '-' for storage.",
        )
        if new_user:
            _switch_user(new_user.replace(".", "-"))
    else:
        _switch_user(chosen)

    if st.button(
        "🏁 End & start new conversation",
        help=(
            "Runs the distillation pass on the current thread's episodes "
            "(folds any explicit preferences + recurring patterns into "
            "semantic memory), then resets the thread."
        ),
    ):
        if store is not None and st.session_state.get("history"):
            with st.spinner("Saving preferences…"):
                n = distill_user(
                    store,
                    user_id=st.session_state.get("user_id", DEFAULT_USER),
                    llm=_distill_llm(),
                    min_episodes=1,
                )
            if n:
                st.toast(f"Distilled {n} preference update(s).", icon="🧠")
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
    st.caption(f"thread: `{st.session_state.get('thread_id', '—')[:8]}…`")

    if store is None:
        st.warning(
            "Lakebase not configured. Set "
            "`LAKEBASE_AUTOSCALING_PROJECT` and `LAKEBASE_AUTOSCALING_BRANCH` "
            "in your `.env`."
        )
    else:
        active_user = st.session_state.get("user_id", DEFAULT_USER)

        st.divider()
        st.markdown("### 🧠 User preferences")
        st.caption(
            "Distilled from past conversations. ✦ = user-stated, "
            "otherwise inferred from recurring patterns."
        )
        semantic = get_semantic_memories(store, active_user)
        if semantic:
            for s in semantic:
                value = s.get("value") or json.dumps(
                    {k: v for k, v in s.items() if k not in {"key", "updated_at"}}
                )
                conf = s.get("confidence")
                stated = "✦ " if s.get("stated_by_user") else ""
                suffix = f"  ·  conf {conf:.2f}" if isinstance(conf, (int, float)) else ""
                st.markdown(f"- {stated}**{s['key']}**: {value}{suffix}")
        else:
            st.caption("_(none yet — chat for a few turns then click 'End & start new')_")

        st.markdown("### 💬 Past sessions")
        st.caption("Threads this user has had. Metadata only.")
        sessions = get_session_summaries(store, active_user)
        if sessions:
            for sess in sessions[:10]:
                last_at = (sess["last_at"] or "")[:16].replace("T", " ")
                turns = sess["turn_count"]
                turn_word = "turn" if turns == 1 else "turns"
                tid_short = sess["thread_id"][:12]
                marker = " ← current" if sess["thread_id"] == st.session_state.get("thread_id") else ""
                st.markdown(f"- `{tid_short}…` · {turns} {turn_word} · {last_at}{marker}")
            if len(sessions) > 10:
                st.caption(f"+ {len(sessions) - 10} older")
        else:
            st.caption("_(none yet)_")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content) if content is not None else ""


def _extract_streaming_delta(content) -> str:
    if not isinstance(content, list):
        return ""
    return "".join(
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text" and "annotations" not in b
    )


_INTERESTING_NAMES = {"LangGraph", "supervisor", "subagents", "persist_episode"}
_INTERESTING_SPAN_TYPES = {"TOOL", "CHAT_MODEL", "LLM", "AGENT"}


def _is_interesting(span) -> bool:
    return span.name in _INTERESTING_NAMES or span.span_type in _INTERESTING_SPAN_TYPES


def _render_span_tree(trace) -> None:
    spans = trace.data.spans
    children: dict[str, list] = defaultdict(list)
    roots = []
    for s in spans:
        if s.parent_id:
            children[s.parent_id].append(s)
        else:
            roots.append(s)

    icons = {"AGENT": "🤖", "TOOL": "🔧", "CHAT_MODEL": "🧠", "LLM": "🧠"}

    def render(span, depth: int):
        if not _is_interesting(span):
            for ch in children[span.span_id]:
                render(ch, depth)
            return
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1_000_000
        icon = icons.get(span.span_type, "🔗")
        indent = "&nbsp;" * (depth * 4)
        st.markdown(
            f"{indent}{icon} **{span.name}** "
            f"<span style='color:#888'>· {span.span_type} · {duration_ms:.0f} ms</span>",
            unsafe_allow_html=True,
        )
        for ch in children[span.span_id]:
            render(ch, depth + 1)

    for root in roots:
        render(root, 0)


# ---------------------------------------------------------------------------
# Streaming turn
# ---------------------------------------------------------------------------

def _stream_assistant_turn(query: str) -> None:
    status = st.status("🧭 Supervisor routing your query…", expanded=False)
    answer_placeholder = st.empty()
    accumulated = ""
    subagents_invoked = False
    subagents_called: set[str] = set()

    start = time.perf_counter()

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "user_id": st.session_state.user_id,
        }
    }

    for mode, payload in _agent().stream(
        {"messages": [HumanMessage(query)]},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if mode == "updates":
            for node_name, state_update in payload.items():
                if node_name == "subagents":
                    subagents_invoked = True
                    status.update(label="📦 Running tools…")
                elif node_name == "supervisor":
                    new_messages = (
                        state_update.get("messages", [])
                        if isinstance(state_update, dict)
                        else []
                    )
                    for msg in new_messages:
                        for tc in getattr(msg, "tool_calls", None) or []:
                            name = tc.get("name", "")
                            if name.endswith("_subagent"):
                                subagents_called.add(
                                    name[len("call_") : -len("_subagent")]
                                )
                    if subagents_invoked:
                        status.update(label="✍️ Synthesizing final answer…")
                elif node_name == "persist_episode":
                    status.update(label="💾 Saving episode…")
        elif mode == "messages":
            chunk_msg, metadata = payload
            if metadata.get("langgraph_node") != "supervisor":
                continue
            text = _extract_streaming_delta(chunk_msg.content)
            if text:
                accumulated += text
                answer_placeholder.markdown(accumulated)

    total_ms = int((time.perf_counter() - start) * 1000)
    status.update(state="complete", expanded=False, label="✓ Done")

    routing_line = (
        f"Routed to: **{', '.join(sorted(subagents_called))}**"
        if subagents_called
        else "Answered directly — no sub-agents invoked."
    )
    st.markdown(f"_{routing_line}  ·  ⏱ {total_ms} ms_")

    trace_id = mlflow.get_last_active_trace_id()
    trace = mlflow.get_trace(trace_id) if trace_id else None
    if trace:
        with st.expander("🔀 full span trace"):
            _render_span_tree(trace)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": accumulated,
            "routing": routing_line,
            "trace_id": trace_id,
            "total_ms": total_ms,
        }
    )


def _render_history_turn(turn) -> None:
    with st.chat_message(turn["role"]):
        if turn["role"] != "assistant":
            st.markdown(turn["content"])
            return

        st.markdown(turn["content"])
        header_parts = []
        if turn.get("routing"):
            header_parts.append(turn["routing"])
        if turn.get("total_ms") is not None:
            header_parts.append(f"⏱ {turn['total_ms']} ms")
        if header_parts:
            st.markdown(f"_{'  ·  '.join(header_parts)}_")
        if turn.get("trace_id"):
            with st.expander("🔀 full span trace"):
                trace = mlflow.get_trace(turn["trace_id"])
                if trace:
                    _render_span_tree(trace)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

for past_turn in st.session_state.history:
    _render_history_turn(past_turn)

if query := st.chat_input("Ask about weather, currency conversion, forecasts…"):
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        _stream_assistant_turn(query)
