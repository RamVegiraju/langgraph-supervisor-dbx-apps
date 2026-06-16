"""Streamlit UI for the LangGraph supervisor.

MLflow tracing is enabled before the agent is built — every node, tool, and
LLM call is captured automatically. We stream from the graph so the user sees
live status updates ("routing", "running sub-agents", "synthesizing") and
token-by-token output for the final answer.
"""

import os
import time
from collections import defaultdict

# Local dev: write traces synchronously so the span tree is readable the moment
# the stream ends. In production (Databricks Apps + remote MLflow), flip this
# back to true so response latency doesn't depend on the tracing backend.
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

import mlflow
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

mlflow.langchain.autolog()
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "langgraph-supervisor-local"))

from agent import build_agent  # noqa: E402


st.set_page_config(page_title="Trip Helper Supervisor", layout="wide")
st.title("🧭 Trip Helper Supervisor")
st.caption(
    "LangGraph supervisor delegating to weather and finance specialists, "
    "running on Databricks GPT-5.5 via the AI Gateway."
)


@st.cache_resource
def _agent():
    return build_agent()


if "history" not in st.session_state:
    st.session_state.history = []


def _extract_text(content) -> str:
    """Responses API content is a list of typed blocks; flatten to text."""
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
    """Pull text from a streaming chunk, skipping the final consolidated block.

    The Databricks Responses API emits incremental delta chunks during a stream
    AND a final block containing the full text. The final block is identifiable
    by the presence of an ``annotations`` key on the text block.
    """
    if not isinstance(content, list):
        return ""
    parts = []
    for b in content:
        if not isinstance(b, dict) or b.get("type") != "text":
            continue
        if "annotations" in b:
            continue
        parts.append(b.get("text", ""))
    return "".join(parts)


# --- trace rendering --------------------------------------------------------

_INTERESTING_NAMES = {"LangGraph", "supervisor", "subagents"}
_INTERESTING_SPAN_TYPES = {"TOOL", "CHAT_MODEL", "LLM", "AGENT"}


def _is_interesting(span) -> bool:
    return span.name in _INTERESTING_NAMES or span.span_type in _INTERESTING_SPAN_TYPES


def _summarize_routing(spans) -> str:
    subagents = set()
    for s in spans:
        if (
            s.span_type == "TOOL"
            and s.name.startswith("call_")
            and s.name.endswith("_subagent")
        ):
            subagents.add(s.name[len("call_") : -len("_subagent")])
    if not subagents:
        return "Answered directly — no sub-agents invoked."
    return f"Routed to: **{', '.join(sorted(subagents))}**"


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


# --- streaming turn ---------------------------------------------------------


def _stream_assistant_turn(query: str) -> None:
    status = st.status("🧭 Supervisor routing your query…", expanded=False)
    answer_placeholder = st.empty()
    accumulated = ""
    subagents_invoked = False
    subagents_called: set[str] = set()

    start = time.perf_counter()

    for mode, payload in _agent().stream(
        {"messages": [HumanMessage(query)]},
        stream_mode=["updates", "messages"],
    ):
        if mode == "updates":
            for node_name, state_update in payload.items():
                if node_name == "subagents":
                    subagents_invoked = True
                    status.update(label="📦 Running sub-agents in parallel…")
                elif node_name == "supervisor":
                    # Capture which sub-agents the supervisor decided to call —
                    # we get this from the stream and don't need the MLflow trace.
                    new_messages = (
                        state_update.get("messages", [])
                        if isinstance(state_update, dict)
                        else []
                    )
                    for msg in new_messages:
                        for tc in getattr(msg, "tool_calls", None) or []:
                            name = tc.get("name", "")
                            if name.startswith("call_") and name.endswith("_subagent"):
                                subagents_called.add(
                                    name[len("call_") : -len("_subagent")]
                                )
                    if subagents_invoked:
                        status.update(label="✍️ Synthesizing final answer…")
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


# --- render --------------------------------------------------------------

for past_turn in st.session_state.history:
    _render_history_turn(past_turn)

if query := st.chat_input("Ask about weather, currency conversion, forecasts…"):
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        _stream_assistant_turn(query)
