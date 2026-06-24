import os
from typing import Optional

from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

from agent.memory import (
    load_episodic_context,
    load_semantic_context,
    persist_episode,
)
from agent.subagents import build_finance_agent, build_weather_agent


class State(MessagesState):
    """State for the supervisor graph.

    Two memory caches ride on top of MessagesState:

    - `semantic_context`: the user's KNOWN_USER_PREFERENCES block. Query-
      independent, so it's loaded ONCE on the first turn of a thread and then
      reused (the checkpointer rehydrates it on later turns). `None` until
      loaded; `""` means "loaded, user has no prefs" — both skip a re-query.

    - `memory_context`: the full working-memory string (semantic + the per-turn
      episodic block) for the current turn. Recomputed each routing pass and
      reused on the synthesis pass after sub-agents return.
    """
    semantic_context: Optional[str]
    memory_context: str

SUPERVISOR_PROMPT = (
    "You are a supervisor coordinating two specialist sub-agents: "
    "call_weather_subagent (for weather questions) and call_finance_subagent "
    "(for currency conversion or exchange rates). "
    "Delegate the user's question to the right sub-agent — call both in parallel "
    "in a single turn if the question needs both. "
    "After the sub-agents respond, synthesize their answers into one short reply.\n\n"
    "For greetings, memory-style questions (\"what do you remember\", "
    "\"what's my preferred X\"), or anything else not requiring real-time data: "
    "answer directly from the user context provided in your system prompt "
    "(KNOWN_USER_PREFERENCES, RELATED_PAST_INTERACTIONS). Do NOT call a tool "
    "to look up things already in your context."
)


def _extract_text(content) -> str:
    """Responses API messages carry a list of typed blocks; flatten to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def build_agent(
    endpoint: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
):
    """Build the supervisor graph with optional memory backing.

    Shape:

        START -> supervisor --(tools_condition)--+-> subagents -> supervisor
                                                 +-> persist_episode -> END

    If `checkpointer` is provided, multi-turn state per thread_id is persisted.
    If `store` is provided, long-term memory tools become functional and the
    persist_episode node writes an episode at the end of every run.
    """
    llm = ChatDatabricks(
        endpoint=endpoint or os.environ.get("MODEL_ENDPOINT", "databricks-gpt-5-5"),
        use_responses_api=True,
    )

    weather_agent = build_weather_agent()
    finance_agent = build_finance_agent()

    @tool
    def call_weather_subagent(query: str) -> str:
        """Delegate a weather question to the weather specialist.

        Args:
            query: The user's weather question, e.g. "What's the weather in Paris?".
        """
        result = weather_agent.invoke({"messages": [HumanMessage(query)]})
        return _extract_text(result["messages"][-1].content)

    @tool
    def call_finance_subagent(query: str) -> str:
        """Delegate a currency / FX question to the finance specialist.

        Args:
            query: The user's finance question, e.g. "Convert 100 USD to EUR".
        """
        result = finance_agent.invoke({"messages": [HumanMessage(query)]})
        return _extract_text(result["messages"][-1].content)

    subagent_tools = [call_weather_subagent, call_finance_subagent]
    supervisor_llm = llm.bind_tools(subagent_tools)

    def supervisor_node(state: State, config: RunnableConfig) -> dict:
        # Routing pass = last message is a HumanMessage (new turn just started).
        # Synthesis pass = last message is a ToolMessage (sub-agents just returned).
        # We only recompute memory_context on the routing pass; the synthesis
        # pass reuses what's already in state.
        msgs = state["messages"]
        is_routing_pass = bool(msgs) and isinstance(msgs[-1], HumanMessage)

        user_id = config.get("configurable", {}).get("user_id")
        update: dict = {}

        if is_routing_pass and user_id and store is not None:
            query = str(msgs[-1].content)

            # Episodic is query-dependent -> reload every turn.
            episodic = load_episodic_context(store, user_id=user_id, query=query)

            # Semantic is query-independent -> load once per thread, then reuse
            # the checkpointed value. None = not yet loaded; "" = loaded, empty.
            semantic = state.get("semantic_context")
            if semantic is None:
                semantic = load_semantic_context(store, user_id=user_id)
                update["semantic_context"] = semantic

            context = "\n".join(p for p in (semantic, episodic) if p)
            update["memory_context"] = context
        else:
            context = state.get("memory_context", "")

        prompt_messages: list = [SystemMessage(SUPERVISOR_PROMPT)]
        if context:
            prompt_messages.append(
                SystemMessage(f"Context about this user:\n{context}")
            )
        prompt_messages.extend(msgs)

        response = supervisor_llm.invoke(prompt_messages)
        update["messages"] = [response]
        return update

    def persist_episode_node(state: State, config: RunnableConfig) -> dict:
        if store is None:
            return {}

        cfg = config.get("configurable", {})
        user_id = cfg.get("user_id")
        thread_id = cfg.get("thread_id", "unknown")
        if not user_id:
            return {}

        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if not last_human or not last_ai:
            return {}

        subagents_used = []
        for m in state["messages"]:
            for tc in getattr(m, "tool_calls", None) or []:
                name = tc.get("name", "")
                if name.startswith("call_") and name.endswith("_subagent"):
                    subagents_used.append(name[len("call_") : -len("_subagent")])

        persist_episode(
            store,
            user_id=user_id,
            thread_id=thread_id,
            query=str(last_human.content),
            answer=_extract_text(last_ai.content),
            subagents_used=subagents_used,
        )
        return {}

    builder = (
        StateGraph(State)
        .add_node(
            "supervisor",
            supervisor_node,
            retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
        )
        .add_node("subagents", ToolNode(subagent_tools, handle_tool_errors=True))
        .add_node("persist_episode", persist_episode_node)
        .add_edge(START, "supervisor")
        .add_conditional_edges(
            "supervisor",
            tools_condition,
            {"tools": "subagents", END: "persist_episode"},
        )
        .add_edge("subagents", "supervisor")
        .add_edge("persist_episode", END)
    )

    return builder.compile(checkpointer=checkpointer, store=store)
