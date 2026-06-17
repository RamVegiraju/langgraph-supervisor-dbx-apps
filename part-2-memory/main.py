"""CLI entry point. By default runs WITHOUT memory (no Lakebase connection).

Pass --with-memory to wire up the same checkpointer + store the Streamlit app
uses — useful for scripted multi-turn tests.
"""

import argparse
import atexit
import contextlib
import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import build_agent


def _print_tool_trace(messages) -> None:
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                print(f"  ↪ tool call: {call['name']}({call['args']})")
        if msg.__class__.__name__ == "ToolMessage":
            print(f"  ↪ tool result: {msg.content}")


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def _ask(agent, query: str, thread_id: str, user_id: str | None, verbose: bool) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    if user_id:
        config["configurable"]["user_id"] = user_id
    result = agent.invoke({"messages": [HumanMessage(query)]}, config=config)
    if verbose:
        _print_tool_trace(result["messages"])
    print(_extract_text(result["messages"][-1].content))


def _open_memory_resources():
    """Open Lakebase CheckpointSaver + DatabricksStore. Cleanup on exit."""
    from databricks_langchain import CheckpointSaver, DatabricksStore

    project = os.environ.get("LAKEBASE_AUTOSCALING_PROJECT")
    branch = os.environ.get("LAKEBASE_AUTOSCALING_BRANCH")
    if not (project and branch):
        raise SystemExit(
            "--with-memory requires LAKEBASE_AUTOSCALING_PROJECT and "
            "LAKEBASE_AUTOSCALING_BRANCH in your .env."
        )

    stack = contextlib.ExitStack()
    atexit.register(stack.close)
    cp = stack.enter_context(CheckpointSaver(project=project, branch=branch))
    cp.setup()  # idempotent: creates checkpoint tables if absent
    # Sync DatabricksStore doesn't implement context-manager protocol —
    # instantiate directly and call setup() once to create its tables.
    store = DatabricksStore(
        project=project,
        branch=branch,
        embedding_endpoint=os.environ.get(
            "DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
        ),
        embedding_dims=1024,
    )
    store.setup()
    return cp, store


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="LangGraph trip helper agent")
    parser.add_argument("query", nargs="?", help="One-shot question. Omit for REPL.")
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--user-id", default=None, help="Required with --with-memory.")
    parser.add_argument(
        "--with-memory",
        action="store_true",
        help="Wire up Lakebase short-term + long-term memory.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    checkpointer, store = (None, None)
    if args.with_memory:
        if not args.user_id:
            print("--with-memory requires --user-id <email-or-id>.", file=sys.stderr)
            return 1
        checkpointer, store = _open_memory_resources()

    agent = build_agent(checkpointer=checkpointer, store=store)
    thread_id = args.thread_id or str(uuid.uuid4())

    if args.query:
        _ask(agent, args.query, thread_id, args.user_id, args.verbose)
        return 0

    print(f"Thread: {thread_id}  (Ctrl-D or 'exit' to quit)")
    while True:
        try:
            query = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if query.lower() in {"exit", "quit"}:
            return 0
        if not query:
            continue
        print()
        _ask(agent, query, thread_id, args.user_id, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
