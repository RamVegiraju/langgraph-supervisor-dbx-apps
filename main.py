import argparse
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import build_agent


def _print_tool_trace(messages) -> None:
    """Show tool calls and tool outputs so you can see what the agent did."""
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                print(f"  ↪ tool call: {call['name']}({call['args']})")
        if msg.__class__.__name__ == "ToolMessage":
            print(f"  ↪ tool result: {msg.content}")


def _extract_text(content) -> str:
    """Responses API returns a list of typed blocks (reasoning, text, ...).
    Chat completions returns a string. Normalize to plain text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _ask(agent, query: str, thread_id: str, verbose: bool) -> None:
    result = agent.invoke(
        {"messages": [HumanMessage(query)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    if verbose:
        _print_tool_trace(result["messages"])
    print(_extract_text(result["messages"][-1].content))


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="LangGraph trip helper agent")
    parser.add_argument("query", nargs="?", help="One-shot question. Omit for REPL.")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Conversation thread id (defaults to a fresh UUID).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print tool calls and results."
    )
    args = parser.parse_args()

    agent = build_agent()
    thread_id = args.thread_id or str(uuid.uuid4())

    if args.query:
        _ask(agent, args.query, thread_id, args.verbose)
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
        _ask(agent, query, thread_id, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
