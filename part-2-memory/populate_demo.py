"""One-off driver to populate the Lakebase memory tables with realistic traffic.

Opens the checkpointer + store once, then replays several multi-turn
conversations across a few users. Mixes weather-only, finance-only, both, and
chitchat turns, and plants explicit user preferences so the semantic/distill
layer has signal. Runs a distillation pass per user at the end.
"""

import os
import sys
import contextlib

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import build_agent

load_dotenv()


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


# (user_id, thread_id, [turns]) — multiple threads per user => multiple sessions
CONVERSATIONS = [
    (
        "alice@example.com",
        "alice-paris-trip",
        [
            "I'm planning a trip to Paris. What's the weather there right now?",
            "How much is $500 in euros for the trip?",
            "Remember that I always prefer temperatures in Fahrenheit.",
            "What's the 4-day forecast for Paris?",
        ],
    ),
    (
        "alice@example.com",
        "alice-tokyo-trip",
        [
            "Now I'm thinking about Tokyo instead. Weather there and convert $1000 to yen?",
            "What's the air quality like in Tokyo today?",
        ],
    ),
    (
        "bob@example.com",
        "bob-fx-desk",
        [
            "Convert 250 GBP to USD.",
            "Compare the US dollar against EUR, GBP, and JPY.",
            "Please always show me amounts in US dollars.",
            "What was the USD to EUR rate a year ago versus today?",
        ],
    ),
    (
        "carol@example.com",
        "carol-weather",
        [
            "What's the weather in London?",
            "I prefer Celsius, by the way.",
            "Is the air quality safe to run outside in Delhi right now?",
            "Thanks, that's all I needed!",
        ],
    ),
]


def open_memory():
    from databricks_langchain import CheckpointSaver, DatabricksStore

    project = os.environ["LAKEBASE_AUTOSCALING_PROJECT"]
    branch = os.environ["LAKEBASE_AUTOSCALING_BRANCH"]
    stack = contextlib.ExitStack()
    cp = stack.enter_context(CheckpointSaver(project=project, branch=branch))
    cp.setup()
    store = DatabricksStore(
        project=project,
        branch=branch,
        embedding_endpoint=os.environ.get(
            "DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
        ),
        embedding_dims=1024,
    )
    store.setup()
    return cp, store, stack


def main() -> int:
    cp, store, stack = open_memory()
    agent = build_agent(checkpointer=cp, store=store)

    users = []
    with stack:
        for user_id, thread_id, turns in CONVERSATIONS:
            if user_id not in users:
                users.append(user_id)
            print(f"\n{'='*70}\nUSER {user_id}  /  THREAD {thread_id}\n{'='*70}")
            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            for turn in turns:
                print(f"\nyou> {turn}")
                result = agent.invoke({"messages": [HumanMessage(turn)]}, config=config)
                print(f"bot> {_text(result['messages'][-1].content)[:280]}")

        # Distill semantic preferences from the episodes we just wrote.
        print(f"\n{'='*70}\nDISTILLATION\n{'='*70}")
        from agent.distiller import distill_user
        from databricks_langchain import ChatDatabricks

        llm = ChatDatabricks(
            endpoint=os.environ.get("MODEL_ENDPOINT", "databricks-gpt-5-5"),
            use_responses_api=True,
        )
        for user_id in users:
            print(f"\n-- distilling {user_id} --")
            try:
                n = distill_user(store, user_id=user_id, llm=llm, min_episodes=1)
                print(f"   applied {n} update(s)")
            except Exception as exc:
                print(f"   distill error: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
