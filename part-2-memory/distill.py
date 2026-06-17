"""On-demand distillation CLI.

Runs a single distillation pass for one user (or all users) — turns recent
episodic memory into stable semantic preferences via one LLM call.

Usage:
    python distill.py                                # distill all known users
    python distill.py --user ramvegdev@gmail.com     # one user
    python distill.py --user X --min-episodes 1      # force a pass with few episodes
"""

import argparse
import contextlib
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Run L2 semantic-memory distillation.")
    p.add_argument("--user", help="user_id to distill (default: all users)")
    p.add_argument(
        "--min-episodes",
        type=int,
        default=int(os.environ.get("DISTILL_MIN_NEW_EPISODES", "3")),
        help="Skip users with fewer undistilled episodes than this.",
    )
    return p.parse_args()


def _discover_users(store) -> list[str]:
    """Walk the store namespaces to find user_ids that have any episodic memory."""
    namespaces = store.list_namespaces(prefix=("memories", "episodic"))
    return sorted({ns[-1] for ns in namespaces}) if namespaces else []


def main() -> int:
    args = parse_args()

    project = os.environ.get("LAKEBASE_AUTOSCALING_PROJECT")
    branch = os.environ.get("LAKEBASE_AUTOSCALING_BRANCH")
    if not (project and branch):
        print("Error: LAKEBASE_AUTOSCALING_PROJECT and LAKEBASE_AUTOSCALING_BRANCH required.")
        return 1

    from databricks_langchain import ChatDatabricks, DatabricksStore
    from agent.distiller import distill_user

    llm = ChatDatabricks(
        endpoint=os.environ.get("MODEL_ENDPOINT", "databricks-gpt-5-5"),
        use_responses_api=True,
    )

    store = DatabricksStore(
        project=project,
        branch=branch,
        embedding_endpoint=os.environ.get(
            "DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
        ),
        embedding_dims=1024,
    )
    store.setup()

    if args.user:
        users = [args.user.replace(".", "-")]
    else:
        users = _discover_users(store)
        if not users:
            print("No users with episodic memories found.")
            return 0

    for user_id in users:
        print(f"distilling user_id={user_id} …", flush=True)
        n = distill_user(
            store, user_id=user_id, llm=llm, min_episodes=args.min_episodes
        )
        print(f"  applied {n} update(s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
