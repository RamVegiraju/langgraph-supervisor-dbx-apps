"""
Provision a Lakebase Autoscale project for agent memory.

Creates a project with two branches: the auto-created `production` (untouched
during dev) and a long-lived `development` branch we work against. Verifies
that credentials can be generated.

Requirements:
    pip install "databricks-sdk>=0.81.0" python-dotenv

Usage:
    python provision_lakebase.py
    python provision_lakebase.py --name my-memory --dev-branch dev
"""

import argparse
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    Branch,
    BranchSpec,
    Project,
    ProjectSpec,
)
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(
        description="Provision a Lakebase Autoscale project for agent memory"
    )
    p.add_argument(
        "--name",
        default="langgraph-supervisor-memory",
        help="Project name (lowercase letters, digits, hyphens; max 63 chars)",
    )
    p.add_argument(
        "--dev-branch",
        default="development",
        help="Name of the development branch forked from production",
    )
    p.add_argument(
        "--pg-version",
        default="17",
        help="Postgres version: 16 or 17",
    )
    p.add_argument(
        "--min-cu",
        type=float,
        default=0.5,
        help="Autoscaling minimum compute units (0.5-32)",
    )
    p.add_argument(
        "--max-cu",
        type=float,
        default=2.0,
        help="Autoscaling maximum compute units (max - min <= 8)",
    )
    p.add_argument(
        "--scale-to-zero-seconds",
        type=int,
        default=300,
        help="Inactivity timeout before suspending the dev branch",
    )
    return p.parse_args()


def main():
    args = parse_args()

    host = os.environ.get("DATABRICKS_HOST") or ""
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if not (host or profile):
        print(
            "Error: set DATABRICKS_HOST + DATABRICKS_TOKEN in .env, "
            "or DATABRICKS_CONFIG_PROFILE pointing to a valid profile."
        )
        raise SystemExit(1)

    w = WorkspaceClient()

    print(f"\n[1/3] Creating project '{args.name}' (pg {args.pg_version}) …")
    print("      This is a long-running operation; may take a couple of minutes.")
    try:
        op = w.postgres.create_project(
            project=Project(
                spec=ProjectSpec(
                    display_name=args.name,
                    pg_version=args.pg_version,
                )
            ),
            project_id=args.name,
        )
        project_result = op.wait()
        print(f"      Created: {project_result.name}")
    except Exception as exc:
        if "already exists" in str(exc).lower() or "ALREADY_EXISTS" in str(exc):
            print(f"      Already exists: projects/{args.name} — skipping create.")
        else:
            raise

    print(
        f"\n[2/3] Creating dev branch '{args.dev_branch}' "
        f"(autoscale {args.min_cu}-{args.max_cu} CU, "
        f"scale-to-zero {args.scale_to_zero_seconds}s) …"
    )
    branch_op = w.postgres.create_branch(
        parent=f"projects/{args.name}",
        branch=Branch(
            spec=BranchSpec(
                source_branch=f"projects/{args.name}/branches/production",
                no_expiry=True,
            )
        ),
        branch_id=args.dev_branch,
    )
    branch_result = branch_op.wait()
    print(f"      Created: {branch_result.name}")

    print("\n[3/3] Verifying credentials …")
    endpoints = list(
        w.postgres.list_endpoints(
            parent=f"projects/{args.name}/branches/{args.dev_branch}"
        )
    )
    if not endpoints:
        print("      Warning: no endpoints found on dev branch; skipping cred check.")
        cred = None
    else:
        endpoint_path = endpoints[0].name
        print(f"      Using endpoint: {endpoint_path}")
        cred = w.postgres.generate_database_credential(endpoint=endpoint_path)
    if not cred.token:
        print("      Warning: no token returned; verify endpoint name.")
    else:
        me = w.current_user.me().user_name
        print(f"      Credentials OK for user '{me}'")

    print("\n" + "=" * 60)
    print("Add this to part-2-memory/.env:")
    print("=" * 60)
    print(f"LAKEBASE_AUTOSCALING_PROJECT={args.name}")
    print(f"LAKEBASE_AUTOSCALING_BRANCH={args.dev_branch}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Update .env with the two lines above.")
    print("  2. uv run streamlit run app.py")
    print("  3. The first connection will trigger table creation in the dev branch.")
    print("\nTo create an experiment branch later (cheap, copy-on-write):")
    print(
        f"  databricks postgres create-branch projects/{args.name} experiment-foo \\"
    )
    print(
        f'    --json \'{{"spec": {{"source_branch": '
        f'"projects/{args.name}/branches/{args.dev_branch}", "ttl": "604800s"}}}}\''
    )


if __name__ == "__main__":
    main()
