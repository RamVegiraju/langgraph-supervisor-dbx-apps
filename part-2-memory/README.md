# Part 2 — Memory (Lakebase-backed)

Adds three memory layers on top of the Part 1 supervisor, all backed by one
Lakebase Autoscale project. Aligned with the
[Databricks Memory Scaling blog](https://www.databricks.com/blog/memory-scaling-ai-agents)
terminology: episodic memories are raw records of past interactions; semantic
memories are generalized facts distilled from them.

> Part 2 is intentionally a prototype. Known limitations and the production
> evolution path are at the bottom of this README.

## The three memory layers

| | **Checkpointer** (short-term) | **Episodic** (long-term, raw) | **Semantic** (long-term, distilled) |
|---|---|---|---|
| **What it captures** | Full message list + cached memory context at every step of the graph | One `{query, answer, sub-agents used, thread_id, occurred_at}` per user turn | One stable preference per row: `{value, confidence, stated_by_user, updated_at}` |
| **Granularity** | One row per node firing | One row per turn | One row per distinct preference |
| **When written** | After every node firing (~5× per turn) | At the end of every turn (`persist_episode` node) | "🏁 End & start new conversation" button OR `distill.py` CLI |
| **Who writes it** | LangGraph runtime auto-persists state | A graph node — pure data extraction, no LLM | A separate LLM call (the distiller) reads recent episodes and proposes updates |
| **Read pattern** | Auto-loaded by LangGraph at the start of every `invoke()` for the active `thread_id` | Top-3 by **pgvector cosine similarity** to the current user query, loaded into the supervisor's system prompt every turn | All rows loaded into the supervisor's system prompt every turn (no vector search — small per-user table) |
| **Format & where** | msgpack binary in `checkpoints` + `checkpoint_blobs` tables, keyed by `(thread_id, checkpoint_ns, checkpoint_id)` | JSONB in `store` + 1024-dim embedding in `store_vectors`, namespace `memories.episodic.<user_id>` | JSONB in `store` + 1024-dim embedding in `store_vectors`, namespace `memories.semantic.<user_id>` |

### What happens in one turn

```
User: "What's the weather in Paris?"
  │
  ├─► [checkpointer load]     prior state for this thread_id (empty for new thread)
  │
  ├─► supervisor node
  │     ├─► load_memory_context:
  │     │     • SELECT all rows in memories.semantic.<user_id>       (plain SQL)
  │     │     • embed("What's the weather in Paris?") → 1024-dim
  │     │     • SELECT top-3 from memories.episodic.<user_id>
  │     │       ORDER BY embedding <=> :query_vector LIMIT 3         (pgvector!)
  │     ├─► LLM call → AIMessage(tool_calls=[call_weather_subagent])
  │     └─► [checkpointer write]
  │
  ├─► subagents node → runs sub-agent → ToolMessage
  │     └─► [checkpointer write]
  │
  ├─► supervisor node (synthesis) → AIMessage("Paris is 20°C…")
  │     └─► [checkpointer write]
  │
  ├─► persist_episode node
  │     ├─► extract last HumanMessage and last AIMessage from state
  │     ├─► store.put(("memories","episodic",user_id), random_id, {  ◄── EPISODIC write
  │     │       query: "What's the weather in Paris?",
  │     │       answer: "Paris is 20°C, cloudy",
  │     │       subagents_used: ["weather"],
  │     │       thread_id: "...",
  │     │       occurred_at: "...",
  │     │       distilled_at: null })
  │     └─► [checkpointer write]
  │
  └─► END
```

Semantic is **not** written this turn. It only gets written when:
- The user clicks the end-of-conversation button → `distill_user` reads recent
  episodic rows, makes one LLM call to propose preference updates, writes them
  to the semantic namespace, marks the episodes as `distilled_at=now()`.
- OR you run `python distill.py --user <id>` manually.

### Where it all lives

```
Lakebase Autoscale  /  project langgraph-supervisor-memory  /  branch development

  Checkpointer (short-term)              Store (long-term)
  ─────────────────────────              ─────────────────────────────────
  checkpoints                            store
  checkpoint_blobs                       store_vectors  (1024-dim embeddings)
  checkpoint_writes                          ↑
  checkpoint_migrations                      └ namespaces:
                                                ("memories","episodic",<user_id>)
                                                ("memories","semantic",<user_id>)
                                                ("meta",<user_id>)
```

## Branching strategy

Lakebase Autoscale branches are copy-on-write — cheap to spawn, isolated.
Treat them like git branches:

| Branch | Purpose | When you touch it |
|---|---|---|
| `production` | Real user data | Only on deploy |
| `development` | Day-to-day dev | Default for local app |
| `experiment-*` | Try alternate schemas / distillation prompts | Short TTL (3-7d) |

Switching is a one-line env change: `LAKEBASE_AUTOSCALING_BRANCH=...`.

## Setup

```bash
cd part-2-memory
uv venv && uv pip install -e .

databricks auth login --profile <your-profile>
cp .env.example .env
# edit .env: DATABRICKS_CONFIG_PROFILE=<your-profile>

uv run python provision_lakebase.py
# copy the two LAKEBASE_AUTOSCALING_* lines it prints into .env
```

## Run

```bash
# Streamlit UI (recommended) — sidebar has user_id selector + preference panels
uv run streamlit run app.py

# CLI — multi-turn with memory
uv run python main.py --with-memory --user-id alice@example.com --thread-id demo \
    "Convert 100 USD to JPY"
uv run python main.py --with-memory --user-id alice@example.com --thread-id demo \
    "And in EUR?"

# Force a distillation pass on demand
uv run python distill.py --user alice@example.com --min-episodes 1
```

## Files

| File | Purpose |
|---|---|
| `agent/tools.py` | 6 raw tool functions (unchanged from Part 1) |
| `agent/subagents.py` | Weather + finance sub-agents (unchanged) |
| `agent/graph.py` | Supervisor graph — loads memory context, persists episodes |
| `agent/memory.py` | Namespaces, `persist_episode`, `load_memory_context`, sidebar helpers |
| `agent/distiller.py` | L2 distillation — `distill_user(...)` (LLM call + JSON parse + merge) |
| `app.py` | Streamlit UI — user_id sidebar, Lakebase init, streaming, "End & save" button |
| `main.py` | CLI — `--with-memory` opt-in |
| `provision_lakebase.py` | One-shot Lakebase project + dev branch creation |
| `distill.py` | On-demand CLI for L2 distillation |
| `.env.example` | Template incl. `LAKEBASE_AUTOSCALING_*` and embedding endpoint |
| `pyproject.toml` | Adds `databricks-sdk>=0.81.0`, `pydantic`, `databricks-langchain[memory]` |

## Distillation: how semantic gets written

When the distiller runs, it makes one LLM call over recent undistilled
episodic rows and tags each proposed semantic update as either:

- **`stated_by_user: true`** with **confidence ≥ 0.9** — user explicitly said
  it ("remember I prefer Fahrenheit").
- **`stated_by_user: false`** with **confidence 0.5–0.8** — inferred from a
  recurring pattern across multiple turns.

Updates merge into `memories.semantic.<user_id>` by key; old episodes get
their `distilled_at` set so they aren't reprocessed.

---

# Known limitations & production evolution path

This is a prototype. Below is what we *know* is rough about the current
implementation and the direction each piece should evolve in for production.

## Memory architecture

| Limitation | Production evolution |
|---|---|
| **Confidence score is LLM-judged vibes**, not a calibrated probability. Different distillation runs can pick different numbers for the same input. | Count-based reinforcement: each distillation increments a `times_observed` counter; `confidence = min(1.0, times_observed / N)`. Number becomes a real signal. |
| **All semantic prefs loaded unconditionally** into every supervisor system prompt. Scales linearly with prefs per user (currently capped at `limit=20`). | (a) Confidence threshold filter (load only `confidence >= 0.6`). (b) Vector search relevance: if a user has >20 prefs, load top-k similar to the current query instead of all. |
| **No memory decay or pruning**. Semantic prefs persist forever; old ones can become stale or contradictory. | Time-based decay (`new_conf = old_conf * 0.95^days_since_update`); periodic consolidation job that prunes anything below threshold. |
| **No conflict resolution between similar keys**. Distiller can produce `temp_unit_pref` AND `preferred_temperature_unit` for the same idea over time. | Distillation step that semantically clusters keys and merges duplicates; or a pre-defined `Literal[...]` enum of valid keys to constrain the LLM. |
| **`DatabricksStore` embeds the whole JSON value on every write** — including fields that aren't used for search. | Set `embedding_fields=["query"]` so only the relevant text gets embedded; save compute and improve search quality. |
| **Episode schema is minimal** — just `{query, answer, sub-agents used, timestamps}`. No token counts, model used, latency, cost. | Richer episode payload with `token_usage`, `model_endpoint`, `latency_ms`, `tools_called` — enables cost reporting, perf analysis, evaluation. |
| **Episodic stored per-turn with no rollup** — storage grows linearly with turns (~5.5 KB/turn ≈ 20 MB/year for a heavy user; fine for now, unbounded long-term). Per-turn is intentional — distillation needs fine-grained frequency signal, and you can derive coarser views from fine, not the reverse. | Layered storage: keep per-turn raw for recent ~90 days (feeds distillation + similarity search), then a scheduled consolidation job collapses old turns into per-conversation summary rows under a new `("memories", "sessions", user_id)` namespace and prunes the raw rows. TTL keeps total bounded; summaries keep "browse my history" UX working forever. |
| **No schema versioning** on stored values. Changing the episode shape will break readers of old rows. | Add a `schema_version` field; readers branch on it; or run a one-shot migration job to upgrade old rows. |

## Distillation

| Limitation | Production evolution |
|---|---|
| **Distillation triggers only on explicit "End conversation" button**. Tab-close = orphaned episodes that may never get distilled. | Scheduled **Databricks Job** that runs nightly (or hourly), scans all users with undistilled episodes older than N hours, distills them. Plus retain the End button for the in-app explicit flow. |
| **Distillation is non-deterministic** — same episodes can produce different semantic updates across runs. | Run distillation in a fixed-seed mode, OR compute deterministically via clustering+counting before involving the LLM, OR run N times and take majority vote. |
| **Distiller can hallucinate preferences** with no validation against actual user behavior. | Hold proposed updates in a `pending` namespace; require confirmation before merging into `semantic`. For high-stakes prefs, surface them in the UI for the user to confirm. |
| **Distillation prompt embedded in code**, not versioned. Can't A/B test or roll back. | Move prompts into a versioned registry (MLflow Prompt Registry or simple Git-tracked JSON) with the prompt version recorded on each `last_distill` meta entry. |
| **Distillation reads ALL episodes each time and filters in Python** for `distilled_at IS NULL`. | Use `store.search(..., filter={"distilled_at": None})` if the store supports filter pushdown, or maintain a `meta.last_distilled_episode_id` watermark to skip processed rows in SQL. |
| **No "dry run" or audit log** for distillation. Can't preview what would change before applying. | Add `distill_user(..., dry_run=True)` returning the proposed updates without applying. Write a row to a `distillation_audits` table on every real run. |
| **No rollback path** — once distillation applies updates, no way to undo. | Soft-delete model: keep old semantic rows with a `superseded_at` timestamp; ability to "rewind to point-in-time" using Lakebase branching. |

## LLM cost / latency

| Limitation | Production evolution |
|---|---|
| **GPT-5.5 is used for every LLM call** — supervisor routing, sub-agent reasoning, sub-agent synthesis, supervisor synthesis, distillation. | Tier the models: supervisor routing → cheaper/faster (e.g. Llama-3-70b), sub-agents → GPT-5.5 only for hard reasoning, distillation → small model with structured-output support. Could halve cost. |
| **4 LLM calls per turn for tool-using queries** (supervisor routes → worker reasons → worker synthesizes → supervisor synthesizes). | (a) For single-subagent queries, skip the supervisor synthesis and return the worker's answer directly. (b) Or switch to a "handoff" pattern where sub-agents write the user-facing reply themselves. |
| **`with_structured_output` doesn't work cleanly with GPT-5.5 Responses API** — we fall back to JSON-in-prompt + regex-cleaned parse, which is fragile. | Either switch the distillation model to one that supports clean structured output, OR validate aggressively and fall back to a retry loop, OR migrate to JSON Schema mode once databricks-langchain supports it for the Responses API. |
| **Per-sub-agent prompts are hardcoded**; can't be tuned per user/context. | Make prompts loadable from the store at runtime so they can be A/B tested or personalized. |

## Operational

| Limitation | Production evolution |
|---|---|
| **`load_memory_context` runs on every supervisor firing** — twice per tool-using turn. Same SQL, same embedding, redundant. *Fixed in this iteration via a `memory_context` state field.* | Already fixed — keep an eye on cache invalidation if we add user-impact mid-turn (e.g., a tool that updates a preference). |
| **All store/checkpoint operations are sync** — works for local Streamlit but blocks the event loop. | For Databricks Apps deployment, swap to `AsyncCheckpointSaver` + `AsyncDatabricksStore` + FastAPI handlers (the Databricks template's pattern). |
| **MLflow async tracing disabled** for local dev (so traces appear instantly). | Re-enable for production via env var; tail latency goes down. |
| **No connection pool tuning** — defaults from `databricks-langchain`. | Tune pool size to expected concurrency; monitor connection saturation in production. |
| **No automatic Lakebase OAuth token refresh in long-running workers**. The library handles short ops but a multi-hour distillation could hit the 1h expiry. | Wrap long-running ops in a retry-on-401 with re-auth; or chunk into batches that complete inside 1h. |
| **No retry policy on distillation** if the LLM call fails or returns malformed JSON. | Add backoff + retry with stricter system prompt on each attempt; flag persistent failures in MLflow. |

## Identity & security

| Limitation | Production evolution |
|---|---|
| **User_id is free-text from a sidebar input** — no auth, no validation, easy to spoof or typo. | In Databricks Apps deployment, read `X-Forwarded-User` from the request — workspace identity is the source of truth. Drop the sidebar input. |
| **No "forget this user" operation** (GDPR right-to-erasure). | Add a `delete_user(user_id)` that wipes all `memories.*.<user_id>` namespaces, all checkpoints for that user's threads, and logs the deletion. |
| **No row-level security** — all data lives in one branch, accessible by anyone with the connection string. | Use Unity Catalog ABAC/RLS to restrict `user_id` access by workspace identity. Becomes critical when multiple users share an environment. |
| **Secrets in `.env`** — fine for local, leaks-prone for prod. | Use Databricks Secrets / Azure Key Vault; never commit env files. |
| **No PII handling** — emails stored verbatim as namespace segments and in episode `query` text. | Hash user_ids for the namespace key; redact PII from episode payloads before persisting (regex on common patterns, or run through an LLM-based scrubber). |

## UX / Streamlit-specific

| Limitation | Production evolution |
|---|---|
| **Streamlit reruns the entire script on every interaction** — sidebar memory panels re-query Lakebase on every keystroke in the chat input. | Cache panel reads in `st.session_state` with TTL; invalidate on known write events. |
| **No way to switch to a past thread** — sidebar lists past sessions but can't reload one. | Add a click-to-load action on each session entry; restore the conversation via the existing `thread_id` and have the checkpointer hydrate the history. |
| **No pagination on past sessions** — `limit=500` is the hard cap. | Add `LIMIT/OFFSET` pagination and a "load more" affordance, or summarize sessions older than N days into a single "archive" entry. |
| **No streaming for the distillation pass** — user sees a blocking spinner. | Stream tokens from the distiller LLM into a status panel so the user can watch it think. |

## Testing & deployment

| Limitation | Production evolution |
|---|---|
| **Zero automated test coverage**. | Add: (a) unit tests with mock store/LLM for `persist_episode`, `load_memory_context`, distillation merge logic. (b) integration tests against an ephemeral Lakebase branch (`experiment-test-*`) that gets torn down after CI runs. |
| **No CI/CD** — purely local development. | GitHub Actions: lint + unit tests on every PR; integration tests against an ephemeral branch on `main`. Auto-deploy to a staging Databricks App on merge. |
| **No rate limiting** — a runaway client could hammer the model endpoint. | Token bucket per user_id; surface 429s gracefully in the UI. |
| **No deployment story yet** — that's Part 3. | Part 3: package as a Databricks App + add the scheduled distillation Job + swap sync→async + bind to workspace identity. |

---

These are intentionally listed up front so anyone reading the code knows
where the seams are. Part 3 (or beyond) is where most of these get hardened.
