# Part 2 — Memory (Lakebase-backed)

Three memory layers on top of the Part 1 supervisor, all backed by one Lakebase
Autoscale project.

Terminology follows two Databricks references:

- [Memory Scaling](https://www.databricks.com/blog/memory-scaling-ai-agents) —
  *episodic* memories are raw records of past interactions; *semantic* memories
  are generalized facts distilled from them. The payoff comes from keeping
  memory **efficient** — distilling raw turns into a few stable facts and
  retrieving only what's relevant per query — not from piling more context into
  the prompt, which dilutes it. No retraining either way.
- [MemAlign](https://www.databricks.com/blog/memalign-building-better-llm-judges-human-feedback-scalable-memory) —
  the **dual-memory** idea this borrows: distilled semantic *principles* +
  retrieved episodic *examples*, assembled per call. (MemAlign itself targets
  LLM judges aligned by expert feedback; here memory is built automatically from
  user turns, so this takes the shape, not the whole framework.)

> Part 2 is intentionally a prototype. Known limitations and the production
> evolution path are at the bottom of this README.

## The three memory layers

| | **Checkpointer** (short-term) | **Episodic** (long-term, raw) | **Semantic** (long-term, distilled) |
|---|---|---|---|
| **What it captures** | Full message list + cached memory context at every step | One `{query, answer, sub-agents used, thread_id, occurred_at}` per turn | One stable preference per row: `{value, confidence, stated_by_user, updated_at}` |
| **Granularity** | One row per node firing | One row per turn | One row per distinct preference |
| **When written** | After every node firing (~5× per turn) | End of every turn (`persist_episode` node) | "🏁 End conversation" button OR `distill.py` |
| **Who writes it** | LangGraph runtime (auto) | A graph node — pure data extraction, no LLM | The distiller — one LLM call over recent episodes |
| **How it's read** | Auto-loaded by LangGraph each `invoke()` for the active `thread_id` | **Top-3 by pgvector cosine similarity** to the current query — reloaded **every turn** | **All** rows, no vector search — loaded **once per thread** and cached in graph state |
| **Format & where** | msgpack in `checkpoints` + `checkpoint_blobs`, keyed by `(thread_id, checkpoint_ns, checkpoint_id)` | JSONB in `store` + embedding in `store_vectors`, ns `memories.episodic.<user_id>` | JSONB in `store` + embedding in `store_vectors`, ns `memories.semantic.<user_id>` |

**Why semantic loads once but episodic every turn:** semantic prefs don't depend
on the question (same answer regardless of what's asked), so they're loaded on
the first turn of a thread and reused from graph state thereafter. Episodic
recall *is* query-dependent (top-3 most similar past turns), so it's recomputed
each turn. This saves one embedding round-trip + one query on every turn after
the first.

### What happens in one turn

```
User: "What's the weather in Paris?"
  │
  ├─► [checkpointer load]   prior state for this thread_id (empty for a new thread)
  │
  ├─► supervisor node  — assembles "working memory":
  │     ├─ load_episodic_context  (EVERY turn, query-dependent):
  │     │     embed(query) → 1024-dim
  │     │     SELECT top-3 FROM memories.episodic.<user_id>
  │     │       ORDER BY embedding <=> :query_vector LIMIT 3      (pgvector)
  │     ├─ load_semantic_context  (FIRST turn only, then cached in state):
  │     │     SELECT all rows FROM memories.semantic.<user_id>    (plain SQL)
  │     ├─ LLM call → AIMessage(tool_calls=[call_weather_subagent])
  │     └─ [checkpointer write]
  │
  ├─► subagents node → runs sub-agent → ToolMessage
  │     └─ [checkpointer write]
  │
  ├─► supervisor node (synthesis) → AIMessage("Paris is 20°C…")
  │     └─ reuses cached working memory · [checkpointer write]
  │
  ├─► persist_episode node                                        ◄── EPISODIC write
  │     store.put(("memories","episodic",user_id), id, {
  │       query, answer, subagents_used, thread_id, occurred_at, distilled_at: null })
  │     └─ [checkpointer write]
  │
  └─► END
```

Semantic is **not** written this turn. It's written only when distillation runs
(the "End conversation" button or `distill.py`): `distill_user` reads recent
episodic rows, makes one LLM call to propose preference updates, writes them to
the semantic namespace, and marks those episodes `distilled_at=now()`.

### Where it all lives

`project langgraph-supervisor-memory / branch development`. Two subsystems, each
splitting its data from a library-managed `*_migrations` version table:

| Table(s) | Holds |
|---|---|
| `checkpoints` · `checkpoint_blobs` · `checkpoint_writes` | **Short-term**: per-`thread_id` state snapshots (index · payload · pending writes) |
| `store` | **Long-term values** (JSONB) — episodic episodes + semantic prefs |
| `store_vectors` | **Long-term embeddings** (1024-dim) for pgvector similarity search |

Every searchable item appears in **both** `store` (its value) and `store_vectors`
(its embedding). Inside both, rows are grouped by namespace:

```
("memories","episodic",<user_id>)   raw turns
("memories","semantic",<user_id>)   distilled prefs
("meta",<user_id>)                  distillation watermark
```

## Branching strategy

Lakebase Autoscale branches are copy-on-write — cheap to spawn, isolated. Treat
them like git branches:

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

# Seed the tables with demo traffic across several users, then distill each
uv run python populate_demo.py
```

## Files

| File | Purpose |
|---|---|
| `agent/tools.py` | 6 raw tool functions (unchanged from Part 1) |
| `agent/subagents.py` | Weather + finance sub-agents (unchanged) |
| `agent/graph.py` | Supervisor graph — assembles working memory, persists episodes |
| `agent/memory.py` | Namespaces, `persist_episode`, `load_semantic_context` / `load_episodic_context`, sidebar helpers |
| `agent/distiller.py` | Distillation — `distill_user(...)` (LLM call + JSON parse + merge) |
| `app.py` | Streamlit UI — user_id sidebar, Lakebase init, streaming, "End & save" button |
| `main.py` | CLI — `--with-memory` opt-in |
| `provision_lakebase.py` | One-shot Lakebase project + dev branch creation |
| `distill.py` | On-demand CLI for distillation |
| `populate_demo.py` | Dev seeder — replays multi-turn conversations across several users, then distills each |
| `.env.example` | Template incl. `LAKEBASE_AUTOSCALING_*` and embedding endpoint |
| `pyproject.toml` | Adds `databricks-sdk>=0.81.0`, `pydantic`, `databricks-langchain[memory]` |

## Distillation: how semantic gets written

When the distiller runs, it makes one LLM call over recent undistilled episodic
rows and tags each proposed semantic update as either:

- **`stated_by_user: true`**, confidence **≥ 0.9** — user explicitly said it
  ("remember I prefer Fahrenheit").
- **`stated_by_user: false`**, confidence **0.5–0.8** — inferred from a recurring
  pattern across turns.

Updates merge into `memories.semantic.<user_id>` by key; processed episodes get
`distilled_at` set so they aren't reprocessed.

---

# Known limitations & production evolution path

This is a prototype. Each row is **what's rough now → where it should go**. Listed
up front so anyone reading the code knows where the seams are; Part 3 (or beyond)
hardens most of them.

## Memory & distillation

| Limitation | Production evolution |
|---|---|
| Confidence is LLM-guessed, not calibrated — same input can score differently across runs. | Count-based reinforcement: `confidence = min(1, times_observed / N)`. |
| **All** semantic prefs load into the prompt (no confidence/relevance filter; capped at 20). | Filter by `confidence >= 0.6`; switch to vector top-k once a user exceeds ~20 prefs. |
| No decay or pruning — prefs live forever and can go stale or contradict. | Time-decay (`conf *= 0.95^days`) + a consolidation job that prunes below threshold. |
| No conflict resolution — distiller can emit `temp_unit` *and* `preferred_temperature_unit` for one idea. | Cluster + merge duplicate keys, or constrain keys to a fixed `Literal[...]` enum. |
| Distillation only fires on the "End conversation" button — tab-close orphans episodes. | Nightly Databricks Job distills any user with stale undistilled episodes (keep the button too). |
| Distillation is non-deterministic and can hallucinate prefs. | Fixed-seed / majority-vote; stage proposals in a `pending` ns and confirm before merging. |
| `DatabricksStore` embeds the whole JSON value, not just the searchable text. | `embedding_fields=["query"]` — cheaper writes, better recall. |
| Episode schema is minimal and stored per-turn with no rollup (grows ~20 MB/yr/heavy user). | Richer payload (tokens, latency, cost); keep raw ~90d, then roll up to per-session summaries + TTL. |
| No schema versioning on stored values — shape changes break old readers. | Add `schema_version`; branch on read, or migrate old rows once. |

Per-turn episodic storage is **intentional** — distillation needs fine-grained
frequency signal, and you can derive coarse views from fine, not the reverse.

## LLM cost & latency

| Limitation | Production evolution |
|---|---|
| GPT-5.5 on every call (routing, both sub-agent steps, synthesis, distillation). | Tier models: cheap/fast for routing + distillation, GPT-5.5 only for hard reasoning. |
| 4 LLM calls per tool-using turn. | Skip supervisor synthesis for single-sub-agent queries, or hand off the reply to the sub-agent. |
| `with_structured_output` is fragile on the GPT-5.5 Responses API — falls back to regex JSON parse. | Use a model with clean structured output, or JSON Schema mode once supported. |

## Operational & security

| Limitation | Production evolution |
|---|---|
| All store/checkpoint ops are **sync** — fine for Streamlit, blocks the event loop. | Async checkpointer + store + FastAPI handlers for the Databricks Apps deploy. |
| `user_id` is free-text from the sidebar — spoofable, typo-prone. | Read workspace identity (`X-Forwarded-User`) in the Apps deploy; drop the input. |
| No "forget this user" (GDPR), no row-level security, no PII handling. | `delete_user(user_id)` wipe; UC ABAC/RLS by identity; hash user_ids + redact PII. |
| Secrets live in `.env`; no Lakebase token refresh for multi-hour workers; no distillation retry. | Databricks Secrets; retry-on-401 re-auth; backoff + stricter retry on malformed JSON. |

## Testing & deployment

| Limitation | Production evolution |
|---|---|
| Zero test coverage, no CI/CD, no rate limiting. | Unit tests (mock store/LLM) + integration tests on an ephemeral branch; GitHub Actions; per-user token bucket. |
| No deployment story yet. | **Part 3**: package as a Databricks App + scheduled distillation Job + sync→async + workspace identity. |
