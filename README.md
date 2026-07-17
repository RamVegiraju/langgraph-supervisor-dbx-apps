# LangGraph Supervisor on Databricks
A LangGraph supervisor agent that delegates to specialist sub-agents (weather +
finance), running against Databricks foundation models. Built up over two parts.

## [YouTube Memory Enablement Series](A LangGraph supervisor agent that delegates to specialist sub-agents (weather +
finance), running against Databricks foundation models. Built up over two parts.)

## Directories

### [`part-1-local-supervisor`](./part-1-local-supervisor)
The base agent. A LangGraph supervisor that routes each question to a weather or
finance sub-agent (or both, in parallel), then synthesizes one answer. Runs
locally against a Databricks model endpoint, with a Streamlit UI showing live
routing and MLflow traces. No memory — each query is independent.

### [`part-2-memory`](./part-2-memory)
Adds memory on top of Part 1, backed by Lakebase (Postgres). Three layers:
- **Short-term** — a LangGraph checkpointer persists conversation state per thread.
- **Episodic** (long-term) — every turn is saved and retrieved by similarity to the current query.
- **Semantic** (long-term) — an LLM distills past turns into stable user preferences.

Each part has its own README with setup and run instructions.
