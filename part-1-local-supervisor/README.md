# Part 1 — Local LangGraph Supervisor

A LangGraph supervisor agent that delegates to two specialist sub-agents
(weather + finance), running locally against Databricks GPT-5.5 through the AI
Gateway. Ships with a Streamlit UI showing live routing, token streaming, and
MLflow span traces with latencies.

**Part 1** of a series. Future parts will add Lakebase-backed memory and
deployment to Databricks Apps.

## Architecture

```
   Streamlit UI ──► Supervisor (custom StateGraph)
                       │
                       ├─► call_weather_subagent ─► weather sub-agent (ReAct)
                       │                              │
                       │                              └── get_weather · get_forecast · get_air_quality
                       │
                       └─► call_finance_subagent ─► finance sub-agent (ReAct)
                                                      │
                                                      └── convert_currency · get_historical_rate · compare_currencies
```

The supervisor exposes each sub-agent as a single tool (`call_*_subagent`). The
supervisor LLM decides which to delegate to — multiple in parallel when a query
needs both, none at all for chitchat. After sub-agents respond, the supervisor
synthesizes one final answer.

Every node, tool, and LLM call is auto-instrumented by
`mlflow.langchain.autolog()` and rendered as a span tree in the UI.

## Setup

```bash
cd part-1-local-supervisor

# Install
uv venv && uv pip install -e .

# Databricks auth — OAuth into your workspace
databricks auth login --profile <your-profile>

# Point the app at that profile
cp .env.example .env
# edit .env and set DATABRICKS_CONFIG_PROFILE=<your-profile>
```

## Run

```bash
# Streamlit UI at http://localhost:8501
uv run streamlit run app.py

# CLI — one-shot question, -v shows tool calls
uv run python main.py -v "What's the weather in Paris?"
```

## Files

| File | Purpose |
|---|---|
| `agent/tools.py` | 6 raw tool functions (Open-Meteo, Frankfurter APIs — both free, no key) |
| `agent/subagents.py` | Builds the weather and finance ReAct sub-agents |
| `agent/graph.py` | Supervisor `StateGraph`: wraps each sub-agent as a tool, wires the loop |
| `app.py` | Streamlit UI — streams tokens, captures routing live, renders MLflow trace |
| `main.py` | CLI entry point |
| `.env.example` | Template for `DATABRICKS_CONFIG_PROFILE` and `MODEL_ENDPOINT` |
| `pyproject.toml` | Python dependencies |

## Example queries

- `What's the weather in Paris?` — single sub-agent, single tool
- `Tokyo right now and the 4-day forecast?` — single sub-agent, two tools in parallel
- `Trip to Tokyo with $500 — what's the weather and how much is that in yen?` — both sub-agents in parallel
- `What can you help me with?` — supervisor answers directly, no sub-agents invoked
