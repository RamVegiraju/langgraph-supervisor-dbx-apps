import os

from databricks_langchain import ChatDatabricks
from langgraph.prebuilt import create_react_agent

from agent.tools import (
    compare_currencies,
    convert_currency,
    get_air_quality,
    get_forecast,
    get_historical_rate,
    get_weather,
)

WEATHER_TOOLS = [get_weather, get_forecast, get_air_quality]
FINANCE_TOOLS = [convert_currency, get_historical_rate, compare_currencies]

WEATHER_PROMPT = (
    "You are a weather specialist. You can look up current conditions "
    "(get_weather), multi-day forecasts (get_forecast), and air quality "
    "(get_air_quality). Pick the right tool(s) for the question — call multiple "
    "in parallel if the question needs more than one — and answer concisely."
)

FINANCE_PROMPT = (
    "You are a finance specialist. You can convert at the latest rate "
    "(convert_currency), look up historical rates (get_historical_rate), and "
    "convert one amount to many currencies at once (compare_currencies). "
    "Prefer compare_currencies when the user is asking about multiple targets. "
    "Pick the right tool(s) and answer concisely."
)


def _llm():
    return ChatDatabricks(
        endpoint=os.environ.get("MODEL_ENDPOINT", "databricks-gpt-5-5"),
        use_responses_api=True,
    )


def build_weather_agent():
    return create_react_agent(_llm(), WEATHER_TOOLS, prompt=WEATHER_PROMPT)


def build_finance_agent():
    return create_react_agent(_llm(), FINANCE_TOOLS, prompt=FINANCE_PROMPT)
