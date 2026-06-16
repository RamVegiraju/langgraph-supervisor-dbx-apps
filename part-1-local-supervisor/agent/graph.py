import os

from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from agent.subagents import build_finance_agent, build_weather_agent

SUPERVISOR_PROMPT = (
    "You are a supervisor coordinating two specialist sub-agents: "
    "call_weather_subagent (for weather questions) and call_finance_subagent "
    "(for currency conversion or exchange rates). "
    "Delegate the user's question to the right sub-agent — call both in parallel "
    "in a single turn if the question needs both. "
    "After the sub-agents respond, synthesize their answers into one short reply. "
    "If the question does not need a sub-agent (e.g. greetings), respond directly."
)


def _extract_text(content) -> str:
    """Responses API messages carry a list of typed blocks; flatten to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def build_agent(endpoint: str | None = None):
    """Build the supervisor graph.

    Shape (same as Part 2's custom ReAct, but the tools are sub-agent delegations):

        START -> supervisor --(tools_condition)--+-> subagents -> supervisor
                                                 +-> END
    """
    llm = ChatDatabricks(
        endpoint=endpoint or os.environ.get("MODEL_ENDPOINT", "databricks-gpt-5-5"),
        use_responses_api=True,
    )

    weather_agent = build_weather_agent()
    finance_agent = build_finance_agent()

    @tool
    def call_weather_subagent(query: str) -> str:
        """Delegate a weather question to the weather specialist.

        Args:
            query: The user's weather question, e.g. "What's the weather in Paris?".
        """
        result = weather_agent.invoke({"messages": [HumanMessage(query)]})
        return _extract_text(result["messages"][-1].content)

    @tool
    def call_finance_subagent(query: str) -> str:
        """Delegate a currency / FX question to the finance specialist.

        Args:
            query: The user's finance question, e.g. "Convert 100 USD to EUR".
        """
        result = finance_agent.invoke({"messages": [HumanMessage(query)]})
        return _extract_text(result["messages"][-1].content)

    subagent_tools = [call_weather_subagent, call_finance_subagent]
    supervisor_llm = llm.bind_tools(subagent_tools)

    def supervisor_node(state: MessagesState) -> dict:
        response = supervisor_llm.invoke(
            [SystemMessage(SUPERVISOR_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    return (
        StateGraph(MessagesState)
        .add_node(
            "supervisor",
            supervisor_node,
            retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
        )
        .add_node("subagents", ToolNode(subagent_tools, handle_tool_errors=True))
        .add_edge(START, "supervisor")
        .add_conditional_edges(
            "supervisor", tools_condition, {"tools": "subagents", END: END}
        )
        .add_edge("subagents", "supervisor")
        .compile()
    )
