"""Weather agent using Xweather MCP server and LangGraph.

This module provides a LangGraph-based weather agent that connects to the
Xweather MCP server for real-time weather data.

Implements a Professional ReAct Loop:
- Analyzer: Classifies query complexity (simple_forecast vs deep_reanalysis)
- Tools: Executes Xweather MCP tools via ToolNode
- Synthesizer: Merges data into professional meteorological reports

Requires Python 3.10+ and langchain-mcp-adapters package.
"""

import asyncio
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.state import AgentState, WeatherConfig


# System prompts for each node
ANALYZER_SYSTEM_PROMPT = """You are a Senior Meteorological Research Lead at a top-tier climate research institution.

Your task is to analyze the user's weather query and determine the appropriate response type:

1. **simple_forecast**: Basic weather queries like current conditions, simple forecasts, or single-location requests.
   Examples: "What's the weather in Paris?", "Will it rain tomorrow in NYC?"

2. **deep_reanalysis**: Complex queries requiring scientific analysis, multi-factor assessment, or professional-grade insights.
   Examples: "Analyze weather patterns for outdoor event planning this week", "What are the agricultural impacts of current conditions?"

After classifying, use the appropriate weather tools to gather data. Always provide evidence-backed insights using precise scientific terminology."""

SYNTHESIZER_SYSTEM_PROMPT = """You are a Senior Meteorological Research Lead providing a professional analyst report.

Based on the weather data gathered, synthesize a comprehensive response following this structure:

## Summary
A 1-2 sentence executive summary of the key findings.

## Analysis
Detailed breakdown of the weather conditions with:
- Precise measurements and scientific terminology
- Confidence levels where applicable
- Relevant meteorological context

## Recommendations
Actionable insights based on the analysis (if applicable to the query).

Always cite data sources implicitly by referencing the specific measurements. Maintain a professional yet accessible tone."""


class MCPConnectionError(Exception):
    """Raised when connection to the MCP server fails."""

    pass


class WeatherAgent:
    """LangGraph-based weather agent using Xweather MCP tools.

    Implements a Professional ReAct Loop with three specialized nodes:
    - Analyzer: Classifies query and initiates tool calls
    - Tools: Executes Xweather MCP operations
    - Synthesizer: Creates professional meteorological reports

    Args:
        config: WeatherConfig with connection settings.

    Raises:
        MCPConnectionError: If connection to MCP server fails.
        ValueError: If required credentials are not provided.
    """

    TOOL_NAMES = [
        "xweather_get_current_weather",
        "xweather_get_weather_forecast",
        "xweather_get_weather_impacts",
    ]

    def __init__(self, config: WeatherConfig | None = None):
        """Initialize the weather agent.

        Args:
            config: Optional WeatherConfig. If not provided, loads from
                environment variables via pydantic-settings.

        Raises:
            ValueError: If required environment variables are not set.
        """
        if config is None:
            config = WeatherConfig()

        self.config = config
        self._tools: list[BaseTool] = []
        self._graph: StateGraph | None = None
        self._compiled_graph: Any = None
        self._mcp_client: MultiServerMCPClient | None = None
        self._checkpointer: MemorySaver = MemorySaver()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the MCP client and build the graph.

        This method must be called before using the agent.

        Raises:
            MCPConnectionError: If connection to MCP server fails.
        """
        if self._initialized:
            return

        await self._initialize_tools()
        self._build_graph()
        self._initialized = True

    async def _initialize_tools(self) -> None:
        """Initialize MCP tools for weather operations.

        Raises:
            MCPConnectionError: If connection to MCP server fails.
        """
        try:
            if self.config.transport == "stdio":
                self._mcp_client = MultiServerMCPClient(
                    {
                        "xweather": {
                            "transport": "stdio",
                            "command": "npx",
                            "args": ["-y", "@xweather/mcp-server"],
                            "env": {
                                "XWEATHER_CLIENT_ID": self.config.xweather_client_id,
                                "XWEATHER_CLIENT_SECRET": self.config.xweather_client_secret,
                            },
                        }
                    }
                )
            else:
                self._mcp_client = MultiServerMCPClient(
                    {
                        "xweather": {
                            "transport": "streamable_http",
                            "url": self.config.mcp_endpoint,
                            "headers": {
                                "Authorization": f"Bearer {self.config.api_key}"
                            },
                        }
                    }
                )

            all_tools = await self._mcp_client.get_tools()
            self._tools = [
                tool for tool in all_tools if tool.name in self.TOOL_NAMES
            ]

            if not self._tools:
                self._tools = all_tools

            if not self._tools:
                raise MCPConnectionError(
                    "No tools available from MCP server. "
                    "Check your credentials and connection."
                )

        except MCPConnectionError:
            raise
        except Exception as e:
            raise MCPConnectionError(
                f"Failed to connect to Xweather MCP server: {e}"
            ) from e

    def _build_graph(self) -> None:
        """Build the Professional ReAct Loop workflow."""
        # Primary LLM for analysis with tools
        analyzer_llm = ChatAnthropic(
            model=self.config.model_name,
            temperature=0,
        )
        analyzer_with_tools = analyzer_llm.bind_tools(self._tools)

        # Synthesizer LLM for professional reports
        synthesizer_llm = ChatAnthropic(
            model=self.config.synthesizer_model,
            temperature=0.3,
        )

        def analyzer_node(state: AgentState) -> dict:
            """Analyze query and determine response strategy.

            Classifies the query as simple_forecast or deep_reanalysis,
            then initiates appropriate tool calls.

            Args:
                state: Current agent state with messages.

            Returns:
                Updated state with analysis type and tool calls.
            """
            messages = state["messages"]

            # Inject system prompt for analyzer
            system_msg = SystemMessage(content=ANALYZER_SYSTEM_PROMPT)
            messages_with_system = [system_msg] + list(messages)

            response = analyzer_with_tools.invoke(messages_with_system)

            # Determine analysis type from response
            # Handle content that can be string or list of content blocks
            content = response.content
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            content_lower = content.lower() if content else ""

            if any(kw in content_lower for kw in ["deep", "complex", "analysis", "impact", "pattern"]):
                analysis_type = "deep_reanalysis"
            else:
                analysis_type = "simple_forecast"

            return {
                "messages": [response],
                "analysis_type": analysis_type,
            }

        def synthesizer_node(state: AgentState) -> dict:
            """Synthesize weather data into professional report.

            Uses Claude 3.7 to create a comprehensive meteorological
            analysis report based on gathered data.

            Args:
                state: Current agent state with tool results.

            Returns:
                Final synthesized response.
            """
            messages = state["messages"]
            analysis_type = state.get("analysis_type", "simple_forecast")

            # For simple queries, return the last AI response directly
            if analysis_type == "simple_forecast":
                # Find the last AI message that isn't a tool call
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                        return {"messages": []}  # No additional message needed

            # For deep analysis, synthesize a professional report
            system_msg = SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT)

            # Build context from conversation
            synthesis_prompt = HumanMessage(
                content="Based on the weather data gathered above, provide a professional meteorological analysis report."
            )

            synthesis_messages = [system_msg] + list(messages) + [synthesis_prompt]
            response = synthesizer_llm.invoke(synthesis_messages)

            return {"messages": [response]}

        def route_after_analyzer(state: AgentState) -> Literal["tools", "synthesizer"]:
            """Route based on whether tools need to be called.

            Args:
                state: Current agent state.

            Returns:
                Next node: 'tools' if tool calls present, else 'synthesizer'.
            """
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "synthesizer"

        def route_after_tools(state: AgentState) -> Literal["analyzer", "synthesizer"]:
            """Route after tool execution.

            Args:
                state: Current agent state.

            Returns:
                Next node: 'analyzer' for more tools, 'synthesizer' when done.
            """
            # Check if the previous AI message still has pending logic
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        # Tool was just executed, go back to analyzer for potential follow-up
                        return "analyzer"
                    break

            return "synthesizer"

        # Build the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyzer", analyzer_node)
        workflow.add_node("tools", ToolNode(self._tools))
        workflow.add_node("synthesizer", synthesizer_node)

        # Set entry point
        workflow.set_entry_point("analyzer")

        # Add conditional edges
        workflow.add_conditional_edges(
            "analyzer",
            route_after_analyzer,
            {"tools": "tools", "synthesizer": "synthesizer"},
        )
        workflow.add_conditional_edges(
            "tools",
            route_after_tools,
            {"analyzer": "analyzer", "synthesizer": "synthesizer"},
        )
        workflow.add_edge("synthesizer", END)

        self._graph = workflow
        self._compiled_graph = workflow.compile(checkpointer=self._checkpointer)

    @property
    def tools(self) -> list[BaseTool]:
        """Get the list of MCP tools.

        Returns:
            List of initialized LangChain tool instances.
        """
        return self._tools

    @property
    def graph(self) -> StateGraph | None:
        """Get the LangGraph workflow.

        Returns:
            The StateGraph instance or None if not initialized.
        """
        return self._graph

    def _check_initialized(self) -> None:
        """Check if the agent has been initialized.

        Raises:
            RuntimeError: If the agent has not been initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "Agent not initialized. Call 'await agent.initialize()' first."
            )

    def invoke(self, query: str, thread_id: str | None = None) -> dict:
        """Process a weather-related query synchronously.

        Args:
            query: Natural language weather query.
            thread_id: Optional thread ID for multi-turn conversation persistence.
                If provided, conversation history is maintained across calls.

        Returns:
            Agent response with weather information.

        Raises:
            MCPConnectionError: If MCP server connection fails during query.
            RuntimeError: If agent has not been initialized.
        """
        self._check_initialized()
        try:
            input_state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "analysis_type": "simple_forecast",
            }

            # Configure with thread_id for persistence
            config = {}
            if thread_id:
                config = {"configurable": {"thread_id": thread_id}}

            result = self._compiled_graph.invoke(input_state, config)
            return result
        except Exception as e:
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                raise MCPConnectionError(
                    f"Lost connection to Xweather MCP server: {e}"
                ) from e
            raise

    async def ainvoke(self, query: str, thread_id: str | None = None) -> dict:
        """Process a weather-related query asynchronously.

        Args:
            query: Natural language weather query.
            thread_id: Optional thread ID for multi-turn conversation persistence.
                If provided, conversation history is maintained across calls.

        Returns:
            Agent response with weather information.

        Raises:
            MCPConnectionError: If MCP server connection fails during query.
            RuntimeError: If agent has not been initialized.
        """
        self._check_initialized()
        try:
            input_state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "analysis_type": "simple_forecast",
            }

            # Configure with thread_id for persistence
            config = {}
            if thread_id:
                config = {"configurable": {"thread_id": thread_id}}

            result = await self._compiled_graph.ainvoke(input_state, config)
            return result
        except Exception as e:
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                raise MCPConnectionError(
                    f"Lost connection to Xweather MCP server: {e}"
                ) from e
            raise

    def get_conversation_history(self, thread_id: str) -> list:
        """Retrieve conversation history for a thread.

        Args:
            thread_id: The thread ID to retrieve history for.

        Returns:
            List of messages in the conversation.
        """
        self._check_initialized()
        config = {"configurable": {"thread_id": thread_id}}
        state = self._compiled_graph.get_state(config)
        if state and state.values:
            return state.values.get("messages", [])
        return []

    async def close(self) -> None:
        """Close the MCP client connection."""
        if self._mcp_client:
            if hasattr(self._mcp_client, "close"):
                await self._mcp_client.close()
            self._mcp_client = None
            self._initialized = False


async def create_weather_agent(config: WeatherConfig | None = None) -> WeatherAgent:
    """Factory function to create and initialize a WeatherAgent.

    Args:
        config: Optional WeatherConfig. If not provided, loads from
            environment variables.

    Returns:
        Configured and initialized WeatherAgent instance.

    Raises:
        ValueError: If required environment variables are not set.
        MCPConnectionError: If connection to MCP server fails.
    """
    agent = WeatherAgent(config)
    await agent.initialize()
    return agent


def create_weather_agent_sync(config: WeatherConfig | None = None) -> WeatherAgent:
    """Synchronous factory function to create and initialize a WeatherAgent.

    Args:
        config: Optional WeatherConfig. If not provided, loads from
            environment variables.

    Returns:
        Configured and initialized WeatherAgent instance.

    Raises:
        ValueError: If required environment variables are not set.
        MCPConnectionError: If connection to MCP server fails.
    """
    return asyncio.run(create_weather_agent(config))
