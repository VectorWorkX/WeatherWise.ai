"""Weather agent using Xweather MCP server and LangGraph.

This module provides a LangGraph-based weather agent that connects to the
Xweather MCP server for real-time weather data.

Supports both local (stdio) and remote (SSE) MCP server connections.
Requires Python 3.10+ and langchain-mcp-adapters package.
"""

import asyncio
import os
from typing import Annotated, Any, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


class WeatherAgentConfig(BaseModel):
    """Configuration for the Weather Agent.

    Args:
        transport: Transport type - 'stdio' for local, 'streamable_http' for remote.
        client_id: Xweather client ID.
        client_secret: Xweather client secret.
        mcp_endpoint: URL of remote MCP server (for streamable_http transport).
        model_name: Anthropic model to use.
    """

    transport: Literal["stdio", "streamable_http"] = Field(
        default="streamable_http",
        description="MCP transport type",
    )
    client_id: str | None = Field(
        default=None,
        description="Xweather client ID",
    )
    client_secret: str | None = Field(
        default=None,
        description="Xweather client secret",
    )
    mcp_endpoint: str = Field(
        default="https://mcp.api.xweather.com/mcp",
        description="Xweather Remote MCP server endpoint",
    )
    model_name: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model identifier",
    )


class AgentState(TypedDict):
    """State schema for the weather agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]


class MCPConnectionError(Exception):
    """Raised when connection to the MCP server fails."""

    pass


class WeatherAgent:
    """LangGraph-based weather agent using Xweather MCP tools.

    This agent connects to the Xweather MCP server and provides
    weather-related capabilities through LangGraph orchestration.

    Supports two modes:
    - Local (stdio): Uses npx @xweather/mcp-server with CLIENT_ID/SECRET
    - Remote (sse): Uses https endpoint with OAuth token

    Args:
        config: WeatherAgentConfig with connection settings.

    Raises:
        MCPConnectionError: If connection to MCP server fails.
        ValueError: If required credentials are not provided.
    """

    TOOL_NAMES = [
        "xweather_get_current_weather",
        "xweather_get_weather_forecast",
        "xweather_get_weather_impacts",
    ]

    def __init__(self, config: WeatherAgentConfig | None = None):
        """Initialize the weather agent.

        Args:
            config: Optional configuration. If not provided, uses environment
                variables for credentials and defaults to streamable_http transport.

        Raises:
            ValueError: If required environment variables are not set.
        """
        if config is None:
            client_id = os.getenv("XWEATHER_CLIENT_ID")
            client_secret = os.getenv("XWEATHER_CLIENT_SECRET")
            if not client_id or not client_secret:
                raise ValueError(
                    "XWEATHER_CLIENT_ID and XWEATHER_CLIENT_SECRET "
                    "environment variables are required"
                )
            config = WeatherAgentConfig(
                transport="streamable_http",
                client_id=client_id,
                client_secret=client_secret,
            )

        self.config = config
        self._tools: list[BaseTool] = []
        self._graph: StateGraph | None = None
        self._compiled_graph: Any = None
        self._mcp_client: MultiServerMCPClient | None = None
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
            if not self.config.client_id or not self.config.client_secret:
                raise ValueError("client_id and client_secret are required")

            # Create API key by combining client_id and client_secret
            api_key = f"{self.config.client_id}_{self.config.client_secret}"

            if self.config.transport == "stdio":
                # Local MCP server via npx (if package becomes available)
                self._mcp_client = MultiServerMCPClient(
                    {
                        "xweather": {
                            "transport": "stdio",
                            "command": "npx",
                            "args": ["-y", "@xweather/mcp-server"],
                            "env": {
                                "XWEATHER_CLIENT_ID": self.config.client_id,
                                "XWEATHER_CLIENT_SECRET": self.config.client_secret,
                            },
                        }
                    }
                )
            else:
                # Remote MCP server via streamable_http
                self._mcp_client = MultiServerMCPClient(
                    {
                        "xweather": {
                            "transport": "streamable_http",
                            "url": self.config.mcp_endpoint,
                            "headers": {
                                "Authorization": f"Bearer {api_key}"
                            },
                        }
                    }
                )

            # Get all tools from the MCP server
            all_tools = await self._mcp_client.get_tools()

            # Filter to only the tools we want
            self._tools = [
                tool for tool in all_tools if tool.name in self.TOOL_NAMES
            ]

            if not self._tools:
                # If no specific tools found, use all available tools
                self._tools = all_tools

            if not self._tools:
                raise MCPConnectionError(
                    "No tools available from MCP server. "
                    "Check your credentials and connection."
                )

        except ValueError:
            raise
        except MCPConnectionError:
            raise
        except Exception as e:
            raise MCPConnectionError(
                f"Failed to connect to Xweather MCP server: {e}"
            ) from e

    def _build_graph(self) -> None:
        """Build the LangGraph workflow for the weather agent."""
        llm = ChatAnthropic(
            model=self.config.model_name,
            temperature=0,
        )
        llm_with_tools = llm.bind_tools(self._tools)

        def call_model(state: AgentState) -> dict:
            """Invoke the LLM with the current state.

            Args:
                state: Current agent state with messages.

            Returns:
                Dictionary with updated messages.
            """
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            """Determine if the agent should continue or end.

            Args:
                state: Current agent state.

            Returns:
                Next node name: 'tools' or 'end'.
            """
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        # Build the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self._tools))

        # Set entry point
        workflow.set_entry_point("agent")

        # Add edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "agent")

        self._graph = workflow
        self._compiled_graph = workflow.compile()

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

    def invoke(self, query: str) -> dict:
        """Process a weather-related query synchronously.

        Args:
            query: Natural language weather query.

        Returns:
            Agent response with weather information.

        Raises:
            MCPConnectionError: If MCP server connection fails during query.
            RuntimeError: If agent has not been initialized.
        """
        self._check_initialized()
        try:
            initial_state = {"messages": [HumanMessage(content=query)]}
            result = self._compiled_graph.invoke(initial_state)
            return result
        except Exception as e:
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                raise MCPConnectionError(
                    f"Lost connection to Xweather MCP server: {e}"
                ) from e
            raise

    async def ainvoke(self, query: str) -> dict:
        """Process a weather-related query asynchronously.

        Args:
            query: Natural language weather query.

        Returns:
            Agent response with weather information.

        Raises:
            MCPConnectionError: If MCP server connection fails during query.
            RuntimeError: If agent has not been initialized.
        """
        self._check_initialized()
        try:
            initial_state = {"messages": [HumanMessage(content=query)]}
            result = await self._compiled_graph.ainvoke(initial_state)
            return result
        except Exception as e:
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                raise MCPConnectionError(
                    f"Lost connection to Xweather MCP server: {e}"
                ) from e
            raise

    async def close(self) -> None:
        """Close the MCP client connection."""
        if self._mcp_client:
            if hasattr(self._mcp_client, "close"):
                await self._mcp_client.close()
            self._mcp_client = None
            self._initialized = False


async def create_weather_agent() -> WeatherAgent:
    """Factory function to create and initialize a WeatherAgent.

    Uses stdio transport with credentials from environment variables.

    Returns:
        Configured and initialized WeatherAgent instance.

    Raises:
        ValueError: If required environment variables are not set.
        MCPConnectionError: If connection to MCP server fails.
    """
    agent = WeatherAgent()
    await agent.initialize()
    return agent


def create_weather_agent_sync() -> WeatherAgent:
    """Synchronous factory function to create and initialize a WeatherAgent.

    Returns:
        Configured and initialized WeatherAgent instance.

    Raises:
        ValueError: If required environment variables are not set.
        MCPConnectionError: If connection to MCP server fails.
    """
    return asyncio.run(create_weather_agent())
