"""State and configuration definitions for WeatherWise.ai agents.

This module provides the core state schema and configuration management
using Pydantic BaseSettings for environment-based configuration.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QueryClassification(BaseModel):
    """Structured output for LLM-based query classification.

    The LLM uses this schema to classify user queries along two dimensions:
    1. Complexity: simple (just format data) vs complex (needs analysis)
    2. Requires tools: whether MCP tools are needed to answer
    """

    query_type: Literal["simple", "complex"] = Field(
        description=(
            "'simple' for straightforward queries that just need data formatted "
            "(e.g., 'What is the weather in Tokyo?', 'Temperature in NYC', 'Hello'); "
            "'complex' for queries requiring interpretation, comparison, reasoning, "
            "or professional insights (e.g., 'Why is it so cold?', 'Compare NYC to LA', "
            "'What's the best day for hiking?')."
        )
    )

    requires_tools: bool = Field(
        description=(
            "True if the query needs weather/MCP tools to answer "
            "(e.g., 'weather in Tokyo' needs tools). "
            "False for greetings, general questions, or queries answerable without tools "
            "(e.g., 'Hello', 'What can you do?', 'Thanks')."
        )
    )

    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen."
    )


class AgentState(TypedDict):
    """Conversational state with message history for LangGraph agents.

    Attributes:
        messages: List of conversation messages with automatic merging
            via the add_messages reducer.
        query_type: 'simple' for data formatting, 'complex' for analysis.
        requires_tools: Whether the query needs MCP tool calls.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    query_type: Literal["simple", "complex"]
    requires_tools: bool


class WeatherConfig(BaseSettings):
    """Weather agent configuration loaded from environment variables.

    Automatically reads from .env file and environment variables.
    All Xweather and Anthropic credentials are required.

    Attributes:
        xweather_client_id: Xweather API client ID.
        xweather_client_secret: Xweather API client secret.
        anthropic_api_key: Anthropic API key for Claude models.
        model_name: Primary Claude model for agent reasoning.
        synthesizer_model: Claude model for report synthesis.
        mcp_endpoint: Xweather MCP server endpoint URL.
        transport: MCP transport type (stdio or streamable_http).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Xweather credentials
    xweather_client_id: str = Field(
        ...,
        description="Xweather API client ID",
    )
    xweather_client_secret: str = Field(
        ...,
        description="Xweather API client secret",
    )

    # Anthropic credentials
    anthropic_api_key: str = Field(
        ...,
        description="Anthropic API key for Claude models",
    )

    # Model configuration
    model_name: str = Field(
        default="claude-sonnet-4-20250514",
        description="Primary Claude model for agent reasoning",
    )
    synthesizer_model: str = Field(
        default="claude-3-7-sonnet-latest",
        description="Claude model for professional report synthesis",
    )

    # MCP configuration
    mcp_endpoint: str = Field(
        default="https://mcp.api.xweather.com/mcp",
        description="Xweather MCP server endpoint URL",
    )
    transport: Literal["stdio", "streamable_http"] = Field(
        default="streamable_http",
        description="MCP transport type",
    )

    @property
    def api_key(self) -> str:
        """Generate the combined API key for MCP authentication.

        Returns:
            Combined client_id and client_secret for Bearer auth.
        """
        return f"{self.xweather_client_id}_{self.xweather_client_secret}"
