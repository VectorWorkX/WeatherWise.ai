"""State and configuration definitions for WeatherWise.ai agents.

This module provides the core state schema and configuration management
using Pydantic BaseSettings for environment-based configuration.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentState(TypedDict):
    """Conversational state with message history for LangGraph agents.

    Attributes:
        messages: List of conversation messages with automatic merging
            via the add_messages reducer.
        analysis_type: Classification of query complexity - either
            'simple_forecast' for basic queries or 'deep_reanalysis'
            for scientific analysis requests.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    analysis_type: str


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
