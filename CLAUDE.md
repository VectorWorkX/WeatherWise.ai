# WeatherWise.ai

An intelligent weather assistant powered by LangGraph orchestration and Xweather MCP server for real-time weather data.

## Tech Stack

- **Python**: 3.11+
- **Package Manager**: uv
- **Orchestration**: LangGraph with MemorySaver persistence
- **LLM**: Claude (via langchain-anthropic)
- **Weather Data**: Xweather MCP server
- **Configuration**: pydantic-settings (BaseSettings)
- **Data Validation**: Pydantic v2

## Project Structure

```
WeatherWise.ai/
├── src/
│   ├── agents/
│   │   ├── __init__.py       # Public exports
│   │   ├── state.py          # AgentState + WeatherConfig
│   │   └── weather_agent.py  # Professional ReAct Loop agent
│   ├── models/               # Pydantic data models
│   ├── tools/                # MCP tool integrations
│   └── utils/                # Helper utilities
├── tests/                    # Test suite
├── pyproject.toml            # uv dependencies
├── .env                      # Environment variables
├── CLAUDE.md
└── README.md
```

## LangGraph Architecture

### Professional ReAct Loop

The weather agent implements a three-node graph:

```
┌──────────────┐     ┌─────────┐     ┌──────────────┐
│   analyzer   │────▶│  tools  │────▶│  synthesizer │───▶ END
│ (Claude 4)   │     │(ToolNode)│    │ (Claude 3.7) │
└──────────────┘     └─────────┘     └──────────────┘
       │                                     ▲
       └──────── (no tools needed) ──────────┘
```

### Node Functions

1. **analyzer_node**: Classifies query as `simple_forecast` or `deep_reanalysis`, initiates tool calls
2. **tools_node**: ToolNode executes Xweather MCP tools
3. **synthesizer_node**: Claude 3.7 creates professional meteorological reports

### Node Function Conventions

- Each node function takes `AgentState` and returns a `dict` with state updates
- Use `Literal` return types for routing functions
- System prompts are injected per-node, not globally

```python
def analyzer_node(state: AgentState) -> dict:
    """Analyze query and determine response strategy."""
    messages = state["messages"]
    system_msg = SystemMessage(content=ANALYZER_SYSTEM_PROMPT)
    messages_with_system = [system_msg] + list(messages)
    response = llm_with_tools.invoke(messages_with_system)
    return {"messages": [response], "analysis_type": "simple_forecast"}
```

### State Definition

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    analysis_type: str  # "simple_forecast" or "deep_reanalysis"
```

### Persistence

Multi-turn conversations use `MemorySaver` with `thread_id`:

```python
result = await agent.ainvoke("Weather in Paris?", thread_id="session-1")
result = await agent.ainvoke("Compare to yesterday?", thread_id="session-1")
```

## Coding Style

### Pydantic Models
- Use strict Pydantic models for all data structures
- Define explicit field types and validators
- Use `Field()` with descriptions for documentation
- Use `pydantic-settings` BaseSettings for environment config

```python
from pydantic import Field
from pydantic_settings import BaseSettings

class WeatherConfig(BaseSettings):
    xweather_client_id: str = Field(..., description="Xweather API client ID")
    model_name: str = Field(default="claude-sonnet-4-20250514")

    model_config = {"env_file": ".env", "extra": "ignore"}
```

### Docstrings
- All public functions and classes must have docstrings
- Use Google-style docstring format
- Include Args, Returns, and Raises sections

```python
def get_forecast(location: str, days: int = 7) -> WeatherForecast:
    """Retrieve weather forecast for a location.

    Args:
        location: City name or lat,lon coordinates.
        days: Number of forecast days (1-14).

    Returns:
        WeatherForecast object with daily predictions.

    Raises:
        LocationNotFoundError: If location cannot be resolved.
    """
```

### Modular Files
- One primary class or related group per file
- Keep files under 300 lines
- Use clear, descriptive filenames (snake_case)

## Running the Project

### Setup with uv

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Or install specific groups
uv sync --group dev
```

### Environment Variables

Create a `.env` file:

```bash
XWEATHER_CLIENT_ID=your_client_id
XWEATHER_CLIENT_SECRET=your_client_secret
ANTHROPIC_API_KEY=your_anthropic_key
```

### Run

```bash
# Run the CLI
uv run python -m src.main

# Or with explicit Python
uv run python -c "from src.main import main; main()"
```

### CLI Commands

Once running, use these commands:
- `/help` - Show available commands
- `/new` - Start a new conversation session
- `/history` - Show conversation history
- `/quit` - Exit (also `/exit`, `/q`)

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_agents.py

# Run with verbose output
uv run pytest -v
```

## Linting & Formatting

```bash
# Lint code
uv run ruff check src/

# Auto-fix lint issues
uv run ruff check src/ --fix

# Format code
uv run ruff format src/

# Type checking
uv run mypy src/
```

## Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add dev dependency
uv add --group dev package-name

# Remove dependency
uv remove package-name
```

## MCP Server Configuration

The Xweather MCP server provides weather data tools. The agent uses `streamable_http` transport by default:

```python
# Remote MCP (default)
config = WeatherConfig(transport="streamable_http")

# Local MCP via npx
config = WeatherConfig(transport="stdio")
```

For local MCP, configure in your MCP settings:

```json
{
  "mcpServers": {
    "xweather": {
      "command": "npx",
      "args": ["-y", "@xweather/mcp-server"],
      "env": {
        "XWEATHER_CLIENT_ID": "your_client_id",
        "XWEATHER_CLIENT_SECRET": "your_client_secret"
      }
    }
  }
}
```

## Common Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install/sync dependencies |
| `uv run python -m src.main` | Run CLI |
| `uv run pytest` | Run tests |
| `uv run ruff check src/` | Lint code |
| `uv run ruff format src/` | Format code |
| `uv run mypy src/` | Type check |
| `uv add <pkg>` | Add dependency |
