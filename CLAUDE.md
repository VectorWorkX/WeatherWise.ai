# WeatherWise.ai

An intelligent weather assistant powered by LangGraph orchestration and Xweather MCP server for real-time weather data.

## Tech Stack

- **Python**: 3.11
- **Orchestration**: LangGraph
- **Weather Data**: Xweather MCP server
- **Data Validation**: Pydantic

## Project Structure

```
WeatherWise.ai/
├── src/
│   ├── agents/        # LangGraph agent definitions
│   ├── models/        # Pydantic data models
│   ├── tools/         # MCP tool integrations
│   └── utils/         # Helper utilities
├── tests/             # Test suite
├── CLAUDE.md
└── README.md
```

## Coding Style

### Pydantic Models
- Use strict Pydantic models for all data structures
- Define explicit field types and validators
- Use `Field()` with descriptions for documentation

```python
from pydantic import BaseModel, Field

class WeatherRequest(BaseModel):
    location: str = Field(..., description="City name or coordinates")
    units: str = Field(default="metric", pattern="^(metric|imperial)$")
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

### Setup
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables
```bash
export XWEATHER_CLIENT_ID="your_client_id"
export XWEATHER_CLIENT_SECRET="your_client_secret"
```

### Run
```bash
python -m src.main
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_agents.py

# Run with verbose output
pytest -v
```

## MCP Server Configuration

The Xweather MCP server provides weather data tools. Configure in your MCP settings:

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

- `pytest` - Run test suite
- `ruff check src/` - Lint code
- `ruff format src/` - Format code
- `mypy src/` - Type checking
