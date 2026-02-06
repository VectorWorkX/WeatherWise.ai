# Project: WeatherWise.ai
# Role: Senior AI Engineer & Architect

## Project Context
We are building an agentic weather system that acts as an "Expert Analyst," not just a data fetcher. The system must handle simple lookups, complex comparisons, deep atmospheric explanations (RAG), and mathematical operations on weather data.

## Tech Stack
- **Orchestration:** LangGraph
- **Tools:** Xweather MCP Server (via `langchain-mcp-adapters`)
- **Integration:** `MultiServerMCPClient` to bridge MCP tools to LangChain tools.
- **LLM:** Anthropic Claude 3.5 Sonnet (via API)
- **Data Source (Real-Time):** Xweather API (via MCP or direct HTTP)
- **Data Source (Knowledge):** Vector Store (ChromaDB) for atmospheric papers/historical data
- **Data Source (Math):** Python REPL or `numexpr` for calculation logic
- **Validation:** Pydantic models for structured output
- **Environment:** Python 3.10+, Poetry for dependency management

## Coding Standards
1. **Type Safety:** All functions must have Python type hints.
2. **Error Handling:** Every API call must be wrapped in `try/except` blocks with meaningful error messages.
3. **Modularity:** One file per logical component (e.g., `tools/weather.py`, `graph/router.py`).
4. **Documentation:** Add docstrings to complex logic (especially the Router).
5. **No Hardcoding:** Use `.env` for all keys (CLAUDE_API_KEY, XWEATHER_ID, etc.).

## Design Philosophy
"Reliability first. If an API fails, degrade gracefully. If a user asks 'Why', provide the science, not just the statistic."