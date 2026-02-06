# Project Skills & Knowledge

## Xweather API Patterns
*To be populated once we make our first successful call. Keep notes here on JSON structure.*
- **Example Call:** `GET /v1/current.json?q=San+Jose&key=...`

## LangGraph Patterns
- **State Updates:** Remember that LangGraph state is immutable; nodes must return *updates* to the state, not the full state object.
- **Tool Calling:** When the Router selects a tool, the output must be a standard `ToolMessage` to be compatible with LangChain's pre-built nodes.

## Math Operations
- **Data Consistency:** When performing math on temperatures, always ensure both sources are in the same unit (C or F) before calculating.