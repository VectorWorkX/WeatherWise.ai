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
- **Document Parsing:** `unstructured` (`hi_res` strategy) for PDF partitioning into text, tables, and images
- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dimensions) via `langchain-openai`
- **Vector Store:** ChromaDB (local persistent at `data/chroma_db/`) for summary embeddings + local file store (`data/docstore/`) for parent documents
- **Chunking:** Element-based semantic grouping via `unstructured` `chunk_by_title` (NOT token-based splitting)
- **RAG Pattern:** Multi-Vector Retrieval (Summary-to-Source) — embed summaries, retrieve originals
- **Image Grounding:** Claude 3.5 Sonnet (Vision) generates text summaries of weather charts/graphs for embedding (no CLIP/multimodal embeddings)
- **Data Source (Math):** Python REPL or `numexpr` for calculation logic
- **Validation:** Pydantic models for structured output
- **Environment:** Python 3.10+, Poetry for dependency management

## Coding Standards
1. **Type Safety:** All functions must have Python type hints.
2. **Error Handling:** Every API call must be wrapped in `try/except` blocks with meaningful error messages.
3. **Modularity:** One file per logical component (e.g., `tools/weather.py`, `graph/router.py`).
4. **Documentation:** Add docstrings to complex logic (especially the Router).
5. **No Hardcoding:** Use `.env` for all keys (CLAUDE_API_KEY, XWEATHER_ID, etc.).
6. **RAG Separation:** Heavy ingestion/indexing logic (PDF parsing, image summarization, embedding) must live in `scripts/`. Only lightweight retrieval logic (query ChromaDB, fetch from docstore) belongs in `tools/`. Never import ingestion code at runtime.

## Workflow Rules
7. **Plan Tracking:** After completing any task that corresponds to an item in `.claude/plan.md`, immediately mark it as `[x]` in the plan file. Do not batch updates — check off each item as soon as the work is done and verified.
8. **Golden Test Cases:** When adding a new golden case in `tests/eval/golden_responses.py`, always populate the `locations` field with the city/location names relevant to the query and response. The `code_assertions.py` location checker derives its known locations list automatically from this field — no separate file to update.

## Design Philosophy
"Reliability first. If an API fails, degrade gracefully. If a user asks 'Why', provide the science, not just the statistic."