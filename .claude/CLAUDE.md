# Project: WeatherWise.ai
# Role: Senior AI Engineer & Architect

## Project Context
WeatherWise.ai is a stateful multi-agent weather system using a Planner-Executor pattern with dynamic parallel execution (fan-out/fan-in). It handles compound queries: real-time weather lookups for multiple cities, vector DB retrieval (RAG), and mathematical operations on weather data.

## Tech Stack
- **Orchestration:** LangGraph v0.2+ (StateGraph with Send-based fan-out)
- **LLM:** Anthropic Claude (via `langchain-anthropic`)
- **Classifier:** Claude Haiku (fast, cheap — for ambiguity detection)
- **Weather Data:** Xweather MCP Server (`https://mcp.api.xweather.com/mcp`) via `langchain-mcp-adapters`
- **RAG:** ChromaDB (local persistent) + OpenAI `text-embedding-3-small` (1536 dims), summary-to-source multi-vector retrieval
- **Document Parsing:** `unstructured` (`hi_res` strategy) for PDF text, tables, images — ingestion only
- **Tools:** Xweather MCP tools (real-time), ChromaDB RAG retrieval, math operations
- **Checkpointing:** AsyncRedisSaver (production, 24h TTL), MemorySaver (dev), AsyncSqliteSaver (local persistent)
- **Prompt Caching:** Anthropic native prompt caching via `cache_control` content blocks (GA, no beta header)
- **Validation:** Pydantic models for structured output
- **Environment:** Python 3.11+, uv for dependency management

## Architecture
- **Pattern:** Planner-Executor with Send-based fan-out/fan-in
- **State:** Single `WeatherState` TypedDict with Annotated reducers
- **Fan-out:** Executor returns `Send` objects to spawn parallel worker nodes
- **Fan-in:** Workers loop back to executor; reducers merge parallel writes
- **Two-phase execution:** Phase 1 = fetch + research (parallel), Phase 2 = calculate (has fetched data)
- **Graph flow:** START -> supervisor -> planner -> executor -> [phase 1 workers] -> executor -> [phase 2 workers] -> executor -> synthesizer -> END
- **Async-native:** All nodes are `async def`, graph uses `ainvoke()`/`astream()`

## Key Files
- `src/weatherwise/state.py` — State definition with reducers
- `src/weatherwise/graph.py` — StateGraph builder and wiring
- `src/weatherwise/nodes/` — Node implementations (supervisor, planner, executor, fetcher, retriever, calculator, synthesizer)
- `src/weatherwise/tools/xweather_mcp.py` — Xweather MCP client config
- `src/weatherwise/tools/vector_search.py` — ChromaDB retrieval (runtime, lightweight)
- `src/weatherwise/tools/` — Tool functions (MCP bridge, RAG retrieval, math)
- `src/weatherwise/main.py` — Entry point with checkpointing
- `scripts/build_knowledge.py` — Offline RAG ingestion pipeline (PDF/image -> ChromaDB)
- `scripts/verify_retrieval.py` — RAG retrieval test harness

## Coding Standards
1. **Type Safety:** All functions must have Python type hints.
2. **Error Handling:** Every LLM/API call wrapped in try/except with meaningful errors.
3. **Modularity:** One file per node, one file per tool category.
4. **No Hardcoding:** Use `.env` for all keys (ANTHROPIC_API_KEY, etc.).
5. **Reducers:** Every state field updated by parallel nodes MUST use Annotated reducers.
6. **Send import:** Always `from langgraph.types import Send` (NOT `langgraph.graph`).
7. **RAG Separation:** Heavy ingestion logic (PDF parsing, image summarization, embedding) lives in `scripts/`. Only lightweight retrieval (query ChromaDB, fetch from docstore) belongs in `src/weatherwise/tools/`. Never import ingestion code at runtime.

## Workflow Rules
7. **Plan Tracking:** After completing any task in `.claude/plan.md`, mark it `[x]` immediately.
8. **No Guessing:** Only factual answers based on verified documentation or code. If unsure, say so.
9. **Suggest Improvements:** If the current approach is suboptimal, flag it and propose alternatives.
10. **Informed Architectural Judgments:** Before recommending any architectural change, you MUST: (a) fully read and understand the existing implementation — every node, every state field, every design decision, (b) understand WHY the current architecture was chosen — ask the user if the intent isn't clear from the code, (c) identify the specific implementation gaps vs. actual architectural flaws — "broken" does not mean "wrong design", (d) never conflate "a different branch works" with "a different branch has better architecture." A working prototype proves execution quality, not design superiority.

## Design Philosophy
"Reliability first. No infinite loops. No race conditions. Degrade gracefully on failure."
