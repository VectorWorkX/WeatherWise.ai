# Project: WeatherWise.ai
# Role: Senior AI Engineer & Architect

## Project Context
WeatherWise.ai is a stateful multi-agent weather system using a Planner-Executor pattern with dynamic parallel execution (fan-out/fan-in). It handles compound queries: real-time weather lookups for multiple cities, vector DB retrieval (RAG), and mathematical operations on weather data.

## Tech Stack
- **Orchestration:** LangGraph v0.2+ (StateGraph with Send-based fan-out)
- **LLM:** Anthropic Claude (via `langchain-anthropic`). Sonnet for all agents (supervisor, fetcher, planner, calculator, synthesizer, visualizer).
- **Weather Data:** Xweather MCP Server via `langchain-mcp-adapters`, Bearer auth header, tool filtering (`exclude_tags=tropical,lightning`)
- **RAG:** Pinecone (serverless, free tier) + OpenAI `text-embedding-3-small` (1536 dims), summary-to-source multi-vector retrieval, parent content in vector metadata
- **Document Parsing:** `unstructured` (`hi_res` strategy) for PDF text, tables, images — ingestion only
- **Tools:** Xweather MCP tools (real-time), Pinecone RAG retrieval, sandboxed Python calculator (math + statistics), ECharts visualization (LLM generates complete ECharts JSON directly)
- **Checkpointing:** MemorySaver (default), AsyncSqliteSaver (local persistent), AsyncRedisSaver (implemented, toggleable via `CHECKPOINTER_MODE` env var). Redis not active in production deployment — GCP Memorystore lacks RediSearch module needed by langgraph-checkpoint-redis v0.3+.
- **Prompt Caching:** Anthropic native prompt caching via `cache_control` content blocks (GA, no beta header)
- **Validation:** Pydantic models for structured output
- **Environment:** Python 3.11+, uv for dependency management

## Architecture
- **Pattern:** Planner-Executor with Send-based fan-out/fan-in
- **State:** Single `WeatherState` TypedDict with Annotated reducers
- **Fan-out:** Executor returns `Send` objects to spawn parallel worker nodes
- **Fan-in:** Workers loop back to executor; reducers merge parallel writes
- **Dependency-aware execution:** Planner assigns `depends_on` indices to each task. Executor dispatches tasks whose dependencies are satisfied, producing dynamic parallel waves. No hardcoded phase ordering.
- **Graph flow:** START -> supervisor -> planner -> executor -> [ready workers] -> executor -> [next wave] -> ... -> executor -> synthesizer -> END
- **Visualization:** Visualizer agent runs as a plan task (op="visualize") with dependency-aware execution. Single structured output call: LLM reads JSON data directly and generates complete ECharts option JSON via Pydantic VisualReport model. No ReAct loop, no code execution for chart generation. Frontend renders echarts_option with theme="dark".
- **Data Pipeline:** Fetcher stores both `raw_data` (structured JSON from MCP) and `raw_response` (LLM text). Calculator/visualizer use `raw_data` (clean JSON); synthesizer uses `raw_response` (human-readable text).
- **Session Persistence:** Frontend stores session_id in sessionStorage, sends with every message. Backend resumes checkpoint on reconnect.
- **Thinking Timeline:** ChatGPT-style collapsible disclosure showing supervisor reasoning, planner summary, task progress. Dynamic header (Thinking → Searching → Analyzing → Writing).
- **Follow-up Suggestions:** Synthesizer generates `>>>` prefixed suggestions, server extracts and sends as clickable chips. Suggestions stripped from checkpoint to keep conversation history clean.
- **Async-native:** All nodes are `async def`, graph uses `ainvoke()`/`astream()`

## Key Files
- `src/weatherwise/state.py` — State definition with reducers
- `src/weatherwise/graph.py` — StateGraph builder and wiring
- `src/weatherwise/nodes/` — Node implementations (supervisor, planner, executor, fetcher, retriever, calculator, visualizer, synthesizer)
- `src/weatherwise/tools/xweather_mcp.py` — Xweather MCP client config
- `src/weatherwise/tools/vector_search.py` — Pinecone retrieval (runtime, lightweight)
- `src/weatherwise/tools/math_ops.py` — Calculator tools: `compute_expression` (arithmetic) + `run_calculation` (sandboxed Python with math/statistics)
- `src/weatherwise/nodes/visualizer.py` — Visualizer agent: generates complete ECharts option JSON directly (generative approach, no templates)
- `src/weatherwise/server.py` — FastAPI + WebSocket server (REST, streaming, session management)
- `src/weatherwise/utils.py` — Shared utilities (chat history trimming)
- `src/weatherwise/main.py` — CLI entry point with checkpointing
- `scripts/build_knowledge.py` — Offline RAG ingestion pipeline (PDF/image -> Pinecone)
- `scripts/verify_retrieval.py` — RAG retrieval test harness

## Coding Standards
1. **Type Safety:** All functions must have Python type hints.
2. **Error Handling:** Every LLM/API call wrapped in try/except with meaningful errors.
3. **Modularity:** One file per node, one file per tool category.
4. **No Hardcoding:** Use `.env` for all keys (ANTHROPIC_API_KEY, etc.).
5. **Reducers:** Every state field updated by parallel nodes MUST use Annotated reducers.
6. **Send import:** Always `from langgraph.types import Send` (NOT `langgraph.graph`).
7. **RAG Separation:** Heavy ingestion logic (PDF parsing, image summarization, embedding) lives in `scripts/`. Only lightweight retrieval (query Pinecone) belongs in `src/weatherwise/tools/`. Never import ingestion code at runtime.

## Workflow Rules
7. **Plan Tracking:** After completing any task in `.claude/plan.md`, mark it `[x]` immediately.
8. **No Guessing:** Only factual answers based on verified documentation or code. If unsure, say so.
9. **Suggest Improvements:** If the current approach is suboptimal, flag it and propose alternatives.
10. **Informed Architectural Judgments:** Before recommending any architectural change, you MUST: (a) fully read and understand the existing implementation — every node, every state field, every design decision, (b) understand WHY the current architecture was chosen — ask the user if the intent isn't clear from the code, (c) identify the specific implementation gaps vs. actual architectural flaws — "broken" does not mean "wrong design", (d) never conflate "a different branch works" with "a different branch has better architecture." A working prototype proves execution quality, not design superiority.

## Deployment
- **Platform:** GCP Cloud Run (us-central1), project `weatherwise-486608`
- **Infrastructure:** Terraform-managed (`infra/` directory)
- **Image:** `us-central1-docker.pkg.dev/weatherwise-486608/weatherwise/app:latest`
- **Build:** Cloud Build (`gcloud builds submit`)
- **URLs:** `https://weatherwise-228097686588.us-central1.run.app` (Cloud Run), `https://weather-wise.ai` (custom domain via Cloudflare)
- **Frontend:** Served from same container (`SERVE_FRONTEND=1`, `FRONTEND_DIST_DIR=/app/frontend/dist`)
- **Secrets:** GCP Secret Manager (6 keys), accessed via Cloud Run SA IAM
- **Cost:** ~$3-8/mo (usage-based, $0 when idle)

## Design Philosophy
"Reliability first. No infinite loops. No race conditions. Degrade gracefully on failure."
