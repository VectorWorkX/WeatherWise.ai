# Plan: WeatherWise.ai Multi-Agent Architecture

## Phase 1: Foundation
- [x] Create project structure and .claude config files
- [x] Define WeatherState with correct reducers (`state.py`)
- [x] Define TaskPayload for Send-based worker input (`state.py`)
- [x] Create mock tools: weather API, vector search, math (`tools/`)
- [x] Set up pyproject.toml with dependencies

## Phase 2: Node Implementation
- [x] Implement supervisor node (ambiguity classifier)
- [x] Implement planner node (query decomposition)
- [x] Implement executor node (fan-out router)
- [x] Implement fetcher_agent node (weather API worker)
- [x] Implement retriever_agent node (RAG worker)
- [x] Implement calculator_agent node (math worker)
- [x] Implement synthesizer node (response writer)

## Phase 3: Graph Wiring
- [x] Build StateGraph with all nodes registered
- [x] Wire conditional edges: supervisor routing (ambiguous vs clear)
- [x] Wire Send-based fan-out from executor
- [x] Wire fan-in edges from workers to synthesizer
- [x] Add MemorySaver checkpointing

## Phase 3.5: Xweather MCP Integration
- [x] Create Xweather MCP client config (`tools/xweather_mcp.py`)
- [x] Replace mock fetcher with MCP-backed manual tool-calling loop (`nodes/fetcher.py`)
- [x] Add `langchain-mcp-adapters` dependency
- [x] Update env config with Xweather credentials
- [x] Remove mock weather API (replaced by Xweather MCP)

## Phase 3.6: Async + Two-Phase Executor
- [x] Convert all nodes to async (production-safe, no `asyncio.run()` hacks)
- [x] Implement two-phase executor: fetch/research first, then calculate with data
- [x] Rewire graph: workers -> executor loop (not workers -> synthesizer)
- [x] Split TaskPayload into FetchPayload + CalcPayload (calculator gets fetched data)
- [x] Update main.py to use `ainvoke()` via `asyncio.run()`

## Phase 3.7: RAG Pipeline (ChromaDB)
- [x] Port ingestion script (`scripts/build_knowledge.py`) — PDF/image -> summaries -> ChromaDB
- [x] Replace mock vector_search.py with real ChromaDB retrieval tool
- [x] Port verification script (`scripts/verify_retrieval.py`)
- [x] Add dependencies: `langchain-chroma`, `langchain-openai`, `unstructured[pdf]`, `Pillow`
- [x] Add `OPENAI_API_KEY` to env config

## Phase 4: Entry Point & Testing
- [x] Create main.py entry point with CLI interaction loop
- [ ] Test with simple single-city query
- [ ] Test with compound multi-city + RAG query
- [ ] Test ambiguity detection flow
- [ ] Verify parallel fan-out/fan-in produces correct merged state

## Phase 5: Production Hardening
- [x] P0.1: Error handling + timeouts on all nodes (try/except, timeout param)
- [x] P0.2: Recursion limit (25) + executor max-iteration safeguard
- [x] P0.3: Fix eval() security — replaced with AST-based safe evaluator
- [x] P0.4: Fix malformed pyproject.toml (merged optional-dependencies)
- [x] P1.5: Structured logging (structlog, JSON/console modes)
- [x] P1.6: Retry wrapper (tenacity, exponential backoff for 429/503/timeout)
- [x] P1.7: Pydantic structured output for planner (PlanTask/PlanOutput models)
- [x] P1.8: Unit + integration tests (math_ops, executor, nodes, graph)
- [x] P2.9: Production checkpointer option (sqlite via --checkpointer flag)
- [x] P2.10: Streaming support (astream_events via --stream flag)
- [x] P2.11: LangSmith tracing (auto-enabled when LANGCHAIN_API_KEY set)
- [x] P3.12: FastAPI/WebSocket server (POST /query, WS /ws)
- [x] P3.13: Graceful degradation (all nodes return partial results on failure)

## Phase 6: Redis Persistence & Prompt Caching

### 6.1 — Add `langgraph-checkpoint-redis` dependency
- [x] Add `langgraph-checkpoint-redis>=0.3.0` to `pyproject.toml` dependencies
- [x] Run `uv sync` to install the package and its transitive deps (`redis`, `redisvl`)

### 6.2 — Refactor `graph.py` to accept a pre-created checkpointer
**Why:** `AsyncRedisSaver.__init__` calls `asyncio.get_running_loop()` (line 102 of aio.py), so it MUST be created inside an async context. Our current `build_graph()` is a sync function called at module level. We cannot instantiate `AsyncRedisSaver` inside it.
**What:**
- [x] Add a `checkpointer_instance` parameter to `build_graph()`:
  ```python
  def build_graph(
      checkpointer: Literal["memory", "sqlite", "none"] | bool = True,
      checkpointer_instance=None,  # pre-created async saver (e.g., AsyncRedisSaver)
  ) -> StateGraph:
  ```
  If `checkpointer_instance` is provided, use it directly; otherwise fall back to `_get_checkpointer()` for memory/sqlite/none (existing behavior, untouched).
- [x] Keep `_get_checkpointer()` as-is — it only handles memory/sqlite/none. No "redis" case inside it.

### 6.3 — Update `server.py` to create `AsyncRedisSaver` in async lifespan
**Why:** `server.py` currently creates the graph at module level (`_graph = build_graph(...)`). For Redis, we need an async context. FastAPI's lifespan is the correct place.
**What:**
- [x] Add a FastAPI `lifespan` async context manager:
  ```python
  from contextlib import asynccontextmanager

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      global _graph
      mode = os.environ.get("CHECKPOINTER_MODE", "memory")
      if mode == "redis":
          from langgraph.checkpoint.redis.aio import AsyncRedisSaver
          redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
          async with AsyncRedisSaver.from_conn_string(
              redis_url,
              ttl={"default_ttl": 1440, "refresh_on_read": True},
          ) as saver:
              _graph = build_graph(checkpointer_instance=saver)
              yield
      else:
          _graph = build_graph(checkpointer=mode)
          yield
  ```
  - `from_conn_string` is an `@asynccontextmanager` on `AsyncRedisSaver` — it creates the instance, calls `asetup()` (creates Redis search indices), and yields the ready saver.
  - `ttl={"default_ttl": 1440}` = 24 hours in minutes.
  - `refresh_on_read: True` resets the 24h TTL each time a session is accessed, so active conversations never expire mid-use.
- [x] Pass `lifespan=lifespan` to the `FastAPI()` constructor.
- [x] Move the current module-level `_graph = build_graph(...)` inside the `else` branch of the lifespan (for non-redis modes).
- [x] Initialize `_graph = None` at module level (endpoints will use it after lifespan sets it).

### 6.4 — Update `main.py` to support Redis checkpointer
**Why:** `main.py` runs `run_interactive()` which is already `async def`, so creating `AsyncRedisSaver` is straightforward.
**What:**
- [x] Add `"redis"` to the `--checkpointer` CLI choices.
- [x] Inside `run_interactive()`, if `checkpointer == "redis"`:
  ```python
  if checkpointer == "redis":
      from langgraph.checkpoint.redis.aio import AsyncRedisSaver
      redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
      async with AsyncRedisSaver.from_conn_string(
          redis_url,
          ttl={"default_ttl": 1440, "refresh_on_read": True},
      ) as saver:
          app = build_graph(checkpointer_instance=saver)
          # run interactive loop inside this context
  else:
      app = build_graph(checkpointer=checkpointer)
  ```
- [x] Refactor the interactive while-loop into a helper to avoid duplicating the loop code between the two branches.

### 6.5 — Add `REDIS_URL` environment variable
- [x] Add `REDIS_URL` to `.env`: `REDIS_URL=redis://localhost:6379`
- [x] Document in CLAUDE.md under Tech Stack

### 6.6 — Prompt caching: Synthesizer (`nodes/synthesizer.py`)
**Why:** The synthesizer uses Claude Sonnet 4 (most expensive model). Its system prompt (~300 tokens) is identical on every call. No beta header needed — prompt caching is GA with Anthropic SDK v0.77.0 and langchain-anthropic v1.3.1 (verified: `anthropic.types.CacheControlEphemeralParam` exists).
**What:**
- [x] Change the `SystemMessage` from a plain string to content block list with `cache_control`:
  ```python
  SystemMessage(content=[
      {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
  ])
  ```
- [x] Change the `HumanMessage` (data block) to content block format with `cache_control`:
  ```python
  HumanMessage(content=[
      {"type": "text", "text": f"User query: {user_query}\n\nGathered data:\n{data_block}",
       "cache_control": {"type": "ephemeral"}}
  ])
  ```
- [x] Verify `_extract_text()` in `server.py` handles list content blocks — it already does (returns `block.get("text", "")` for dict blocks). No server.py change needed.

### 6.7 — Prompt caching: Planner (`nodes/planner.py`)
**Why:** The planner also uses Claude Sonnet 4. Its system prompt (~400 tokens with examples) is identical on every call.
**What:**
- [x] Change the `SystemMessage` to content block format with `cache_control`:
  ```python
  SystemMessage(content=[
      {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
  ])
  ```
- [x] The planner's second argument is the user's message history (varies every call) — no `cache_control` on it.

### 6.8 — Prompt caching: Supervisor (`nodes/supervisor.py`)
**Why:** The supervisor is called on every single user message. It uses Claude Haiku 4.5 (cheap per-call), but the system prompt is ~1,200 tokens. Caching it is a one-line change with zero downside.
**What:**
- [x] Change the `SystemMessage` to content block format:
  ```python
  SystemMessage(content=[
      {"type": "text", "text": f"{date_ctx}\n\n{SYSTEM_PROMPT}",
       "cache_control": {"type": "ephemeral"}}
  ])
  ```
  Note: `date_ctx` changes every minute, but Anthropic's prompt cache TTL is 5 minutes, so within a burst of messages the prefix is stable and will hit.

### 6.9 — Sort weather keys for cache consistency (`nodes/synthesizer.py`)
**Why:** Anthropic prompt caching is prefix-based. If two requests produce the same data but in different dict iteration order, the cache misses. Sorting guarantees identical prefixes for identical data.
**What:**
- [x] Sort `fetched_weather` keys alphabetically when building the data block:
  ```python
  for loc in sorted(weather.keys()):
      lines.append(f"  {loc}: {weather[loc]}")
  ```
  One-line change: `sorted(weather.keys())` instead of `weather.items()`.

## Phase 7: Pinecone Migration & Speed Optimization
- [x] Migrate vector DB from ChromaDB to Pinecone serverless (free tier)
- [x] Write migration script (`scripts/migrate_to_pinecone.py`)
- [x] Update `vector_search.py` for Pinecone (cosine similarity, threshold 0.3)
- [x] Store parent content in vector metadata (eliminate docstore dependency)
- [x] Update `build_knowledge.py` for Pinecone upserts
- [x] Update `verify_retrieval.py` with Pinecone stats function
- [x] Remove `_get_index_stats()` pre-check (save ~200-400ms/query)
- [x] ~Switch planner LLM from Sonnet to Haiku~ (reverted — planner currently uses Sonnet: `claude-sonnet-4-20250514`)

## Phase 8: GCP Deployment
- [x] Create multi-stage Dockerfile (Node frontend + Python backend)
- [x] Lock CORS to production domains
- [x] Create Terraform infrastructure (Cloud Run, Secret Manager, Artifact Registry)
- [x] Build and push Docker image via Cloud Build
- [x] Deploy to Cloud Run (`https://weatherwise-228097686588.us-central1.run.app`)
- [x] Verify health endpoint and frontend serving
- [ ] Cloudflare DNS setup for weather-wise.ai
- [ ] Rotate all API keys (exposed in git history)
- [x] CI/CD: Cloud Build trigger `deploy-on-push-to-main` (runs cloudbuild.yaml on push to main)

### Files modified (summary)
| File | Change |
|---|---|
| `pyproject.toml` | Add `langgraph-checkpoint-redis` dep |
| `src/weatherwise/graph.py` | Add `checkpointer_instance` param to `build_graph()` |
| `src/weatherwise/server.py` | FastAPI lifespan for async Redis saver creation |
| `src/weatherwise/main.py` | Add `"redis"` CLI option, async saver in `run_interactive()` |
| `src/weatherwise/nodes/synthesizer.py` | Prompt caching (system + data blocks) + sort weather keys |
| `src/weatherwise/nodes/planner.py` | Prompt caching (system block) |
| `src/weatherwise/nodes/supervisor.py` | Prompt caching (system block) |
| `.env` | Add `REDIS_URL` |
