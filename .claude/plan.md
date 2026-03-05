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
- [x] Replace mock fetcher with MCP-backed ReAct agent (`nodes/fetcher.py`)
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
