# WeatherWise.ai Implementation Plan

---

## Foundation (Complete)

### Phase 1: Foundation
- [x] Set up Python environment (Poetry) and install `langgraph`, `langchain-anthropic`, `httpx`.
- [x] Create `.env` file for API keys.
- [x] Create `skills.md` to log API patterns.

### Phase 2: The Tools (MCP Edition)
- [x] **Install Adapter:** `pip install langchain-mcp-adapters` (and `npm` if needed for the server).
- [x] **Configure Server:** Identify the execution command for Xweather MCP (likely `npx -y @aerisweather/mcp-server-xweather` or similar).
- [x] **Create Bridge:** Create `tools/mcp_bridge.py` to initialize `MultiServerMCPClient` and load tools.
- [x] **Verification:** Create `verify_mcp.py` to list available tools and run a test query.
- [x] **Calculator:** Create `tools/math_tools.py` with `numexpr`-backed calculator tool.

### Phase 3: The Brain (Router)
- [x] Design the `AgentState` (TypedDict) with `messages` + `intent` fields.
- [x] Implement `analyze_intent` node using `.with_structured_output(RouteDecision)` to classify into 6 Buckets.
- [x] Create `tool_invoker` worker node with intent-specific system prompts + `.bind_tools()`.
- [x] Wire full graph: `analyze_intent → tool_invoker ⇄ tools → END` (all intents route through tool_invoker, including explanation).

### Phase 4a: The Builder (Offline Ingestion Pipeline)

**Strategy:** Multi-Vector Retrieval (Summary-to-Source). We embed summaries but retrieve original parent documents (text, tables, images).
**Dataset:** Real-world only — no synthetic data. Sources documented in `data/DATASET_SOURCES.md`.

- [x] **Collect Golden Dataset (Wave 1):** Download 3 seed documents: CRS Extreme Heat Report, Frontiers Heatwave Paper, 1 GOES satellite snapshot. Place in `data/raw_pdfs/` and `data/satellite_imagery/`.
- [x] **Install Dependencies:** `pip install "unstructured[pdf,image]" langchain-chroma chromadb pillow langchain-openai` + system deps (`brew install poppler tesseract libmagic`).
- [x] **Create Ingestion Script:** `scripts/build_knowledge.py` — the offline pipeline that processes atmospheric PDFs.
- [x] **PDF Parsing:** Use `unstructured` with `hi_res` strategy to partition PDFs into text chunks, tables, and images.
- [x] **Image Summarization Pipeline:** For every extracted chart/graph, call Claude 3.5 Sonnet (Vision) to generate a detailed technical text summary describing axes, trends, anomalies, and meteorological significance.
- [x] **Summary Generation:** For text and table elements, generate concise summaries using Claude that preserve scientific terminology.
- [x] **Embedding & Storage:** Embed all summaries into ChromaDB. Store original parent documents (full text, raw tables, original images) in a local file store (`data/docstore/`) keyed by `doc_id`.
- [x] **Metadata Tagging:** Each vector entry must carry metadata: `source_pdf`, `element_type` (text|table|image), `doc_id` (pointer to parent), and `page_number`.
- [x] **Verification (Wave 1):** Ingest the 3 seed documents and validate retrieval returns correct parent documents.
- [x] **Expand Dataset (Wave 2):** Added NCA5 Ch.2 (91 vectors), AMS/NASA Great Lakes paper (42 vectors), NOAA Tornado Dec 2021 (40 vectors), NOAA Climate Risk CPO003 (156 vectors), 3 satellite images (wildfire smoke, Hurricane Helene VIIRS & PACE). Fixed Claude Vision 5MB limit with `_resize_for_vision()`. Total corpus: 429 vectors, 3/3 verification queries pass.

### Phase 4b: The Runtime (RAG Integration)

**Strategy:** Lightweight retrieval tool wired into the Bucket 5 (Explainability) path of our LangGraph.

- [x] **Create RAG Tool:** `tools/rag_tool.py` — a LangChain `@tool` that queries ChromaDB, retrieves matching summaries, then fetches the original parent documents from the file store.
- [x] **Response Enrichment:** The tool must return both the retrieved text *and* descriptions of any associated visual data (chart summaries) so the synthesizer can cite them.
- [x] **Graph Wiring:** Integrate `rag_tool` into the existing tool set, ensuring the Router dispatches Bucket 5 ("Why" questions) to the `tool_invoker` node with the RAG tool available.
- [x] **Hybrid Context:** For Bucket 5 queries, the synthesizer node must combine RAG-retrieved knowledge with live Xweather data (e.g., current pressure maps alongside historical explanations).
- [x] **End-to-End Test:** "Why is there a heat dome over the Midwest?" — verified: RAG retrieves from CRS Extreme Heat + Frontiers Heatwave + chart/figure analysis; MCP fetches live weather for Chicago & Kansas City; response includes source citations, meteorological terminology, and visual data references. All 5 acceptance criteria pass.

### Phase 4c: Observability (LangSmith)

- [x] **LangSmith Integration:** Added `langsmith` SDK, configured env vars (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT=weatherwise-ai`).
- [x] **Tracing Utility:** Created `src/utils/tracing.py` with config validation and `get_run_config()` helper for run metadata, tags, and custom names.
- [x] **Verification:** Confirmed traces in LangSmith dashboard — all nodes (router, tool_invoker, tools), LLM calls, and tool invocations traced with tags and metadata.

### Phase 5: The Interface & Polish
- [x] **Chainlit UI:** Professional dark-mode chat interface with streaming responses, collapsible tool steps, intent classification display, and LangSmith header link. Launch: `chainlit run app.py`.
- [x] **Premium Web UI:** React + Vite + Tailwind + tsparticles frontend with FastAPI WebSocket backend. Dynamic animated weather background (rain, snow, sun, lightning), glassmorphism landing page with 6 capability cards, aesthetic chat UI with streaming, tool call steps, and intent badges. Launch: `uvicorn server:app --port 8000` + `cd frontend && npm run dev`.

---

## Multi-Agent Refactor + Performance Optimization

**Context:** The current system is a routed single-agent ReAct loop — a Haiku classifier picks one of 6 intent buckets, then a single Sonnet worker handles everything with all tools bound. The `_route_by_intent()` conditional edge is dead code (always returns `"tool_invoker"`). This underutilizes LangGraph (subgraphs, Send(), MemorySaver, interrupt) and leaves significant performance on the table.

**Goal:** Refactor into a true multi-agent system with 3 specialized agents, add streaming, caching, memory, and quality improvements — all incrementally deployable.

**Target Architecture:**
```
BEFORE (single-agent):
  START → analyze_intent → tool_invoker ⇄ tools → END

AFTER (multi-agent):
  START → guardrails → supervisor → ┬─ forecast_agent ⇄ weather_tools ─┐
                                     ├─ research_agent ⇄ rag+weather ──├→ synthesizer → END
                                     └─ analyst_agent  ⇄ calc+weather ──┘
```

### Phase 6: Performance Quick Wins (no architecture changes)

- [x] **6.1 Cache Router LLM** — `src/graph/router.py`: Replace `_get_router_llm()` (lines 63-70) with module-level singleton pattern using lazy initialization. Saves ~100-300ms/query.
- [x] **6.2 Cache ChromaDB Vectorstore** — `src/tools/rag_tool.py`: Add `_vectorstore_cache = None` at module level; modify `_get_vectorstore()` (lines 36-49) to return cached instance. Saves ~200-500ms/RAG query.
- [x] **6.3 Set Worker max_tokens** — `src/graph/nodes/worker.py`: Add `max_tokens=2048` to `ChatAnthropic()` at line 72. ~10-20% faster generation, lower cost.
- [x] **6.4 Bounded Message History** — `server.py` + `app.py`: Add `_trim_history(history, max=20)` helper; apply before `graph_input` in both the WebSocket handler (`server.py:217`) and Chainlit handler (`app.py`). Prevents latency degradation in long sessions.
- [x] **6.5 Anthropic Prompt Caching** — `router.py`, `worker.py`: Add `cache_control={"type": "ephemeral"}` via `additional_kwargs` on `SystemMessage` in both nodes. ~40-60% input token savings for repeated intent paths.
- [x] **6.6 RAG Relevance Score Filtering** — `src/tools/rag_tool.py`: Change `k=5` → `k=8` (fetch more candidates), add score threshold filter (`score < 1.2` cosine distance), take top 5 of filtered results. Better RAG quality, fewer wasted tokens.
- [ ] **Verification:** Run Chainlit, test all 6 buckets, confirm faster response times in LangSmith traces.

### Phase 7: State & Graph Restructure

- [x] **7.1 Expand AgentState** — `src/graph/state.py`: Add new fields: `route_decision` (full routing decision with reasoning), `agent_name` (which agent handles query), `requires_human_approval` / `human_approved` (safety flags), `rag_documents` (retrieved RAG docs for Corrective RAG), `rag_grade` ("relevant"|"irrelevant"|"ambiguous"), `structured_response` (structured output from agents). Move `RouteDecision` from `router.py` to `state.py`.
- [x] **7.2 Create Input Guardrails** — NEW `src/graph/nodes/guardrails.py`: `check_guardrails(state)` → keyword check for weather-related terms (~30 weather keywords). If off-topic: set `intent="rejected"`, inject polite refusal `AIMessage`. If weather-related or follow-up (`len(messages) > 2`): pass through.
- [x] **7.3 Create Supervisor Node** — NEW `src/graph/nodes/supervisor.py`: Replaces `analyze_intent` + dead `_route_by_intent`. `supervisor_node(state)` → Haiku with `RouteDecision` structured output. `route_to_agent(state)` → conditional edge mapping: `real_time/comparative/lifestyle/safety` → `"forecast_agent"`, `explanation` → `"research_agent"`, `math` → `"analyst_agent"`, `rejected` → `END`. Only sends last message to Haiku (not full history).
- [x] **7.4 Restructure builder.py** — `src/graph/builder.py` (rewrite): New signature: `build_graph(mcp_tools, rag_tool, calculator_tool, *, checkpointer=None)`. Wire: `guardrails → supervisor → [forecast_agent | research_agent | analyst_agent] → END`. Conditional edges on both guardrails (`rejected → END`) and supervisor (route to agent). Updated `server.py` + `app.py` to use new signature and `"supervisor"` event name.
- [x] **Verification:** Import test — `build_graph()` compiles without error. Route test — each intent dispatches to the correct agent. Guardrails tested: weather queries pass, off-topic rejected, follow-ups pass.

### Phase 8: Specialized Agents

- [x] **8.1 Forecast Agent** — NEW `src/graph/nodes/forecast_agent.py`: Handles Buckets 1-4 (real_time, comparative, lifestyle, safety). Tools: Xweather MCP only (no calculator, no RAG). Internal ReAct loop: `forecast_worker ⇄ ToolNode(mcp_tools)`. Intent-specific prompts migrated from `worker.py INTENT_PROMPTS`. Module-level LLM singleton. `build_forecast_subgraph(mcp_tools)` → returns compiled subgraph.
- [x] **8.2 Research Agent** — NEW `src/graph/nodes/research_agent.py`: Handles Bucket 5 (explanation). Tools: RAG + Xweather MCP (no calculator). Internal ReAct loop: `research_worker ⇄ ToolNode(mcp_tools + [rag_tool])`. Explanation-specific prompt migrated from `worker.py`. Module-level LLM singleton. `build_research_subgraph(mcp_tools, rag_tool)` → returns compiled subgraph.
- [x] **8.3 Analyst Agent** — NEW `src/graph/nodes/analyst_agent.py`: Handles Bucket 6 (math). Tools: Calculator + Xweather MCP (no RAG). Internal ReAct loop: `analyst_worker ⇄ ToolNode(mcp_tools + [calculator_tool])`. Math-specific prompt migrated from `worker.py`. Module-level LLM singleton. `build_analyst_subgraph(mcp_tools, calculator_tool)` → returns compiled subgraph.
- [x] **8.4 Synthesizer Node** — NEW `src/graph/nodes/synthesizer.py`: Runs after any agent completes, before END. Uses Haiku (fast, cheap) for response formatting. Prompt: format with markdown, preserve citations, add units, use warning tone for safety. Module-level LLM singleton.
- [x] **8.5 Update server.py + app.py**: `build_graph` signature updated in Phase 7. Added `{"type": "agent", "name": "..."}` WebSocket event for agent identification. Event filtering updated for supervisor + agent-specific events.
- [x] **8.6 Deprecate Old Files**: `src/graph/router.py` → superseded by `supervisor.py` (keep for reference). `src/graph/nodes/worker.py` → superseded by 3 specialized agents (keep for reference). Both marked with DEPRECATED docstrings.
- [ ] **Verification:** Test each agent independently:
  1. "What is the temp in San Jose?" → `forecast_agent` → Xweather tools
  2. "Why do heat domes form?" → `research_agent` → RAG + Xweather tools
  3. "Sum temps of Boston and NYC in Kelvin" → `analyst_agent` → Xweather + calculator
  4. "What is the GDP of France?" → guardrails rejection

### Phase 9: Advanced LangGraph Features

- [x] **9.1 Parallel Tool Execution** — `src/graph/nodes/forecast_agent.py`: Updated comparative system prompt to instruct LLM to emit ALL tool calls in a single response. LangGraph's `ToolNode` already runs multiple `tool_calls` concurrently when batched. Saves 1-3s on multi-location queries.
- [x] **9.2 MemorySaver Checkpointing** — `builder.py`, `server.py`, `app.py`: Pass `checkpointer` to `graph.compile(checkpointer=checkpointer)`. Created `MemorySaver()` in server lifespan and app startup, added `config["configurable"] = {"thread_id": session_id}`. Removed manual in-memory session stores — LangGraph manages conversation state via checkpointer. Simplified `graph_input` to `{"messages": [HumanMessage(content=msg)]}`.
- [x] **9.3 Adapt Streaming for Multi-Agent Events** — `server.py` + `app.py`: Updated event filtering for multi-agent graph: supervisor intent detection, synthesizer-only token streaming (via tag filtering), agent identification WebSocket messages. Converted `app.py` from `graph.ainvoke()` to `astream_events` for parity.
- [x] **9.4 Human-in-the-Loop for Safety** — NEW `src/graph/nodes/safety_review.py` + `frontend/src/components/chat/ApprovalPrompt.tsx`: `safety_review_node(state)` checks safety intent + severe alert keywords → `interrupt()` pauses graph. In `builder.py`: inserted between `forecast_agent` and `synthesizer`. In `server.py`: after streaming, checks `graph.aget_state()` for pending tasks (interrupt), sends `approval_required` to client, handles `user_approval` message to resume graph with `Command(resume=True)`. Frontend: `ApprovalPrompt` component with Approve/Dismiss buttons, wired through `MessageBubble` → `MessageList` → `ChatPage`.
- [ ] **Verification:** Comparative query → confirm parallel tool calls in LangSmith. 10+ questions → confirm MemorySaver maintains context. Watch tokens stream in Premium Web UI (`uvicorn server:app --port 8000` + `cd frontend && npm run dev`).

### Phase 10: Quality & Reliability

- [x] **10.1 Corrective RAG** — `src/graph/nodes/research_agent.py` (modified): Added `grade_documents` node using Haiku to grade RAG relevance, `rewrite_query` node to rewrite for better retrieval if irrelevant. Subgraph: `research_worker → tools → grade_documents → [relevant] → research_worker / [irrelevant] → rewrite_query → research_worker`. `RelevanceGrade` Pydantic model with grade + reasoning fields.
- [x] **10.2 MCP Retry with Backoff** — NEW `src/tools/retry_wrapper.py`: `with_retry(tools)` wraps each tool's `ainvoke` with 3 retries, exponential backoff (1s, 2s, 4s). Applied in both `server.py` and `app.py` after loading MCP tools.
- [x] **10.3 Structured Output Models** — NEW `src/graph/models.py`: `ForecastResponse` (location, temperature, conditions, wind, advisory, sources), `ResearchResponse` (explanation, key_concepts, citations), `AnalystResponse` (fetched_values, expression, result). Python 3.9 compatible with `Optional`/`List`/`Dict` typing.
- [x] **10.4 Evaluation Framework** — NEW `tests/eval/test_evaluation.py`: 11 golden routing cases + 4 off-topic + 4 on-topic + 1 follow-up = 15 tests across `TestGuardrails` and `TestRouting` classes. LLM-based `TestRoutingAccuracy` (skipped without API key). LangSmith `run_langsmith_evaluation()` runner with `routing_accuracy` evaluator.
- [x] **10.5 Evaluation Framework Improvements (5 Priorities)** — Industry best-practice upgrades to the LLM-as-Judge eval pipeline:
  - **P2: GPT-4o Judge** — Eliminates self-enhancement bias (Claude judging Claude). Added `_register_openai()` to `src/utils/model_registry.py`. Refactored `llm_judge.py` singleton to registry-based `_get_judge_llm(model_key="gpt-4o")` with `reset_judge_llm()` for test isolation. Added `requires_openai` skip marker in `conftest.py`.
  - **P1: Code-Assertion Layer** — NEW `tests/eval/code_assertions.py`: 5 deterministic checks (numeric units, expected locations, math steps, citations, comparative locations) that run BEFORE LLM judge — fast, cheap, reproducible. `run_assertions()` orchestrator. Word-boundary matching for short aliases (NYC→New York, LA→Los Angeles). Bucket-aware skipping (explanation skips units, lifestyle skips location check).
  - **P3: Binary PASS/FAIL per Criterion** — Added `CriterionResult`, `BinaryVerdict` models and `evaluate_criteria()` to `llm_judge.py`. `_classify_criterion()` parses Must vs Should requirements. Overall pass = all Must pass AND ≥75% Should pass.
  - **P4: RAG-Specific Metrics** — Added 3 new `ScoringDimension` objects to `rubrics.py`: `faithfulness` (weight=1.5), `context_precision` (1.0), `context_recall` (1.0). Updated `BUCKET_APPLICABILITY["explanation"]` and `golden_responses.py` scoring weights. Aligns with RAGAS framework.
  - **P6: Pairwise Comparison** — Added `PairwiseVerdict` model, `pairwise_evaluate()`, and `pairwise_evaluate_debiased()` (swap-debiasing) to `llm_judge.py`. If forward and reverse passes disagree, returns tie with confidence=0.5 and position bias flag. Added `--pairwise` CLI flag to `model_comparison.py`.
  - **Wiring:** `model_comparison.py` extended with `--judge-model` CLI arg, code assertions, binary criteria, and pairwise comparison sections. `test_evaluation.py` gained 6 new test classes: `TestCodeAssertions`, `TestCriterionClassification`, `TestRubrics`, `TestJudgeModelSwitch`, `TestBinaryVerdict`, `TestPairwiseEvaluation`.
- [x] **Verification:** `uv run pytest tests/eval/` — 40/40 tests pass (unit tests). Integration tests require OPENAI_API_KEY.
- [x] **10.6 HTML Evaluation Report** — NEW `tests/eval/report.py`: `generate_html_report(all_results, output_path, pairwise_results)` produces a standalone `.html` file with inline CSS (no external deps). Sections: summary table with color-coded PASS/FAIL badges, expandable dimension breakdowns with score bars, code assertion results, binary criteria detail, and optional pairwise comparison tally. Added `--html [PATH]` flag to `model_comparison.py` (default: `reports/eval_report.html`). Added `reports/` to `.gitignore`.

### Remaining from Original Plan

- [ ] **Full Corpus (Wave 3):** Ingest remaining 15-20 documents (NCA5 chapters, IPCC AR6, NHC reports, Yale CC articles, full satellite set). See `data/DATASET_SOURCES.md` for complete manifest.
- [x] **Create `prompts.py`** — `src/graph/prompts.py`: Centralized all system prompts, keyword lists, and user-facing messages. Extracted 9 prompt constants from 7 agent files: `SUPERVISOR_SYSTEM_PROMPT`, `FORECAST_PROMPTS` (4 intents), `EXPLANATION_PROMPT`, `RAG_GRADER_PROMPT`, `QUERY_REWRITER_PROMPT`, `MATH_PROMPT`, `SYNTHESIZER_PROMPT`, `WEATHER_KEYWORDS`, `REFUSAL_MESSAGE`, `SEVERE_KEYWORDS`, `SAFETY_APPROVAL_MESSAGE`. All agent nodes now import from `prompts.py` instead of defining inline.

---

## Phase Dependencies

```
Phase 6  (Quick Wins)      ← standalone, do first
   ↓
Phase 7  (State + Graph)   ← depends on P6 caching patterns
   ↓
Phase 8  (Agents)          ← depends on P7 state + builder
   ↓
Phase 9  (Advanced)        ← depends on P8 agents
   ↓
Phase 10 (Quality)         ← depends on P8 research agent for Corrective RAG
Wave 3 + prompts.py        ← independent, can be done anytime
```

## File Manifest

### New Files (10)
| File | Phase | Lines (est.) |
|------|-------|-------------|
| `src/graph/nodes/guardrails.py` | 7 | ~40 |
| `src/graph/nodes/supervisor.py` | 7 | ~70 |
| `src/graph/nodes/forecast_agent.py` | 8 | ~80 |
| `src/graph/nodes/research_agent.py` | 8 | ~80 |
| `src/graph/nodes/analyst_agent.py` | 8 | ~70 |
| `src/graph/nodes/synthesizer.py` | 8 | ~45 |
| `src/graph/nodes/safety_review.py` | 9 | ~45 |
| `frontend/src/components/chat/ApprovalPrompt.tsx` | 9 | ~40 |
| `src/graph/models.py` | 10 | ~35 |
| `src/graph/prompts.py` | Remaining | ~155 |
| `src/tools/retry_wrapper.py` | 10 | ~35 |
| `tests/eval/test_evaluation.py` | 10 | ~60 |
| `tests/eval/code_assertions.py` | 10.5 | ~233 |
| `tests/eval/report.py` | 10.6 | ~230 |

### Modified Files (7+)
| File | Phase(s) | Key Changes |
|------|----------|-------------|
| `src/graph/state.py` | 7 | +7 state fields, move `RouteDecision` here |
| `src/graph/builder.py` | 7-9 | Full rewrite: multi-agent graph wiring |
| `src/tools/rag_tool.py` | 6 | Vectorstore caching, score filtering |
| `src/graph/router.py` | 6 | LLM caching (then deprecated by Phase 7) |
| `src/graph/nodes/worker.py` | 6 | max_tokens (then deprecated by Phase 8) |
| `server.py` | 6-9 | History trimming, multi-agent event filtering, MemorySaver, new `build_graph` signature, approval WebSocket messages |
| `app.py` | 6-9 | History trimming, streaming, MemorySaver, new `build_graph` signature |
| `src/utils/model_registry.py` | 10.5 | Added `_register_openai()` with gpt-4o factory |
| `tests/eval/conftest.py` | 10.5 | Added `requires_openai` skip marker |
| `tests/eval/llm_judge.py` | 10.5 | Registry-based judge, binary PASS/FAIL, pairwise comparison |
| `tests/eval/rubrics.py` | 10.5 | 3 RAG dimensions + updated BUCKET_APPLICABILITY |
| `tests/eval/golden_responses.py` | 10.5 | Updated explanation scoring_weights for RAG metrics |
| `tests/eval/model_comparison.py` | 10.5 | Extended ModelResult, CLI args, assertions, binary, pairwise |
| `tests/eval/test_evaluation.py` | 10.5 | 6 new test classes (40 total tests) |

### Deprecated Files (kept for reference)
| File | Replaced By |
|------|-------------|
| `src/graph/router.py` | `src/graph/nodes/supervisor.py` |
| `src/graph/nodes/worker.py` | `forecast_agent.py`, `research_agent.py`, `analyst_agent.py` |

---

## End-to-End Verification (after all phases)

1. Launch Premium Web UI: `uvicorn server:app --port 8000` + `cd frontend && npm run dev`
2. Test each bucket with streaming visible in the React chat UI
3. Confirm LangSmith traces show: `guardrails → supervisor → [correct agent] → synthesizer`
4. Confirm MemorySaver: ask follow-up question referencing previous context
5. Confirm parallel tools: "Is it warmer in NYC or LA?" shows batched tool calls
6. Confirm RAG quality: "Why do heat domes form?" uses Corrective RAG grading
7. Confirm guardrails: "What is 2+2?" gets polite rejection
8. Confirm agent badge: React UI shows which specialist agent handled the query (forecast/research/analyst)
9. Confirm safety approval: severe weather query triggers approval prompt in React UI
10. Run `uv run pytest tests/eval/` — routing accuracy > 95%
