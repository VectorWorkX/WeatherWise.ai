# WeatherWise.ai Implementation Plan

## Phase 1: Foundation
- [x] Set up Python environment (Poetry) and install `langgraph`, `langchain-anthropic`, `httpx`.
- [x] Create `.env` file for API keys.
- [x] Create `skills.md` to log API patterns.

## Phase 2: The Tools (MCP Edition)
- [x] **Install Adapter:** `pip install langchain-mcp-adapters` (and `npm` if needed for the server).
- [x] **Configure Server:** Identify the execution command for Xweather MCP (likely `npx -y @aerisweather/mcp-server-xweather` or similar).
- [x] **Create Bridge:** Create `tools/mcp_bridge.py` to initialize `MultiServerMCPClient` and load tools.
- [x] **Verification:** Create `verify_mcp.py` to list available tools and run a test query.
- [x] **Calculator:** Create `tools/math_tools.py` with `numexpr`-backed calculator tool.

## Phase 3: The Brain (Router)
- [x] Design the `AgentState` (TypedDict) with `messages` + `intent` fields.
- [x] Implement `analyze_intent` node using `.with_structured_output(RouteDecision)` to classify into 6 Buckets.
- [x] Create `tool_invoker` worker node with intent-specific system prompts + `.bind_tools()`.
- [x] Wire full graph: `analyze_intent → tool_invoker ⇄ tools → END` (all intents route through tool_invoker, including explanation).

## Phase 4a: The Builder (Offline Ingestion Pipeline)

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
- [ ] **Full Corpus (Wave 3):** Ingest remaining 15-20 documents (NCA5 chapters, IPCC AR6, NHC reports, Yale CC articles, full satellite set). See `data/DATASET_SOURCES.md` for complete manifest.

## Phase 4b: The Runtime (RAG Integration)

**Strategy:** Lightweight retrieval tool wired into the Bucket 5 (Explainability) path of our LangGraph.

- [x] **Create RAG Tool:** `tools/rag_tool.py` — a LangChain `@tool` that queries ChromaDB, retrieves matching summaries, then fetches the original parent documents from the file store.
- [x] **Response Enrichment:** The tool must return both the retrieved text *and* descriptions of any associated visual data (chart summaries) so the synthesizer can cite them.
- [x] **Graph Wiring:** Integrate `rag_tool` into the existing tool set, ensuring the Router dispatches Bucket 5 ("Why" questions) to the `tool_invoker` node with the RAG tool available.
- [x] **Hybrid Context:** For Bucket 5 queries, the synthesizer node must combine RAG-retrieved knowledge with live Xweather data (e.g., current pressure maps alongside historical explanations).
- [x] **End-to-End Test:** "Why is there a heat dome over the Midwest?" — verified: RAG retrieves from CRS Extreme Heat + Frontiers Heatwave + chart/figure analysis; MCP fetches live weather for Chicago & Kansas City; response includes source citations, meteorological terminology, and visual data references. All 5 acceptance criteria pass.

## Phase 4c: Observability (LangSmith)

- [x] **LangSmith Integration:** Added `langsmith` SDK, configured env vars (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT=weatherwise-ai`).
- [x] **Tracing Utility:** Created `src/utils/tracing.py` with config validation and `get_run_config()` helper for run metadata, tags, and custom names.
- [x] **Verification:** Confirmed traces in LangSmith dashboard — all nodes (router, tool_invoker, tools), LLM calls, and tool invocations traced with tags and metadata.

## Phase 5: The Interface & Polish
- [ ] Create `prompts.py` using the tone/style examples from `scope.md`.
- [ ] Build a simple Streamlit or CLI interface for testing.