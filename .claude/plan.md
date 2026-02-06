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
- [x] Wire full graph: `analyze_intent → tool_invoker ⇄ tools → END` (explanation → END for now).

## Phase 4: The Knowledge (RAG)
- [ ] Set up Vector DB (ChromaDB) for atmospheric papers.
- [ ] Ingest sample atmospheric documents (e.g., definitions of "Heat Dome", "Lake Effect").
- [ ] Implement the `retrieve_knowledge` node for "Why" questions.

## Phase 5: The Interface & Polish
- [ ] Create `prompts.py` using the tone/style examples from `scope.md`.
- [ ] Build a simple Streamlit or CLI interface for testing.