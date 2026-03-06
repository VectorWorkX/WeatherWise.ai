# WeatherWise.ai — Multi-Agent Architecture Discussion (Complete)

## Context
This document captures the FULL architecture discussion for transforming WeatherWise.ai from a routed-specialist pattern to a true Plan-and-Execute multi-agent system. Includes all iterations, concerns, reviews, and the final design.

---

## 1. Current Architecture (What We Have — Before Refactor)

```
START → supervisor (Haiku, classifies 1 of 8 buckets)
    → routes to exactly 1 agent:
        forecast_agent (Buckets 1-4: real_time, comparative, lifestyle, safety)
        research_agent (Bucket 5: explanation, with Corrective RAG)
        analyst_agent (Bucket 6: math)
    → safety_review (forecast_agent path ONLY)
    → synthesizer (Haiku, reformats text — redundant)
    → END
```

### Problems Identified
1. **Not truly multi-agent** — only 1 agent runs per query, no collaboration
2. **No parallel execution** — Send() not used
3. **No sequential dependency handling** — can't chain agent outputs
4. **Synthesizer is wasteful** — just reformats prose to prose
5. **Safety review only on forecast path** — research/analyst bypass it
6. **Structured output models exist but unused** — ForecastResponse, ResearchResponse, AnalystResponse in models.py
7. **ForecastResponse only supports single location** — needs multi-location
8. **LangGraph underutilized** — no Send(), no operator.add, no loops, no subgraph state isolation

### What We Were Using vs What LangGraph Offers

| LangGraph Feature | Available | Were We Using It? |
|---|---|---|
| StateGraph | Yes | Yes, but as a linear pipe |
| Send() (parallel fan-out) | Yes | NO |
| operator.add reducer | Yes | NO |
| Conditional edges (loops) | Yes | NO — graph was acyclic |
| Global state as agent communication | Yes | NO |
| interrupt() | Yes | Partially — only forecast path |
| Subgraphs | Yes | Yes |
| Checkpointer (MemorySaver) | Yes | Yes |

---

## 2. Architecture Evolution — Three Approaches Discussed

### Attempt 1: One-Shot Send() Fan-Out
- Supervisor selects 1+ agents via multi-label classification
- Send() dispatches all agents in parallel
- **Rejected:** Can't handle sequential dependencies (agent B needing agent A's output)

### Attempt 2: Iterative Orchestrator Loop
- Orchestrator loops: dispatch → collect findings → decide next → loop or done
- `is_final` flag for zero overhead on simple queries
- **Rejected:** Orchestrator re-reads state and re-reasons at every iteration — redundant LLM calls. User's insight: "State is global — why does an orchestrator need to re-read and re-reason when agents can just read findings from state?"

### Attempt 3: Plan-and-Execute (Selected — Best Architecture)
- Supervisor creates a FULL execution plan in ONE call
- Plan specifies ordered steps, each step can have 1+ agents (parallel via Send())
- Graph executes plan step-by-step using LangGraph's global state
- Each agent reads accumulated findings from state — no orchestrator needed
- **ONE supervisor LLM call regardless of query complexity**
- **Zero LLM calls in the execution loop**

### Why Plan-and-Execute Wins:
- vs Attempt 1: Handles sequential dependencies via multi-step plans
- vs Attempt 2: Zero LLM calls in the execution loop (saves N-1 calls per multi-step query)
- Simple queries (80%+): identical performance to current system
- Compound/sequential: new capability, optimal performance

---

## 3. Key Design Decisions

### Decision 1: Split Supervisor (Haiku classifier + Sonnet planner)
**Problem:** Asking one model to classify intent + detect compound queries + generate multi-step plans + decide parallel vs sequential is too much for a single call.
**Solution:** Two stages:
- Stage 1 (Haiku): Classifies 8 buckets + binary `is_compound` detection. For simple queries (80%+), creates deterministic plan via AGENT_MAP — no second call needed.
- Stage 2 (Sonnet): Only invoked for compound queries (20%). Generates multi-step ExecutionPlan with correct agent assignments and parallel/sequential ordering.
**Result:** Simple queries stay fast (1 Haiku call). Compound queries get accuracy (Sonnet reasoning).

### Decision 2: Subgraph State Isolation
**Problem:** Every Send() passes the full parent state. Agent internal messages (system prompts, tool calls, tool results) leak back to parent. By step 2, parent messages has 50+ messages of noise.
**Solution:** Two state types:
- `AgentState` (parent): messages, intent, execution_plan, current_step, findings
- `AgentInternalState` (subgraph): messages, rag_documents, rag_grade, structured_output
The wrapper constructs MINIMAL input (user query + step description + prior findings summary) for the subgraph. Only the distilled Finding crosses back. Parent messages stays constant size.

### Decision 3: Structured Output Extraction (No Extra LLM)
**Problem:** structured_data was empty — models existed but were never used.
**Solution:** Add `extract_findings` node to each agent subgraph. This node parses MCP tool results (already JSON) into typed Pydantic models (ForecastResponse, ResearchResponse, AnalystResponse). Pure JSON parsing — no LLM call needed.

### Decision 4: Synthesizer as Merger
In single-agent system: synthesizer was wasteful (just reformatted prose).
In multi-agent system: synthesizer is ESSENTIAL — merges N structured findings from N agents into one coherent narrative. Reads findings sorted by (step_number, agent_name) for deterministic ordering.

### Decision 5: Universal Safety Review
Old: Only checked forecast_agent path for safety intent.
New: Scans ALL findings from ALL agents for severe keywords. Research agent explaining "catastrophic Hurricane Katrina" now triggers safety review.

### Decision 6: Agent Error Recovery
Each agent wrapper has try/except with 30-second timeout. On failure, creates a Finding with `finding_type="error"` and `confidence=0.0`. Synthesizer works with partial data: "I was able to get the forecast but the research component encountered an error."

### Decision 7: Prior Findings Context Injection
For sequential dependencies, the wrapper injects prior findings summary into the agent's context:
```
"Context from prior steps:
- forecast_agent (step 0): Chicago temperature is 72°F
Your task (step 1): Square the temperature value from step 0"
```
This gives agents explicit context instead of making them parse through raw message history.

### Decision 8: Confidence Scoring
Each Finding gets a confidence score (0.0-1.0) calculated from: tool success, content length, structured data completeness, RAG grade. Synthesizer caveats low-confidence findings.

### Decision 9: Findings Order Handling
Parallel agents complete in arbitrary order. operator.add concatenates in completion order. All consumers sort findings by (step_number, agent_name) before use for deterministic behavior.

### Decision 10: is_final Removed
Originally proposed as optimization for simple queries. Removed because the split supervisor already provides this optimization — simple queries get deterministic AGENT_MAP plans, never invoke the plan generator. is_final was dead code.

---

## 4. Final Architecture — Complete Graph

```
START → intent_classifier (Haiku — classifies bucket + is_compound?)
    ├─ greeting/off_topic → direct response → END
    ├─ simple (is_compound=False) → deterministic AGENT_MAP plan → step_executor
    └─ compound (is_compound=True) → plan_generator (Sonnet) → plan_validation → step_executor

step_executor (reads plan[current_step], returns Send())
    → Send(agent_1, state), Send(agent_2, state)  [parallel via asyncio]
        Each agent wrapped in make_agent_wrapper:
          - Constructs MINIMAL AgentInternalState (user query + step desc + prior findings)
          - Agent subgraph runs ReAct loop + extract_findings (structured output from JSON)
          - try/except with 30s timeout → fallback Finding on error
          - Returns ONLY Finding to parent (internal messages stay in subgraph)
    → collector (current_step += 1, no LLM)
    → should_continue? (also checks if prev step critically failed)
        ├─ more steps + prev step OK → step_executor (loop)
        ├─ prev step ALL errors + next step depends on it → safety_review (abort)
        └─ done → safety_review (universal, ALL findings)
                 → synthesizer (merger, reads ALL findings sorted by step+agent)
                 → END
```

### Edge Map

| From | To | Type | Condition |
|---|---|---|---|
| START | intent_classifier | entry | — |
| intent_classifier | END | conditional | greeting/off_topic |
| intent_classifier | step_executor | conditional | simple (plan set) |
| intent_classifier | plan_generator | conditional | compound (no plan) |
| plan_generator | step_executor | direct | — |
| step_executor | forecast/research/analyst_agent | Send() | per plan step |
| forecast_agent | collector | direct | — |
| research_agent | collector | direct | — |
| analyst_agent | collector | direct | — |
| collector | should_continue | conditional | more steps / abort / done |
| safety_review | synthesizer | direct | — |
| synthesizer | END | direct | — |

### Node Inventory

| Node | LLM? | Model | Purpose |
|---|---|---|---|
| intent_classifier | Yes | Haiku | Classify 8 buckets + is_compound |
| plan_generator | Yes | Sonnet | Multi-step plan (compound only, 20% of queries) |
| step_executor | No | — | Read plan, dispatch via Send() |
| forecast_agent wrapper | Yes | Sonnet (inside subgraph) | Weather data retrieval |
| research_agent wrapper | Yes | Sonnet + Haiku grader | RAG + atmospheric science |
| analyst_agent wrapper | Yes | Sonnet (inside subgraph) | Math/calculations |
| collector | No | — | Increment step counter |
| safety_review | No | — | Keyword scan all findings |
| synthesizer | Yes | Haiku | Merge N findings into 1 response |

### LLM Call Counts

| Query Type | Classifier | Planner | Agent ReAct | Synthesizer | Total |
|---|---|---|---|---|---|
| Simple (80%) | 1 Haiku | 0 | N Sonnet | 1 Haiku | 2 + N |
| Compound (20%) | 1 Haiku | 1 Sonnet | N Sonnet | 1 Haiku | 3 + N |
| Greeting/off-topic | 1 Haiku | 0 | 0 | 0 | 1 |

LLM calls in execution loop (step_executor, collector, should_continue): **always 0**.

---

## 5. State Design

### Parent State (AgentState)
```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # only user queries + synthesizer responses
    intent: str                                           # bucket classification
    execution_plan: dict | None                           # serialized ExecutionPlan
    current_step: int                                     # 0-indexed, advances per step
    findings: Annotated[list[dict], operator.add]         # Finding dicts, merged via add
    requires_human_approval: bool                         # safety flag
    human_approved: bool                                  # user approval
```

### Subgraph State (AgentInternalState) — Isolated from parent
```python
class AgentInternalState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # internal tool calls, stays in subgraph
    rag_documents: list[str]                              # research agent only
    rag_grade: str                                        # research agent only
    structured_output: dict | None                        # populated by extract_findings
```

### State Size: Constant
Parent messages grows by exactly 1 AIMessage per query (synthesizer output). Agent internal messages (system prompts, tool calls, tool results) live and die inside subgraphs.

---

## 6. Pydantic Models

```python
VALID_AGENTS = frozenset({"forecast_agent", "research_agent", "analyst_agent"})

class LocationData(BaseModel):       # multi-location forecast support
    location: str
    temperature: str
    conditions: str
    wind: str
    humidity: Optional[str] = None
    advisory: Optional[str] = None

class ForecastResponse(BaseModel):   # expanded from single-location
    locations: List[LocationData]
    comparison_summary: Optional[str] = None
    sources: List[str] = []

class ResearchResponse(BaseModel):   # unchanged
    explanation: str
    key_concepts: List[str] = []
    citations: List[str] = []

class AnalystResponse(BaseModel):    # unchanged
    fetched_values: Dict[str, str] = {}
    expression: str
    result: str

class ClassificationResult(BaseModel):  # Stage 1 output
    bucket: Literal["real_time", "comparative", "lifestyle", "safety", "explanation", "math", "greeting", "off_topic"]
    is_compound: bool
    reasoning: str
    response: Optional[str] = None

class PlanStep(BaseModel):           # one step in execution plan
    step_number: int
    agents: List[str]
    description: str

class ExecutionPlan(BaseModel):      # full plan from supervisor
    steps: List[PlanStep]

class Finding(BaseModel):            # universal agent output
    agent_name: str
    finding_type: Literal["forecast", "research", "analysis", "error"]
    structured_data: Dict[str, Any]
    summary: str
    tools_used: List[str] = []
    step_number: int
    confidence: float = 1.0
```

---

## 7. Query Execution Traces

### Simple: "What is the temp in SF?"
```
intent_classifier (Haiku) → bucket: real_time, is_compound: False
    → deterministic plan: [Step(0, ["forecast_agent"], "Fetch SF weather")]
    → step_executor → Send("forecast_agent")
    → forecast_agent runs ReAct (worker ⇄ MCP tools → extract_findings)
    → wrapper returns Finding(structured_data=ForecastResponse, summary="SF is 65°F...")
    → collector → current_step = 1
    → should_continue → done (1 < 1 = False)
    → safety_review → pass (no severe keywords)
    → synthesizer → formats single finding → AIMessage
    → END

LLM calls: 1 Haiku + ~2 Sonnet (forecast ReAct) + 1 Haiku = same as current
```

### Compound Parallel: "Compare Buffalo vs Boston AND explain lake-effect snow"
```
intent_classifier (Haiku) → bucket: comparative, is_compound: True
    → no plan set, routes to plan_generator

plan_generator (Sonnet) → plan: [
    Step(0, ["forecast_agent", "research_agent"], "Compare cities AND explain science")
]
    → plan_validation → OK

step_executor → Send("forecast_agent", state), Send("research_agent", state)
    Both run in PARALLEL:
    - forecast_agent: fetches Buffalo + Boston data → Finding(forecast)
    - research_agent: RAG for lake-effect → Finding(research)

    operator.add merges: [forecast_finding, research_finding]

collector → current_step = 1
should_continue → done (1 < 1 = False)
safety_review → pass
synthesizer → merges 2 findings into unified narrative
END
```

### Sequential: "Fetch Chicago's temp, square it, find city with that zipcode"
```
intent_classifier → is_compound: True → plan_generator

plan_generator → plan: [
    Step(0, ["forecast_agent"], "Fetch Chicago temperature"),
    Step(1, ["analyst_agent"], "Square the temperature from step 0"),
    Step(2, ["forecast_agent"], "Look up city for zipcode from step 1 result")
]

Step 0: step_executor → Send("forecast_agent")
    → Finding: "Chicago: 72°F"
    → collector → step 1
    → should_continue → more steps

Step 1: step_executor → Send("analyst_agent")
    → wrapper injects: "Prior: forecast_agent step 0: Chicago 72°F. Your task: Square it."
    → analyst reads context, calculates: 72² = 5184
    → Finding: "5184"
    → collector → step 2
    → should_continue → more steps

Step 2: step_executor → Send("forecast_agent")
    → wrapper injects: "Prior: forecast 72°F, analyst 5184. Your task: Look up zipcode 05184."
    → forecast fetches weather for zipcode 05184
    → Finding: weather data for that location
    → collector → step 3
    → should_continue → done (3 < 3 = False)

safety_review → synthesizer merges 3 findings → END
```

### Greeting: "Hello!"
```
intent_classifier → bucket: greeting, response: "Welcome to WeatherWise.ai..."
    → route_after_classifier → END (direct response, no agents)
```

### Safety across agents: "Explain why Hurricane Katrina was catastrophic"
```
intent_classifier → bucket: explanation → research_agent
    → research_agent Finding: "...catastrophic...life-threatening..."
    → safety_review → keyword match on "catastrophic" → interrupt()
    → user approves → synthesizer → END
```

---

## 8. Concerns Raised and Addressed

### Concern: Supervisor doing too much
**Resolution:** Split into intent_classifier (Haiku) + plan_generator (Sonnet). Each does one job well.

### Concern: Structured data empty
**Resolution:** Added extract_findings node to each agent subgraph. Parses MCP JSON into Pydantic models. No extra LLM call.

### Concern: State size growth through message accumulation
**Resolution:** AgentInternalState isolated from parent AgentState. Wrapper is the bridge. Internal messages stay in subgraph. Only Finding crosses back.

### Concern: is_final never used
**Resolution:** Removed. Split supervisor already provides the optimization.

### Concern: Agent name validation
**Resolution:** VALID_AGENTS frozenset. Plan generator validates. Step executor filters.

### Concern: Safety review timing
**Resolution:** End-of-pipeline is correct. User sees full picture before approval. Non-blocking warning event sent during execution.

### Concern: Findings order in parallel
**Resolution:** Sort by (step_number, agent_name) before all uses. Synthesizer prompt handles arbitrary order.

### Concern: XWeather MCP tool name discovery
**Resolution:** Already runtime-discovered via get_mcp_client().get_tools(). Agents bind tools via schemas, not hardcoded names.

### Concern: Parallel agents can't see each other
**Resolution:** By design — parallel agents are independent. If integration needed, supervisor should plan sequential steps. Supervisor prompt includes rule: "If sub-task B requires data from sub-task A, use separate sequential steps."

### Concern: Error detection in execution loop
**Resolution:** Folded into `should_continue`. After collector increments the step counter, should_continue checks if ALL findings from the previous step are type "error". If so and more steps remain, it aborts to safety_review with partial results (the next step would lack data it depends on). No extra node needed — should_continue already reads execution_plan to decide loop vs done.

### Concern: Agents can't request help from other agents
**Resolution:** Acceptable for bounded domain. Collaboration is orchestrated (via plan), not emergent. For v2 if needed.

---

## 9. A2A Protocol Assessment

Google's A2A (Agent-to-Agent) protocol is for **inter-system** agent communication — agents on different platforms/organizations talking to each other. Our agents are **intra-system** — all in the same LangGraph application, sharing the same Python process and global state. Using A2A would be like using HTTP to call a function in the same file. LangGraph's state-based communication is the correct choice.

A2A would apply if we wanted to expose WeatherWise.ai as a service for external agents, or integrate with third-party agent platforms. Not in current scope.

---

## 10. Agentic Assessment

| Pillar | Score | Status |
|---|---|---|
| Autonomy (agents decide actions) | 8/10 | Agents run independent ReAct loops, choose tools |
| Collaboration (agents work together) | 7/10 | Parallel execution, shared findings, sequential context |
| Reactivity (adapt to changes) | 4/10 | Plan is fixed once created; error detection aborts gracefully |
| Proactivity (take initiative) | 3/10 | Agents can't escalate or request other agents |
| Fault tolerance | 8/10 | try/except, timeout, fallback Finding, confidence scoring |

Verdict: Strong multi-agent system for a domain-specific application. Reactivity and proactivity gaps are conscious tradeoffs — for a bounded weather domain, predictable orchestrated collaboration is better than unpredictable emergent behavior.

---

## 11. Performance Characteristics

| Query Type | Classifier | Planner | Agent Parallel? | Loop Overhead |
|---|---|---|---|---|
| Simple (80%+) | 1 Haiku | 0 | N/A | <1ms |
| Compound parallel | 1 Haiku | 1 Sonnet | YES | <1ms per step |
| Sequential | 1 Haiku | 1 Sonnet | Per step | <1ms per step |
| Greeting/off-topic | 1 Haiku | 0 | N/A | 0ms |

---

## 12. Files Changed (Phase 11)

| File | Change |
|---|---|
| `src/graph/models.py` | REWRITE — 8 models, VALID_AGENTS |
| `src/graph/state.py` | REWRITE — AgentState + AgentInternalState |
| `src/graph/prompts.py` | MODIFY — classifier, planner, domain-scope, merger |
| `src/graph/nodes/supervisor.py` | REWRITE — split into classifier + planner |
| `src/graph/nodes/forecast_agent.py` | MODIFY — AgentInternalState, extract_findings |
| `src/graph/nodes/research_agent.py` | MODIFY — AgentInternalState, extract_findings |
| `src/graph/nodes/analyst_agent.py` | MODIFY — AgentInternalState, extract_findings |
| `src/graph/builder.py` | REWRITE — full Plan-Execute loop with Send() |
| `src/graph/nodes/safety_review.py` | MODIFY — universal findings check |
| `src/graph/nodes/synthesizer.py` | REWRITE — merger with sorted findings |
| `server.py` | MODIFY — new events, classifier rename |
| `app.py` | MODIFY — same as server.py |
| `frontend/src/types/chat.ts` | MODIFY — agents, step, warning fields |
| `frontend/src/hooks/useWeatherChat.ts` | MODIFY — new event handlers |
| `frontend/src/components/chat/MessageBubble.tsx` | MODIFY — multi-agent badges |
| `.claude/plan.md` | MODIFY — add Phase 11 |
| `tests/eval/test_evaluation.py` | MODIFY — update for new models |

---

## 13. Deployment Architecture

The system is deployed on GCP Cloud Run with Terraform-managed infrastructure:

- **Cloud Run** serves both FastAPI backend (API + WebSocket) and React frontend (static files) from one container
- **Checkpointing:** In-memory (MemorySaver) — GCP Memorystore Redis lacks RediSearch module required by `langgraph-checkpoint-redis` v0.3+. Redis Cloud free tier (30MB, has RediSearch) is a future option for cross-session persistence.
- **Vector DB:** Pinecone serverless (free tier, AWS us-east-1, cosine metric, 1536 dims) — migrated from ChromaDB. Parent content stored in vector metadata (eliminates docstore dependency at runtime).
- **Secrets:** GCP Secret Manager (6 API keys), accessed via IAM by Cloud Run service account
- **DNS:** Cloudflare (proxied CNAME to Cloud Run URL, Full strict SSL)
- **CI/CD:** Cloud Build for Docker image builds (~2.5 min, ~$0.003/build)

See [deployment-plan.md](deployment-plan.md) for full infrastructure details, costs, and Terraform configuration.
