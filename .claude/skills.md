# Skills: WeatherWise.ai Development

## LangGraph Patterns
- **StateGraph with TypedDict:** Define state schema with Annotated reducers for parallel-safe updates
- **Send fan-out:** Return `list[Send]` from conditional edge routers to spawn parallel node instances
- **Fan-in convergence:** Workers write to reducer-backed fields; downstream node reads merged state
- **Conditional routing:** `add_conditional_edges(source, router_fn, path_map)` for dynamic graph traversal
- **Checkpointing:** `graph.compile(checkpointer=MemorySaver())` for stateful conversations

## State Reducer Rules
- `Annotated[list, operator.add]` — list concatenation (append-only, safe for parallel writes)
- `Annotated[dict, merge_fn]` — custom dict merger (safe for parallel writes to different keys)
- `add_messages` — LangChain message dedup by ID + auto-deserialization
- No reducer (bare type) — last-write-wins overwrite semantics

## Node Implementation Pattern
- Nodes receive full `WeatherState` (or custom payload if invoked via Send)
- Nodes return a **partial dict** of state updates (only keys being changed)
- Worker nodes invoked via Send receive typed payloads (`FetchPayload`, `CalcPayload`, `VisualizePayload`), return updates to parent state keys

## Testing Approach
- Unit test each node function in isolation with mock state
- Integration test the compiled graph with mock tools
- Verify fan-out/fan-in by checking reducer-merged results
