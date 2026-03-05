# Scope: WeatherWise.ai Multi-Agent Architecture

## In Scope (v1)
- Stateful multi-agent graph with 7 nodes (supervisor, planner, executor, fetcher, retriever, calculator, synthesizer)
- Planner-Executor pattern with Send-based dynamic fan-out for parallel task execution
- Fan-in via Annotated reducers on shared state fields
- Supervisor node for ambiguity detection (fast model)
- Planner node for query decomposition into independent sub-tasks
- Mock tools: weather API, vector DB search, math operations
- In-memory checkpointing (MemorySaver) for conversation persistence
- Correct reducer definitions to prevent race conditions during parallel writes

## In Scope (v2 — future)
- Real Xweather MCP integration replacing mock weather tool
- ChromaDB RAG pipeline replacing mock vector search
- Dependent task chaining (e.g., math step that needs weather data first)
- Human-in-the-loop safety review for severe weather alerts
- Streaming support for real-time token output
- Production checkpointer (SQLite/Redis)
- Frontend integration

## Out of Scope
- Frontend development (existing React app is separate)
- PDF ingestion / embedding pipeline
- Deployment / CI/CD
- Authentication / multi-tenancy
