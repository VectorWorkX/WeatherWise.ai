# Scope: WeatherWise.ai Multi-Agent Architecture

## Implemented
- Stateful multi-agent graph with 8 nodes (supervisor, planner, executor, fetcher, retriever, calculator, visualizer, synthesizer)
- Planner-Executor pattern with Send-based dynamic fan-out for parallel task execution
- Fan-in via Annotated reducers on shared state fields
- Supervisor node for intent classification and ambiguity detection (Sonnet)
- Planner node for query decomposition into dependency-aware sub-tasks
- Xweather MCP integration for real-time weather data
- Pinecone RAG pipeline (serverless, free tier) with OpenAI embeddings
- Dependency-aware task chaining (e.g., calculate step depends on fetch results)
- Sandboxed Python calculator (AST-validated, signal-based timeout)
- ECharts visualization via structured output (LLM generates complete chart JSON)
- FastAPI + WebSocket server with SSE-like streaming
- React frontend with thinking timeline, tool call visualization, follow-up suggestions
- Multiple checkpointer backends (MemorySaver, AsyncSqliteSaver, AsyncRedisSaver)
- Prompt caching on system prompts (Anthropic native cache_control)
- Token-budgeted chat history trimming for LLM context management
- PDF/image ingestion pipeline (scripts/build_knowledge.py) with Claude Vision
- GCP Cloud Run deployment with Terraform-managed infrastructure
- Cloud Build CI for Docker image builds

## Out of Scope
- Authentication / multi-tenancy
- Human-in-the-loop safety review for severe weather alerts
- GitHub Actions CI/CD (Cloud Build trigger on push to main handles this)
