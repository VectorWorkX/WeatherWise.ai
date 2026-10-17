# WeatherWise.ai

A weather analyst you can talk to. It plans each question, pulls live data from Xweather over MCP, adds context from a knowledge base of NOAA, NASA and National Climate Assessment reports, and answers with a written analysis, interactive charts and suggested follow-up questions.

https://github.com/user-attachments/assets/8d0edeee-3438-42ed-931a-97b1560b84af

Live at [weather-wise.ai](https://weather-wise.ai) · [Watch the full-resolution demo](https://github.com/user-attachments/assets/8d0edeee-3438-42ed-931a-97b1560b84af) (1080p, 36 s)

## How it works

The agent is a LangGraph state graph with eight nodes; the reasoning steps run on Claude Sonnet 4.

1. **Route.** A supervisor classifies the message. Greetings and off-topic questions get a direct reply; weather questions go to the planner.
2. **Plan.** The planner turns relative dates into exact ranges and breaks the question into tasks with dependencies, for example one fetch per city followed by a calculation.
3. **Execute.** The executor fans tasks out to workers with LangGraph `Send` in two parallel phases: weather fetches and knowledge-base lookups first, then calculations over the fetched data. Workers report back to the executor until the plan is done.
4. **Answer.** The synthesizer streams the analysis to the browser over a WebSocket and closes with two or three follow-up questions the user can click. When a chart is requested, a visualizer node builds ECharts specs (line, bar, radar, heatmap and others) with metric cards and key insights from the fetched data.

Calculations run in a sandboxed Python evaluator with AST checks and a 5-second timeout. Conversation state is kept per session by a LangGraph checkpointer (in memory by default, Redis or SQLite optional), and runs are traced in LangSmith. A detailed walkthrough is in [docs/WeatherWise_Architecture.md](docs/WeatherWise_Architecture.md).

## Knowledge base

`scripts/build_knowledge.py` parses six reports (NOAA, NASA/AMS, the Fifth National Climate Assessment, CRS and Frontiers) and four satellite images with `unstructured`. Every text chunk, table and figure gets a short Claude summary, with figures going through the vision model. The summaries are embedded with OpenAI `text-embedding-3-small` and stored in Pinecone; a search matches against the summaries and returns the original passage with its source and page. Sources are listed in [data/DATASET_SOURCES.md](data/DATASET_SOURCES.md).

## Running locally

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 18+ and API keys for Anthropic, Xweather, OpenAI and Pinecone.

```bash
git clone https://github.com/VectorWorkX/WeatherWise.ai.git
cd WeatherWise.ai
cp .env.example .env          # fill in the API keys
uv sync --extra server --extra dev
uv run uvicorn weatherwise.server:app --port 8000
```

In a second terminal, start the frontend. It proxies API and WebSocket calls to port 8000.

```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

There is also a terminal client: `uv run weatherwise`.

Tests: `uv run pytest` (the live tests run only when API keys are set).

## Deployment

The Dockerfile builds the React frontend and the FastAPI server into one image. Cloud Build pushes it to Artifact Registry and deploys it to Cloud Run. Terraform in `infra/` provisions the Cloud Run service, Secret Manager secrets, a VPC connector and an optional Memorystore Redis instance.

## Project layout

```
src/weatherwise/
  graph.py       graph wiring
  nodes/         supervisor, planner, executor, workers and synthesizer
  tools/         Xweather MCP client, Pinecone search, sandboxed math
  server.py      FastAPI app with WebSocket streaming
frontend/        React, TypeScript, Tailwind and ECharts
scripts/         knowledge-base ingestion and retrieval checks
infra/           Terraform for Google Cloud
docs/            architecture write-up
tests/
```

## Stack

LangGraph, Claude Sonnet 4, Xweather MCP, Pinecone, OpenAI embeddings, FastAPI, React, TypeScript, ECharts, Docker, Cloud Run, Terraform, LangSmith.
