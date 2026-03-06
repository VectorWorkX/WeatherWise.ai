# WeatherWise.ai — GCP Deployment Plan (Terraform)

## Status

- [x] Pinecone migration (ChromaDB -> Pinecone serverless) — DONE
- [ ] Rotate all API keys (they're in git history)
- [x] Dockerfile (multi-stage: Node frontend build + Python backend)
- [x] Terraform infra provisioning (Cloud Run, Secret Manager, Artifact Registry)
- [x] Cloud Build image push to Artifact Registry
- [x] Cloud Run deployment — LIVE
- [x] Lock CORS to production domain
- [ ] Cloudflare DNS setup for weather-wise.ai
- [ ] GitHub Actions CI/CD

### Deployment URLs
- **Cloud Run:** `https://weatherwise-228097686588.us-central1.run.app`
- **Custom domain:** `https://weather-wise.ai` (pending Cloudflare DNS)
- **GCP Project:** `weatherwise-486608`
- **Region:** `us-central1`

---

## Architecture (As Deployed)

```
    User (browser)
         |
         v
    Cloudflare (DNS + CDN + DDoS)       <-- Free plan, manual setup
         |
         v
    Cloud Run (FastAPI + frontend)       <-- Serves both API/WS and static files
         |
    +----+----+-----------+
    v         v           v
  In-Memory  Pinecone    External APIs
  Checkpoints (free tier) (Anthropic, Xweather, OpenAI)
```

**Key decisions made during deployment:**
- **No Memorystore Redis.** `langgraph-checkpoint-redis` v0.3+ requires RediSearch
  module (FT.* commands) which GCP Memorystore doesn't support. Using in-memory
  checkpointing instead. Can add Redis Cloud free tier (30MB, has RediSearch) later.
- **In-memory checkpointing** is faster (0ms vs 1-5ms per checkpoint) and free.
  Conversations persist within a WebSocket session. Lost on instance restart/deploy.
- **Cloud Build** for Docker image builds (Docker Desktop not available locally).
  ~$0.003 per build, ~2.5 min build time.
- **SERVE_FRONTEND=1** + `FRONTEND_DIST_DIR=/app/frontend/dist` serves the React
  frontend from the same container. No separate GCS bucket or load balancer needed.

---

## Active GCP Resources

| Resource | Terraform ID | Monthly Cost |
|---|---|---|
| Cloud Run service | `google_cloud_run_v2_service.app` | ~$3-8 (usage-based, $0 idle) |
| Artifact Registry | `google_artifact_registry_repository.app` | ~$0.10 |
| Secret Manager (6 secrets) | `google_secret_manager_secret.secrets[*]` | $0 |
| Service Account | `google_service_account.cloud_run` | $0 |
| Public IAM | `google_cloud_run_v2_service_iam_member.public` | $0 |
| GCP API enablement (6 APIs) | `google_project_service.apis[*]` | $0 |
| **Total** | | **~$3-8/mo** |

### Not Deployed (toggle available)
| Resource | Toggle | Monthly Cost |
|---|---|---|
| Memorystore Redis | `enable_redis = true` | ~$36.50 |
| VPC Connector | `enable_redis = true` | ~$7.30 |

---

## Terraform File Structure

```
infra/
  main.tf                  # Provider (google ~> 5.0), API enablement (6 APIs)
  variables.tf             # All input variables with descriptions + sensitive flags
  network.tf               # VPC connector (conditional on enable_redis)
  redis.tf                 # Memorystore Redis (conditional on enable_redis)
  secrets.tf               # Secret Manager: 6 secrets + versions
  registry.tf              # Artifact Registry with keep-3 cleanup policy
  cloud_run.tf             # Cloud Run service + SA + secret IAM + public access
  outputs.tf               # service_url, image_url, redis_host
  terraform.tfvars.example # Example values (never commit real .tfvars)
  terraform.tfvars         # Actual values (in .gitignore, never committed)
```

---

## Cloud Run Configuration

| Setting | Value | Why |
|---|---|---|
| CPU | 1 | I/O-bound (waiting on LLM APIs), async concurrency handles it |
| Memory | 1Gi | Python + LangGraph + all deps |
| Min instances | 0 | $0 when idle, 5-10s cold start acceptable |
| Max instances | 3 | Hard cap, 30 concurrent users max |
| Timeout | 300s | Complex multi-city + RAG queries can take 30-60s |
| Concurrency | 10 | WebSocket connections hold 10-60s during streaming |
| Port | 8000 | Uvicorn default |
| Startup probe | GET /api/health | 5s initial delay, 5s period, 6 failures max |

### Environment Variables (non-secret)
```
SERVE_FRONTEND=1
FRONTEND_DIST_DIR=/app/frontend/dist
ENVIRONMENT=production
CHECKPOINTER_MODE=memory
LOG_FORMAT=json
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=weatherwise-ai
PINECONE_INDEX_NAME=weatherwise-knowledge
XWEATHER_MCP_URL=https://mcp.api.xweather.com/mcp
```

### Secrets (from Secret Manager)
```
ANTHROPIC_API_KEY
OPENAI_API_KEY
PINECONE_API_KEY
XWEATHER_CLIENT_ID
XWEATHER_CLIENT_SECRET
LANGCHAIN_API_KEY
```

---

## Dockerfile (Multi-Stage)

```dockerfile
# Stage 1: Build frontend (Node 20)
FROM node:20-slim AS frontend-builder
# npm ci + npm run build

# Stage 2: Python application (Python 3.11)
FROM python:3.11-slim
# pip install ".[server]" (includes source for hatch build)
# Copy frontend/dist from stage 1
# CMD: uvicorn weatherwise.server:app
```

---

## Deployment Commands

### Build + Push Image
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/weatherwise-486608/weatherwise/app:latest
```

### Update Cloud Run (after image push)
```bash
gcloud run services update weatherwise --region=us-central1 \
  --image=us-central1-docker.pkg.dev/weatherwise-486608/weatherwise/app:latest
```

### Terraform Apply (infra changes)
```bash
cd infra && terraform apply
```

---

## DNS Setup (Cloudflare — Pending)

In Cloudflare dashboard for `weather-wise.ai`:

| Type | Name | Target | Proxy |
|---|---|---|---|
| CNAME | `@` | `weatherwise-228097686588.us-central1.run.app` | Proxied (orange) |
| CNAME | `www` | `weatherwise-228097686588.us-central1.run.app` | Proxied (orange) |

SSL/TLS mode: **Full (strict)**

---

## CORS Origins (server.py)

```python
_cors_origins = [
    "https://weather-wise.ai",
    "https://www.weather-wise.ai",
    "https://weatherwise-2i5llicxea-uc.a.run.app",
    "https://weatherwise-228097686588.us-central1.run.app",
]
# + localhost:5173/5174 in non-production
```

---

## Rotate API Keys (CRITICAL — Pre-Production)

Every key in `.env` has been in git history. Before going live:

- [ ] Anthropic: Regenerate at console.anthropic.com
- [ ] OpenAI: Regenerate at platform.openai.com/api-keys
- [ ] Pinecone: Regenerate at app.pinecone.io
- [ ] Xweather: Regenerate at xweather.com/account
- [ ] LangSmith: Regenerate at smith.langchain.com/settings

After rotating: update `infra/terraform.tfvars` → `terraform apply` → redeploy.

---

## Billing Safety

1. **Budget alert:** GCP Console > Billing > Budgets — set $25/mo with alerts at 50%, 80%, 100%
2. **Max instances = 3** caps compute costs
3. **No GPUs, no GKE, no Compute Engine**
4. **Free tier services:** Pinecone, Cloudflare, Secret Manager, Artifact Registry (~$0.10)
5. **Real cost is API usage:** ~$0.02-0.05 per complex query in LLM tokens

---

## Future Enhancements

- **Redis Cloud free tier** (30MB, has RediSearch) for cross-session persistence
- **GitHub Actions CI/CD** — automate build+deploy on push to main
- **Cloud Run min_instances=1** if cold start UX is unacceptable (~$10/mo extra)
