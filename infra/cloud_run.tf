locals {
  image = "${var.region}-docker.pkg.dev/${var.project_id}/weatherwise/app:latest"

  # Non-secret environment variables
  env_vars = {
    SERVE_FRONTEND        = "1"
    FRONTEND_DIST_DIR     = "/app/frontend/dist"
    ENVIRONMENT           = "production"
    CHECKPOINTER_MODE     = var.enable_redis ? "redis" : "memory"
    LOG_FORMAT            = "json"
    LANGCHAIN_TRACING_V2  = "true"
    LANGCHAIN_PROJECT     = "weatherwise-ai"
    PINECONE_INDEX_NAME   = "weatherwise-knowledge"
    XWEATHER_MCP_URL      = "https://mcp.api.xweather.com/mcp"
  }

  # Map secret variable names to Secret Manager IDs and env var names
  secret_env_vars = {
    ANTHROPIC_API_KEY      = "anthropic-api-key"
    OPENAI_API_KEY         = "openai-api-key"
    PINECONE_API_KEY       = "pinecone-api-key"
    XWEATHER_CLIENT_ID     = "xweather-client-id"
    XWEATHER_CLIENT_SECRET = "xweather-client-secret"
    LANGCHAIN_API_KEY      = "langchain-api-key"
  }
}

resource "google_cloud_run_v2_service" "app" {
  name     = "weatherwise"
  location = var.region

  template {
    scaling {
      min_instance_count = var.cloud_run_min_instances
      max_instance_count = var.cloud_run_max_instances
    }

    # VPC connector for Redis access (only when Redis enabled)
    dynamic "vpc_access" {
      for_each = var.enable_redis ? [1] : []
      content {
        connector = google_vpc_access_connector.connector[0].id
        egress    = "PRIVATE_RANGES_ONLY"
      }
    }

    containers {
      image = local.image

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle = true
      }

      # Non-secret env vars
      dynamic "env" {
        for_each = local.env_vars
        content {
          name  = env.key
          value = env.value
        }
      }

      # Redis URL (constructed from Memorystore IP)
      dynamic "env" {
        for_each = var.enable_redis ? [1] : []
        content {
          name  = "REDIS_URL"
          value = "redis://${google_redis_instance.checkpoints[0].host}:${google_redis_instance.checkpoints[0].port}"
        }
      }

      # Secret env vars from Secret Manager
      dynamic "env" {
        for_each = local.secret_env_vars
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.secrets[env.value].secret_id
              version = "latest"
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/api/health"
          port = 8000
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 6
        timeout_seconds       = 3
      }
    }

    timeout = "300s"
    max_instance_request_concurrency = 10

    service_account = google_service_account.cloud_run.email
  }

  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_version.versions,
  ]
}

# Service account for Cloud Run
resource "google_service_account" "cloud_run" {
  account_id   = "weatherwise-run"
  display_name = "WeatherWise Cloud Run SA"
}

# Grant Cloud Run SA access to secrets
resource "google_secret_manager_secret_iam_member" "secret_access" {
  for_each  = local.secret_env_vars
  secret_id = google_secret_manager_secret.secrets[each.value].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Make Cloud Run service publicly accessible
resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.app.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
