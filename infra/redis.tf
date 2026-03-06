# Memorystore Redis — stores LangGraph conversation checkpoints.
# Only created when enable_redis = true.

resource "google_redis_instance" "checkpoints" {
  count = var.enable_redis ? 1 : 0

  name           = "weatherwise-checkpoints"
  tier           = "BASIC"
  memory_size_gb = 1
  region         = var.region
  redis_version  = "REDIS_7_0"

  display_name = "WeatherWise Checkpoints"

  depends_on = [google_project_service.apis]
}
