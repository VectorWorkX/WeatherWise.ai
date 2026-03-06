resource "google_artifact_registry_repository" "app" {
  location      = var.region
  repository_id = "weatherwise"
  format        = "DOCKER"
  description   = "Docker images for WeatherWise.ai"

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 3
    }
  }

  depends_on = [google_project_service.apis]
}
