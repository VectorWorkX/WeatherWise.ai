output "service_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_v2_service.app.uri
}

output "image_url" {
  description = "Artifact Registry image URL"
  value       = local.image
}

output "redis_host" {
  description = "Memorystore Redis host (if enabled)"
  value       = var.enable_redis ? google_redis_instance.checkpoints[0].host : "N/A (in-memory mode)"
}
