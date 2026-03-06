# VPC connector — only needed when Redis is enabled.
# Bridges Cloud Run (serverless) to Memorystore (private VPC).

resource "google_vpc_access_connector" "connector" {
  count = var.enable_redis ? 1 : 0

  name          = "weatherwise-vpc"
  region        = var.region
  network       = "default"
  ip_cidr_range = "10.8.0.0/28"
  machine_type  = "e2-micro"
  min_instances = 2
  max_instances = 3

  depends_on = [google_project_service.apis]
}
