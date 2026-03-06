variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-central1"
}

variable "enable_redis" {
  description = "Enable Memorystore Redis for persistent checkpointing. Set to false to save ~$44/mo (uses in-memory checkpoints instead)."
  type        = bool
  default     = true
}

variable "cloud_run_min_instances" {
  description = "Minimum Cloud Run instances. 0 = scale to zero (saves ~$10/mo but has cold starts)."
  type        = number
  default     = 0
}

variable "cloud_run_max_instances" {
  description = "Maximum Cloud Run instances. Hard cap to prevent runaway costs."
  type        = number
  default     = 3
}

# Secrets — provided via terraform.tfvars (never committed)
variable "anthropic_api_key" {
  type      = string
  sensitive = true
}

variable "openai_api_key" {
  type      = string
  sensitive = true
}

variable "pinecone_api_key" {
  type      = string
  sensitive = true
}

variable "xweather_client_id" {
  type      = string
  sensitive = true
}

variable "xweather_client_secret" {
  type      = string
  sensitive = true
}

variable "langchain_api_key" {
  type      = string
  sensitive = true
}
