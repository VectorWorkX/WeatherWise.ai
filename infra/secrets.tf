locals {
  secrets = {
    anthropic-api-key      = var.anthropic_api_key
    openai-api-key         = var.openai_api_key
    pinecone-api-key       = var.pinecone_api_key
    xweather-client-id     = var.xweather_client_id
    xweather-client-secret = var.xweather_client_secret
    langchain-api-key      = var.langchain_api_key
  }
}

resource "google_secret_manager_secret" "secrets" {
  for_each  = local.secrets
  secret_id = each.key

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "versions" {
  for_each    = local.secrets
  secret      = google_secret_manager_secret.secrets[each.key].id
  secret_data = each.value
}
