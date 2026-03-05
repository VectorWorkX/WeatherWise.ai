"""LangSmith tracing configuration for the WeatherWise agent.

Sets up and validates the LangSmith connection. All LangGraph nodes,
LLM calls, and tool invocations are traced automatically once the
environment variables are present.

Environment variables (set in .env):
    LANGCHAIN_TRACING_V2: "true" to enable tracing
    LANGCHAIN_API_KEY: LangSmith API key
    LANGCHAIN_PROJECT: Project name in LangSmith dashboard

Ported from feature/claude-integration branch (src/utils/tracing.py).
"""

from __future__ import annotations

import os


_REQUIRED_VARS = ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT")

LANGSMITH_BASE_URL = "https://smith.langchain.com"


def is_tracing_enabled() -> bool:
    """Check whether LangSmith tracing is configured and enabled."""
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"


def setup_tracing() -> dict[str, str]:
    """Enable LangSmith tracing if API key is available.

    Auto-sets LANGCHAIN_TRACING_V2 and LANGCHAIN_PROJECT if LANGCHAIN_API_KEY
    is present. Validates configuration and returns status.

    Returns:
        Dict with tracing status info.

    Raises:
        RuntimeError: If tracing is enabled but API key is missing.
    """
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")

    if api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "weatherwise")

    tracing_on = is_tracing_enabled()
    project = os.getenv("LANGCHAIN_PROJECT", "default")

    if tracing_on and not api_key:
        raise RuntimeError(
            "LANGCHAIN_TRACING_V2 is enabled but LANGCHAIN_API_KEY is not set. "
            "Get your key from https://smith.langchain.com/settings"
        )

    return {
        "tracing_enabled": str(tracing_on),
        "project": project,
        "api_key_set": str(bool(api_key)),
    }


def get_run_config(
    *,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    run_name: str | None = None,
) -> dict:
    """Build a LangGraph-compatible RunnableConfig with tracing metadata.

    Args:
        tags: Labels for filtering runs (e.g., ["bucket:explanation", "test"]).
        metadata: Key-value pairs attached to the run (e.g., {"user_id": "123"}).
        run_name: Custom name shown in the LangSmith trace tree.

    Returns:
        Config dict suitable for ``graph.ainvoke(inputs, config=...)``.
    """
    config: dict = {}

    if tags:
        config["tags"] = tags
    if metadata:
        config["metadata"] = metadata
    if run_name:
        config["run_name"] = run_name

    return config
