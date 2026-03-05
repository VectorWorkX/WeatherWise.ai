"""Retry wrapper for transient API failures (429, 503, timeouts)."""

from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from weatherwise.logging import get_logger

logger = get_logger(__name__)

# Exceptions that indicate transient failures worth retrying
_RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    OSError,
)

try:
    from anthropic import APIConnectionError, APITimeoutError, RateLimitError

    _RETRYABLE_EXCEPTIONS = (  # type: ignore[assignment]
        *_RETRYABLE_EXCEPTIONS,
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
    )
except ImportError:
    pass


def llm_retry(max_attempts: int = 3):
    """Decorator for retrying LLM/API calls with exponential backoff."""
    return retry(
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=lambda retry_state: logger.warning(
            "retrying_api_call",
            attempt=retry_state.attempt_number,
            exception=str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
        ),
        reraise=True,
    )
