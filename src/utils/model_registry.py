"""Centralized model factory registry for LLM backends.

Provides a pluggable way to register and instantiate different LLM backends
(Claude, Gemini, etc.) for model comparison in evaluation.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List

from langchain_core.language_models import BaseChatModel

# Registry: model_key -> factory function
_MODEL_REGISTRY: Dict[str, Callable[..., BaseChatModel]] = {}


def register_model(key: str, factory: Callable[..., BaseChatModel]) -> None:
    """Register a model factory under a key like 'claude-sonnet' or 'gemini-flash'."""
    _MODEL_REGISTRY[key] = factory


def get_model(key: str, **kwargs) -> BaseChatModel:
    """Instantiate a registered model by key."""
    if key not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{key}' not registered. Available: {available}")
    return _MODEL_REGISTRY[key](**kwargs)


def list_models() -> List[str]:
    """Return all registered model keys."""
    return list(_MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------


def _register_defaults() -> None:
    """Register the default Anthropic models."""
    from langchain_anthropic import ChatAnthropic

    def claude_sonnet_factory(**kwargs) -> BaseChatModel:
        defaults = dict(
            model="claude-sonnet-4-20250514",
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            temperature=0,
            max_tokens=2048,
        )
        defaults.update(kwargs)
        return ChatAnthropic(**defaults)

    def claude_haiku_factory(**kwargs) -> BaseChatModel:
        defaults = dict(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            temperature=0,
            max_tokens=256,
        )
        defaults.update(kwargs)
        return ChatAnthropic(**defaults)

    register_model("claude-sonnet", claude_sonnet_factory)
    register_model("claude-haiku", claude_haiku_factory)


def _register_gemini() -> None:
    """Register Gemini models if langchain-google-genai is installed."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        def gemini_flash_factory(**kwargs) -> BaseChatModel:
            defaults = dict(
                model="gemini-2.0-flash",
                google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
                temperature=0,
                max_output_tokens=2048,
            )
            defaults.update(kwargs)
            return ChatGoogleGenerativeAI(**defaults)

        register_model("gemini-flash", gemini_flash_factory)
    except ImportError:
        pass  # graceful skip if not installed


def _register_openai() -> None:
    """Register OpenAI models if langchain-openai is installed."""
    try:
        from langchain_openai import ChatOpenAI

        def gpt4o_factory(**kwargs) -> BaseChatModel:
            defaults = dict(
                model="gpt-4o",
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                temperature=0,
                max_tokens=1024,
            )
            defaults.update(kwargs)
            return ChatOpenAI(**defaults)

        register_model("gpt-4o", gpt4o_factory)
    except ImportError:
        pass  # graceful skip if not installed


_register_defaults()
_register_gemini()
_register_openai()
