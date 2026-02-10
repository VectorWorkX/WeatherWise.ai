"""Agent LLM singleton swapping for model comparison tests.

Monkey-patches the module-level LLM singletons in each agent module
to use a different model backend. Only swaps the 3 worker agent LLMs —
supervisor and synthesizer stay as Haiku for consistent routing.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")


def swap_agent_llms(model_key: str) -> None:
    """Swap all agent-level LLM singletons to use a different model.

    Must be called BEFORE build_graph().

    Args:
        model_key: A registered model key (e.g., 'claude-sonnet', 'gemini-flash').
    """
    from utils.model_registry import get_model

    import graph.nodes.analyst_agent as aa
    import graph.nodes.forecast_agent as fa
    import graph.nodes.research_agent as ra

    new_worker_llm = get_model(model_key, max_tokens=2048)

    fa._forecast_llm = new_worker_llm
    ra._research_llm = new_worker_llm
    aa._analyst_llm = new_worker_llm


def reset_agent_llms() -> None:
    """Reset all singletons back to None so defaults reload on next use."""
    import graph.nodes.analyst_agent as aa
    import graph.nodes.forecast_agent as fa
    import graph.nodes.research_agent as ra

    fa._forecast_llm = None
    ra._research_llm = None
    aa._analyst_llm = None
