"""Safety review node — pauses the graph for human approval on severe alerts.

Inserted between ``forecast_agent`` and ``synthesizer`` for safety-intent queries.
Uses ``langgraph.types.interrupt()`` to pause execution and request user approval.
"""

from __future__ import annotations

import re

from langgraph.types import interrupt

from graph.prompts import SAFETY_APPROVAL_MESSAGE, SEVERE_KEYWORDS
from graph.state import AgentState


def safety_review_node(state: AgentState) -> dict:
    """Check if the safety response mentions severe alerts; if so, pause for approval.

    Only triggers for ``safety`` intent queries where the agent's response
    contains keywords indicating severe or life-threatening conditions.
    For non-severe safety responses, passes through without interruption.
    """
    intent = state.get("intent", "")
    if intent != "safety":
        return {}

    # Check the agent's response for severe alert keywords
    last_msg = state["messages"][-1]
    raw = last_msg.content if hasattr(last_msg, "content") else ""
    # content may be a str or a list of content blocks from Anthropic
    if isinstance(raw, list):
        raw = " ".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in raw
        )
    content_lower = raw.lower()

    is_severe = any(kw in content_lower for kw in SEVERE_KEYWORDS)

    if not is_severe:
        return {}

    # Pause the graph and request human approval
    approval = interrupt({
        "type": "approval_required",
        "message": SAFETY_APPROVAL_MESSAGE,
        "severity": "high",
    })

    # If we reach here, the user approved (graph was resumed)
    return {"human_approved": True}
