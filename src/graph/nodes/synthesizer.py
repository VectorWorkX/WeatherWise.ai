"""Synthesizer node — formats agent responses before returning to the user.

Runs after any specialist agent completes, before END.
Uses Haiku (fast, cheap) for response formatting.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from graph.prompts import SYNTHESIZER_PROMPT
from graph.state import AgentState

load_dotenv()

_synthesizer_llm: ChatAnthropic | None = None


def _get_synthesizer_llm() -> ChatAnthropic:
    """Return a cached Haiku instance for fast response formatting."""
    global _synthesizer_llm
    if _synthesizer_llm is None:
        _synthesizer_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0,
            max_tokens=2048,
        )
    return _synthesizer_llm


async def synthesizer_node(state: AgentState) -> dict:
    """Format the specialist agent's response for the end user."""
    last_msg = state["messages"][-1]
    intent = state.get("intent", "")

    llm = _get_synthesizer_llm()
    formatted = await llm.ainvoke(
        [
            SystemMessage(
                content=SYNTHESIZER_PROMPT,
                additional_kwargs={"cache_control": {"type": "ephemeral"}},
            ),
            HumanMessage(
                content=f"Intent: {intent}\n\nAgent response:\n{last_msg.content}"
            ),
        ],
        config={"tags": ["synthesizer"]},
    )

    return {"messages": [formatted]}
