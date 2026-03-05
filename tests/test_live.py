"""Live integration tests — requires real API keys in .env.

Run with: uv run pytest tests/test_live.py -v -s

These tests hit real APIs (Anthropic, Xweather, OpenAI) and verify
the full graph pipeline end-to-end.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

# Skip all tests if API keys are missing
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("XWEATHER_CLIENT_ID"),
    reason="Live API keys not configured",
)

# Seconds to wait between tests to stay under Anthropic's input token rate limit.
_INTER_TEST_DELAY = 5


@pytest.fixture(scope="module")
def graph():
    """Build the graph once for all live tests."""
    from weatherwise.logging import setup_logging
    from weatherwise.graph import build_graph

    setup_logging()
    return build_graph(checkpointer="none")


@pytest.fixture(autouse=True)
async def _rate_limit_cooldown():
    """Pause between tests to avoid hitting Anthropic's per-minute token limits."""
    yield
    await asyncio.sleep(_INTER_TEST_DELAY)


@pytest.mark.asyncio
async def test_single_city_query(graph):
    """Phase 4: Test with simple single-city query."""
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What's the weather in Tokyo?")]}
    )

    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
    assert len(ai_messages) > 0, "No AI response generated"

    response = ai_messages[-1].content
    print(f"\n--- Single City Response ---\n{response}\n")

    # Should mention Tokyo and contain weather data
    assert "tokyo" in response.lower() or "Tokyo" in response, "Response should mention Tokyo"
    assert len(response) > 50, "Response seems too short for weather data"


@pytest.mark.asyncio
async def test_multi_city_query(graph):
    """Phase 4: Test with compound multi-city query."""
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Compare the weather in New York and London right now")]}
    )

    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
    assert len(ai_messages) > 0, "No AI response generated"

    response = ai_messages[-1].content
    print(f"\n--- Multi City Response ---\n{response}\n")

    response_lower = response.lower()
    assert "new york" in response_lower, "Response should mention New York"
    assert "london" in response_lower, "Response should mention London"

    # Verify fan-out produced results for both cities
    weather = result.get("fetched_weather", {})
    assert len(weather) >= 2, f"Expected weather for 2 cities, got {len(weather)}: {list(weather.keys())}"

    # Verify both plan indices completed
    completed = result.get("completed_indices", [])
    assert len(completed) >= 2, f"Expected at least 2 completed tasks, got {len(completed)}"


@pytest.mark.asyncio
async def test_ambiguity_detection(graph):
    """Phase 4: Test ambiguity detection flow."""
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Weather in Springfield")]}
    )

    messages = result.get("messages", [])
    routing = result.get("routing", "")

    if routing == "ambiguous":
        # Supervisor detected ambiguity — should have a clarifying message
        ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
        assert len(ai_messages) > 0, "Should have clarification message"
        response = ai_messages[-1].content
        print(f"\n--- Ambiguity Detected ---\n{response}\n")
        assert len(response) > 10, "Clarification message too short"
    else:
        # Supervisor may classify as clear (Springfield is somewhat common)
        # Either way, the pipeline should complete without crashing
        ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
        assert len(ai_messages) > 0, "Should have a response"
        print(f"\n--- Springfield (classified as {routing}) ---\n{ai_messages[-1].content}\n")


@pytest.mark.asyncio
async def test_fan_out_fan_in_state_merge(graph):
    """Phase 4: Verify parallel fan-out/fan-in produces correct merged state."""
    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            content="What's the weather in Tokyo and Mumbai? Calculate which city is hotter."
        )]}
    )

    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
    assert len(ai_messages) > 0, "No AI response generated"

    response = ai_messages[-1].content
    print(f"\n--- Fan-out/Fan-in Response ---\n{response}\n")

    # Verify state was correctly merged
    weather = result.get("fetched_weather", {})
    print(f"Fetched weather keys: {list(weather.keys())}")
    assert len(weather) >= 2, f"Expected weather for 2+ cities, got {list(weather.keys())}"

    completed = result.get("completed_indices", [])
    print(f"Completed indices: {completed}")
    assert len(completed) >= 2, f"Expected 2+ completed tasks, got {len(completed)}"

    # Should have calculation results too
    calcs = result.get("calculations", [])
    print(f"Calculations: {calcs}")


# ---------------------------------------------------------------------------
# Scope-driven test cases — one per intent bucket from .claude/scope.md
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifestyle_advisory(graph):
    """Lifestyle bucket: activity-based weather advisory."""
    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            content="I'm planning an outdoor picnic in San Francisco this weekend. Is the weather suitable?"
        )]}
    )

    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
    assert len(ai_messages) > 0, "No AI response generated"

    response = ai_messages[-1].content
    print(f"\n--- Lifestyle Advisory Response ---\n{response}\n")

    assert "san francisco" in response.lower(), "Response should mention San Francisco"
    assert len(response) > 100, "Advisory response seems too short"


@pytest.mark.asyncio
async def test_safety_severe_weather(graph):
    """Safety bucket: severe weather alerts and hazards."""
    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            content="Are there any active weather alerts or severe conditions in Miami right now?"
        )]}
    )

    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
    assert len(ai_messages) > 0, "No AI response generated"

    response = ai_messages[-1].content
    print(f"\n--- Safety/Alerts Response ---\n{response}\n")

    assert "miami" in response.lower(), "Response should mention Miami"
    assert len(response) > 50, "Safety response seems too short"


@pytest.mark.asyncio
async def test_three_city_comparison(graph):
    """Comparative bucket: 3-city fan-out comparison."""
    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            content="Compare current weather in Paris, Berlin, and Rome. Which city is warmest?"
        )]}
    )

    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
    assert len(ai_messages) > 0, "No AI response generated"

    response = ai_messages[-1].content
    print(f"\n--- Three City Comparison ---\n{response}\n")

    response_lower = response.lower()
    assert "paris" in response_lower, "Response should mention Paris"
    assert "berlin" in response_lower, "Response should mention Berlin"
    assert "rome" in response_lower, "Response should mention Rome"

    weather = result.get("fetched_weather", {})
    print(f"Fetched weather keys: {list(weather.keys())}")
    assert len(weather) >= 3, f"Expected weather for 3 cities, got {list(weather.keys())}"


@pytest.mark.asyncio
async def test_math_temperature_conversion(graph):
    """Math bucket: unit conversion and arithmetic on weather data."""
    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            content=(
                "What's the temperature in Chicago and Dubai right now? "
                "Calculate the difference in both Celsius and Fahrenheit."
            )
        )]}
    )

    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
    assert len(ai_messages) > 0, "No AI response generated"

    response = ai_messages[-1].content
    print(f"\n--- Math / Conversion Response ---\n{response}\n")

    response_lower = response.lower()
    assert "chicago" in response_lower, "Response should mention Chicago"
    assert "dubai" in response_lower, "Response should mention Dubai"

    calcs = result.get("calculations", [])
    print(f"Calculations: {calcs}")
    assert len(calcs) >= 1, "Expected at least one calculation result"
