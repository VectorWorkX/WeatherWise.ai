"""Integration tests for the compiled graph with mocked tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from weatherwise.graph import build_graph, route_from_supervisor
from weatherwise.nodes.supervisor import RouteDecision
from weatherwise.nodes.planner import PlanTask, PlanOutput


class TestRouteFromSupervisor:
    def test_clear_routes_to_planner(self):
        state = {"routing": "clear"}
        assert route_from_supervisor(state) == "planner"

    def test_ambiguous_routes_to_end(self):
        from langgraph.graph import END

        state = {"routing": "ambiguous"}
        assert route_from_supervisor(state) == END

    def test_missing_routing_routes_to_planner(self):
        state = {}
        assert route_from_supervisor(state) == "planner"


class TestBuildGraph:
    def test_graph_compiles_no_checkpointer(self):
        graph = build_graph(checkpointer="none")
        assert graph is not None

    def test_graph_compiles_with_memory(self):
        graph = build_graph(checkpointer="memory")
        assert graph is not None

    def test_graph_backwards_compat_bool(self):
        graph = build_graph(checkpointer=False)
        assert graph is not None

    def _mock_supervisor(self, mock_get_llm):
        """Set up supervisor mock to return clear weather routing."""
        decision = RouteDecision(
            bucket="weather",
            reasoning="Clear weather query",
            needs_clarification=False,
        )
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=decision)
        mock_get_llm.return_value.with_structured_output.return_value = structured_llm

    def _mock_planner(self, mock_get_llm, tasks: list[PlanTask]):
        """Set up planner mock to return a plan with given tasks."""
        plan_output = PlanOutput(
            summary="Fetching weather data",
            tasks=tasks,
        )
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=plan_output)
        mock_get_llm.return_value.with_structured_output.return_value = structured_llm

    def _mock_synthesizer(self, mock_get_llm, content: str):
        """Set up synthesizer mock to return given content."""
        response = MagicMock()
        response.content = content
        mock_get_llm.return_value.ainvoke = AsyncMock(return_value=response)

    @pytest.mark.asyncio
    async def test_full_pipeline_mocked(self):
        """End-to-end test: supervisor -> planner -> executor -> fetcher -> synthesizer."""
        tasks = [
            PlanTask(op="fetch", location="Tokyo", query=""),
        ]

        with (
            patch("weatherwise.nodes.supervisor._get_supervisor_llm") as mock_super,
            patch("weatherwise.nodes.planner._get_planner_llm") as mock_planner,
            patch("weatherwise.nodes.fetcher.get_xweather_tools", new_callable=AsyncMock, return_value=[]),
            patch("weatherwise.nodes.fetcher._invoke_fetch_loop", new_callable=AsyncMock, return_value=("Tokyo: 25C, clear, wind 10kph", '{"weather":{"tempC":25}}')),
            patch("weatherwise.nodes.synthesizer._get_synthesizer_llm") as mock_synth,
        ):
            self._mock_supervisor(mock_super)
            self._mock_planner(mock_planner, tasks)
            self._mock_synthesizer(mock_synth, "Tokyo is currently 25C with clear skies.")

            graph = build_graph(checkpointer="none")
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="Weather in Tokyo")]}
            )

            messages = result.get("messages", [])
            ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
            assert len(ai_messages) > 0
            assert "Tokyo" in ai_messages[-1].content

    @pytest.mark.asyncio
    async def test_fan_out_fan_in_merges_state(self):
        """Verify that parallel fetchers correctly merge their results."""
        tasks = [
            PlanTask(op="fetch", location="Tokyo", query=""),
            PlanTask(op="fetch", location="London", query=""),
        ]

        with (
            patch("weatherwise.nodes.supervisor._get_supervisor_llm") as mock_super,
            patch("weatherwise.nodes.planner._get_planner_llm") as mock_planner,
            patch("weatherwise.nodes.fetcher.get_xweather_tools", new_callable=AsyncMock, return_value=[]),
            patch("weatherwise.nodes.fetcher._invoke_fetch_loop", new_callable=AsyncMock, return_value=("Weather data here", '{"weather":{"tempC":20}}')),
            patch("weatherwise.nodes.synthesizer._get_synthesizer_llm") as mock_synth,
        ):
            self._mock_supervisor(mock_super)
            self._mock_planner(mock_planner, tasks)
            self._mock_synthesizer(mock_synth, "Tokyo is 25C, London is 11C.")

            graph = build_graph(checkpointer="none")
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="Compare Tokyo and London weather")]}
            )

            # Both cities should be in fetched_weather (merged by reducer)
            weather = result.get("fetched_weather", {})
            assert len(weather) >= 2

            # Both plan indices should be completed
            completed = result.get("completed_indices", [])
            assert 0 in completed
            assert 1 in completed
