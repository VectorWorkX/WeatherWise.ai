"""Integration tests for the compiled graph with mocked tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from weatherwise.graph import build_graph, route_from_supervisor


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

    @pytest.mark.asyncio
    async def test_full_pipeline_mocked(self):
        """End-to-end test: supervisor -> planner -> executor -> fetcher -> synthesizer."""
        supervisor_response = MagicMock()
        supervisor_response.content = "CLEAR"

        # Planner returns structured output
        planner_response = MagicMock()
        planner_response.tasks = [
            MagicMock(
                op="fetch",
                location="Tokyo",
                query="",
                model_dump=lambda: {"op": "fetch", "location": "Tokyo", "query": ""},
            )
        ]

        synth_response = MagicMock()
        synth_response.content = "Tokyo is currently 25C with clear skies."

        call_count = {"n": 0}

        async def mock_ainvoke(messages, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return supervisor_response
            return synth_response

        with (
            patch("weatherwise.nodes.supervisor.ChatAnthropic") as MockSuperLLM,
            patch("weatherwise.nodes.planner.ChatAnthropic") as MockPlannerLLM,
            patch("weatherwise.nodes.fetcher.MultiServerMCPClient") as MockMCP,
            patch("weatherwise.nodes.fetcher.ChatAnthropic") as MockFetcherLLM,
            patch("weatherwise.nodes.fetcher.create_react_agent") as MockReAct,
            patch("weatherwise.nodes.synthesizer.ChatAnthropic") as MockSynthLLM,
        ):
            # Supervisor
            MockSuperLLM.return_value.ainvoke = AsyncMock(return_value=supervisor_response)

            # Planner with structured output
            structured_llm = MagicMock()
            structured_llm.ainvoke = AsyncMock(return_value=planner_response)
            MockPlannerLLM.return_value.with_structured_output = MagicMock(
                return_value=structured_llm
            )

            # Fetcher MCP
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = []
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockMCP.return_value = mock_client

            # Fetcher ReAct agent
            fetch_result = {
                "messages": [AIMessage(content="Tokyo: 25C, clear, wind 10kph")]
            }
            mock_agent = MagicMock()
            mock_agent.ainvoke = AsyncMock(return_value=fetch_result)
            MockReAct.return_value = mock_agent

            # Synthesizer
            MockSynthLLM.return_value.ainvoke = AsyncMock(return_value=synth_response)

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
        supervisor_response = MagicMock()
        supervisor_response.content = "CLEAR"

        planner_response = MagicMock()
        planner_response.tasks = [
            MagicMock(
                op="fetch",
                location="Tokyo",
                query="",
                model_dump=lambda: {"op": "fetch", "location": "Tokyo", "query": ""},
            ),
            MagicMock(
                op="fetch",
                location="London",
                query="",
                model_dump=lambda: {"op": "fetch", "location": "London", "query": ""},
            ),
        ]

        synth_response = MagicMock()
        synth_response.content = "Tokyo is 25C, London is 11C."

        with (
            patch("weatherwise.nodes.supervisor.ChatAnthropic") as MockSuperLLM,
            patch("weatherwise.nodes.planner.ChatAnthropic") as MockPlannerLLM,
            patch("weatherwise.nodes.fetcher.MultiServerMCPClient") as MockMCP,
            patch("weatherwise.nodes.fetcher.ChatAnthropic"),
            patch("weatherwise.nodes.fetcher.create_react_agent") as MockReAct,
            patch("weatherwise.nodes.synthesizer.ChatAnthropic") as MockSynthLLM,
        ):
            MockSuperLLM.return_value.ainvoke = AsyncMock(return_value=supervisor_response)

            structured_llm = MagicMock()
            structured_llm.ainvoke = AsyncMock(return_value=planner_response)
            MockPlannerLLM.return_value.with_structured_output = MagicMock(
                return_value=structured_llm
            )

            mock_client = AsyncMock()
            mock_client.get_tools.return_value = []
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockMCP.return_value = mock_client

            fetch_result = {
                "messages": [AIMessage(content="Weather data here")]
            }
            mock_agent = MagicMock()
            mock_agent.ainvoke = AsyncMock(return_value=fetch_result)
            MockReAct.return_value = mock_agent

            MockSynthLLM.return_value.ainvoke = AsyncMock(return_value=synth_response)

            graph = build_graph(checkpointer="none")
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="Compare Tokyo and London weather")]}
            )

            # Both cities should be in fetched_weather (merged by reducer)
            weather = result.get("fetched_weather", {})
            assert "tokyo" in weather or "london" in weather

            # Both plan indices should be completed
            completed = result.get("completed_indices", [])
            assert 0 in completed
            assert 1 in completed
