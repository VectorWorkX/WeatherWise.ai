"""Tests for individual nodes with mocked LLM/API calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from weatherwise.nodes.supervisor import RouteDecision


class TestSupervisor:
    @pytest.mark.asyncio
    async def test_clear_routing(self):
        decision = RouteDecision(
            bucket="weather",
            reasoning="Clear weather query about Tokyo",
            needs_clarification=False,
        )
        with patch("weatherwise.nodes.supervisor._get_supervisor_llm") as mock_get_llm:
            structured_llm = MagicMock()
            structured_llm.ainvoke = AsyncMock(return_value=decision)
            mock_get_llm.return_value.with_structured_output.return_value = structured_llm

            from weatherwise.nodes.supervisor import supervisor

            state = {
                "messages": [HumanMessage(content="Weather in Tokyo")],
                "plan": [],
                "completed_indices": [],
                "fetched_weather": {},
                "retrieved_knowledge": [],
                "calculations": [],
                "routing": "clear",
            }
            result = await supervisor(state)
            assert result["routing"] == "clear"
            assert result["intent"] == "weather"

    @pytest.mark.asyncio
    async def test_ambiguous_routing(self):
        decision = RouteDecision(
            bucket="weather",
            reasoning="Paris is ambiguous — multiple cities named Paris",
            needs_clarification=True,
            response="Did you mean Paris, France or Paris, Texas?",
        )
        with patch("weatherwise.nodes.supervisor._get_supervisor_llm") as mock_get_llm:
            structured_llm = MagicMock()
            structured_llm.ainvoke = AsyncMock(return_value=decision)
            mock_get_llm.return_value.with_structured_output.return_value = structured_llm

            from weatherwise.nodes.supervisor import supervisor

            state = {
                "messages": [HumanMessage(content="Weather in Paris")],
                "plan": [],
                "completed_indices": [],
                "fetched_weather": {},
                "retrieved_knowledge": [],
                "calculations": [],
                "routing": "clear",
            }
            result = await supervisor(state)
            assert result["routing"] == "ambiguous"
            assert "Paris" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_error_defaults_to_clear(self):
        with patch("weatherwise.nodes.supervisor._get_supervisor_llm") as mock_get_llm:
            structured_llm = MagicMock()
            structured_llm.ainvoke = AsyncMock(side_effect=TimeoutError("timeout"))
            mock_get_llm.return_value.with_structured_output.return_value = structured_llm

            from weatherwise.nodes.supervisor import supervisor

            state = {
                "messages": [HumanMessage(content="Weather in Tokyo")],
                "plan": [],
                "completed_indices": [],
                "fetched_weather": {},
                "retrieved_knowledge": [],
                "calculations": [],
                "routing": "clear",
            }
            result = await supervisor(state)
            assert result["routing"] == "clear"


class TestRetriever:
    @pytest.mark.asyncio
    async def test_retriever_returns_results(self):
        with patch(
            "weatherwise.nodes.retriever.search_knowledge_base"
        ) as mock_search:
            mock_search.return_value = ["Result 1", "Result 2"]

            from weatherwise.nodes.retriever import retriever_agent

            state = {
                "plan_index": 0,
                "op": "research",
                "location": "",
                "query": "port closures",
            }
            result = await retriever_agent(state)
            assert result["retrieved_knowledge"] == ["Result 1", "Result 2"]
            assert result["completed_indices"] == [0]

    @pytest.mark.asyncio
    async def test_retriever_handles_error(self):
        with patch(
            "weatherwise.nodes.retriever.search_knowledge_base",
            side_effect=RuntimeError("DB down"),
        ):
            from weatherwise.nodes.retriever import retriever_agent

            state = {
                "plan_index": 1,
                "op": "research",
                "location": "",
                "query": "wind thresholds",
            }
            result = await retriever_agent(state)
            assert result["completed_indices"] == [1]
            assert "failed" in result["retrieved_knowledge"][0].lower()


class TestCalculator:
    @pytest.mark.asyncio
    async def test_pure_arithmetic(self):
        """LLM calls compute_expression tool, which runs pure Python."""
        # First LLM call: decides to use compute_expression tool
        tool_response = MagicMock()
        tool_response.tool_calls = [
            {"name": "compute_expression", "args": {"expression": "28.5 - 25.0"}, "id": "call_1"}
        ]
        tool_response.content = ""

        # Second LLM call: reads tool result, returns final text
        final_response = MagicMock()
        final_response.tool_calls = []
        final_response.content = "The result of 28.5 - 25.0 is 3.5"

        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(side_effect=[tool_response, final_response])

        with patch("weatherwise.nodes.calculator._get_calc_llm") as mock_get_llm:
            mock_get_llm.return_value.bind_tools.return_value = mock_bound_llm

            from weatherwise.nodes.calculator import calculator_agent

            state = {
                "plan_index": 2,
                "op": "calculate",
                "location": "",
                "query": "28.5 - 25.0",
                "fetched_weather": {},
            }
            result = await calculator_agent(state)
            assert result["completed_indices"] == [2]
            assert "3.5" in result["calculations"][0]

    @pytest.mark.asyncio
    async def test_llm_calculation(self):
        """LLM calls run_calculation tool for complex math."""
        tool_response = MagicMock()
        tool_response.tool_calls = [
            {
                "name": "run_calculation",
                "args": {"code": "result = 28.5 - 25.0", "data": "{}"},
                "id": "call_1",
            }
        ]
        tool_response.content = ""

        final_response = MagicMock()
        final_response.tool_calls = []
        final_response.content = "Temperature anomaly: 28.5 - 25.0 = +3.5C above normal"

        mock_bound_llm = MagicMock()
        mock_bound_llm.ainvoke = AsyncMock(side_effect=[tool_response, final_response])

        with patch("weatherwise.nodes.calculator._get_calc_llm") as mock_get_llm:
            mock_get_llm.return_value.bind_tools.return_value = mock_bound_llm

            from weatherwise.nodes.calculator import calculator_agent

            state = {
                "plan_index": 2,
                "op": "calculate",
                "location": "Santos",
                "query": "wind anomaly Santos",
                "fetched_weather": {
                    "santos": {"location": "Santos", "raw_response": "wind: 18kph"},
                },
            }
            result = await calculator_agent(state)
            assert result["completed_indices"] == [2]
            assert "anomaly" in result["calculations"][0].lower()
