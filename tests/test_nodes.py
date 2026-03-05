"""Tests for individual nodes with mocked LLM/API calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    response = MagicMock()
    response.content = "CLEAR"
    return response


class TestSupervisor:
    @pytest.mark.asyncio
    async def test_clear_routing(self, mock_llm_response):
        mock_llm_response.content = "CLEAR"
        with patch(
            "weatherwise.nodes.supervisor.ChatAnthropic"
        ) as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_llm_response)

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

    @pytest.mark.asyncio
    async def test_ambiguous_routing(self, mock_llm_response):
        mock_llm_response.content = "AMBIGUOUS\nDid you mean Paris, France or Paris, Texas?"
        with patch(
            "weatherwise.nodes.supervisor.ChatAnthropic"
        ) as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_llm_response)

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
        with patch(
            "weatherwise.nodes.supervisor.ChatAnthropic"
        ) as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(side_effect=TimeoutError("timeout"))

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
    async def test_llm_calculation(self, mock_llm_response):
        mock_llm_response.content = "Temperature anomaly: 28.5 - 25.0 = +3.5C above normal"
        with patch(
            "weatherwise.nodes.calculator.ChatAnthropic"
        ) as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_llm_response)

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
