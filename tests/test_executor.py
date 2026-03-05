"""Tests for the executor node — routing logic and safeguards."""

from weatherwise.nodes.executor import route_from_executor, MAX_EXECUTOR_ITERATIONS


class TestRouteFromExecutor:
    def _make_state(self, plan=None, completed=None, fetched_weather=None):
        return {
            "messages": [],
            "plan": plan or [],
            "completed_indices": completed or [],
            "fetched_weather": fetched_weather or {},
            "retrieved_knowledge": [],
            "calculations": [],
            "routing": "clear",
        }

    def test_empty_plan_routes_to_synthesizer(self):
        state = self._make_state(plan=[])
        assert route_from_executor(state) == "synthesizer"

    def test_single_fetch_sends_to_fetcher(self):
        state = self._make_state(
            plan=[{"op": "fetch", "location": "Tokyo", "query": ""}]
        )
        result = route_from_executor(state)
        assert len(result) == 1
        assert result[0].node == "fetcher_agent"

    def test_single_research_sends_to_retriever(self):
        state = self._make_state(
            plan=[{"op": "research", "location": "", "query": "port closures"}]
        )
        result = route_from_executor(state)
        assert len(result) == 1
        assert result[0].node == "retriever_agent"

    def test_phase1_before_phase2(self):
        """Fetch tasks should run before calculate tasks."""
        state = self._make_state(
            plan=[
                {"op": "fetch", "location": "Tokyo", "query": ""},
                {"op": "calculate", "location": "Tokyo", "query": "temp anomaly"},
            ]
        )
        result = route_from_executor(state)
        # Should only return phase 1 sends
        assert len(result) == 1
        assert result[0].node == "fetcher_agent"

    def test_phase2_after_phase1_complete(self):
        """Calculate tasks run after fetch tasks are completed."""
        state = self._make_state(
            plan=[
                {"op": "fetch", "location": "Tokyo", "query": ""},
                {"op": "calculate", "location": "Tokyo", "query": "temp anomaly"},
            ],
            completed=[0],
            fetched_weather={"tokyo": {"location": "Tokyo", "raw_response": "data"}},
        )
        result = route_from_executor(state)
        assert len(result) == 1
        assert result[0].node == "calculator_agent"

    def test_all_complete_routes_to_synthesizer(self):
        state = self._make_state(
            plan=[
                {"op": "fetch", "location": "Tokyo", "query": ""},
                {"op": "calculate", "location": "Tokyo", "query": "temp anomaly"},
            ],
            completed=[0, 1],
        )
        assert route_from_executor(state) == "synthesizer"

    def test_unknown_op_skipped(self):
        """Unrecognized ops should be skipped, not cause infinite loops."""
        state = self._make_state(
            plan=[{"op": "unknown_op", "location": "X", "query": ""}]
        )
        assert route_from_executor(state) == "synthesizer"

    def test_max_iterations_safeguard(self):
        """Should force synthesizer after MAX_EXECUTOR_ITERATIONS."""
        state = self._make_state(
            plan=[{"op": "fetch", "location": "A", "query": ""}],
            completed=list(range(MAX_EXECUTOR_ITERATIONS)),
        )
        assert route_from_executor(state) == "synthesizer"

    def test_parallel_fan_out(self):
        """Multiple fetch tasks should fan out in parallel."""
        state = self._make_state(
            plan=[
                {"op": "fetch", "location": "Tokyo", "query": ""},
                {"op": "fetch", "location": "London", "query": ""},
                {"op": "research", "location": "", "query": "wind patterns"},
            ]
        )
        result = route_from_executor(state)
        assert len(result) == 3
        nodes = {s.node for s in result}
        assert "fetcher_agent" in nodes
        assert "retriever_agent" in nodes

    def test_calculator_receives_fetched_weather(self):
        """Calculator payload should include fetched_weather."""
        weather = {"tokyo": {"location": "Tokyo", "raw_response": "25C"}}
        state = self._make_state(
            plan=[
                {"op": "fetch", "location": "Tokyo", "query": ""},
                {"op": "calculate", "location": "Tokyo", "query": "anomaly"},
            ],
            completed=[0],
            fetched_weather=weather,
        )
        result = route_from_executor(state)
        assert result[0].arg["fetched_weather"] == weather
