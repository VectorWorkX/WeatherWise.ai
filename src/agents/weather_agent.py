"""Weather agent using Xweather MCP server and LangGraph.

This module provides a LangGraph-based weather agent that connects to the
Xweather MCP server for real-time weather data.

Architecture:
- Classifier: LLM-based query classification using structured output
- Router: Routes to appropriate processing path based on classification
- Tools: Executes Xweather MCP tools via ToolNode
- Formatter: Returns structured data for simple queries
- Synthesizer: Generates detailed analysis for complex queries

Future: RAG integration for document retrieval on analysis queries.

Requires Python 3.10+ and langchain-mcp-adapters package.
"""

import asyncio
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.state import AgentState, QueryClassification, WeatherConfig


# System prompt for query classification
CLASSIFIER_SYSTEM_PROMPT = """You are a query classifier for a weather assistant system.

Classify the user's LATEST query along TWO dimensions:

## 1. Query Type (simple vs complex)

**simple**: Straightforward queries that just need data presented clearly.
- "What's the weather in Tokyo?"
- "Current temperature in NYC"
- "Will it rain tomorrow?"
- "Hello" / "Thanks" / "What can you do?"
- User wants a direct, concise answer.

**complex**: Queries requiring interpretation, comparison, reasoning, or professional analysis.
- "Why is it so hot today?"
- "Compare weather in Paris vs Berlin"
- "What's the best day for outdoor activities this week?"
- "How will the weather impact my flight?"
- "Analyze the storm system approaching"
- User wants understanding, insights, or recommendations.

## 2. Requires Tools (true vs false)

**true**: Query needs weather data from MCP tools to answer.
- Any question about actual weather conditions, forecasts, AQI, etc.

**false**: Query can be answered without fetching weather data.
- Greetings: "Hello", "Hi", "Thanks"
- Meta questions: "What can you do?", "Who are you?"
- Off-topic: "What time is it?", "Tell me a joke"

Be precise. Most weather questions are simple + requires_tools=true."""

# System prompt for tool execution (ReAct agent)
TOOL_EXECUTOR_PROMPT = """You are a weather data assistant using a ReAct (Reasoning + Acting) approach.

Your job is to gather weather data by calling tools. You can make multiple rounds of tool calls.

INSTRUCTIONS:
1. Look at what data you already have (from previous tool results in the conversation)
2. Determine what data you still need
3. If you need more data: call the appropriate tool(s)
4. If you have ALL the data needed: respond with a brief summary like "I have gathered the weather data."

TIPS:
- Call tools for EACH location the user asks about
- You can call multiple tools in one turn (e.g., weather for NYC AND LA)
- Check tool results before deciding if you need more calls
- When done gathering data, just say you're done - don't format the final answer

Example flow:
User: "Compare NYC and LA weather"
Turn 1: Call get_current_weather for NYC, call get_current_weather for LA
Turn 2: [See results] "I have gathered weather data for both NYC and LA." """

# System prompt for simple responses (formatter handles both data and no-tool queries)
FORMATTER_SYSTEM_PROMPT = """You are WeatherWise, a friendly weather assistant.

## For weather queries (with tool data):
- ONLY include information that directly answers what the user asked
- If they asked about temperature, just give temperature
- If they asked about rain, just give precipitation info
- If they asked a general "what's the weather" question, give a brief overview
- Use bullet points or a simple format
- Include units (°F/°C, mph/kph)
- Keep it concise - users want quick facts, not essays

## For non-weather queries (no tool data available):
- Greetings: Respond warmly, offer to help with weather questions
- Meta questions ("What can you do?"): Briefly explain you provide weather info worldwide
- Off-topic: Politely explain you're a weather assistant and suggest a weather question
- Keep responses to 1-2 sentences

Example - Weather:
**Tokyo, Japan**: 45°F (7°C), Partly Cloudy

Example - Greeting:
Hello! I'm WeatherWise - ask me about weather anywhere in the world!"""

# System prompt for deep analysis (complex queries)
SYNTHESIZER_SYSTEM_PROMPT = """You are a Senior Meteorological Research Lead.

Provide a comprehensive analysis based on the weather data gathered:

## Summary
1-2 sentence executive summary.

## Analysis
- Detailed breakdown with scientific terminology
- Comparisons if multiple locations
- Causal explanations where relevant

## Recommendations
Actionable insights based on the analysis.

Use precise measurements and maintain a professional tone."""


class MCPConnectionError(Exception):
    """Raised when connection to the MCP server fails."""

    pass


class WeatherAgent:
    """LangGraph-based weather agent using Xweather MCP tools.

    Architecture (simple/complex × requires_tools):

    ```
                           ┌─→ tools → formatter → END (simple + tools)
    classifier → router ───┤
      (simple    (needs    ├─→ tools → synthesizer → END (complex + tools)
       vs         tools?)  │
       complex)            ├─→ formatter → END (simple, no tools)
                           └─→ synthesizer → END (complex, no tools)
    ```

    Args:
        config: WeatherConfig with connection settings.

    Raises:
        MCPConnectionError: If connection to MCP server fails.
        ValueError: If required credentials are not provided.
    """


    def __init__(self, config: WeatherConfig | None = None):
        """Initialize the weather agent.

        Args:
            config: Optional WeatherConfig. If not provided, loads from
                environment variables via pydantic-settings.

        Raises:
            ValueError: If required environment variables are not set.
        """
        if config is None:
            config = WeatherConfig()

        self.config = config
        self._tools: list[BaseTool] = []
        self._graph: StateGraph | None = None
        self._compiled_graph: Any = None
        self._mcp_client: MultiServerMCPClient | None = None
        self._checkpointer: MemorySaver = MemorySaver()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the MCP client and build the graph.

        This method must be called before using the agent.

        Raises:
            MCPConnectionError: If connection to MCP server fails.
        """
        if self._initialized:
            return

        await self._initialize_tools()
        self._build_graph()
        self._initialized = True

    async def _initialize_tools(self) -> None:
        """Initialize MCP tools for weather operations.

        Raises:
            MCPConnectionError: If connection to MCP server fails.
        """
        try:
            if self.config.transport == "stdio":
                self._mcp_client = MultiServerMCPClient(
                    {
                        "xweather": {
                            "transport": "stdio",
                            "command": "npx",
                            "args": ["-y", "@xweather/mcp-server"],
                            "env": {
                                "XWEATHER_CLIENT_ID": self.config.xweather_client_id,
                                "XWEATHER_CLIENT_SECRET": self.config.xweather_client_secret,
                            },
                        }
                    }
                )
            else:
                self._mcp_client = MultiServerMCPClient(
                    {
                        "xweather": {
                            "transport": "streamable_http",
                            "url": self.config.mcp_endpoint,
                            "headers": {
                                "Authorization": f"Bearer {self.config.api_key}"
                            },
                        }
                    }
                )

            # Get all tools from MCP server - don't filter
            self._tools = await self._mcp_client.get_tools()

            if not self._tools:
                raise MCPConnectionError(
                    "No tools available from MCP server. "
                    "Check your credentials and connection."
                )

        except MCPConnectionError:
            raise
        except Exception as e:
            raise MCPConnectionError(
                f"Failed to connect to Xweather MCP server: {e}"
            ) from e

    def _build_graph(self) -> None:
        """Build the LLM-routed workflow with ReAct loop for tool execution.

        Classification dimensions:
        1. query_type: 'simple' (format data) vs 'complex' (analyze data)
        2. requires_tools: True (need MCP tools) vs False (can answer directly)

        Graph Architecture with ReAct Loop:

        classifier ─────────────────────────────────────┐
            │                                           │
            │ (requires_tools=True)                     │ (requires_tools=False)
            ▼                                           │
        tool_caller ◄───────────────┐                   │
            │                       │                   │
            │ (has tool calls)      │ (loop)            │
            ▼                       │                   │
          tools ────────────────────┘                   │
            │                                           │
            │ (no more tool calls)                      │
            ▼                                           ▼
        formatter/synthesizer ◄─────────────────────────┘
            │
            ▼
           END

        The ReAct loop allows the agent to make multiple rounds of tool calls
        until it has gathered all necessary data.
        """
        # Classifier LLM with structured output
        classifier_llm = ChatAnthropic(
            model="claude-3-5-haiku-latest",  # Fast, cheap for classification
            temperature=0,
        )
        classifier_with_schema = classifier_llm.with_structured_output(
            QueryClassification
        )

        # Tool caller LLM (lightweight, just for tool selection)
        tool_caller_llm = ChatAnthropic(
            model="claude-3-5-haiku-latest",
            temperature=0,
        )
        tool_caller_with_tools = tool_caller_llm.bind_tools(self._tools)

        # Formatter LLM for structured data output
        formatter_llm = ChatAnthropic(
            model="claude-3-5-haiku-latest",
            temperature=0,
        )

        # Synthesizer LLM for deep analysis
        synthesizer_llm = ChatAnthropic(
            model=self.config.synthesizer_model,
            temperature=0.3,
        )

        def classifier_node(state: AgentState) -> dict:
            """Classify the user query using structured output.

            Uses LLM with Pydantic schema to determine query type.
            Includes full conversation history for context on follow-up questions.

            Args:
                state: Current agent state with messages.

            Returns:
                Updated state with query classification.
            """
            messages = state["messages"]

            # Build conversation context for classification
            # Include history so follow-ups like "what about AQI?" are understood
            conversation_context = []
            for msg in messages:
                # Handle content that may be string or list
                raw_content = msg.content
                if isinstance(raw_content, list):
                    # Join list parts into a string
                    raw_content = " ".join(
                        part.get("text", str(part)) if isinstance(part, dict) else str(part)
                        for part in raw_content
                    )
                content_str = str(raw_content)

                if isinstance(msg, HumanMessage):
                    conversation_context.append(f"User: {content_str}")
                elif isinstance(msg, AIMessage) and content_str:
                    # Include AI responses for context but keep it brief
                    brief = content_str[:200] + "..." if len(content_str) > 200 else content_str
                    conversation_context.append(f"Assistant: {brief}")

            # Use full conversation context - MemorySaver manages the history
            # For very long conversations, consider truncating to last N exchanges
            context_str = "\n".join(conversation_context)

            # Classify using structured output with full context
            classification_prompt = [
                SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(content=f"Conversation context:\n{context_str}\n\nClassify the LATEST user query in this conversation."),
            ]

            classification: QueryClassification = classifier_with_schema.invoke(
                classification_prompt
            )

            return {
                "query_type": classification.query_type,
                "requires_tools": classification.requires_tools,
            }

        def tool_caller_node(state: AgentState) -> dict:
            """Call appropriate weather tools based on the query.

            Lightweight node that just selects and calls tools.

            Args:
                state: Current agent state.

            Returns:
                State with tool call messages.
            """
            messages = state["messages"]

            system_msg = SystemMessage(content=TOOL_EXECUTOR_PROMPT)
            messages_with_system = [system_msg] + list(messages)

            response = tool_caller_with_tools.invoke(messages_with_system)
            return {"messages": [response]}

        def formatter_node(state: AgentState) -> dict:
            """Handle simple queries - format weather data or respond to greetings.

            Used for all 'simple' query types, with or without tool data.

            Args:
                state: Current agent state with messages (and possibly tool results).

            Returns:
                Formatted response.
            """
            messages = state["messages"]

            system_msg = SystemMessage(content=FORMATTER_SYSTEM_PROMPT)
            format_prompt = HumanMessage(
                content="Respond to the user's query based on the conversation above."
            )

            format_messages = [system_msg] + list(messages) + [format_prompt]
            response = formatter_llm.invoke(format_messages)

            return {"messages": [response]}

        def synthesizer_node(state: AgentState) -> dict:
            """Generate comprehensive analysis for complex queries.

            For analysis queries - provides detailed interpretation.

            Args:
                state: Current agent state with tool results.

            Returns:
                Detailed analysis response.
            """
            messages = state["messages"]

            system_msg = SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT)
            synthesis_prompt = HumanMessage(
                content="Provide a comprehensive meteorological analysis based on the data above."
            )

            synthesis_messages = [system_msg] + list(messages) + [synthesis_prompt]
            response = synthesizer_llm.invoke(synthesis_messages)

            return {"messages": [response]}

        def route_after_classifier(
            state: AgentState,
        ) -> Literal["tool_caller", "formatter", "synthesizer"]:
            """Route based on LLM classification (simple/complex × requires_tools).

            Routing logic:
            - requires_tools=True  → tool_caller (then to formatter or synthesizer)
            - requires_tools=False, simple  → formatter (handles greetings, etc.)
            - requires_tools=False, complex → synthesizer (rare, but possible)

            Args:
                state: Current agent state with classification.

            Returns:
                Next node based on query_type and requires_tools.
            """
            query_type = state.get("query_type", "simple")
            requires_tools = state.get("requires_tools", True)

            if requires_tools:
                return "tool_caller"

            # No tools needed - route based on complexity
            if query_type == "complex":
                return "synthesizer"
            return "formatter"  # Simple queries without tools (greetings, etc.)

        def route_after_tool_caller(
            state: AgentState,
        ) -> Literal["tools", "formatter", "synthesizer"]:
            """Route after tool caller - execute tools or generate response.

            ReAct decision point:
            - If tool_caller made tool calls → execute them
            - If no tool calls → done gathering data, go to formatter/synthesizer

            Args:
                state: Current agent state.

            Returns:
                'tools' if tool calls present, else 'formatter'/'synthesizer'.
            """
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            # No more tool calls - route to response generation based on query type
            query_type = state.get("query_type", "simple")
            if query_type == "complex":
                return "synthesizer"
            return "formatter"

        def route_after_tools(state: AgentState) -> Literal["tool_caller"]:
            """Route after tool execution - always return to tool_caller (ReAct loop).

            The tool_caller will see the tool results and decide whether to:
            - Make more tool calls (continues the loop)
            - Generate a response (exits the loop via route_after_tool_caller)

            Args:
                state: Current agent state with tool results.

            Returns:
                Always 'tool_caller' to continue the ReAct loop.
            """
            return "tool_caller"

        # Create ToolNode with error handling
        tool_node = ToolNode(self._tools)

        async def safe_tools_node(state: AgentState) -> dict:
            """Execute tools with error handling (async for MCP tools).

            Catches tool errors and returns proper ToolMessage responses
            to keep conversation state valid.

            Args:
                state: Current agent state with tool calls.

            Returns:
                State with tool results or error messages.
            """
            try:
                # MCP tools require async invocation
                return await tool_node.ainvoke(state)
            except Exception as e:
                # Get the tool call IDs from the last message
                last_message = state["messages"][-1]
                error_messages = []

                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        # Create a proper ToolMessage with the error
                        error_msg = self._get_friendly_error_message(str(e)) or str(e)
                        error_messages.append(
                            ToolMessage(
                                content=f"Error: {error_msg}",
                                tool_call_id=tool_call["id"],
                            )
                        )

                return {"messages": error_messages} if error_messages else {"messages": []}

        # Build the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classifier", classifier_node)
        workflow.add_node("tool_caller", tool_caller_node)
        workflow.add_node("tools", safe_tools_node)
        workflow.add_node("formatter", formatter_node)
        workflow.add_node("synthesizer", synthesizer_node)

        # Set entry point
        workflow.set_entry_point("classifier")

        # Add conditional edges
        # Classifier routes based on: requires_tools × simple/complex
        workflow.add_conditional_edges(
            "classifier",
            route_after_classifier,
            {"tool_caller": "tool_caller", "formatter": "formatter", "synthesizer": "synthesizer"},
        )
        # Tool caller routes to tools (if has calls) or to response generation (if done)
        workflow.add_conditional_edges(
            "tool_caller",
            route_after_tool_caller,
            {"tools": "tools", "formatter": "formatter", "synthesizer": "synthesizer"},
        )
        # ReAct loop: tools always routes back to tool_caller
        workflow.add_conditional_edges(
            "tools",
            route_after_tools,
            {"tool_caller": "tool_caller"},
        )
        workflow.add_edge("formatter", END)
        workflow.add_edge("synthesizer", END)

        self._graph = workflow
        # Compile with recursion limit to prevent infinite ReAct loops
        self._compiled_graph = workflow.compile(
            checkpointer=self._checkpointer,
        )
        # Set recursion limit for ReAct loop (max tool call rounds)
        self._recursion_limit = 10

    @property
    def tools(self) -> list[BaseTool]:
        """Get the list of MCP tools.

        Returns:
            List of initialized LangChain tool instances.
        """
        return self._tools

    @property
    def graph(self) -> StateGraph | None:
        """Get the LangGraph workflow.

        Returns:
            The StateGraph instance or None if not initialized.
        """
        return self._graph

    def _check_initialized(self) -> None:
        """Check if the agent has been initialized.

        Raises:
            RuntimeError: If the agent has not been initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "Agent not initialized. Call 'await agent.initialize()' first."
            )

    def _get_friendly_error_message(self, error: str) -> str | None:
        """Convert tool errors into user-friendly messages.

        Args:
            error: The raw error message string.

        Returns:
            A friendly message string, or None if error should be re-raised.
        """
        error_lower = error.lower()

        # Invalid location
        if "invalid" in error_lower and "location" in error_lower:
            return (
                "I couldn't find that location. Please check the spelling "
                "or try a different city name (e.g., 'Buffalo, NY' or 'Tokyo, Japan')."
            )

        # Rate limiting
        if "rate limit" in error_lower or "too many requests" in error_lower:
            return "I'm getting too many requests right now. Please try again in a moment."

        # Authentication errors
        if "unauthorized" in error_lower or "401" in error_lower:
            return "There's an issue with the weather service authentication. Please try again later."

        # Generic tool errors
        if "error calling tool" in error_lower:
            return "I had trouble getting that weather data. Please try rephrasing your question."

        # Malformed conversation state (tool_use without tool_result)
        if "tool_use" in error_lower and "tool_result" in error_lower:
            return "Something went wrong with the previous request. Please try again or start a new session with /new."

        return None

    def invoke(self, query: str, thread_id: str | None = None) -> dict:
        """Process a weather-related query synchronously.

        Note: Uses asyncio.run internally since MCP tools require async execution.

        Args:
            query: Natural language weather query.
            thread_id: Optional thread ID for multi-turn conversation persistence.
                If provided, conversation history is maintained across calls.

        Returns:
            Agent response with weather information.

        Raises:
            MCPConnectionError: If MCP server connection fails during query.
            RuntimeError: If agent has not been initialized.
        """
        # MCP tools require async - wrap ainvoke for sync interface
        return asyncio.run(self.ainvoke(query, thread_id))

    async def ainvoke(self, query: str, thread_id: str | None = None) -> dict:
        """Process a weather-related query asynchronously.

        Args:
            query: Natural language weather query.
            thread_id: Optional thread ID for multi-turn conversation persistence.
                If provided, conversation history is maintained across calls.

        Returns:
            Agent response with weather information.

        Raises:
            MCPConnectionError: If MCP server connection fails during query.
            RuntimeError: If agent has not been initialized.
        """
        self._check_initialized()
        try:
            input_state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "query_type": "simple",  # Default, will be overwritten by classifier
                "requires_tools": True,
            }

            # Configure with thread_id for persistence and recursion limit for ReAct loop
            config: dict[str, Any] = {"recursion_limit": self._recursion_limit}
            if thread_id:
                config["configurable"] = {"thread_id": thread_id}

            result = await self._compiled_graph.ainvoke(input_state, config)
            return result
        except Exception as e:
            error_str = str(e).lower()
            if "connection" in error_str or "timeout" in error_str:
                raise MCPConnectionError(
                    f"Lost connection to Xweather MCP server: {e}"
                ) from e
            # Handle tool errors gracefully with friendly messages
            friendly_msg = self._get_friendly_error_message(str(e))
            if friendly_msg:
                return {
                    "messages": [AIMessage(content=friendly_msg)],
                    "query_type": "simple",
                    "requires_tools": False,
                }
            raise

    def get_conversation_history(self, thread_id: str) -> list:
        """Retrieve conversation history for a thread.

        Args:
            thread_id: The thread ID to retrieve history for.

        Returns:
            List of messages in the conversation.
        """
        self._check_initialized()
        config = {"configurable": {"thread_id": thread_id}}
        state = self._compiled_graph.get_state(config)
        if state and state.values:
            return state.values.get("messages", [])
        return []

    async def close(self) -> None:
        """Close the MCP client connection."""
        if self._mcp_client:
            if hasattr(self._mcp_client, "close"):
                await self._mcp_client.close()
            self._mcp_client = None
            self._initialized = False


async def create_weather_agent(config: WeatherConfig | None = None) -> WeatherAgent:
    """Factory function to create and initialize a WeatherAgent.

    Args:
        config: Optional WeatherConfig. If not provided, loads from
            environment variables.

    Returns:
        Configured and initialized WeatherAgent instance.

    Raises:
        ValueError: If required environment variables are not set.
        MCPConnectionError: If connection to MCP server fails.
    """
    agent = WeatherAgent(config)
    await agent.initialize()
    return agent


def create_weather_agent_sync(config: WeatherConfig | None = None) -> WeatherAgent:
    """Synchronous factory function to create and initialize a WeatherAgent.

    Args:
        config: Optional WeatherConfig. If not provided, loads from
            environment variables.

    Returns:
        Configured and initialized WeatherAgent instance.

    Raises:
        ValueError: If required environment variables are not set.
        MCPConnectionError: If connection to MCP server fails.
    """
    return asyncio.run(create_weather_agent(config))
