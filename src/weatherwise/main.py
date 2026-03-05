"""WeatherWise.ai — main entry point with interactive CLI loop."""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from weatherwise.graph import build_graph
from weatherwise.logging import setup_logging, get_logger
from weatherwise.tracing import setup_tracing


async def run_interactive(*, stream: bool = False, checkpointer: str = "memory") -> None:
    """Run the WeatherWise interactive CLI (async)."""
    logger = get_logger("weatherwise.main")

    app = build_graph(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("WeatherWise.ai — Expert Weather Analysis")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        logger.info("user_query", query=user_input, thread_id=thread_id)

        if stream:
            # Stream tokens as they arrive
            print("\nWeatherWise: ", end="", flush=True)
            async for event in app.astream_events(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    # Only stream from synthesizer node
                    tags = event.get("tags", [])
                    if "synthesizer" in str(tags) or event.get("name", "") == "ChatAnthropic":
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content") and chunk.content:
                            print(chunk.content, end="", flush=True)
            print("\n")
        else:
            result = await app.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "ai":
                    print(f"\nWeatherWise: {msg.content}\n")
                    break


def main() -> None:
    """Sync entry point — spins up the async event loop."""
    load_dotenv()

    # Enable LangSmith tracing if configured (ported from feature/claude-integration)
    tracing_info = setup_tracing()
    if tracing_info["tracing_enabled"] == "True":
        print(f"LangSmith tracing enabled (project: {tracing_info['project']})")

    parser = argparse.ArgumentParser(description="WeatherWise.ai CLI")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming token output"
    )
    parser.add_argument(
        "--checkpointer",
        choices=["memory", "sqlite", "none"],
        default="memory",
        help="Checkpointer backend (default: memory)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()
    os.environ["LOG_LEVEL"] = args.log_level
    setup_logging()

    asyncio.run(run_interactive(stream=args.stream, checkpointer=args.checkpointer))


if __name__ == "__main__":
    main()
