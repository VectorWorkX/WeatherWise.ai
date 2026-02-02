"""Main entry point for WeatherWise.ai.

This module provides the CLI interface for interacting with the weather agent.
"""

import asyncio
import sys

from src.agents.weather_agent import (
    MCPConnectionError,
    WeatherAgent,
    create_weather_agent,
)


async def main_async() -> None:
    """Run the weather agent CLI asynchronously.

    Initializes the weather agent and processes user queries interactively.
    """
    print("WeatherWise.ai - Intelligent Weather Assistant")
    print("=" * 50)
    print("Connecting to Xweather MCP server...")

    try:
        agent = await create_weather_agent()
        print("Connected successfully!")
        print(f"Available tools: {[t.name for t in agent.tools]}\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set the XWEATHER_API_KEY environment variable.")
        sys.exit(1)
    except MCPConnectionError as e:
        print(f"Connection error: {e}")
        print("Please check your network connection and API credentials.")
        sys.exit(1)

    print("Enter your weather queries (type 'quit' to exit):\n")

    try:
        while True:
            try:
                query = input("You: ").strip()
                if not query:
                    continue
                if query.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                result = await agent.ainvoke(query)
                last_message = result["messages"][-1]
                print(f"\nAssistant: {last_message.content}\n")

            except MCPConnectionError as e:
                print(f"\nConnection error: {e}")
                print("Please check your network connection.\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}\n")
    finally:
        await agent.close()


def main() -> None:
    """Synchronous entry point for the CLI."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
