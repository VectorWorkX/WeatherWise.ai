"""Main entry point for WeatherWise.ai.

This module provides the CLI interface for interacting with the weather agent.
Supports continuous multi-turn conversations with session persistence.
"""

import asyncio
import sys
import uuid

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.agents.weather_agent import (
    MCPConnectionError,
    create_weather_agent,
)


WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                     WeatherWise.ai                           ║
║         Senior Meteorological Research Assistant             ║
╠══════════════════════════════════════════════════════════════╣
║  Powered by LangGraph + Xweather MCP + Claude                ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  /help     - Show this help message
  /new      - Start a new conversation session
  /history  - Show conversation history for current session
  /quit     - Exit the application (also: /exit, /q)

Tips:
  - Ask simple questions: "What's the weather in Tokyo?"
  - Request analysis: "Analyze weather impacts for farming in Iowa"
  - Follow-up questions remember context from the current session
"""


def generate_session_id() -> str:
    """Generate a unique session ID for conversation tracking.

    Returns:
        A unique session identifier string.
    """
    return f"session-{uuid.uuid4().hex[:8]}"


def format_response(content: str | list) -> str:
    """Format the response content for display.

    Args:
        content: Response content (string or list of content blocks).

    Returns:
        Formatted string for display.
    """
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return content


async def main_async() -> None:
    """Run the weather agent CLI asynchronously.

    Provides a continuous chat loop with multi-turn conversation support.
    """
    print(WELCOME_BANNER)
    print("Connecting to Xweather MCP server...")

    try:
        agent = await create_weather_agent()
        print(f"Connected successfully!")
        print(f"Available tools: {[t.name for t in agent.tools]}")
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        print("Please ensure XWEATHER_CLIENT_ID and XWEATHER_CLIENT_SECRET are set in .env")
        sys.exit(1)
    except MCPConnectionError as e:
        print(f"\nConnection error: {e}")
        print("Please check your network connection and API credentials.")
        sys.exit(1)

    # Initialize session
    session_id = generate_session_id()
    print(f"\nSession started: {session_id}")
    print("Type /help for available commands.\n")
    print("-" * 60)

    try:
        while True:
            try:
                query = input("\nYou: ").strip()

                if not query:
                    continue

                # Handle commands
                if query.startswith("/"):
                    cmd = query.lower()

                    if cmd in ("/quit", "/exit", "/q"):
                        print("\nThank you for using WeatherWise.ai. Goodbye!")
                        break

                    elif cmd == "/help":
                        print(HELP_TEXT)
                        continue

                    elif cmd == "/new":
                        session_id = generate_session_id()
                        print(f"\nNew session started: {session_id}")
                        print("Conversation history cleared.")
                        continue

                    elif cmd == "/history":
                        history = agent.get_conversation_history(session_id)
                        if not history:
                            print("\nNo conversation history in current session.")
                        else:
                            print(f"\nConversation history ({len(history)} messages):")
                            print("-" * 40)
                            for i, msg in enumerate(history, 1):
                                msg_type = type(msg).__name__.replace("Message", "")
                                content = format_response(msg.content)
                                preview = content[:100] + "..." if len(content) > 100 else content
                                print(f"  [{i}] {msg_type}: {preview}")
                        continue

                    else:
                        print(f"Unknown command: {query}")
                        print("Type /help for available commands.")
                        continue

                # Process weather query
                print("\nAnalyzing...")
                result = await agent.ainvoke(query, thread_id=session_id)

                # Get the last AI message
                last_message = result["messages"][-1]
                content = format_response(last_message.content)

                print(f"\n{'─' * 60}")
                print("Assistant:")
                print(f"{'─' * 60}")
                print(content)
                print(f"{'─' * 60}")

            except MCPConnectionError as e:
                print(f"\nConnection error: {e}")
                print("Please check your network connection.")
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
            except EOFError:
                print("\n\nThank you for using WeatherWise.ai. Goodbye!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")

    finally:
        await agent.close()


def main() -> None:
    """Synchronous entry point for the CLI."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
