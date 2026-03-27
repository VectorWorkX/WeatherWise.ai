"""Tests for shared utilities — trim_chat_history."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from weatherwise.utils import trim_chat_history


def test_empty_messages():
    assert trim_chat_history([]) == []


def test_short_conversation_kept():
    msgs = [
        HumanMessage(content="What's the weather in Tokyo?"),
        AIMessage(content="It's 22°C and partly cloudy."),
        HumanMessage(content="And London?"),
    ]
    result = trim_chat_history(msgs)
    assert len(result) == 3


def test_long_conversation_trimmed():
    """Verbose assistant responses should be trimmed to fit the token budget."""
    msgs = []
    for i in range(10):
        msgs.append(HumanMessage(content=f"Weather in city {i}?"))
        # ~800 tokens each (3200 chars / ~4 chars per token)
        msgs.append(AIMessage(content=f"Weather report for city {i}:\n" + "data " * 600))

    result = trim_chat_history(msgs, max_tokens=4000)
    assert len(result) < len(msgs)


def test_result_starts_on_human():
    msgs = [
        AIMessage(content="Previous response"),
        HumanMessage(content="Follow-up question"),
        AIMessage(content="Answer"),
        HumanMessage(content="Another question"),
    ]
    result = trim_chat_history(msgs)
    assert result[0].type == "human"


def test_system_messages_excluded():
    msgs = [
        SystemMessage(content="You are a weather assistant."),
        HumanMessage(content="Weather in NYC?"),
        AIMessage(content="It's 18°C."),
    ]
    result = trim_chat_history(msgs)
    assert all(m.type != "system" for m in result)


def test_custom_budget():
    msgs = [
        HumanMessage(content="First question " * 50),
        AIMessage(content="First answer " * 50),
        HumanMessage(content="Second question"),
    ]
    tight = trim_chat_history(msgs, max_tokens=100)
    loose = trim_chat_history(msgs, max_tokens=4000)
    assert len(tight) <= len(loose)
