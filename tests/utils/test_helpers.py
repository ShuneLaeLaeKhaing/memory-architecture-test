"""Helper utilities for testing."""

from typing import Dict, List

from memory.event_capture import EventLogger


def log_conversation(
    event_logger: EventLogger,
    thread_id: str,
    session_id: str,
    user_id: str,
    agent_id: str,
    turns: List[tuple],
) -> None:
    """Log a conversation to EventLogger as (user_msg, agent_msg) turns."""
    for user_msg, agent_msg in turns:
        event_logger.log_event({
            "event_type": "user_message",
            "thread_id": thread_id,
            "session_id": session_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "payload": {"content": user_msg},
        })
        event_logger.log_event({
            "event_type": "agent_response",
            "thread_id": thread_id,
            "session_id": session_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "payload": {"content": agent_msg},
        })


def assert_contains(text: str, keywords: List[str], case_sensitive: bool = False) -> None:
    """Assert that text contains all keywords."""
    search_text = text if case_sensitive else text.lower()
    missing = []
    for keyword in keywords:
        search_keyword = keyword if case_sensitive else keyword.lower()
        if search_keyword not in search_text:
            missing.append(keyword)
    if missing:
        raise AssertionError(
            f"Missing keywords in text: {missing}\n"
            f"Text: {text[:200]}..."
        )


def assert_not_contains(text: str, keywords: List[str], case_sensitive: bool = False) -> None:
    """Assert that text does NOT contain any keywords."""
    search_text = text if case_sensitive else text.lower()
    found = []
    for keyword in keywords:
        search_keyword = keyword if case_sensitive else keyword.lower()
        if search_keyword in search_text:
            found.append(keyword)
    if found:
        raise AssertionError(
            f"Unexpected keywords found in text: {found}\n"
            f"Text: {text[:200]}..."
        )


def get_thread_messages(event_logger: EventLogger, thread_id: str) -> List[Dict]:
    """Get all messages for a thread in chronological order."""
    events = event_logger.get_events_for_thread(
        thread_id=thread_id,
        event_types=["user_message", "agent_response"],
    )
    return [
        {
            "role": "user" if event.event_type == "user_message" else "assistant",
            "content": event.payload.get("content", ""),
            "timestamp": event.timestamp,
        }
        for event in events
    ]


def count_messages(event_logger: EventLogger, thread_id: str) -> int:
    """Count total messages in a thread."""
    return len(get_thread_messages(event_logger, thread_id))
