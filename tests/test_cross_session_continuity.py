"""Category 2: Cross-session thread continuity tests."""

from datetime import datetime, timedelta

import pytest

from tests.utils.test_helpers import (
    assert_contains,
    log_conversation,
)


@pytest.mark.slow
class TestCrossSessionContext:
    """Category 2: Thread resumption across sessions."""

    def test_2_1_failed_action_retry(self, agent, event_logger, test_thread, test_user):
        thread_id = test_thread.thread_id
        log_conversation(
            event_logger,
            thread_id,
            "sess_1",
            test_user["user_id"],
            test_user["agent_id"],
            turns=[
                ("Submit my PTO request for next week", "Created ticket #1234"),
                ("No, that's wrong", "I apologize for the confusion"),
            ],
        )
        response = agent.respond(
            user_message="Do it correctly",
            user_id=test_user["user_id"],
            session_id="sess_2",
            thread_id=thread_id,
        )
        assert_contains(response, ["PTO", "submit"])
        assert_contains(response, ["apolog", "sorry", "correct"])

    def test_2_2_information_continuation(
        self, agent, event_logger, test_thread, test_user
    ):
        thread_id = test_thread.thread_id
        log_conversation(
            event_logger,
            thread_id,
            "sess_1",
            test_user["user_id"],
            test_user["agent_id"],
            turns=[
                ("What benefits are available?", "We offer health insurance, 401k, PTO..."),
                ("Tell me more about the 401k", "Our 401k plan includes 5% matching..."),
            ],
        )
        response = agent.respond(
            user_message="What about health insurance?",
            user_id=test_user["user_id"],
            session_id="sess_2",
            thread_id=thread_id,
        )
        assert_contains(response, ["health", "insurance"])

    def test_2_3_correction_across_sessions(
        self, agent, event_logger, test_thread, test_user
    ):
        thread_id = test_thread.thread_id
        log_conversation(
            event_logger,
            thread_id,
            "sess_1",
            test_user["user_id"],
            test_user["agent_id"],
            turns=[("My manager is Bob", "Got it, your manager is Bob")],
        )
        response = agent.respond(
            user_message="Actually, my manager is Alice",
            user_id=test_user["user_id"],
            session_id="sess_2",
            thread_id=thread_id,
        )
        assert_contains(response, ["Alice", "manager"])
        assert_contains(response, ["Bob", "Alice", "chang"])

    def test_2_4_resumption_after_interruption(
        self, agent, event_logger, test_thread, test_user
    ):
        thread_id = test_thread.thread_id
        log_conversation(
            event_logger,
            thread_id,
            "sess_1",
            test_user["user_id"],
            test_user["agent_id"],
            turns=[
                ("Walk me through setting up my dev environment", "Step 1: Install Homebrew..."),
                ("Done", "Step 2: Install Git..."),
            ],
        )
        response = agent.respond(
            user_message="What's next?",
            user_id=test_user["user_id"],
            session_id="sess_2",
            thread_id=thread_id,
        )
        assert_contains(response, ["dev", "environment", "step"])
        assert_contains(response, ["3", "next", "install"])

    def test_4_4_stale_prior_context_over_4_hours(
        self, agent, event_logger, test_thread, test_user
    ):
        thread_id = test_thread.thread_id
        event_logger.log_event({
            "event_type": "user_message",
            "thread_id": thread_id,
            "session_id": "sess_1",
            "user_id": test_user["user_id"],
            "agent_id": test_user["agent_id"],
            "payload": {"content": "I need help with GitHub"},
            "timestamp": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
        })
        response = agent.respond(
            user_message="How do I set up my account?",
            user_id=test_user["user_id"],
            session_id="sess_2",
            thread_id=thread_id,
        )
        assert response is not None
