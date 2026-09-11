"""Category 1: Within-session thread memory tests."""

import pytest

from tests.utils.test_helpers import (
    assert_contains,
    assert_not_contains,
    count_messages,
    log_conversation,
)


@pytest.mark.slow
class TestWithinSessionContext:
    """Category 1: Basic thread memory within same session."""

    def test_1_1_multi_turn_information_request(
        self, agent, event_logger, test_thread, test_session, test_user
    ):
        thread_id = test_thread.thread_id
        session_id = test_session

        response1 = agent.respond(
            user_message="What's the company's PTO policy?",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        event_logger.log_event({
            "event_type": "user_message",
            "thread_id": thread_id,
            "session_id": session_id,
            "user_id": test_user["user_id"],
            "agent_id": test_user["agent_id"],
            "payload": {"content": "What's the company's PTO policy?"},
        })
        event_logger.log_event({
            "event_type": "agent_response",
            "thread_id": thread_id,
            "session_id": session_id,
            "user_id": test_user["user_id"],
            "agent_id": test_user["agent_id"],
            "payload": {"content": response1},
        })
        assert_contains(response1, ["PTO", "policy"])

        response2 = agent.respond(
            user_message="How do I request it?",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        event_logger.log_event({
            "event_type": "user_message",
            "thread_id": thread_id,
            "session_id": session_id,
            "user_id": test_user["user_id"],
            "agent_id": test_user["agent_id"],
            "payload": {"content": "How do I request it?"},
        })
        event_logger.log_event({
            "event_type": "agent_response",
            "thread_id": thread_id,
            "session_id": session_id,
            "user_id": test_user["user_id"],
            "agent_id": test_user["agent_id"],
            "payload": {"content": response2},
        })
        assert_contains(response2, ["PTO", "request"])

        response3 = agent.respond(
            user_message="Who approves it?",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        event_logger.log_event({
            "event_type": "user_message",
            "thread_id": thread_id,
            "session_id": session_id,
            "user_id": test_user["user_id"],
            "agent_id": test_user["agent_id"],
            "payload": {"content": "Who approves it?"},
        })
        event_logger.log_event({
            "event_type": "agent_response",
            "thread_id": thread_id,
            "session_id": session_id,
            "user_id": test_user["user_id"],
            "agent_id": test_user["agent_id"],
            "payload": {"content": response3},
        })
        assert_contains(response3, ["PTO", "manager"])
        assert count_messages(event_logger, thread_id) == 6

    def test_1_2_clarification_chain(
        self, agent, event_logger, test_thread, test_session, test_user
    ):
        thread_id = test_thread.thread_id
        session_id = test_session

        response1 = agent.respond(
            user_message="I need help with GitHub",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        log_conversation(
            event_logger,
            thread_id,
            session_id,
            test_user["user_id"],
            test_user["agent_id"],
            turns=[("I need help with GitHub", response1)],
        )

        response2 = agent.respond(
            user_message="Setting up my account",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        log_conversation(
            event_logger,
            thread_id,
            session_id,
            test_user["user_id"],
            test_user["agent_id"],
            turns=[("Setting up my account", response2)],
        )
        assert_contains(response2, ["GitHub", "account"])

        response3 = agent.respond(
            user_message="What if I already have a personal account?",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        assert_contains(response3, ["GitHub", "work"])

    def test_1_3_pronoun_resolution(
        self, agent, event_logger, test_thread, test_session, test_user
    ):
        thread_id = test_thread.thread_id
        session_id = test_session

        response1 = agent.respond(
            user_message="Tell me about the PTO approval process",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        log_conversation(
            event_logger,
            thread_id,
            session_id,
            test_user["user_id"],
            test_user["agent_id"],
            turns=[("Tell me about the PTO approval process", response1)],
        )

        response2 = agent.respond(
            user_message="How long does it take?",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        log_conversation(
            event_logger,
            thread_id,
            session_id,
            test_user["user_id"],
            test_user["agent_id"],
            turns=[("How long does it take?", response2)],
        )
        assert_contains(response2, ["PTO", "approval"])
        assert_not_contains(response2, ["what takes", "which process"])

        response3 = agent.respond(
            user_message="Can I expedite that?",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        assert_contains(response3, ["PTO", "approval"])

    def test_1_4_preference_application(
        self, agent, event_logger, test_thread, test_session, test_user
    ):
        thread_id = test_thread.thread_id
        session_id = test_session

        response1 = agent.respond(
            user_message="I prefer meetings in the afternoon",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        log_conversation(
            event_logger,
            thread_id,
            session_id,
            test_user["user_id"],
            test_user["agent_id"],
            turns=[("I prefer meetings in the afternoon", response1)],
        )
        assert_contains(response1, ["afternoon", "prefer"])

        response2 = agent.respond(
            user_message="Schedule a 1:1 with my manager",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        assert_contains(response2, ["afternoon"])
        assert_not_contains(response2, ["morning", "9 AM", "10 AM"])


@pytest.mark.slow
class TestEdgeCases:
    """Category 4: Edge cases and error handling."""

    def test_4_1_no_prior_context_fresh_thread(
        self, agent, test_thread, test_session, test_user
    ):
        response = agent.respond(
            user_message="Hello",
            user_id=test_user["user_id"],
            session_id=test_session,
            thread_id=test_thread.thread_id,
        )
        assert response
        assert_contains(response, ["hello", "hi", "help"])

    def test_4_2_context_window_overflow(
        self, agent, event_logger, test_thread, test_session, test_user
    ):
        thread_id = test_thread.thread_id
        session_id = test_session
        for i in range(15):
            log_conversation(
                event_logger,
                thread_id,
                session_id,
                test_user["user_id"],
                test_user["agent_id"],
                turns=[
                    (
                        f"Question {i + 1}: Tell me something",
                        f"Answer {i + 1}: Here's the information",
                    )
                ],
            )
        assert count_messages(event_logger, thread_id) == 30

        response = agent.respond(
            user_message="What did we discuss?",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        assert response is not None

    def test_4_3_thread_switch_mid_conversation(
        self, agent, event_logger, thread_manager, test_user
    ):
        thread1 = thread_manager.create_thread(
            user_id=test_user["user_id"],
            agent_id=test_user["agent_id"],
            initial_message="PTO question",
        )
        response1 = agent.respond(
            user_message="Tell me about PTO",
            user_id=test_user["user_id"],
            session_id="sess_1",
            thread_id=thread1.thread_id,
        )
        log_conversation(
            event_logger,
            thread1.thread_id,
            "sess_1",
            test_user["user_id"],
            test_user["agent_id"],
            turns=[("Tell me about PTO", response1)],
        )

        thread2 = thread_manager.create_thread(
            user_id=test_user["user_id"],
            agent_id=test_user["agent_id"],
            initial_message="GitHub question",
        )
        response2 = agent.respond(
            user_message="What's the GitHub workflow?",
            user_id=test_user["user_id"],
            session_id="sess_2",
            thread_id=thread2.thread_id,
        )
        assert_not_contains(response2, ["PTO", "time off"])
        assert_contains(response2, ["GitHub"])
