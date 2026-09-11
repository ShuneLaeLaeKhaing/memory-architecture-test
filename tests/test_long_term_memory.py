"""Category 3: Long-term memory integration with threads."""

import pytest

from tests.utils.test_helpers import assert_contains, log_conversation


@pytest.mark.slow
class TestLongTermMemory:
    """Category 3: Semantic and procedural memory integration."""

    def test_3_1_fact_recall_across_threads(
        self,
        agent,
        event_logger,
        thread_manager,
        semantic_memory,
        consolidation_engine,
        test_user,
    ):
        thread1 = thread_manager.create_thread(
            user_id=test_user["user_id"],
            agent_id=test_user["agent_id"],
            initial_message="Preference setting",
        )
        log_conversation(
            event_logger,
            thread1.thread_id,
            "sess_1",
            test_user["user_id"],
            test_user["agent_id"],
            turns=[
                (
                    "I prefer dark mode for all applications",
                    "Noted. I'll remember you prefer dark mode",
                )
            ],
        )
        episodes = consolidation_engine.consolidate_session(
            session_id="sess_1",
            user_id=test_user["user_id"],
        )
        assert len(episodes) > 0, "Episode should be created"

        facts = semantic_memory.get_all_facts(
            user_id=test_user["user_id"],
            status="current",
        )
        dark_mode_fact = next(
            (f for f in facts if "dark mode" in f.content.lower()),
            None,
        )
        assert dark_mode_fact is not None, "Dark mode preference should be stored"

        thread2 = thread_manager.create_thread(
            user_id=test_user["user_id"],
            agent_id=test_user["agent_id"],
            initial_message="IDE setup",
        )
        response = agent.respond(
            user_message="Set up my IDE",
            user_id=test_user["user_id"],
            session_id="sess_2",
            thread_id=thread2.thread_id,
        )
        assert_contains(response, ["dark"])

    def test_3_3_policy_reference(
        self,
        agent,
        event_logger,
        thread_manager,
        consolidation_engine,
        test_user,
    ):
        thread1 = thread_manager.create_thread(
            user_id=test_user["user_id"],
            agent_id=test_user["agent_id"],
            initial_message="Remote work policy",
        )
        log_conversation(
            event_logger,
            thread1.thread_id,
            "sess_1",
            test_user["user_id"],
            test_user["agent_id"],
            turns=[
                (
                    "What's the policy on remote work?",
                    "Company policy allows 2 days/week remote work",
                )
            ],
        )
        consolidation_engine.consolidate_session(
            session_id="sess_1",
            user_id=test_user["user_id"],
        )

        thread2 = thread_manager.create_thread(
            user_id="test_bob",
            agent_id=test_user["agent_id"],
            initial_message="Remote work question",
        )
        response = agent.respond(
            user_message="Can I work from home?",
            user_id="test_bob",
            session_id="sess_2",
            thread_id=thread2.thread_id,
        )
        assert_contains(response, ["2 days", "remote", "week"])


@pytest.mark.slow
class TestConsolidationIntegration:
    """Category 5: Consolidation and promotion integration."""

    def test_5_1_mid_conversation_consolidation(
        self, agent, event_logger, consolidation_engine, test_thread, test_user
    ):
        thread_id = test_thread.thread_id
        session_id = "sess_1"
        for i in range(8):
            log_conversation(
                event_logger,
                thread_id,
                session_id,
                test_user["user_id"],
                test_user["agent_id"],
                turns=[
                    (
                        f"Question {i + 1} about PTO",
                        f"Answer {i + 1} about PTO policy",
                    )
                ],
            )
        episodes = consolidation_engine.consolidate_session(
            session_id=session_id,
            user_id=test_user["user_id"],
        )
        assert len(episodes) > 0, "Episode should be created"

        response = agent.respond(
            user_message="One more question about PTO",
            user_id=test_user["user_id"],
            session_id=session_id,
            thread_id=thread_id,
        )
        assert response is not None
        assert_contains(response, ["PTO"])
