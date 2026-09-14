"""Retrieval quality against the current retrieve() path. No new features."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from memory.consolidation import EpisodeRecord
from memory.retrieval import _rule_route


AGENT_ID = "hr_assistant"


def _blob(context) -> str:
    parts = []
    for fact in context.get("semantic") or []:
        parts.append(getattr(fact, "content", "") or "")
    for item in context.get("procedural") or []:
        parts.append(str(item.get("title", "")))
        parts.append(str(item.get("content", "")))
    for item in context.get("episodic") or []:
        parts.append(str(item.get("summary", "")))
        parts.append(str(item.get("goal", "")))
    return " ".join(parts).lower()


def _add_user_fact(semantic, user_id, content, key, value, episode_id):
    return semantic.add_fact(
        content=content,
        confidence=0.95,
        source_episode_id=episode_id,
        trigger="test",
        status="current",
        user_id=user_id,
        scope=f"user:{user_id}",
        metadata={
            "category": "preference",
            "fact_key": key,
            "fact_value": value,
        },
    )


def _delete_user(semantic, user_id):
    try:
        semantic.mem0.delete_all(user_id=user_id)
    except Exception:
        pass


def _index_episode(episodic, episode):
    dest = Path(episodic.base_path) / f"{episode.episode_id}.json"
    dest.write_text(episode.model_dump_json(), encoding="utf-8")
    episodic.index_episode(episode)


@pytest.fixture
def quality_retriever(retriever):
    retriever.router.route = _rule_route
    return retriever


@pytest.fixture
def users(iso_users, semantic_memory):
    yield iso_users
    for user_id in iso_users.values():
        _delete_user(semantic_memory, user_id)


@pytest.mark.integration
class TestRetrievalQuality:
    def test_user_b_does_not_get_user_a_facts(
        self, quality_retriever, semantic_memory, users
    ):
        token = f"zebrastripe-{users['a']}"
        _add_user_fact(
            semantic_memory,
            users["a"],
            f"User UI theme preference is {token}",
            "ui_theme",
            token,
            "ep_quality_iso",
        )

        context = quality_retriever.retrieve(
            query="What is my UI theme preference?",
            user_id=users["b"],
            agent_id=AGENT_ID,
        )

        assert token.lower() not in _blob(context)

    def test_superseded_facts_do_not_come_back(
        self, quality_retriever, semantic_memory, users
    ):
        user_a = users["a"]
        old_token = f"oldtheme-{user_a}"
        new_token = f"newtheme-{user_a}"

        old = _add_user_fact(
            semantic_memory,
            user_a,
            f"User UI theme preference is {old_token}",
            "ui_theme",
            old_token,
            "ep_quality_old",
        )
        semantic_memory.supersede_fact(
            old_fact=old,
            new_content=f"User UI theme preference is {new_token}",
            scope_type="user",
            scope_id=user_a,
            source_episode_id="ep_quality_new",
            trigger="test",
            extra_metadata={"fact_key": "ui_theme", "fact_value": new_token},
        )

        context = quality_retriever.retrieve(
            query="What is my UI theme preference?",
            user_id=user_a,
            agent_id=AGENT_ID,
        )
        blob = _blob(context)

        assert old_token.lower() not in blob
        assert new_token.lower() in blob

    def test_same_key_update_keeps_only_new_value(
        self, quality_retriever, semantic_memory, users
    ):
        user_a = users["a"]
        old_value = f"slack-{user_a}"
        new_value = f"teams-{user_a}"

        old = _add_user_fact(
            semantic_memory,
            user_a,
            f"User prefers {old_value} notifications",
            "notify_channel",
            old_value,
            "ep_quality_notify_old",
        )
        semantic_memory.supersede_fact(
            old_fact=old,
            new_content=f"User prefers {new_value} notifications",
            scope_type="user",
            scope_id=user_a,
            source_episode_id="ep_quality_notify_new",
            trigger="test",
            extra_metadata={"fact_key": "notify_channel", "fact_value": new_value},
        )

        facts = semantic_memory.get_all_facts(user_id=user_a, status="current")
        notify = [
            f for f in facts
            if f.metadata.get("fact_key") == "notify_channel"
            or new_value in f.content
            or old_value in f.content
        ]
        contents = " ".join(f.content for f in notify)

        assert old_value not in contents
        assert new_value in contents
        assert sum(1 for f in notify if f.status == "current") == 1

    def test_how_do_i_hits_procedural_what_is_hits_semantic(
        self, quality_retriever, semantic_memory, procedural_memory, users
    ):
        user_a = users["a"]
        theme = f"coraltheme-{user_a}"
        _add_user_fact(
            semantic_memory,
            user_a,
            f"User UI theme preference is {theme}",
            "ui_theme",
            theme,
            "ep_quality_route",
        )
        procedural_memory.add_skill(
            title="Calculate PTO balance",
            content="PTO = (months_employed x 1.5) - days_used",
            description="How to calculate remaining PTO days",
            scope="shared:organization",
            status="approved",
            source_episode_ids=["ep_quality_pto"],
            confidence=0.9,
            trigger="test",
        )

        how = _rule_route("How do I calculate my PTO balance?")
        what = _rule_route("What is my UI theme?")
        assert how["primary"] == "procedural"
        assert what["primary"] == "semantic"

        how_ctx = quality_retriever.retrieve(
            query="How do I calculate my PTO balance?",
            user_id=user_a,
            agent_id=AGENT_ID,
        )
        what_ctx = quality_retriever.retrieve(
            query="What is my UI theme?",
            user_id=user_a,
            agent_id=AGENT_ID,
        )

        assert how_ctx["procedural"], "how-do-I should retrieve the PTO skill"
        assert any("1.5" in str(item.get("content", "")) for item in how_ctx["procedural"])
        assert theme.lower() in _blob(what_ctx)

    def test_new_thread_does_not_invent_old_episodic_detail(
        self,
        quality_retriever,
        event_logger,
        thread_manager,
        episodic_memory,
        users,
    ):
        user_a = users["a"]
        marker = f"Quetzalcoatl-{user_a}"

        old_thread = thread_manager.create_thread(
            user_id=user_a,
            agent_id=AGENT_ID,
            initial_message="Secret project",
        )
        event_logger.log_event({
            "event_type": "user_message",
            "user_id": user_a,
            "session_id": f"{user_a}_old",
            "thread_id": old_thread.thread_id,
            "agent_id": AGENT_ID,
            "payload": {"content": f"I am working on Project {marker}"},
        })

        episode = EpisodeRecord(
            episode_id=f"ep_{user_a}_phoenix",
            session_id=f"{user_a}_old",
            user_id=user_a,
            agent_id=AGENT_ID,
            timestamp=datetime.now(timezone.utc).isoformat(),
            goal=f"Discuss Project {marker}",
            outcome="success",
            summary=f"User described secret internal tool Project {marker}",
            trajectory=["mentioned project"],
            lessons=[],
            entities=[marker],
            source_event_ids=[],
            retention_score=0.9,
            scope=f"user:{user_a}",
            thread_id=old_thread.thread_id,
        )
        _index_episode(episodic_memory, episode)

        new_thread = thread_manager.create_thread(
            user_id=user_a,
            agent_id=AGENT_ID,
            initial_message="PTO question",
        )
        new_events = event_logger.get_events_for_thread(new_thread.thread_id)
        assert not any(marker in str(event.payload) for event in new_events)

        context = quality_retriever.retrieve(
            query="How do I request PTO?",
            user_id=user_a,
            agent_id=AGENT_ID,
        )
        assert marker.lower() not in _blob(context)
