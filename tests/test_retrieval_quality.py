"""Retrieval quality against the current retrieve() path. No new features."""

from datetime import datetime, timezone
from pathlib import Path
import uuid

import pytest

from agent.simple_agent import SimpleAgent
from memory.consolidation import EpisodeRecord
from memory.retrieval import _rule_route, apply_semantic_authority, scope_kind
from memory.semantic_memory import SemanticFact


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


def _add_org_fact(semantic, content, key, value, episode_id):
    return semantic.add_fact(
        content=content,
        confidence=0.95,
        source_episode_id=episode_id,
        trigger="test",
        status="current",
        scope_type="organization",
        scope_id="organization",
        metadata={
            "category": "policy",
            "fact_key": key,
            "fact_value": value,
        },
    )


def _delete_user(semantic, user_id):
    try:
        semantic.mem0.delete_all(user_id=user_id)
    except Exception:
        pass


def _delete_fact(semantic, fact):
    if fact is None:
        return
    try:
        semantic.mem0.delete(fact.fact_id)
    except Exception:
        pass
    semantic._recent = [
        item for item in semantic._recent if item.fact_id != fact.fact_id
    ]


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

    def test_user_notify_pref_beats_org_same_key(
        self, quality_retriever, semantic_memory, users
    ):
        user_a = users["a"]
        mark = uuid.uuid4().hex[:10]
        slack = f"slackchan{mark}"
        email = f"emailchan{mark}"
        org_fact = None
        _add_user_fact(
            semantic_memory,
            user_a,
            f"User prefers {slack} notifications",
            "notify_channel",
            slack,
            "ep_auth_user_notify",
        )
        try:
            org_fact = _add_org_fact(
                semantic_memory,
                f"Organization HR updates go by {email}",
                "notify_channel",
                email,
                "ep_auth_org_notify",
            )
            context = quality_retriever.retrieve(
                query="How should you notify me?",
                user_id=user_a,
                agent_id=AGENT_ID,
            )
            blob = _blob(context)
            assert slack.lower() in blob
            assert email.lower() not in blob
        finally:
            _delete_fact(semantic_memory, org_fact)

    def test_unrelated_org_policy_not_returned(
        self, quality_retriever, semantic_memory, users
    ):
        user_a = users["a"]
        mark = uuid.uuid4().hex[:10]
        theme = f"coraltheme{mark}"
        notice = f"noticedays{mark}"
        org_fact = None
        _add_user_fact(
            semantic_memory,
            user_a,
            f"User UI theme preference is {theme}",
            "ui_theme",
            theme,
            "ep_auth_theme",
        )
        try:
            org_fact = _add_org_fact(
                semantic_memory,
                f"Employees must give {notice} leave notice",
                "leave_notice",
                notice,
                "ep_auth_leave",
            )
            context = quality_retriever.retrieve(
                query="UI theme preference and leave notice",
                user_id=user_a,
                agent_id=AGENT_ID,
            )
            facts = context.get("semantic") or []
            blob = _blob(context)
            assert theme.lower() in blob
            assert notice.lower() not in blob
            assert "shared:organization" not in [
                getattr(fact, "scope", "") for fact in facts
            ]
        finally:
            _delete_fact(semantic_memory, org_fact)


def _authority_fact(fact_id, content, scope, key="", score=0.5, user_id="alice"):
    return SemanticFact(
        fact_id=fact_id,
        content=content,
        user_id=user_id,
        scope=scope,
        confidence=0.9,
        metadata={"fact_key": key, "_score": score},
    )


class TestSemanticAuthority:
    """In-memory authority rules. No Mem0."""

    def test_user_same_key_beats_higher_scoring_org(self):
        user = _authority_fact(
            "u1", "Notify this user via Slack", f"user:alice", "notify_channel", 0.4
        )
        org = _authority_fact(
            "o1",
            "HR updates go by email",
            "shared:organization",
            "notify_channel",
            0.99,
        )
        ranked = apply_semantic_authority([org, user], user_id="alice")
        assert [fact.fact_id for fact in ranked] == ["u1"]

    def test_different_keys_keep_both_user_first(self):
        user = _authority_fact(
            "u1", "UI theme is dark", f"user:alice", "ui_theme", 0.3
        )
        org = _authority_fact(
            "o1",
            "Employees must give 14 days leave notice",
            "shared:organization",
            "leave_notice",
            0.99,
        )
        ranked = apply_semantic_authority([org, user], user_id="alice")
        assert [fact.fact_id for fact in ranked] == ["u1", "o1"]

    def test_org_fact_naming_this_user_is_dropped(self):
        user = _authority_fact(
            "u1", "Notify this user via Slack", f"user:alice", "notify_channel", 0.5
        )
        org = _authority_fact(
            "o1",
            "alice organization sends HR updates by email",
            "shared:organization",
            "hr_channel",
            0.95,
        )
        ranked = apply_semantic_authority([org, user], user_id="alice")
        blob = " ".join(fact.content for fact in ranked).lower()
        assert "slack" in blob
        assert "email" not in blob

    def test_prompt_puts_user_before_agent_and_states_override(self):
        agent = SimpleAgent(
            agent_config={"agent_id": AGENT_ID, "role": "HR assistant"},
            retriever=object(),
        )
        prompt = agent._build_context_prompt({
            "semantic": [
                _authority_fact(
                    "a1",
                    "PTO formula is months x 1.5 minus used",
                    "agent:hr_assistant",
                    "pto_formula",
                    0.99,
                ),
                _authority_fact(
                    "u1",
                    "Notify this user via Slack",
                    "user:alice",
                    "notify_channel",
                    0.2,
                ),
            ]
        })
        assert "User-specific" in prompt
        assert "Agent-learned" in prompt
        assert prompt.index("User-specific") < prompt.index("Agent-learned")
        assert prompt.index("Slack") < prompt.index("PTO formula")
        assert "override" in prompt.lower()
        guidelines = agent._system_prompt()
        assert "user-specific memory conflicts" in guidelines.lower()

    def test_scope_kind(self):
        assert scope_kind("user:alice") == "user"
        assert scope_kind("agent:hr_assistant") == "agent"
        assert scope_kind("shared:organization") == "shared"
