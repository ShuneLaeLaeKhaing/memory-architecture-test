"""Adversarial cases: try to break isolation, promotion, and retrieve."""

from datetime import datetime, timezone

import pytest

from memory.consolidation import EpisodeRecord
from memory.promotion import Candidate
from memory.retrieval import _rule_route


AGENT_ID = "hr_assistant"


def _episode(user_id, outcome="success", **kwargs):
    return EpisodeRecord(
        episode_id=kwargs.get("episode_id", f"ep_{user_id}_{outcome}"),
        session_id=kwargs.get("session_id", f"{user_id}_s"),
        user_id=user_id,
        agent_id=kwargs.get("agent_id", AGENT_ID),
        timestamp=datetime.now(timezone.utc).isoformat(),
        goal=kwargs.get("goal", "set a preference"),
        outcome=outcome,
        summary=kwargs.get("summary", "User stated a preference"),
        trajectory=kwargs.get("trajectory", ["asked", "answered"]),
        lessons=kwargs.get("lessons", []),
        entities=[],
        source_event_ids=["evt_adv_1"],
        retention_score=0.9,
        scope=f"user:{user_id}",
    )


def _cand(user_id, key, value, fact_type="preference", confidence=0.95, **kwargs):
    return Candidate(
        content=kwargs.get("content") or f"{user_id} UI theme preference is {value}",
        fact_type=fact_type,
        key=key,
        value=value,
        scope_hint=kwargs.get("scope_hint", "user"),
        confidence=confidence,
        evidence="adversarial",
    )


def _blob(context) -> str:
    parts = [getattr(f, "content", "") or "" for f in context.get("semantic") or []]
    return " ".join(parts).lower()


def _fact_text(facts) -> str:
    return " ".join(f.content for f in facts)


def _delete_user(semantic, user_id):
    try:
        semantic.mem0.delete_all(user_id=user_id)
    except Exception:
        pass


@pytest.fixture
def quality_retriever(retriever):
    retriever.router.route = _rule_route
    return retriever


@pytest.fixture
def users(iso_users, semantic_memory):
    yield iso_users
    for user_id in iso_users.values():
        _delete_user(semantic_memory, user_id)


class TestAdversarialPolicyGates:
    """No Mem0 writes. Attack apply_policy / identity / outcome gates."""

    def test_failed_episode_does_not_call_extract(
        self, promotion_engine, semantic_memory, users, monkeypatch
    ):
        called = {"extract": False}

        def boom(episode):
            called["extract"] = True
            return []

        monkeypatch.setattr(promotion_engine, "extract_candidates", boom)
        promotion_engine._run_semantic_pipeline(
            _episode(users["a"], outcome="failed")
        )
        assert called["extract"] is False

    def test_abandoned_episode_does_not_promote(self, promotion_engine, monkeypatch):
        called = {"extract": False}

        def boom(episode):
            called["extract"] = True
            return []

        monkeypatch.setattr(promotion_engine, "extract_candidates", boom)
        promotion_engine._run_semantic_pipeline(
            _episode("iso_gate", outcome="abandoned")
        )
        assert called["extract"] is False

    def test_preference_identity_is_user_not_shared(self, promotion_engine, users):
        ep = _episode(users["a"])
        cand = _cand(users["a"], "ui_theme", "dark")
        scope_type, scope_id = promotion_engine.identity_for(cand, ep)
        assert scope_type == "user"
        assert scope_id == users["a"]

    def test_policy_type_goes_to_organization(self, promotion_engine, users):
        ep = _episode(users["a"])
        cand = _cand(
            users["a"],
            "remote_days",
            "2",
            fact_type="policy",
            content="Company policy requires 2 remote days",
        )
        scope_type, scope_id = promotion_engine.identity_for(cand, ep)
        assert scope_type == "organization"

    def test_low_confidence_discarded(self, promotion_engine, users):
        ep = _episode(users["a"])
        cand = _cand(users["a"], "ui_theme", "dark", confidence=0.4)
        assert promotion_engine.apply_policy(cand, [], ep).action == "discard"

    def test_transient_statement_discarded(self, promotion_engine, users):
        ep = _episode(users["a"])
        cand = _cand(
            users["a"],
            "setup",
            "workday",
            content=f"{users['a']} is currently setting up Workday",
        )
        assert promotion_engine.apply_policy(cand, [], ep).action == "discard"

    def test_generic_advice_discarded(self, promotion_engine, users):
        ep = _episode(users["a"])
        cand = _cand(
            users["a"],
            "advice",
            "best",
            content=f"{users['a']} should familiarize with best practice",
        )
        assert promotion_engine.apply_policy(cand, [], ep).action == "discard"

    def test_procedural_skips_failed_and_short_trajectory(
        self, promotion_engine, monkeypatch, users
    ):
        called = {"llm": False}

        def fake_llm():
            called["llm"] = True
            raise AssertionError("procedural LLM should not run")

        monkeypatch.setattr("memory.promotion.llm_client", fake_llm)

        promotion_engine._run_procedural_pipeline(
            _episode(
                users["a"],
                outcome="failed",
                trajectory=["clicked", "opened", "submitted", "configured"],
            )
        )
        promotion_engine._run_procedural_pipeline(
            _episode(users["a"], outcome="success", trajectory=["clicked"])
        )
        assert called["llm"] is False


@pytest.mark.integration
class TestAdversarialWrites:
    """Mem0 writes. Attack leak, wrong owner, leftover stale facts."""

    def test_user_b_cannot_retrieve_user_a_promoted_preference(
        self, promotion_engine, quality_retriever, semantic_memory, users
    ):
        user_a, user_b = users["a"], users["b"]
        token = f"advtheme-{user_a}"
        ep = _episode(user_a)
        cand = _cand(user_a, "ui_theme", token)
        decision = promotion_engine.apply_policy(cand, [], ep)
        assert decision.action == "promote"
        promotion_engine._execute_decision(decision, cand, "user", user_a, ep)

        context = quality_retriever.retrieve(
            query="What is my UI theme preference?",
            user_id=user_b,
            agent_id=AGENT_ID,
        )
        assert token.lower() not in _blob(context)

        a_facts = semantic_memory.get_all_facts(user_id=user_a, status="current")
        assert token in _fact_text(a_facts)

    def test_claim_about_other_user_stays_on_speaker(
        self, promotion_engine, quality_retriever, semantic_memory, users
    ):
        user_a, user_b = users["a"], users["b"]
        secret = f"secretpass-{user_a}"
        ep = _episode(user_a)
        cand = _cand(
            user_a,
            "other_password",
            secret,
            content=f"{user_a} claims {user_b} password is {secret}",
        )
        decision = promotion_engine.apply_policy(cand, [], ep)
        assert decision.action == "promote"
        scope_type, scope_id = promotion_engine.identity_for(cand, ep)
        assert scope_type == "user"
        assert scope_id == user_a
        promotion_engine._execute_decision(decision, cand, scope_type, scope_id, ep)

        bob = quality_retriever.retrieve(
            query="What is my password?",
            user_id=user_b,
            agent_id=AGENT_ID,
        )
        assert secret.lower() not in _blob(bob)

        bob_facts = semantic_memory.get_all_facts(user_id=user_b, status="current")
        assert secret not in _fact_text(bob_facts)

    def test_same_key_new_value_supersedes_old(
        self, promotion_engine, quality_retriever, semantic_memory, users
    ):
        user_a = users["a"]
        old_val = f"slack-{user_a}"
        new_val = f"teams-{user_a}"
        ep1 = _episode(user_a, episode_id=f"ep_{user_a}_old")
        first = _cand(
            user_a,
            "notify_channel",
            old_val,
            content=f"{user_a} notification preference is {old_val}",
        )
        d1 = promotion_engine.apply_policy(first, [], ep1)
        promotion_engine._execute_decision(d1, first, "user", user_a, ep1)

        existing = promotion_engine.find_existing("user", user_a, first)
        ep2 = _episode(user_a, episode_id=f"ep_{user_a}_new")
        second = _cand(
            user_a,
            "notify_channel",
            new_val,
            content=f"{user_a} notification preference is {new_val}",
        )
        d2 = promotion_engine.apply_policy(second, existing, ep2)
        assert d2.action == "supersede"
        promotion_engine._execute_decision(d2, second, "user", user_a, ep2)

        current = semantic_memory.get_all_facts(user_id=user_a, status="current")
        text = _fact_text(current)
        assert old_val not in text
        assert new_val in text

        context = quality_retriever.retrieve(
            query="What is my notification preference?",
            user_id=user_a,
            agent_id=AGENT_ID,
        )
        blob = _blob(context)
        assert old_val.lower() not in blob
        assert new_val.lower() in blob

    def test_same_key_same_value_merges(
        self, promotion_engine, semantic_memory, users
    ):
        user_a = users["a"]
        value = f"email-{user_a}"
        ep1 = _episode(user_a, episode_id=f"ep_{user_a}_m1")
        cand = _cand(
            user_a,
            "notify_channel",
            value,
            content=f"{user_a} notification preference is {value}",
        )
        d1 = promotion_engine.apply_policy(cand, [], ep1)
        promotion_engine._execute_decision(d1, cand, "user", user_a, ep1)

        existing = promotion_engine.find_existing("user", user_a, cand)
        ep2 = _episode(user_a, episode_id=f"ep_{user_a}_m2")
        d2 = promotion_engine.apply_policy(cand, existing, ep2)
        assert d2.action == "merge"
        promotion_engine._execute_decision(d2, cand, "user", user_a, ep2)

        current = [
            f for f in semantic_memory.get_all_facts(user_id=user_a, status="current")
            if value in f.content
        ]
        assert len(current) == 1
