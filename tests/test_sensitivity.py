"""Sensitivity: public user facts cross agents; agent_only stays with source agent."""

from datetime import datetime, timezone

import pytest

from memory.consolidation import EpisodeRecord
from memory.promotion import Candidate
from memory.retrieval import _rule_route
from memory.sensitivity import (
    SENSITIVITY_AGENT_ONLY,
    SENSITIVITY_PUBLIC,
    classify_sensitivity,
    filter_user_facts_for_agent,
    sensitivity_metadata,
    user_fact_visible_to_agent,
)
from memory.semantic_memory import SemanticFact


AGENT_HR = "hr_assistant"
AGENT_ONBOARD = "onboarding_assistant"


def _fact(content, sensitivity=SENSITIVITY_PUBLIC, source=AGENT_HR, user_id="alice"):
    accessible = "all" if sensitivity == SENSITIVITY_PUBLIC else [source]
    return SemanticFact(
        fact_id="f1",
        content=content,
        user_id=user_id,
        scope=f"user:{user_id}",
        confidence=0.95,
        metadata={
            "sensitivity": sensitivity,
            "source_agent_id": source,
            "accessible_by": accessible,
            "fact_key": "test_key",
        },
    )


class TestSensitivityRules:
    """No Mem0. Classification and retrieve filter only."""

    def test_salary_is_agent_only(self):
        assert classify_sensitivity(
            "User's salary is $120,000", AGENT_HR, use_llm=False
        ) == SENSITIVITY_AGENT_ONLY

    def test_dark_mode_is_public(self):
        assert classify_sensitivity(
            "User prefers dark mode for all interfaces",
            AGENT_ONBOARD,
            use_llm=False,
        ) == SENSITIVITY_PUBLIC

    def test_unknown_fails_closed_without_llm(self):
        assert classify_sensitivity(
            "User's favorite internal project codename is Orion",
            AGENT_HR,
            use_llm=False,
        ) == SENSITIVITY_AGENT_ONLY

    def test_agent_scoped_write_skips_sensitivity(self):
        assert sensitivity_metadata(
            "PTO formula is months x 1.5",
            scope_type="agent",
            agent_id=AGENT_HR,
            use_llm=False,
        ) == {}

    def test_onboarding_cannot_see_hr_salary(self):
        salary = _fact("User's salary is $120,000", SENSITIVITY_AGENT_ONLY)
        pref = _fact("User prefers Slack notifications", SENSITIVITY_PUBLIC)
        visible = filter_user_facts_for_agent([salary, pref], AGENT_ONBOARD)
        blob = " ".join(f.content for f in visible).lower()
        assert "120" not in blob
        assert "slack" in blob

    def test_hr_can_see_own_salary_and_public_pref(self):
        salary = _fact("User's salary is $120,000", SENSITIVITY_AGENT_ONLY)
        pref = _fact("User prefers Slack notifications", SENSITIVITY_PUBLIC)
        visible = filter_user_facts_for_agent([salary, pref], AGENT_HR)
        blob = " ".join(f.content for f in visible).lower()
        assert "120" in blob
        assert "slack" in blob

    def test_untagged_old_fact_stays_public(self):
        old = SemanticFact(
            fact_id="old",
            content="User prefers Slack notifications",
            user_id="alice",
            scope="user:alice",
            confidence=0.9,
            metadata={},
        )
        assert user_fact_visible_to_agent(old, AGENT_ONBOARD) is True

    def test_hr_canonical_id_can_see_hr_assistant_lock(self):
        salary = _fact(
            "User's salary is $120,000",
            SENSITIVITY_AGENT_ONLY,
            source="hr_assistant",
        )
        assert user_fact_visible_to_agent(salary, "hr_assistant_001") is True


class TestSensitivityPromotionMetadata:
    """No Mem0 write. Execute path metadata via classify rules."""

    def test_promote_salary_tags_agent_only(self, promotion_engine, iso_users):
        user_a = iso_users["a"]
        ep = EpisodeRecord(
            episode_id="ep_sens_salary",
            session_id="s1",
            user_id=user_a,
            agent_id=AGENT_HR,
            timestamp=datetime.now(timezone.utc).isoformat(),
            goal="discuss compensation",
            outcome="success",
            summary="User stated salary",
            trajectory=["asked", "answered"],
            lessons=[],
            entities=[],
            source_event_ids=["e1"],
            retention_score=0.9,
            scope=f"user:{user_a}",
        )
        cand = Candidate(
            content=f"{user_a} salary is $120000",
            fact_type="identity",
            key="salary",
            value="120000",
            scope_hint="user",
            confidence=0.95,
            evidence="test",
        )
        meta = promotion_engine._semantic_write_metadata(
            cand, ep, "user", "promote: test"
        )
        assert meta["sensitivity"] == SENSITIVITY_AGENT_ONLY
        assert meta["accessible_by"] == [AGENT_HR]
        assert meta["source_agent_id"] == AGENT_HR

    def test_promote_theme_tags_public(self, promotion_engine, iso_users):
        user_a = iso_users["a"]
        ep = EpisodeRecord(
            episode_id="ep_sens_theme",
            session_id="s1",
            user_id=user_a,
            agent_id=AGENT_ONBOARD,
            timestamp=datetime.now(timezone.utc).isoformat(),
            goal="set UI preference",
            outcome="success",
            summary="User prefers dark mode",
            trajectory=["asked", "answered"],
            lessons=[],
            entities=[],
            source_event_ids=["e1"],
            retention_score=0.9,
            scope=f"user:{user_a}",
        )
        cand = Candidate(
            content=f"{user_a} prefers dark mode",
            fact_type="preference",
            key="ui_theme",
            value="dark",
            scope_hint="user",
            confidence=0.95,
            evidence="test",
        )
        meta = promotion_engine._semantic_write_metadata(
            cand, ep, "user", "promote: test"
        )
        assert meta["sensitivity"] == SENSITIVITY_PUBLIC
        assert meta["accessible_by"] == "all"


@pytest.mark.integration
class TestSensitivityRetrieve:
    """Retrieve path: salary locked to HR, dark mode shared."""

    @pytest.fixture
    def quality_retriever(self, retriever):
        retriever.router.route = _rule_route
        return retriever

    def test_hr_salary_not_retrieved_by_onboarding(
        self, quality_retriever, semantic_memory, iso_users, monkeypatch
    ):
        user_a = iso_users["a"]
        salary_token = f"sal{user_a[-8:]}"
        theme_token = f"thm{user_a[-8:]}"
        salary_fact = SemanticFact(
            fact_id="mem_salary",
            content=f"User's salary is {salary_token}",
            user_id=user_a,
            scope=f"user:{user_a}",
            confidence=0.95,
            metadata={
                "sensitivity": SENSITIVITY_AGENT_ONLY,
                "source_agent_id": AGENT_HR,
                "accessible_by": [AGENT_HR],
                "fact_key": "salary",
                "_score": 0.9,
            },
        )
        theme_fact = SemanticFact(
            fact_id="mem_theme",
            content=f"User prefers dark mode {theme_token}",
            user_id=user_a,
            scope=f"user:{user_a}",
            confidence=0.95,
            metadata={
                "sensitivity": SENSITIVITY_PUBLIC,
                "source_agent_id": AGENT_ONBOARD,
                "accessible_by": "all",
                "fact_key": "ui_theme",
                "_score": 0.9,
            },
        )

        def fake_search(query, user_id=None, scopes=None, **kwargs):
            scopes = scopes or []
            if any(str(scope).startswith("user:") for scope in scopes):
                return [salary_fact, theme_fact]
            return []

        monkeypatch.setattr(semantic_memory, "search", fake_search)

        hr = quality_retriever.retrieve(
            query="What is my salary and UI preference?",
            user_id=user_a,
            agent_id=AGENT_HR,
        )
        onboard = quality_retriever.retrieve(
            query="What is my salary and UI preference?",
            user_id=user_a,
            agent_id=AGENT_ONBOARD,
        )
        hr_blob = " ".join(
            getattr(f, "content", "") for f in (hr.get("semantic") or [])
        ).lower()
        on_blob = " ".join(
            getattr(f, "content", "") for f in (onboard.get("semantic") or [])
        ).lower()
        assert salary_token.lower() in hr_blob
        assert salary_token.lower() not in on_blob
        assert theme_token.lower() in on_blob
        assert theme_token.lower() in hr_blob
