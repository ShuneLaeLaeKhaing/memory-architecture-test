"""
Smoke tests for promotion failure categories (brute-force, Burmese-friendly).

Categories:
  1. extraction error  — LLM pulls the right topic from MY+EN episode
  2. key-identity error — synonym / same-concept keys collapse
  3. scope error — personal pref resolves to user, not agent/org
  4. evidence/promotion error — ungrounded value is discarded

Run:
  pytest tests/test_burmese_failure_smoke.py -q
  pytest tests/test_burmese_failure_smoke.py -q -m slow   # includes LLM extract
"""

from datetime import datetime, timezone

import pytest

from memory.consolidation import EpisodeRecord
from memory.key_normalizer import keys_are_same_concept, normalize_key
from memory.promotion import Candidate, PromotionEngine


AGENT_ID = "hr_assistant"
USER = "eve"


def _ep(**kwargs) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=kwargs.get("episode_id", "ep_my_smoke"),
        session_id=kwargs.get("session_id", "sess_my_smoke"),
        user_id=kwargs.get("user_id", USER),
        agent_id=kwargs.get("agent_id", AGENT_ID),
        timestamp=datetime.now(timezone.utc).isoformat(),
        goal=kwargs.get(
            "goal",
            "Record UI theme preference from Burmese conversation",
        ),
        outcome=kwargs.get("outcome", "success"),
        summary=kwargs.get(
            "summary",
            "User said they want dark mode for all interfaces.",
        ),
        trajectory=kwargs.get(
            "trajectory",
            [
                "eve: ကျွန်တော် dark mode ကို နှစ်သက်တယ်။",
                "agent: Got it — dark mode.",
            ],
        ),
        lessons=kwargs.get("lessons", []),
        entities=kwargs.get("entities", ["eve", "dark mode"]),
        source_event_ids=kwargs.get("source_event_ids", ["evt_my_1"]),
        retention_score=0.9,
        scope=kwargs.get("scope", f"user:{USER}"),
        thread_id=kwargs.get("thread_id", "thread_my_smoke"),
    )


def _cand(**kwargs) -> Candidate:
    return Candidate(
        content=kwargs.get(
            "content",
            f"{USER} prefers dark mode for all user interfaces.",
        ),
        fact_type=kwargs.get("fact_type", "preference"),
        key=kwargs.get("key", "ui_theme"),
        value=kwargs.get("value", "dark mode"),
        scope_hint=kwargs.get("scope_hint", "user"),
        confidence=kwargs.get("confidence", 0.95),
        evidence=kwargs.get(
            "evidence",
            "ကျွန်တော် dark mode ကို နှစ်သက်တယ်။",
        ),
    )


class TestBurmeseFailureCategorySmoke:
    """Brute-force smokes: one check per failure category."""

    def test_key_identity_burmese_aliases_collapse(self):
        """Key-identity: invented aliases must match canonical ui_theme."""
        assert normalize_key("preferred_ui_theme") == "ui_theme"
        assert keys_are_same_concept("ui_theme", "preferred_ui_theme")
        assert keys_are_same_concept("ui_theme", "interface_theme_preference")
        # Opposite topic must not collapse into theme
        assert not keys_are_same_concept("ui_theme", "notification_channel")

    def test_scope_preference_is_user_not_agent(self):
        """Scope: personal preference from MY chat stays user-scoped."""
        eng = PromotionEngine.__new__(PromotionEngine)
        ep = _ep()
        cand = _cand()
        scope_type, scope_id = eng.identity_for(cand, ep)
        assert scope_type == "user"
        assert scope_id == USER

        lesson = _cand(
            fact_type="lesson",
            key="reply_style",
            value="concise",
            content=f"{AGENT_ID} should reply concisely to {USER}",
            evidence="lesson",
        )
        st, sid = eng.identity_for(lesson, ep)
        assert st == "agent"
        assert sid == AGENT_ID

    def test_evidence_grounds_burmese_latin_value(self):
        """Evidence OK: Latin value appears inside Burmese utterance."""
        eng = PromotionEngine.__new__(PromotionEngine)
        ep = _ep()
        cand = _cand(
            value="dark mode",
            evidence="ကျွန်တော် dark mode ကို နှစ်သက်တယ်။",
        )
        assert eng._evidence_supports_candidate(cand, ep) is True
        decision = eng.apply_policy(cand, existing=[], ep=ep)
        assert decision.action == "promote"

    def test_evidence_rejects_invented_secret(self):
        """Evidence/promotion error: invented token not in MY episode → discard."""
        eng = PromotionEngine.__new__(PromotionEngine)
        ep = _ep(
            summary="User only talked about dark mode theme.",
            trajectory=["eve: ကျွန်တော် dark mode ပဲ သုံးချင်တယ်။"],
            lessons=[],
        )
        cand = _cand(
            key="teams_notification_token",
            value="TeamsKeyTest77",
            content=f"{USER} Teams token is TeamsKeyTest77",
            evidence="ကျွန်တော် dark mode ပဲ သုံးချင်တယ်။",
            fact_type="identity",
        )
        assert eng._evidence_supports_candidate(cand, ep) is False
        decision = eng.apply_policy(cand, existing=[], ep=ep)
        assert decision.action == "discard"
        assert "grounded" in decision.reason.lower() or "evidence" in decision.reason.lower()

    def test_key_identity_policy_supersedes_not_duplicate(self):
        """Key-identity at policy: same concept key + new value → supersede."""
        from memory.semantic_memory import SemanticFact

        eng = PromotionEngine.__new__(PromotionEngine)
        ep = _ep(
            summary="User switched to light mode.",
            trajectory=["eve: အခု light mode ပြောင်းမယ်။"],
        )
        old = SemanticFact(
            fact_id="old_dark",
            content=f"{USER} prefers dark mode",
            user_id=USER,
            scope=f"user:{USER}",
            confidence=0.9,
            metadata={"fact_key": "ui_theme", "fact_value": "dark mode", "status": "current"},
        )
        cand = _cand(
            key="preferred_ui_theme",
            value="light mode",
            content=f"{USER} prefers light mode for all interfaces.",
            evidence="အခု light mode ပြောင်းမယ်။",
        )
        # Normalize like extract_candidates does
        cand.key = normalize_key(cand.key)
        decision = eng.apply_policy(cand, existing=[old], ep=ep)
        assert decision.action == "supersede"
        assert decision.all_targets or decision.target

    @pytest.mark.slow
    def test_extraction_from_burmese_episode_gets_theme(self):
        """Extraction: LLM should surface a theme/dark preference from MY episode."""
        eng = PromotionEngine()
        ep = _ep(
            goal="Save UI theme preference",
            summary=(
                "The user stated in Burmese that they prefer dark mode "
                "for the interface; the assistant confirmed."
            ),
            trajectory=[
                "eve: ကျေးဇူးပြုပြီး မှတ်ထားပေးပါ။ UI ကို dark mode ပဲ သုံးချင်တယ်။",
                "agent: Got it — I'll remember dark mode.",
            ],
            lessons=["User prefers dark mode"],
        )
        cands = eng.extract_candidates(ep)
        assert cands, "expected at least one candidate from Burmese theme episode"
        blob = " ".join(
            f"{c.key} {c.value} {c.content}".lower() for c in cands
        )
        assert any(
            tok in blob
            for tok in ("theme", "dark", "ui_theme", "interface")
        ), f"extraction missed theme signal: {blob}"
        # Grounded candidates should pass evidence gate
        grounded = [
            c for c in cands
            if eng._evidence_supports_candidate(c, ep)
        ]
        assert grounded, "extracted candidates were not grounded in episode text"
