"""Unit tests for ChatGPT-style preference conflict authority."""

from types import SimpleNamespace

from memory.conflict_authority import (
    filter_stale_episodes,
    format_conflict_block,
    resolve_preference_conflicts,
)
from memory.semantic_memory import SemanticFact


def _fact(content, key, created_at, fact_id="f1"):
    return SemanticFact(
        fact_id=fact_id,
        content=content,
        user_id="frank",
        scope="user:frank",
        confidence=0.95,
        created_at=created_at,
        metadata={
            "fact_key": key,
            "category": "preference",
            "created_at": created_at,
            "version": 2,
        },
    )


def _user_msg(content, ts):
    return SimpleNamespace(
        event_type="user_message",
        timestamp=ts,
        payload={"content": content},
    )


class TestConflictAuthority:
    def test_newer_semantic_beats_old_thread_teams(self):
        fact = _fact(
            "Frank's notification preference is set to Slack.",
            "notification_channel",
            "2026-09-17T13:49:41",
        )
        events = [
            _user_msg(
                "Please remember my notification preference is Microsoft Teams",
                "2026-09-17T13:47:52",
            )
        ]
        resolved = resolve_preference_conflicts(facts=[fact], thread_events=events)
        assert len(resolved) == 1
        assert resolved[0].winner == "semantic"
        assert "slack" in resolved[0].winning_value
        assert resolved[0].acknowledge is True
        block = format_conflict_block(resolved)
        assert "SAVED FACT" in block or "WINNER = SAVED FACT" in block
        assert "slack" in block.lower()
        assert "acknowledge" in block.lower()

    def test_current_thread_beats_older_semantic(self):
        fact = _fact(
            "Frank's preferred notification channel is Microsoft Teams.",
            "notification_channel",
            "2026-09-17T13:48:45",
        )
        events = [
            _user_msg(
                "I prefer slack notifications now",
                "2026-09-17T13:49:10",
            )
        ]
        resolved = resolve_preference_conflicts(facts=[fact], thread_events=events)
        assert len(resolved) == 1
        assert resolved[0].winner == "thread"
        assert "slack" in resolved[0].winning_value
        assert resolved[0].acknowledge is True
        assert "acknowledge" in format_conflict_block(resolved).lower()
        assert "previously" in format_conflict_block(resolved).lower() or "losing" in format_conflict_block(resolved).lower()

    def test_stale_teams_episode_filtered_when_slack_wins(self):
        fact = _fact(
            "Frank's notification preference is set to Slack.",
            "notification_channel",
            "2026-09-17T13:49:41",
        )
        events = [
            _user_msg(
                "Please remember my notification preference is Microsoft Teams",
                "2026-09-17T13:47:52",
            )
        ]
        resolved = resolve_preference_conflicts(
            facts=[fact],
            thread_events=events,
            episodes=[
                {
                    "goal": "Update preference to Microsoft Teams",
                    "outcome": "success",
                    "summary": "User set Teams",
                }
            ],
        )
        kept = filter_stale_episodes(
            [
                {
                    "goal": "Update preference to Microsoft Teams",
                    "outcome": "success",
                    "summary": "User set Teams",
                },
                {
                    "goal": "Ask about salary",
                    "outcome": "success",
                    "summary": "Salary discussed",
                },
            ],
            resolved,
        )
        assert len(kept) == 1
        assert "salary" in kept[0]["summary"].lower()
