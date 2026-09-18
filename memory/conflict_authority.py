"""Explicit thread ↔ semantic conflict resolution (ChatGPT-style policy).

Policy for personal preferences (same user):
1. Current/recent thread statement newer than the fact → thread wins
2. Older thread/episode vs newer semantic fact → semantic wins

Same-key semantic duplicates are handled upstream (supersede + apply_semantic_authority),
not here. When resolving a conflict, briefly acknowledge the losing side once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from memory.retrieval import fact_timestamp, parse_memory_timestamp, scope_kind


# Topic buckets: if thread and fact mention different values in the same bucket, conflict.
_TOPIC_VALUES: Dict[str, Tuple[str, ...]] = {
    "notification_channel": (
        "slack",
        "teams",
        "microsoft teams",
        "email",
        "sms",
        "phone",
        "whatsapp",
    ),
    "ui_theme": ("dark mode", "light mode", "dark", "light", "beige"),
}


@dataclass
class ConflictResolution:
    topic: str
    winner: str  # "thread" | "semantic"
    winning_value: str
    losing_value: str
    fact_content: str
    acknowledge: bool
    reason: str

    def prompt_line(self) -> str:
        if self.winner == "thread":
            return (
                f"- Topic '{self.topic}': WINNER = CURRENT CHAT → {self.winning_value}. "
                f"LOSING side = older saved fact ({self.losing_value}). "
                f"Required: answer with {self.winning_value}, and in the SAME reply "
                f"briefly acknowledge the losing side once "
                f"(e.g. \"I'll use {self.winning_value} — I previously had "
                f"{self.losing_value} saved.\")."
            )
        return (
            f"- Topic '{self.topic}': WINNER = SAVED FACT → {self.winning_value}. "
            f"LOSING side = older chat/episode ({self.losing_value}). "
            f"Fact: {self.fact_content} "
            f"Required: answer with {self.winning_value}, and in the SAME reply "
            f"briefly acknowledge the losing side once "
            f"(e.g. \"I'll use {self.winning_value} — earlier you mentioned "
            f"{self.losing_value}.\")."
        )


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _find_values(text: str, topic: str) -> List[str]:
    blob = _norm(text)
    found: List[str] = []
    for value in _TOPIC_VALUES.get(topic, ()):
        if value in blob and value not in found:
            found.append(value)
    return found


def _primary_value(values: Sequence[str]) -> Optional[str]:
    if not values:
        return None
    # Prefer longer/more specific match (microsoft teams > teams)
    return sorted(values, key=len, reverse=True)[0]


def _fact_is_preference(fact: Any) -> bool:
    meta = getattr(fact, "metadata", None) or {}
    category = str(meta.get("category") or "").lower()
    key = str(meta.get("fact_key") or "").lower()
    if category in {"preference", "identity", "constraint"}:
        return True
    return any(token in key for token in ("prefer", "notification", "channel", "theme", "ui_"))


def _topic_for_fact(fact: Any) -> Optional[str]:
    meta = getattr(fact, "metadata", None) or {}
    key = str(meta.get("fact_key") or "").lower()
    content = getattr(fact, "content", "") or ""
    if any(t in key for t in ("notification", "channel", "notify")):
        return "notification_channel"
    if any(t in key for t in ("theme", "ui_")):
        return "ui_theme"
    for topic in _TOPIC_VALUES:
        if _find_values(content, topic):
            return topic
    return None


def resolve_preference_conflicts(
    *,
    facts: Sequence[Any],
    thread_events: Sequence[Any],
    episodes: Optional[Sequence[Any]] = None,
) -> List[ConflictResolution]:
    """
    Compare preference facts with thread (and episode text).

    Thread wins only when a user message mentioning a competing value is
    strictly newer than the fact timestamp.
    """
    resolutions: List[ConflictResolution] = []
    user_facts = [
        f for f in facts
        if scope_kind(getattr(f, "scope", "") or "") == "user" and _fact_is_preference(f)
    ]

    for fact in user_facts:
        topic = _topic_for_fact(fact)
        if not topic:
            continue
        fact_value = _primary_value(_find_values(getattr(fact, "content", "") or "", topic))
        if not fact_value:
            continue
        fact_ts = fact_timestamp(fact)

        # Newest user message that mentions this topic with any known value
        latest_thread_value: Optional[str] = None
        latest_thread_ts: Optional[datetime] = None
        for event in thread_events:
            etype = getattr(event, "event_type", None) or (
                event.get("event_type") if isinstance(event, dict) else None
            )
            if etype != "user_message":
                continue
            payload = getattr(event, "payload", None) or (
                event.get("payload") if isinstance(event, dict) else {}
            ) or {}
            content = payload.get("content", "") if isinstance(payload, dict) else ""
            values = _find_values(content, topic)
            if not values:
                continue
            ts = parse_memory_timestamp(
                getattr(event, "timestamp", None)
                or (event.get("timestamp") if isinstance(event, dict) else None)
            )
            if latest_thread_ts is None or (ts and ts > latest_thread_ts):
                latest_thread_ts = ts
                latest_thread_value = _primary_value(values)

        episode_mentions_other = False
        for ep in episodes or []:
            if isinstance(ep, dict):
                blob = f"{ep.get('goal', '')} {ep.get('summary', '')}"
            else:
                blob = f"{getattr(ep, 'goal', '')} {getattr(ep, 'summary', '')}"
            values = _find_values(blob, topic)
            other = [v for v in values if v != fact_value and fact_value not in v and v not in fact_value]
            if other:
                episode_mentions_other = True
                break

        if latest_thread_value and latest_thread_value != fact_value:
            thread_newer = (
                fact_ts is None
                or latest_thread_ts is None
                or latest_thread_ts > fact_ts
            )
            if thread_newer:
                resolutions.append(
                    ConflictResolution(
                        topic=topic,
                        winner="thread",
                        winning_value=latest_thread_value,
                        losing_value=fact_value,
                        fact_content=getattr(fact, "content", ""),
                        acknowledge=True,
                        reason="current thread newer than semantic fact",
                    )
                )
            else:
                resolutions.append(
                    ConflictResolution(
                        topic=topic,
                        winner="semantic",
                        winning_value=fact_value,
                        losing_value=latest_thread_value,
                        fact_content=getattr(fact, "content", ""),
                        acknowledge=True,
                        reason="semantic fact newer than thread statement",
                    )
                )
        elif episode_mentions_other:
            # No competing current-thread value, but stale episode still says otherwise
            resolutions.append(
                ConflictResolution(
                    topic=topic,
                    winner="semantic",
                    winning_value=fact_value,
                    losing_value="older episode",
                    fact_content=getattr(fact, "content", ""),
                    acknowledge=True,
                    reason="semantic fact overrides older episode",
                )
            )

    if resolutions:
        logger.info(
            "[ConflictAuthority] "
            + "; ".join(
                f"{r.topic}:{r.winner}={r.winning_value} ({r.reason})" for r in resolutions
            )
        )
    return resolutions


def format_conflict_block(resolutions: Sequence[ConflictResolution]) -> str:
    if not resolutions:
        return (
            "No explicit preference conflicts detected between chat and saved facts.\n"
            "Still apply: current clear chat prefs beat older saved facts; "
            "newer saved facts beat older chat."
        )
    lines = [
        "RESOLVED PREFERENCE CONFLICTS (code-decided — obey these over raw chat/episodes):",
    ]
    for r in resolutions:
        lines.append(r.prompt_line())
    lines.append(
        "When a conflict is resolved above: state the winner AND briefly "
        "acknowledge the losing side once in the same reply. "
        "Ignore contradictory older episodes/assistant lines for that topic."
    )
    return "\n".join(lines)


def filter_stale_episodes(
    episodes: Sequence[Any],
    resolutions: Sequence[ConflictResolution],
) -> List[Any]:
    """Drop episodes that only push a losing preference value."""
    if not resolutions:
        return list(episodes)
    semantic_wins = [r for r in resolutions if r.winner == "semantic"]
    if not semantic_wins:
        return list(episodes)

    kept: List[Any] = []
    for ep in episodes:
        if isinstance(ep, dict):
            blob = f"{ep.get('goal', '')} {ep.get('summary', '')}"
        else:
            blob = f"{getattr(ep, 'goal', '')} {getattr(ep, 'summary', '')}"
        drop = False
        for r in semantic_wins:
            values = _find_values(blob, r.topic)
            if not values:
                continue
            # Episode pushes the losing channel and not the winning one
            has_lose = any(
                v == r.losing_value or r.losing_value in v or v in r.losing_value
                for v in values
                if r.losing_value != "older episode"
            )
            has_win = any(
                v == r.winning_value or r.winning_value in v or v in r.winning_value
                for v in values
            )
            if has_lose and not has_win:
                drop = True
                break
        if not drop:
            kept.append(ep)
    return kept
