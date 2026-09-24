"""Storage contract for semantic facts.

Promotion and retrieval depend on this interface. SemanticMemory is the
current Mem0 Cloud implementation. A later PostgreSQL store can implement
the same methods without changing promotion or conflict rules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from memory.semantic_memory import SemanticFact


class SemanticStore(Protocol):
    """Operations the memory policy layer already uses."""

    def add_fact(
        self,
        content: str,
        confidence: float,
        source_episode_id: str,
        trigger: str,
        status: str = "current",
        metadata: Optional[Dict[str, Any]] = None,
        supersedes_id: Optional[str] = None,
        user_id: Optional[str] = None,
        scope: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> SemanticFact:
        """Insert one fact and return the stored record."""

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        top_k: int = 10,
        threshold: float = 0.45,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        include_org_facts: bool = False,
    ) -> List[SemanticFact]:
        """Return similar facts for a query."""

    def get_all_facts(
        self,
        user_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        status: Optional[str] = "current",
    ) -> List[SemanticFact]:
        """List facts for the requested owners."""

    def get_fact(self, fact_id: str, user_id: str) -> Optional[SemanticFact]:
        """Load one fact by id."""

    def find_in_recent(self, fact_key: str, scope: str) -> List[SemanticFact]:
        """Return same-session facts that match this key and scope."""

    def supersede_fact(
        self,
        old_fact: SemanticFact,
        new_content: str,
        scope_type: str,
        scope_id: str,
        source_episode_id: str,
        trigger: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticFact:
        """Mark the old fact superseded and write the replacement."""

    def mark_superseded(self, fact_id: str, superseded_by: str) -> None:
        """Mark one fact as no longer current."""

    def bump_confidence(
        self,
        fact_id: str,
        extra_episode_id: str,
        delta: float = 0.05,
        fact: Optional[SemanticFact] = None,
        extra_event_ids: Optional[List[str]] = None,
        extra_session_ids: Optional[List[str]] = None,
        extra_thread_ids: Optional[List[str]] = None,
    ) -> None:
        """Update confidence and lineage metadata on an existing fact."""
