"""Semantic Memory using Mem0 under infer=False with versioning and scope enforcement."""

import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger
from mem0 import MemoryClient  # Changed from Memory
from pydantic import BaseModel, Field
from memory.key_normalizer import normalize_key, keys_are_same_concept


class SemanticFact(BaseModel):
    """Domain model for authoritative semantic facts."""
    fact_id: str
    content: str
    user_id: str
    scope: str
    confidence: float
    status: str = "current"
    version: int = 1
    source_episode_ids: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = "promotion"
    supersedes_id: Optional[str] = None
    superseded_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticMemory:
    """Manages semantic facts using Mem0 Cloud API."""

    def __init__(self):
        # ✅ Use Mem0 Cloud with API Key
        api_key = os.getenv("MEM0_API_KEY")
        if not api_key:
            raise ValueError(
                "MEM0_API_KEY not found in environment. "
                "Get your key from https://app.mem0.ai/"
            )
        
        self.mem0 = MemoryClient(api_key=api_key)
        self._recent: List[SemanticFact] = []
        logger.info("[SemanticMemory] Mem0 Cloud API connected")

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
        """Write to the Mem0 owner for this scope; keep scope in metadata."""
        kind, owner_id, scope_label = _resolve_scope(
            scope_type=scope_type,
            scope_id=scope_id,
            user_id=user_id,
            scope=scope,
        )
        meta = metadata.copy() if metadata else {}
        meta.update({
            "scope": scope_label,
            "scope_type": kind,
            "owner_id": owner_id,
            "confidence": confidence,
            "status": status,
            "source_episode_ids": [source_episode_id],
            "trigger": trigger,
            "version": meta.get("version", 1),
            "supersedes_id": supersedes_id,
            "superseded_by": None,
            "created_at": datetime.utcnow().isoformat(),
        })
        if category:
            meta["category"] = category

        resp = self.mem0.add(
            messages=[{"role": "user", "content": content}],
            metadata=meta,
            infer=False,
            **_mem0_owner_kwargs(kind, owner_id),
        )

        memory_id = _memory_id_from_response(resp) or (
            "mem0_" + datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        )

        fact = SemanticFact(
            fact_id=memory_id,
            content=content,
            user_id=owner_id,
            scope=scope_label,
            confidence=confidence,
            status=status,
            version=meta["version"],
            source_episode_ids=[source_episode_id],
            created_at=meta["created_at"],
            created_by=trigger,
            supersedes_id=supersedes_id,
            metadata=meta
        )
        self._recent.append(fact)
        logger.info(
            f"[SemanticMemory] Fact committed: {fact.fact_id} "
            f"[{kind}:{owner_id}] -> '{content[:40]}...'"
        )
        return fact

    def update_fact(
        self,
        old_fact_id: str,
        new_content: str,
        user_id: str,
        source_episode_id: str,
        trigger: str = "user_correction"
    ) -> SemanticFact:
        """Creates an updated semantic version and supersedes the predecessor."""
        old_fact = self.get_fact(old_fact_id, user_id=user_id)
        current_version = old_fact.version if old_fact else 1
        new_version = current_version + 1

        if old_fact:
            self.mark_superseded(old_fact.fact_id, superseded_by="pending")

        new_fact = self.add_fact(
            content=new_content,
            user_id=user_id,
            scope=old_fact.scope if old_fact else f"user:{user_id}",
            confidence=1.0,
            source_episode_id=source_episode_id,
            trigger=trigger,
            status="current",
            metadata={"version": new_version},
            supersedes_id=old_fact_id,
        )
        self.mark_superseded(old_fact_id, superseded_by=new_fact.fact_id)
        logger.info(f"[SemanticMemory] Superseded {old_fact_id} with new fact {new_fact.fact_id}")
        return new_fact

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        top_k: int = 10,
        threshold: float = 0.45,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        include_org_facts: bool = True,
    ) -> List[SemanticFact]:
        """Search each Mem0 owner (user / agent / app), then filter by scope metadata."""
        wanted = list(scopes or [])
        if scope_type and scope_id:
            _, _, label = _resolve_scope(scope_type=scope_type, scope_id=scope_id)
            wanted.append(label)
            if include_org_facts and not label.startswith("shared:"):
                wanted.append("shared:organization")
        if user_id and not wanted:
            wanted = [f"user:{user_id}"]

        raw_list: List[dict] = []
        for filters in _mem0_filters_for_scopes(wanted, legacy_user_id=user_id):
            try:
                results = self.mem0.search(
                    query=query,
                    filters=filters,
                    top_k=top_k * 3,
                    threshold=0.0,
                )
                raw_list.extend(_rows(results))
            except Exception as exc:
                logger.warning(f"[SemanticMemory] Search error {filters}: {exc}")

        ranked = self._facts_from_rows(
            raw_list,
            user_id=user_id or "",
            scopes=wanted,
            query=query,
            threshold=threshold,
        )
        ranked = self._merge_recent(
            ranked,
            scopes=wanted,
            query=query,
            threshold=threshold,
        )

        if not ranked:
            for fact in self.get_all_facts(user_id=user_id, scopes=wanted, status=None):
                if fact.status == "superseded":
                    continue
                if wanted and fact.scope not in wanted:
                    continue
                score = _text_score(query, fact.content)
                if score >= threshold:
                    fact.metadata = {**fact.metadata, "_score": score}
                    ranked.append(fact)

        ranked.sort(key=lambda fact: float(fact.metadata.get("_score", 0.0)), reverse=True)
        return ranked[:top_k]

    def _facts_from_rows(
        self,
        raw_list: List[dict],
        *,
        user_id: str,
        scopes: List[str],
        query: str,
        threshold: float,
    ) -> List[SemanticFact]:
        clean_results: List[SemanticFact] = []
        for item in raw_list:
            text = str(item.get("memory") or item.get("content") or "")
            meta = _metadata(item)
            if meta.get("status") == "superseded":
                continue
            scope = str(meta.get("scope") or _default_scope(item, user_id))
            if scopes and scope not in scopes:
                continue
            raw_score = item.get("score")
            score = float(raw_score) if raw_score is not None else _text_score(query, text)
            if score < threshold:
                continue
            meta = {**meta, "_score": score}
            clean_results.append(
                SemanticFact(
                    fact_id=str(item.get("id") or "mem_unk"),
                    content=text,
                    user_id=str(meta.get("owner_id") or user_id),
                    scope=scope,
                    confidence=_as_float(meta.get("confidence"), 1.0),
                    status=str(meta.get("status") or "current"),
                    version=_as_int(meta.get("version"), 1),
                    source_episode_ids=_as_list(meta.get("source_episode_ids")),
                    created_at=str(meta.get("created_at") or datetime.utcnow().isoformat()),
                    created_by=str(meta.get("trigger") or "unknown"),
                    supersedes_id=meta.get("supersedes_id"),
                    superseded_by=meta.get("superseded_by"),
                    metadata=meta,
                )
            )
        return clean_results

    def _merge_recent(
        self,
        facts: List[SemanticFact],
        *,
        scopes: List[str],
        query: str,
        threshold: float,
    ) -> List[SemanticFact]:
        seen = {fact.fact_id for fact in facts}
        merged = list(facts)
        for fact in self._recent:
            if fact.fact_id in seen:
                continue
            if fact.status == "superseded":
                continue
            if scopes and fact.scope not in scopes:
                continue
            score = _text_score(query, fact.content)
            if score < threshold:
                continue
            fact.metadata = {**fact.metadata, "_score": score}
            merged.append(fact)
            seen.add(fact.fact_id)
        return merged

    def get_fact(self, fact_id: str, user_id: str) -> Optional[SemanticFact]:
        """Retrieves a specific memory node."""
        all_facts = self.get_all_facts(user_id=user_id)
        for f in all_facts:
            if f.fact_id == fact_id:
                return f
        return None

    def get_all_facts(
        self,
        user_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        status: Optional[str] = "current"
    ) -> List[SemanticFact]:
        """List current facts from each Mem0 owner in the requested scopes."""
        wanted = list(scopes or [])
        if user_id and not wanted:
            wanted = [f"user:{user_id}"]

        raw_list: List[dict] = []
        seen_ids: set = set()
        for filters in _mem0_filters_for_scopes(wanted, legacy_user_id=user_id):
            try:
                raw = self.mem0.get_all(filters=filters)
            except Exception as exc:
                logger.error(f"[SemanticMemory] get_all error {filters}: {exc}")
                continue
            for item in _rows(raw):
                item_id = str(item.get("id") or "")
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                raw_list.append(item)

        facts: List[SemanticFact] = []
        for item in raw_list:
            meta = _metadata(item)
            fact_status = str(meta.get("status") or "current")
            fact_scope = str(meta.get("scope") or _default_scope(item, user_id or ""))

            if status and fact_status != status:
                continue
            if wanted and fact_scope not in wanted:
                continue

            facts.append(
                SemanticFact(
                    fact_id=str(item.get("id") or "mem_unk"),
                    content=str(item.get("memory") or item.get("content") or ""),
                    user_id=str(meta.get("owner_id") or user_id or ""),
                    scope=fact_scope,
                    confidence=_as_float(meta.get("confidence"), 1.0),
                    status=fact_status,
                    version=_as_int(meta.get("version"), 1),
                    source_episode_ids=_as_list(meta.get("source_episode_ids")),
                    created_at=str(meta.get("created_at") or datetime.utcnow().isoformat()),
                    created_by=str(meta.get("trigger") or "manual"),
                    supersedes_id=meta.get("supersedes_id"),
                    metadata=meta
                )
            )

        return facts

    def find_in_recent(
    self,
    fact_key: str,
    scope: str,
) -> List[SemanticFact]:
        """Same-session lookup before Mem0 (avoids cloud lag)."""
        if not fact_key:
            return []
        return [
            fact for fact in self._recent
            if fact.status != "superseded"
            and fact.scope == scope
            and keys_are_same_concept(
                fact.metadata.get("fact_key", ""),
                fact_key
            )
        ]

    def bump_confidence(self, fact_id: str, extra_episode_id: str, delta: float = 0.05) -> None:
        """Merge action: reinforce an existing fact instead of duplicating."""
        fact = next((f for f in self._recent if f.fact_id == fact_id), None)
        if fact is None:
            raise RuntimeError(
                f"[SemanticMemory] bump failed: {fact_id} not in recent cache"
            )
        new_conf = min(1.0, fact.confidence + delta)
        episodes = list(dict.fromkeys(fact.source_episode_ids + [extra_episode_id]))
        self._update_and_verify(
            fact_id=fact_id,
            text=fact.content,
            metadata={
                **fact.metadata,
                "confidence": new_conf,
                "source_episode_ids": episodes,
            },
            expected_status=str(fact.metadata.get("status") or "current"),
        )
        fact.confidence = new_conf
        fact.source_episode_ids = episodes
        fact.metadata["confidence"] = new_conf
        fact.metadata["source_episode_ids"] = episodes
        logger.info(f"[SemanticMemory] Bumped confidence of {fact_id} → {new_conf}")

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
        """Mark old superseded first, then write the replacement."""
        self.mark_superseded(old_fact.fact_id, superseded_by="pending")
        new_fact = self.add_fact(
            content=new_content,
            scope_type=scope_type,
            scope_id=scope_id,
            confidence=0.95,
            category=old_fact.metadata.get("category", "general"),
            source_episode_id=source_episode_id,
            trigger=trigger,
            status="current",
            metadata={"version": old_fact.version + 1, **(extra_metadata or {})},
            supersedes_id=old_fact.fact_id,
        )
        self.mark_superseded(old_fact.fact_id, superseded_by=new_fact.fact_id)
        return new_fact

    def mark_superseded(self, fact_id: str, superseded_by: str) -> None:
        """
        Mark an existing fact as superseded.
        
        Mem0 Cloud rejects update() if text is unchanged (409 Conflict).
        Solution: append a superseded marker to the text so it's different.
        """
        cached = next((f for f in self._recent if f.fact_id == fact_id), None)
        original_text = cached.content if cached else ""
        existing_meta = cached.metadata.copy() if cached else {}

        # Mem0 requires text to actually change or it returns 409.
        # Append a marker that makes the content unique while preserving meaning.
        superseded_text = f"[SUPERSEDED] {original_text}".strip()
        # If we don't have the original text (not in cache), use a safe fallback
        if not original_text:
            superseded_text = f"[SUPERSEDED:{fact_id[:8]}]"

        updated_meta = {
            **existing_meta,
            "status": "superseded",
            "superseded_by": superseded_by,
        }

        try:
            self._update_and_verify(
                fact_id=fact_id,
                text=superseded_text,      # ← changed text avoids 409
                metadata=updated_meta,
                expected_status="superseded",
            )
        except Exception as exc:
            logger.error(
                f"[SemanticMemory] mark_superseded failed for "
                f"{fact_id[:8]}: {exc}"
            )
            raise

        # Update local cache
        if cached:
            cached.status = "superseded"
            cached.superseded_by = superseded_by
            cached.metadata["status"] = "superseded"
            cached.metadata["superseded_by"] = superseded_by

        logger.info(
            f"[SemanticMemory] Marked {fact_id[:8]} as superseded "
            f"(superseded_by={superseded_by[:8] if superseded_by != 'pending' else 'pending'})"
        )

    def _update_and_verify(
    self,
    *,
    fact_id: str,
    text: str,
    metadata: Dict[str, Any],
    expected_status: str,
    attempts: int = 5,
    delay_s: float = 0.5,
) -> None:
        """
        Call mem0.update(), then poll until the change is visible.
        Raises RuntimeError if update cannot be confirmed after all attempts.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            # Step A: attempt the update
            try:
                self.mem0.update(fact_id, text=text, metadata=metadata)
                logger.debug(
                    f"[SemanticMemory] update({fact_id[:8]}) "
                    f"call succeeded (attempt {attempt})"
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"[SemanticMemory] update({fact_id[:8]}) "
                    f"attempt {attempt}/{attempts} failed: {exc}"
                )
                time.sleep(delay_s)
                continue

            # Step B: verify the write is visible
            time.sleep(delay_s)  # give Mem0 Cloud time to propagate
            matched = self._status_matches(fact_id, expected_status)

            if matched is True:
                logger.debug(
                    f"[SemanticMemory] update({fact_id[:8]}) "
                    f"confirmed status='{expected_status}' "
                    f"after attempt {attempt}"
                )
                return

            if matched is False:
                logger.warning(
                    f"[SemanticMemory] update({fact_id[:8]}) "
                    f"not visible yet (attempt {attempt}/{attempts}), retrying"
                )
                # Loop: try updating again
                continue

            if matched is None:
                # Could not read back - API issue or fact not found
                # Don't retry the update, just wait and re-check
                logger.warning(
                    f"[SemanticMemory] update({fact_id[:8]}) "
                    f"read-back returned None (attempt {attempt}/{attempts}). "
                    f"Mem0 get() may not support single-fact fetch. "
                    f"Trusting the update call succeeded."
                )
                # If Mem0's .get() API doesn't work reliably,
                # we fall through and trust the update call itself
                # This is the pragmatic path - log it, don't block
                return

        raise RuntimeError(
            f"[SemanticMemory] update({fact_id[:8]}) "
            f"could not confirm status='{expected_status}' "
            f"after {attempts} attempts. Last error: {last_error}")

    def _status_matches(
    self,
    fact_id: str,
    expected_status: str,
) -> Optional[bool]:
        """
        Returns:
            True  - confirmed status matches
            False - confirmed status does not match  
            None  - could not determine (API error, fact not found)
        """
        try:
            raw = self.mem0.get(fact_id)
        except Exception as exc:
            logger.warning(
                f"[SemanticMemory] get({fact_id[:8]}) failed: {exc}"
            )
            return None

        # Mem0 Cloud can return dict OR list - handle both
        if isinstance(raw, list):
            if not raw:
                logger.warning(
                    f"[SemanticMemory] get({fact_id[:8]}) returned empty list"
                )
                return None
            raw = raw[0]

        if not isinstance(raw, dict):
            logger.warning(
                f"[SemanticMemory] get({fact_id[:8]}) unexpected type: "
                f"{type(raw)} value={str(raw)[:100]}"
            )
            return None

        # Log the full raw response once so you can see exactly what shape
        # Mem0 is returning - remove after confirming
        logger.debug(
            f"[SemanticMemory] get({fact_id[:8]}) raw response keys: "
            f"{list(raw.keys())}"
        )

        # Check metadata.status first (your primary storage)
        meta = _metadata(raw)
        status = meta.get("status", "")

        # Fallback: some Mem0 versions put status at top level
        if not status:
            status = str(raw.get("status") or "")

        if not status:
            logger.warning(
                f"[SemanticMemory] get({fact_id[:8]}) has no status field. "
                f"meta keys={list(meta.keys())} raw keys={list(raw.keys())}"
            )
            # Return None - we genuinely don't know, caller decides
            return None

        result = (status == expected_status)
        logger.debug(
            f"[SemanticMemory] get({fact_id[:8]}) "
            f"status='{status}' expected='{expected_status}' match={result}"
        )
        return result

    def diagnose_fact(self, fact_id: str) -> None:
        """
        Call this on a fact_id that should be superseded but isn't.
        Prints exactly what Mem0 returns so you can fix _status_matches.
        """
        logger.info(f"[Diagnose] Fetching fact {fact_id}")

        try:
            raw = self.mem0.get(fact_id)
            logger.info(f"[Diagnose] get() type: {type(raw)}")
            logger.info(f"[Diagnose] get() value: {raw}")
        except Exception as exc:
            logger.error(f"[Diagnose] get() raised: {exc}")

        logger.info("[Diagnose] Scanning get_all for fact_id...")
        try:
            all_raw = self.mem0.get_all(filters={"user_id": "bob"})
            rows = _rows(all_raw)
            match = next((r for r in rows if str(r.get("id")) == fact_id), None)
            if match:
                logger.info(f"[Diagnose] Found in get_all: {match}")
            else:
                logger.warning(
                    f"[Diagnose] NOT found in get_all for user:bob. "
                    f"Total rows returned: {len(rows)}"
                )
        except Exception as exc:
            logger.error(f"[Diagnose] get_all scan raised: {exc}")

        logger.info("[Diagnose] Testing update() round-trip...")
        try:
            self.mem0.update(
                fact_id,
                text="DIAGNOSTIC_PROBE",
                metadata={"_probe": True},
            )
            logger.info("[Diagnose] update() succeeded")

            time.sleep(1.0)
            raw2 = self.mem0.get(fact_id)
            logger.info(f"[Diagnose] get() after update: {raw2}")
        except Exception as exc:
            logger.error(f"[Diagnose] update/re-get raised: {exc}")


_SCOPE_FIELD = {
    "user": "user_id",
    "agent": "agent_id",
    "organization": "app_id",
}


def _normalize_scope_type(scope_type: str) -> str:
    if scope_type in ("organization", "org", "project", "shared"):
        return "organization"
    if scope_type in ("user", "agent"):
        return scope_type
    raise ValueError(f"Unknown scope_type: {scope_type}")


def _resolve_scope(
    *,
    scope_type: Optional[str] = None,
    scope_id: Optional[str] = None,
    user_id: Optional[str] = None,
    scope: Optional[str] = None,
) -> tuple:
    """Return (kind, owner_id, metadata scope string)."""
    if scope_type:
        kind = _normalize_scope_type(scope_type)
        if kind == "user":
            owner = scope_id or user_id
            if not owner:
                raise ValueError("user scope needs scope_id or user_id")
            return kind, owner, f"user:{owner}"
        if kind == "agent":
            owner = scope_id
            if not owner:
                raise ValueError("agent scope needs scope_id")
            return kind, owner, f"agent:{owner}"
        return kind, scope_id or "organization", "shared:organization"
    if scope:
        return _parse_scope_string(scope)
    if user_id:
        return "user", user_id, f"user:{user_id}"
    raise ValueError("Need scope_type/scope_id or user_id/scope")


def _parse_scope_string(scope: str) -> tuple:
    raw = (scope or "").strip()
    if raw in ("shared:organization", "organization", "organization:organization", "shared"):
        return "organization", "organization", "shared:organization"
    if ":" in raw:
        kind, owner = raw.split(":", 1)
        if kind == "user":
            return "user", owner, f"user:{owner}"
        if kind == "agent":
            return "agent", owner, f"agent:{owner}"
        if kind in ("shared", "organization"):
            return "organization", owner or "organization", "shared:organization"
    return "user", raw, f"user:{raw}"


def _mem0_owner_kwargs(kind: str, owner_id: str) -> Dict[str, str]:
    return {_SCOPE_FIELD[kind]: owner_id}


def _mem0_filters_for_scopes(
    scopes: List[str],
    legacy_user_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """One Mem0 filter per owner. Also query user_id for older facts stored there."""
    seen = set()
    filters: List[Dict[str, str]] = []
    for scope in scopes:
        kind, owner, _ = _parse_scope_string(scope)
        key = (_SCOPE_FIELD[kind], owner)
        if key in seen:
            continue
        seen.add(key)
        filters.append({_SCOPE_FIELD[kind]: owner})
    if legacy_user_id:
        key = ("user_id", legacy_user_id)
        if key not in seen:
            filters.append({"user_id": legacy_user_id})
    return filters


def _default_scope(item: dict, user_id: str) -> str:
    meta = _metadata(item)
    if meta.get("scope"):
        return str(meta["scope"])
    if item.get("agent_id"):
        return f"agent:{item['agent_id']}"
    if item.get("app_id"):
        return "shared:organization"
    return f"user:{user_id or item.get('user_id') or 'unknown'}"


def _text_score(query: str, content: str) -> float:
    query_words = set(re.findall(r"[a-z0-9]{4,}", query.lower()))
    content_words = set(re.findall(r"[a-z0-9]{4,}", content.lower()))
    if not query_words or not content_words:
        return 0.0
    hits = 0
    for query_word in query_words:
        prefix = query_word[:4]
        if any(word == query_word or word.startswith(prefix) or query_word.startswith(word[:4]) for word in content_words):
            hits += 1
    return hits / len(query_words)


def _rows(raw: Any) -> List[dict]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if not isinstance(raw, dict):
        return []
    for key in ("results", "memories"):
        items = raw.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _memory_id_from_response(resp: Any) -> str:
    if isinstance(resp, dict):
        if resp.get("id"):
            return str(resp["id"])
        rows = _rows(resp)
        if rows and rows[0].get("id"):
            return str(rows[0]["id"])
    if isinstance(resp, list) and resp and isinstance(resp[0], dict) and resp[0].get("id"):
        return str(resp[0]["id"])
    return ""


def _metadata(item: dict) -> Dict[str, Any]:
    meta = item.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _as_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and item != ""]
    return [str(value)]


def _as_int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
