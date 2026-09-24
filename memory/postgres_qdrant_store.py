"""Semantic facts in PostgreSQL, similarity search in Qdrant."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from memory.key_normalizer import keys_are_same_concept
from memory.semantic_memory import (
    SemanticFact,
    _as_float,
    _as_int,
    _as_list,
    _resolve_scope,
    _text_score,
)

_SQL_PATH = Path(__file__).resolve().parent / "sql" / "semantic_facts.sql"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dsn() -> str:
    url = os.getenv("SEMANTIC_DATABASE_URL", "").strip()
    if url:
        return url
    user = os.getenv("DB_USERNAME", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "5433")
    database = os.getenv("DB_DATABASE", "memory_architecture")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _report(line: str) -> None:
    logger.info(line)
    print(line)


class PostgresFactRepository:
    """SQL only. No vectors."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = None

    def connect(self) -> str:
        import psycopg

        try:
            self._conn = psycopg.connect(self._dsn, autocommit=True, connect_timeout=5)
            sql = _SQL_PATH.read_text(encoding="utf-8")
            with self._conn.cursor() as cur:
                cur.execute(sql)
            return "connected"
        except Exception as exc:
            return f"failed: {exc}"

    def insert(
        self,
        *,
        fact_id: str,
        kind: str,
        owner_id: str,
        scope_label: str,
        fact_key: str,
        content: str,
        status: str,
        version: int,
        confidence: float,
        category: Optional[str],
        supersedes_id: Optional[str],
        episode_ids: List[str],
        meta: Dict[str, Any],
        created_at: str,
        trigger: str,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO semantic_facts (
                    fact_id, owner_type, owner_id, scope, fact_key, content, status,
                    version, confidence, category, supersedes_id, superseded_by,
                    source_episode_ids, metadata, created_at, created_by
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, NULL,
                    %s::jsonb, %s::jsonb, %s, %s
                )
                """,
                (
                    fact_id, kind, owner_id, scope_label, fact_key or None, content, status,
                    version, confidence, category, supersedes_id,
                    json.dumps(episode_ids), json.dumps(meta), created_at, trigger,
                ),
            )

    def fetch_all(
        self,
        *,
        scopes: List[str],
        status: Optional[str],
    ) -> List[SemanticFact]:
        sql = "SELECT * FROM semantic_facts WHERE 1=1"
        params: List[Any] = []
        if status:
            sql += " AND status = %s"
            params.append(status)
        if scopes:
            sql += " AND scope = ANY(%s)"
            params.append(scopes)
        sql += " ORDER BY created_at DESC"
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
        return [self.row_to_fact(dict(zip(columns, row))) for row in rows]

    def fetch_one(self, fact_id: str) -> Optional[SemanticFact]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM semantic_facts WHERE fact_id = %s", (fact_id,))
            row = cur.fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in cur.description]
        return self.row_to_fact(dict(zip(columns, row)))

    def mark_superseded(self, fact_id: str, superseded_by: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE semantic_facts
                SET status = 'superseded',
                    superseded_by = %s,
                    metadata = jsonb_set(
                        jsonb_set(metadata, '{status}', '"superseded"'),
                        '{superseded_by}',
                        to_jsonb(%s::text)
                    )
                WHERE fact_id = %s
                """,
                (superseded_by, superseded_by, fact_id),
            )

    def update_confidence(self, fact_id: str, confidence: float, episodes: List[str], meta: Dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE semantic_facts
                SET confidence = %s,
                    source_episode_ids = %s::jsonb,
                    metadata = %s::jsonb
                WHERE fact_id = %s
                """,
                (confidence, json.dumps(episodes), json.dumps(meta), fact_id),
            )

    def row_to_fact(self, row: Dict[str, Any]) -> SemanticFact:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        episodes = row.get("source_episode_ids") or []
        if isinstance(episodes, str):
            episodes = json.loads(episodes)
        created = row.get("created_at")
        created_at = created.isoformat() if hasattr(created, "isoformat") else str(created or "")
        return SemanticFact(
            fact_id=str(row["fact_id"]),
            content=str(row.get("content") or ""),
            user_id=str(row.get("owner_id") or ""),
            scope=str(row.get("scope") or ""),
            confidence=_as_float(row.get("confidence"), 1.0),
            status=str(row.get("status") or "current"),
            version=_as_int(row.get("version"), 1),
            source_episode_ids=[str(item) for item in episodes],
            created_at=created_at,
            created_by=str(row.get("created_by") or "promotion"),
            supersedes_id=row.get("supersedes_id"),
            superseded_by=row.get("superseded_by"),
            metadata=dict(meta),
        )


class QdrantFactIndex:
    """Vectors only. Returns fact ids; PostgreSQL loads the rows."""

    def __init__(self, url: str, collection: str, vector_size: int) -> None:
        self._url = url
        self._collection = collection
        self._vector_size = vector_size
        self._client = None

    def connect(self) -> str:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        try:
            self._client = QdrantClient(url=self._url, timeout=10)
            names = {item.name for item in self._client.get_collections().collections}
            if self._collection not in names:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
                )
            return "connected"
        except Exception as exc:
            return f"failed: {exc}"

    def upsert(self, fact: SemanticFact) -> str:
        vector = self._embed(fact.content)
        if vector is None:
            return "skipped"
        from qdrant_client.models import PointStruct

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=fact.fact_id,
                    vector=vector,
                    payload={"scope": fact.scope, "fact_key": fact.metadata.get("fact_key") or ""},
                )
            ],
        )
        return "ok"

    def delete(self, fact_id: str) -> str:
        from qdrant_client.models import PointIdsList

        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=[fact_id]),
            )
            return "ok"
        except Exception as exc:
            logger.warning(f"[QdrantFactIndex] delete failed fact_id={fact_id}: {exc}")
            return "failed"

    def search_ids(self, query: str, scopes: List[str], top_k: int, threshold: float) -> List[Tuple[str, float]]:
        vector = self._embed(query)
        if vector is None or self._client is None:
            return []
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        query_filter = None
        if scopes:
            query_filter = Filter(
                must=[FieldCondition(key="scope", match=MatchAny(any=scopes))]
            )
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=max(top_k, 1),
            score_threshold=threshold,
        )
        return [(str(hit.id), float(hit.score)) for hit in response.points]

    def _embed(self, text: str) -> Optional[List[float]]:
        try:
            from memory.embeddings import get_encoder

            vector = get_encoder().encode(text)
            return [float(value) for value in vector]
        except Exception as exc:
            logger.warning(f"[QdrantFactIndex] embedding skipped: {exc}")
            return None


class PostgresQdrantStore:
    """PostgreSQL is authoritative. Qdrant only returns similar fact ids."""

    def __init__(self) -> None:
        self._sql = PostgresFactRepository(_dsn())
        self._index = QdrantFactIndex(
            url=os.getenv("QDRANT_URL", "http://127.0.0.1:6335"),
            collection=os.getenv("QDRANT_SEMANTIC_COLLECTION", "memory_semantic_facts"),
            vector_size=int(os.getenv("EMBEDDING_DIM", "1024")),
        )
        self._recent: List[SemanticFact] = []
        health = self.health()
        if not health["postgres"].startswith("connected") or not health["qdrant"].startswith("connected"):
            raise RuntimeError(
                f"Semantic store health failed: postgres={health['postgres']} qdrant={health['qdrant']}"
            )

    def health(self) -> Dict[str, str]:
        postgres = self._sql.connect()
        qdrant = self._index.connect()
        _report(f"[health] PostgreSQL {postgres}")
        _report(f"[health] Qdrant {qdrant}")
        return {"postgres": postgres, "qdrant": qdrant}

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
        kind, owner_id, scope_label = _resolve_scope(
            scope_type=scope_type,
            scope_id=scope_id,
            user_id=user_id,
            scope=scope,
        )
        meta = dict(metadata or {})
        if category:
            meta["category"] = category
        episode_ids = list(dict.fromkeys(_as_list(meta.get("source_episode_ids")) + [source_episode_id]))
        created_at = _utcnow()
        meta.update({
            "scope": scope_label,
            "scope_type": kind,
            "owner_id": owner_id,
            "confidence": confidence,
            "status": status,
            "source_episode_ids": episode_ids,
            "trigger": trigger,
            "version": _as_int(meta.get("version"), 1),
            "supersedes_id": supersedes_id,
            "superseded_by": None,
            "created_at": created_at,
            "fact_key": meta.get("fact_key") or "",
        })
        fact_id = str(uuid.uuid4())
        fact_key = str(meta.get("fact_key") or "")
        self._sql.insert(
            fact_id=fact_id,
            kind=kind,
            owner_id=owner_id,
            scope_label=scope_label,
            fact_key=fact_key,
            content=content,
            status=status,
            version=meta["version"],
            confidence=confidence,
            category=meta.get("category"),
            supersedes_id=supersedes_id,
            episode_ids=episode_ids,
            meta=meta,
            created_at=created_at,
            trigger=trigger,
        )
        fact = SemanticFact(
            fact_id=fact_id,
            content=content,
            user_id=owner_id,
            scope=scope_label,
            confidence=confidence,
            status=status,
            version=meta["version"],
            source_episode_ids=episode_ids,
            created_at=created_at,
            created_by=trigger,
            supersedes_id=supersedes_id,
            metadata=meta,
        )
        self._recent.append(fact)
        qdrant_status = self._index.upsert(fact)
        _report(f"[fact {fact_id}] postgres=ok qdrant={qdrant_status}")
        return fact

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
        wanted = list(scopes or [])
        if scope_type and scope_id:
            _, _, label = _resolve_scope(scope_type=scope_type, scope_id=scope_id)
            wanted.append(label)
            if include_org_facts and not label.startswith("shared:"):
                wanted.append("shared:organization")
        if user_id and not wanted:
            wanted = [f"user:{user_id}"]

        ranked: List[SemanticFact] = []
        for fact_id, score in self._index.search_ids(query, wanted, top_k, threshold):
            fact = self.get_fact(fact_id, user_id="")
            if fact is None or fact.status == "superseded":
                continue
            fact.metadata = {**fact.metadata, "_score": score}
            ranked.append(fact)
        if not ranked:
            for fact in self.get_all_facts(user_id=user_id, scopes=wanted, status=None):
                if fact.status == "superseded":
                    continue
                score = _text_score(query, fact.content)
                if score >= threshold:
                    fact.metadata = {**fact.metadata, "_score": score}
                    ranked.append(fact)
        seen = {fact.fact_id for fact in ranked}
        for fact in self._recent:
            if fact.fact_id in seen or fact.status == "superseded":
                continue
            if wanted and fact.scope not in wanted:
                continue
            score = _text_score(query, fact.content)
            if score < threshold:
                continue
            fact.metadata = {**fact.metadata, "_score": score}
            ranked.append(fact)
        ranked.sort(key=lambda fact: float(fact.metadata.get("_score", 0.0)), reverse=True)
        return ranked[:top_k]

    def get_all_facts(
        self,
        user_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        status: Optional[str] = "current",
    ) -> List[SemanticFact]:
        wanted = list(scopes or [])
        if user_id and not wanted:
            wanted = [f"user:{user_id}"]
        return self._sql.fetch_all(scopes=wanted, status=status)

    def get_fact(self, fact_id: str, user_id: str) -> Optional[SemanticFact]:
        return self._sql.fetch_one(fact_id)

    def find_in_recent(self, fact_key: str, scope: str) -> List[SemanticFact]:
        if not fact_key:
            return []
        return [
            fact for fact in self._recent
            if fact.status != "superseded"
            and fact.scope == scope
            and keys_are_same_concept(str(fact.metadata.get("fact_key") or ""), fact_key)
        ]

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
        self.mark_superseded(old_fact.fact_id, superseded_by="pending")
        meta = {
            "version": old_fact.version + 1,
            "category": old_fact.metadata.get("category", "general"),
            **(extra_metadata or {}),
        }
        new_fact = self.add_fact(
            content=new_content,
            scope_type=scope_type,
            scope_id=scope_id,
            confidence=0.95,
            source_episode_id=source_episode_id,
            trigger=trigger,
            status="current",
            metadata=meta,
            supersedes_id=old_fact.fact_id,
        )
        self.mark_superseded(old_fact.fact_id, superseded_by=new_fact.fact_id)
        return new_fact

    def mark_superseded(self, fact_id: str, superseded_by: str) -> None:
        self._sql.mark_superseded(fact_id, superseded_by)
        for fact in self._recent:
            if fact.fact_id == fact_id:
                fact.status = "superseded"
                fact.superseded_by = superseded_by
                fact.metadata = {**fact.metadata, "status": "superseded", "superseded_by": superseded_by}
        qdrant_status = self._index.delete(fact_id)
        _report(f"[fact {fact_id}] postgres=superseded qdrant={qdrant_status}")

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
        target = fact or self.get_fact(fact_id, user_id="")
        if target is None:
            raise RuntimeError(f"[PostgresQdrantStore] bump failed: {fact_id} not found")
        new_conf = min(1.0, target.confidence + delta)
        episodes = list(dict.fromkeys(target.source_episode_ids + [extra_episode_id]))
        meta = dict(target.metadata)
        meta["confidence"] = new_conf
        meta["source_episode_ids"] = episodes
        meta["source_event_ids"] = list(dict.fromkeys(
            _as_list(meta.get("source_event_ids")) + list(extra_event_ids or [])
        ))
        meta["source_session_ids"] = list(dict.fromkeys(
            _as_list(meta.get("source_session_ids")) + list(extra_session_ids or [])
        ))
        meta["source_thread_ids"] = list(dict.fromkeys(
            _as_list(meta.get("source_thread_ids")) + list(extra_thread_ids or [])
        ))
        self._sql.update_confidence(fact_id, new_conf, episodes, meta)
        target.confidence = new_conf
        target.source_episode_ids = episodes
        target.metadata = meta
        _report(f"[fact {fact_id}] postgres=updated qdrant=unchanged")
