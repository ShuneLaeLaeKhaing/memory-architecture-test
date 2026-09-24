"""Choose Mem0 or PostgreSQL + Qdrant from SEMANTIC_STORE."""

from __future__ import annotations

import os

from memory.semantic_store import SemanticStore


def semantic_store_name() -> str:
    return os.getenv("SEMANTIC_STORE", "mem0").strip().lower()


def open_semantic_store() -> SemanticStore:
    name = semantic_store_name()
    if name in {"postgres", "postgresql", "qdrant"}:
        from memory.postgres_qdrant_store import PostgresQdrantStore

        return PostgresQdrantStore()
    from memory.semantic_memory import SemanticMemory

    return SemanticMemory()
