"""Shared Chroma query helpers. Chroma 1.x uses L2 by default, so 1 - distance is not cosine."""

from __future__ import annotations

import math
from typing import Any


def cosine(left: list[float], right: list[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for x, y in zip(left, right):
        dot += x * y
        left_norm += x * x
        right_norm += y * y
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def has_hits(result: dict[str, Any]) -> bool:
    return bool(_first_batch(result, "ids"))


def query_collection(
    collection: Any,
    query_vector: list[float],
    n_results: int,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    empty = {"ids": [[]], "distances": [[]], "metadatas": [[]], "embeddings": [[]]}
    count = collection.count()
    if count == 0:
        return empty
    n_results = max(1, min(n_results, count))
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_vector],
        "n_results": n_results,
        "include": ["metadatas", "documents", "embeddings"],
    }
    if where:
        try:
            return collection.query(**kwargs, where=where)
        except Exception:
            pass
    return collection.query(**kwargs)


def _first_batch(result: dict[str, Any], key: str) -> list[Any]:
    raw = result.get(key)
    if raw is None:
        return []
    if len(raw) == 0:
        return []
    batch = raw[0]
    if batch is None:
        return []
    return list(batch)


def hit_similarity(query_vector: list[float], result: dict[str, Any], index: int) -> float:
    embeddings = _first_batch(result, "embeddings")
    if index < len(embeddings) and embeddings[index] is not None:
        return cosine(query_vector, [float(x) for x in embeddings[index]])
    distances = _first_batch(result, "distances")
    if index >= len(distances):
        return 0.0
    distance = float(distances[index])
    # Unit-vector L2: cosine = 1 - (d^2)/2. Also works as a soft fallback.
    return max(0.0, 1.0 - (distance ** 2) / 2.0)
