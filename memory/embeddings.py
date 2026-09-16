"""Shared multilingual encoder. BGE-M3 covers Burmese; MiniLM does not."""

import os
from functools import lru_cache

from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def collection_name(base: str) -> str:
    tag = EMBEDDING_MODEL.rsplit("/", 1)[-1].replace("-", "_").lower()
    return f"{base}_{tag}"


class Encoder:
    """SentenceTransformer wrapper that always L2-normalizes vectors."""

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def encode(self, text):
        return self._model.encode(text, normalize_embeddings=True)


@lru_cache(maxsize=1)
def get_encoder() -> Encoder:
    """Load BGE-M3 once. First run downloads ~2GB."""
    logger.info(f"[embeddings] Loading {EMBEDDING_MODEL} (first run can take a few minutes)")
    AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"[embeddings] {EMBEDDING_MODEL} ready")
    return Encoder(model)
