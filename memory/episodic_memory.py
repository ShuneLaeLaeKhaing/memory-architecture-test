"""Episodic Memory: Search over historical episodes with exponential recency decay."""

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from loguru import logger
from sentence_transformers import SentenceTransformer

from memory.consolidation import EpisodeRecord
from memory.vector_util import has_hits, hit_similarity, query_collection


def _agent_aliases(agent_id: str) -> set:
    """Accept both config key (hr_assistant) and canonical id (hr_assistant_001)."""
    aliases = {agent_id}
    if not agent_id:
        return aliases
    try:
        from config.agents import AGENT_CONFIGS
    except Exception:
        return aliases
    if agent_id in AGENT_CONFIGS:
        aliases.add(AGENT_CONFIGS[agent_id]["agent_id"])
    for key, cfg in AGENT_CONFIGS.items():
        if cfg.get("agent_id") == agent_id:
            aliases.add(key)
    return aliases


class EpisodicMemory:
    """Stores consolidated episodes and handles vector search with recency weighting."""

    def __init__(
        self,
        base_path: str = "data/episodes",
        persist_dir: str = "vector_store"
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="episodes_history",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("[EpisodicMemory] Initialized episodic vector storage")

    def index_episode(self, episode: EpisodeRecord) -> None:
        """Embeds and indexes an episode record."""
        text = f"Goal: {episode.goal}\nSummary: {episode.summary}\nLessons: {', '.join(episode.lessons)}"
        embedding = self.encoder.encode(text).tolist()

        self.collection.upsert(
            ids=[episode.episode_id],
            embeddings=[embedding],
            metadatas=[{
                "episode_id": episode.episode_id,
                "session_id": episode.session_id,
                "user_id": episode.user_id,
                "scope": episode.scope,
                "timestamp": episode.timestamp,
                "outcome": episode.outcome
            }],
            documents=[text]
        )
        logger.info(f"[EpisodicMemory] Indexed episode {episode.episode_id}")

    def search(
        self,
        query: str,
        scopes: Optional[List[str]] = None,
        top_k: int = 5,
        threshold: float = 0.45,
        recency_weight: float = 0.3,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves episodes, reweighting similarity using exponential time decay.
        Score = (1 - w) * Similarity + w * exp(-age_days / 14)
        """
        query_vector = self.encoder.encode(query).tolist()
        res = query_collection(self.collection, query_vector, n_results=top_k * 3)

        ranked: List[Dict[str, Any]] = []
        if not has_hits(res):
            return ranked

        now = datetime.utcnow()
        for idx, ep_id in enumerate(res["ids"][0]):
            similarity = hit_similarity(query_vector, res, idx)
            if similarity < threshold:
                continue

            episode = self.get_episode(ep_id)
            if (
                not episode
                or episode.user_id != user_id
                or episode.agent_id not in _agent_aliases(agent_id)
            ):
                continue

            try:
                ep_time = datetime.fromisoformat(episode.timestamp)
            except Exception:
                ep_time = now

            age_days = max(0, (now - ep_time).days)
            # Exponential decay with 14-day half-life factor
            recency_multiplier = math.exp(-age_days / 14.0)
            composite_score = ((1.0 - recency_weight) * similarity) + (recency_weight * recency_multiplier)

            ranked.append({
                "episode_id": episode.episode_id,
                "goal": episode.goal,
                "outcome": episode.outcome,
                "summary": episode.summary,
                "lessons": episode.lessons,
                "timestamp": episode.timestamp,
                "similarity": similarity,
                "composite_score": composite_score
            })

        ranked.sort(key=lambda x: x["composite_score"], reverse=True)
        return ranked[:top_k]

    def get_episode(self, episode_id: str) -> Optional[EpisodeRecord]:
        """Loads an episode from disk."""
        target = self.base_path / f"{episode_id}.json"
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as f:
            return EpisodeRecord.model_validate_json(f.read())

    def get_all_episodes(
        self,
        user_id: str,
        agent_id: str,
        session_id: Optional[str] = None,
    ) -> List[EpisodeRecord]:
        """Lists this user's episodes for this agent only."""
        results: List[EpisodeRecord] = []
        for item in self.base_path.glob("*.json"):
            with open(item, "r", encoding="utf-8") as f:
                ep = EpisodeRecord.model_validate_json(f.read())
                if ep.user_id != user_id or ep.agent_id not in _agent_aliases(agent_id):
                    continue
                if session_id and ep.session_id != session_id:
                    continue
                results.append(ep)
        return sorted(results, key=lambda e: e.timestamp, reverse=True)
