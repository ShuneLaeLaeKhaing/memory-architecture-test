"""Knowledge Base: Manages agent-specific reference markdown documents."""

import json
from pathlib import Path
from typing import Any, Dict, List
import chromadb
from loguru import logger
from sentence_transformers import SentenceTransformer

from memory.vector_util import has_hits, hit_similarity, query_collection


class KnowledgeBase:
    """Stores static organizational reference manuals and guides."""

    def __init__(
        self,
        base_path: str = "data/knowledge_base",
        persist_dir: str = "vector_store"
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base_catalog",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("[KnowledgeBase] Initialized Knowledge Base repository")

    def add_document(
        self,
        title: str,
        content: str,
        agent_id: str,
        category: str = "general"
    ) -> str:
        """Stores markdown document and vectorizes it."""
        slug = title.lower().replace(" ", "_").replace("/", "_")
        doc_id = f"kb_{agent_id}_{slug}"

        md_file = self.base_path / f"{doc_id}.md"
        meta_file = self.base_path / f"{doc_id}.meta.json"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(content)

        meta = {
            "doc_id": doc_id,
            "title": title,
            "agent_id": agent_id,
            "category": category,
            "md_file": f"{doc_id}.md"
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        vector = self.encoder.encode(content).tolist()
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            metadatas=[{
                "doc_id": doc_id,
                "agent_id": agent_id,
                "title": title,
                "category": category
            }],
            documents=[content]
        )
        logger.info(f"[KnowledgeBase] Stored document '{title}' ({doc_id})")
        return doc_id

    def search(
        self,
        query: str,
        agent_id: str,
        top_k: int = 5,
        threshold: float = 0.60
    ) -> List[Dict[str, Any]]:
        """Vector search filtered by agent identifier."""
        q_vec = self.encoder.encode(query).tolist()
        res = query_collection(
            self.collection,
            q_vec,
            n_results=top_k * 2,
            where={"agent_id": agent_id},
        )

        matches: List[Dict[str, Any]] = []
        if not has_hits(res):
            return matches

        for idx, doc_id in enumerate(res["ids"][0]):
            sim = hit_similarity(q_vec, res, idx)
            if sim < threshold:
                continue

            content = self.get_document_content(doc_id)
            meta = res["metadatas"][0][idx]
            matches.append({
                "doc_id": doc_id,
                "title": meta.get("title", ""),
                "content": content,
                "category": meta.get("category", ""),
                "score": sim
            })
            if len(matches) >= top_k:
                break

        return matches

    def get_document_content(self, doc_id: str) -> str:
        """Fetches document markdown content from disk."""
        path = self.base_path / f"{doc_id}.md"
        if not path.exists():
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def get_all_items(self, agent_id: str) -> List[Dict[str, Any]]:
        """Lists all document manifests matching an agent ID."""
        items: List[Dict[str, Any]] = []
        for meta_file in self.base_path.glob("*.meta.json"):
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("agent_id") == agent_id:
                    items.append(data)
        return items
