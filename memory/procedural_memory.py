"""Procedural Memory: Stores Markdown procedures with sidecar JSON metadata."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from loguru import logger
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from memory.vector_util import has_hits, hit_similarity, query_collection


class ProceduralSkill(BaseModel):
    """Metadata schema for versioned markdown skills."""
    skill_id: str
    title: str
    description: str
    scope: str
    status: str  # approved, candidate, superseded, rejected
    version: int = 1
    supersedes_id: Optional[str] = None
    superseded_by: Optional[str] = None
    source_episode_ids: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = "manual"
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    md_file: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProceduralMemory:
    """Manages procedural skills on disk as .md files and indexes them via ChromaDB."""

    def __init__(
        self,
        base_path: str = "data/procedural_skills",
        persist_dir: str = "vector_store"
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="procedural_skills",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("[ProceduralMemory] Initialized disk store and Chroma collection")

    def add_skill(
        self,
        title: str,
        content: str,
        description: str,
        scope: str,
        status: str,
        source_episode_ids: List[str],
        confidence: float,
        trigger: str,
        approved_by: Optional[str] = None,
        supersedes_id: Optional[str] = None
    ) -> ProceduralSkill:
        """Writes markdown file and metadata, indexing the skill."""
        stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        skill_id = f"proc_{stamp}"

        version = 1
        if supersedes_id:
            ancestor = self.get_skill(supersedes_id)
            if ancestor:
                version = ancestor.version + 1

        md_filename = f"{skill_id}.md"
        meta_filename = f"{skill_id}.meta.json"

        skill = ProceduralSkill(
            skill_id=skill_id,
            title=title,
            description=description,
            scope=scope,
            status=status,
            version=version,
            supersedes_id=supersedes_id,
            source_episode_ids=source_episode_ids,
            confidence=confidence,
            created_at=datetime.utcnow().isoformat(),
            created_by=trigger,
            approved_by=approved_by,
            approved_at=datetime.utcnow().isoformat() if approved_by else None,
            md_file=md_filename
        )

        # Write markdown content
        with open(self.base_path / md_filename, "w", encoding="utf-8") as f:
            f.write(content)

        # Write sidecar JSON
        with open(self.base_path / meta_filename, "w", encoding="utf-8") as f:
            f.write(skill.model_dump_json(indent=2))

        # Vector Indexing
        indexed_text = f"Title: {title}\nDescription: {description}\n\n{content}"
        vector = self.encoder.encode(indexed_text).tolist()

        self.collection.upsert(
            ids=[skill_id],
            embeddings=[vector],
            metadatas=[{
                "skill_id": skill_id,
                "title": title,
                "scope": scope,
                "status": status,
                "version": version
            }],
            documents=[indexed_text]
        )

        if supersedes_id:
            self._mark_superseded(supersedes_id, successor_id=skill_id)

        logger.info(f"[ProceduralMemory] Skill saved: {skill_id} ('{title}')")
        return skill

    def _mark_superseded(self, old_skill_id: str, successor_id: str) -> None:
        """Updates predecessor metadata and purges it from search index."""
        old_skill = self.get_skill(old_skill_id)
        if not old_skill:
            return

        old_skill.status = "superseded"
        old_skill.superseded_by = successor_id

        meta_path = self.base_path / f"{old_skill_id}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(old_skill.model_dump_json(indent=2))

        # Drop from index to keep reads deduplicated
        try:
            self.collection.delete(ids=[old_skill_id])
            logger.info(f"[ProceduralMemory] Removed superseded skill {old_skill_id} from vector index")
        except Exception as exc:
            logger.warning(f"[ProceduralMemory] Index purge exception: {exc}")

    def update_skill_status(
        self,
        skill_id: str,
        status: str,
        approved_by: Optional[str] = None
    ) -> None:
        """Promotes or alters lifecycle status of a procedure."""
        skill = self.get_skill(skill_id)
        if not skill:
            raise KeyError(f"Procedural skill {skill_id} not found")

        skill.status = status
        if approved_by:
            skill.approved_by = approved_by
            skill.approved_at = datetime.utcnow().isoformat()

        meta_path = self.base_path / f"{skill_id}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(skill.model_dump_json(indent=2))

        self.collection.update(
            ids=[skill_id],
            metadatas=[{
                "skill_id": skill.skill_id,
                "title": skill.title,
                "scope": skill.scope,
                "status": skill.status,
                "version": skill.version
            }]
        )
        logger.info(f"[ProceduralMemory] Skill {skill_id} status updated to {status}")

    def search(
        self,
        query: str,
        scopes: List[str],
        top_k: int = 10,
        threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """Searches indexed skills, lazily fetching Markdown content only for matches."""
        query_vector = self.encoder.encode(query).tolist()
        res = query_collection(
            self.collection,
            query_vector,
            n_results=top_k * 2,
            where={"status": {"$in": ["approved", "candidate"]}},
        )

        matches: List[Dict[str, Any]] = []
        if not has_hits(res):
            return matches

        for idx, sid in enumerate(res["ids"][0]):
            similarity = hit_similarity(query_vector, res, idx)
            if similarity < threshold:
                continue

            skill = self.get_skill(sid)
            if not skill or skill.scope not in scopes:
                continue

            content = self.get_skill_content(sid)
            matches.append({
                "skill_id": skill.skill_id,
                "title": skill.title,
                "description": skill.description,
                "content": content,
                "status": skill.status,
                "version": skill.version,
                "scope": skill.scope,
                "score": similarity
            })

            if len(matches) >= top_k:
                break

        return matches

    def get_skill(self, skill_id: str) -> Optional[ProceduralSkill]:
        """Loads skill metadata from disk."""
        meta_file = self.base_path / f"{skill_id}.meta.json"
        if not meta_file.exists():
            return None
        with open(meta_file, "r", encoding="utf-8") as f:
            return ProceduralSkill.model_validate_json(f.read())

    def get_skill_content(self, skill_id: str) -> str:
        """Reads the Markdown file for a skill."""
        skill = self.get_skill(skill_id)
        if not skill:
            return ""
        md_path = self.base_path / skill.md_file
        if not md_path.exists():
            return ""
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_all_skills(self, scopes: List[str]) -> List[ProceduralSkill]:
        """Retrieves all registered skills within authorized scopes."""
        skills: List[ProceduralSkill] = []
        for meta_file in self.base_path.glob("*.meta.json"):
            with open(meta_file, "r", encoding="utf-8") as f:
                sk = ProceduralSkill.model_validate_json(f.read())
                if sk.scope in scopes:
                    skills.append(sk)
        return sorted(skills, key=lambda s: s.created_at, reverse=True)
