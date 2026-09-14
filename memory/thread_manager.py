from datetime import datetime
from pathlib import Path
import json
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger


class Thread(BaseModel):
    """Persistent conversation thread (like Claude sidebar item)."""
    thread_id: str
    user_id: str
    agent_id: str
    title: str  # Auto-generated from first message
    created_at: str
    last_activity: str
    status: str = "active"  # active, archived, deleted
    message_count: int = 0
    session_ids: List[str] = []  # All sessions in this thread
    metadata: dict = {}


class ThreadManager:
    """Manages persistent conversation threads."""
    
    def __init__(self, threads_path: str = "data/threads"):
        self.base_path = Path(threads_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_thread(
        self,
        user_id: str,
        agent_id: str,
        initial_message: str
    ) -> Thread:
        """Create new thread with auto-generated title."""
        from uuid import uuid4
        
        thread_id = f"thread_{uuid4().hex[:12]}"
        
        # Generate title from first message (simple version)
        title = self._generate_title(initial_message)
        
        thread = Thread(
            thread_id=thread_id,
            user_id=user_id,
            agent_id=agent_id,
            title=title,
            created_at=datetime.utcnow().isoformat(),
            last_activity=datetime.utcnow().isoformat(),
            status="active",
            message_count=1
        )
        
        self._save_thread(thread)
        logger.info(f"[ThreadManager] Created thread {thread_id}: '{title}'")
        return thread
    
    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Load thread by ID."""
        path = self.base_path / f"{thread_id}.json"
        if not path.exists():
            return None
        return self._load_thread_file(path)

    def list_threads(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        status: str = "active"
    ) -> List[Thread]:
        """List all threads for user (for sidebar)."""
        threads = []

        for path in self.base_path.glob("*.json"):
            thread = self._load_thread_file(path)
            if thread is None:
                continue

            # Filter
            if thread.user_id != user_id:
                continue
            if agent_id and thread.agent_id != agent_id:
                continue
            if thread.status != status:
                continue

            threads.append(thread)

        # Sort by last_activity (newest first)
        threads.sort(key=lambda t: t.last_activity, reverse=True)
        return threads

    def _load_thread_file(self, path: Path) -> Optional[Thread]:
        """Skip empty or corrupt thread files from a failed save."""
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                logger.warning(f"[ThreadManager] Skipping empty thread file: {path.name}")
                return None
            return Thread(**json.loads(raw))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning(f"[ThreadManager] Skipping bad thread file {path.name}: {exc}")
            return None
    
    def update_thread(self, thread: Thread) -> None:
        """Save updated thread."""
        self._save_thread(thread)
    
    def add_session_to_thread(self, thread_id: str, session_id: str) -> None:
        """Track session in thread."""
        thread = self.get_thread(thread_id)
        if thread:
            thread.session_ids.append(session_id)
            thread.last_activity = datetime.utcnow().isoformat()
            self._save_thread(thread)
    
    def increment_message_count(self, thread_id: str) -> None:
        """Increment message counter."""
        thread = self.get_thread(thread_id)
        if thread:
            thread.message_count += 1
            thread.last_activity = datetime.utcnow().isoformat()
            self._save_thread(thread)
    
    def _save_thread(self, thread: Thread) -> None:
        """Write thread to disk."""
        path = self.base_path / f"{thread.thread_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(thread.model_dump_json(indent=2))
    
    def _generate_title(self, message: str) -> str:
        """Generate thread title from first message."""
        # Simple version: truncate
        if len(message) <= 40:
            return message
        return message[:37] + "..."
        
        # TODO: Later, use LLM to generate better titles
        # e.g., "PTO Request Help" instead of "How do I request PTO and what..."