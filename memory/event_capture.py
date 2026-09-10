"""Step 1: Event Capture - Append-only raw event logging."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger
from pydantic import BaseModel, Field


class EventRecord(BaseModel):
    """Immutable audit record of runtime agent interactions."""
    event_id: str
    event_type: str
    timestamp: str
    session_id: str
    user_id: str
    agent_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EventLogger:
    """Handles continuous, append-only disk logging of system interaction events."""

    def __init__(self, base_path: str = "data/events"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_data: Dict[str, Any]) -> EventRecord:
        """Persists a single event synchronously to an append-only JSONL log."""
        timestamp = event_data.get("timestamp") or datetime.utcnow().isoformat()
        unique_stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        event_id = event_data.get("event_id") or f"evt_{unique_stamp}"

        record = EventRecord(
            event_id=event_id,
            event_type=event_data["event_type"],
            timestamp=timestamp,
            session_id=event_data["session_id"],
            user_id=event_data["user_id"],
            agent_id=event_data.get("agent_id", "system"),
            payload=event_data.get("payload", {}),
            metadata=event_data.get("metadata", {})
        )

        log_file = self.base_path / f"{record.session_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

        logger.debug(f"[EventLogger] Persisted {record.event_type} ({record.event_id})")
        return record

    def get_events(
        self,
        session_id: str,
        unprocessed_only: bool = False,
        event_types: Optional[List[str]] = None
    ) -> List[EventRecord]:
        """Retrieves raw interaction records for a specific session."""
        log_file = self.base_path / f"{session_id}.jsonl"
        if not log_file.exists():
            return []

        results: List[EventRecord] = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                evt = EventRecord.model_validate_json(stripped)
                if event_types and evt.event_type not in event_types:
                    continue
                if unprocessed_only and evt.metadata.get("processed", False):
                    continue
                results.append(evt)

        return results

    def mark_events_processed(self, event_ids: List[str], session_id: str) -> None:
        """Atomically marks specific events as processed to avoid re-consolidation."""
        log_file = self.base_path / f"{session_id}.jsonl"
        if not log_file.exists() or not event_ids:
            return

        id_set = set(event_ids)
        all_events: List[EventRecord] = []

        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    evt = EventRecord.model_validate_json(stripped)
                    if evt.event_id in id_set:
                        evt.metadata["processed"] = True
                        evt.metadata["processed_at"] = datetime.utcnow().isoformat()
                    all_events.append(evt)

        temp_file = self.base_path / f"{session_id}.jsonl.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            for evt in all_events:
                f.write(evt.model_dump_json() + "\n")

        os.replace(temp_file, log_file)
        logger.debug(f"[EventLogger] Marked {len(event_ids)} events as processed in {session_id}")
