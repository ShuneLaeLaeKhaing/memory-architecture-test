"""Step 2: Consolidation - Transform raw event streams into structured Episodes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger
import openai
from pydantic import BaseModel, Field

from memory.event_capture import EventLogger, EventRecord
from memory.llm import llm_client, llm_model


class EpisodeRecord(BaseModel):
    """Consolidated unit of experience synthesized from multiple causal events."""
    episode_id: str
    session_id: str
    user_id: str
    agent_id: str
    timestamp: str
    goal: str
    outcome: str  # success, failed, abandoned, correction
    summary: str
    trajectory: List[str] = Field(default_factory=list)
    lessons: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    source_event_ids: List[str] = Field(default_factory=list)
    retention_score: float
    scope: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    promoted_to: List[Dict[str, Any]] = Field(default_factory=list)


class ConsolidationEngine:
    """Segments, filters, and synthesizes event logs into structured memory records."""

    def __init__(
        self,
        episodes_path: str = "data/episodes",
        event_logger: Optional[EventLogger] = None,
        promotion_engine: Optional[Any] = None
    ):
        self.base_path = Path(episodes_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.event_logger = event_logger or EventLogger()
        self._promotion_engine = promotion_engine

    def set_promotion_engine(self, engine: Any) -> None:
        """Binds the promotion engine after circular setup if necessary."""
        self._promotion_engine = engine

    def consolidate_session(self, session_id: str, user_id: str) -> List[EpisodeRecord]:
        """Runs segmentation, retention scoring, LLM synthesis, and promotion."""
        events = self.event_logger.get_events(session_id=session_id, unprocessed_only=True)
        if not events:
            logger.info(f"[ConsolidationEngine] No unprocessed events found for session: {session_id}")
            return []

        logger.info(f"[ConsolidationEngine] Processing {len(events)} events for session {session_id}")
        segments = self._segment_events(events)

        created_episodes: List[EpisodeRecord] = []
        for segment in segments:
            retention_score, should_retain = self._evaluate_retention(segment)
            segment["retention_score"] = retention_score
            if not should_retain:
                logger.debug(f"[ConsolidationEngine] Dropped low-retention segment (score={retention_score:.2f})")
                continue

            episode = self._synthesize_episode(segment, user_id)
            self._save_episode(episode)
            self.event_logger.mark_events_processed(
                event_ids=episode.source_event_ids,
                session_id=session_id
            )
            created_episodes.append(episode)

        if created_episodes and self._promotion_engine:
            logger.info(f"[ConsolidationEngine] Triggering promotion analysis on {len(created_episodes)} episodes")
            self._promotion_engine.detect_promotions(created_episodes)

        return created_episodes

    def _segment_events(self, events: List[EventRecord]) -> List[Dict[str, Any]]:
        """LLM-powered segmentation with conversation context."""
        if not events:
            return []
        
        segments: List[Dict[str, Any]] = []
        current_chunk: List[EventRecord] = []
        previous_user_messages: List[str] = []  # ← Track conversation
        
        for evt in events:
            current_chunk.append(evt)
            
            # Check intent on user messages
            if evt.event_type == "user_message":
                content = evt.payload.get("content", "")
                
                # Classify with context
                intent = self._classify_intent(content, previous_user_messages)
                
                # Split if NEW topic detected
                if intent.startswith("new_") and len(current_chunk) > 1:
                    # Remove current event, save segment, start new
                    current_chunk.pop()  # Remove current message
                    
                    logger.info(f"[Segmentation] New topic detected: '{intent}'")
                    segments.append({
                        "events": list(current_chunk),
                        "start": current_chunk[0].timestamp,
                        "end": current_chunk[-1].timestamp
                    })
                    
                    current_chunk = [evt]  # Start new segment with current event
                    previous_user_messages.clear()
                
                # Track user messages for context
                previous_user_messages.append(content)
                if len(previous_user_messages) > 3:
                    previous_user_messages.pop(0)  # Keep last 3
            
            # Terminal events
            if evt.event_type in ["session_end", "user_feedback"]:
                if current_chunk:
                    segments.append({
                        "events": list(current_chunk),
                        "start": current_chunk[0].timestamp,
                        "end": current_chunk[-1].timestamp
                    })
                    current_chunk.clear()
                    previous_user_messages.clear()
            
            # Safety limit
            elif len(current_chunk) >= 15:
                segments.append({
                    "events": list(current_chunk),
                    "start": current_chunk[0].timestamp,
                    "end": current_chunk[-1].timestamp
                })
                current_chunk.clear()
                previous_user_messages.clear()
        
        # Remaining
        if current_chunk:
            segments.append({
                "events": list(current_chunk),
                "start": current_chunk[0].timestamp,
                "end": current_chunk[-1].timestamp
            })
        
        logger.info(f"[Segmentation] Created {len(segments)} segment(s)")
        return segments


    def _classify_intent(self, user_message: str, previous_messages: List[str] = None) -> str:
        """
        LLM-based intent classification with conversation context.
        
        Args:
            user_message: Current message
            previous_messages: Last 2-3 messages for context
        
        Returns:
            Intent label for boundary detection
        """
        # Build context
        context = ""
        if previous_messages:
            context = "\n".join([f"- {msg}" for msg in previous_messages[-3:]])
        
        prompt = f"""Determine if this message starts a NEW topic or continues the PREVIOUS one.

    Previous messages:
    {context if context else "(start of conversation)"}

    Current message:
    "{user_message}"

    Rules:
    - "Remember X" → always NEW topic
    - Follow-up/clarification → SAME topic
    - Error correction ("no I mean...") → SAME topic
    - Completely different subject → NEW topic

    Return ONE of:
    - "continue" (if same topic as previous)
    - "new_[topic]" (if new topic, name it: new_pto, new_github, new_workday, etc.)

    Return ONLY the label:"""

        try:
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            intent = resp.choices[0].message.content.strip().lower()
            logger.debug(f"[Intent] '{user_message[:30]}...' → '{intent}'")
            return intent
        
        except Exception as exc:
            logger.warning(f"[Intent] LLM failed: {exc}, using fallback")
            # Fallback to simple heuristic
            if user_message.lower().startswith("remember"):
                return f"new_remember_{hash(user_message) % 100}"
            return "continue"

    def _evaluate_retention(self, segment: Dict[str, Any]) -> tuple[float, bool]:
        """
        Calculate retention score.
        
        New logic: Keep most multi-turn conversations by default.
        Promotion engine will handle quality filtering.
        """
        events: List[EventRecord] = segment["events"]
        score = 0.0
        
        # Combine all text
        all_text = " ".join([
            str(e.payload.get("content", "")).lower() 
            for e in events 
            if e.event_type in ["user_message", "agent_response"]
        ])
        
        # --- Strong signals (always keep) ---
        
        # Explicit memory requests
        memory_keywords = ["remember", "save", "note", "policy", "rule", "always", "never"]
        if any(kw in all_text for kw in memory_keywords):
            score += 2.0
        
        # User feedback
        if any(e.event_type == "user_feedback" for e in events):
            score += 2.0
        
        # Failures/errors (learn from mistakes)
        if any(e.event_type == "error" for e in events):
            score += 1.5
        
        # --- Correction signals (NEW - this fixes your issue) ---
        
        correction_phrases = [
            "actually", "no i mean", "correction", "switched to",
            "changed to", "now using", "instead of", "not anymore"
        ]
        if any(phrase in all_text for phrase in correction_phrases):
            score += 1.5  # High value - corrections are important
        
        # --- Preference/identity signals ---
        
        preference_words = ["prefer", "like", "want", "need", "my manager", "my team"]
        if any(word in all_text for word in preference_words):
            score += 1.0
        
        # --- Complexity signals ---
        
        # Multi-turn conversation
        if len(events) >= 4:
            score += 0.5
        
        # Tool usage
        if any(e.event_type == "tool_call" for e in events):
            score += 0.5
        
        # --- Negative signals ---
        
        # Single turn (probably trivial)
        if len(events) <= 2:
            score -= 0.3
        
        # Very short messages
        word_count = len(all_text.split())
        if word_count < 10:
            score -= 0.3
        
        final_score = max(0.0, score)
        threshold = 0.5
        
        return final_score, final_score >= threshold

    def _synthesize_episode(self, segment: Dict[str, Any], user_id: str) -> EpisodeRecord:
        """Invokes structured LLM extraction to generate an EpisodeRecord."""
        events: List[EventRecord] = segment["events"]
        transcript = "\n".join([
            f"[{e.timestamp}] {e.event_type} ({e.user_id}->{e.agent_id}): {e.payload.get('content', '')}"
            for e in events
        ])

        system_prompt = (
            "You are a conversation memory synthesizer. Analyze the raw conversation logs and extract:\n"
            "1. goal: What was the primary objective of the user?\n"
            "2. outcome: Outcome class ('success', 'failed', 'abandoned', 'correction').\n"
            "3. summary: 2-3 concise summary sentences.\n"
            "4. trajectory: Array of atomic logical steps executed.\n"
            "5. lessons: Generalizable principles or facts learned.\n"
            "6. entities: Important nouns, systems, or people referenced.\n\n"
            "Return valid JSON matching this schema exactly."
        )

        try:
            # client = openai.OpenAI()
            # response = client.chat.completions.create(
            #     model="gpt-4o-mini",
            #     messages=[
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": f"Interaction Logs:\n{transcript}"}
            #     ],
            #     response_format={"type": "json_object"},
            #     temperature=0.2
            # )
            client = llm_client()
            response = client.chat.completions.create(
                model=llm_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Interaction Logs:\n{transcript}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            parsed = json.loads(response.choices[0].message.content)
        except Exception as exc:
            logger.error(f"[ConsolidationEngine] Synthesis failure: {exc}")
            parsed = {
                "goal": "Unspecified conversational inquiry",
                "outcome": "success",
                "summary": "Interaction occurred between user and agent.",
                "trajectory": ["Interaction processed"],
                "lessons": [],
                "entities": []
            }

        stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        episode = EpisodeRecord(
            episode_id=f"ep_{stamp}",
            session_id=events[0].session_id,
            user_id=user_id,
            agent_id=events[0].agent_id,
            timestamp=events[0].timestamp,
            goal=parsed.get("goal", "Execute query"),
            outcome=parsed.get("outcome", "success"),
            summary=parsed.get("summary", ""),
            trajectory=parsed.get("trajectory", []),
            lessons=parsed.get("lessons", []),
            entities=parsed.get("entities", []),
            source_event_ids=[e.event_id for e in events],
            retention_score=segment["retention_score"],
            scope=f"user:{user_id}"
        )
        return episode

    def _save_episode(self, episode: EpisodeRecord) -> None:
        """Writes episode JSON to disk."""
        dest = self.base_path / f"{episode.episode_id}.json"
        with open(dest, "w", encoding="utf-8") as f:
            f.write(episode.model_dump_json(indent=2))
        logger.info(f"[ConsolidationEngine] Saved Episode {episode.episode_id} to disk")
