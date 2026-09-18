"""
Consolidation Scheduler - Auto-trigger consolidation based on conversation signals.

Triggers:
1. End signal detection ("bye", "thanks", etc.)
2. Inactivity timeout (default 5 min) via on_timeout callback
3. Manual (/consolidate, /new, /exit) — handled by terminal chat
"""

import threading
from datetime import datetime
from typing import Callable, Dict, Optional

from loguru import logger


class ConsolidationScheduler:
    """Tracks activity and decides when consolidation should run."""

    def __init__(
        self,
        inactivity_timeout: int = 300,
        on_timeout: Optional[Callable[[str], None]] = None,
    ):
        self.inactivity_timeout = inactivity_timeout
        self.on_timeout = on_timeout
        self.timers: Dict[str, threading.Timer] = {}
        self.last_activity: Dict[str, datetime] = {}
        self._lock = threading.Lock()

        logger.info(
            f"[ConsolidationScheduler] Initialized with "
            f"{inactivity_timeout}s inactivity timeout"
        )

    def on_message(
        self,
        session_id: str,
        message_content: str,
        message_type: str = "user",
    ) -> bool:
        """
        Call after each user/agent message.

        Returns True if consolidation should happen NOW (user end signal).
        """
        with self._lock:
            self.last_activity[session_id] = datetime.utcnow()
            self._cancel_timer(session_id)

            if message_type == "user" and self._is_end_signal(message_content):
                logger.info(
                    f"[ConsolidationScheduler] End signal in session "
                    f"{session_id[:8]}: '{message_content[:30]}...'"
                )
                return True

            self._start_timer(session_id)
            return False

    def force_consolidate(self, session_id: str) -> None:
        """Cancel idle timer after a consolidation already ran."""
        with self._lock:
            self._cancel_timer(session_id)
            self.last_activity.pop(session_id, None)

    def _is_end_signal(self, message: str) -> bool:
        content = message.lower().strip()
        exact_signals = [
            "bye", "goodbye", "thanks", "thank you",
            "thanks bye", "goodbye thanks", "that's all",
            "done", "ok bye", "see you",
            "thanks!", "thank you!", "thanks.", "thank you.",
        ]
        if content in exact_signals:
            return True
        end_signals = [
            "bye", "goodbye", "thanks", "thank you",
            "that's all", "see you",
        ]
        return any(content.endswith(signal) for signal in end_signals)

    def _start_timer(self, session_id: str) -> None:
        def timeout_handler():
            logger.info(
                f"[ConsolidationScheduler] Inactivity timeout for "
                f"session {session_id[:8]} ({self.inactivity_timeout}s)"
            )
            with self._lock:
                self.timers.pop(session_id, None)
            if self.on_timeout:
                try:
                    self.on_timeout(session_id)
                except Exception as exc:
                    logger.error(
                        f"[ConsolidationScheduler] on_timeout failed: {exc}"
                    )

        timer = threading.Timer(self.inactivity_timeout, timeout_handler)
        timer.daemon = True
        timer.start()
        self.timers[session_id] = timer
        logger.debug(
            f"[ConsolidationScheduler] Timer started for {session_id[:8]} "
            f"({self.inactivity_timeout}s)"
        )

    def _cancel_timer(self, session_id: str) -> None:
        if session_id in self.timers:
            self.timers[session_id].cancel()
            del self.timers[session_id]
            logger.debug(
                f"[ConsolidationScheduler] Timer cancelled for {session_id[:8]}"
            )

    def has_timed_out(self, session_id: str) -> bool:
        if session_id not in self.last_activity:
            return False
        elapsed = (
            datetime.utcnow() - self.last_activity[session_id]
        ).total_seconds()
        return elapsed >= self.inactivity_timeout

    def cleanup(self) -> None:
        with self._lock:
            for session_id in list(self.timers.keys()):
                self._cancel_timer(session_id)
        logger.info("[ConsolidationScheduler] Cleanup complete")
