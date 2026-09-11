"""
Consolidation Scheduler - Auto-trigger consolidation based on conversation signals.

Triggers:
1. End signal detection ("bye", "thanks", etc.)
2. Inactivity timeout (default 5 min)
3. Manual (existing /consolidate command)
"""

import threading
from typing import Optional, Dict
from datetime import datetime
from loguru import logger


class ConsolidationScheduler:
    """
    Manages when consolidation happens automatically.
    
    Usage:
        scheduler = ConsolidationScheduler()
        
        # After each message:
        should_consolidate = scheduler.on_message(session_id, user_message)
        
        if should_consolidate:
            _trigger_consolidation()
    """
    
    def __init__(self, inactivity_timeout: int = 300):
        """
        Args:
            inactivity_timeout: Seconds of inactivity before auto-consolidate (default 5 min)
        """
        self.inactivity_timeout = inactivity_timeout
        self.timers: Dict[str, threading.Timer] = {}
        self.last_activity: Dict[str, datetime] = {}
        
        logger.info(
            f"[ConsolidationScheduler] Initialized with "
            f"{inactivity_timeout}s inactivity timeout"
        )
    
    def on_message(
        self,
        session_id: str,
        message_content: str,
        message_type: str = "user"
    ) -> bool:
        """
        Call this after EVERY message (user or agent).
        
        Args:
            session_id: Current session ID
            message_content: The message text
            message_type: "user" or "agent"
        
        Returns:
            True if consolidation should happen NOW, False otherwise
        """
        # Update activity timestamp
        self.last_activity[session_id] = datetime.utcnow()
        
        # Cancel existing timer (user is active)
        self._cancel_timer(session_id)
        
        # Check for end signal (only on user messages)
        if message_type == "user" and self._is_end_signal(message_content):
            logger.info(
                f"[ConsolidationScheduler] End signal detected in session "
                f"{session_id[:8]}: '{message_content[:30]}...'"
            )
            return True  # Consolidate NOW
        
        # Start new inactivity timer
        self._start_timer(session_id)
        
        return False  # Don't consolidate yet
    
    def force_consolidate(self, session_id: str) -> None:
        """
        Manually mark session for consolidation (called by /consolidate, /exit, etc.)
        Cancels the timer since we're consolidating anyway.
        """
        self._cancel_timer(session_id)
        if session_id in self.last_activity:
            del self.last_activity[session_id]
    
    def _is_end_signal(self, message: str) -> bool:
        """
        Detect if message indicates conversation end.
        
        Signals:
        - "bye", "goodbye"
        - "thanks", "thank you" (as final message)
        - "that's all", "done"
        """
        content = message.lower().strip()
        
        # Exact matches
        exact_signals = [
            "bye", "goodbye", "thanks", "thank you",
            "thanks bye", "goodbye thanks", "that's all",
            "done", "ok bye", "see you"
        ]
        
        if content in exact_signals:
            return True
        
        # Ends with signal (e.g., "perfect, thanks!")
        end_signals = [
            "bye", "goodbye", "thanks", "thank you",
            "that's all", "see you"
        ]
        
        if any(content.endswith(signal) for signal in end_signals):
            return True
        
        # Special case: "thanks" or "thank you" followed by punctuation only
        if content in ["thanks!", "thank you!", "thanks.", "thank you."]:
            return True
        
        return False
    
    def _start_timer(self, session_id: str) -> None:
        """Start inactivity countdown."""
        
        def timeout_handler():
            logger.info(
                f"[ConsolidationScheduler] Inactivity timeout triggered for "
                f"session {session_id[:8]} ({self.inactivity_timeout}s)"
            )
            # NOTE: We don't consolidate here - we just log
            # The terminal_chat.py will check if consolidation is needed
            # This is because we can't call _trigger_consolidation from here
            # (it's in TerminalApp instance)
            
            # Instead, we'll use a callback pattern (see below)
        
        timer = threading.Timer(self.inactivity_timeout, timeout_handler)
        timer.daemon = True  # Don't block shutdown
        timer.start()
        
        self.timers[session_id] = timer
        
        logger.debug(
            f"[ConsolidationScheduler] Timer started for {session_id[:8]} "
            f"({self.inactivity_timeout}s)"
        )
    
    def _cancel_timer(self, session_id: str) -> None:
        """Cancel pending timer."""
        if session_id in self.timers:
            self.timers[session_id].cancel()
            del self.timers[session_id]
            logger.debug(f"[ConsolidationScheduler] Timer cancelled for {session_id[:8]}")
    
    def has_timed_out(self, session_id: str) -> bool:
        """
        Check if session has been inactive longer than timeout.
        Call this at the START of new messages to catch timeouts.
        """
        if session_id not in self.last_activity:
            return False
        
        elapsed = (datetime.utcnow() - self.last_activity[session_id]).total_seconds()
        return elapsed >= self.inactivity_timeout
    
    def cleanup(self) -> None:
        """Cancel all timers (call on shutdown)."""
        for session_id in list(self.timers.keys()):
            self._cancel_timer(session_id)
        logger.info("[ConsolidationScheduler] Cleanup complete")