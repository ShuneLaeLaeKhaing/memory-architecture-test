"""Simple Agent: Assembles memory context without graph orchestration overhead."""

from typing import Any, Dict, List, Optional
from loguru import logger
import openai

from memory.event_capture import EventLogger
from memory.llm import llm_client, llm_model_fast, llm_no_thinking
from memory.retrieval import MemoryRetriever


class SimpleAgent:
    """Direct LLM execution harness incorporating authoritative memory context."""

    def __init__(
        self,
        agent_config: Dict[str, Any],
        retriever: MemoryRetriever,
        event_logger: Optional[EventLogger] = None,
    ):
        self.config = agent_config
        self.retriever = retriever
        self.event_logger = event_logger or EventLogger()

    def respond(
        self, 
        user_message: str, 
        user_id: str, 
        session_id: str,
        thread_id: Optional[str] = None
    ) -> str:
        """Retrieves targeted memory and conversation history."""
        
        scopes = [
            f"user:{user_id}",
            f"agent:{self.config['agent_id']}",
            "shared:organization"
        ]

        # Long-term memory (facts, procedures)
        context = self.retriever.retrieve(
            query=user_message,
            user_id=user_id,
            agent_id=self.config["agent_id"],
            session_id=session_id
        )

        formatted_context = self._build_context_prompt(context)
        
        system_prompt = (
            f"{self.config['role']}\n\n"
            "Operational Guidelines:\n"
            "- Answer using the factual knowledge and official procedures provided.\n"
            "- If a candidate procedure is provided, warn the user it is pending approval.\n"
            "- Reference past user preferences or constraints when relevant.\n"
            "- If you lack sufficient context, be transparent and ask for clarification.\n"
            "- Reply with the user-facing answer only. No analysis, steps, or thinking."
        )
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"AUTHORITATIVE CONTEXT:\n{formatted_context}"},
        ]
        
        # ✅ Thread history (includes all sessions automatically)
        conversation_history = []
        if thread_id:
            conversation_history = self._get_thread_history(
                thread_id=thread_id,
                max_messages=10
            )
            messages.extend(conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})

        logger.debug(
            f"[SimpleAgent] Invoking with {len(conversation_history)} "
            f"history messages for '{user_message[:30]}...'"
        )
        
        client = llm_client()
        resp = client.chat.completions.create(
            model=llm_model_fast(),
            messages=messages,
            temperature=0.4,
            max_tokens=600,
            **llm_no_thinking(),
        )
        
        return resp.choices[0].message.content

    def _get_thread_history(
        self,
        thread_id: str,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation history for this thread.
        
        Includes ALL sessions in the thread (no session boundaries).
        This is how Claude works - thread history persists across browser sessions.
        
        Args:
            thread_id: Thread identifier
            max_messages: Maximum number of recent messages to retrieve
        
        Returns:
            List of messages in OpenAI format [{"role": "user", "content": "..."}]
        """
        events = self.event_logger.get_events_for_thread(
            thread_id=thread_id,
            event_types=["user_message", "agent_response"]
        )
        
        messages = []
        for event in events[-max_messages:]:
            if event.event_type == "user_message":
                messages.append({
                    "role": "user",
                    "content": event.payload.get("content", "")
                })
            elif event.event_type == "agent_response":
                messages.append({
                    "role": "assistant",
                    "content": event.payload.get("content", "")
                })
        
        return messages

    def _build_context_prompt(self, context: Dict[str, Any]) -> str:
        """Formats retrieved items into organized Markdown sections for the LLM prompt."""
        sections: List[str] = []

        # 1. Procedures
        procs = context.get("procedural", [])
        if procs:
            sections.append("### Procedural Guidelines")
            for p in procs:
                status_flag = "[PENDING APPROVAL] " if p.get("status") == "candidate" else ""
                sections.append(f"#### {status_flag}{p['title']}\n{p['content']}\n")

        # 2. Knowledge Base
        kb = context.get("knowledge_base", [])
        if kb:
            sections.append("### Corporate Knowledge Base")
            for doc in kb:
                sections.append(f"#### {doc['title']}\n{doc['content']}\n")

        # 3. Semantic Facts
        facts = context.get("semantic", [])
        if facts:
            sections.append("### Authoritative Facts & Constraints")
            for f in facts:
                sections.append(f"- {f.content}")

        # 4. Episodic Past Experiences
        episodes = context.get("episodic", [])
        if episodes:
            sections.append("### Prior Related Episodes")
            for ep in episodes:
                sections.append(f"- Goal: {ep['goal']} -> Outcome: {ep['outcome']} ({ep['summary']})")

        return "\n\n".join(sections) if sections else "No prior memory context found."