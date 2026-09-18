"""Simple Agent: Assembles memory context without graph orchestration overhead."""

from typing import Any, Dict, List, Optional
from loguru import logger

from memory.conflict_authority import (
    filter_stale_episodes,
    format_conflict_block,
    resolve_preference_conflicts,
)
from memory.event_capture import EventLogger
from memory.llm import llm_client, llm_model_fast, llm_no_thinking
from memory.retrieval import MemoryRetriever, fact_timestamp, scope_kind


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
        thread_id: Optional[str] = None,
    ) -> str:
        """Retrieves targeted memory and conversation history."""
        context = self.retriever.retrieve(
            query=user_message,
            user_id=user_id,
            agent_id=self.config["agent_id"],
            session_id=session_id,
        )

        thread_events = []
        if thread_id:
            thread_events = self.event_logger.get_events_for_thread(
                thread_id=thread_id,
                event_types=["user_message", "agent_response"],
            )

        resolutions = resolve_preference_conflicts(
            facts=context.get("semantic") or [],
            thread_events=thread_events,
            episodes=context.get("episodic") or [],
        )
        if resolutions:
            context = {
                **context,
                "episodic": filter_stale_episodes(
                    context.get("episodic") or [],
                    resolutions,
                ),
            }

        formatted_context = self._build_context_prompt(context)
        conflict_block = format_conflict_block(resolutions)

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "system", "content": f"AUTHORITATIVE CONTEXT:\n{formatted_context}"},
            {"role": "system", "content": conflict_block},
        ]

        conversation_history = self._history_from_events(thread_events, max_messages=10)
        messages.extend(conversation_history)
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
            **llm_no_thinking(llm_model_fast()),
        )
        return resp.choices[0].message.content

    def _history_from_events(
        self,
        events: List[Any],
        max_messages: int = 10,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for event in events[-max_messages:]:
            stamp = (getattr(event, "timestamp", None) or "")[:19]
            prefix = f"[{stamp}] " if stamp else ""
            if event.event_type == "user_message":
                messages.append({
                    "role": "user",
                    "content": prefix + event.payload.get("content", ""),
                })
            elif event.event_type == "agent_response":
                messages.append({
                    "role": "assistant",
                    "content": prefix + event.payload.get("content", ""),
                })
        return messages

    def _get_thread_history(
        self,
        thread_id: str,
        max_messages: int = 10,
    ) -> List[Dict[str, str]]:
        events = self.event_logger.get_events_for_thread(
            thread_id=thread_id,
            event_types=["user_message", "agent_response"],
        )
        return self._history_from_events(events, max_messages=max_messages)

    def _system_prompt(self) -> str:
        return (
            f"{self.config['role']}\n\n"
            "Operational Guidelines:\n"
            "- Answer using the factual knowledge and official procedures provided.\n"
            "- If a candidate procedure is provided, warn the user it is pending approval.\n"
            "- Reference past user preferences or constraints when relevant.\n"
            "- When user-specific memory conflicts with agent or handbook "
            "memory, follow the user-specific memory for this person's "
            "preferences and how to reach them.\n"
            "- Obey RESOLVED PREFERENCE CONFLICTS when present — those are "
            "code-decided winners (clear current chat wins if newer; "
            "otherwise saved fact wins over stale chat/episodes).\n"
            "- When a conflict is listed: say the winning preference AND "
            "briefly acknowledge the losing side once in the same reply "
            "(one short clause). Do not dump memory internals.\n"
            "- If you lack sufficient context, be transparent and ask for clarification.\n"
            "- Reply with the user-facing answer only. No analysis, steps, or thinking."
        )

    def _build_context_prompt(self, context: Dict[str, Any]) -> str:
        """Formats retrieved items into organized Markdown sections for the LLM prompt."""
        sections: List[str] = []

        procs = context.get("procedural", [])
        if procs:
            sections.append("### Procedural Guidelines")
            for p in procs:
                status_flag = "[PENDING APPROVAL] " if p.get("status") == "candidate" else ""
                sections.append(f"#### {status_flag}{p['title']}\n{p['content']}\n")

        kb = context.get("knowledge_base", [])
        if kb:
            sections.append("### Corporate Knowledge Base")
            for doc in kb:
                sections.append(f"#### {doc['title']}\n{doc['content']}\n")

        facts = context.get("semantic", [])
        if facts:
            sections.append("### Authoritative Facts & Constraints")
            sections.append(
                "User-specific facts override agent facts "
                "for this person's preferences and how to contact them. "
                "Each fact includes as_of= timestamp for conflict checks."
            )
            buckets = {"user": [], "agent": [], "shared": []}
            for fact in facts:
                buckets[scope_kind(getattr(fact, "scope", "") or "")].append(fact)
            for heading, kind in (
                ("User-specific", "user"),
                ("Agent-learned", "agent"),
                ("Organization default", "shared"),
            ):
                items = buckets[kind]
                if not items:
                    continue
                sections.append(f"#### {heading}")
                for fact in items:
                    ts = fact_timestamp(fact)
                    as_of = ts.isoformat(timespec="seconds") if ts else "unknown"
                    sections.append(f"- (as_of={as_of}) {fact.content}")

        episodes = context.get("episodic", [])
        if episodes:
            sections.append("### Prior Related Episodes")
            for ep in episodes:
                if isinstance(ep, dict):
                    sections.append(
                        f"- Goal: {ep['goal']} -> Outcome: {ep['outcome']} ({ep['summary']})"
                    )
                else:
                    sections.append(
                        f"- Goal: {ep.goal} -> Outcome: {ep.outcome} ({ep.summary})"
                    )

        return "\n\n".join(sections) if sections else "No prior memory context found."
