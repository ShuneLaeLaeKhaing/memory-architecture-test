"""Simple Agent: Assembles memory context without graph orchestration overhead."""

from typing import Any, Dict, List
from loguru import logger
import openai

from memory.llm import llm_client, llm_model
from memory.retrieval import MemoryRetriever


class SimpleAgent:
    """Direct LLM execution harness incorporating authoritative memory context."""

    def __init__(self, agent_config: Dict[str, Any], retriever: MemoryRetriever):
        self.config = agent_config
        self.retriever = retriever

    def respond(self, user_message: str, user_id: str, session_id: str) -> str:
        """Retrieves targeted memory across stores and streams LLM completion."""
        scopes = [
            f"user:{user_id}",
            f"agent:{self.config['agent_id']}",
            "shared:organization"
        ]

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
            "- If you lack sufficient context, be transparent and ask for clarification."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"AUTHORITATIVE CONTEXT:\n{formatted_context}"},
            {"role": "user", "content": user_message}
        ]

        logger.debug(f"[SimpleAgent] Invoking OpenAI completion for '{user_message[:30]}...'")
        # client = openai.OpenAI()
        # resp = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=messages,
        #     temperature=0.4
        # )
        client = llm_client()
        resp = client.chat.completions.create(
            model=llm_model(),
            messages=messages,
            temperature=0.4
        )
        return resp.choices[0].message.content

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
