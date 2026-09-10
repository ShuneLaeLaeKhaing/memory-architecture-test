"""Intent-Routed Unified Memory Retrieval with Primary top-K & Secondary low-K budgets."""

import json
import os
from typing import Any, Dict, List, Optional
from loguru import logger
import openai

from memory.llm import llm_client, llm_model
from memory.semantic_memory import SemanticMemory
from memory.procedural_memory import ProceduralMemory
from memory.episodic_memory import EpisodicMemory, _agent_aliases
from memory.knowledge_base import KnowledgeBase


def semantic_recall_scopes(user_id: str, agent_id: str) -> List[str]:
    """Same scopes chat recall uses: this user, this agent, shared."""
    scopes = [f"user:{user_id}", "shared:organization"]
    for alias in _agent_aliases(agent_id):
        scopes.append(f"agent:{alias}")
    return scopes

_CHANNELS = ("procedural", "semantic", "episodic")


# def _openai_client() -> openai.OpenAI:
#     key = os.getenv("OPENAI_API_KEY", "")
#     if key.startswith("sk-or-"):
#         return openai.OpenAI(
#             api_key=key,
#             base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
#         )
#     return openai.OpenAI()
#
#
# def _llm_model() -> str:
#     model = os.getenv("LLM_MODEL", "gpt-4o-mini")
#     key = os.getenv("OPENAI_API_KEY", "")
#     if key.startswith("sk-or-") and "/" not in model:
#         return f"openai/{model}"
#     return model


def _rule_route(query: str) -> Dict[str, Any]:
    """Same three channels as the router prompt."""
    low = query.lower()
    if any(marker in low for marker in ("how do i", "how to", "steps to", "procedure", "workflow")):
        return {"primary": "procedural", "secondary": ["semantic", "episodic"], "confidence": 0.7}
    if any(marker in low for marker in ("last time", "yesterday", "we discussed", "previous conversation", "remember when")):
        return {"primary": "episodic", "secondary": ["semantic", "procedural"], "confidence": 0.7}
    return {"primary": "semantic", "secondary": ["procedural", "episodic"], "confidence": 0.5}


class IntentRouter:
    """Classifies user inquiries to configure asymmetric retrieval budgets."""

    def route(self, query: str) -> Dict[str, Any]:
        """Maps query to primary and secondary memory channels."""
        prompt = (
            "Analyze the following user query for memory retrieval routing:\n\n"
            f"Query: \"{query}\"\n\n"
            "Classify intent into:\n"
            "- 'procedural': User asks 'how to', workflows, multi-step actions.\n"
            "- 'semantic': User asks for direct facts, rules, identity, parameters.\n"
            "- 'episodic': User mentions past conversations, prior discussions.\n\n"
            "Output JSON format:\n"
            "{\n"
            "  \"primary\": \"procedural\" | \"semantic\" | \"episodic\",\n"
            "  \"secondary\": [\"semantic\", \"episodic\"],\n"
            "  \"confidence\": 0.0 to 1.0\n"
            "}"
        )

        try:
            # client = _openai_client()
            # res = client.chat.completions.create(
            #     model=_llm_model(),
            #     messages=[{"role": "user", "content": prompt}],
            #     response_format={"type": "json_object"},
            #     temperature=0.0
            # )
            client = llm_client()
            res = client.chat.completions.create(
                model=llm_model(),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(res.choices[0].message.content)
            if data.get("primary") not in _CHANNELS:
                raise ValueError(f"Invalid primary channel: {data.get('primary')}")
            logger.debug(f"[IntentRouter] Query routed -> Primary: {data.get('primary')}")
            return data
        except Exception as exc:
            logger.warning(f"[IntentRouter] Routing fallback triggered: {exc}")
            return _rule_route(query)


class MemoryRetriever:
    """Coordinates retrieval across all memory systems according to routed intent budgets."""

    BUDGETS = {
        "primary": {"procedural": 8, "semantic": 10, "episodic": 5},
        "secondary": {"procedural": 3, "semantic": 4, "episodic": 2},
        "tertiary": {"procedural": 1, "semantic": 2, "episodic": 1}
    }

    def __init__(
        self,
        semantic: Optional[SemanticMemory] = None,
        procedural: Optional[ProceduralMemory] = None,
        episodic: Optional[EpisodicMemory] = None,
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        self.semantic = semantic or SemanticMemory()
        self.procedural = procedural or ProceduralMemory()
        self.episodic = episodic or EpisodicMemory()
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.router = IntentRouter()


    def retrieve(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        session_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves context from all scopes:
        - User-specific facts
        - Agent-learned patterns
        - Organization-wide knowledge
        """
        routing = self.router.route(query)
        primary = routing.get("primary", "semantic")
        secondaries = routing.get("secondary", [])
        limits = self._calculate_budgets(primary, secondaries)
        logger.info(f"[MemoryRetriever] Retrieval allocations: {limits}")

        recall_scopes = semantic_recall_scopes(user_id, agent_id)
        allowed = scopes or recall_scopes
        user_shared = [s for s in recall_scopes if not s.startswith("agent:")]
        agent_scopes = [s for s in recall_scopes if s.startswith("agent:")]

        user_facts = self.semantic.search(
            query=query,
            user_id=user_id,
            scopes=user_shared,
            top_k=max(1, limits["semantic"] // 2),
        )
        agent_facts = self.semantic.search(
            query=query,
            user_id=user_id,
            scopes=agent_scopes,
            top_k=max(1, limits["semantic"] // 4),
        )

        seen_ids = set()
        semantic_hits: List[Any] = []
        for fact in user_facts + agent_facts:
            if fact.fact_id in seen_ids:
                continue
            seen_ids.add(fact.fact_id)
            semantic_hits.append(fact)
        semantic_hits.sort(key=lambda fact: float(fact.metadata.get("_score", 0.0)), reverse=True)

        return {
            "routing": routing,
            "limits": limits,
            "procedural": self.procedural.search(
                query=query,
                scopes=allowed,
                top_k=limits["procedural"],
            ),
            "semantic": semantic_hits[: limits["semantic"]],
            "episodic": self.episodic.search(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                top_k=limits["episodic"],
            ),
            "knowledge_base": self.knowledge_base.search(
                query=query,
                agent_id=agent_id,
                top_k=4,
            ),
        }

    def _calculate_budgets(self, primary: str, secondaries: List[str]) -> Dict[str, int]:
        limits: Dict[str, int] = {}
        for channel in ["procedural", "semantic", "episodic"]:
            if channel == primary:
                limits[channel] = self.BUDGETS["primary"][channel]
            elif channel in secondaries:
                limits[channel] = self.BUDGETS["secondary"][channel]
            else:
                limits[channel] = self.BUDGETS["tertiary"][channel]
        return limits

    def get_semantic_facts(self, user_id: str, agent_id: str):
        return self.semantic.get_all_facts(
            user_id=user_id,
            scopes=semantic_recall_scopes(user_id, agent_id),
        )

    def get_procedural_skills(self, scopes: List[str]):
        return self.procedural.get_all_skills(scopes=scopes)

    def get_episodes(
        self,
        user_id: str,
        agent_id: str,
        session_id: Optional[str] = None,
    ):
        return self.episodic.get_all_episodes(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

    def get_knowledge_base_items(self, agent_id: str):
        return self.knowledge_base.get_all_items(agent_id=agent_id)
