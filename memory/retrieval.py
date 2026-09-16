"""Intent-Routed Unified Memory Retrieval with Primary top-K & Secondary low-K budgets."""

import json
import os
from typing import Any, Dict, List, Optional, Set
from loguru import logger
import openai

from memory.key_normalizer import normalize_key
from memory.llm import llm_client, llm_model_fast, llm_no_thinking
from memory.semantic_memory import SemanticMemory
from memory.procedural_memory import ProceduralMemory
from memory.episodic_memory import EpisodicMemory, _agent_aliases
from memory.knowledge_base import KnowledgeBase
from memory.provenance import MemoryMetadata, Provenance
from memory.sensitivity import filter_user_facts_for_agent, user_fact_visible_to_agent


_CHANNELS = ("procedural", "semantic", "episodic")
_AUTHORITY_RANK = {"user": 0, "agent": 1, "shared": 2}


def semantic_recall_scopes(user_id: str, agent_id: str) -> List[str]:
    """Semantic recall: this user OR this agent. Not shared:organization."""
    scopes = [f"user:{user_id}"]
    for alias in _agent_aliases(agent_id):
        scopes.append(f"agent:{alias}")
    return scopes


def scope_kind(scope: str) -> str:
    """user | agent | shared from a scope label."""
    raw = (scope or "").strip().lower()
    if raw.startswith("user:"):
        return "user"
    if raw.startswith("agent:"):
        return "agent"
    return "shared"


def apply_semantic_authority(facts: List[Any], user_id: Optional[str] = None) -> List[Any]:
    """
    Retrieve-time authority, not write-time supersede.

    - Drop leftover org facts that name this user (shared leak).
    - Same fact_key: keep highest authority (user > agent > org).
    - Remaining facts: user first, then agent, then leftover org, then score.
    """
    survivors: List[Any] = []
    seen_ids = set()
    user_marker = (user_id or "").strip().lower()

    for fact in facts:
        fact_id = getattr(fact, "fact_id", None) or id(fact)
        if fact_id in seen_ids:
            continue
        seen_ids.add(fact_id)

        kind = scope_kind(getattr(fact, "scope", "") or "")
        content = (getattr(fact, "content", "") or "").lower()
        if kind == "shared" and user_marker and user_marker in content:
            logger.debug(
                f"[Authority] Dropped org leak naming user '{user_id}': "
                f"{content[:60]}"
            )
            continue
        survivors.append(fact)

    best_by_key: Dict[str, Any] = {}
    unmatched: List[Any] = []
    for fact in survivors:
        meta = getattr(fact, "metadata", None) or {}
        key = normalize_key(str(meta.get("fact_key") or "").strip())
        if not key:
            unmatched.append(fact)
            continue
        previous = best_by_key.get(key)
        if previous is None or _authority_better(fact, previous):
            best_by_key[key] = fact

    ranked = list(best_by_key.values()) + unmatched
    ranked.sort(
        key=lambda fact: (
            _AUTHORITY_RANK[scope_kind(getattr(fact, "scope", "") or "")],
            -float((getattr(fact, "metadata", None) or {}).get("_score", 0.0)),
        )
    )
    return ranked


def _authority_better(candidate: Any, incumbent: Any) -> bool:
    cand_rank = _AUTHORITY_RANK[scope_kind(getattr(candidate, "scope", "") or "")]
    inc_rank = _AUTHORITY_RANK[scope_kind(getattr(incumbent, "scope", "") or "")]
    if cand_rank != inc_rank:
        return cand_rank < inc_rank
    cand_score = float((getattr(candidate, "metadata", None) or {}).get("_score", 0.0))
    inc_score = float((getattr(incumbent, "metadata", None) or {}).get("_score", 0.0))
    return cand_score > inc_score

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
                model=llm_model_fast(),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=256,
                **llm_no_thinking(llm_model_fast()),
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
        Retrieves user semantic facts and this agent's semantic facts.
        User facts tagged agent_only stay with the source agent.
        Company handbook is knowledge_base / approved procedures, not Mem0 org.
        """
        routing = self.router.route(query)
        primary = routing.get("primary", "semantic")
        secondaries = routing.get("secondary", [])
        limits = self._calculate_budgets(primary, secondaries)
        logger.info(f"[MemoryRetriever] Retrieval allocations: {limits}")

        recall_scopes = semantic_recall_scopes(user_id, agent_id)
        allowed = set(scopes or recall_scopes)
        semantic_limit = limits["semantic"]

        user_facts = filter_user_facts_for_agent(
            self._search_semantic(
                query,
                owner_user_id=user_id,
                wanted=[f"user:{user_id}"],
                allowed=allowed,
                top_k=semantic_limit,
            ),
            agent_id,
        )
        agent_facts = self._search_semantic(
            query,
            owner_user_id=user_id,
            wanted=[s for s in recall_scopes if s.startswith("agent:")],
            allowed=allowed,
            top_k=max(1, semantic_limit // 2),
        )

        semantic_hits = apply_semantic_authority(
            user_facts + agent_facts,
            user_id=user_id,
        )[: semantic_limit]

        # Curated SOP skills still live as shared:organization in this playground.
        procedural_scopes = list(allowed | {"shared:organization"})

        return {
            "routing": routing,
            "limits": limits,
            "procedural": self.procedural.search(
                query=query,
                scopes=procedural_scopes,
                top_k=limits["procedural"],
            ),
            "semantic": semantic_hits,
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

    def _search_semantic(
        self,
        query: str,
        *,
        owner_user_id: Optional[str],
        wanted: List[str],
        allowed: Set[str],
        top_k: int,
    ) -> List[Any]:
        scopes = [scope for scope in wanted if scope in allowed]
        if not scopes:
            return []
        return self.semantic.search(
            query=query,
            user_id=owner_user_id,
            scopes=scopes,
            top_k=max(1, top_k),
        )

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
        facts = self.semantic.get_all_facts(
            user_id=user_id,
            scopes=semantic_recall_scopes(user_id, agent_id),
        )
        return [
            fact
            for fact in facts
            if scope_kind(getattr(fact, "scope", "") or "") != "user"
            or user_fact_visible_to_agent(fact, agent_id)
        ]

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

    def get_memory_provenance(self, memory_id: str) -> Dict[str, Any]:
        """
        Debug tool: Trace a memory back to its original conversations.
        
        Returns complete lineage including:
        - Original events
        - Episodes that created it
        - Sessions and threads
        - Why it was promoted
        """
        mem0 = getattr(self, "mem0", None) or self.semantic.mem0
        memory = mem0.get(memory_id)

        if not memory:
            return {"error": f"Memory {memory_id} not found"}

        metadata_dict = memory.get("metadata") or {}
        provenance = _provenance_from_fact_metadata(metadata_dict)

        episodes = []
        for episode_id in provenance.source_episode_ids:
            episode = self._get_episode(episode_id)
            if episode:
                episodes.append(episode)

        events = []
        event_logger = getattr(self, "event_logger", None)
        for event_id in provenance.source_event_ids:
            event = event_logger.get_event(event_id) if event_logger and hasattr(event_logger, "get_event") else None
            if event:
                events.append(event)

        conversations = self._format_provenance_conversations(events)

        return {
            "memory_id": memory_id,
            "memory_text": memory.get("memory", ""),
            "metadata": metadata_dict,
            "source_episodes": episodes,
            "source_events": events,
            "original_conversations": conversations,
            "confidence_score": provenance.confidence_score,
            "promotion_reason": provenance.promotion_reason,
            "created_at": provenance.created_at,
            "last_reinforced_at": provenance.last_reinforced_at,
        }

    def _format_provenance_conversations(self, events: List) -> List[Dict]:
        """Format events into readable conversation format"""
        conversations = []
        
        for event in events:
            payload = getattr(event, "payload", None) or getattr(event, "data", {}) or {}
            content = payload.get("content") or payload.get("message") or payload.get("response") or ""
            if event.event_type == "user_message":
                conversations.append({
                    'role': 'user',
                    'content': content,
                    'timestamp': event.timestamp,
                    'session_id': event.session_id,
                    'thread_id': event.thread_id,
                })
            elif event.event_type == "agent_response":
                conversations.append({
                    'role': 'assistant',
                    'content': content,
                    'timestamp': event.timestamp,
                    'session_id': event.session_id,
                    'thread_id': event.thread_id,
                })
        
        return sorted(conversations, key=lambda x: x['timestamp'])
    
    def _get_episode(self, episode_id: str) -> Optional[Dict]:
        """
        Retrieve episode from storage.
        Implement based on your episode storage (markdown files or database).
        """
        from pathlib import Path
        import json
        
        # Assuming episodes stored in data/episodes/
        episode_file = Path(f"data/episodes/{episode_id}.json")
        
        if episode_file.exists():
            with open(episode_file, 'r') as f:
                return json.load(f)
        
        return None


def _provenance_from_fact_metadata(metadata_dict: Dict[str, Any]) -> Provenance:
    """Read lineage from add_fact metadata (nested provenance or flat ids)."""
    raw = metadata_dict.get("provenance")
    if isinstance(raw, dict) and raw:
        return Provenance.from_dict(raw)

    if all(key in metadata_dict for key in ("memory_id", "memory_type", "memory_scope", "status")):
        try:
            return MemoryMetadata.from_dict(metadata_dict).provenance
        except Exception:
            pass

    return Provenance(
        source_episode_ids=list(metadata_dict.get("source_episode_ids") or []),
        source_event_ids=list(metadata_dict.get("source_event_ids") or []),
        source_session_ids=list(metadata_dict.get("source_session_ids") or []),
        source_thread_ids=list(metadata_dict.get("source_thread_ids") or []),
        promotion_reason=str(metadata_dict.get("promotion_reason") or ""),
        confidence_score=float(metadata_dict.get("confidence") or 1.0),
    )
