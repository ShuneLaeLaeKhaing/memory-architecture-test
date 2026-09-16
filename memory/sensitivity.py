"""User-fact sensitivity: public across agents, or locked to the source agent.

Semantic facts stay user XOR agent. This only tags user-scoped facts.
Missing sensitivity on old facts defaults to public.
"""

from typing import Any, Iterable, List, Optional

from loguru import logger

from memory.episodic_memory import _agent_aliases
from memory.llm import llm_client, llm_model_fast, llm_no_thinking


SENSITIVITY_PUBLIC = "public"
SENSITIVITY_AGENT_ONLY = "agent_only"

_AGENT_ONLY_MARKERS = (
    "salary", "compensation", "base pay", "annual pay", "paycheck",
    "wage", "bonus", "equity grant", "stock grant",
    "medical", "health condition", "diagnosis", "disability",
    "mental health", "medical leave", "sick leave", "pregnancy",
    "performance review", "performance rating", "pip ", "on a pip",
    "terminated", "disciplinary", "write-up",
    "budget denied", "ssn", "social security", "passport",
    "bank account", "password",
)

_PUBLIC_MARKERS = (
    "prefer", "preference", "dark mode", "light mode", "ui theme",
    "slack", "teams", "language", "burmese", "english",
    "notify", "notification", "remote on", "work remotely",
)


def classify_sensitivity(
    memory_text: str,
    agent_id: str = "",
    *,
    use_llm: bool = True,
) -> str:
    """Return public or agent_only. Rules first; LLM only if unknown."""
    ruled = _rule_classify(memory_text)
    if ruled:
        return ruled
    if use_llm:
        judged = _llm_classify(memory_text, agent_id)
        if judged:
            return judged
    logger.info(
        "[Sensitivity] Unknown fact; fail closed to agent_only: "
        f"{(memory_text or '')[:60]}"
    )
    return SENSITIVITY_AGENT_ONLY


def sensitivity_metadata(
    memory_text: str,
    *,
    scope_type: str,
    agent_id: str,
    use_llm: bool = True,
) -> dict:
    """Metadata for a user fact. Agent-scoped lessons get nothing extra."""
    if scope_type != "user":
        return {}
    level = classify_sensitivity(memory_text, agent_id, use_llm=use_llm)
    if level == SENSITIVITY_AGENT_ONLY:
        return {
            "sensitivity": SENSITIVITY_AGENT_ONLY,
            "source_agent_id": agent_id,
            "accessible_by": [agent_id],
        }
    return {
        "sensitivity": SENSITIVITY_PUBLIC,
        "source_agent_id": agent_id,
        "accessible_by": "all",
    }


def user_fact_visible_to_agent(fact: Any, agent_id: str) -> bool:
    """True if this agent may see a user-scoped semantic fact."""
    meta = getattr(fact, "metadata", None) or {}
    if isinstance(fact, dict):
        meta = fact.get("metadata") or meta
    level = str(meta.get("sensitivity") or SENSITIVITY_PUBLIC).strip().lower()
    if level != SENSITIVITY_AGENT_ONLY:
        return True

    allowed = _allowed_agents(meta)
    if not allowed:
        return False
    return bool(_agent_aliases(agent_id) & allowed)


def filter_user_facts_for_agent(facts: Iterable[Any], agent_id: str) -> List[Any]:
    """Drop agent_only user facts the current agent must not see."""
    visible = []
    for fact in facts:
        if user_fact_visible_to_agent(fact, agent_id):
            visible.append(fact)
            continue
        content = getattr(fact, "content", None) or ""
        logger.info(
            f"[Sensitivity] Hid agent_only fact from {agent_id}: {content[:50]}"
        )
    return visible


def _rule_classify(memory_text: str) -> Optional[str]:
    text = (memory_text or "").lower()
    if any(marker in text for marker in _AGENT_ONLY_MARKERS):
        return SENSITIVITY_AGENT_ONLY
    if any(marker in text for marker in _PUBLIC_MARKERS):
        return SENSITIVITY_PUBLIC
    return None


def _llm_classify(memory_text: str, agent_id: str) -> Optional[str]:
    prompt = (
        "Classify this user fact for cross-agent sharing.\n\n"
        f'Fact: "{memory_text}"\n'
        f"Source agent: {agent_id}\n\n"
        "Should other agents (IT, Onboarding, Finance) see this?\n"
        "- Preferences, settings, language, work schedule → public\n"
        "- Salary, health, performance, budget, credentials → agent_only\n\n"
        "Return ONLY: public OR agent_only"
    )
    try:
        client = llm_client()
        res = client.chat.completions.create(
            model=llm_model_fast(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
            **llm_no_thinking(llm_model_fast()),
        )
        raw = (res.choices[0].message.content or "").strip().lower()
        if SENSITIVITY_AGENT_ONLY in raw:
            return SENSITIVITY_AGENT_ONLY
        if SENSITIVITY_PUBLIC in raw:
            return SENSITIVITY_PUBLIC
    except Exception as exc:
        logger.warning(f"[Sensitivity] LLM classify failed: {exc}")
    return None


def _allowed_agents(meta: dict) -> set:
    names: set = set()
    source = meta.get("source_agent_id")
    if source:
        names |= _agent_aliases(str(source))
    raw = meta.get("accessible_by")
    if raw in (None, "all"):
        return names
    if isinstance(raw, str):
        names |= _agent_aliases(raw)
        return names
    if isinstance(raw, list):
        for item in raw:
            names |= _agent_aliases(str(item))
    return names
