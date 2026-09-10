"""Automated pipeline test validating each memory component in sequence."""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from memory.event_capture import EventLogger
from memory.consolidation import ConsolidationEngine
from memory.semantic_memory import SemanticMemory
from memory.procedural_memory import ProceduralMemory
from memory.episodic_memory import EpisodicMemory
from memory.retrieval import MemoryRetriever
from memory.promotion import PromotionEngine


def run_checks():
    print("============================================")
    print("RUNNING MEMORY SUBSYSTEM VERIFICATION CHECKS")
    print("============================================")

    # 1. Event Capture
    el = EventLogger()
    evt = el.log_event({
        "event_type": "user_message",
        "session_id": "test_sess_001",
        "user_id": "unit_test_user",
        "payload": {"content": "Hello! Please remember my favorite IDE is Neovim."}
    })
    assert evt.event_id is not None
    print("[PASS] EventLogger recorded event successfully")

    # 2. Procedural Engine
    pm = ProceduralMemory()
    skill = pm.add_skill(
        title="Deploy Hotfix",
        content="# Deployment SOP\n1. Checkout main branch\n2. Run migrations.",
        description="Emergency hotfix deploy steps",
        scope="user:unit_test_user",
        status="approved",
        source_episode_ids=["ep_test"],
        confidence=1.0,
        trigger="test"
    )
    assert skill.skill_id is not None
    res = pm.search("deploy hotfix", scopes=["user:unit_test_user"], top_k=1)
    assert len(res) > 0
    print("[PASS] ProceduralMemory written and retrieved via vector search")

    # 3. Semantic Engine
    sm = SemanticMemory()
    fact = sm.add_fact(
        content="Unit test user prefers Neovim",
        user_id="unit_test_user",
        scope="user:unit_test_user",
        confidence=1.0,
        source_episode_id="ep_test",
        trigger="test"
    )
    assert fact.fact_id is not None
    facts = sm.search("What is my preferred IDE?", user_id="unit_test_user", scopes=["user:unit_test_user"])
    assert len(facts) > 0
    print("[PASS] SemanticMemory Mem0 (infer=False) validated")

    # 4. Consolidation & Promotion
    ep_mem = EpisodicMemory()
    pe = PromotionEngine(semantic_memory=sm, procedural_memory=pm, episodic_memory=ep_mem)
    ce = ConsolidationEngine(promotion_engine=pe)
    episodes = ce.consolidate_session(session_id="test_sess_001", user_id="unit_test_user")
    print(f"[PASS] Consolidation completed ({len(episodes)} episodes generated)")

    # 5. Retrieval routing check
    retriever = MemoryRetriever(semantic=sm, procedural=pm, episodic=ep_mem)
    ctx = retriever.retrieve(
        query="How do I deploy a hotfix?",
        user_id="unit_test_user",
        scopes=["user:unit_test_user"],
        agent_id="hr_assistant_001"
    )
    assert ctx["routing"]["primary"] == "procedural"
    print(f"[PASS] MemoryRetriever dynamically prioritized '{ctx['routing']['primary']}'")

    print("\nALL ARCHITECTURAL SANITY CHECKS PASSED.")


if __name__ == "__main__":
    run_checks()
