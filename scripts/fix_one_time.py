# scripts/fix_one_time.py
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from memory.semantic_memory import SemanticMemory

# scripts/fix_one_time.py - replace your existing script

# scripts/diagnose_alice.py

from memory.semantic_memory import SemanticMemory
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="DEBUG", format="{time:HH:mm:ss} | {level:<8} | {message}")

semantic = SemanticMemory()

# Step 1: show everything Mem0 has for alice
print("\n=== ALL FACTS FOR ALICE ===")
all_facts = semantic.get_all_facts(
    scopes=["user:alice"],
    status=None,   # get everything including superseded
)

for f in all_facts:
    print(f"""
  fact_id  : {f.fact_id[:8]}
  content  : {f.content}
  status   : {f.status}
  fact_key : {f.metadata.get('fact_key')}
  fact_value: {f.metadata.get('fact_value')}
  version  : {f.version}
    """)

# Step 2: show what find_existing returns for a theme query
print("\n=== find_existing for 'ui_theme' key ===")
from memory.promotion import PromotionEngine, Candidate

engine = PromotionEngine(semantic_memory=semantic)

# Simulate what find_existing would be called with
fake_cand = Candidate(
    content="Alice prefers light mode for all applications",
    fact_type="preference",
    key="ui_theme",          # after normalization
    value="light mode",
    scope_hint="user",
    confidence=0.95,
    evidence="test",
)

existing = engine.find_existing("user", "alice", fake_cand)
print(f"find_existing returned {len(existing)} facts:")
for f in existing:
    print(f"  [{f.fact_id[:8]}] key={f.metadata.get('fact_key')} status={f.status}")

# Step 3: show what keys are actually stored and whether normalizer catches them
print("\n=== KEY NORMALIZATION CHECK ===")
from memory.key_normalizer import normalize_key, keys_are_same_concept

for f in all_facts:
    raw_key = f.metadata.get("fact_key", "MISSING")
    norm = normalize_key(raw_key)
    matches_ui_theme = keys_are_same_concept(raw_key, "ui_theme")
    print(f"  [{f.fact_id[:8]}] raw_key='{raw_key}' → normalized='{norm}' matches_ui_theme={matches_ui_theme}")