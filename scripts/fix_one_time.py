# scripts/fix_one_time.py
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from memory.semantic_memory import SemanticMemory

sem = SemanticMemory()
all_facts = sem.get_all_facts(user_id="carol", status=None)

current = [f for f in all_facts if f.status == "current"]
superseded = [f for f in all_facts if f.status == "superseded"]

print(f"Carol's semantic facts:")
print(f"  Current: {len(current)}")
print(f"  Superseded: {len(superseded)}")

if len(current) == 1 and len(superseded) == 2:
    print("\n✅ PERFECT! 1 current + 2 superseded = 3 versions total")
else:
    print(f"\n⚠️  Expected 1 current + 2 superseded, got {len(current)} + {len(superseded)}")