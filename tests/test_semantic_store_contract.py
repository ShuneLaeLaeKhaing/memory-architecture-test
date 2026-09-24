"""SemanticMemory must keep the SemanticStore methods. No Mem0 connection."""

import inspect

from memory.semantic_memory import SemanticMemory
from memory.semantic_store import SemanticStore


def test_mem0_memory_implements_semantic_store():
    required = {
        name
        for name, value in inspect.getmembers(SemanticStore)
        if inspect.isfunction(value) and not name.startswith("_")
    }
    missing = sorted(name for name in required if not hasattr(SemanticMemory, name))
    assert missing == []
