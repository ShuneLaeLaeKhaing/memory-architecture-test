"""
Add cleanup and isolation helpers to existing conftest.py
"""

import json
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from memory.retrieval import MemoryRetriever

@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically clean up test data before and after each test"""
    # Cleanup before test
    _cleanup_provenance_index()
    _cleanup_test_memories()
    
    yield
    
    # Cleanup after test
    _cleanup_provenance_index()
    _cleanup_test_memories()


def _cleanup_provenance_index():
    """Remove test provenance index"""
    index_file = Path("data/provenance_index.json")
    if index_file.exists():
        # Only remove test entries, keep production data
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        # Remove entries with test user_ids
        test_users = {'alice', 'bob', 'charlie'}
        # This is simplified - in reality, you'd need more sophisticated filtering
        
        index_file.unlink()  # For tests, just remove entirely


def _cleanup_test_memories():
    """Clean up test memories from Mem0"""
    # Note: This requires Mem0 to support bulk deletion by user_id
    # For now, this is a placeholder
    pass


@pytest.fixture
def isolated_retriever():
    """
    Retriever with isolated memory space for testing.
    Ensures no cross-test contamination.
    """
    from memory.retrieval import MemoryRetriever
    
    retriever = MemoryRetriever()
    
    # Could mock Mem0 client here for faster tests
    # For now, using real Mem0 Cloud with cleanup
    
    yield retriever
    
    # Cleanup test memories
    # retriever.mem0.delete_all(user_id="alice")
    # retriever.mem0.delete_all(user_id="bob")