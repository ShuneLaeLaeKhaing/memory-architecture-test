"""Pytest fixtures for memory system testing."""

import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from agent.simple_agent import SimpleAgent
from config.agents import AGENT_CONFIGS
from memory.consolidation import ConsolidationEngine
from memory.episodic_memory import EpisodicMemory
from memory.event_capture import EventLogger
from memory.knowledge_base import KnowledgeBase
from memory.procedural_memory import ProceduralMemory
from memory.promotion import PromotionEngine
from memory.retrieval import MemoryRetriever
from memory.semantic_memory import SemanticMemory
from memory.thread_manager import ThreadManager


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def event_logger(temp_data_dir):
    """EventLogger with temp storage."""
    events_path = Path(temp_data_dir) / "events"
    events_path.mkdir(parents=True, exist_ok=True)
    return EventLogger(base_path=str(events_path))


@pytest.fixture
def thread_manager(temp_data_dir):
    """ThreadManager with temp storage."""
    threads_path = Path(temp_data_dir) / "threads"
    threads_path.mkdir(parents=True, exist_ok=True)
    return ThreadManager(threads_path=str(threads_path))


@pytest.fixture
def semantic_memory():
    """SemanticMemory instance (uses Mem0 Cloud)."""
    return SemanticMemory()


@pytest.fixture
def chroma_dir(temp_data_dir):
    path = Path(temp_data_dir) / "vector_store"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@pytest.fixture
def procedural_memory(temp_data_dir, chroma_dir):
    """ProceduralMemory with temp storage."""
    skills_path = Path(temp_data_dir) / "procedural_skills"
    skills_path.mkdir(parents=True, exist_ok=True)
    return ProceduralMemory(base_path=str(skills_path), persist_dir=chroma_dir)


@pytest.fixture
def episodic_memory(temp_data_dir, chroma_dir):
    """EpisodicMemory instance."""
    episodes_path = Path(temp_data_dir) / "episodes_index"
    episodes_path.mkdir(parents=True, exist_ok=True)
    return EpisodicMemory(base_path=str(episodes_path), persist_dir=chroma_dir)


@pytest.fixture
def knowledge_base(temp_data_dir, chroma_dir):
    """KnowledgeBase with temp storage."""
    kb_path = Path(temp_data_dir) / "knowledge_base"
    kb_path.mkdir(parents=True, exist_ok=True)
    return KnowledgeBase(base_path=str(kb_path), persist_dir=chroma_dir)


@pytest.fixture
def promotion_engine(semantic_memory, procedural_memory, episodic_memory):
    """PromotionEngine with all memory stores."""
    return PromotionEngine(
        semantic_memory=semantic_memory,
        procedural_memory=procedural_memory,
        episodic_memory=episodic_memory,
    )


@pytest.fixture
def consolidation_engine(temp_data_dir, event_logger, promotion_engine):
    """ConsolidationEngine with temp storage."""
    episodes_path = Path(temp_data_dir) / "episodes"
    episodes_path.mkdir(parents=True, exist_ok=True)
    return ConsolidationEngine(
        episodes_path=str(episodes_path),
        event_logger=event_logger,
        promotion_engine=promotion_engine,
    )


@pytest.fixture
def retriever(semantic_memory, procedural_memory, episodic_memory, knowledge_base):
    """MemoryRetriever with all stores."""
    return MemoryRetriever(
        semantic=semantic_memory,
        procedural=procedural_memory,
        episodic=episodic_memory,
        knowledge_base=knowledge_base,
    )


@pytest.fixture
def agent(retriever, event_logger):
    """SimpleAgent with HR Assistant config."""
    return SimpleAgent(
        agent_config=AGENT_CONFIGS["hr_assistant"],
        retriever=retriever,
        event_logger=event_logger,
    )


@pytest.fixture
def test_user():
    """Standard test user."""
    return {
        "user_id": "test_alice",
        "agent_id": "hr_assistant",
    }


@pytest.fixture
def test_thread(thread_manager, test_user):
    """Create a test thread."""
    return thread_manager.create_thread(
        user_id=test_user["user_id"],
        agent_id=test_user["agent_id"],
        initial_message="Test conversation",
    )


@pytest.fixture
def test_session():
    """Generate test session ID."""
    return f"sess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
