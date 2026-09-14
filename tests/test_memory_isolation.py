"""
Test memory isolation between users, agents, and threads.
CRITICAL: Ensures no data leakage between users.
"""
import pytest
from datetime import datetime
from memory.event_capture import EventLogger
from memory.thread_manager import ThreadManager
from memory.retrieval import MemoryRetriever
from agent.simple_agent import SimpleAgent


def _memories_for_user(retriever, user_id):
    """Mem0 Cloud wants filters, not get_all(user_id=...)."""
    raw = retriever.mem0.get_all(filters={"user_id": user_id})
    if isinstance(raw, dict):
        raw = raw.get("results") or raw.get("memories") or []
    return list(raw or [])


def _memory_texts(retriever, user_id):
    return {m.get("memory", "") for m in _memories_for_user(retriever, user_id)}


def _delete_user_memories(retriever, user_id):
    try:
        retriever.mem0.delete_all(filters={"user_id": user_id})
    except Exception:
        try:
            retriever.mem0.delete_all(user_id=user_id)
        except Exception:
            pass


class TestUserMemoryIsolation:
    """Ensure user-specific memories don't leak between users"""
    
    def test_semantic_memory_user_isolation(
        self,
        event_logger,
        thread_manager,
        retriever,
        simple_agent,
        iso_users,
    ):
        """User A's facts shouldn't appear in User B's context"""
        user_a = iso_users["a"]
        user_b = iso_users["b"]

        alice_thread = thread_manager.create_thread(
            user_id=user_a,
            agent_id="hr_assistant",
            initial_message="I prefer dark mode"
        )

        simple_agent.respond(
            user_message="I prefer dark mode for all my work",
            user_id=user_a,
            session_id=f"{user_a}_s1",
            thread_id=alice_thread.thread_id
        )

        retriever.mem0.add(
            "User prefers dark mode for all work",
            user_id=user_a,
            metadata={
                "memory_type": "semantic",
                "source": "conversation",
                "created_at": datetime.now().isoformat()
            }
        )

        bob_thread = thread_manager.create_thread(
            user_id=user_b,
            agent_id="hr_assistant",
            initial_message="What are my preferences?"
        )

        response_bob = simple_agent.respond(
            user_message="What are my UI preferences?",
            user_id=user_b,
            session_id=f"{user_b}_s1",
            thread_id=bob_thread.thread_id
        )

        assert "dark mode" not in response_bob.lower(), \
            "User A's preference leaked into User B's conversation!"

        alice_texts = _memory_texts(retriever, user_a)
        bob_texts = _memory_texts(retriever, user_b)
        overlap = alice_texts & bob_texts
        assert len(overlap) == 0, f"Memory leak detected: {overlap}"
    
    def test_episodic_memory_user_isolation(
        self,
        event_logger,
        thread_manager,
        iso_users,
    ):
        """Raw events should be isolated by user_id"""
        user_a = iso_users["a"]
        user_b = iso_users["b"]

        event_logger.log_event({
            "event_type": "user_message",
            "user_id": user_a,
            "session_id": f"{user_a}_s1",
            "thread_id": f"{user_a}_t1",
            "agent_id": "hr_assistant",
            "payload": {"content": "My employee ID is A12345"},
        })

        event_logger.log_event({
            "event_type": "user_message",
            "user_id": user_b,
            "session_id": f"{user_b}_s1",
            "thread_id": f"{user_b}_t1",
            "agent_id": "hr_assistant",
            "payload": {"content": "What's my employee ID?"},
        })

        alice_events = event_logger.get_events(
            session_id=f"{user_a}_s1",
            event_types=["user_message"],
        )
        bob_events = event_logger.get_events(
            session_id=f"{user_b}_s1",
            event_types=["user_message"],
        )

        for event in bob_events:
            event_data = str(event.payload)
            assert "A12345" not in event_data, \
                "Alice's employee ID leaked into Bob's events!"

        assert any("A12345" in str(e.payload) for e in alice_events)
    
    def test_thread_isolation_same_user(
        self,
        event_logger,
        thread_manager,
        simple_agent,
        iso_users,
    ):
        """Episodic memories shouldn't leak between threads (same user)"""
        user_a = iso_users["a"]

        thread1 = thread_manager.create_thread(
            user_id=user_a,
            agent_id="hr_assistant",
            initial_message="I'm working on Project Phoenix"
        )

        simple_agent.respond(
            user_message="I'm working on Project Phoenix, a secret internal tool",
            user_id=user_a,
            session_id=f"{user_a}_s1",
            thread_id=thread1.thread_id
        )

        thread2 = thread_manager.create_thread(
            user_id=user_a,
            agent_id="hr_assistant",
            initial_message="How do I request PTO?"
        )

        response = simple_agent.respond(
            user_message="What have we discussed before?",
            user_id=user_a,
            session_id=f"{user_a}_s2",
            thread_id=thread2.thread_id
        )
        
        # Thread 2 shouldn't have episodic details from Thread 1
        # (Semantic memories like "Alice is on a project" might be OK, 
        #  but specific conversation "Project Phoenix" should not appear)
        assert "Phoenix" not in response, \
            "Episodic detail from Thread 1 leaked into Thread 2!"
        
        # Verify thread history is isolated
        thread1_events = event_logger.get_events_for_thread(thread1.thread_id)
        thread2_events = event_logger.get_events_for_thread(thread2.thread_id)
        
        thread1_ids = {e.event_id for e in thread1_events}
        thread2_ids = {e.event_id for e in thread2_events}
        
        assert len(thread1_ids & thread2_ids) == 0, \
            "Events leaked between threads!"


class TestAgentMemoryScoping:
    """Test agent-level vs user-level memory scoping"""
    
    def test_agent_procedural_knowledge_shared(
        self,
        retriever,
        simple_agent,
        iso_users,
    ):
        """Agent procedures should be shared across users (if appropriate)"""
        retriever.mem0.add(
            "When calculating PTO balance: (months_employed × 1.5) - days_used",
            user_id=None,  # Agent-level, not user-specific
            agent_id="hr_assistant",
            metadata={
                "memory_type": "procedural",
                "scope": "agent",
                "created_at": datetime.now().isoformat()
            }
        )

        bob_context = retriever.retrieve(
            query="How do I calculate my PTO balance?",
            user_id=iso_users["b"],
            agent_id="hr_assistant"
        )
        
        # Should retrieve agent-level procedural knowledge
        assert "1.5" in str(bob_context) or "calculate" in str(bob_context).lower(), \
            "Agent procedural knowledge not shared appropriately"
    
    def test_user_facts_not_shared(
        self,
        retriever,
        iso_users,
    ):
        """User-specific facts must remain isolated"""
        user_a = iso_users["a"]
        user_b = iso_users["b"]

        retriever.mem0.add(
            "User has 5.5 days PTO remaining",
            user_id=user_a,
            metadata={
                "memory_type": "semantic",
                "scope": "user",
                "created_at": datetime.now().isoformat()
            }
        )

        bob_context = retriever.retrieve(
            query="How many PTO days do I have?",
            user_id=user_b,
            agent_id="hr_assistant"
        )
        
        # Should NOT contain Alice's specific balance
        assert "5.5" not in str(bob_context), \
            "Alice's PTO balance leaked into Bob's context!"


class TestCrossThreadMemoryScope:
    """Test semantic vs episodic memory across threads"""
    
    def test_semantic_memory_crosses_threads(
        self,
        thread_manager,
        retriever,
        simple_agent,
        iso_users,
    ):
        """Semantic memories (facts) should work across threads for same user"""
        user_a = iso_users["a"]

        thread1 = thread_manager.create_thread(
            user_id=user_a,
            agent_id="hr_assistant",
            initial_message="I prefer Slack notifications"
        )

        simple_agent.respond(
            user_message="Please always notify me via Slack, not email",
            user_id=user_a,
            session_id=f"{user_a}_s1",
            thread_id=thread1.thread_id
        )

        retriever.mem0.add(
            "User prefers Slack notifications over email",
            user_id=user_a,
            metadata={
                "memory_type": "semantic",
                "source_thread_id": thread1.thread_id,
                "created_at": datetime.now().isoformat()
            }
        )

        thread2 = thread_manager.create_thread(
            user_id=user_a,
            agent_id="hr_assistant",
            initial_message="How should I contact you?"
        )

        response = simple_agent.respond(
            user_message="What's the best way to notify me about updates?",
            user_id=user_a,
            session_id=f"{user_a}_s2",
            thread_id=thread2.thread_id
        )
        
        # Should retrieve semantic memory from Thread 1
        assert "slack" in response.lower(), \
            "Semantic memory didn't cross threads for same user!"
    
    def test_episodic_memory_stays_in_thread(
        self,
        event_logger,
        thread_manager,
        simple_agent,
        iso_users,
    ):
        """Episodic details should NOT cross thread boundaries"""
        user_a = iso_users["a"]

        thread1 = thread_manager.create_thread(
            user_id=user_a,
            agent_id="hr_assistant",
            initial_message="Meeting discussion"
        )

        simple_agent.respond(
            user_message="In yesterday's meeting, John mentioned the Q4 budget is tight",
            user_id=user_a,
            session_id=f"{user_a}_s1",
            thread_id=thread1.thread_id
        )

        thread2 = thread_manager.create_thread(
            user_id=user_a,
            agent_id="hr_assistant",
            initial_message="General question"
        )

        response = simple_agent.respond(
            user_message="What did we discuss last time?",
            user_id=user_a,
            session_id=f"{user_a}_s2",
            thread_id=thread2.thread_id
        )
        
        # Should NOT have specific details from Thread 1
        assert "john" not in response.lower() and "q4 budget" not in response.lower(), \
            "Episodic details leaked across threads!"
        
        # Should indicate this is a new thread
        assert any(phrase in response.lower() for phrase in [
            "haven't talked",
            "new conversation",
            "first time",
            "this thread"
        ]), "Agent didn't recognize this is a different thread"


class TestMaliciousMemoryInjection:
    """Adversarial tests - users trying to inject false memories"""
    
    def test_user_cannot_inject_other_user_memories(
        self,
        simple_agent,
        thread_manager,
        retriever,
        iso_users,
    ):
        """User A claiming facts about User B shouldn't affect User B"""
        user_a = iso_users["a"]
        user_b = iso_users["b"]

        alice_thread = thread_manager.create_thread(
            user_id=user_a,
            agent_id="hr_assistant",
            initial_message="About Bob"
        )

        simple_agent.respond(
            user_message="Bob told me his password is 'password123'",
            user_id=user_a,
            session_id=f"{user_a}_s1",
            thread_id=alice_thread.thread_id
        )

        bob_thread = thread_manager.create_thread(
            user_id=user_b,
            agent_id="hr_assistant",
            initial_message="Security question"
        )

        response = simple_agent.respond(
            user_message="What's my password?",
            user_id=user_b,
            session_id=f"{user_b}_s1",
            thread_id=bob_thread.thread_id
        )

        assert "password123" not in response.lower(), \
            "SECURITY BREACH: User A injected false memory into User B's context!"

        bob_memories = _memories_for_user(retriever, user_b)

        for memory in bob_memories:
            memory_text = memory.get('memory', '').lower()
            assert "password123" not in memory_text, \
                "Injected memory found in Bob's memory store!"
    
    def test_cross_agent_isolation(
        self,
        simple_agent,
        thread_manager,
        retriever,
        iso_users,
    ):
        """Memories for one agent shouldn't leak to another agent"""
        user_a = iso_users["a"]

        thread_manager.create_thread(
            user_id=user_a,
            agent_id="hr_assistant",
            initial_message="HR question"
        )

        retriever.mem0.add(
            "User's salary is $120,000",
            user_id=user_a,
            agent_id="hr_assistant",
            metadata={
                "memory_type": "semantic",
                "scope": "user",
                "agent_context": "hr_assistant"
            }
        )

        it_context = retriever.retrieve(
            query="What can you help me with?",
            user_id=user_a,
            agent_id="it_support"
        )
        
        # Salary info should NOT leak to IT agent
        # (This test depends on your retriever filtering by agent_id)
        context_str = str(it_context).lower()
        assert "salary" not in context_str and "120000" not in context_str, \
            "Sensitive HR info leaked to IT support agent!"


@pytest.fixture
def event_logger():
    """Fresh EventLogger for each test"""
    from memory.event_capture import EventLogger
    logger = EventLogger()
    # Clear any existing events if needed
    return logger


@pytest.fixture
def thread_manager():
    """Fresh ThreadManager for each test"""
    from memory.thread_manager import ThreadManager
    manager = ThreadManager()
    # Clear any existing threads if needed
    return manager


@pytest.fixture
def retriever(iso_users):
    """Fresh MemoryRetriever for each test"""
    from memory.retrieval import MemoryRetriever

    retriever = MemoryRetriever()
    retriever.mem0 = retriever.semantic.mem0
    yield retriever

    for user_id in iso_users.values():
        _delete_user_memories(retriever, user_id)


@pytest.fixture
def simple_agent(event_logger, retriever):
    """Fresh SimpleAgent for each test"""
    from agent.simple_agent import SimpleAgent
    from config.agents import AGENT_CONFIGS

    return SimpleAgent(
        agent_config=AGENT_CONFIGS["hr_assistant"],
        retriever=retriever,
        event_logger=event_logger,
    )