# tests/test_thread_scenarios.py

def test_scenario_1_1_multi_turn():
    """Test multi-turn information request."""
    agent = SimpleAgent(...)
    thread_id = "test_thread_123"
    
    # Turn 1
    response1 = agent.respond(
        user_message="What's the company's PTO policy?",
        user_id="alice",
        session_id="sess_1",
        thread_id=thread_id
    )
    assert "15 days" in response1
    
    # Turn 2
    response2 = agent.respond(
        user_message="How do I request it?",
        user_id="alice",
        session_id="sess_1",
        thread_id=thread_id
    )
    assert "PTO" in response2  # Should mention PTO context
    
    # Turn 3
    response3 = agent.respond(
        user_message="Who approves it?",
        user_id="alice",
        session_id="sess_1",
        thread_id=thread_id
    )
    assert "manager" in response3.lower()
    assert "PTO" in response3  # Should still reference PTO