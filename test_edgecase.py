"""
test_promotion_edge_cases.py

Tests semantic fact promotion edge cases:
- Duplicate keys with different values
- Same concept expressed differently
- Generic vs specific facts
- Transient vs durable statements
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from memory.consolidation import EpisodeRecord
from memory.semantic_memory import SemanticMemory
from memory.promotion import PromotionEngine


# ============================================================
# TEST SCENARIOS
# ============================================================

class TestScenario:
    """A test case for promotion logic."""
    
    def __init__(
        self,
        name: str,
        episodes: List[dict],
        expected_outcome: str,
        description: str
    ):
        self.name = name
        self.episodes = episodes
        self.expected_outcome = expected_outcome
        self.description = description


# Edge Case 1: Same key, value changes over time
SCENARIO_VALUE_CHANGE = TestScenario(
    name="Value Change - Same Key",
    description="Bob's UI theme preference changes from dark to light",
    expected_outcome="Should supersede old fact, leave only 1 current fact",
    episodes=[
        {
            "episode_id": "ep_001",
            "user_id": "bob",
            "agent_id": "hr_assistant",
            "goal": "Set up my workspace preferences",
            "summary": "Bob configured his UI theme to dark mode",
            "lessons": ["Bob prefers dark mode for the interface"],
            "outcome": "success",
            "trajectory": [
                "User asked about theme settings",
                "Agent explained theme options",
                "User selected dark mode",
                "System applied dark theme"
            ]
        },
        {
            "episode_id": "ep_002",
            "user_id": "bob",
            "agent_id": "hr_assistant",
            "goal": "Change my UI theme",
            "summary": "Bob switched to light mode due to eye strain",
            "lessons": ["Bob now prefers light mode for the interface"],
            "outcome": "success",
            "trajectory": [
                "User reported eye strain with dark mode",
                "Agent suggested light theme",
                "User switched to light mode",
                "System applied light theme"
            ]
        }
    ]
)

# Edge Case 2: Same concept, different wording
SCENARIO_KEY_NORMALIZATION = TestScenario(
    name="Key Normalization - Same Concept",
    description="Laptop preference expressed with different keys",
    expected_outcome="Should recognize as same concept, merge or supersede",
    episodes=[
        {
            "episode_id": "ep_003",
            "user_id": "bob",
            "agent_id": "hr_assistant",
            "goal": "Request equipment",
            "summary": "Bob requested a MacBook for development work",
            "lessons": ["Bob's preferred laptop is MacBook Pro"],
            "outcome": "success",
            "trajectory": [
                "User requested development equipment",
                "User specified MacBook Pro",
                "System recorded laptop preference"
            ]
        },
        {
            "episode_id": "ep_004",
            "user_id": "bob",
            "agent_id": "hr_assistant",
            "goal": "Confirm equipment setup",
            "summary": "Confirmed Bob uses MacBook for coding",
            "lessons": ["Bob's device preference for development is MacBook"],
            "outcome": "success",
            "trajectory": [
                "Agent asked about current setup",
                "User confirmed using MacBook",
                "System updated records"
            ]
        }
    ]
)

# Edge Case 3: Generic vs Specific
SCENARIO_GENERIC_REJECTION = TestScenario(
    name="Generic Statement Rejection",
    description="Generic advice should be rejected, specific facts promoted",
    expected_outcome="Generic rejected, specific promoted",
    episodes=[
        {
            "episode_id": "ep_005",
            "user_id": "bob",
            "agent_id": "hr_assistant",
            "goal": "Learn about company tools",
            "summary": "Discussion about general onboarding best practices",
            "lessons": [
                "It's important for new employees to familiarize themselves with company tools",
                "Bob's assigned team is Platform Engineering"
            ],
            "outcome": "success",
            "trajectory": [
                "User asked about onboarding",
                "Agent explained general process",
                "User was assigned to Platform Engineering team"
            ]
        }
    ]
)

# Edge Case 4: Transient vs Durable
SCENARIO_TRANSIENT_REJECTION = TestScenario(
    name="Transient Statement Rejection",
    description="Temporary states should be rejected",
    expected_outcome="Transient rejected, durable promoted",
    episodes=[
        {
            "episode_id": "ep_006",
            "user_id": "bob",
            "agent_id": "hr_assistant",
            "goal": "Set up communication tools",
            "summary": "Bob is currently setting up Slack",
            "lessons": [
                "Bob is currently configuring Slack notifications",
                "Bob's Slack username is @bob.smith"
            ],
            "outcome": "success",
            "trajectory": [
                "User started Slack setup",
                "User configured notification preferences",
                "User created username @bob.smith"
            ]
        }
    ]
)

# Edge Case 5: Multiple duplicates (zombie facts)
SCENARIO_MULTIPLE_DUPLICATES = TestScenario(
    name="Multiple Duplicates Cleanup",
    description="Superseding should clean up ALL duplicates",
    expected_outcome="All duplicates marked superseded, only 1 current",
    episodes=[
        {
            "episode_id": "ep_007",
            "user_id": "bob",
            "agent_id": "hr_assistant",
            "goal": "Set manager info",
            "summary": "Bob's manager is Alice",
            "lessons": ["Bob reports to Alice Johnson"],
            "outcome": "success",
            "trajectory": ["User confirmed manager is Alice"]
        },
        {
            "episode_id": "ep_008",
            "user_id": "bob",
            "agent_id": "hr_assistant",
            "goal": "Update manager info",
            "summary": "Bob's manager is Alice (confirmation)",
            "lessons": ["Bob's direct manager is Alice Johnson"],
            "outcome": "success",
            "trajectory": ["User reconfirmed manager"]
        },
        {
            "episode_id": "ep_009",
            "user_id": "bob",
            "agent_id": "hr_assistant",
            "goal": "Manager change",
            "summary": "Bob's manager changed to Carol",
            "lessons": ["Bob now reports to Carol Davis"],
            "outcome": "success",
            "trajectory": ["User notified of manager change to Carol"]
        }
    ]
)

# Edge Case 6: Confidence threshold
SCENARIO_LOW_CONFIDENCE = TestScenario(
    name="Low Confidence Rejection",
    description="Facts below confidence threshold rejected",
    expected_outcome="Low confidence fact rejected",
    episodes=[
        {
            "episode_id": "ep_010",
            "user_id": "bob",
            "agent_id": "hr_assistant",
            "goal": "Discuss preferences",
            "summary": "Bob mentioned possibly preferring remote work",
            "lessons": [
                "Bob might prefer remote work",  # Low confidence
                "Bob's office location is Building A"  # High confidence
            ],
            "outcome": "success",
            "trajectory": [
                "User mentioned remote work interest",
                "User confirmed office location"
            ]
        }
    ]
)


# ============================================================
# TEST RUNNER
# ============================================================

class PromotionTester:
    """Runs promotion scenarios and validates outcomes."""
    
    def __init__(self):
        self.semantic = SemanticMemory()
        self.engine = PromotionEngine(semantic_memory=self.semantic)
        self.results = []
    
    def run_scenario(self, scenario: TestScenario) -> dict:
        """Run a single test scenario."""
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST: {scenario.name}")
        logger.info(f"{'='*60}")
        logger.info(f"Description: {scenario.description}")
        logger.info(f"Expected: {scenario.expected_outcome}\n")
        
        # Convert dicts to EpisodeRecord objects
        episodes = [
            EpisodeRecord(
                episode_id=ep["episode_id"],
                session_id=ep.get("session_id", f"sess_{ep['user_id']}_test"),
                user_id=ep["user_id"],
                agent_id=ep["agent_id"],
                timestamp=ep.get("timestamp", datetime.now(timezone.utc).isoformat()),
                goal=ep["goal"],
                summary=ep["summary"],
                lessons=ep["lessons"],
                outcome=ep["outcome"],
                trajectory=ep["trajectory"],
                retention_score=ep.get("retention_score", 0.9),
                scope=f"user:{ep['user_id']}",
                promoted_to=[]
            )
            for ep in scenario.episodes
        ]
        
        # Run promotion
        self.engine.detect_promotions(episodes)
        
        # Analyze results
        user_id = scenario.episodes[0]["user_id"]
        all_facts = self.semantic.get_all_facts(
            user_id=user_id,
            status=None  # Get ALL (current + superseded)
        )
        
        current_facts = [f for f in all_facts if f.status == "current"]
        superseded_facts = [f for f in all_facts if f.status == "superseded"]
        
        result = {
            "scenario": scenario.name,
            "total_episodes": len(episodes),
            "total_facts": len(all_facts),
            "current_facts": len(current_facts),
            "superseded_facts": len(superseded_facts),
            "facts": current_facts,
            "superseded": superseded_facts
        }
        
        # Print results
        logger.info(f"\n--- RESULTS ---")
        logger.info(f"Total facts created: {len(all_facts)}")
        logger.info(f"Current facts: {len(current_facts)}")
        logger.info(f"Superseded facts: {len(superseded_facts)}")
        
        logger.info(f"\n--- CURRENT FACTS ---")
        for fact in current_facts:
            logger.info(
                f"  [{fact.fact_id[:8]}] {fact.content[:60]}... "
                f"(key={fact.metadata.get('fact_key')}, "
                f"value={fact.metadata.get('fact_value')})"
            )
        
        if superseded_facts:
            logger.info(f"\n--- SUPERSEDED FACTS ---")
            for fact in superseded_facts:
                logger.info(
                    f"  [{fact.fact_id[:8]}] {fact.content[:60]}... "
                    f"(superseded_by={fact.superseded_by[:8] if fact.superseded_by else 'N/A'})"
                )
        
        self.results.append(result)
        return result
    
    def run_all_scenarios(self):
        """Run all test scenarios."""
        scenarios = [
            SCENARIO_VALUE_CHANGE,
            SCENARIO_KEY_NORMALIZATION,
            SCENARIO_GENERIC_REJECTION,
            SCENARIO_TRANSIENT_REJECTION,
            SCENARIO_MULTIPLE_DUPLICATES,
            SCENARIO_LOW_CONFIDENCE,
        ]
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"RUNNING {len(scenarios)} PROMOTION EDGE CASE TESTS")
        logger.info(f"{'#'*60}\n")
        
        for scenario in scenarios:
            try:
                self.run_scenario(scenario)
            except Exception as exc:
                logger.error(f"SCENARIO FAILED: {scenario.name}")
                logger.exception(exc)
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        logger.info(f"\n{'#'*60}")
        logger.info(f"TEST SUMMARY")
        logger.info(f"{'#'*60}\n")
        
        for result in self.results:
            logger.info(f"{result['scenario']}:")
            logger.info(f"  Episodes processed: {result['total_episodes']}")
            logger.info(f"  Current facts: {result['current_facts']}")
            logger.info(f"  Superseded facts: {result['superseded_facts']}")
            logger.info("")


# ============================================================
# INTERACTIVE TEST MODE
# ============================================================

def test_custom_sentence():
    """Test a custom sentence for promotion quality."""
    print("\n" + "="*60)
    print("CUSTOM SENTENCE TESTER")
    print("="*60)
    print("\nTest if a sentence would be promoted as a semantic fact.\n")
    
    user_id = input("User ID (default: bob): ").strip() or "bob"
    agent_id = input("Agent ID (default: hr_assistant): ").strip() or "hr_assistant"
    
    print("\nEnter a sentence to test (or 'quit' to exit):")
    print("Examples:")
    print("  - 'Bob prefers dark mode for the interface'")
    print("  - 'It's important for new employees to learn the tools'")
    print("  - 'Bob is currently setting up his laptop'\n")
    
    sentence = input("> ").strip()
    
    if sentence.lower() in ('quit', 'exit', 'q'):
        return
    
    # Create test episode
    episode = EpisodeRecord(
        episode_id=f"test_{hash(sentence) % 10000}",
        session_id=f"sess_{user_id}_test",
        user_id=user_id,
        agent_id=agent_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        goal="Test custom sentence",
        summary=sentence,
        lessons=[sentence],
        outcome="success",
        trajectory=["Test trajectory"],
        retention_score=0.9,
        scope=f"user:{user_id}",
        promoted_to=[]
    )
    
    # Run promotion
    semantic = SemanticMemory()
    engine = PromotionEngine(semantic_memory=semantic)
    
    logger.info(f"\n--- TESTING SENTENCE ---")
    logger.info(f"Input: '{sentence}'")
    logger.info(f"User: {user_id}, Agent: {agent_id}\n")
    
    engine.detect_promotions([episode])
    
    # Check results
    facts = semantic.get_all_facts(user_id=user_id, status="current")
    
    if facts:
        logger.info(f"\n✅ PROMOTED - Created {len(facts)} fact(s):")
        for fact in facts:
            logger.info(f"  Content: {fact.content}")
            logger.info(f"  Type: {fact.metadata.get('category')}")
            logger.info(f"  Key: {fact.metadata.get('fact_key')}")
            logger.info(f"  Value: {fact.metadata.get('fact_value')}")
            logger.info(f"  Confidence: {fact.confidence:.2f}")
    else:
        logger.info(f"\n❌ REJECTED - No facts created")
        logger.info(f"Check logs above for rejection reason")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Run tests based on user choice."""
    print("\n" + "="*60)
    print("SEMANTIC FACT PROMOTION TESTER")
    print("="*60)
    print("\nChoose test mode:")
    print("1. Run all edge case scenarios")
    print("2. Test custom sentence")
    print("3. Run specific scenario")
    
    choice = input("\nChoice [1/2/3] (1): ").strip() or "1"
    
    if choice == "1":
        tester = PromotionTester()
        tester.run_all_scenarios()
    
    elif choice == "2":
        test_custom_sentence()
    
    elif choice == "3":
        scenarios = [
            ("Value Change", SCENARIO_VALUE_CHANGE),
            ("Key Normalization", SCENARIO_KEY_NORMALIZATION),
            ("Generic Rejection", SCENARIO_GENERIC_REJECTION),
            ("Transient Rejection", SCENARIO_TRANSIENT_REJECTION),
            ("Multiple Duplicates", SCENARIO_MULTIPLE_DUPLICATES),
            ("Low Confidence", SCENARIO_LOW_CONFIDENCE),
        ]
        
        print("\nAvailable scenarios:")
        for i, (name, _) in enumerate(scenarios, 1):
            print(f"{i}. {name}")
        
        idx = int(input("\nScenario number: ").strip()) - 1
        
        if 0 <= idx < len(scenarios):
            tester = PromotionTester()
            tester.run_scenario(scenarios[idx][1])
        else:
            print("Invalid scenario number")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()