"""Seed sample data to verify the memory architecture."""

import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from config.agents import AGENT_CONFIGS
from memory.semantic_memory import SemanticMemory
from memory.procedural_memory import ProceduralMemory
from memory.knowledge_base import KnowledgeBase

console = Console()


# scripts/setup_sample_data.py

def seed():
    console.print("[bold yellow]Initializing sample data...[/bold yellow]")
    
    kb = KnowledgeBase()
    pm = ProceduralMemory()
    sm = SemanticMemory()
    
    for agent_key, cfg in AGENT_CONFIGS.items():
        console.print(f"\n[cyan]Seeding data for agent: {cfg['name']}[/cyan]")
        
        # Knowledge base (unchanged)
        for doc in cfg.get("knowledge_base", []):
            kb.add_document(
                title=doc["title"],
                content=doc["content"],
                agent_id=cfg["agent_id"],
                category=doc.get("category", "general")
            )
            console.print(f"  [green]✓[/green] Added KB doc: {doc['title']}")
        
        # Procedural skills (unchanged)
        for skill in cfg.get("procedural_skills", []):
            pm.add_skill(
                title=skill["title"],
                content=skill["content"],
                description=skill["description"],
                scope=skill["scope"],
                status=skill["status"],
                source_episode_ids=[],
                confidence=1.0,
                trigger="bootstrap_seed",
                approved_by="system_admin"
            )
            console.print(f"  [green]✓[/green] Added Procedural Skill: {skill['title']}")
        
        # Semantic facts - NOW WITH SCOPES
        for fact in cfg.get("semantic_facts", []):
            # Determine scope from old format
            old_scope = fact["scope"]
            
            if old_scope.startswith("user:"):
                scope_type = "user"
                scope_id = old_scope.split(":")[1]
            elif old_scope.startswith("agent:"):
                scope_type = "agent"
                scope_id = old_scope.split(":")[1]
            else:  # shared:organization
                scope_type = "organization"
                scope_id = "organization"
            
            sm.add_fact(
                content=fact["content"],
                scope_type=scope_type,        # ← NEW
                scope_id=scope_id,            # ← NEW
                category="policy",            # ← NEW
                confidence=fact["confidence"],
                source_episode_id="seed_initial",
                trigger="bootstrap_seed",
                status="current"
            )
            console.print(f"  [green]✓[/green] Added Semantic Fact: {fact['content'][:50]}...")
    
    console.print("\n[bold green]✓ Memory seed completed successfully.[/bold green]\n")


if __name__ == "__main__":
    seed()
