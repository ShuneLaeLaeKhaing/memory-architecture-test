"""Terminal Interactive Chat: Test harness for the unified memory architecture."""

import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent / ".env")

from config.agents import AGENT_CONFIGS
from memory.event_capture import EventLogger
from memory.consolidation import ConsolidationEngine
from memory.semantic_memory import SemanticMemory
from memory.procedural_memory import ProceduralMemory
from memory.episodic_memory import EpisodicMemory
from memory.knowledge_base import KnowledgeBase
from memory.promotion import PromotionEngine
from memory.retrieval import MemoryRetriever
from agent.simple_agent import SimpleAgent

# Mute noisy internal loggers on stdout
logger.remove()
logger.add("logs/system.log", rotation="5 MB", level="DEBUG")

console = Console()


class TerminalApp:
    def __init__(self):
        # Create data directories
        for p in ["data/events", "data/episodes", "data/procedural_skills", "data/knowledge_base", "logs", "vector_store"]:
            Path(p).mkdir(parents=True, exist_ok=True)

        self.event_logger = EventLogger()
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()
        self.episodic_memory = EpisodicMemory()
        self.knowledge_base = KnowledgeBase()

        self.promotion_engine = PromotionEngine(
            semantic_memory=self.semantic_memory,
            procedural_memory=self.procedural_memory,
            episodic_memory=self.episodic_memory
        )

        self.consolidation_engine = ConsolidationEngine(
            event_logger=self.event_logger,
            promotion_engine=self.promotion_engine
        )

        self.retriever = MemoryRetriever(
            semantic=self.semantic_memory,
            procedural=self.procedural_memory,
            episodic=self.episodic_memory,
            knowledge_base=self.knowledge_base
        )

        self.current_user_id = "alice"
        self.current_agent_id = "hr_assistant"
        self.session_id = ""
        self.agent: SimpleAgent = None

    def start(self):
        console.clear()
        console.print(Panel.fit(
            "[bold green]Enterprise Agent Memory Terminal[/bold green]\n"
            "[white]Complete Architecture: Events -> Episodes -> Mem0 Semantic & Markdown Procedural[/white]",
            border_style="green"
        ))

        self._configure_identity()
        self._new_session()
        self._loop()

    def _configure_identity(self):
        console.print("\n[bold cyan]Select User Identity:[/bold cyan]")
        console.print("1. [bold]alice[/bold] (Engineering Lead)")
        console.print("2. [bold]bob[/bold] (Senior Operations Specialist)")
        console.print("3. [bold]carol[/bold] (New Hire Employee)")

        choice = Prompt.ask("Choose user", choices=["1", "2", "3"], default="1")
        user_map = {"1": "alice", "2": "bob", "3": "carol"}
        self.current_user_id = user_map[choice]

        console.print("\n[bold cyan]Select Agent:[/bold cyan]")
        console.print("1. [bold]HR Assistant[/bold]")
        console.print("2. [bold]Onboarding Assistant[/bold]")

        a_choice = Prompt.ask("Choose agent", choices=["1", "2"], default="1")
        agent_map = {"1": "hr_assistant", "2": "onboarding_assistant"}
        self.current_agent_id = agent_map[a_choice]

        cfg = AGENT_CONFIGS[self.current_agent_id]
        self.agent = SimpleAgent(agent_config=cfg, retriever=self.retriever)
        console.print(f"[green]Authenticated as {self.current_user_id} -> Talking to {cfg['name']}[/green]\n")

    def _new_session(self):
        from datetime import datetime
        self.session_id = f"sess_{self.current_user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.event_logger.log_event({
            "event_type": "session_start",
            "session_id": self.session_id,
            "user_id": self.current_user_id,
            "agent_id": self.current_agent_id,
            "payload": {"status": "started"}
        })
        console.print(f"[dim]Session created: {self.session_id}[/dim]")

    def _loop(self):
        console.print("[dim]Type your message or command (/memory, /consolidate, /user, /new, /exit)[/dim]\n")
        while True:
            try:
                user_msg = Prompt.ask(f"[bold blue]{self.current_user_id}[/bold blue]")
                if not user_msg.strip():
                    continue

                if user_msg.startswith("/"):
                    cmd = user_msg.strip().lower()
                    if cmd == "/exit":
                        self._trigger_consolidation()
                        console.print("[yellow]Exiting chat session. Goodbye![/yellow]")
                        break
                    elif cmd == "/memory":
                        self._render_memory_inspection()
                        continue
                    elif cmd == "/consolidate":
                        self._trigger_consolidation()
                        continue
                    elif cmd == "/new":
                        self._trigger_consolidation()
                        self._new_session()
                        continue
                    elif cmd == "/user":
                        self._trigger_consolidation()
                        self._configure_identity()
                        self._new_session()
                        continue
                    else:
                        console.print(f"[red]Unknown command: {cmd}[/red]")
                        continue

                # 1. Log incoming user event
                self.event_logger.log_event({
                    "event_type": "user_message",
                    "session_id": self.session_id,
                    "user_id": self.current_user_id,
                    "agent_id": self.current_agent_id,
                    "payload": {"content": user_msg}
                })

                # 2. Generate agent response
                with console.status("[dim cyan]Retrieving context & querying model...[/dim cyan]"):
                    reply = self.agent.respond(
                        user_message=user_msg,
                        user_id=self.current_user_id,
                        session_id=self.session_id
                    )

                # 3. Log agent response event
                self.event_logger.log_event({
                    "event_type": "agent_response",
                    "session_id": self.session_id,
                    "user_id": self.current_user_id,
                    "agent_id": self.current_agent_id,
                    "payload": {"content": reply}
                })

                cfg = AGENT_CONFIGS[self.current_agent_id]
                console.print(f"\n[bold green]{cfg['name']}:[/bold green]")
                console.print(Markdown(reply))
                console.print("")

            except (KeyboardInterrupt, EOFError):
                self._trigger_consolidation()
                break

    def _trigger_consolidation(self):
        with console.status("[dim yellow]Running consolidation & promotion...[/dim yellow]"):
            episodes = self.consolidation_engine.consolidate_session(
                session_id=self.session_id,
                user_id=self.current_user_id
            )
        if episodes:
            console.print(f"[green]✓ Consolidated {len(episodes)} new episode(s)[/green]")
            for ep in episodes:
                if ep.promoted_to:
                    console.print(f"  [cyan]-> Promoted items:[/cyan] {ep.promoted_to}")
        else:
            console.print(
                "[yellow]No episodes kept.[/yellow] "
                "Ordinary Q&A is discarded. Keep needs remember/save/note/policy, "
                "a failure, or user feedback (retention ≥ 0.5)."
            )

    def _render_memory_inspection(self):
        console.print("\n")
        console.rule("[bold magenta]AUTHORITATIVE MEMORY STATE[/bold magenta]")
        
        # Same scopes as chat recall: this user + this agent + shared
        recall_facts = self.retriever.get_semantic_facts(
            user_id=self.current_user_id,
            agent_id=self.current_agent_id,
        )

        t_facts = Table(
            title="Semantic Facts (eligible for recall: user + this agent + shared)",
            header_style="bold cyan",
        )
        t_facts.add_column("Fact ID", style="dim", width=18)
        t_facts.add_column("Content")
        t_facts.add_column("Bucket", width=8)
        t_facts.add_column("Scope", width=22)
        t_facts.add_column("Category", width=12)
        t_facts.add_column("Ver", width=4)

        for f in recall_facts:
            if f.scope.startswith("agent:"):
                bucket = "agent"
            elif f.scope.startswith("user:"):
                bucket = "user"
            else:
                bucket = "shared"
            t_facts.add_row(
                f.fact_id[:18],
                f.content,
                bucket,
                f.scope,
                str(f.metadata.get("category") or f.created_by),
                str(f.version),
            )
        
        console.print(t_facts)

        cfg = AGENT_CONFIGS[self.current_agent_id]
        scopes = [
            f"user:{self.current_user_id}",
            f"agent:{cfg['agent_id']}",
            "shared:organization",
        ]

        skills = self.retriever.get_procedural_skills(scopes=scopes)
        t_skills = Table(title="Procedural Skills (.md On-Disk)", header_style="bold green")
        t_skills.add_column("Skill ID", style="dim", width=18)
        t_skills.add_column("Title")
        t_skills.add_column("Status", width=10)
        t_skills.add_column("Scope", width=22)
        t_skills.add_column("Ver", width=4)
        for skill in skills:
            color = "green" if skill.status == "approved" else "yellow"
            t_skills.add_row(
                skill.skill_id,
                skill.title,
                f"[{color}]{skill.status}[/{color}]",
                skill.scope,
                str(skill.version),
            )
        console.print(t_skills)

        episodes = self.retriever.get_episodes(
            user_id=self.current_user_id,
            agent_id=self.current_agent_id,
        )
        t_eps = Table(title="Recent Episodes", header_style="bold blue")
        t_eps.add_column("Episode ID", style="dim", width=18)
        t_eps.add_column("Goal")
        t_eps.add_column("Outcome", width=10)
        t_eps.add_column("Scope", width=18)
        for episode in episodes[:6]:
            t_eps.add_row(episode.episode_id, episode.goal, episode.outcome, episode.scope)
        console.print(t_eps)

        console.rule()
        console.print("")


if __name__ == "__main__":
    app = TerminalApp()
    app.start()
