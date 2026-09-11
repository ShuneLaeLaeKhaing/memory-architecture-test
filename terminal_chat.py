"""Terminal Interactive Chat: Test harness for the unified memory architecture."""

import sys
from pathlib import Path
from typing import Optional

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
from memory.consolidation_scheduler import ConsolidationScheduler
from memory.thread_manager import ThreadManager,Thread
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
        self.thread_manager = ThreadManager()
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
        self.consolidation_scheduler = ConsolidationScheduler(
            inactivity_timeout=300
        )

        self.current_user_id = "alice"
        self.current_agent_id = "hr_assistant"
        self.session_id = ""
        self.agent: SimpleAgent = None
        self.current_thread: Optional[Thread] = None

    def start(self):
        console.clear()
        console.print(Panel.fit(
            "[bold green]Enterprise Agent Memory Terminal[/bold green]\n"
            "[white]Complete Architecture: Events -> Episodes -> Mem0 Semantic & Markdown Procedural[/white]",
            border_style="green"
        ))

        self._configure_identity()
        self._new_session()
        self._show_thread_selection()
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
        self.agent = SimpleAgent(
            agent_config=cfg,
            retriever=self.retriever,
            event_logger=self.event_logger,
        )
        console.print(f"[green]Authenticated as {self.current_user_id} -> Talking to {cfg['name']}[/green]\n")

    def _show_thread_selection(self):
        """Show thread list (like Claude sidebar) or create new."""
        threads = self.thread_manager.list_threads(
            user_id=self.current_user_id,
            agent_id=self.current_agent_id
        )
        
        if not threads:
            console.print("\n[dim]No existing threads. Starting new conversation...[/dim]")
            self._create_new_thread()
            return
        
        console.print("\n[bold cyan]Your Conversations:[/bold cyan]")
        for i, thread in enumerate(threads[:10], 1):  # Show last 10
            age = self._format_age(thread.last_activity)
            console.print(
                f"{i}. [bold]{thread.title}[/bold] "
                f"[dim]({thread.message_count} messages, {age})[/dim]"
            )
        
        console.print(f"{len(threads) + 1}. [green]+ Start New Conversation[/green]")
        
        choice = Prompt.ask(
            "Select conversation",
            choices=[str(i) for i in range(1, len(threads) + 2)],
            default="1"
        )
        
        choice_idx = int(choice) - 1
        
        if choice_idx < len(threads):
            # Resume existing thread
            self.current_thread = threads[choice_idx]
            console.print(f"\n[green]Resuming: {self.current_thread.title}[/green]")
            self._show_thread_history()
            self._new_session()
        else:
            # Create new thread
            self._create_new_thread()
    
    def _create_new_thread(self):
        """Start new thread (will get title from first message)."""
        self.current_thread = None  # Will be created on first message
        self._new_session()

    def _new_session(self):
        """Start new session within current thread."""
        from datetime import datetime
        
        self.session_id = f"sess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        if self.current_thread:
            # Add session to existing thread
            self.thread_manager.add_session_to_thread(
                self.current_thread.thread_id,
                self.session_id
            )
            
            self.event_logger.log_event({
                "event_type": "session_start",
                "session_id": self.session_id,
                "thread_id": self.current_thread.thread_id,  # ✅ NEW
                "user_id": self.current_user_id,
                "agent_id": self.current_agent_id,
                "payload": {"status": "resumed_thread"}
            })
            
            console.print(f"[dim]Session started in thread: {self.current_thread.thread_id}[/dim]")
        else:
            # First session, thread will be created on first message
            console.print(f"[dim]Session created: {self.session_id}[/dim]")
    
    def _show_thread_history(self):
        """Show message history for current thread (like loading a Claude conversation)."""
        if not self.current_thread:
            return
        
        events = self.event_logger.get_events_for_thread(
            thread_id=self.current_thread.thread_id,
            event_types=["user_message", "agent_response"]
        )
        
        if not events:
            return
        
        console.print("\n[dim]─── Previous Messages ───[/dim]")
        
        for event in events[-10:]:  # Show last 10 messages
            if event.event_type == "user_message":
                console.print(f"[bold blue]{self.current_user_id}:[/bold blue] {event.payload['content']}")
            elif event.event_type == "agent_response":
                cfg = AGENT_CONFIGS[self.current_agent_id]
                console.print(f"[bold green]{cfg['name']}:[/bold green] {event.payload['content'][:100]}...")
        
        console.print("[dim]─── End of History ───[/dim]\n")

    def _loop(self):
        console.print("[dim]Commands: /threads (switch), /new (new thread), /memory, /exit[/dim]\n")
        
        while True:
            try:
                user_msg = Prompt.ask(f"[bold blue]{self.current_user_id}[/bold blue]")
                
                if not user_msg.strip():
                    continue
                
                # Handle commands
                if user_msg.startswith("/"):
                    cmd = user_msg.strip().lower()
                    
                    if cmd == "/exit":
                        self._trigger_consolidation()
                        console.print("[yellow]Goodbye![/yellow]")
                        break
                    
                    elif cmd == "/threads":
                        # Switch threads (like clicking sidebar)
                        self._trigger_consolidation()
                        self._show_thread_selection()
                        continue
                    
                    elif cmd == "/new":
                        # Create new thread
                        self._trigger_consolidation()
                        self._create_new_thread()
                        continue
                    
                    elif cmd == "/memory":
                        self._render_memory_inspection()
                        continue
                    
                    elif cmd == "/consolidate":
                        self._trigger_consolidation()
                        continue
                    
                    else:
                        console.print(f"[red]Unknown: {cmd}[/red]")
                        continue
                
                # Create thread on first message if needed
                if not self.current_thread:
                    self.current_thread = self.thread_manager.create_thread(
                        user_id=self.current_user_id,
                        agent_id=self.current_agent_id,
                        initial_message=user_msg
                    )
                    
                    # Log session start now that we have thread_id
                    self.event_logger.log_event({
                        "event_type": "session_start",
                        "session_id": self.session_id,
                        "thread_id": self.current_thread.thread_id,
                        "user_id": self.current_user_id,
                        "agent_id": self.current_agent_id,
                        "payload": {"status": "new_thread"}
                    })
                
                # Log user message
                self.event_logger.log_event({
                    "event_type": "user_message",
                    "session_id": self.session_id,
                    "thread_id": self.current_thread.thread_id,  
                    "user_id": self.current_user_id,
                    "agent_id": self.current_agent_id,
                    "payload": {"content": user_msg}
                })
                
                # Update thread
                self.thread_manager.increment_message_count(self.current_thread.thread_id)
                
                # Get response (with thread context!)
                with console.status("[dim cyan]Thinking...[/dim cyan]"):
                    reply = self.agent.respond(
                        user_message=user_msg,
                        user_id=self.current_user_id,
                        session_id=self.session_id,
                        thread_id=self.current_thread.thread_id if self.current_thread else None  # ✅ NEW
                    )
                
                # Log agent response
                self.event_logger.log_event({
                    "event_type": "agent_response",
                    "session_id": self.session_id,
                    "thread_id": self.current_thread.thread_id, 
                    "user_id": self.current_user_id,
                    "agent_id": self.current_agent_id,
                    "payload": {"content": reply}
                })
                
                # Update thread
                self.thread_manager.increment_message_count(self.current_thread.thread_id)
                
                # Display
                cfg = AGENT_CONFIGS[self.current_agent_id]
                console.print(f"\n[bold green]{cfg['name']}:[/bold green]")
                console.print(Markdown(reply))
                console.print("")
                
            except (KeyboardInterrupt, EOFError):
                self._trigger_consolidation()
                break
    
    def _format_age(self, timestamp_str: str) -> str:
        """Format 'last_activity' as human-readable age."""
        from datetime import datetime
        
        ts = datetime.fromisoformat(timestamp_str)
        delta = datetime.utcnow() - ts
        
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds >= 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds >= 60:
            return f"{delta.seconds // 60}m ago"
        else:
            return "just now"

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
    import signal
    
    app = TerminalApp()
    
    def shutdown_handler(signum, frame):
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        app.consolidation_scheduler.cleanup()
        app._trigger_consolidation()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    app.start()
