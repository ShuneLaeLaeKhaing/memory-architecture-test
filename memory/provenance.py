"""
Provenance tracking for memory lineage.
Allows tracing any memory back to original conversations.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class MemoryStatus(Enum):
    """Lifecycle status of a memory"""
    ACTIVE = "active"  # Currently valid and retrievable
    DEPRECATED = "deprecated"  # Replaced by newer memory
    CONFLICTED = "conflicted"  # Contradicts another memory
    MERGED = "merged"  # Combined with other memories
    UNCERTAIN = "uncertain"  # Low confidence, needs verification


class MemoryScope(Enum):
    """Scope of memory applicability"""
    USER = "user"  # User-specific fact
    AGENT = "agent"  # Agent-level knowledge (shared across users)
    THREAD = "thread"  # Thread-specific context
    GLOBAL = "global"  # Universal knowledge


class MemoryType(Enum):
    """Type of memory content"""
    SEMANTIC = "semantic"  # Facts: "User prefers X"
    PROCEDURAL = "procedural"  # Procedures: "When X, do Y"
    EPISODIC = "episodic"  # Raw conversation episodes


@dataclass
class Provenance:
    """
    Tracks the lineage of a memory - where it came from and why it exists.
    Essential for debugging bad memories and building trust.
    """
    # Source tracking
    source_episode_ids: List[str] = field(default_factory=list)
    source_event_ids: List[str] = field(default_factory=list)
    source_thread_ids: List[str] = field(default_factory=list)
    source_session_ids: List[str] = field(default_factory=list)
    
    # Creation context
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"  # or user_id if explicit user input
    promotion_reason: str = ""  # Why this was promoted to semantic
    
    # Evidence & confidence
    supporting_evidence_count: int = 0  # How many episodes support this
    contradicting_evidence_count: int = 0
    confidence_score: float = 1.0  # 0-1, decays over time or with contradictions
    
    # Temporal tracking
    last_reinforced_at: Optional[datetime] = None  # Last time an episode supported this
    last_accessed_at: Optional[datetime] = None  # Last time this was retrieved
    last_updated_at: Optional[datetime] = None
    
    # Relationships
    supersedes: List[str] = field(default_factory=list)  # Memories this replaced
    superseded_by: Optional[str] = None  # Memory that replaced this
    related_memories: List[str] = field(default_factory=list)  # Similar/connected memories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'source_episode_ids': self.source_episode_ids,
            'source_event_ids': self.source_event_ids,
            'source_thread_ids': self.source_thread_ids,
            'source_session_ids': self.source_session_ids,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'promotion_reason': self.promotion_reason,
            'supporting_evidence_count': self.supporting_evidence_count,
            'contradicting_evidence_count': self.contradicting_evidence_count,
            'confidence_score': self.confidence_score,
            'last_reinforced_at': self.last_reinforced_at.isoformat() if self.last_reinforced_at else None,
            'last_accessed_at': self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            'last_updated_at': self.last_updated_at.isoformat() if self.last_updated_at else None,
            'supersedes': self.supersedes,
            'superseded_by': self.superseded_by,
            'related_memories': self.related_memories,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Provenance':
        """Create from stored dictionary"""
        # Convert ISO strings back to datetime
        created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now()
        last_reinforced = datetime.fromisoformat(data['last_reinforced_at']) if data.get('last_reinforced_at') else None
        last_accessed = datetime.fromisoformat(data['last_accessed_at']) if data.get('last_accessed_at') else None
        last_updated = datetime.fromisoformat(data['last_updated_at']) if data.get('last_updated_at') else None
        
        return cls(
            source_episode_ids=data.get('source_episode_ids', []),
            source_event_ids=data.get('source_event_ids', []),
            source_thread_ids=data.get('source_thread_ids', []),
            source_session_ids=data.get('source_session_ids', []),
            created_at=created_at,
            created_by=data.get('created_by', 'system'),
            promotion_reason=data.get('promotion_reason', ''),
            supporting_evidence_count=data.get('supporting_evidence_count', 0),
            contradicting_evidence_count=data.get('contradicting_evidence_count', 0),
            confidence_score=data.get('confidence_score', 1.0),
            last_reinforced_at=last_reinforced,
            last_accessed_at=last_accessed,
            last_updated_at=last_updated,
            supersedes=data.get('supersedes', []),
            superseded_by=data.get('superseded_by'),
            related_memories=data.get('related_memories', []),
        )


@dataclass
class MemoryMetadata:
    """
    Complete metadata for a memory, including provenance and lifecycle.
    """
    memory_id: str
    memory_type: MemoryType
    memory_scope: MemoryScope
    status: MemoryStatus
    
    # User/Agent context
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    # Provenance
    provenance: Provenance = field(default_factory=Provenance)
    
    # Content metadata
    language: str = "en"  # Support for Burmese
    tags: List[str] = field(default_factory=list)
    
    # Quality metrics
    retrieval_count: int = 0  # How often this is retrieved
    success_rate: Optional[float] = None  # For procedural memories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Mem0 storage"""
        return {
            'memory_id': self.memory_id,
            'memory_type': self.memory_type.value,
            'memory_scope': self.memory_scope.value,
            'status': self.status.value,
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'provenance': self.provenance.to_dict(),
            'language': self.language,
            'tags': self.tags,
            'retrieval_count': self.retrieval_count,
            'success_rate': self.success_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryMetadata':
        """Create from stored dictionary"""
        return cls(
            memory_id=data['memory_id'],
            memory_type=MemoryType(data['memory_type']),
            memory_scope=MemoryScope(data['memory_scope']),
            status=MemoryStatus(data['status']),
            user_id=data.get('user_id'),
            agent_id=data.get('agent_id'),
            provenance=Provenance.from_dict(data.get('provenance', {})),
            language=data.get('language', 'en'),
            tags=data.get('tags', []),
            retrieval_count=data.get('retrieval_count', 0),
            success_rate=data.get('success_rate'),
        )
    
    def update_on_retrieval(self):
        """Update metadata when memory is retrieved"""
        self.retrieval_count += 1
        self.provenance.last_accessed_at = datetime.now()
    
    def update_on_reinforcement(self, episode_id: str):
        """Update when new evidence supports this memory"""
        self.provenance.supporting_evidence_count += 1
        self.provenance.last_reinforced_at = datetime.now()
        if episode_id not in self.provenance.source_episode_ids:
            self.provenance.source_episode_ids.append(episode_id)
    
    def update_on_contradiction(self):
        """Update when contradicting evidence found"""
        self.provenance.contradicting_evidence_count += 1
        self.provenance.confidence_score *= 0.8  # Decay confidence
        
        # If too many contradictions, mark as conflicted
        if self.provenance.contradicting_evidence_count > self.provenance.supporting_evidence_count:
            self.status = MemoryStatus.CONFLICTED
    
    def deprecate(self, superseded_by_id: str, reason: str):
        """Mark this memory as deprecated"""
        self.status = MemoryStatus.DEPRECATED
        self.provenance.superseded_by = superseded_by_id
        self.provenance.last_updated_at = datetime.now()