"""Episode → Semantic/Procedural promotion with 6-stage pipeline."""

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger

from memory.llm import llm_client, llm_model
from memory.consolidation import EpisodeRecord
from memory.semantic_memory import SemanticMemory, SemanticFact
from memory.procedural_memory import ProceduralMemory
from memory.episodic_memory import EpisodicMemory
from memory.key_normalizer import normalize_key, keys_are_same_concept


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_TYPES = {"preference", "identity", "constraint", "lesson", "policy"}
MIN_CONFIDENCE = 0.85
MAX_CANDIDATES = 3
DUP_THRESHOLD = 0.88

GENERIC_PATTERNS = [
    "it's important", "it is important", "should familiarize", "may require",
    "can vary", "is helpful", "are useful", "might need", "typically",
    "generally", "usually", "in most cases", "employees should",
    "users should", "new hires should", "best practice", "smooth start",
]

TRANSIENT_PATTERNS = [
    "is currently", "just started", "recently", "was asked",
    "inquired about", "is learning", "is trying", "is setting up",
]


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Candidate:
    """Extracted fact candidate before policy decision."""
    content: str
    fact_type: str
    key: str
    value: str
    scope_hint: str
    confidence: float
    evidence: str


@dataclass
class PolicyDecision:
    action: str
    reason: str
    target: Optional[SemanticFact] = None
    all_targets: Optional[List[SemanticFact]] = None  # add this

    def targets_to_supersede(self) -> List[SemanticFact]:
        """All facts that need to be marked superseded."""
        if self.all_targets:
            return self.all_targets
        if self.target:
            return [self.target]
        return []


# ============================================================
# PROMOTION ENGINE
# ============================================================

class PromotionEngine:
    """Detects durable semantic facts and procedures from episodes."""

    def __init__(
        self,
        semantic_memory: Optional[SemanticMemory] = None,
        procedural_memory: Optional[ProceduralMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None
    ):
        self.semantic = semantic_memory or SemanticMemory()
        self.procedural = procedural_memory or ProceduralMemory()
        self.episodic = episodic_memory or EpisodicMemory()

    def detect_promotions(self, episodes: List[EpisodeRecord]) -> None:
        """Main entry: process episodes through both pipelines."""
        for episode in episodes:
            logger.info(f"[Promotion] Episode: {episode.episode_id}")
            
            # Index episode for historical retrieval
            self.episodic.index_episode(episode)
            
            # Run pipelines
            self._run_semantic_pipeline(episode)
            self._run_procedural_pipeline(episode)

    # ================================================================
    # SEMANTIC PIPELINE (6 stages)
    # ================================================================

    def _run_semantic_pipeline(self, episode: EpisodeRecord) -> None:
        """6-stage semantic fact promotion pipeline."""
        if episode.outcome not in ("success", "correction"):
            logger.info(
                f"[Promotion] Skip semantic (outcome={episode.outcome})"
            )
            return

        # Stage 1+2: Extract & Classify
        candidates = self.extract_candidates(episode)
        if not candidates:
            logger.debug("[Promotion] No candidates extracted")
            return

        for cand in candidates[:MAX_CANDIDATES]:
            # Stage 3: Identity Resolution
            scope_type, scope_id = self.identity_for(cand, episode)

            # Stage 4: Existing Fact Lookup
            existing = self.find_existing(scope_type, scope_id, cand)

            # Stage 5: Policy Decision
            decision = self.apply_policy(cand, existing, episode)

            # Stage 6: Execute
            self._execute_decision(decision, cand, scope_type, scope_id, episode)

    # ---------- Stage 1+2: Extract & Classify ----------

    def extract_candidates(self, ep: EpisodeRecord) -> List[Candidate]:
        """Extract durable facts with key/value structure."""
        
        prompt = f"""Extract durable, specific facts from this episode. Be strict.

        EPISODE:
        Goal: {ep.goal}
        Summary: {ep.summary}
        Lessons: {', '.join(ep.lessons)}
        User: {ep.user_id}
        Agent: {ep.agent_id}

        REQUIREMENTS (ALL must be met):
        1. SPECIFIC - about {ep.user_id}, this agent, or a named company system/team
        2. DURABLE - still true next month (not "is currently setting up")
        3. NON-GENERIC - never general advice

        REJECT (do not extract):
        - "It's important for new employees to..." (generic)
        - "Technical setups may require..." (vague)
        - "New hires should familiarize..." (obvious)

        ACCEPT examples:
        - "{ep.user_id} prefers MacBook for development" → key: preferred_laptop, value: MacBook
        - "{ep.user_id}'s manager is Sarah Chen" → key: manager_name, value: Sarah Chen
        - "IT department uses Microsoft Teams" → key: it_communication_tool, value: Microsoft Teams

        OUTPUT JSON:
        {{
        "candidates": [
            {{
            "content": "Clean standalone sentence, no pronouns",
            "fact_type": "preference|identity|constraint|lesson|policy",
            "key": "snake_case_key",
            "value": "the essential value",
            "scope_hint": "user|agent|organization",
            "confidence": 0.85-1.0,
            "evidence": "quote from episode"
            }}
        ]
        }}

        Return empty array if nothing qualifies."""

        try:
            client = llm_client()
            resp = client.chat.completions.create(
                model=llm_model(),
                messages=[
                    {
                        "role": "system",
                        "content": "Extract durable facts with precision. Reject generic advice."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(resp.choices[0].message.content)
            raw = data.get("candidates", [])
        except Exception as exc:
            logger.error(f"[Promotion] Extraction failed: {exc}")
            return []

        return [
                Candidate(
                    content=str(c.get("content", "")).strip(),
                    fact_type=str(c.get("fact_type", "")).strip().lower(),
                    key=normalize_key(str(c.get("key", "")).strip().lower()),  # ← normalize here
                    value=str(c.get("value", "")).strip(),
                    scope_hint=str(c.get("scope_hint", "user")).strip().lower(),
                    confidence=float(c.get("confidence", 0.0)),
                    evidence=str(c.get("evidence", ""))
                )
                for c in raw
                if c.get("content")
            ]

    # ---------- Stage 3: Identity Resolution ----------

    def identity_for(self, cand: Candidate, ep: EpisodeRecord) -> Tuple[str, str]:
        """Determine who owns this fact (user/agent/organization)."""
        
        if cand.fact_type == "lesson":
            return "agent", ep.agent_id
        
        if cand.fact_type == "policy":
            return "organization", "organization"
        
        return "user", ep.user_id

    # ---------- Stage 4: Existing Fact Lookup ----------

    def find_existing(
    self,
    scope_type: str,
    scope_id: str,
    cand: Candidate,
) -> List[SemanticFact]:

        if scope_type in ("organization", "org", "shared"):
            scope_label = "shared:organization"
        else:
            scope_label = f"{scope_type}:{scope_id}"

        # Pass 0: local cache
        cached = self.semantic.find_in_recent(
            fact_key=cand.key,
            scope=scope_label,
        )
        if cached:
            logger.debug(
                f"[find_existing] Cache hit key='{cand.key}': "
                f"{[f.fact_id[:8] for f in cached]}"
            )
            return cached

        # Pass 1: fetch all in scope, match on NORMALIZED key
        try:
            all_in_scope = self.semantic.get_all_facts(
                scopes=[scope_label],
                status=None,
            )
        except Exception as exc:
            logger.warning(f"[find_existing] get_all_facts failed: {exc}")
            all_in_scope = []

        key_matches = [
            f for f in all_in_scope
            if keys_are_same_concept(
                f.metadata.get("fact_key", ""),
                cand.key
            )
            and f.status != "superseded"
        ]

        if key_matches:
            logger.debug(
                f"[find_existing] Key match (normalized) '{cand.key}': "
                f"{[(f.fact_id[:8], f.metadata.get('fact_key')) for f in key_matches]}"
            )
            return key_matches

        # Pass 2: content similarity fallback
        try:
            similarity_matches = self.semantic.search(
                query=cand.content,
                scope_type=scope_type,
                scope_id=scope_id,
                top_k=3,
                threshold=0.45,
                include_org_facts=False,
            )
            return [f for f in similarity_matches if f.status != "superseded"]
        except Exception as exc:
            logger.warning(f"[find_existing] search fallback failed: {exc}")
            return []

    # ---------- Stage 5: Policy Decision ----------

    def apply_policy(
    self,
    cand: Candidate,
    existing: List[SemanticFact],
    ep: EpisodeRecord,
) -> PolicyDecision:

        # Hard gates
        if cand.confidence < MIN_CONFIDENCE:
            return PolicyDecision(
                "discard",
                f"confidence {cand.confidence:.2f} < {MIN_CONFIDENCE}"
            )
        if cand.fact_type not in ALLOWED_TYPES:
            return PolicyDecision(
                "discard",
                f"unknown fact_type '{cand.fact_type}'"
            )
        if not cand.key or not cand.value:
            return PolicyDecision("discard", "missing key or value")
        if self._matches_any(cand.content, GENERIC_PATTERNS):
            return PolicyDecision("discard", "generic advice pattern detected")
        if self._matches_any(cand.content, TRANSIENT_PATTERNS):
            return PolicyDecision("discard", "transient statement (not durable)")
        if not self._is_specific_enough(cand, ep):
            return PolicyDecision("discard", "not specific to user/agent/organization")

        # No existing fact
        if not existing:
            return PolicyDecision("promote", "no existing fact in this scope")

        # Key match path
        key_matches = [
            f for f in existing
            if f.metadata.get("fact_key") == cand.key
        ]

        if key_matches:
            same_value = next(
                (f for f in key_matches
                if self._values_equal(f.metadata.get("fact_value", ""), cand.value)),
                None,
            )
            if same_value and len(key_matches) == 1:
                return PolicyDecision(
                    "merge",
                    f"same key '{cand.key}', same value → reinforce",
                    target=same_value,
                )
            # Different value or multiple survivors → supersede all
            return PolicyDecision(
                "supersede",
                f"key '{cand.key}' changed to '{cand.value}' "
                f"({len(key_matches)} existing)",
                target=key_matches[0],
                all_targets=key_matches,
            )

        # Content similarity fallback (no key match)
        nearest = existing[0]
        if self._content_overlaps(nearest.content, cand.content):
            return PolicyDecision(
                "supersede",
                "near-duplicate content with different value",
                target=nearest,
                all_targets=existing,
            )

        return PolicyDecision("promote", "similar content but distinct fact")

    # ---------- Stage 6: Execute Decision ----------

    def _execute_decision(
        self,
        decision: PolicyDecision,
        cand: Candidate,
        scope_type: str,
        scope_id: str,
        ep: EpisodeRecord
    ) -> None:
        """Execute the policy decision."""
        
        if decision.action == "discard":
            logger.debug(
                f"[Policy] DISCARD ({decision.reason}): '{cand.content[:50]}'"
            )
            return

        if decision.action == "promote":
            fact = self.semantic.add_fact(
                content=cand.content,
                scope_type=scope_type,
                scope_id=scope_id,
                confidence=cand.confidence,
                category=cand.fact_type,
                source_episode_id=ep.episode_id,
                trigger="promotion",
                status="current",
                metadata={"fact_key": cand.key, "fact_value": cand.value}
            )
            
            ep.promoted_to.append({
                "type": "semantic_new",
                "scope": scope_type,
                "scope_id": scope_id,
                "category": cand.fact_type,
                "content": cand.content,
                "fact_id": fact.fact_id
            })
            
            logger.info(
                f"[Policy] PROMOTE [{scope_type}:{scope_id}] '{cand.content[:50]}'"
            )
            return

        if decision.action == "merge":
            old = decision.target
            
            self.semantic.bump_confidence(
                fact_id=old.fact_id,
                extra_episode_id=ep.episode_id
            )
            
            ep.promoted_to.append({
                "type": "semantic_reinforced",
                "fact_id": old.fact_id,
                "content": cand.content
            })
            
            logger.info(
                f"[Policy] MERGE into {old.fact_id}: '{cand.content[:50]}'"
            )
            return

        if decision.action == "supersede":
            targets = decision.targets_to_supersede()

            if not targets:
                logger.warning("[Policy] SUPERSEDE with no targets - promoting instead")
                decision.action = "promote"
                # fall through to promote block... 
                # easier: just call add_fact directly here
                fact = self.semantic.add_fact(
                    content=cand.content,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    confidence=cand.confidence,
                    category=cand.fact_type,
                    source_episode_id=ep.episode_id,
                    trigger="promotion",
                    status="current",
                    writing_agent_id=ep.agent_id,
                    metadata={"fact_key": cand.key, "fact_value": cand.value},
                )
                ep.promoted_to.append({
                    "type": "semantic_new",
                    "fact_id": fact.fact_id,
                    "content": cand.content,
                })
                return

            primary_old = targets[0]
            new_fact = self.semantic.supersede_fact(
                old_fact=primary_old,
                new_content=cand.content,
                scope_type=scope_type,
                scope_id=scope_id,
                source_episode_id=ep.episode_id,
                trigger="user_correction",
                extra_metadata={"fact_key": cand.key, "fact_value": cand.value},
            )

            logger.info(
                f"[Policy] SUPERSEDE {primary_old.fact_id[:8]} → "
                f"{new_fact.fact_id[:8]}: '{cand.content[:50]}'"
            )

            # Mark ALL remaining duplicates as superseded too
            # This fixes your current state where two facts survived
            for old_fact in targets[1:]:
                try:
                    self.semantic.mark_superseded(
                        fact_id=old_fact.fact_id,
                        superseded_by=new_fact.fact_id,
                    )
                    logger.info(
                        f"[Policy] Cleaned up duplicate {old_fact.fact_id[:8]} "
                        f"(also superseded by {new_fact.fact_id[:8]})"
                    )
                except Exception as exc:
                    logger.error(
                        f"[Policy] Duplicate cleanup failed "
                        f"{old_fact.fact_id[:8]}: {exc}"
                    )
                    raise

            ep.promoted_to.append({
                "type": "semantic_superseded",
                "old_fact_ids": [f.fact_id for f in targets],
                "new_fact_id": new_fact.fact_id,
                "content": cand.content,
            })

    # ---------- Helper Methods ----------

    def _matches_any(self, text: str, patterns: List[str]) -> bool:
        """Check if text contains any pattern."""
        t = text.lower()
        return any(p in t for p in patterns)

    def _is_specific_enough(self, cand: Candidate, ep: EpisodeRecord) -> bool:
        """Verify fact mentions specific entities, not generic advice."""
        t = cand.content.lower()
        
        # User-specific facts must mention user
        if ep.user_id.lower() in t:
            return True
        
        # Agent lessons must mention agent
        if cand.fact_type == "lesson":
            if ep.agent_id.lower() in t or "assistant" in t:
                return True
        
        # Org facts must name a team/system
        if cand.fact_type == "policy":
            required = ["team", "department", "company", "uses", "requires", "runs"]
            return any(w in t for w in required)
        
        return False

    def _values_equal(self, old_value: str, new_value: str) -> bool:
        """Check if two values are semantically equal."""
        stopwords = {"the", "a", "an", "is", "are", "for", "to", "of", "and"}
        
        normalize = lambda s: {
            w for w in re.findall(r"[a-z0-9]+", s.lower())
            if w not in stopwords
        }
        
        return normalize(old_value) == normalize(new_value)

    def _content_overlaps(self, old: str, new: str) -> bool:
        """Check if two fact contents are similar (Jaccard similarity)."""
        words = lambda s: {w for w in re.findall(r"[a-z0-9]{4,}", s.lower())}
        
        old_words, new_words = words(old), words(new)
        
        if not old_words or not new_words:
            return False
        
        intersection = len(old_words & new_words)
        union = len(old_words | new_words)
        
        return (intersection / union) >= 0.5

    # ================================================================
    # PROCEDURAL PIPELINE
    # ================================================================

    def _run_procedural_pipeline(self, episode: EpisodeRecord) -> None:
        """Detect and promote procedural skills."""
        
        # Hard gates
        if episode.outcome != "success":
            logger.debug(
                f"[Procedural] Skipped (outcome={episode.outcome})"
            )
            return
        
        if len(episode.trajectory) < 4:
            logger.debug(
                f"[Procedural] Skipped (only {len(episode.trajectory)} steps)"
            )
            return
        
        # Check for execution verbs
        trajectory_text = " ".join(episode.trajectory).lower()
        action_verbs = [
            "clicked", "opened", "submitted", "configured",
            "ran", "executed", "installed"
        ]
        
        if not any(verb in trajectory_text for verb in action_verbs):
            logger.debug(
                "[Procedural] Skipped (no execution verbs - likely Q&A)"
            )
            return

        # LLM classification
        prompt = f"""Is this a REUSABLE PROCEDURE the user actually PERFORMED?
Not Q&A, not a preference, not information retrieval.

Goal: {episode.goal}
Steps:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(episode.trajectory))}

Return JSON:
{{
  "is_procedure": true/false,
  "reasoning": "...",
  "title": "...",
  "markdown_content": "..."
}}"""

        try:
            client = llm_client()
            resp = client.chat.completions.create(
                model=llm_model(),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception as exc:
            logger.error(f"[Procedural] Classification failed: {exc}")
            return

        # Policy gates
        if not data.get("is_procedure", False):
            logger.debug(
                f"[Procedural] LLM rejected: {data.get('reasoning', 'N/A')}"
            )
            return

        title = data.get("title", "")
        if any(w in title.lower() for w in ["prefer", "like", "want", "need"]):
            logger.debug(
                "[Procedural] Rejected (preference in title, not procedure)"
            )
            return

        # Promote as candidate
        self.procedural.add_skill(
            title=title or "Procedure",
            content=data.get("markdown_content", ""),
            description=data.get("description", ""),
            scope=episode.scope,
            status="candidate",
            source_episode_ids=[episode.episode_id],
            confidence=0.7,
            trigger="trajectory_promotion"
        )
        
        episode.promoted_to.append({
            "type": "procedural_candidate",
            "title": title
        })
        
        logger.info(f"[Procedural] Created candidate: '{title}'")