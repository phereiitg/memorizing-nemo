"""
MNEMOSYNE — Curator Agent (Layer 5)
ASYNC — runs after inference. Manages the full memory lifecycle.

Operations:
  ADD    → new memory, no conflict
  UPDATE → existing memory extended or corrected
  DELETE → direct contradiction of existing fact
  NOOP   → duplicate or irrelevant

Also runs the heat decay pass each turn.
"""

import json
import re
from .models import MemoryObject, MemoryType, CuratorOp, CuratorDecision, MemoryStatus
from .memory_store import MemoryStore


# Conflict Detection Thresholds

SIMILARITY_FOR_CONFLICT   = 0.55   # score above this → run conflict/duplicate checks
SIMILARITY_FOR_DUPLICATE  = 0.75   # score above this AND same value → NOOP

NON_CONTRADICTION_PAIRS = [
    {"vegetarian", "vegan"},
    {"vegan", "plant-based"},
    {"vegetarian", "plant-based"},
]


# Curator Agent

class Curator:
    """
    Layer 5: Determines what happens to each extracted memory candidate.
    Resolves conflicts, merges duplicates, runs decay each turn.
    """

    def __init__(self, store: MemoryStore):
        self.store = store

    # Main Entry

    def process(self, candidates: list, turn: int) -> list:
        """
        Process a list of MemoryObject candidates from Sentinel.
        Returns list of CuratorDecision objects (for logging/debugging).
        """
        decisions = []

        # Build O(1) lookup once for all candidates this turn
        active_by_key: dict = {}
        for mem in self.store.get_all_active():
            active_by_key[(mem.key, mem.type)] = mem

        for candidate in candidates:
            decision = self._decide(candidate, turn, active_by_key)
            decisions.append(decision)
            self._execute(decision, turn)

            if decision.operation == CuratorOp.ADD:
                active_by_key[(candidate.key, candidate.type)] = candidate
            elif decision.operation == CuratorOp.UPDATE and decision.target_id:
                # Update the value in cache so subsequent lookups are fresh
                existing = active_by_key.get((candidate.key, candidate.type))
                if existing:
                    existing.value = candidate.value

        return decisions

    def run_decay(self, turn: int) -> list:
        """
        Trigger heat decay pass. Returns list of evicted memory_ids.
        Called once per turn after processing.
        """
        return self.store.apply_decay(turn)

    # Decision Logic

    def _decide(self, candidate: MemoryObject, turn: int,
                active_by_key: dict) -> CuratorDecision:
        """
        Core logic: compare candidate against existing memories to decide operation.
        Uses O(1) key lookup first, then semantic search for cross-key conflicts.
        """

        # Check 1: Same key exact match (O(1) dict lookup)
        existing_same_key = active_by_key.get((candidate.key, candidate.type))
        if existing_same_key:
            if self._values_are_same(candidate.value, existing_same_key.value):
                return CuratorDecision(
                    operation=CuratorOp.NOOP,
                    candidate=candidate,
                    target_id=existing_same_key.memory_id,
                    reason="Duplicate: same key and value already stored",
                )
            else:
                # Different value for same key → UPDATE regardless of contradiction.
                # The key match is already a strong signal this replaces the old memory.
                return CuratorDecision(
                    operation=CuratorOp.UPDATE,
                    candidate=candidate,
                    target_id=existing_same_key.memory_id,
                    reason=f"Update: '{existing_same_key.value}' → '{candidate.value}'",
                )

        # Check 2: Semantic similarity for cross-key conflicts
        similar = self.store.semantic_search(
            query=candidate.embed_text,
            top_k=3,
            threshold=SIMILARITY_FOR_CONFLICT,  
        )

        for existing_mem, score in similar:

            if score >= SIMILARITY_FOR_DUPLICATE:
                # Very high similarity — check for duplicate first.
                if self._values_are_same(candidate.value, existing_mem.value):
                    return CuratorDecision(
                        operation=CuratorOp.NOOP,
                        candidate=candidate,
                        target_id=existing_mem.memory_id,
                        reason=f"Semantic duplicate (score={score:.2f})",
                    )

            if score >= SIMILARITY_FOR_CONFLICT:
                if self._is_contradiction(candidate.value, existing_mem.value):
                    return CuratorDecision(
                        operation=CuratorOp.DELETE,
                        candidate=candidate,
                        target_id=existing_mem.memory_id,
                        reason=f"Contradiction detected (score={score:.2f}): replacing old memory",
                    )

       
        return CuratorDecision(
            operation=CuratorOp.ADD,
            candidate=candidate,
            target_id=None,
            reason="New memory: no conflict or duplicate found",
        )

    # Execution

    def _execute(self, decision: CuratorDecision, turn: int):
        """Apply the decided operation to the memory store."""
        op  = decision.operation
        mem = decision.candidate

        if op == CuratorOp.ADD:
            self.store.add(mem)

        elif op == CuratorOp.UPDATE:
            if decision.target_id:
                self.store.update(decision.target_id, mem.value, turn)
            else:
                self.store.add(mem)   # fallback: add if target missing

        elif op == CuratorOp.DELETE:
            if decision.target_id:
                self.store.delete(decision.target_id)
            self.store.add(mem)       # replace with the new (correct) version

        elif op == CuratorOp.NOOP:
            pass

    # Helpers

    def _values_are_same(self, v1: str, v2: str) -> bool:
        """Normalise and compare two values."""
        return v1.lower().strip() == v2.lower().strip()

    def _is_contradiction(self, new_val: str, old_val: str) -> bool:
        """
        Heuristic contradiction detection.

        FIX: Added NON_CONTRADICTION_PAIRS guard.
        Without it, "vegetarian" vs "vegan" both have zero word overlap
        (100% different words), triggering a false contradiction and
        deleting a valid dietary memory.
        """
        new_lower = new_val.lower().strip()
        old_lower = old_val.lower().strip()

        # Guard: known synonym pairs are never contradictions
        pair = {new_lower, old_lower}
        if pair in NON_CONTRADICTION_PAIRS:
            return False

        # Number change (age, times, phone numbers)
        new_nums = set(re.findall(r'\d+', new_lower))
        old_nums = set(re.findall(r'\d+', old_lower))
        if new_nums and old_nums and new_nums != old_nums:
            return True

        # Word overlap — less than 20% shared words = likely contradiction
        new_words = set(new_lower.split())
        old_words = set(old_lower.split())
        overlap   = len(new_words & old_words)
        total     = len(new_words | old_words)
        if total > 0 and (overlap / total) < 0.2:
            return True

        return False

    # LLM-Based Decision (bonus path)

    def build_conflict_prompt(self, candidate: MemoryObject, similar: list) -> str:
        """
        Builds prompt for LLM-based conflict resolution.
        Use this when rule-based detection is uncertain.
        """
        similar_text = "\n".join([
            f"  - [{m.memory_id}] {m.key}: {m.value} (heat={m.heat:.2f})"
            for m, _ in similar[:3]
        ])

        return f"""You are a memory management agent. Decide what operation to perform.

New candidate memory:
  type: {candidate.type.value}
  key: {candidate.key}
  value: {candidate.value}

Similar existing memories:
{similar_text}

Choose ONE operation and return ONLY JSON:
  {{"operation": "ADD|UPDATE|DELETE|NOOP", "target_id": "<id or null>", "reason": "<one sentence>"}}

Rules:
- ADD: if this is genuinely new information
- UPDATE: if this refines or extends an existing memory (use target_id of that memory)
- DELETE: if this directly contradicts an existing memory (use target_id of contradicted memory)
- NOOP: if this is a duplicate or not worth storing

JSON:"""

    def parse_llm_decision(self, response: str, candidate: MemoryObject) -> CuratorDecision:
        """Parse LLM's JSON response into a CuratorDecision."""
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = re.sub(r"```(?:json)?", "", clean).strip().strip("```")
            data  = json.loads(clean)
            op    = CuratorOp(data.get("operation", "NOOP"))
            return CuratorDecision(
                operation=op,
                candidate=candidate,
                target_id=data.get("target_id"),
                reason=data.get("reason", "LLM-decided"),
            )
        except (json.JSONDecodeError, ValueError):
            return CuratorDecision(
                operation=CuratorOp.ADD,
                candidate=candidate,
                reason="Fallback: LLM parse failed, defaulting to ADD",
            )