"""
MNEMOSYNE — Oracle Agent (Layer 3)
SYNC — runs before every LLM inference call.

Strategy:
  1. Semantic search: vector similarity against current query
  2. Structural search: type-based lookup (always inject constraints + commitments)
  3. Merge + deduplicate results
  4. Relevance gate: score >= RELEVANCE_THRESHOLD
  5. Token budget: inject at most MAX_MEMORY_TOKENS into the prompt
  6. Mark all retrieved memories as recalled (heat boost)
"""

from .models import MemoryObject, MemoryType, MemoryStatus, RetrievalResult
from .memory_store import MemoryStore


# Oracle Config

RELEVANCE_THRESHOLD = 0.50

MAX_MEMORY_TOKENS   = 300      # hard cap on tokens injected per turn
ALWAYS_INJECT_TYPES = [        # these types always get injected regardless of score
    MemoryType.CONSTRAINT,
    MemoryType.COMMITMENT,
]
TOP_K_SEMANTIC      = 8        # how many semantic candidates to consider


# Oracle Agent

class Oracle:
    """
    Layer 3: Retrieves the most relevant memories for a given turn.
    Output is a compact prompt block injected before the LLM call.

    """

    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve(self, query: str, turn_number: int) -> RetrievalResult:
        """
        Main retrieval method. Returns a RetrievalResult containing:
          - memories: the selected MemoryObjects
          - prompt_block: ready-to-inject string for the LLM
          - stats: semantic_hits, structural_hits, total_tokens
        """
        selected: dict = {}   # memory_id → MemoryObject (deduplication map)
        semantic_hits   = 0
        structural_hits = 0

        # Step 1: Semantic Search
        semantic_results = self.store.semantic_search(
            query=query,
            top_k=TOP_K_SEMANTIC,
            threshold=RELEVANCE_THRESHOLD,
        )
        for mem, score in semantic_results:
            if mem.memory_id not in selected:
                selected[mem.memory_id] = mem
                semantic_hits += 1

        # Step 2: Structural Search (always-inject types)
        for mem_type in ALWAYS_INJECT_TYPES:
            type_memories = self.store.get_by_type(mem_type)
            for mem in type_memories:
                if mem.status != MemoryStatus.EVICTED:
                    if mem.memory_id not in selected:
                        selected[mem.memory_id] = mem
                        structural_hits += 1

        # Step 3: Sort by relevance (heat * recency)
        def relevance_score(mem: MemoryObject) -> float:
            recency = 1.0 / (1.0 + (turn_number - mem.last_recalled_turn) * 0.01)
            return mem.heat * 0.6 + recency * 0.4

        sorted_memories = sorted(
            selected.values(),
            key=relevance_score,
            reverse=True,
        )

        # Step 4: Apply Token Budget
        final_memories = []
        total_tokens   = 0
        for mem in sorted_memories:
            fragment         = mem.to_prompt_fragment()
            estimated_tokens = len(fragment.split()) + 3   # rough token estimate
            if total_tokens + estimated_tokens > MAX_MEMORY_TOKENS:
                break
            final_memories.append(mem)
            total_tokens += estimated_tokens

        # Step 5: Mark recalled (heat boost)
        for mem in final_memories:
            self.store.mark_recalled(mem.memory_id, turn_number)

        # Step 6: Build prompt block
        prompt_block = self._build_prompt_block(final_memories)

        return RetrievalResult(
            memories=final_memories,
            total_tokens=total_tokens,
            semantic_hits=semantic_hits,
            structural_hits=structural_hits,
            prompt_block=prompt_block,
        )

    def _build_prompt_block(self, memories: list) -> str:
        """
        Builds the compact memory context block injected into the LLM prompt.
        Grouped by type for clarity. Constraints and commitments come first.
        """
        if not memories:
            return ""

        groups: dict = {}
        for mem in memories:
            t = mem.type.value
            if t not in groups:
                groups[t] = []
            groups[t].append(mem)

        lines = ["<memory_context>"]
        priority_order = [
            MemoryType.CONSTRAINT.value,
            MemoryType.COMMITMENT.value,
            MemoryType.PREFERENCE.value,
            MemoryType.FACT.value,
            MemoryType.ENTITY.value,
        ]
        for type_key in priority_order:
            if type_key in groups:
                lines.append(f"  [{type_key.upper()}S]")
                for mem in groups[type_key]:
                    lines.append(f"    {mem.key}: {mem.value}")

        lines.append("</memory_context>")
        return "\n".join(lines)

    def build_system_prompt(self, base_system: str, retrieval: RetrievalResult) -> str:
        """
        Injects the memory block into the full system prompt.
        Memory is placed AFTER the base instructions so it's fresh in context.

        """
        if not retrieval.prompt_block:
            return base_system

        return f"""{base_system}

{retrieval.prompt_block}

Use the memory context above to inform your response naturally.
Do NOT mention that you have a memory system or that you're recalling stored facts.
Apply memories implicitly — adapt your response as if you simply know these things."""