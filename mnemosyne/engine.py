"""
MNEMOSYNE — Engine (Orchestrator)
The single public interface. Wires all agents together.

Usage:
    engine = MnemosyneEngine()
    result = engine.chat(user_message="Hi, my name is Arjun")
    print(result.response)        # LLM response
    print(result.memories_used)   # which memories were injected
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable

from .models       import MemoryObject, MemoryType, CuratorOp
from .memory_store import MemoryStore
from .sentinel     import Sentinel
from .oracle       import Oracle
from .curator      import Curator


# Turn Result

@dataclass
class TurnResult:
    turn_number:        int
    user_message:       str
    response:           str   = ""
    memories_used:      list  = field(default_factory=list)  # MemoryObjects injected
    memories_added:     list  = field(default_factory=list)  # new memories this turn
    evicted:            list  = field(default_factory=list)  # evicted memory IDs
    prompt_block:       str   = ""
    latency_ms:         float = 0.0
    sentinel_extracted: int   = 0
    curator_ops:        dict  = field(default_factory=dict)  # op counts per CuratorOp


# Engine

# Rolling window for conversation history passed to LLM.
# 20 messages = 10 full turns. Prevents unbounded RAM growth at 1000+ turns.
HISTORY_WINDOW = 20


class MnemosyneEngine:
    """
    Main orchestrator. Coordinates:
      Oracle (SYNC) → LLM (SYNC) → Sentinel + Curator (background thread)

    Threading model:
      - Oracle.retrieve() and LLM call run on the calling thread.
      - Sentinel.extract() and Curator.process() run in a daemon thread
        started after the LLM response is ready.
      - thread.join(timeout) is used so TurnResult fields are populated
        before chat() returns. This is intentionally blocking for demo
        visibility. In a streaming production system, remove join() and
        read async results via engine.last_async_result.
      - self._lock guards Sentinel+Curator so concurrent chat() calls
        from different threads never write to the store simultaneously.
        For true concurrent multi-user use, give each session its own
        MnemosyneEngine instance instead.
    """

    BASE_SYSTEM_PROMPT = """You are a helpful, intelligent assistant.
You have access to remembered context about this user from previous conversations.
Use that context naturally — never mention that you have a memory system."""

    def __init__(
        self,
        db_path: str = ":memory:",
        llm_fn: Optional[Callable] = None,
        confidence_threshold: float = 0.65,
        verbose: bool = False,
    ):
        """
        Args:
            db_path: SQLite path. \":memory:\" for ephemeral, file path for persistence.
                     ChromaDB path is derived automatically from this value.
            llm_fn:  Optional callable(system_prompt, user_message, history) → str.
                     If None, engine works in extraction/retrieval-only mode.
            confidence_threshold: Sentinel gate — only memories above this are stored.
            verbose: Print per-turn debug info to stdout.
        """
        self.db_path   = db_path

        self.store     = MemoryStore(db_path=db_path)
        self.sentinel  = Sentinel(confidence_threshold=confidence_threshold)
        self.oracle    = Oracle(store=self.store)
        self.curator   = Curator(store=self.store)
        self.llm_fn    = llm_fn
        self.verbose   = verbose

        self.turn_number = 0

        self.history: list = []

        self._lock = threading.Lock()

        # Populated by the async thread; readable after join() returns.
        self._last_async: dict = {}

    # Public API

    def chat(self, user_message: str) -> TurnResult:
        """
        Process one conversation turn end-to-end.
        Returns a TurnResult with response + full audit trail.

        Execution order:
          1. Oracle retrieves relevant memories (SYNC, ~20ms)
          2. LLM is called with memory-injected prompt (SYNC, ~200ms)
          3. History is updated
          4. Sentinel + Curator run in background thread
          5. join(timeout) waits for background thread so TurnResult is complete
          6. Latency is measured and returned
        """
        t_start = time.time()
        self.turn_number += 1
        turn = self.turn_number

        result = TurnResult(turn_number=turn, user_message=user_message)

        # SYNC: Oracle retrieves relevant memories BEFORE inference
        retrieval = self.oracle.retrieve(query=user_message, turn_number=turn)
        result.memories_used = retrieval.memories
        result.prompt_block  = retrieval.prompt_block

        if self.verbose:
            print(f"\n[Turn {turn}] Oracle: {retrieval.semantic_hits} semantic + "
                  f"{retrieval.structural_hits} structural hits, "
                  f"~{retrieval.total_tokens} tokens injected")

        # SYNC: Build full system prompt and call LLM
        system_prompt = self.oracle.build_system_prompt(
            self.BASE_SYSTEM_PROMPT, retrieval
        )
        response = self._call_llm(system_prompt, user_message)
        result.response = response

        # Update rolling history window.
        # FIX: slice to HISTORY_WINDOW AFTER appending so list never grows
        # beyond the cap. Previously the full history accumulated forever.
        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant",  "content": response})
        self.history = self.history[-HISTORY_WINDOW:]   # keep last 10 turns

        # BACKGROUND: Sentinel extracts + Curator updates
        # Capture turn in closure to avoid closing over a mutable variable.
        captured_turn    = turn
        captured_message = user_message

        async_output: dict = {}

        def async_memory_ops():
            with self._lock:
                extraction = self.sentinel.extract(captured_message, captured_turn)

                if self.verbose and extraction.filtered_in:
                    print(f"[Turn {captured_turn}] Sentinel extracted: "
                          + ", ".join(f"{m.key}={m.value}"
                                      for m in extraction.filtered_in))

                decisions = self.curator.process(extraction.filtered_in, captured_turn)

                op_counts = {op.value: 0 for op in CuratorOp}
                added = []
                for d in decisions:
                    op_counts[d.operation.value] += 1
                    if d.operation in (CuratorOp.ADD, CuratorOp.UPDATE):
                        added.append(d.candidate)

                evicted = self.curator.run_decay(captured_turn)

                if self.verbose and evicted:
                    print(f"[Turn {captured_turn}] Curator evicted: {evicted}")

                # Write to local dict — no shared mutable state with main thread
                async_output["sentinel_extracted"] = len(extraction.filtered_in)
                async_output["memories_added"]     = added
                async_output["curator_ops"]        = op_counts
                async_output["evicted"]            = evicted

        thread = threading.Thread(target=async_memory_ops, daemon=True)
        thread.start()

        # Block until background work completes (or times out).
        # Intentionally blocking here so TurnResult is fully populated
        # before returning — useful for demos, tests, and sequential chat.
        # In a streaming production system, remove join() and read
        # self._last_async after the fact.
        thread.join(timeout=5.0)

        # FIX: only copy async results into TurnResult if the thread actually
        # finished. If it timed out, fields stay at their safe defaults.
        if not thread.is_alive():
            result.sentinel_extracted = async_output.get("sentinel_extracted", 0)
            result.memories_added     = async_output.get("memories_added", [])
            result.curator_ops        = async_output.get("curator_ops", {})
            result.evicted            = async_output.get("evicted", [])
        elif self.verbose:
            print(f"[Turn {turn}] WARNING: async thread timed out after 5s")

        result.latency_ms = round((time.time() - t_start) * 1000, 1)
        return result

    # Convenience Methods

    def get_memory_snapshot(self) -> list:
        """Return all current memories sorted by heat (highest first)."""
        return self.store.get_snapshot()

    def get_memories_by_type(self, memory_type: MemoryType) -> list:
        return self.store.get_by_type(memory_type)

    def inject_memory(self, key: str, value: str,
                      mem_type: MemoryType = MemoryType.FACT,
                      confidence: float = 0.99) -> MemoryObject:
        """Manually inject a memory — useful for seeding known facts."""
        mem = MemoryObject(
            type=mem_type, key=key, value=value,
            source_turn=self.turn_number,
            last_recalled_turn=self.turn_number,
            heat=confidence, confidence=confidence,
        )
        self.store.add(mem)
        return mem

    def reset(self):
        """
        Full reset — clears all memory and conversation history.

        FIX 1: Passes self.db_path to new MemoryStore so persistence
        path stays consistent. Previously called MemoryStore() with no
        args, which defaulted to ':memory:' even if the engine was
        initialized with a file path.

        FIX 2: Re-initialises all four agents (including Sentinel) for
        consistency, even though Sentinel carries no state.
        """
        self.store    = MemoryStore(db_path=self.db_path)
        self.sentinel = Sentinel(
            confidence_threshold=self.sentinel.threshold
        )
        self.oracle   = Oracle(store=self.store)
        self.curator  = Curator(store=self.store)
        self.turn_number = 0
        self.history  = []

    # LLM Call

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Route to provided LLM function or return a stub."""
        if self.llm_fn is None:
            return f"[LLM stub — turn {self.turn_number}] No LLM function provided."
        try:
            return self.llm_fn(
                system_prompt=system_prompt,
                user_message=user_message,
                history=self.history,   # already windowed to last 10 turns
            )
        except Exception as e:
            return f"[LLM error: {e}]"

    # Stats

    def stats(self) -> dict:
        """Summary stats for the current session."""
        all_mems = self.store.get_snapshot()
        type_counts = {}
        for m in all_mems:
            t = m.type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "turn":           self.turn_number,
            "total_memories": len(all_mems),
            "by_type":        type_counts,
            "avg_heat":       round(
                sum(m.heat for m in all_mems) / len(all_mems), 3
            ) if all_mems else 0,
            "history_turns":  len(self.history) // 2,
        }