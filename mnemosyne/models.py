"""
MNEMOSYNE — Data Models
All memory objects, types, and operations are defined here.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from enum import Enum
import time
import uuid

# Memory Types

class MemoryType(str, Enum):
    PREFERENCE   = "preference"    # user likes/dislikes, habits
    FACT         = "fact"          # stated facts about the user's world
    ENTITY       = "entity"        # named people, places, things
    CONSTRAINT   = "constraint"    # hard limits
    COMMITMENT   = "commitment"    # promises made

class MemoryStatus(str, Enum):
    ACTIVE   = "active"     # heat > 0.3, in full use
    DECAYING = "decaying"   # 0.1 <= heat <= 0.3, watch list
    EVICTED  = "evicted"    # heat < 0.1, removed from active retrieval

class CuratorOp(str, Enum):
    ADD    = "ADD"    # brand new memory
    UPDATE = "UPDATE" # extend/replace existing
    DELETE = "DELETE" # contradiction — remove old
    NOOP   = "NOOP"   # duplicate or irrelevant

# Core Memory Object

@dataclass
class MemoryObject:
    """
    A single unit of memory. Every memory the system holds is one of these.
    """
    memory_id:         str          = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    type:              MemoryType   = MemoryType.FACT
    key:               str          = ""          # short label, e.g. "language_preference"
    value:             str          = ""          # the actual remembered value
    source_turn:       int          = 0           # which turn this was extracted from
    last_recalled_turn: int         = 0           # last turn this was retrieved by Oracle
    heat:              float        = 1.0         # 0.0–1.0 decay score
    confidence:        float        = 1.0         # Sentinel extraction confidence
    status:            MemoryStatus = MemoryStatus.ACTIVE
    created_at:        float        = field(default_factory=time.time)
    updated_at:        float        = field(default_factory=time.time)
    # Embedding is stored separately in the warm tier; this is the raw text for embedding
    embed_text:        str          = ""

    def __post_init__(self):
        if not self.embed_text:
            self.embed_text = f"{self.key}: {self.value}"

    def to_dict(self) -> dict:
        return {
            "memory_id":          self.memory_id,
            "type":               self.type.value,
            "key":                self.key,
            "value":              self.value,
            "source_turn":        self.source_turn,
            "last_recalled_turn": self.last_recalled_turn,
            "heat":               round(self.heat, 3),
            "confidence":         round(self.confidence, 3),
            "status":             self.status.value,
            "created_at":         self.created_at,
            "updated_at":         self.updated_at,
        }

    def to_prompt_fragment(self) -> str:
        """Compact representation injected into LLM prompts — no wasted tokens."""
        return f"[{self.type.value.upper()}] {self.key}: {self.value}"
    
# Extraction Result (Sentinel Output)

@dataclass
class ExtractionResult:
    candidates : list
    raw_turn : str
    turn_number : int
    filtered_in : list = field(default_factory=list)  # candidates that passed confidence filter
    filtered_out : list = field(default_factory=list) # candidates that failed confidence filter

# Retrieval Result (Oracle Output)

@dataclass
class RetrievalResult:
    memories:         list   # list of MemoryObject
    total_tokens:     int    = 0
    semantic_hits:    int    = 0
    structural_hits:  int    = 0
    prompt_block:     str    = ""  # the final string injected into the LLM prompt


# Curator Decision

@dataclass
class CuratorDecision:
    operation:    CuratorOp
    candidate:    object  # MemoryObject
    target_id:    Optional[str] = None  # for UPDATE/DELETE, the existing memory's ID
    reason:       str           = ""

