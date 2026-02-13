"""
MNEMOSYNE — Multi-Agent Long-Form Memory System
================================================
A 5-layer agent architecture for AI systems that need to remember
information accurately across 1000+ conversation turns.

Quick start:
    - Run python main.py
 
"""

from .engine       import MnemosyneEngine, TurnResult
from .models       import MemoryObject, MemoryType, MemoryStatus, CuratorOp
from .memory_store import MemoryStore
from .sentinel     import Sentinel
from .oracle       import Oracle
from .curator      import Curator

__version__ = "1.0.0"
__all__ = [
    "MnemosyneEngine",
    "TurnResult",
    "MemoryObject",
    "MemoryType",
    "MemoryStatus",
    "CuratorOp",
    "MemoryStore",
    "Sentinel",
    "Oracle",
    "Curator",
]
