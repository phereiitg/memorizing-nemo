"""
MNEMOSYNE — Memory Store (Layer 2) [UPGRADED — ChromaDB]
3-Tier hybrid storage: Hot (ring buffer) → Warm (ChromaDB) → Cold (SQLite)

Hot  Tier: Python deque, last 40 extracted memories. O(1) access, no I/O.
Warm Tier: ChromaDB with cosine similarity embeddings. Semantic search.
Cold Tier: SQLite for structured persistence. Survives restarts.
"""

import sqlite3
import time
import chromadb
from collections import deque
from typing import Optional
from .models import MemoryObject, MemoryType, MemoryStatus


# Hot Tier: Ring Buffer

class HotTier:
    """
    Holds the last HOT_SIZE extracted MemoryObjects in a ring buffer.
    Instant O(1) access. Auto-evicts oldest when full.
    """
    HOT_SIZE = 40

    def __init__(self):
        self._buffer: deque = deque(maxlen=self.HOT_SIZE)

    def add(self, memory: MemoryObject):
        self._buffer.appendleft(memory)

    def get_all(self) -> list:
        return list(self._buffer)

    def find_by_id(self, memory_id: str) -> Optional[MemoryObject]:
        for m in self._buffer:
            if m.memory_id == memory_id:
                return m
        return None

    def __len__(self):
        return len(self._buffer)


# Warm Tier: ChromaDB Vector Store

class WarmTier:
    """
    Stores memories using ChromaDB for semantic vector search.
    Wraps the vector DB to present the same interface as the old TF-IDF tier.

    FIX BUG-1 + BUG-2: Accepts persistence_path and collection_name so that
    multiple sessions never share a collection.

    FIX WARN-1: Explicitly sets hnsw:space=cosine so distances are always
    in the range [0, 2], making 1/(1+d) scores consistent and predictable.
    """

    def __init__(self, persistence_path: str = "mnemosyne_chroma",
                 collection_name: str = "memories"):
        # 1. ChromaDB persistent client (saves to disk at persistence_path)

        self.client = chromadb.PersistentClient(path=persistence_path)

        # 2. Get or create collection.

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 3. In-memory cache for fast full-object access.
        #    Chroma stores text + metadata but not our full MemoryObject.
        self._memories: dict = {}   # memory_id -> MemoryObject

    def upsert(self, memory: MemoryObject):
        """Add or update a memory in both ChromaDB and the local cache."""
        self._memories[memory.memory_id] = memory
        self.collection.upsert(
            ids=[memory.memory_id],
            documents=[memory.embed_text],
            metadatas=[{
                "type":   memory.type.value,
                "heat":   memory.heat,
                "status": memory.status.value,
            }],
        )

    def remove(self, memory_id: str):
        """Delete from both local cache and ChromaDB."""
        self._memories.pop(memory_id, None)
        try:
            self.collection.delete(ids=[memory_id])
        except (ValueError, Exception):
            pass

    def search(self, query: str, top_k: int = 8,
               threshold: float = 0.15) -> list:
        """
        Semantic search using ChromaDB vector embeddings.
        Returns list of (MemoryObject, score) sorted descending.

        FIX: threshold default changed from 0.0 to 0.15 to match
        MemoryStore.semantic_search and avoid returning noise.

        FIX: n_results capped at len(_memories) to prevent ChromaDB
        ValueError when top_k > number of documents in collection.
        """
        if not self._memories:
            return []

        # ChromaDB raises ValueError if n_results > collection size
        k = min(top_k, len(self._memories))

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
        )

        hits = []
        ids       = results["ids"][0]
        distances = results["distances"][0]

        for i, mem_id in enumerate(ids):
            mem = self._memories.get(mem_id)
            if mem is None:
                continue

            # Convert cosine distance (0-2) to similarity score (0.33-1.0).
            # distance=0.0 -> score=1.0 (perfect match)
            # distance=1.0 -> score=0.5
            # distance=2.0 -> score=0.33 (opposite)
            raw_score = 1.0 / (1.0 + distances[i])

            # Weight by heat: hotter memories rank higher when scores are close
            adjusted_score = raw_score * (0.7 + 0.3 * mem.heat)

            if adjusted_score >= threshold:
                hits.append((mem, adjusted_score))

        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    def get_all(self) -> list:
        return list(self._memories.values())

    def get_by_id(self, memory_id: str) -> Optional[MemoryObject]:
        return self._memories.get(memory_id)

    def get_by_type(self, memory_type: MemoryType) -> list:
        return [m for m in self._memories.values() if m.type == memory_type]


# Cold Tier: SQLite Persistent Store
class ColdTier:
    """
    SQLite-backed persistent store. Survives restarts.
    Used for structured lookup of facts, preferences, constraints.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id          TEXT PRIMARY KEY,
                type               TEXT NOT NULL,
                key_name           TEXT NOT NULL,
                value              TEXT NOT NULL,
                source_turn        INTEGER DEFAULT 0,
                last_recalled_turn INTEGER DEFAULT 0,
                heat               REAL DEFAULT 1.0,
                confidence         REAL DEFAULT 1.0,
                status             TEXT DEFAULT 'active',
                embed_text         TEXT DEFAULT '',
                created_at         REAL,
                updated_at         REAL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON memories(status)")
        self._conn.commit()

    def upsert(self, memory: MemoryObject):
        self._conn.execute("""
            INSERT INTO memories
                (memory_id, type, key_name, value, source_turn, last_recalled_turn,
                 heat, confidence, status, embed_text, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(memory_id) DO UPDATE SET
                value              = excluded.value,
                last_recalled_turn = excluded.last_recalled_turn,
                heat               = excluded.heat,
                confidence         = excluded.confidence,
                status             = excluded.status,
                updated_at         = excluded.updated_at
        """, (
            memory.memory_id, memory.type.value, memory.key, memory.value,
            memory.source_turn, memory.last_recalled_turn,
            memory.heat, memory.confidence, memory.status.value,
            memory.embed_text, memory.created_at, memory.updated_at,
        ))
        self._conn.commit()

    def delete(self, memory_id: str):
        self._conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        self._conn.commit()

    def get_by_type(self, memory_type: MemoryType) -> list:
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE type = ? AND status != 'evicted'",
            (memory_type.value,)
        ).fetchall()
        return [self._row_to_obj(r) for r in rows]

    def get_all_active(self) -> list:
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE status IN ('active', 'decaying')"
        ).fetchall()
        return [self._row_to_obj(r) for r in rows]

    def get_by_id(self, memory_id: str) -> Optional[MemoryObject]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_obj(row) if row else None

    def _row_to_obj(self, row) -> MemoryObject:
        return MemoryObject(
            memory_id=row[0], type=MemoryType(row[1]),
            key=row[2], value=row[3],
            source_turn=row[4], last_recalled_turn=row[5],
            heat=row[6], confidence=row[7],
            status=MemoryStatus(row[8]), embed_text=row[9],
            created_at=row[10], updated_at=row[11],
        )

    def export_all(self) -> list:
        return [m.to_dict() for m in self.get_all_active()]


# Unified Memory Store

class MemoryStore:
    HEAT_DECAY_PER_TURN  = 0.04
    HEAT_RECALL_BOOST    = 0.25
    HEAT_EVICT_THRESHOLD = 0.08

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path

        # Derive isolated ChromaDB path and collection name from db_path.
        if db_path == ":memory:":
            chroma_path     = "mnemosyne_chroma_ephemeral"
            collection_name = "mem_ephemeral"
        else:
            base            = db_path.replace(".db", "")
            chroma_path     = f"{base}_chroma"
            collection_name = f"mem_{base.split('/')[-1]}"

        self.hot  = HotTier()
        self.warm = WarmTier(
            persistence_path=chroma_path,
            collection_name=collection_name,
        )
        self.cold = ColdTier(db_path)

        # Sync persisted SQLite memories back into ChromaDB warm tier on startup.
        # upsert() is idempotent -- safe even if Chroma already has these IDs.
        for m in self.cold.get_all_active():
            self.warm.upsert(m)

    # Write Operations

    def add(self, memory: MemoryObject):
        self.hot.add(memory)
        self.warm.upsert(memory)
        self.cold.upsert(memory)

    def update(self, memory_id: str, new_value: str, turn: int):
        mem = self.warm.get_by_id(memory_id)
        if mem is None:
            mem = self.cold.get_by_id(memory_id)
        if mem is None:
            return
        mem.value      = new_value
        mem.updated_at = time.time()
        mem.last_recalled_turn = turn
        mem.embed_text = f"{mem.key}: {new_value}"
        self.warm.upsert(mem)
        self.cold.upsert(mem)

    def delete(self, memory_id: str):
        self.warm.remove(memory_id)
        self.cold.delete(memory_id)

    def mark_recalled(self, memory_id: str, turn: int):
        mem = self.warm.get_by_id(memory_id)
        if mem:
            mem.heat = min(1.0, mem.heat + self.HEAT_RECALL_BOOST)
            mem.last_recalled_turn = turn
            mem.status = MemoryStatus.ACTIVE
            self.warm.upsert(mem)
            self.cold.upsert(mem)

    # Read Operations

    def semantic_search(self, query: str, top_k: int = 6,
                        threshold: float = 0.15) -> list:
        return self.warm.search(query, top_k=top_k, threshold=threshold)

    def get_by_type(self, memory_type: MemoryType) -> list:
        return self.warm.get_by_type(memory_type)

    def get_all_active(self) -> list:
        return [m for m in self.warm.get_all()
                if m.status != MemoryStatus.EVICTED]

    def get_snapshot(self) -> list:
        return sorted(self.warm.get_all(), key=lambda m: m.heat, reverse=True)

    # Decay Pass

    def apply_decay(self, current_turn: int) -> list:
        """
        Subtract HEAT_DECAY_PER_TURN from all memories not recalled this turn.
        Returns list of evicted memory_ids.
        Iterates list() copy of warm._memories -- safe against mutation.
        """
        evicted = []
        for mem in self.warm.get_all():   # get_all() returns a list copy
            if mem.last_recalled_turn < current_turn:
                mem.heat = max(0.0, mem.heat - self.HEAT_DECAY_PER_TURN)

                if mem.heat < self.HEAT_EVICT_THRESHOLD:
                    mem.status = MemoryStatus.EVICTED
                    self.warm.remove(mem.memory_id)
                    self.cold.delete(mem.memory_id)
                    evicted.append(mem.memory_id)
                elif mem.heat < 0.3:
                    mem.status = MemoryStatus.DECAYING
                    self.warm.upsert(mem)
                    self.cold.upsert(mem)
                else:
                    mem.status = MemoryStatus.ACTIVE
                    self.warm.upsert(mem)
                    self.cold.upsert(mem)
        return evicted