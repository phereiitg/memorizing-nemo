"""
MNEMOSYNE — Memory Store (Layer 2)
3-Tier hybrid storage: Hot (ring buffer) → Warm (vector) → Cold (SQLite)

Hot  Tier: Python deque, last 20 turns' extracted memories. O(1) access, no I/O.
Warm Tier: Cosine similarity over TF-IDF vectors (no external deps). Semantic search.
Cold Tier: SQLite for structured facts, preferences, constraints. Survives restarts.
"""