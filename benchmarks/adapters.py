"""
benchmarks/adapters.py — EpochDB adapter for the benchmark suite.
"""

import numpy as np
from typing import List, Any, Optional
from epochdb import EpochDB


class EpochDBStoreAdapter:
    """Thin wrapper around EpochDB for use in benchmark scripts."""

    def __init__(self, db: EpochDB):
        self.db = db

    def add(self, payload: Any, embedding: Any, triples: Optional[List[tuple]] = None):
        self.db.add_memory(payload=payload, embedding=embedding, triples=triples or [])

    def add_batch(self, items: List[dict]):
        self.db.add_memory_batch(items)

    def query(self, query_emb: Any, top_k: int = 5, **kwargs) -> List[Any]:
        expand_hops    = kwargs.get("expand_hops", 1)
        query_entities = kwargs.get("query_entities", None)
        return self.db.recall(
            query_emb,
            top_k=top_k,
            expand_hops=expand_hops,
            query_entities=query_entities,
        )

    def clear(self):
        self.db.hot_tier.clear()
        self.db.wal.clear()

    def checkpoint(self):
        self.db.force_checkpoint()
