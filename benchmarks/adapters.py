import numpy as np
from typing import List, Any, Optional
from .base import VectorStoreAdapter
from epochdb import EpochDB

class EpochDBStoreAdapter(VectorStoreAdapter):
    def __init__(self, db: EpochDB):
        self.db = db
        
    def add(self, payload: Any, embedding: Any, triples: Optional[List[tuple]] = None):
        self.db.add_memory(payload=payload, embedding=embedding, triples=triples)
        
    def query(self, query_emb: Any, top_k: int = 5, **kwargs) -> List[Any]:
        expand_hops = kwargs.get("expand_hops", 0)
        if expand_hops > 0:
            return self.db.retriever.search(query_emb, top_k=top_k, expand_hops=expand_hops)
        return self.db.recall(query_emb, top_k=top_k)
        
    def clear(self):
        # We handle clearing in the client/runner side by recreating the DB or clearing tiers
        self.db.hot_tier.clear()
        self.db.wal.clear()
        
    def checkpoint(self):
        self.db.force_checkpoint()

class ChromaDBStoreAdapter(VectorStoreAdapter):
    def __init__(self, collection_name="benchmark_collection"):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is not installed. Please install it to use this adapter.")
            
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.id_counter = 0

    def add(self, payload: Any, embedding: Any, triples: Optional[List[tuple]] = None):
        # Chroma doesn't use triples, we ignore them
        self.collection.add(
            embeddings=[embedding.tolist() if isinstance(embedding, np.ndarray) else embedding],
            documents=[str(payload)],
            ids=[f"id_{self.id_counter}"]
        )
        self.id_counter += 1
        
    def query(self, query_emb: Any, top_k: int = 5, **kwargs) -> List[Any]:
        results = self.collection.query(
            query_embeddings=[query_emb.tolist() if isinstance(query_emb, np.ndarray) else query_emb],
            n_results=top_k
        )
        
        # Wrap results to match EpochDB's UnifiedMemoryAtom-like structure (with a .payload attribute)
        class MockResult:
            def __init__(self, payload):
                self.payload = payload
        
        return [MockResult(doc) for doc in results["documents"][0]]
        
    def clear(self):
        self.client.reset()
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
        self.id_counter = 0
        
    def checkpoint(self):
        # Chroma is persistent or in-memory, no explicit cold-tier checkpoint
        pass
