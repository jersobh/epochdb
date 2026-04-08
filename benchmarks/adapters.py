import numpy as np
from typing import List, Any, Optional
from .base import VectorStoreAdapter
from epochdb import EpochDB

class EpochDBStoreAdapter(VectorStoreAdapter):
    def __init__(self, db: EpochDB):
        self.db = db
        
    def add(self, payload: Any, embedding: Any, triples: Optional[List[tuple]] = None):
        self.db.add_memory(payload=payload, embedding=embedding, triples=triples)

    def add_batch(self, payloads: List[Any], embeddings: List[Any]):
        for p, e in zip(payloads, embeddings):
            self.db.add_memory(payload=p, embedding=e)
        
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

    def add_batch(self, payloads: List[Any], embeddings: List[Any]):
        import numpy as np
        ids = [f"id_{self.id_counter + i}" for i in range(len(payloads))]
        embs = [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]
        docs = [str(p) for p in payloads]
        self.collection.add(embeddings=embs, documents=docs, ids=ids)
        self.id_counter += len(payloads)
        
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

class LanceDBStoreAdapter(VectorStoreAdapter):
    def __init__(self, uri="./.lancedb_data", table_name="benchmark"):
        import lancedb
        self.uri = uri
        self.table_name = table_name
        self.db = lancedb.connect(self.uri)
        self.table = None

    def add(self, payload: Any, embedding: Any, triples: Optional[List[tuple]] = None):
        data = [{"vector": embedding.tolist(), "payload": str(payload)}]
        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")
        else:
            self.table.add(data)

    def add_batch(self, payloads: List[Any], embeddings: List[Any]):
        data = []
        for p, e in zip(payloads, embeddings):
            data.append({"vector": e.tolist(), "payload": str(p)})
        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")
        else:
            self.table.add(data)
            
    def query(self, query_emb: Any, top_k: int = 5, **kwargs) -> List[Any]:
        if self.table is None: return []
        results = self.table.search(query_emb.tolist()).limit(top_k).to_pandas()
        
        class MockResult:
            def __init__(self, payload):
                self.payload = payload
        
        return [MockResult(row["payload"]) for _, row in results.iterrows()]
        
    def clear(self):
        import shutil
        import os
        if os.path.exists(self.uri):
            shutil.rmtree(self.uri)
        import lancedb
        self.db = lancedb.connect(self.uri)
        self.table = None
        
    def checkpoint(self):
        pass

class FAISSStoreAdapter(VectorStoreAdapter):
    def __init__(self, dim=384):
        import faiss
        import numpy as np
        self.dim = dim
        self.index = faiss.IndexFlatL2(self.dim)
        self.payloads = []

    def add(self, payload: Any, embedding: Any, triples: Optional[List[tuple]] = None):
        import numpy as np
        emb = np.array([embedding]).astype('float32')
        self.index.add(emb)
        self.payloads.append(payload)

    def add_batch(self, payloads: List[Any], embeddings: List[Any]):
        import numpy as np
        embs = np.array(embeddings).astype('float32')
        self.index.add(embs)
        self.payloads.extend(payloads)
        
    def query(self, query_emb: Any, top_k: int = 5, **kwargs) -> List[Any]:
        import numpy as np
        q_emb = np.array([query_emb]).astype('float32')
        D, I = self.index.search(q_emb, top_k)
        
        class MockResult:
            def __init__(self, payload):
                self.payload = payload
        
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.payloads):
                results.append(MockResult(self.payloads[idx]))
        return results
        
    def clear(self):
        import faiss
        self.index = faiss.IndexFlatL2(self.dim)
        self.payloads = []
        
    def checkpoint(self):
        pass

class QdrantStoreAdapter(VectorStoreAdapter):
    def __init__(self, collection_name="benchmark"):
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:") # Reliable local-in-memory
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        self.id_counter = 0

    def add(self, payload: Any, embedding: Any, triples: Optional[List[tuple]] = None):
        from qdrant_client.http.models import PointStruct
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=self.id_counter,
                    vector=embedding.tolist(),
                    payload={"text": str(payload)}
                )
            ]
        )
        self.id_counter += 1

    def add_batch(self, payloads: List[Any], embeddings: List[Any]):
        from qdrant_client.http.models import PointStruct
        points = [
            PointStruct(
                id=self.id_counter + i,
                vector=embeddings[i].tolist(),
                payload={"text": str(payloads[i])}
            ) for i in range(len(payloads))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        self.id_counter += len(payloads)
        
    def query(self, query_emb: Any, top_k: int = 5, **kwargs) -> List[Any]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb.tolist(),
            limit=top_k
        ).points
        
        class MockResult:
            def __init__(self, payload):
                self.payload = payload
        
        return [MockResult(r.payload["text"]) for r in results]
        
    def clear(self):
        from qdrant_client.http.models import Distance, VectorParams
        self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        self.id_counter = 0
        
    def checkpoint(self):
        pass
