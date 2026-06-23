import asyncio
import inspect
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

try:
    from pydantic import ConfigDict
    _HAS_CONFIG_DICT = True
except ImportError:
    _HAS_CONFIG_DICT = False

try:
    from langchain_core.vectorstores import VectorStore
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
except ImportError:
    raise ImportError(
        "langchain-core is required to use epochdb vector store and retriever. "
        "Please install it via `pip install langchain-core` or `pip install epochdb[langgraph]`."
    )



class EpochDBVectorStore(VectorStore):
    """
    EpochDB VectorStore wrapper for LangChain.
    """

    def __init__(self, db: Any, **kwargs: Any):
        """
        Initialize the vector store.
        
        Args:
            db: An instance of EpochDB or AsyncEpochDB.
        """
        self.db = db
        self.is_async = hasattr(db, "_get_db_sync") or (
            hasattr(db, "remember") and inspect.iscoroutinefunction(db.remember)
        )

    @property
    def embeddings(self) -> Optional[Any]:
        # Return the underlying embedder if available
        if self.is_async:
            sync_db = self.db._get_db_sync()
            return sync_db._get_embedder() if sync_db._model_name else None
        return self.db._get_embedder() if self.db._model_name else None

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run texts through database and add to the vector store."""
        items = []
        metadatas_list = list(metadatas) if metadatas else [{}] * len(list(texts))
        for text, meta in zip(texts, metadatas_list):
            items.append({"text": text, "metadata": meta})

        if self.is_async:
            return self.db._get_db_sync().remember_batch(items)
        return self.db.remember_batch(items)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run texts through database and add to the vector store asynchronously."""
        items = []
        metadatas_list = list(metadatas) if metadatas else [{}] * len(list(texts))
        for text, meta in zip(texts, metadatas_list):
            items.append({"text": text, "metadata": meta})

        if self.is_async:
            return await self.db.remember_batch(items)
        return await asyncio.to_thread(self.db.remember_batch, items)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        if self.is_async:
            memories = self.db._get_db_sync().query(query, k=k, **kwargs)
        else:
            memories = self.db.query(query, k=k, **kwargs)

        return [
            Document(
                page_content=m.text,
                metadata={**m.metadata, "id": m.id, "created_at": m.created_at}
            )
            for m in memories
        ]

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query asynchronously."""
        if self.is_async:
            memories = await self.db.query(query, k=k, **kwargs)
        else:
            memories = await asyncio.to_thread(self.db.query, query, k=k, **kwargs)

        return [
            Document(
                page_content=m.text,
                metadata={**m.metadata, "id": m.id, "created_at": m.created_at}
            )
            for m in memories
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query along with their similarity scores."""
        sync_db = self.db._get_db_sync() if self.is_async else self.db
        
        # Perform query
        if self.is_async:
            memories = self.db._get_db_sync().query(query, k=k, **kwargs)
        else:
            memories = self.db.query(query, k=k, **kwargs)

        # Re-calculate similarity score if possible
        results = []
        if sync_db._model_name:
            embedder = sync_db._get_embedder()
            query_emb = np.array(embedder.encode(query, normalize_embeddings=True), dtype=np.float32)
            
            for m in memories:
                atom = getattr(m, "_atom", None)
                score = 0.0
                if atom and query_emb.any() and atom.embedding.any():
                    score = float(
                        np.dot(atom.embedding, query_emb) / (
                            np.linalg.norm(atom.embedding) * np.linalg.norm(query_emb) + 1e-10
                        )
                    )
                doc = Document(
                    page_content=m.text,
                    metadata={**m.metadata, "id": m.id, "created_at": m.created_at}
                )
                results.append((doc, score))
        else:
            for m in memories:
                doc = Document(
                    page_content=m.text,
                    metadata={**m.metadata, "id": m.id, "created_at": m.created_at}
                )
                results.append((doc, 0.0))
                
        return results

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query along with similarity scores asynchronously."""
        # Simple fallback: run similarity_search_with_score in thread
        return await asyncio.to_thread(self.similarity_search_with_score, query, k=k, **kwargs)

    @classmethod
    def from_texts(
        cls: Type["EpochDBVectorStore"],
        texts: List[str],
        embedding: Any = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "EpochDBVectorStore":
        """Return VectorStore initialized from texts."""
        db = kwargs.get("db")
        if db is None:
            raise ValueError("An EpochDB or AsyncEpochDB instance must be passed as 'db' keyword argument.")
        
        store = cls(db)
        store.add_texts(texts, metadatas=metadatas)
        return store


class EpochDBMultiHopRetriever(BaseRetriever):
    """
    EpochDB Multi-Hop Retriever for LangChain.
    """
    db: Any
    k: int = 5
    hops: int = 2

    if _HAS_CONFIG_DICT:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:
        class Config:
            arbitrary_types_allowed = True



    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        is_async = hasattr(self.db, "_get_db_sync") or (
            hasattr(self.db, "multi_hop") and inspect.iscoroutinefunction(self.db.multi_hop)
        )
        if is_async:
            memories = self.db._get_db_sync().multi_hop(query, hops=self.hops, k=self.k)
        else:
            memories = self.db.multi_hop(query, hops=self.hops, k=self.k)

        return [
            Document(
                page_content=m.text,
                metadata={**m.metadata, "id": m.id, "created_at": m.created_at}
            )
            for m in memories
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        is_async = hasattr(self.db, "_get_db_sync") or (
            hasattr(self.db, "multi_hop") and inspect.iscoroutinefunction(self.db.multi_hop)
        )
        if is_async:
            memories = await self.db.multi_hop(query, hops=self.hops, k=self.k)
        else:
            memories = await asyncio.to_thread(self.db.multi_hop, query, hops=self.hops, k=self.k)

        return [
            Document(
                page_content=m.text,
                metadata={**m.metadata, "id": m.id, "created_at": m.created_at}
            )
            for m in memories
        ]
