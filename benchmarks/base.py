from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = Any

logger = logging.getLogger(__name__)

class VectorStoreAdapter(ABC):
    """Abstract interface for different vector database backends."""
    
    @abstractmethod
    def add(self, payload: Any, embedding: Any, triples: Optional[List[tuple]] = None):
        pass

    @abstractmethod
    def query(self, query_emb: Any, top_k: int = 5, **kwargs) -> List[Any]:
        pass
    
    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def checkpoint(self):
        """Optional: handle tier transitions or persistence flushes."""
        pass

class BenchmarkAdapter(ABC):
    name = "BaseBenchmark"

    def __init__(self, store: VectorStoreAdapter, embedder: SentenceTransformer):
        self.store = store
        self.embedder = embedder
        
    @abstractmethod
    def load_dataset(self) -> Any:
        pass
        
    def _extract_heuristic_triples(self, text: str) -> List[tuple]:
        """A mocked triplet extraction for the MVP benchmark."""
        # For a true implementation, integrating an LLM call here is standard.
        return [] 

    @abstractmethod
    def ingest(self):
        """Pushes data into the vector store."""
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Runs test queries against the store and returns metrics."""
        pass
