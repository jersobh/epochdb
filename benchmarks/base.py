from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from epochdb import EpochDB

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = Any

logger = logging.getLogger(__name__)

class BenchmarkAdapter(ABC):
    name = "BaseBenchmark"

    def __init__(self, db: EpochDB, embedder: SentenceTransformer):
        self.db = db
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
        """Pushes data into EpochDB."""
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Runs test queries against EpochDB and returns metrics."""
        pass
