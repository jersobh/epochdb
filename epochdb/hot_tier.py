import hnswlib
import numpy as np
import logging
from typing import Dict, List, Optional
from .atom import UnifiedMemoryAtom

logger = logging.getLogger(__name__)

class HotTier:
    """
    L1 Working Memory. Resides in RAM.
    Houses Active Partition, vector index. (Global KG is now in engine).
    """
    def __init__(self, dim: int, max_elements: int = 10000):
        self.dim = dim
        self.max_elements = max_elements
        
        # HNSW Index for Vectors
        self.vector_index = hnswlib.Index(space='cosine', dim=self.dim)
        self.vector_index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        
        # Atom storage: id -> UnifiedMemoryAtom
        self.atoms: Dict[str, UnifiedMemoryAtom] = {}

        # map string uuid to integer for hnsw
        self.uuid_to_int = {}
        self.int_to_uuid = {}
        self._next_int_id = 0

    def _add_atom(self, atom: UnifiedMemoryAtom):
        """Internal method called by MIT transaction."""
        if atom.id in self.atoms:
            return  # Already exists

        int_id = self._next_int_id
        self.uuid_to_int[atom.id] = int_id
        self.int_to_uuid[int_id] = atom.id
        self._next_int_id += 1
        
        # 1. Add to Vector space
        if atom.embedding is not None and len(atom.embedding) == self.dim:
            self.vector_index.add_items([atom.embedding], [int_id])
        else:
            logger.warning(f"Atom {atom.id} has no valid embedding for dim {self.dim}.")
        
        # 3. Store payload
        self.atoms[atom.id] = atom

    def query_vector(self, query_emb: np.ndarray, top_k: int = 5) -> List[UnifiedMemoryAtom]:
        if len(self.atoms) == 0:
            return []
        
        actual_k = min(top_k, len(self.atoms))
        # knn_query returns (labels, distances)
        labels, distances = self.vector_index.knn_query([query_emb], k=actual_k)
        
        results = []
        for int_lbl in labels[0]:
            if int_lbl in self.int_to_uuid:
                uuid_str = self.int_to_uuid[int_lbl]
                results.append(self.atoms[uuid_str])
                
        return results
        
    def clear(self):
        """Called upon Epoch Expiry after serialization."""
        self.atoms.clear()
        self.uuid_to_int.clear()
        self.int_to_uuid.clear()
        self._next_int_id = 0
        self.vector_index = hnswlib.Index(space='cosine', dim=self.dim)
        self.vector_index.init_index(max_elements=self.max_elements, ef_construction=200, M=16)
