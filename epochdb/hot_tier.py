import hnswlib
import numpy as np
import logging
from typing import Dict, List
from .atom import UnifiedMemoryAtom

logger = logging.getLogger(__name__)

# When the index is this fraction full, double its capacity.
_RESIZE_THRESHOLD = 0.90


class HotTier:
    """
    L1 Working Memory. Resides in RAM.
    Houses the Active Partition and HNSW vector index.
    Auto-resizes the HNSW index as capacity approaches the limit.
    """

    def __init__(self, dim: int, max_elements: int = 10_000):
        self.dim = dim
        self.max_elements = max_elements

        # HNSW Index for Vectors
        self.vector_index = hnswlib.Index(space="cosine", dim=self.dim)
        self.vector_index.init_index(
            max_elements=max_elements, ef_construction=200, M=16
        )

        # Atom storage: id → UnifiedMemoryAtom
        self.atoms: Dict[str, UnifiedMemoryAtom] = {}

        # Map string UUID to integer label for hnswlib (which requires int IDs).
        self.uuid_to_int: Dict[str, int] = {}
        self.int_to_uuid: Dict[int, str] = {}
        self._next_int_id = 0

    def _maybe_resize(self):
        """Double the index capacity if we're approaching the limit."""
        if self._next_int_id >= int(self.max_elements * _RESIZE_THRESHOLD):
            new_capacity = self.max_elements * 2
            logger.info(
                f"HotTier near capacity ({self._next_int_id}/{self.max_elements}). "
                f"Resizing HNSW index to {new_capacity}."
            )
            self.vector_index.resize_index(new_capacity)
            self.max_elements = new_capacity

    def _add_atom(self, atom: UnifiedMemoryAtom):
        """Internal method called by MultiIndexTransaction."""
        if atom.id in self.atoms:
            return  # Already exists; idempotent.

        self._maybe_resize()

        int_id = self._next_int_id
        self.uuid_to_int[atom.id] = int_id
        self.int_to_uuid[int_id] = atom.id
        self._next_int_id += 1

        # Add to vector space.
        if atom.embedding is not None and len(atom.embedding) == self.dim:
            self.vector_index.add_items([atom.embedding], [int_id])
        else:
            logger.warning(
                f"Atom {atom.id} has no valid embedding for dim {self.dim}."
            )

        # Store atom payload.
        self.atoms[atom.id] = atom

    def query_vector(self, query_emb: np.ndarray, top_k: int = 5) -> List[UnifiedMemoryAtom]:
        if len(self.atoms) == 0:
            return []

        actual_k = min(top_k, len(self.atoms))
        labels, _ = self.vector_index.knn_query([query_emb], k=actual_k)

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
        # Reset to original capacity — new epoch starts fresh.
        self.vector_index = hnswlib.Index(space="cosine", dim=self.dim)
        self.vector_index.init_index(
            max_elements=self.max_elements, ef_construction=200, M=16
        )
