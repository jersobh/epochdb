import hnswlib
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from epochdb.core.atom import UnifiedMemoryAtom
from epochdb.retrieval.quantitative_index import QuantitativeIndexManager

logger = logging.getLogger(__name__)

# When the index is this fraction full, double its capacity.
_RESIZE_THRESHOLD = 0.90


@dataclass
class StagedMemory:
    """Transient intent held in the staging pool prior to symbolic commit."""

    memory_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    triples: List[Tuple[Any, ...]] = field(default_factory=list)
    status: str = "PENDING"
    created_at: float = field(default_factory=time.time)
    epoch_id: Optional[str] = None
    namespace: Optional[str] = None


class HotTier:
    """
    L1 Working Memory. Resides in RAM.
    Houses the Active Partition and HNSW vector index.
    Auto-resizes the HNSW index as capacity approaches the limit.

    Also maintains a transient Staging Area for two-phase neuro-symbolic
    commits: intents are registered as PENDING without touching HNSW/WAL/KG
    until ``commit_staged_memory`` promotes them.
    """

    def __init__(self, dim: int, max_elements: int = 10_000, storage_dir: Optional[str] = None):
        self.dim = dim
        self.max_elements = max_elements
        self.storage_dir = storage_dir

        # HNSW Index for Vectors
        self.vector_index = hnswlib.Index(space="cosine", dim=self.dim)
        self.vector_index.init_index(
            max_elements=max_elements, ef_construction=200, M=16,
            allow_replace_deleted=True,
        )

        # Atom storage: id → UnifiedMemoryAtom
        self.atoms: Dict[str, UnifiedMemoryAtom] = {}

        # Staging pool: memory_id → StagedMemory (PENDING intents only)
        self.staging: Dict[str, StagedMemory] = {}

        # Map string UUID to integer label for hnswlib (which requires int IDs).
        self.uuid_to_int: Dict[str, int] = {}
        self.int_to_uuid: Dict[int, str] = {}
        self._next_int_id = 0

        # Quantitative Indices
        from epochdb.core.units import UnitRegistry
        self.unit_registry = UnitRegistry()
        self.quant_index = QuantitativeIndexManager(self.unit_registry, storage_dir=self.storage_dir)

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

        # Index quantitative data.
        self.quant_index.index_atom(atom)

    # -------------------------------------------------------------------------
    # Staging Area (two-phase neuro-symbolic commit)
    # -------------------------------------------------------------------------

    def stage_memory(
        self,
        memory_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        triples: Optional[List[Tuple[Any, ...]]] = None,
        epoch_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """
        Register a write intent as PENDING.

        Does **not** write to the WAL, HNSW index, or active Knowledge Graph.
        """
        self.staging[memory_id] = StagedMemory(
            memory_id=memory_id,
            text=text,
            metadata=dict(metadata or {}),
            embedding=embedding,
            triples=list(triples or []),
            status="PENDING",
            epoch_id=epoch_id,
            namespace=namespace,
        )
        return memory_id

    def get_staged_memory(self, memory_id: str) -> Optional[StagedMemory]:
        """Return a staged intent without removing it."""
        return self.staging.get(memory_id)

    def discard_staged_memory(self, memory_id: str) -> bool:
        """Abort a PENDING intent and remove it from the staging pool."""
        return self.staging.pop(memory_id, None) is not None

    def _atom_from_staged(self, staged: StagedMemory) -> UnifiedMemoryAtom:
        emb = staged.embedding
        if emb is None:
            emb = np.zeros(self.dim, dtype=np.float32)
        return UnifiedMemoryAtom(
            id=staged.memory_id,
            payload=staged.text,
            embedding=emb,
            triples=list(staged.triples),
            epoch_id=staged.epoch_id or "",
            created_at=staged.created_at,
            namespace=staged.namespace,
            metadata=dict(staged.metadata),
        )

    def commit_staged_memory(
        self,
        memory_id: str,
        wal=None,
    ) -> Optional[str]:
        """
        Promote a staged memory into the permanent Hot Tier (HNSW + atoms).

        When ``wal`` is provided, the promote is wrapped in a
        ``MultiIndexTransaction`` so WAL ADD/COMMIT and HNSW stay atomic.
        Staged entries are never written to the Knowledge Graph here; the
        caller (engine / facade) owns KG association updates.
        """
        staged = self.staging.pop(memory_id, None)
        if staged is None:
            return None

        atom = self._atom_from_staged(staged)
        staged.status = "APPROVED"

        if wal is not None:
            from epochdb.core.transaction import MultiIndexTransaction
            with MultiIndexTransaction(wal, self) as tx:
                tx.add(atom)
        else:
            self._add_atom(atom)

        return atom.id

    def update_atom(self, atom: UnifiedMemoryAtom):
        """Replace an existing atom and its vector index entry."""
        if atom.id not in self.atoms:
            self._add_atom(atom)
            return

        int_id = self.uuid_to_int.get(atom.id)
        if int_id is not None:
            try:
                self.vector_index.mark_deleted(int_id)
            except RuntimeError:
                # The atom may not have had a valid vector when it was added.
                pass

            if atom.embedding is not None and len(atom.embedding) == self.dim:
                self.vector_index.add_items(
                    [atom.embedding], [int_id], replace_deleted=True
                )

        self.atoms[atom.id] = atom
        self.quant_index.index_atom(atom)

    def remove_atom(self, atom_id: str) -> bool:
        """Remove an atom from hot storage and prevent stale HNSW hits."""
        atom = self.atoms.pop(atom_id, None)
        if atom is None:
            return False

        int_id = self.uuid_to_int.pop(atom_id, None)
        if int_id is not None:
            self.int_to_uuid.pop(int_id, None)
            try:
                self.vector_index.mark_deleted(int_id)
            except RuntimeError:
                pass
        return True

    def query_vector(self, query_emb: np.ndarray, top_k: int = 5) -> List[UnifiedMemoryAtom]:
        current_count = self.vector_index.get_current_count()
        if current_count == 0 or not self.atoms:
            return []

        if len(query_emb) != self.dim:
            raise ValueError(
                f"Query embedding dimension mismatch: got {len(query_emb)}, expected {self.dim}."
            )

        # HNSW's count includes labels marked deleted. Requesting more
        # neighbours than active atoms makes knn_query raise when a hard delete
        # has removed all (or most) labels from the index.
        actual_k = min(top_k, len(self.atoms))
        labels, _ = self.vector_index.knn_query([query_emb], k=actual_k)

        results = []
        for int_lbl in labels[0]:
            if int_lbl in self.int_to_uuid:
                uuid_str = self.int_to_uuid[int_lbl]
                atom = self.atoms.get(uuid_str)
                if atom is not None:
                    results.append(atom)

        return results

    def clear(self):
        """Called upon Epoch Expiry after serialization."""
        self.atoms.clear()
        self.staging.clear()
        self.uuid_to_int.clear()
        self.int_to_uuid.clear()
        self._next_int_id = 0
        # Reset to original capacity — new epoch starts fresh.
        self.vector_index = hnswlib.Index(space="cosine", dim=self.dim)
        self.vector_index.init_index(
            max_elements=self.max_elements, ef_construction=200, M=16,
            allow_replace_deleted=True,
        )
        self.quant_index.clear()
