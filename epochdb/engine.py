import os
import time
import uuid
import json
import logging
import threading
from typing import List, Optional, Dict, Any

import numpy as np

from .atom import UnifiedMemoryAtom
from .hot_tier import HotTier
from .cold_tier import ColdTier
from .transaction import WriteAheadLog, FileLock, MultiIndexTransaction
from .retrieval import RetrievalManager

logger = logging.getLogger(__name__)

# Save the Global Entity Index to disk every N dirty writes.
_KG_FLUSH_INTERVAL = 50


class EpochDB:
    """
    The main client for the EpochDB Agentic Memory Engine.

    Usage (recommended):
        with EpochDB(storage_dir="./memory", dim=384) as db:
            db.add_memory("User said hello", embedding)
            results = db.recall(query_emb)

    Convenience (auto-embedding):
        with EpochDB(storage_dir="./memory", model="all-MiniLM-L6-v2") as db:
            db.remember("User said hello")
            results = db.recall_text("what did the user say?")
    """

    def __init__(
        self,
        storage_dir: str = "./.epochdb_data",
        dim: int = 384,
        epoch_duration_secs: int = 3600,
        saliency_threshold: float = 0.1,
        hot_tier_capacity: int = 10_000,
        model: Optional[str] = None,
    ):
        self.storage_dir = os.path.abspath(storage_dir)
        self.dim = dim
        self.epoch_duration_secs = epoch_duration_secs
        self.saliency_threshold = saliency_threshold
        self._model_name = model
        self._embedder = None  # Lazy-loaded on first use.

        os.makedirs(self.storage_dir, exist_ok=True)

        # --- Dimensionality Enforcement ---
        self.metadata_file = os.path.join(self.storage_dir, "metadata.json")
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)
                stored_dim = metadata.get("dim")
                if stored_dim and stored_dim != self.dim:
                    raise ValueError(
                        f"Dimensionality mismatch! Stored data is {stored_dim}d, "
                        f"but engine initialized with {self.dim}d. "
                        f"Please clear '{self.storage_dir}' or use matching dimensions."
                    )
        else:
            with open(self.metadata_file, "w") as f:
                json.dump({"dim": self.dim, "created_at": time.time()}, f)

        # --- Concurrency Lock ---
        self.lock = FileLock(os.path.join(self.storage_dir, ".lock"))
        self.lock.acquire()

        # --- WAL ---
        self.wal = WriteAheadLog(os.path.join(self.storage_dir, "wal.jsonl"))

        # --- Global Entity Index ---
        self.global_kg_file = os.path.join(self.storage_dir, "global_kg.json")
        self.global_kg: Dict[str, List[List[str]]] = {}
        if os.path.exists(self.global_kg_file):
            try:
                with open(self.global_kg_file, "r") as f:
                    self.global_kg = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load global KG: {e}")

        self._kg_dirty_count = 0  # Tracks unsaved KG mutations.

        # --- Epoch State ---
        self.current_epoch_id = f"epoch_{uuid.uuid4().hex[:8]}"
        self.epoch_start_time = time.time()

        # --- Tiers ---
        self.hot_tier = HotTier(dim=self.dim, max_elements=hot_tier_capacity)
        self.cold_tier = ColdTier(self.storage_dir)

        # --- Retrieval ---
        self.retriever = RetrievalManager(self.hot_tier, self.cold_tier, self.global_kg)

        # --- WAL Replay (Crash Recovery) ---
        self._replay_wal()

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # Do not suppress exceptions.

    # -------------------------------------------------------------------------
    # Crash Recovery
    # -------------------------------------------------------------------------

    def _replay_wal(self):
        """
        Replay uncommitted ADD records from the WAL after a crash.
        Any atoms that were ADD'd but not COMMIT'd are restored to the Hot Tier.
        """
        pending = self.wal.replay()
        if not pending:
            return

        logger.warning(
            f"WAL replay: found {len(pending)} uncommitted atom(s) — restoring to Hot Tier."
        )
        for atom_dict in pending:
            try:
                atom = UnifiedMemoryAtom.from_dict(atom_dict)
                self.hot_tier._add_atom(atom)
                # Restore KG associations.
                for subj, pred, obj in atom.triples:
                    if subj not in self.global_kg:
                        self.global_kg[subj] = []
                    if obj not in self.global_kg:
                        self.global_kg[obj] = []
                    entry = [atom.id, atom.epoch_id]
                    if entry not in self.global_kg[subj]:
                        self.global_kg[subj].append(entry)
                    if entry not in self.global_kg[obj]:
                        self.global_kg[obj].append(entry)
            except Exception as e:
                logger.error(f"Failed to replay atom from WAL: {e}")

        # Clear WAL now that atoms are safely in the Hot Tier.
        self.wal.clear()
        self._save_global_kg(force=True)
        logger.info("WAL replay complete.")

    # -------------------------------------------------------------------------
    # Core Memory Operations
    # -------------------------------------------------------------------------

    def add_memory(
        self,
        payload: Any,
        embedding: np.ndarray,
        triples: List[tuple] = None,
    ) -> str:
        """Store a new memory atom with its embedding and optional KG triples."""
        if triples is None:
            triples = []

        atom = UnifiedMemoryAtom(
            payload=payload,
            embedding=embedding,
            triples=triples,
            epoch_id=self.current_epoch_id,
        )

        # ACID Multi-Index Transaction.
        with MultiIndexTransaction(self.wal, self.hot_tier) as tx:
            tx.add(atom)

        # Update Global Entity Index.
        for subj, pred, obj in triples:
            if subj not in self.global_kg:
                self.global_kg[subj] = []
            if obj not in self.global_kg:
                self.global_kg[obj] = []
            self.global_kg[subj].append([atom.id, self.current_epoch_id])
            self.global_kg[obj].append([atom.id, self.current_epoch_id])

        self._save_global_kg()
        self._check_epoch_expiry()
        return atom.id

    def recall(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        expand_hops: int = 1,
        query_entities: List[str] = None,
    ) -> List[UnifiedMemoryAtom]:
        """Query memory using a dense embedding vector."""
        results = self.retriever.search(
            query_emb,
            top_k=top_k,
            expand_hops=expand_hops,
            query_entities=query_entities,
        )
        self._check_epoch_expiry()
        return results

    # -------------------------------------------------------------------------
    # Convenience: Auto-Embedding
    # -------------------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-load the SentenceTransformer model on first use."""
        if self._embedder is None:
            if self._model_name is None:
                raise ValueError(
                    "No embedding model configured. Pass model='all-MiniLM-L6-v2' "
                    "(or any SentenceTransformer model name) to EpochDB.__init__."
                )
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for auto-embedding. "
                    "Install it with: pip install epochdb[embeddings]"
                )
            self._embedder = SentenceTransformer(self._model_name)
        return self._embedder

    def remember(self, text: str, triples: List[tuple] = None) -> str:
        """
        Convenience method: embed `text` automatically and store it.
        Requires EpochDB to be initialized with a `model` name.

        Example:
            db = EpochDB(storage_dir="./mem", model="all-MiniLM-L6-v2")
            db.remember("Alice lives in Paris", triples=[("Alice", "lives_in", "Paris")])
        """
        embedder = self._get_embedder()
        emb = embedder.encode(text, normalize_embeddings=True)
        return self.add_memory(text, np.array(emb, dtype=np.float32), triples or [])

    def recall_text(
        self,
        query: str,
        top_k: int = 5,
        expand_hops: int = 1,
        query_entities: List[str] = None,
    ) -> List[UnifiedMemoryAtom]:
        """
        Convenience method: embed `query` automatically and recall memories.
        Requires EpochDB to be initialized with a `model` name.
        """
        embedder = self._get_embedder()
        emb = embedder.encode(query, normalize_embeddings=True)
        return self.recall(
            np.array(emb, dtype=np.float32),
            top_k=top_k,
            expand_hops=expand_hops,
            query_entities=query_entities,
        )

    # -------------------------------------------------------------------------
    # Epoch Lifecycle
    # -------------------------------------------------------------------------

    def _check_epoch_expiry(self):
        """Trigger a checkpoint if the current epoch has exceeded its duration."""
        if time.time() - self.epoch_start_time > self.epoch_duration_secs:
            self._checkpoint()

    def _checkpoint(self):
        """
        Epoch Checkpoint: flush atoms to Parquet asynchronously.

        Safety guarantee: the WAL is only cleared AFTER the Parquet file
        has been successfully written. If the flush fails, the WAL is preserved
        for crash recovery on the next startup.
        """
        logger.info(
            f"Triggering Asynchronous Epoch Checkpoint for {self.current_epoch_id}"
        )

        atoms_to_flush = list(self.hot_tier.atoms.values())
        epoch_to_flush = self.current_epoch_id

        # Start a new Epoch immediately (synchronous).
        self.current_epoch_id = f"epoch_{uuid.uuid4().hex[:8]}"
        self.epoch_start_time = time.time()

        # Clear the Hot Tier immediately so new writes go to the fresh epoch.
        self.hot_tier.clear()

        # Flush to disk asynchronously.
        # WAL is cleared in the thread ONLY after a successful write.
        def flush_task(epoch_id: str, atoms: list, wal: WriteAheadLog):
            try:
                if atoms:
                    self.cold_tier.serialize_epoch(epoch_id, atoms)
                # Flush succeeded — safe to clear the WAL now.
                wal.clear()
                logger.info(
                    f"Async flush complete for {epoch_id}. WAL cleared."
                )
            except Exception as e:
                logger.error(
                    f"Async epoch serialization FAILED for {epoch_id}: {e}. "
                    f"WAL preserved for crash recovery."
                )

        thread = threading.Thread(
            target=flush_task,
            args=(epoch_to_flush, atoms_to_flush, self.wal),
            daemon=True,
        )
        thread.start()

    def force_checkpoint(self):
        """Manually trigger a synchronous checkpoint (useful for testing)."""
        logger.info(
            f"Triggering Synchronous Epoch Checkpoint for {self.current_epoch_id}"
        )

        atoms_to_flush = list(self.hot_tier.atoms.values())
        epoch_to_flush = self.current_epoch_id

        self.current_epoch_id = f"epoch_{uuid.uuid4().hex[:8]}"
        self.epoch_start_time = time.time()

        self.hot_tier.clear()

        if atoms_to_flush:
            self.cold_tier.serialize_epoch(epoch_to_flush, atoms_to_flush)

        # Synchronous: safe to clear WAL immediately after write.
        self.wal.clear()

    # -------------------------------------------------------------------------
    # Persistence Helpers
    # -------------------------------------------------------------------------

    def _save_global_kg(self, force: bool = False):
        """
        Persist the Global Entity Index to disk.
        Batches writes: only flushes every _KG_FLUSH_INTERVAL mutations
        unless `force=True`.
        """
        self._kg_dirty_count += 1
        if force or self._kg_dirty_count >= _KG_FLUSH_INTERVAL:
            with open(self.global_kg_file, "w") as f:
                json.dump(self.global_kg, f)
            self._kg_dirty_count = 0

    def close(self):
        """Flush all pending state and release resources."""
        self._save_global_kg(force=True)
        self.wal.close()
        self.lock.release()

    def __del__(self):
        # Last-resort cleanup only — prefer using the context manager or close().
        try:
            self.lock.release()
        except Exception:
            pass
