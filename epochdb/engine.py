import os
import time
import json
import logging
from typing import List, Optional, Dict
import numpy as np

from .atom import UnifiedMemoryAtom
from .hot_tier import HotTier
from .cold_tier import ColdTier
from .transaction import WriteAheadLog, FileLock, MultiIndexTransaction
from .retrieval import RetrievalManager

logger = logging.getLogger(__name__)

class EpochDB:
    """The main client for the Agentic Memory Engine."""
    
    def __init__(self, storage_dir: str = "./.epochdb_data", dim: int = 384, epoch_duration_secs: int = 3600, saliency_threshold: float = 0.1):
        self.storage_dir = os.path.abspath(storage_dir)
        self.dim = dim
        self.epoch_duration_secs = epoch_duration_secs
        self.saliency_threshold = saliency_threshold
        
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Metadata Tracking (Dimensionality Enforcement)
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
        
        # Concurrency Lock
        self.lock = FileLock(os.path.join(self.storage_dir, ".lock"))
        self.lock.acquire()
        
        # WAL
        self.wal = WriteAheadLog(os.path.join(self.storage_dir, "wal.jsonl"))

        # Global Entity Index
        self.global_kg_file = os.path.join(self.storage_dir, "global_kg.json")
        self.global_kg: Dict[str, List[List[str]]] = {}
        if os.path.exists(self.global_kg_file):
            try:
                with open(self.global_kg_file, "r") as f:
                    self.global_kg = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load global KG: {e}")
        
        # Epoch State
        self.current_epoch_id = f"epoch_{int(time.time())}"
        self.epoch_start_time = time.time()
        
        # Tiers
        self.hot_tier = HotTier(dim=self.dim)
        self.cold_tier = ColdTier(self.storage_dir)
        
        # Retrieval
        # We pass self.global_kg explicitly to the retriever now
        self.retriever = RetrievalManager(self.hot_tier, self.cold_tier, self.global_kg)

    def _save_global_kg(self):
        with open(self.global_kg_file, "w") as f:
            json.dump(self.global_kg, f)

    def add_memory(self, payload: any, embedding: np.ndarray, triples: List[tuple] = None) -> str:
        """Agent adds a new memory atom."""
        if triples is None:
            triples = []
            
        atom = UnifiedMemoryAtom(
            payload=payload,
            embedding=embedding,
            triples=triples,
            epoch_id=self.current_epoch_id
        )
        
        # ACID Multi-Index Transaction
        with MultiIndexTransaction(self.wal, self.hot_tier) as tx:
            tx.add(atom)

        # Update Global Index
        for subj, pred, obj in triples:
            if subj not in self.global_kg: self.global_kg[subj] = []
            if obj not in self.global_kg: self.global_kg[obj] = []
            
            # Subj/Obj -> [atom.id, epoch_id]
            self.global_kg[subj].append([atom.id, self.current_epoch_id])
            self.global_kg[obj].append([atom.id, self.current_epoch_id])

        self._save_global_kg()
            
        self._check_epoch_expiry()
        return atom.id

    def recall(self, query_emb: np.ndarray, top_k: int = 5, expand_hops: int = 1) -> List[UnifiedMemoryAtom]:
        """Agent queries memory."""
        results = self.retriever.search(query_emb, top_k=top_k, expand_hops=expand_hops)
        self._check_epoch_expiry()
        return results

    def _check_epoch_expiry(self):
        """Lifecycle Management: Hot -> Cold"""
        if time.time() - self.epoch_start_time > self.epoch_duration_secs:
            self._checkpoint()

    def _checkpoint(self):
        """Epoch Checkpoint: Flush to disk, clear memory."""
        logger.info(f"Triggering Epoch Checkpoint for {self.current_epoch_id}")
        
        # Gather atoms
        atoms = list(self.hot_tier.atoms.values())
        
        if atoms:
            self.cold_tier.serialize_epoch(self.current_epoch_id, atoms)
        
        # Clear Hot Tier & WAL
        self.hot_tier.clear()
        self.wal.clear()
        
        # Start new Epoch
        self.current_epoch_id = f"epoch_{int(time.time())}"
        self.epoch_start_time = time.time()

    def force_checkpoint(self):
        """Manually trigger checkpoint for testing."""
        self._checkpoint()

    def close(self):
        self._save_global_kg()
        self.wal.close()
        self.lock.release()

    def __del__(self):
        try:
            self.lock.release()
        except:
            pass
