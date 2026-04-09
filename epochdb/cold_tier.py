import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from typing import List, Dict
from .atom import UnifiedMemoryAtom
import logging

logger = logging.getLogger(__name__)

class ColdTier:
    """
    L2 Historical Archive. Resides on Disk.
    Uses Parquet format.
    """
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def serialize_epoch(self, epoch_id: str, atoms: List[UnifiedMemoryAtom]):
        """Flushes hot partition to Parquet blocks."""
        if not atoms:
            return
            
        file_path = os.path.join(self.storage_dir, f"epoch_{epoch_id}.parquet")
        
        ids = [a.id for a in atoms]
        payloads = [str(a.payload) for a in atoms]
        embeddings_f32 = np.array([a.embedding for a in atoms], dtype=np.float32)
        if len(atoms) > 0 and embeddings_f32.size > 0:
            max_vals = np.abs(embeddings_f32).max(axis=1, keepdims=True)
            max_vals[max_vals == 0] = 1.0
            scaled = (embeddings_f32 / max_vals) * 127.0
            embeddings_i8 = np.clip(np.round(scaled), -128, 127).astype(np.int8)
            embeddings = embeddings_i8.tolist()
            embedding_maxes = max_vals.flatten().tolist()
        else:
            embeddings = []
            embedding_maxes = []
        
        created_ats = [a.created_at for a in atoms]
        access_counts = [a.access_count for a in atoms]
        triples_str = [str(a.triples) for a in atoms]

        table = pa.table({
            "id": ids,
            "payload": payloads,
            "embedding": embeddings,
            "embedding_max": embedding_maxes,
            "triples": triples_str,
            "created_at": created_ats,
            "access_count": access_counts,
            "epoch_id": [epoch_id] * len(atoms)
        })
        
        pq.write_table(table, file_path)
        logger.info(f"Serialized {len(atoms)} atoms to {file_path}")

    def load_epoch(self, epoch_id: str) -> List[UnifiedMemoryAtom]:
        file_path = os.path.join(self.storage_dir, f"epoch_{epoch_id}.parquet")
        if not os.path.exists(file_path):
            return []
            
        table = pq.read_table(file_path)
        rows = table.to_pylist()
        
        atoms = []
        import ast
        for row in rows:
            try:
                triples = ast.literal_eval(row['triples'])
            except:
                triples = []
                
            emb = np.array(row['embedding'], dtype=np.float32)
            if 'embedding_max' in row and row['embedding_max'] is not None:
                emb = (emb / 127.0) * row['embedding_max']

            atom = UnifiedMemoryAtom(
                id=row['id'],
                payload=row['payload'],
                embedding=emb,
                triples=triples,
                created_at=row['created_at'],
                access_count=row['access_count'],
                epoch_id=row['epoch_id']
            )
            atoms.append(atom)
        return atoms

    def get_all_epochs(self) -> List[str]:
        epochs = []
        for f in os.listdir(self.storage_dir):
            if f.startswith("epoch_") and f.endswith(".parquet"):
                epoch_id = f[len("epoch_"):-len(".parquet")]
                epochs.append(epoch_id)
        return epochs
