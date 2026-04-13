import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import hnswlib
from typing import List, Optional
from .atom import UnifiedMemoryAtom
import logging

logger = logging.getLogger(__name__)


class ColdTier:
    """
    L2 Historical Archive. Resides on Disk.
    Uses Parquet format with Zstd compression and INT8 quantization.
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def serialize_epoch(self, epoch_id: str, atoms: List[UnifiedMemoryAtom]):
        """Flushes hot partition to Parquet blocks."""
        if not atoms:
            return

        file_path = os.path.join(self.storage_dir, f"{epoch_id}.parquet")

        ids = [a.id for a in atoms]

        # Payloads: stored as JSON strings for safe round-trip of dicts/lists/strings.
        payloads = []
        for a in atoms:
            try:
                payloads.append(json.dumps(a.payload))
            except (TypeError, ValueError):
                payloads.append(json.dumps(str(a.payload)))

        # Full F32 precision for embeddings to eliminate quantization noise
        embeddings = [a.embedding.tolist() for a in atoms]
        created_ats = [a.created_at for a in atoms]
        access_counts = [a.access_count for a in atoms]
 
        # Triples: stored as JSON arrays (list-of-lists) for safe round-trip.
        # This handles entity strings with quotes, backslashes, or unicode correctly.
        triples_json = []
        for a in atoms:
            try:
                triples_json.append(json.dumps([list(t) for t in a.triples]))
            except (TypeError, ValueError):
                triples_json.append("[]")
 
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("payload", pa.string()),
                ("embedding", pa.list_(pa.float32())),
                ("triples", pa.string()),
                ("created_at", pa.float64()),
                ("access_count", pa.int64()),
                ("epoch_id", pa.string()),
            ]
        )
 
        table = pa.table(
            {
                "id": ids,
                "payload": payloads,
                "embedding": embeddings,
                "triples": triples_json,
                "created_at": created_ats,
                "access_count": access_counts,
                "epoch_id": [epoch_id] * len(atoms),
            },
            schema=schema,
        )
 
        pq.write_table(table, file_path, compression="ZSTD")
        logger.info(f"Serialized {len(atoms)} atoms to {file_path}")
 
        # --- Persistent HNSW Index for this Epoch ---
        embeddings_f32 = np.array([a.embedding for a in atoms], dtype=np.float32)
        self._build_hnsw_index(epoch_id, embeddings_f32)

    def _build_hnsw_index(self, epoch_id: str, embeddings: np.ndarray):
        """Builds and saves an hnswlib index for an epoch's embeddings."""
        if embeddings.size == 0:
            return

        num_elements, dim = embeddings.shape
        index_path = os.path.join(self.storage_dir, f"{epoch_id}.hnsw")

        try:
            # Use cosine space to match HotTier.
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=num_elements, ef_construction=200, M=16)
            
            # Map labels to row indices (0 to N-1).
            labels = np.arange(num_elements)
            index.add_items(embeddings, labels)
            
            index.save_index(index_path)
            logger.info(f"Saved HNSW index to {index_path}")
        except Exception as e:
            logger.error(f"Failed to build HNSW index for {epoch_id}: {e}")

    def load_epoch(self, epoch_id: str) -> List[UnifiedMemoryAtom]:
        file_path = os.path.join(self.storage_dir, f"{epoch_id}.parquet")
        if not os.path.exists(file_path):
            return []

        table = pq.read_table(file_path)
        rows = table.to_pylist()

        atoms = []
        for row in rows:
            atoms.append(self._row_to_atom(row))
        return atoms

    def search_epoch(
        self, epoch_id: str, query_emb: np.ndarray, top_k: int = 5
    ) -> List[UnifiedMemoryAtom]:
        """Queries an epoch using HNSW index; falls back to linear scan if unavailable."""
        file_path = os.path.join(self.storage_dir, f"{epoch_id}.parquet")
        index_path = os.path.join(self.storage_dir, f"{epoch_id}.hnsw")

        if not os.path.exists(file_path):
            return []

        # Case 1: HNSW Fast Path
        if os.path.exists(index_path):
            try:
                # Load index (on-demand to save static RAM)
                index = hnswlib.Index(space="cosine", dim=len(query_emb))
                index.load_index(index_path)
                
                num_elements = index.element_count
                actual_k = min(top_k, num_elements)
                labels, distances = index.knn_query(query_emb, k=actual_k)
                
                # Load only the specific rows from Parquet
                table = pq.read_table(file_path)
                indices = pa.array(labels[0].tolist(), type=pa.int64())
                rows_table = table.take(indices)
                rows = rows_table.to_pylist()
                
                atoms = []
                for i, row in enumerate(rows):
                    atoms.append(self._row_to_atom(row))
                return atoms
            except Exception as e:
                logger.error(f"HNSW query failed for {epoch_id}, falling back to linear: {e}")

        # Case 2: Linear Fallback (Legacy or Failed)
        atoms = self.load_epoch(epoch_id)
        if not atoms:
            return []

        scored = []
        for atom in atoms:
            sim = 1.0 - np.dot(query_emb, atom.embedding) / (
                np.linalg.norm(query_emb) * np.linalg.norm(atom.embedding) + 1e-9
            )
            scored.append((sim, atom))
        
        scored.sort(key=lambda x: x[0], reverse=False)
        return [a for _, a in scored[:top_k]]

    def load_atom_metadata(self, epoch_id: str, atom_ids: List[str]) -> List[UnifiedMemoryAtom]:
        """Efficiently loads specific atoms by ID from an epoch."""
        file_path = os.path.join(self.storage_dir, f"{epoch_id}.parquet")
        if not os.path.exists(file_path):
            return []
        
        table = pq.read_table(file_path)
        # Using pyarrow compute to find indices by ID
        import pyarrow.compute as pc
        mask = pc.is_in(table["id"], value_set=pa.array(atom_ids))
        filtered_table = table.filter(mask)
        
        return [self._row_to_atom(row) for row in filtered_table.to_pylist()]

    def _row_to_atom(self, row: dict) -> UnifiedMemoryAtom:
        """Helper to convert a parquet row dict to UnifiedMemoryAtom."""
        try:
            raw = json.loads(row["triples"])
            triples = [tuple(t) for t in raw]
        except (json.JSONDecodeError, TypeError, ValueError):
            triples = []

        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            payload = row["payload"]

        # Dequantize INT8 embedding
        emb = np.array(row["embedding"], dtype=np.float32)
        if "embedding_max" in row and row["embedding_max"] is not None:
            emb = (emb / 127.0) * row["embedding_max"]

        return UnifiedMemoryAtom(
            id=row["id"],
            payload=payload,
            embedding=emb,
            triples=triples,
            created_at=row["created_at"],
            access_count=row["access_count"],
            epoch_id=row["epoch_id"],
        )

    def get_all_epochs(self) -> List[str]:
        epochs = []
        for f in os.listdir(self.storage_dir):
            if f.endswith(".parquet"):
                epoch_id = f[: -len(".parquet")]
                epochs.append(epoch_id)
        return epochs
