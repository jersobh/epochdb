import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import numpy as np
import hnswlib
from typing import List, Optional, Dict, Any, Tuple
from collections import OrderedDict
from epochdb.core.atom import UnifiedMemoryAtom, PayloadType, ScalarPayload, SeriesPayload, SeriesPoint, ConstraintPayload
from epochdb.core.units import UnitRegistry
import logging
import time

logger = logging.getLogger(__name__)


class ColdTier:
    """
    L2 Historical Archive. Resides on Disk.
    Uses Parquet format with Zstd compression and INT8 quantization.
    """

    def __init__(
        self,
        storage_dir: str,
        index_cache_size: int = 10,
        compression: str = "ZSTD",
        compression_level: int = 3,
    ):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self._index_cache: OrderedDict[str, hnswlib.Index] = OrderedDict()
        self.index_cache_size = index_cache_size
        self.compression = compression
        self.compression_level = compression_level

    def _get_index(self, epoch_id: str, dim: int) -> Optional[hnswlib.Index]:
        """Fetch index from cache or load from disk."""
        if epoch_id in self._index_cache:
            self._index_cache.move_to_end(epoch_id)
            return self._index_cache[epoch_id]

        index_path = os.path.join(self.storage_dir, f"{epoch_id}.hnsw")
        if not os.path.exists(index_path):
            return None

        try:
            index = hnswlib.Index(space="cosine", dim=dim)
            index.load_index(index_path)
            
            # Maintain LRU size
            if len(self._index_cache) >= self.index_cache_size:
                self._index_cache.popitem(last=False)
            
            self._index_cache[epoch_id] = index
            return index
        except Exception as e:
            logger.error(f"Failed to load HNSW index for {epoch_id}: {e}")
            return None

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
 
        # Metadata: stored as JSON strings
        metadatas_json = [json.dumps(a.metadata) for a in atoms]

        schema = pa.schema(
            [
                ("id", pa.string()),
                ("payload", pa.string()),
                ("embedding", pa.list_(pa.float32())),
                ("triples", pa.string()),
                ("created_at", pa.float64()),
                ("access_count", pa.int64()),
                ("epoch_id", pa.string()),
                ("payload_type", pa.string()),
                ("scalar_value", pa.float64()),
                ("scalar_unit", pa.string()),
                ("scalar_uncertainty_low", pa.float64()),
                ("scalar_uncertainty_high", pa.float64()),
                ("series_points", pa.list_(pa.struct([
                    ("timestamp", pa.float64()), 
                    ("value", pa.float64()), 
                    ("uncertainty_low", pa.float64()),
                    ("uncertainty_high", pa.float64())
                ]))),
                ("series_interpolation", pa.string()),
                ("series_aggregation", pa.string()),
                ("series_window", pa.string()),
                ("constraint_expr", pa.string()),
                ("metadata", pa.string()),
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
                "payload_type": [a.payload_type.value for a in atoms],
                "scalar_value": [a.payload.value if a.payload_type == PayloadType.SCALAR else None for a in atoms],
                "scalar_unit": [a.payload.unit if a.payload_type in (PayloadType.SCALAR, PayloadType.SERIES) else None for a in atoms],
                "scalar_uncertainty_low": [a.payload.uncertainty_low if a.payload_type == PayloadType.SCALAR else None for a in atoms],
                "scalar_uncertainty_high": [a.payload.uncertainty_high if a.payload_type == PayloadType.SCALAR else None for a in atoms],
                "series_points": [
                    [{"timestamp": p.timestamp, "value": p.value, "uncertainty_low": p.uncertainty_low, "uncertainty_high": p.uncertainty_high} for p in a.payload.points]
                    if a.payload_type == PayloadType.SERIES else None for a in atoms
                ],
                "series_interpolation": [a.payload.interpolation if a.payload_type == PayloadType.SERIES else None for a in atoms],
                "series_aggregation": [a.payload.aggregation if a.payload_type == PayloadType.SERIES else None for a in atoms],
                "series_window": [a.payload.window if a.payload_type == PayloadType.SERIES else None for a in atoms],
                "constraint_expr": [json.dumps(a.payload.expression) if a.payload_type == PayloadType.CONSTRAINT else None for a in atoms],
                "metadata": metadatas_json,
            },
            schema=schema,
        )
 
        temp_file_path = file_path + ".tmp"
        try:
            write_kwargs = {"compression": self.compression}
            if self.compression_level is not None:
                write_kwargs["compression_level"] = self.compression_level
            pq.write_table(table, temp_file_path, **write_kwargs)
            os.replace(temp_file_path, file_path)
            logger.info(f"Serialized {len(atoms)} atoms to {file_path}")
        except Exception as e:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
            logger.error(f"Failed to serialize epoch {epoch_id} to {file_path}: {e}")
            raise e
 
        # --- Persistent HNSW Index for this Epoch ---
        embeddings_f32 = np.array([a.embedding for a in atoms], dtype=np.float32)
        self._build_hnsw_index(epoch_id, embeddings_f32)

    def _build_hnsw_index(self, epoch_id: str, embeddings: np.ndarray):
        """Builds and saves an hnswlib index for an epoch's embeddings."""
        if embeddings.size == 0:
            return

        num_elements, dim = embeddings.shape
        index_path = os.path.join(self.storage_dir, f"{epoch_id}.hnsw")
        temp_index_path = index_path + ".tmp"

        try:
            # Use cosine space to match HotTier.
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=num_elements, ef_construction=200, M=16)
            
            # Map labels to row indices (0 to N-1).
            labels = np.arange(num_elements)
            index.add_items(embeddings, labels)
            
            index.save_index(temp_index_path)
            os.replace(temp_index_path, index_path)
            logger.info(f"Saved HNSW index to {index_path}")
        except Exception as e:
            if os.path.exists(temp_index_path):
                try:
                    os.remove(temp_index_path)
                except OSError:
                    pass
            logger.error(f"Failed to build HNSW index for {epoch_id}: {e}")

    def load_epoch(self, epoch_id: str) -> List[UnifiedMemoryAtom]:
        file_path = os.path.join(self.storage_dir, f"{epoch_id}.parquet")
        if not os.path.exists(file_path):
            return []

        try:
            table = pq.read_table(file_path)
            rows = table.to_pylist()
            atoms = []
            for row in rows:
                atoms.append(self._row_to_atom(row))
            return atoms
        except Exception as e:
            logger.error(f"Failed to load epoch {epoch_id} from {file_path}: {e}")
            return []

    def search_epoch(
        self, epoch_id: str, query_emb: np.ndarray, top_k: int = 5
    ) -> List[UnifiedMemoryAtom]:
        """Queries an epoch using HNSW index; falls back to linear scan if unavailable."""
        file_path = os.path.join(self.storage_dir, f"{epoch_id}.parquet")
        index_path = os.path.join(self.storage_dir, f"{epoch_id}.hnsw")

        if not os.path.exists(file_path):
            return []

        # Case 1: HNSW Fast Path
        index = self._get_index(epoch_id, len(query_emb))
        if index:
            try:
                num_elements = index.element_count
                actual_k = min(top_k, num_elements)
                labels, distances = index.knn_query(query_emb, k=actual_k)
                
                # OPTIMIZATION: Use Parquet dataset to load only specific rows
                import pyarrow.dataset as ds
                import pyarrow as pa
                dataset = ds.dataset(file_path, format="parquet")
                indices = pa.array(labels[0].tolist(), type=pa.int64())
                rows_table = dataset.take(indices)
                rows = rows_table.to_pylist()
                
                atoms = []
                for row in rows:
                    atoms.append(self._row_to_atom(row))
                return atoms
            except Exception as e:
                logger.error(f"HNSW query failed for {epoch_id}, falling back to linear: {e}")

        # Case 2: Vectorized Linear Fallback (Fast Matrix Ops)
        try:
            schema = pq.read_schema(file_path)
            columns = ["embedding"]
            if "embedding_max" in schema.names:
                columns.append("embedding_max")
            table = pq.read_table(file_path, columns=columns)
            
            # Matrix-based similarity calculation
            # PyArrow fixed-size list to numpy is fast
            embeddings = np.array(table["embedding"].to_pylist(), dtype=np.float32)
            if "embedding_max" in table.column_names:
                max_vals = np.array(table["embedding_max"].to_pylist(), dtype=np.float32)[:, np.newaxis]
                embeddings = (embeddings / 127.0) * max_vals
            
            # Normalize vectors for cosine similarity
            norm_query = np.linalg.norm(query_emb) + 1e-9
            norm_embs = np.linalg.norm(embeddings, axis=1) + 1e-9
            
            similarities = np.dot(embeddings, query_emb) / (norm_embs * norm_query)
            
            # Get top K indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            rows = table.take(top_indices).to_pylist()
            return [self._row_to_atom(row) for row in rows]
        except Exception as e:
            logger.error(f"Linear fallback search failed for {epoch_id} from {file_path}: {e}")
            return []

    def load_atom_metadata(self, epoch_id: str, atom_ids: List[str]) -> List[UnifiedMemoryAtom]:
        """Efficiently loads specific atoms by ID from an epoch."""
        file_path = os.path.join(self.storage_dir, f"{epoch_id}.parquet")
        if not os.path.exists(file_path):
            return []
        
        try:
            table = pq.read_table(file_path)
            # Using pyarrow compute to find indices by ID
            import pyarrow.compute as pc
            mask = pc.is_in(table["id"], value_set=pa.array(atom_ids))
            filtered_table = table.filter(mask)
            
            return [self._row_to_atom(row) for row in filtered_table.to_pylist()]
        except Exception as e:
            logger.error(f"Failed to load atom metadata for {epoch_id} from {file_path}: {e}")
            return []

    def _row_to_atom(self, row: dict) -> UnifiedMemoryAtom:
        """Helper to convert a parquet row dict to UnifiedMemoryAtom."""
        # Dequantize INT8 embedding
        emb = np.array(row["embedding"], dtype=np.float32)
        if "embedding_max" in row and row["embedding_max"] is not None:
            emb = (emb / 127.0) * row["embedding_max"]

        ptype_str = row.get("payload_type", "text")
        ptype = PayloadType(ptype_str)
        payload = row["payload"]

        # Restore triples
        triples = []
        if "triples" in row and row["triples"] is not None:
            try:
                import json
                triples = json.loads(row["triples"])
            except json.JSONDecodeError:
                pass

        # Restore typed payload from quantitative columns if available
        if ptype == PayloadType.SCALAR:
            payload = ScalarPayload(
                value=row["scalar_value"],
                unit=row["scalar_unit"],
                uncertainty_low=row.get("scalar_uncertainty_low", 0.0),
                uncertainty_high=row.get("scalar_uncertainty_high", 0.0),
                timestamp=row["created_at"]
            )
        elif ptype == PayloadType.SERIES:
            points = []
            for p in (row["series_points"] or []):
                points.append(SeriesPoint(
                    timestamp=p["timestamp"],
                    value=p["value"],
                    uncertainty_low=p.get("uncertainty_low", 0.0),
                    uncertainty_high=p.get("uncertainty_high", 0.0),
                ))
            payload = SeriesPayload(
                points=points,
                unit=row["scalar_unit"], # Shared column for unit
                interpolation=row.get("series_interpolation", "linear"),
                aggregation=row.get("series_aggregation", "mean"),
                window=row.get("series_window", "1h"),
            )
        elif ptype == PayloadType.CONSTRAINT:
            try:
                expr = json.loads(row["constraint_expr"])
            except:
                expr = {}
            payload = ConstraintPayload(expression=expr)
        else:
            try:
                payload = json.loads(row["payload"])
            except (json.JSONDecodeError, TypeError):
                payload = row["payload"]

        # Restore metadata
        metadata = {}
        if "metadata" in row and row["metadata"] is not None:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

        return UnifiedMemoryAtom(
            id=row["id"],
            payload=payload,
            payload_type=ptype,
            embedding=emb,
            triples=triples,
            created_at=row["created_at"],
            access_count=row["access_count"],
            epoch_id=row["epoch_id"],
            metadata=metadata,
        )

    def get_total_atoms(self) -> int:
        """Efficiently counts all atoms in the cold tier using Parquet metadata."""
        count = 0
        if not os.path.exists(self.storage_dir):
            return 0
        for f in os.listdir(self.storage_dir):
            if f.endswith(".parquet"):
                try:
                    # read_metadata only reads the footer, very fast.
                    metadata = pq.read_metadata(os.path.join(self.storage_dir, f))
                    count += metadata.num_rows
                except Exception:
                    continue
        return count

    def get_all_epochs(self) -> List[str]:
        epochs = []
        if not os.path.exists(self.storage_dir):
            return epochs
        for f in os.listdir(self.storage_dir):
            if f.endswith(".parquet"):
                epoch_id = f[: -len(".parquet")]
                epochs.append(epoch_id)
        return epochs

    def compact(self, active_epoch_id: str, kg_manager: Any) -> Optional[str]:
        """
        Consolidates historical Parquet files, removes soft-deleted memories,
        resolves supersession, and rebuilds vector index.
        """
        logger.info("Starting historical archive compaction...")
        
        epochs = self.get_all_epochs()
        # Exclude active epoch and any currently executing compacted epoch
        epochs = [e for e in epochs if e != active_epoch_id and not e.startswith("compacted_")]
        
        if not epochs:
            logger.info("No historical epochs found to compact.")
            return None
            
        all_atoms = []
        for epoch in epochs:
            try:
                all_atoms.extend(self.load_epoch(epoch))
            except Exception as e:
                logger.error(f"Failed to load epoch {epoch} during compaction: {e}")
                
        if not all_atoms:
            logger.info("No atoms found in historical epochs.")
            return None
            
        # Filter out soft-deleted and superseded atoms.
        # Process in reverse order (newest first) to easily identify superseded ones.
        all_atoms.sort(key=lambda a: a.created_at, reverse=True)
        
        pruned_count = 0
        pruned_ids = []
        active_states = {}
        final_atoms = []
        seen_ids = set()
        
        for atom in all_atoms:
            if atom.id in seen_ids:
                pruned_count += 1
                pruned_ids.append(atom.id)
                continue
            seen_ids.add(atom.id)

            if atom.metadata.get("_deleted"):
                pruned_count += 1
                pruned_ids.append(atom.id)
                continue
                
            is_superseded = False
            for s, p, o in atom.triples:
                state_key = (str(s).lower(), str(p).lower())
                if state_key in active_states:
                    active_atom = active_states[state_key]
                    if atom.payload_type == PayloadType.SCALAR and active_atom.payload_type == PayloadType.SCALAR:
                        if atom.payload.value != active_atom.payload.value:
                            is_superseded = True
                    elif atom.payload_type == PayloadType.CONSTRAINT and active_atom.payload_type == PayloadType.CONSTRAINT:
                        if atom.payload.expression == active_atom.payload.expression:
                            is_superseded = True
                    elif atom.payload_type == PayloadType.TEXT:
                        is_superseded = True
                else:
                    active_states[state_key] = atom
                    
            if is_superseded:
                pruned_count += 1
                pruned_ids.append(atom.id)
            else:
                final_atoms.append(atom)
                
        # Restore chronological order
        final_atoms.reverse()
        
        logger.info(f"Compaction: loaded {len(all_atoms)} atoms, pruned {pruned_count} (deleted/superseded), keeping {len(final_atoms)}.")
        
        # Update database associations
        if pruned_ids:
            kg_manager.remove_associations(pruned_ids)
            
        if not final_atoms:
            # Clean up all old files if nothing is left
            self._cleanup_epochs(epochs)
            return None
            
        # Write consolidated epoch
        compacted_epoch_id = f"compacted_{int(time.time())}"
        self.serialize_epoch(compacted_epoch_id, final_atoms)
        
        # Update the epoch_id for the consolidated atoms in the global kg database
        kg_manager.update_epoch_ids(epochs, compacted_epoch_id)
        
        # Clean up old files
        self._cleanup_epochs(epochs)
        
        # Clear local cache
        self._index_cache.clear()
        
        logger.info(f"Compaction completed. Consolidated into {compacted_epoch_id}.")
        return compacted_epoch_id
        
    def _cleanup_epochs(self, epochs: List[str]):
        """Delete Parquet and HNSW files for given epochs."""
        for epoch in epochs:
            for ext in (".parquet", ".hnsw"):
                p = os.path.join(self.storage_dir, f"{epoch}{ext}")
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError as e:
                        logger.error(f"Failed to delete {p} during compaction cleanup: {e}")

class ColdTierAnalytics:
    """Analytical engine over cold tier using PyArrow and UnitRegistry."""
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.unit_registry = UnitRegistry()

    def _get_dataset_table(self, field: str, entity: str) -> Optional[pa.Table]:
        """Scans parquet files and filters by triples (LIKE-ish match)."""
        if not os.path.exists(self.storage_dir):
            return None
        
        try:
            # Important: Filter for .parquet files specifically, otherwise ds.dataset 
            # tries to read metadata.json and other control files.
            parquet_files = [
                os.path.join(self.storage_dir, f) 
                for f in os.listdir(self.storage_dir) 
                if f.endswith(".parquet")
            ]
            if not parquet_files:
                return None
                
            dataset = ds.dataset(parquet_files, format="parquet")
            # We filter for records that have a scalar value and match the entity/field in triples
            scanner = dataset.scanner(columns=[
                "id", "scalar_value", "scalar_unit", "triples", "created_at"
            ])
            table = scanner.to_table()
            
            import pyarrow.compute as pc
            # Basic substring filtering for triples JSON
            mask = pc.and_(
                pc.is_valid(table["scalar_value"]),
                pc.and_(
                    pc.match_substring(table["triples"], entity),
                    pc.match_substring(table["triples"], field)
                )
            )
            return table.filter(mask)
        except Exception as e:
            logger.error(f"Failed to scan cold tier for analytics: {e}")
            return None

    def query_trend(self, field: str, entity: str) -> Dict[str, Any]:
        """Detect trends (slope, min, max, std) for a scalar field with unit normalization."""
        table = self._get_dataset_table(field, entity)
        if table is None or len(table) == 0:
            return {}

        # 1. Normalize Units
        # We pick the first record's unit as the base for this trend, or use a schema if we had one.
        # For simplicity, we'll normalize everything to the unit of the first record found.
        base_unit = table["scalar_unit"][0].as_py()
        
        raw_vals = table["scalar_value"].to_numpy()
        raw_units = table["scalar_unit"].to_pylist()
        times = table["created_at"].to_numpy()
        
        normalized_vals = []
        for v, u in zip(raw_vals, raw_units):
            if u == base_unit:
                normalized_vals.append(v)
            else:
                normalized_vals.append(self.unit_registry.convert(v, u, base_unit))
        
        vals = np.array(normalized_vals)
        
        # 2. Compute Stats
        mean_val = np.mean(vals)
        min_val = np.min(vals)
        max_val = np.max(vals)
        std_val = np.std(vals)
        
        # regr_slope (y = vals, x = times)
        if len(vals) > 1:
            try:
                slope, _ = np.polyfit(times, vals, 1)
            except:
                slope = 0.0
        else:
            slope = 0.0
            
        return {
            "mean": float(mean_val),
            "min": float(min_val),
            "max": float(max_val),
            "slope": float(slope),
            "std": float(std_val),
            "unit": base_unit
        }

    def detect_anomalies(self, field: str, entity: str, sigma: float = 3.0) -> List[Dict[str, Any]]:
        """Detect anomalies in a scalar field using normalized standard deviation."""
        trend = self.query_trend(field, entity)
        if not trend: return []
        
        avg = trend["mean"]
        std = trend["std"]
        base_unit = trend["unit"]
        
        threshold_low = avg - sigma * std
        threshold_high = avg + sigma * std
        
        # Scan again to find specific anomalies (or we could have kept the table)
        table = self._get_dataset_table(field, entity)
        if not table: return []
        
        anomalies = []
        for row in table.to_pylist():
            val = row["scalar_value"]
            unit = row["scalar_unit"]
            if unit != base_unit:
                val = self.unit_registry.convert(val, unit, base_unit)
            
            if val < threshold_low or val > threshold_high:
                anomalies.append({
                    "id": row["id"],
                    "value": val,
                    "timestamp": row["created_at"],
                    "unit": base_unit
                })
        return anomalies
