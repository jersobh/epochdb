import os
import time
import uuid
import json
import logging
import threading
from typing import List, Optional, Dict, Any, Set, Union

import numpy as np

from epochdb.core.atom import UnifiedMemoryAtom, PayloadType, ScalarPayload, SeriesPayload, ConstraintPayload, SeriesPoint, MemoryType
from epochdb.storage.hot_tier import HotTier
from epochdb.storage.cold_tier import ColdTier
from epochdb.core.transaction import WriteAheadLog, FileLock, MultiIndexTransaction
from epochdb.retrieval.retrieval import RetrievalManager
from epochdb.entities.kg_manager import KGManager
from epochdb.entities.reflection_quant import QuantitativeReflectionManager
from epochdb.storage.cold_tier import ColdTierAnalytics

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
        model: Optional[str] = "all-MiniLM-L6-v2",
        tenant: Optional[str] = None,
        namespace: Optional[str] = None,
        wal_sync_interval: float = 0.0,
        model_cache_path: Optional[str] = None,
        parquet_compression: str = "ZSTD",
        parquet_compression_level: int = 3,
        wal_use_uring: bool = True,
        auto_extract: bool = False,
        extraction_model: Optional[str] = None,
    ):
        self.tenant = tenant
        self.namespace = namespace
        self.wal_sync_interval = wal_sync_interval
        self.wal_use_uring = wal_use_uring
        self.auto_extract = auto_extract
        self.extraction_model = extraction_model
        self._fact_extractor = None

        # Physical partition for tenant isolation
        if self.tenant:
            storage_dir = os.path.join(storage_dir, "tenants", self.tenant)

        # Namespace subdirectory within the tenant
        if self.namespace:
            storage_dir = os.path.join(storage_dir, "ns", self.namespace)

        self.storage_dir = os.path.abspath(storage_dir)
        self.dim = dim
        self.epoch_duration_secs = epoch_duration_secs
        self.saliency_threshold = saliency_threshold
        self._model_name = model
        self._model_cache_path = model_cache_path
        self._embedder = None  # Lazy-loaded on first use.
        self._internal_lock = threading.RLock() # Protect against concurrent threads in the same process

        os.makedirs(self.storage_dir, exist_ok=True)

        # --- Dimensionality Enforcement ---
        self.metadata_file = os.path.join(self.storage_dir, "metadata.json")
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)
                stored_dim = metadata.get("dim")
                if stored_dim and stored_dim != self.dim:
                    logger.info(
                        f"Dimensionality mismatch! Stored data is {stored_dim}d, "
                        f"but engine initialized with {self.dim}d. Automatically setting to {stored_dim}d."
                    )
                    self.dim = stored_dim
        else:
            with open(self.metadata_file, "w") as f:
                json.dump({"dim": self.dim, "created_at": time.time()}, f)

        # --- Concurrency Lock ---
        self.lock = FileLock(os.path.join(self.storage_dir, ".lock"))
        self.lock.acquire()

        # --- WAL ---
        self.wal = WriteAheadLog(
            os.path.join(self.storage_dir, "wal.jsonl"),
            sync_interval=self.wal_sync_interval,
            use_uring=self.wal_use_uring
        )

        # --- Global Entity Index (Disk-Direct) ---
        self.global_kg_db = os.path.join(self.storage_dir, "global_kg.db")
        self.kg_manager = KGManager(self.global_kg_db)

        self.current_epoch_id = f"epoch_{uuid.uuid4().hex[:8]}"
        self.epoch_start_time = time.time()
        self._last_timestamp = 0.0
        self.predicates: Set[str] = set()  # Track unique predicates for extraction boost

        # --- Tiers ---
        self.hot_tier = HotTier(dim=self.dim, max_elements=hot_tier_capacity, storage_dir=self.storage_dir)
        self.cold_tier = ColdTier(
            self.storage_dir,
            compression=parquet_compression,
            compression_level=parquet_compression_level,
        )

        # --- Retrieval ---
        self.retriever = RetrievalManager(self.hot_tier, self.cold_tier, self.kg_manager)

        # --- Persistent Access Deltas (cold-tier saliency across restarts) ---
        self._access_deltas_file = os.path.join(self.storage_dir, "access_deltas.json")

        if os.path.exists(self._access_deltas_file):
            try:
                with open(self._access_deltas_file, "r") as f:
                    self.retriever._access_deltas = {
                        k: int(v) for k, v in json.load(f).items()
                    }
                logger.info(
                    f"Loaded {len(self.retriever._access_deltas)} access deltas."
                )
            except Exception as e:
                logger.error(f"Failed to load access deltas: {e}")

        # --- WAL Replay (Crash Recovery) ---
        self._replay_wal()

    def get_total_atoms(self) -> int:
        """Returns the total number of atoms across both hot and cold tiers."""
        with self._internal_lock:
            return len(self.hot_tier.atoms) + self.cold_tier.get_total_atoms()

    @property
    def parquet_compression(self) -> str:
        """Returns the current Parquet compression method."""
        return self.cold_tier.compression

    @parquet_compression.setter
    def parquet_compression(self, value: str):
        """Sets a new Parquet compression method."""
        self.cold_tier.compression = value

    @property
    def parquet_compression_level(self) -> Optional[int]:
        """Returns the current Parquet compression level."""
        return self.cold_tier.compression_level

    @parquet_compression_level.setter
    def parquet_compression_level(self, value: Optional[int]):
        """Sets a new Parquet compression level."""
        self.cold_tier.compression_level = value

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
                self.hot_tier.quant_index.index_atom(atom)
                # Restore KG associations.
                associations = []
                for subj, pred, obj in atom.triples:
                    associations.append((subj, atom.id, atom.epoch_id))
                    associations.append((obj, atom.id, atom.epoch_id))
                    self.predicates.add(pred)
                self.kg_manager.add_associations_batch(associations)
            except Exception as e:
                logger.error(f"Failed to replay atom from WAL: {e}")

        # Clear WAL now that atoms are safely in the Hot Tier.
        self.wal.clear()
        self.kg_manager.commit()
        logger.info("WAL replay complete.")

    # -------------------------------------------------------------------------
    # Core Memory Operations
    # -------------------------------------------------------------------------

    def add_memory(
        self,
        payload: Any,
        embedding: np.ndarray,
        triples: List[tuple] = None,
        atom_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a new memory atom with its embedding and optional KG triples."""
        with self._internal_lock:
            if triples is None:
                triples = []

            # Ensure embedding is unit-length for consistent cosine similarity.
            if embedding is None:
                embedding = np.zeros(self.dim, dtype=np.float32)
            
            norm = np.linalg.norm(embedding)
            if norm > 1e-10:
                embedding = embedding / norm

            # Assign strictly monotonic timestamp
            ts = max(time.time(), self._last_timestamp + 0.000001)
            self._last_timestamp = ts

            atom_kwargs = {
                "payload": payload,
                "embedding": embedding,
                "triples": triples,
                "epoch_id": self.current_epoch_id,
                "created_at": ts,
                "namespace": self.namespace,
            }
            if atom_id is not None:
                atom_kwargs["id"] = atom_id
            if metadata is not None:
                atom_kwargs["metadata"] = metadata
            atom = UnifiedMemoryAtom(**atom_kwargs)

            # ACID Multi-Index Transaction.
            with MultiIndexTransaction(self.wal, self.hot_tier) as tx:
                tx.add(atom)

            # Update Global Entity Index.
            associations = []
            for subj, pred, obj in triples:
                associations.append((str(subj), atom.id, self.current_epoch_id))
                associations.append((str(obj), atom.id, self.current_epoch_id))
                self.predicates.add(pred)
            
            self.kg_manager.add_associations_batch(associations)
            self._check_epoch_expiry()
            return atom.id

    def replace_memory(
        self,
        atom_id: str,
        payload: Any,
        embedding: np.ndarray,
        triples: List[tuple],
        metadata: Optional[dict] = None,
    ) -> str:
        """Replace triples (and optional payload/metadata) for an existing hot or cold atom."""
        with self._internal_lock:
            if triples is None:
                triples = []

            norm = np.linalg.norm(embedding)
            if norm > 1e-10:
                embedding = embedding / norm

            self.kg_manager.remove_associations([atom_id])

            atom = self.hot_tier.atoms.get(atom_id)
            if atom is None:
                for epoch_id in self.cold_tier.get_all_epochs():
                    loaded = self.cold_tier.load_atom_metadata(epoch_id, [atom_id])
                    if loaded:
                        atom = loaded[0]
                        break

            if atom is None:
                return self.add_memory(
                    payload,
                    embedding,
                    triples,
                    atom_id=atom_id,
                    metadata=metadata,
                )

            atom.payload = payload
            atom.embedding = embedding
            atom.triples = [tuple(t) for t in triples]
            if metadata is not None:
                atom.metadata = metadata
            atom.epoch_id = self.current_epoch_id

            self.hot_tier._add_atom(atom)
            self.hot_tier.quant_index.index_atom(atom)

            associations = []
            for subj, pred, obj in triples:
                associations.append((str(subj), atom.id, self.current_epoch_id))
                associations.append((str(obj), atom.id, self.current_epoch_id))
                self.predicates.add(pred)
            self.kg_manager.add_associations_batch(associations)

            with MultiIndexTransaction(self.wal, self.hot_tier) as tx:
                tx.add(atom)

            self._check_epoch_expiry()
            return atom.id

    def add_memory_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Bulk-ingest a list of memory atoms in a single call.

        Each item is a dict with keys:
          - ``payload``   (required) – the text or data to store.
          - ``embedding`` (required) – pre-computed numpy array.
          - ``triples``   (optional) – list of (subj, pred, obj) tuples.
          - ``metadata``  (optional) – dict of associated metadata.

        Returns a list of atom IDs in the same order as `items`.

        Example::

            ids = db.add_memory_batch([
                {"payload": "Alice likes Paris",
                 "embedding": emb_a,
                 "triples": [("Alice", "likes", "Paris")]},
                {"payload": "Bob works at CERN",
                 "embedding": emb_b},
            ])
        """
        with self._internal_lock:
            atoms = []
            associations = []
            ids = []
            
            with MultiIndexTransaction(self.wal, self.hot_tier) as tx:
                for item in items:
                    payload = item["payload"]
                    embedding = item["embedding"]
                    triples = item.get("triples", [])
                    metadata = item.get("metadata", {})
                    atom_id = item.get("atom_id")

                    if triples is None:
                        triples = []

                    # Ensure embedding is unit-length for consistent cosine similarity.
                    if embedding is None:
                        embedding = np.zeros(self.dim, dtype=np.float32)
                    
                    norm = np.linalg.norm(embedding)
                    if norm > 1e-10:
                        embedding = embedding / norm

                    # Assign strictly monotonic timestamp
                    ts = max(time.time(), self._last_timestamp + 0.000001)
                    self._last_timestamp = ts

                    atom_kwargs = {
                        "payload": payload,
                        "embedding": embedding,
                        "triples": triples,
                        "epoch_id": self.current_epoch_id,
                        "created_at": ts,
                        "metadata": metadata,
                    }
                    if atom_id is not None:
                        atom_kwargs["id"] = atom_id
                    
                    atom = UnifiedMemoryAtom(**atom_kwargs)
                    tx.add(atom)
                    
                    for subj, pred, obj in triples:
                        associations.append((str(subj), atom.id, self.current_epoch_id))
                        associations.append((str(obj), atom.id, self.current_epoch_id))
                        self.predicates.add(pred)
                    
                    ids.append(atom.id)

            self.kg_manager.add_associations_batch(associations)
            self._check_epoch_expiry()
            return ids

    def recall(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        expand_hops: int = 1,
        query_entities: List[str] = None,
        fork_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> List[UnifiedMemoryAtom]:
        """Query memory using a dense embedding vector."""
        with self._internal_lock:
            results = self.retriever.search(
                query_emb,
                top_k=top_k,
                expand_hops=expand_hops,
                query_entities=query_entities,
                fork_id=fork_id,
            )
            # Filter by memory_type if specified
            if memory_type:
                try:
                    mt = MemoryType(memory_type)
                    results = [a for a in results if a.memory_type == mt]
                except ValueError:
                    pass
            self._check_epoch_expiry()
            return results

    def query_temporal(self, atom_id: str, timestamp: float, fork_id: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Interpolate series value at a specific timestamp with uncertainty propagation."""
        with self._internal_lock:
            return self.retriever.query_temporal(atom_id, timestamp, fork_id)

    def query_aggregate(self, atom_id: str, window: str, t_start: float, t_end: float) -> List[SeriesPoint]:
        """Query pre-materialized aggregated views for a series."""
        with self._internal_lock:
            return self.retriever.query_aggregate(atom_id, window, t_start, t_end)

    def check_feasibility(self, constraint_ids: List[str], state: Dict[str, float], fork_id: Optional[str] = None) -> bool:
        """Check if a set of constraints is feasible against a state."""
        with self._internal_lock:
            return self.retriever.check_feasibility(constraint_ids, state, fork_id)

    def reflect_on_entity(self, entity: str, field: str, confidence_threshold: float = 0.95) -> List[UnifiedMemoryAtom]:
        """Analyze historical data and auto-declare policy constraints."""
        with self._internal_lock:
            analytics = ColdTierAnalytics(self.storage_dir)
            reflector = QuantitativeReflectionManager(analytics, self.hot_tier)
            return reflector.reflect_on_entity(entity, field, confidence_threshold)

    def query_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Run DuckDB SQL analytics directly over Cold Tier Parquet archives."""
        with self._internal_lock:
            analytics = ColdTierAnalytics(self.storage_dir)
            return analytics.query_sql(sql)

    def get_hub_entities(self, limit: int = 50) -> List[str]:
        """Return the most-connected entities for graph visualization."""
        with self._internal_lock:
            return self.kg_manager.get_entities_by_degree(limit=limit)

    def get_entities(self, prefix: str = None) -> List[str]:
        """Fetch all entities or those matching a prefix."""
        with self._internal_lock:
            if prefix:
                return self.kg_manager.get_entities_by_prefix(prefix)
            return self.kg_manager.get_all_entities()

    def claim_entity(self, entity: str, claimer: str, expiry_secs: int = 30) -> bool:
        """Atomically claim an entity for a period of time."""
        with self._internal_lock:
            return self.kg_manager.claim_entity(entity, claimer, expiry_secs)

    def unclaim_entity(self, entity: str, claimer: str) -> bool:
        """Release an atomic claim."""
        with self._internal_lock:
            return self.kg_manager.unclaim_entity(entity, claimer)

    def get_entity_history(self, entity: str) -> List[UnifiedMemoryAtom]:
        """Retrieve all memory atoms associated with an entity, sorted by time."""
        with self._internal_lock:
            lineage = self.kg_manager.get_lineage(entity)
            atoms = []
            for atom_id, epoch_id in lineage:
                # Try hot tier first
                atom = self.hot_tier.atoms.get(atom_id)
                if not atom:
                    # Fallback to cold tier (simplified fetch)
                    atom = self.retriever._fetch_atom_by_id(atom_id, epoch_id)
                if atom:
                    atoms.append(atom)
            return sorted(atoms, key=lambda x: x.created_at)

    def save_state(self, key: str, payload: Any, parent_key: str = None) -> str:
        """
        Specialized memory addition for state management.
        Creates a 'STATE' entity link and optional parent link for lineage.
        Uses dummy embedding (zero vector) as state is usually retrieved via KG.
        """
        with self._internal_lock:
            triples = [(key, "is_state", "True")]
            if parent_key:
                triples.append((key, "parent_state", parent_key))
            
            # Use a small dummy embedding for state atoms
            dummy_emb = np.zeros(self.dim, dtype=np.float32)
            return self.add_memory(payload, dummy_emb, triples)

    def extract_entities(self, text: str) -> List[str]:
        """Heuristically extract entities from text that exist in Global KG."""
        with self._internal_lock:
            found = set()
            text_l = text.lower()
            # Clean possessives and punctuation
            clean_text = text_l.replace("'s", "").replace("?", "").replace(".", "").replace(",", "")
            words = {w.strip() for w in clean_text.split() if len(w) > 2}

            # Expanded blacklist – generic conversational pronouns/determiners (EN + PT)
            blacklist = {"user", "agent", "the", "this", "that", "it", "i", "my", "me",
                         "they", "their", "who", "what", "where", "when", "how",
                         "o", "a", "os", "as", "eu", "meu", "minha", "eles", "elas",
                         "quem", "que", "onde", "quando", "como", "esta", "essa", "aquela"}

            # Get significant words for candidates LIKE query
            sig_words = [w for w in words if w not in blacklist and len(w) > 3]
            candidates = self.kg_manager.get_candidate_entities(sig_words)

            # --- Pass 1: Subjects/Objects must appear in the query ---
            for ent in candidates:
                ent_l = ent.lower()
                ent_parts = [p for p in ent_l.replace("-", " ").split() if len(p) > 3]
                if ent_l in clean_text or any(p in words for p in ent_parts):
                    found.add(ent)
                    continue

                # 2. Acronym Match: if entity name contains "(XYZ)", match "xyz" in query
                if "(" in ent_l and ")" in ent_l:
                    acronym = ent_l[ent_l.find("(")+1:ent_l.find(")")]
                    if len(acronym) >= 2 and acronym in words:
                        found.add(ent)
                        continue
                
                # 3. Technical Phrase Match: for multi-word entities, match if significant significant parts are in query
                if " " in ent_l:
                    ent_parts = [p for p in ent_l.split() if len(p) > 3 and p not in blacklist]
                    if ent_parts:
                        matches = sum(1 for p in ent_parts if p in clean_text)
                        if matches / len(ent_parts) >= 0.5:
                            found.add(ent)
                            continue

            # --- Pass 2: Predicates — substring + cautious prefix fuzzy ---
            for pred in self.predicates:
                pred_l = pred.lower()
                if pred_l in clean_text:
                    found.add(pred)
                    continue
                pred_parts = {p.strip() for p in pred_l.replace("_", " ").split() if len(p) > 3}
                for part in pred_parts:
                    root = part[:4]
                    if any(root in w for w in words):
                        found.add(pred)
                        break

            return [f for f in found if f.lower() not in blacklist]

    def recall_by_entity(self, entity: str) -> List[UnifiedMemoryAtom]:
        """Retrieve all memory atoms associated with a given entity."""
        with self._internal_lock:
            associations = self.kg_manager.get_associations(entity)
            if not associations:
                return []

            epoch_to_atom_ids = {}
            results = []
            hot_ids = set()

            for atom_id, epoch_id in associations:
                if atom_id in self.hot_tier.atoms:
                    results.append(self.hot_tier.atoms[atom_id])
                    hot_ids.add(atom_id)
                else:
                    if epoch_id not in epoch_to_atom_ids:
                        epoch_to_atom_ids[epoch_id] = []
                    epoch_to_atom_ids[epoch_id].append(atom_id)

            for epoch_id, atom_ids in epoch_to_atom_ids.items():
                atoms = self.cold_tier.load_atom_metadata(epoch_id, atom_ids)
                for a in atoms:
                    if a.id not in hot_ids:
                        results.append(a)

            return results

    # -------------------------------------------------------------------------
    # Convenience: Auto-Embedding
    # -------------------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-load the embedding model on first use."""
        if self._embedder is None:
            if self._model_name is None:
                raise ValueError(
                    "No embedding model configured. Pass model='all-MiniLM-L6-v2' "
                    "(or any SentenceTransformer model name) to EpochDB.__init__."
                )
            
            # Normalize common copy-paste Unicode characters (like non-breaking hyphens)
            normalized_model_name = self._model_name.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
            
            if normalized_model_name.startswith("google:"):
                model_id = normalized_model_name.split("google:", 1)[-1]
                self._embedder = GoogleEmbedder(model_id, dim=self.dim)
            elif normalized_model_name.startswith("openai:"):
                model_id = normalized_model_name.split("openai:", 1)[-1]
                self._embedder = OpenAIEmbedder(model_id, dim=self.dim)
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    raise ImportError(
                        "sentence-transformers is required for auto-embedding. "
                        "Install it with: pip install epochdb[embeddings]"
                    )
                # Check for cached local model to avoid downloading on every run
                safe_model_name = normalized_model_name.replace("/", "_")
                if self._model_cache_path:
                    cache_dir = os.path.join(self._model_cache_path, safe_model_name)
                else:
                    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "epochdb_models", safe_model_name)
                if os.path.exists(cache_dir):
                    self._embedder = SentenceTransformer(cache_dir)
                else:
                    self._embedder = SentenceTransformer(normalized_model_name)
                    try:
                        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
                        self._embedder.save(cache_dir)
                    except Exception as e:
                        logger.warning(f"Could not save local model cache to {cache_dir}: {e}")
        return self._embedder

    def remember(
        self,
        text: str,
        triples: Optional[Any] = None,
        metadata: Optional[dict] = None,
        memory_type: Optional[str] = None,
    ) -> str:
        """
        Convenience method: embed `text` automatically and store it.
        Requires EpochDB to be initialized with a `model` name.
        """
        if isinstance(triples, dict) and metadata is None:
            metadata = triples
            triples = None

        with self._internal_lock:
            embedder = self._get_embedder()
            emb = embedder.encode(text, normalize_embeddings=True)
            if triples is None and metadata is not None:
                triples = metadata.get("triples")
            
            if not triples and self.auto_extract:
                from epochdb.core.fact_extractor import FactExtractor
                if self._fact_extractor is None:
                    self._fact_extractor = FactExtractor(self, self.extraction_model)
                triples = self._fact_extractor.extract(text)
            elif not triples:
                triples = []

            atom_id = self.add_memory(
                text,
                np.array(emb, dtype=np.float32),
                triples,
                metadata=metadata,
            )
            # Set memory_type on the atom if specified
            if memory_type:
                try:
                    mt = MemoryType(memory_type)
                    atom = self.hot_tier.atoms.get(atom_id)
                    if atom:
                        atom.memory_type = mt
                except ValueError:
                    pass
            return atom_id

    def remember_batch(self, texts: List[str], triples_list: List[List[tuple]] = None) -> List[str]:
        """
        Store multiple memories at once. Much faster for large datasets.
        """
        with self._internal_lock:
            if not texts:
                return []
            embedder = self._get_embedder()
            embs = embedder.encode_batch(texts)
            
            items = []
            for i, text in enumerate(texts):
                triples = triples_list[i] if triples_list and i < len(triples_list) else []
                items.append({
                    "payload": text,
                    "embedding": embs[i],
                    "triples": triples,
                })
            return self.add_memory_batch(items)

    def forget(self, atom_id: str):
        """
        Permanently remove an atom from the Truth Substrate.
        Logs a DELETE operation to the WAL to ensure persistence.
        """
        with self._internal_lock:
            # 1. Remove from Hot Tier if present
            if atom_id in self.hot_tier.atoms:
                atom = self.hot_tier.atoms.pop(atom_id)
                # 2. Remove from KG associations
                for s, p, o in atom.triples:
                    # Note: Simplified KG removal (doesn't handle Parquet atoms yet)
                    pass
            
            # 3. Log deletion to WAL
            self.wal.append("DELETE", {"id": atom_id})
            logger.info(f"Atom {atom_id} forgotten.")

    def recall_text(
        self,
        query: str,
        top_k: int = 5,
        expand_hops: int = 1,
        query_entities: List[str] = None,
        memory_type: Optional[str] = None,
    ) -> List[UnifiedMemoryAtom]:
        """
        Convenience method: embed `query` automatically and recall memories.
        Automatically extracts known entities from the query string to boost recall.
        """
        # OPTIMIZATION: Extract entities and embed OUTSIDE the lock
        # to allow multiple queries (e.g. Multi-Query Expansion) to run in parallel.
        if query_entities is None:
            query_entities = self.extract_entities(query)
            
        embedder = self._get_embedder()
        emb_list = embedder.encode(query, normalize_embeddings=True)
        emb = np.array(emb_list, dtype=np.float32)

        with self._internal_lock:
            return self.recall(
                emb,
                top_k=top_k,
                expand_hops=expand_hops,
                query_entities=query_entities,
                memory_type=memory_type,
            )

    # -------------------------------------------------------------------------
    # Branching
    # -------------------------------------------------------------------------
    
    def fork(self, parent_epoch_id: str, new_epoch_id: str):
        """
        Creates a logical fork in the memory tree.
        Logs the branching event to the Knowledge Graph to maintain lineage without 
        duplicating massive vector stores.
        """
        with self._internal_lock:
            self.kg_manager.add_associations_batch([
                (parent_epoch_id, "forked_to", new_epoch_id)
            ])
            logger.info(f"Forked epoch {parent_epoch_id} -> {new_epoch_id}")

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
        with self._internal_lock:
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
        with self._internal_lock:
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

    def _save_access_deltas(self):
        """Persist access-count deltas to disk so saliency survives restarts."""
        try:
            with open(self._access_deltas_file, "w") as f:
                json.dump(self.retriever._access_deltas, f)
        except Exception as e:
            logger.error(f"Failed to save access deltas: {e}")

    def flush(self):
        """Synchronously flush current hot tier to cold tier."""
        with self._internal_lock:
            atoms = list(self.hot_tier.atoms.values())
            if atoms:
                logger.info(f"Synchronously flushing {len(atoms)} atoms to cold tier...")
                self.cold_tier.serialize_epoch(self.current_epoch_id, atoms)
                self.wal.clear()
                self.hot_tier.clear()
                self.current_epoch_id = f"epoch_{uuid.uuid4().hex[:8]}"
                self.epoch_start_time = time.time()

    def close(self):
        """Flush all pending state and release resources."""
        self.flush()
        self.kg_manager.close()
        self._save_access_deltas()
        self.wal.close()
        self.lock.release()

    def __del__(self):
        # Last-resort cleanup only — prefer using the context manager or close().
        try:
            self.lock.release()
        except Exception:
            pass


class AsyncEpochDB:
    """
    An asynchronous wrapper for EpochDB that runs blocking operations in threads.
    """
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._db = None

    async def __aenter__(self):
        import asyncio
        self._db = await asyncio.to_thread(EpochDB, **self._kwargs)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        import asyncio
        if self._db:
            await asyncio.to_thread(self._db.close)

    async def query_sql(self, sql: str) -> List[Dict[str, Any]]:
        import asyncio
        return await asyncio.to_thread(self._db.query_sql, sql)

    async def remember(
        self,
        text: str,
        triples: Optional[Any] = None,
        metadata: Optional[dict] = None,
        memory_type: Optional[str] = None,
    ) -> str:
        import asyncio
        return await asyncio.to_thread(self._db.remember, text, triples, metadata, memory_type)

    async def recall_text(self, query: str, **kwargs) -> List[UnifiedMemoryAtom]:
        import asyncio
        return await asyncio.to_thread(self._db.recall_text, query, **kwargs)

    async def save_state(self, key: str, payload: Any, parent_key: str = None) -> str:
        import asyncio
        return await asyncio.to_thread(self._db.save_state, key, payload, parent_key)

    async def claim_entity(self, entity: str, claimer: str, expiry_secs: int = 30) -> bool:
        import asyncio
        return await asyncio.to_thread(self._db.claim_entity, entity, claimer, expiry_secs)

    async def unclaim_entity(self, entity: str, claimer: str) -> bool:
        import asyncio
        return await asyncio.to_thread(self._db.unclaim_entity, entity, claimer)

    async def get_entities(self, prefix: str = None) -> List[str]:
        import asyncio
        return await asyncio.to_thread(self._db.get_entities, prefix)

    async def get_entity_history(self, entity: str) -> List[UnifiedMemoryAtom]:
        import asyncio
        return await asyncio.to_thread(self._db.get_entity_history, entity)

    async def add_memory(self, content: Any, embedding: np.ndarray = None, triples: List[tuple] = None) -> str:
        import asyncio
        return await asyncio.to_thread(self._db.add_memory, content, embedding, triples)

    async def remember_batch(self, texts: List[str], triples_list: List[List[tuple]] = None) -> List[str]:
        import asyncio
        return await asyncio.to_thread(self._db.remember_batch, texts, triples_list)

    async def search(self, query: str, **kwargs) -> List[UnifiedMemoryAtom]:
        import asyncio
        # If it's a semantic search (text query), use recall_text
        if isinstance(query, str):
            return await asyncio.to_thread(self._db.recall_text, query, **kwargs)
        return []

    async def recall_by_entity(self, entity: str) -> List[UnifiedMemoryAtom]:
        import asyncio
        return await asyncio.to_thread(self._db.recall_by_entity, entity)

class GoogleEmbedder:
    """Wrapper for Google GenAI Embedding service."""
    def __init__(self, model_id: str, dim: int = 3072):
        self.model_id = model_id
        self.dim = dim
        self._client = None

    def _get_client(self):
        if self._client is None:
            import os
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai is required to use GoogleEmbedder. "
                    "Install it with: pip install epochdb[google]"
                )
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment.")
            self._client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
        return self._client

    def encode(self, text: Union[str, List[str]], normalize_embeddings: bool = True) -> np.ndarray:
        client = self._get_client()
        logger.debug(f"[GoogleEmbedder] Encoding text (type={type(text)})...")
        model_path = self.model_id if "/" in self.model_id else f"models/{self.model_id}"
        
        is_single = isinstance(text, str)
        contents = [text] if is_single else list(text)
        
        import time
        import random
        max_retries = 6
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                result = client.models.embed_content(
                    model=model_path,
                    contents=contents,
                    config={'output_dimensionality': self.dim}
                )
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Resource exhausted" in err_str:
                    if attempt == max_retries - 1:
                        logger.error(f"[GoogleEmbedder] Rate limit retry limit reached. Propagating exception.")
                        raise e
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"[GoogleEmbedder] Rate limit hit (429). Retrying in {delay:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise e
        
        embs = [np.array(e.values, dtype=np.float32) for e in result.embeddings]
        if normalize_embeddings:
            for i in range(len(embs)):
                norm = np.linalg.norm(embs[i])
                if norm > 1e-10:
                    embs[i] /= norm
                    
        if is_single:
            return embs[0]
        return np.array(embs, dtype=np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return self.encode(texts, normalize_embeddings=True)

class OpenAIEmbedder:
    """Wrapper for OpenAI-compatible Embedding services."""
    def __init__(self, model_id: str, dim: int = 1536):
        self.model_id = model_id
        self.dim = dim
        self._api_key = None
        self._base_url = None

    def _init_config(self):
        if self._api_key is None:
            import os
            self._api_key = os.getenv("OPENAI_API_KEY")
            if not self._api_key:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            self._base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/embeddings")
            if self._base_url.endswith("/v1") or self._base_url.endswith("/v1/"):
                self._base_url = self._base_url.rstrip("/") + "/embeddings"

    def encode(self, text: Union[str, List[str]], normalize_embeddings: bool = True) -> np.ndarray:
        self._init_config()
        import requests
        
        is_single = isinstance(text, str)
        contents = [text] if is_single else list(text)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        payload = {
            "model": self.model_id,
            "input": contents
        }
        
        if "text-embedding-3" in self.model_id:
            payload["dimensions"] = self.dim
            
        logger.debug(f"[OpenAIEmbedder] Sending request to {self._base_url} with model {self.model_id}...")
        
        import time
        import random
        max_retries = 6
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self._base_url, json=payload, headers=headers, timeout=30.0)
                response.raise_for_status()
                break
            except Exception as e:
                err_str = str(e)
                if attempt < max_retries - 1 and ("429" in err_str or "RATE_LIMIT" in err_str or "TOO MANY REQUESTS" in err_str.upper()):
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"[OpenAIEmbedder] Rate limit hit (429). Retrying in {delay:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"[OpenAIEmbedder] Request failed: {e}")
                    raise e
                    
        res_data = response.json()
        embeddings_list = [np.array(item["embedding"], dtype=np.float32) for item in res_data["data"]]
        
        embs = [np.array(e, dtype=np.float32) for e in embeddings_list]
        
        if normalize_embeddings:
            for i in range(len(embs)):
                norm = np.linalg.norm(embs[i])
                if norm > 1e-10:
                    embs[i] /= norm
                    
        if is_single:
            return embs[0]
        return np.array(embs, dtype=np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return self.encode(texts, normalize_embeddings=True)

class OllamaEmbedder:
    """Wrapper for local Ollama Embedding service."""
    def __init__(self, model_name: str = "all-minilm", dim: int = 384):
        self.model_name = model_name
        self.dim = dim
        self.url = "http://localhost:11434/api/embeddings"

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        import requests
        logger.info(f"[OllamaEmbedder] Encoding text with {self.model_name}...")
        response = requests.post(
            self.url,
            json={"model": self.model_name, "prompt": text},
            timeout=10
        )
        response.raise_for_status()
        emb = np.array(response.json()["embedding"], dtype=np.float32)
        
        if normalize_embeddings:
            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                emb = emb / norm
        return emb

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.encode(t) for t in texts])

class FallbackEmbedder:
    """Wraps two embedders and falls back to the second if the first fails with a rate limit error."""
    def __init__(self, primary: Any, secondary: Any):
        self.primary = primary
        self.secondary = secondary

    def encode(self, text: str, **kwargs) -> np.ndarray:
        try:
            return self.primary.encode(text, **kwargs)
        except Exception as e:
            # Check if it's a rate limit or resource exhausted error (429)
            err_str = str(e).upper()
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "TOO MANY REQUESTS" in err_str:
                logger.warning(f"Primary embedder failed (429), falling back to secondary: {e}")
                return self.secondary.encode(text, **kwargs)
            raise e

    def encode_batch(self, texts: List[str], **kwargs) -> np.ndarray:
        try:
            return self.primary.encode_batch(texts, **kwargs)
        except Exception as e:
            err_str = str(e).upper()
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "TOO MANY REQUESTS" in err_str:
                logger.warning(f"Primary batch embedder failed (429), falling back to secondary: {e}")
                return self.secondary.encode_batch(texts, **kwargs)
            raise e

