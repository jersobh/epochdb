import uuid
import numpy as np
import logging
import datetime
from typing import List, Optional, Dict, Any, Tuple, Union

from epochdb.engine import EpochDB as EngineEpochDB
from epochdb.core.atom import UnifiedMemoryAtom, PayloadType, MemoryType

logger = logging.getLogger(__name__)


class Memory:
    """Rich Memory object representing a stored memory atom."""
    def __init__(self, atom: UnifiedMemoryAtom):
        self.id = atom.id
        if isinstance(atom.payload, str):
            self.text = atom.payload
        elif hasattr(atom.payload, "description") and atom.payload.description:
            self.text = atom.payload.description
        else:
            self.text = str(atom.payload)

        self.metadata = atom.metadata or {}
        self.created_at = atom.created_at
        self.access_count = atom.access_count
        self.triples = atom.triples
        self.payload_type = atom.payload_type.value if atom.payload_type else "text"
        self.memory_type = atom.memory_type.value if atom.memory_type else "general"
        self.namespace = atom.namespace
        self._atom = atom

def _build_pairwise_triples(extracted: list) -> list:
    seen = set()
    entities = []
    for e in extracted:
        s = str(e)
        if s and s not in seen:
            seen.add(s)
            entities.append(s)
    if len(entities) >= 2:
        res = []
        for i in range(len(entities) - 1):
            res.append((entities[i], "co_occurs_with", entities[i + 1]))
        if len(entities) > 2:
            res.append((entities[0], "co_occurs_with", entities[-1]))
        return res
    elif len(entities) == 1:
        return [(entities[0], "mentions", entities[0])]
    return []


class Entity(str):
    """Rich Entity object representing a Knowledge Graph entity."""
    def __new__(cls, val, db=None):
        obj = str.__new__(cls, val)
        obj.db = db
        return obj

    @property
    def id(self) -> str:
        return str(self)

    def timeline(self, start: Optional[Any] = None, end: Optional[Any] = None) -> List[Memory]:
        """Chronological evolution of this entity."""
        return self.db.get_timeline(entity_id=self.id, start=start, end=end)

    def related(self, depth: int = 1) -> List['Entity']:
        """Related entities in the Global Entity Index."""
        neighbors = self.db.kg_manager.get_neighbors(self.id)
        return [Entity(n, self.db) for n in neighbors]

    def current_state(self) -> Any:
        """The latest state/payload for this entity."""
        history = self.db.get_entity_history(self.id)
        if not history:
            return None
        return history[-1].payload

    def __repr__(self):
        return f"Entity(id={self.id!r})"


class Graph:
    """Represents a segment of the Global Entity Index graph."""
    def __init__(self, nodes: List[str], edges: List[Dict[str, Any]]):
        self.nodes = nodes
        self.edges = edges

    def __repr__(self):
        return f"Graph(nodes={len(self.nodes)}, edges={len(self.edges)})"


class EpochDB(EngineEpochDB):
    """
    Opinionated, Python-first facade for the EpochDB Agentic Memory Engine.
    """
    def __init__(
        self,
        storage_dir: str = "./memory",
        embedding_model: Optional[str] = None,
        auto_flush: bool = True,
        dim: int = 384,
        tenant: Optional[str] = None,
        namespace: Optional[str] = None,
        wal_sync_interval: float = 0.0,
        parquet_compression: str = "ZSTD",
        parquet_compression_level: int = 3,
        wal_use_uring: bool = True,
        auto_extract: bool = False,
        extraction_model: Optional[str] = None,
        **kwargs
    ):
        self.auto_flush = auto_flush
        self.extraction_model = extraction_model
        if "model" in kwargs:
            val = kwargs.pop("model")
            if embedding_model is None:
                embedding_model = val
        super().__init__(
            storage_dir=storage_dir,
            dim=dim,
            model=embedding_model,
            tenant=tenant,
            namespace=namespace,
            wal_sync_interval=wal_sync_interval,
            parquet_compression=parquet_compression,
            parquet_compression_level=parquet_compression_level,
            wal_use_uring=wal_use_uring,
            auto_extract=auto_extract,
            extraction_model=extraction_model,
            **kwargs
        )

    def remember(
        self,
        text: str,
        triples: Optional[Any] = None,
        metadata: Optional[dict] = None,
        memory_type: Optional[str] = None,
    ) -> str:
        """Primary write method to store memories."""
        if isinstance(triples, dict) and metadata is None:
            metadata = triples
            triples = None

        metadata = metadata or {}
        with self._internal_lock:
            if self._model_name:
                embedder = self._get_embedder()
                emb = embedder.encode(text, normalize_embeddings=True)
                embedding = np.array(emb, dtype=np.float32)
            else:
                embedding = np.zeros(self.dim, dtype=np.float32)

            if triples is None:
                triples = metadata.get("triples")
            
            if not triples and self.auto_extract:
                from epochdb.core.fact_extractor import FactExtractor
                extractor = FactExtractor(self, self.extraction_model)
                triples = extractor.extract(text)
            elif not triples:
                extracted = self.extract_entities(text)
                triples = _build_pairwise_triples(extracted)

            atom_id = self.add_memory(
                payload=text,
                embedding=embedding,
                triples=triples,
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

    def remember_batch(self, items: list) -> List[str]:
        """Store multiple memories at once."""
        with self._internal_lock:
            texts = []
            metadatas = []
            for item in items:
                if isinstance(item, str):
                     texts.append(item)
                     metadatas.append({})
                elif isinstance(item, dict):
                     texts.append(item.get("text", ""))
                     metadatas.append(item.get("metadata", {}))

            if not texts:
                return []

            if self._model_name:
                embedder = self._get_embedder()
                embs = embedder.encode_batch(texts)
            else:
                embs = [np.zeros(self.dim, dtype=np.float32) for _ in texts]

            batch_items = []
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                triples = metadata.get("triples") or []
                if not triples and self.auto_extract:
                    from epochdb.core.fact_extractor import FactExtractor
                    extractor = FactExtractor(self, self.extraction_model)
                    triples = extractor.extract(text)
                elif not triples:
                    extracted = self.extract_entities(text)
                    triples = _build_pairwise_triples(extracted)

                batch_items.append({
                    "payload": text,
                    "embedding": embs[i],
                    "triples": triples,
                    "metadata": metadata,
                })

            return self.add_memory_batch(batch_items)

    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by its ID."""
        with self._internal_lock:
            atom = self.hot_tier.atoms.get(memory_id)
            if not atom:
                for epoch_id in self.cold_tier.get_all_epochs():
                    atoms = self.cold_tier.load_atom_metadata(epoch_id, [memory_id])
                    if atoms:
                        atom = atoms[0]
                        break
            if atom and not atom.metadata.get("_deleted"):
                return Memory(atom)
            return None

    def update(self, memory_id: str, text: Optional[str] = None, metadata: Optional[dict] = None):
        """Update a memory's text or metadata."""
        with self._internal_lock:
            atom = self.hot_tier.atoms.get(memory_id)
            if atom:
                if text is not None:
                    atom.payload = text
                    if self._model_name:
                        embedder = self._get_embedder()
                        emb = embedder.encode(text, normalize_embeddings=True)
                        atom.embedding = np.array(emb, dtype=np.float32)
                if metadata is not None:
                    atom.metadata.update(metadata)
                self.wal.append("ADD", atom.to_dict())
            else:
                old_atom = None
                for epoch_id in self.cold_tier.get_all_epochs():
                    atoms = self.cold_tier.load_atom_metadata(epoch_id, [memory_id])
                    if atoms:
                        old_atom = atoms[0]
                        break
                if old_atom:
                    new_text = text if text is not None else old_atom.payload
                    new_meta = old_atom.metadata.copy()
                    if metadata is not None:
                        new_meta.update(metadata)

                    if text is not None and self._model_name:
                        embedder = self._get_embedder()
                        emb = embedder.encode(text, normalize_embeddings=True)
                        embedding = np.array(emb, dtype=np.float32)
                    else:
                        embedding = old_atom.embedding

                    self.add_memory(
                        payload=new_text,
                        embedding=embedding,
                        triples=old_atom.triples,
                        atom_id=memory_id,
                    )
                    new_atom = self.hot_tier.atoms[memory_id]
                    new_atom.metadata = new_meta
                    self.wal.append("ADD", new_atom.to_dict())

    def delete(self, memory_id: str, hard: bool = False):
        """Delete a memory from the store."""
        with self._internal_lock:
            if hard:
                if memory_id in self.hot_tier.atoms:
                    self.hot_tier.atoms.pop(memory_id)
                self.wal.append("DELETE", {"id": memory_id})
            else:
                self.update(memory_id, metadata={"_deleted": True})

    def fork(self, parent_epoch_id: Optional[str] = None, new_epoch_id: Optional[str] = None) -> str:
        """Create a logical fork branch."""
        with self._internal_lock:
            if parent_epoch_id is None:
                parent_epoch_id = self.current_epoch_id
            if new_epoch_id is None:
                new_epoch_id = f"epoch_{uuid.uuid4().hex[:8]}"
            super().fork(parent_epoch_id, new_epoch_id)
            return new_epoch_id

    def query(self, text: str, k: int = 5, filters: Optional[dict] = None, min_score: float = 0.0, memory_type: Optional[str] = None, context_window: int = 0, expand_hops: int = 0) -> List[Memory]:
        """Query memory using semantic search."""
        query_entities = [str(e) for e in self.extract_entities(text)]
        if self._model_name:
            embedder = self._get_embedder()
            emb = np.array(embedder.encode(text, normalize_embeddings=True), dtype=np.float32)
        else:
            emb = np.zeros(self.dim, dtype=np.float32)

        with self._internal_lock:
            atoms = self.retriever.search(
                query_emb=emb,
                top_k=k,
                expand_hops=max(0, int(expand_hops or 0)),
                query_entities=query_entities,
                filters=filters,
                context_window=context_window,
            )
            # Filter by memory_type if specified
            if memory_type:
                try:
                    mt = MemoryType(memory_type)
                    atoms = [a for a in atoms if a.memory_type == mt]
                except ValueError:
                    pass
            memories = []
            for atom in atoms:
                score = 0.0
                if emb.any() and atom.embedding.any():
                    score = np.dot(atom.embedding, emb) / (
                        np.linalg.norm(atom.embedding) * np.linalg.norm(emb) + 1e-10
                    )
                if score >= min_score:
                    memories.append(Memory(atom))
            return memories

    def analyze(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract triples from text without storing it."""
        from epochdb.core.fact_extractor import FactExtractor
        extractor = FactExtractor(self, self.extraction_model)
        return extractor.extract(text)

    def multi_hop(self, text: str, hops: int = 2, k: int = 5, filters: Optional[dict] = None, context_window: int = 0) -> List[Memory]:
        """Multi-hop relational query."""
        query_entities = [str(e) for e in self.extract_entities(text)]
        if self._model_name:
            embedder = self._get_embedder()
            emb = np.array(embedder.encode(text, normalize_embeddings=True), dtype=np.float32)
        else:
            emb = np.zeros(self.dim, dtype=np.float32)

        with self._internal_lock:
            atoms = self.retriever.search(
                query_emb=emb,
                top_k=k,
                expand_hops=hops,
                query_entities=query_entities,
                filters=filters,
                context_window=context_window,
            )
            return [Memory(atom) for atom in atoms]

    def adaptive_query(self, query: str, k: int = 5, context_window: int = 0) -> List[Memory]:
        """Intelligently route and execute the query using the AdaptiveRouter."""
        from epochdb.retrieval.router import AdaptiveRouter
        router = AdaptiveRouter(self, model_id=self.extraction_model)
        return router.route_and_query(query, k=k, context_window=context_window)

    def get_timeline(self, entity_id: Optional[str] = None, start: Optional[Any] = None, end: Optional[Any] = None) -> List[Memory]:
        """Get chronological history of memories."""
        start_ts = start.timestamp() if isinstance(start, datetime.datetime) else start
        end_ts = end.timestamp() if isinstance(end, datetime.datetime) else end

        with self._internal_lock:
            atoms = []
            if entity_id:
                atoms = self.get_entity_history(entity_id)
            else:
                atoms = list(self.hot_tier.atoms.values())
                for epoch_id in self.cold_tier.get_all_epochs():
                    atoms.extend(self.cold_tier.load_epoch(epoch_id))

            filtered = []
            for a in atoms:
                if a.metadata.get("_deleted"):
                    continue
                if start_ts and a.created_at < start_ts:
                    continue
                if end_ts and a.created_at > end_ts:
                    continue
                filtered.append(Memory(a))

            filtered.sort(key=lambda m: m.created_at)
            return filtered

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract Entity objects from text."""
        entity_names = super().extract_entities(text)
        return [Entity(name, self) for name in entity_names]

    def get_entity(self, entity_id: str) -> Entity:
        """Retrieve a specific Entity by its identifier."""
        return Entity(entity_id, self)

    def entity_graph(self, entity_id: str, depth: int = 2) -> Graph:
        """Construct the relational graph around an entity."""
        with self._internal_lock:
            nodes = set()
            edges = []
            visited = set()
            queue = [(entity_id, 0)]

            while queue:
                curr_ent, curr_depth = queue.pop(0)
                if curr_ent in visited:
                    continue
                visited.add(curr_ent)
                nodes.add(curr_ent)

                atoms = self.recall_by_entity(curr_ent)
                for atom in atoms:
                    if atom.metadata.get("_deleted"):
                        continue
                    for s, p, o in atom.triples:
                        edge = {"source": s, "target": o, "predicate": p, "memory_id": atom.id}
                        if edge not in edges:
                            edges.append(edge)
                        nodes.add(s)
                        nodes.add(o)

                if curr_depth < depth:
                    neighbors = self.kg_manager.get_neighbors(curr_ent)
                    for n in neighbors:
                        if n not in visited:
                            queue.append((n, curr_depth + 1))

            return Graph(nodes=list(nodes), edges=edges)

    def compact(self):
        """Trigger deduplication and compaction in the Cold Tier."""
        with self._internal_lock:
            self.flush()
            self.cold_tier.compact(self.current_epoch_id, self.kg_manager)

    def stats(self) -> dict:
        """Retrieve system diagnostics stats."""
        with self._internal_lock:
            return {
                "memory_count": self.get_total_atoms(),
                "l1_size": len(self.hot_tier.atoms),
                "l2_size": self.cold_tier.get_total_atoms(),
                "entity_count": len(self.get_entities()),
            }


class AsyncEpochDB:
    """Asynchronous wrapper for EpochDB public facade."""
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._db = None

    async def _get_db(self):
        if self._db is None:
            import asyncio
            self._db = await asyncio.to_thread(EpochDB, **self._kwargs)
        return self._db

    def _get_db_sync(self):
        if self._db is None:
            self._db = EpochDB(**self._kwargs)
        return self._db

    @property
    def hot_tier(self):
        return self._get_db_sync().hot_tier

    @property
    def wal(self):
        return self._get_db_sync().wal

    @property
    def cold_tier(self):
        return self._get_db_sync().cold_tier

    @property
    def kg_manager(self):
        return self._get_db_sync().kg_manager

    @property
    def dim(self) -> int:
        return self._get_db_sync().dim

    @property
    def parquet_compression(self) -> str:
        return self._get_db_sync().parquet_compression

    @parquet_compression.setter
    def parquet_compression(self, value: str):
        self._get_db_sync().parquet_compression = value

    @property
    def parquet_compression_level(self) -> Optional[int]:
        return self._get_db_sync().parquet_compression_level

    @parquet_compression_level.setter
    def parquet_compression_level(self, value: Optional[int]):
        self._get_db_sync().parquet_compression_level = value

    @property
    def wal_use_uring(self) -> bool:
        return self._get_db_sync().wal_use_uring

    @property
    def _model_name(self) -> Optional[str]:
        return self._get_db_sync()._model_name

    def _get_embedder(self):
        return self._get_db_sync()._get_embedder()

    async def __aenter__(self):
        self._db = await self._get_db()
        return self

    async def query_sql(self, sql: str) -> List[Dict[str, Any]]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.query_sql, sql)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        import asyncio
        if self._db:
            await asyncio.to_thread(self._db.close)
            self._db = None

    async def remember(
        self,
        text: str,
        triples: Optional[Any] = None,
        metadata: Optional[dict] = None,
        memory_type: Optional[str] = None,
    ) -> str:
        if isinstance(triples, dict) and metadata is None:
            metadata = triples
            triples = None
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.remember, text, triples, metadata, memory_type)

    async def analyze(self, text: str) -> List[Tuple[str, str, str]]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.analyze, text)

    async def remember_batch(self, items: list) -> List[str]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.remember_batch, items)

    async def get(self, memory_id: str) -> Optional[Memory]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.get, memory_id)

    async def update(self, memory_id: str, text: Optional[str] = None, metadata: Optional[dict] = None):
        import asyncio
        db = await self._get_db()
        await asyncio.to_thread(db.update, memory_id, text, metadata)

    async def delete(self, memory_id: str, hard: bool = False):
        import asyncio
        db = await self._get_db()
        await asyncio.to_thread(db.delete, memory_id, hard)

    async def fork(self, parent_epoch_id: Optional[str] = None) -> str:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.fork, parent_epoch_id)

    async def query(self, text: str, **kwargs) -> List[Memory]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.query, text, **kwargs)

    async def multi_hop(self, text: str, **kwargs) -> List[Memory]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.multi_hop, text, **kwargs)

    async def get_timeline(self, **kwargs) -> List[Memory]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.get_timeline, **kwargs)

    async def extract_entities(self, text: str) -> List[Entity]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.extract_entities, text)

    async def get_entity(self, entity_id: str) -> Entity:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.get_entity, entity_id)

    async def entity_graph(self, entity_id: str, depth: int = 2) -> Graph:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.entity_graph, entity_id, depth)

    async def compact(self):
        import asyncio
        db = await self._get_db()
        await asyncio.to_thread(db.compact)

    async def stats(self) -> dict:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.stats)

    async def add_memory(self, payload: Any, embedding: np.ndarray, triples: Optional[List[tuple]] = None, atom_id: Optional[str] = None) -> str:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.add_memory, payload, embedding, triples, atom_id)

    async def recall(self, query_emb: np.ndarray, top_k: int = 5, expand_hops: int = 1, query_entities: Optional[List[str]] = None) -> List[Any]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.recall, query_emb, top_k, expand_hops, query_entities)

    async def force_checkpoint(self):
        import asyncio
        db = await self._get_db()
        await asyncio.to_thread(db.force_checkpoint)

    async def recall_text(self, query: str, top_k: int = 5, expand_hops: int = 1, query_entities: Optional[List[str]] = None) -> List[Any]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.recall_text, query, top_k, expand_hops, query_entities)

    async def recall_by_entity(self, entity: str) -> List[Any]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.recall_by_entity, entity)

    async def get_total_atoms(self) -> int:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.get_total_atoms)

    async def get_entities(self, prefix: Optional[str] = None) -> List[str]:
        import asyncio
        db = await self._get_db()
        return await asyncio.to_thread(db.get_entities, prefix)

    async def close(self):
        import asyncio
        if self._db:
            await asyncio.to_thread(self._db.close)
            self._db = None

    async def flush(self):
        import asyncio
        db = await self._get_db()
        await asyncio.to_thread(db.flush)

