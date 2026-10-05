# EpochDB (Synchronous API)

The `EpochDB` class is the primary entry point for synchronous applications and scripts.

---

## Constructor

```python
EpochDB(
    storage_dir: str = "./.epochdb_data",
    dim: int | None = None,
    embedding_model: str | None = None,
    epoch_duration_secs: int = 3600,
    saliency_threshold: float = 0.1,
    auto_extract: bool = False,
    extraction_model: str = "local",
    async_extract: bool = True,
    validators: list[SymbolicValidator] | None = None,
    tenant: str | None = None,
    wal_sync_interval: float = 0.0,
    cold_search_mode: str = "probe",
    recency_epochs: int = 2,
    centroid_probes: int = 2,
    parquet_compression: str = "zstd",
    parquet_compression_level: int = 3,
)
```

### Parameters

- **`storage_dir`** (`str`): Base filesystem directory where the WAL, Parquet files, and HNSW indexes are stored.
- **`dim`** (`int`): Dimensionality of embedding vectors. Enforced automatically if an `embedding_model` is specified.
- **`embedding_model`** (`str`): Local HuggingFace/SentenceTransformer model name (e.g. `"all-MiniLM-L6-v2"`) or provider prefix (`"openai:text-embedding-3-small"`, `"google:text-embedding-004"`).
- **`epoch_duration_secs`** (`int`): Time window before the Hot Tier is serialized to Cold Tier Parquet archives. Default: `3600`.
- **`saliency_threshold`** (`float`): Minimum cosine similarity required for vector candidate inclusion. Default: `0.1`.
- **`auto_extract`** (`bool`): If `True`, automatically extracts Knowledge Graph triples from inserted text.
- **`extraction_model`** (`str`): Model used for triple extraction (`"local"`, `"hf:small"`, `"google:gemini-2.5-flash"`).
- **`async_extract`** (`bool`): If `True`, extraction runs in a non-blocking background thread.
- **`validators`** (`list[SymbolicValidator]`): Optional list of LCAG symbolic rule checkers for `propose()`.
- **`tenant`** (`str`): Multi-tenant isolation directory partition.
- **`wal_sync_interval`** (`float`): Seconds between WAL disk syncs (`0.0` forces immediate `fsync` for ACID safety).
- **`cold_search_mode`** (`str`): `"probe"` (fast probe set) or `"exhaustive"` (scans all epochs).

---

## Methods

### `remember()`
```python
def remember(
    self,
    text: str,
    vector: list[float] | np.ndarray | None = None,
    metadata: dict | None = None,
) -> str
```
Inserts a memory atom into the Write-Ahead Log, Hot Tier HNSW index, and Active Knowledge Graph.
- **Returns**: The generated `atom_id` (`str`).

---

### `propose()`
```python
def propose(
    self,
    text: str,
    vector: list[float] | np.ndarray | None = None,
    metadata: dict | None = None,
) -> dict
```
Stages a candidate atom and runs registered `SymbolicValidator` instances before committing.
- **Returns**: `{"status": "APPROVED", "memory_id": "..."}` or `{"status": "REJECTED", "error": "..."}`.

---

### `query()`
```python
def query(
    self,
    query_text: str | None = None,
    query_vector: list[float] | None = None,
    k: int = 5,
    filters: dict | None = None,
    query_entities: list[str] | None = None,
    expand_hops: int = 2,
    expand_context: bool = False,
    temporal_window: int = 1,
) -> list[Memory]
```
Executes the 5-stage retrieval pipeline.
- **`filters`**: MongoDB-style metadata filter dictionary (`$gt`, `$in`, etc.).
- **`expand_hops`**: Number of Knowledge Graph relational hops to explore.
- **`expand_context`**: If `True`, returns adjacent chronological turns surrounding matches.
- **Returns**: Ranked list of `Memory` domain objects.

---

### `get_entity()`
```python
def get_entity(self, name: str) -> Entity
```
Retrieves an `Entity` domain object for relational exploration.

---

### `entity_graph()`
```python
def entity_graph(self, name: str, depth: int = 2) -> Graph
```
Extracts a directed subgraph around the target entity up to `depth` hops away.

---

### `query_sql()`
```python
def query_sql(self, sql_query: str) -> list[dict]
```
Executes vectorized DuckDB SQL over the `cold_tier` view covering all historical Parquet archives.

---

### `delete()`
```python
def delete(self, atom_id: str, hard: bool = False) -> bool
```
Marks a memory atom as deleted (`hard=False`) or permanently removes it (`hard=True`).

---

### `compact()`
```python
def compact() -> None
```
Consolidates historical Parquet archives and purges soft-deleted records.
