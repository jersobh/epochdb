# Changelog

All notable changes to **EpochDB** will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.4.0] - 2026-04-13

### Fixed

- **WAL / Async Flush Race Condition** (`engine.py`): The Write-Ahead Log is now
  cleared *inside* the async flush thread, and only *after* the Parquet file is
  successfully written. Previously the WAL was cleared before the async write began,
  meaning a crash between those two points would permanently lose the epoch's data.
- **FileLock Race Condition** (`transaction.py`): Replaced the check-then-create
  pattern (`os.path.exists` → `open`) with an atomic `O_CREAT | O_EXCL` open, which
  eliminates the TOCTOU window between two concurrently starting processes.
- **Stale Lock Auto-Recovery** (`transaction.py`): If the lock file belongs to a PID
  that is no longer alive (e.g., after a `kill -9`), it is automatically removed and
  acquisition retries. Previously, every restart after a crash required manual deletion
  of the `.lock` file.
- **Triples Serialization** (`cold_tier.py`): Triples are now stored as JSON arrays
  (via `json.dumps`) instead of Python `repr` strings. The previous format silently
  dropped triples containing quotes, backslashes, or non-ASCII characters on
  deserialization.
- **Payload Serialization** (`cold_tier.py`): Payloads are now stored as JSON strings,
  enabling safe round-trip of dicts, lists, and strings. Previously, structured payloads
  (e.g., `{"role": "user"}`) were converted to `repr` strings and could not be recovered
  as their original type.
- **WAL Replay on Startup** (`engine.py`): On initialization, EpochDB now reads the
  WAL and re-applies any `ADD` operations that were not followed by a `COMMIT`. This
  provides genuine crash recovery — previously the WAL was only written but never read
  back.
- **`access_count` for Cold Atoms** (`retrieval.py`): Access-count increments for
  cold-tier atoms are now tracked in-memory via a delta map and applied when atoms are
  loaded from Parquet. Previously, `access_count += 1` mutated a temporary in-memory
  copy that was immediately discarded, making the saliency formula inaccurate for
  historical atoms.

### Added

- **Context Manager Support** (`engine.py`): `EpochDB` now implements `__enter__` /
  `__exit__`, enabling `with EpochDB(...) as db:` usage with guaranteed resource cleanup.
- **Auto-Embedding Convenience API** (`engine.py`): Two new methods `remember(text)`
  and `recall_text(query)` embed text automatically when `EpochDB` is initialized with
  a `model` name. The raw `add_memory(embedding=...)` path is fully preserved.

  ```python
  with EpochDB(storage_dir="./mem", model="all-MiniLM-L6-v2") as db:
      db.remember("Alice lives in Paris", triples=[("Alice", "lives_in", "Paris")])
      results = db.recall_text("Where does Alice live?")
  ```

- **HotTier Auto-Resize** (`hot_tier.py`): When the HNSW index reaches 90% of its
  capacity, it is automatically doubled via `resize_index()`. Previously, inserting
  more than `max_elements` (hardcoded at 10,000) atoms would raise an exception.
- **Configurable Hot-Tier Capacity** (`engine.py`): `EpochDB.__init__` now accepts
  a `hot_tier_capacity: int = 10_000` parameter.

### Changed

- **Batched Global Entity Index Saves** (`engine.py`): `global_kg.json` is now
  written to disk every 50 inserts (configurable via `_KG_FLUSH_INTERVAL`) instead of
  on every single `add_memory()` call. It is always flushed on `close()` and at
  checkpoint boundaries.
- **Checkpointer Storage Format** (`checkpointer.py`): Checkpoint files are now stored
  as JSON (`.ckpt.json`) instead of pickle (`.ckpt`). This makes checkpoints
  forward-compatible across Python and LangGraph versions. Existing pickle checkpoints
  are transparently migrated to JSON on first read.
- **Optional Dependencies** (`pyproject.toml`): `langgraph`, `sentence-transformers`,
  and `pytest` are no longer hard runtime dependencies. The core install (`pip install
  epochdb`) now only requires `numpy`, `pyarrow`, `hnswlib`, `zstandard`, and
  `requests`. Use extras for optional features:
  - `pip install epochdb[embeddings]` — for `remember()` / `recall_text()`
  - `pip install epochdb[langgraph]` — for `EpochDBCheckpointer`
  - `pip install epochdb[all]` — for everything
  - `pip install epochdb[dev]` — for development (includes `pytest`)

---

## [0.3.0] - 2026-04-10

### Added

- **3-Way RRF Ranking**: Implemented a comprehensive search ranking engine using
  Reciprocal Rank Fusion of three independent factors: Semantic Similarity, Temporal
  Recency, and Entity Overlap.
- **Query Entity Extraction Support**: The `recall` API now accepts a `query_entities`
  parameter, enabling the engine to prioritize memories that share common actors,
  locations, or identifiers with the current query.

### Changed

- **Default Dimensionality**: Optimized for 384D embeddings (e.g., `all-MiniLM-L6-v2`),
  providing significant memory and disk savings with consistent benchmark performance.

---

## [0.2.2] - 2026-04-10

### Fixed

- **Storage Race Condition**: Epoch IDs now utilize UUID-based hashing (`epoch_uuid8`)
  instead of pure timestamps. This prevents critical data erasure (overwrites) during
  rapid-fire checkpoints in high-ingestion agentic workloads.
- **Longitudinal Recall Optimization**: Resolved a regression in the `LongMemEval`
  benchmark. Historical facts are no longer outranked by recency bias through expanded
  candidate depth and refined RRF constants.

### Changed

- **Retrieval Depth**: Increased internal semantic candidate hook depth to `top_k * 3`
  during the expansion phase to ensure robust retrieval of older memories.

---

## [0.2.1] - 2026-04-10

### Added

- **Zstandard Compression**: Integrated `Zstd` for Parquet archives. When combined with
  INT8 quantization, this provides an enterprise-grade compression ratio for
  vector-heavy datasets.

---

## [0.2.0] - 2026-04-09

### Added

- **Asynchronous Tiering**: Implemented daemon-threaded background flushes. The engine
  no longer blocks the main application loop during Hot-to-Cold tier transitions.
- **INT8 Scalar Quantization**: Native down-casting for embeddings in the Cold Tier,
  providing a 4x reduction in storage footprint with negligible recall loss.
- **Hybrid RRF Ranking**: Implemented Reciprocal Rank Fusion (RRF) to unify semantic
  similarity and access-based saliency scores.
- **Semantic Edge Filtering**: Configurable cosine-similarity thresholds for Knowledge
  Graph traversal (default `> 0.15`) to prevent "super-node" expansion noise.
- **LangGraph Checkpointer**: Native `EpochDBCheckpointer` for persisting agentic graph
  states and thread history.

---

## [0.1.3] - 2026-04-08

### Added

- **ACID Write-Ahead Log (WAL)**: Ensures memory atom safety across crashes.
- **Global Entity Index (GEI)**: Multi-epoch relational expansion bridging
  time-partitioned Parquet files.

---

## [0.1.0] - 2023-11-20

### Added

- Initial release with Tiered Vector Storage (HNSW in-memory + Parquet on-disk).
- `UnifiedMemoryAtom` dataclass: raw text payload paired with dense embedding and
  optional KG triples.
- `EpochDB` engine with configurable epoch duration and saliency threshold.
- Dimensionality enforcement: raises on startup if stored data dimension doesn't match
  the configured dimension.

---

[0.4.0]: https://github.com/jersobh/epochdb/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jersobh/epochdb/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/jersobh/epochdb/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/jersobh/epochdb/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/jersobh/epochdb/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/jersobh/epochdb/compare/v0.1.0...v0.1.3
[0.1.0]: https://github.com/jersobh/epochdb/releases/tag/v0.1.0
