<p align="center">
  <img src="logo-epoch.png" alt="EpochDB Logo" width="120" />
</p>

# Changelog

All notable changes to EpochDB will be documented in this file.

## [1.2.0] - 2026-06-25
### Added
- **SQLite WAL Mode & Concurrency Tuning**: Enabled `PRAGMA journal_mode = WAL`, `PRAGMA synchronous = NORMAL`, and `PRAGMA busy_timeout = 5000` on the Global KG SQLite database connection to eliminate write locks and support concurrent reads.
- **Engine-Level Transaction Batching**: Refactored bulk ingestion loops (`add_memory_batch` and `remember_batch`) to perform a single-commit operation, combining all Write-Ahead Log appends and SQLite Global KG index updates into a single transaction commit, resulting in up to **22x write speedups**.
- **Robust io_uring Queue Backpressure**: Added automatic queue backpressure handling to the `io_uring` system-level WAL writer, preventing submission queue overflow on slow drives.
- **Transactional Crash Recovery**: Implemented a strict transaction recovery parser that rolls back uncommitted writes at EOF while successfully replaying committed transaction blocks into the Hot Tier.
- **Parquet Compression Configuration**: Added `parquet_compression` and `parquet_compression_level` options to both sync (`EpochDB`) and async (`AsyncEpochDB`) database instances to dynamically control compression method (e.g. snappy, zstd, gzip) and level on serialization.

## [1.1.1] - 2026-06-23
### Added
- **LangChain & LangGraph Integrations**:
  - **Structured Agent Tools**: Added `get_epochdb_tools(db)` in `epochdb.tools` creating 7 custom `StructuredTool` objects for agents to read, write, update, delete, and inspect memory atoms and graphs.
  - **Native VectorStore (`EpochDBVectorStore`)**: Custom `VectorStore` wrapper in `epochdb.vectorstore` for standard LangChain similarity search with optional cosine score recalculation.
  - **Custom Multi-Hop Retriever (`EpochDBMultiHopRetriever`)**: Custom LangChain `BaseRetriever` exposing multi-hop relational search directly as a retrieval step.
  - Full compatibility with both sync (`EpochDB`) and async (`AsyncEpochDB`) interfaces with safety locks and background threading.
  - Added self-contained code examples under `examples/example_langchain.py`.

### Changed
- **CI/CD Pipeline**: Migrated the PyPI publishing workflow to Trusted Publishing (OIDC authentication via GitHub Actions), eliminating the need for PyPI API token secrets.


## [1.0.4] - 2026-06-22
### Added
- **Local Model Caching**: Added automatic caching of SentenceTransformers local embedding models to `~/.cache/epochdb_models/{safe_model_name}`. The engine will download the model only on the first run and load it locally thereafter, avoiding repeated downloads. Also added a `model_cache_path` configuration parameter to support custom model cache directories.
- **Model Name Normalization**: Automatically normalizes unicode non-breaking hyphens and dashes (such as `\u2011`) in model names to standard ASCII hyphens to prevent Hugging Face Hub validation failures.

### Changed
- **Optional Dependencies**: Made `google-genai` an optional dependency (moved to `[project.optional-dependencies]` under `google`). It is no longer installed by default with core dependencies.
- **Default Model**: Changed the default embedding model to the local lightweight model `"all-MiniLM-L6-v2"`.

## [1.0.3] - 2026-06-19
### Added
- **Client-Server Architecture**: Introduced `RemoteEpochDB` (in `epochdb/api/client.py`) and a multi-threaded HTTP server `ThreadingEpochDBServer` (in `epochdb/api/server.py`) to run and communicate with a remote instance of EpochDB using HTTP REST endpoints.
- **Multi-Tenant Partitioning**: Added a `tenant` parameter to physically partition database files under `tenants/<tenant_name>/`, ensuring strict data isolation.
- **Asynchronous WAL Syncing**: Added a `wal_sync_interval` configuration parameter to support asynchronous background thread `fsync` flushing on the WAL, optimizing throughput for write-heavy workloads.

### Changed
- **Entity Extraction Enhancements**:
  - Added acronym matching inside parentheses (e.g. mapping an entity `"(XYZ)"` to `"xyz"` in the query).
  - Added common Portuguese stopwords ("o", "a", "os", "as", "eu", "meu", "minha", etc.) to the entity extraction blacklist.

### Fixed
- **WAL Replay COMMIT Handling**: Fixed a database startup crash-recovery issue in `WriteAheadLog.replay()` where the pending operations list was not cleared when encountering a `COMMIT` record. This caused successfully committed historical atoms to be incorrectly replayed and restored to the hot tier at startup, leading to duplicated recovery effort and slower database load times.
- **Thread Safety Guarding**: Enforced concurrency thread locks (`with self._internal_lock`) across all key engine entry points, including `get_total_atoms()`, memory/batch writes, epoch checkpointing, and database lifecycle methods.

## [1.0.1] - 2026-06-18
### Added
- **Query Embedding Dimension Guard**: Added explicit validation checking to `query_vector` in `HotTier` to catch query embedding dimension mismatches and raise a clear ValueError.

### Changed
- **Adaptive Dimension Sync**: Automatically syncs engine dimension `self.dim` to match the stored data dimensionality in `metadata.json` if a mismatch is detected, rather than raising a hard ValueError.
- **Hot Tier Query Coverage**: Transitioned from using `len(self.atoms)` to `self.vector_index.get_current_count()` in `HotTier.query_vector` for accurate dimension queries and k-nearest capping.

### Fixed
- **Cold Tier Schema Safety**: Checks the parquet schema for `"embedding_max"` presence using `pq.read_schema` before reading it, preventing errors on legacy or differently structured parquet files.
- **Async Test Framework**: Migrated async checkpointer tests in `tests/test_checkpointer.py` to use `pytest.mark.anyio` to align with the installed async testing plugins and avoid execution failures.

## [1.0.0] - 2026-06-04
### Added
- **Public API Facade**: Introduced `EpochDB` and `AsyncEpochDB` facades returning rich `Memory`, `Entity`, and `Graph` domain objects.
- **Codebase Modularization**: Reorganized the codebase into structured sub-directories: `core/`, `storage/`, `entities/`, `retrieval/`, and `api/`.
- **Compaction & Soft-Delete**: Added support for historical Parquet archive compaction (`compact()`) and filter check for soft-deleted atoms (`_deleted: True`).
- **Metadata Filtering**: Integrated MongoDB-style comparison operators (`$eq`, `$ne`, `$in`, `$gt`, etc.) in the candidate deduplication phase of vector retrieval.

### Changed
- **Backward-Compatible Shims**: Retained root-level shim files (`atom.py`, `checkpointer.py`, etc.) for zero consumer import disruption.
- **Token Benchmark Optimizations**: Simulated realistic conversation lengths in mock benchmarks to demonstrate true $O(N^2)$ quadratic vs. $O(N)$ linear token growth in offline mock tests.

### Fixed
- **Payload Type Inference**: Fixed a bug where `payload_type` defaulted to text and was not automatically inferred from class type.
- **Constructor Parameter Collision**: Fixed a collision in `EpochDB.__init__` where older scripts passing the `model` keyword argument directly caused a duplicate value error.

## [0.7.2] - 2026-05-25
### Fixed
- **Rate Limit Handling**: Added exponential backoff retry logic for Gemini embedding calls to handle rate limiting gracefully.


## [0.7.1] - 2026-05-25
### Fixed
- **Rebranding Fixes**: Unified duplicate `recall_by_entity` implementations into a single thread-safe, batch-loading method.
- **Bulk retrieval**: Implemented missing `get_associations_batch` in `KGManager` for bulk entity association lookup.

## [0.7.0] - 2026-05-25
### Added
- **Topological Blackboard Serialization Filtering**: Implemented context-aware blackboard property filtering inside Juno, filtering out redundant text attributes for inactive/unmentioned neighbor concepts.
- **Unified Entity Extraction**: Standardized agent recall retrieval using the native `db.extract_entities` API instead of manual blacklist/whitelist heuristics.
- **Checkpointer Prompt Alignment**: Standardized sliding window history representations across Standard, EpochDB, and Juno graph configurations to ensure completely fair benchmarking.

## [0.6.1] - 2026-05-22
### Changed
- **Documentation Overhaul**: Audited and aligned all version references across README, how_it_works, benchmarks, examples, and lockfiles to target v0.6.1.
- **Version Bump**: Official transition to v0.6.1.

## [0.6.0] - 2026-05-16
### Added
- **Analytical Cold Tier**: Fully eliminated DuckDB dependency, replacing the cold tier analytical engine with a native PyArrow-based Scalar Handler for high-performance cross-epoch scanning and numeric aggregation.
- **Parallel Retrieval**: Implemented `ThreadPoolExecutor` in `RetrievalManager` for concurrent historical epoch scanning, significantly reducing latency for multi-epoch queries.
- **Dynamic Signal-to-Noise**: Replaced hardcoded RRF thresholds with a dynamic filtering mechanism and "soft demotion" (0.1x) to preserve context while prioritizing strong signals.
- **Improved Topic Boosting**: Transitioned to cumulative boosting for intent alignment (+15.0) and narrow entity matches (+10.0), improving retrieval precision for complex queries.
- **Quantitative Data Support**: Native support for Scalars, Time-Series, and Constraints.
- **Advanced Indexing**: Integrated B-trees for scalar range queries and R-trees for temporal series indexing.
- **SAT Solver Integration**: Integrated `z3-solver` for logical constraint feasibility checking.
- **Quantitative Intervals**: Replace R-tree with 1D IntervalTree for optimized $O(\log n + k)$ scalar range querying.
- **Dimensional Persistence**: Added `schema_registry.json` for mapping fields to base units, ensuring dimensional consistency across system restarts.
- **Reactive Cascade Graph**: `CascadeManager` dependency graph is now persisted to JSON for fast recovery of reactive constraint relationships.
- **Data Reflection**: Implemented Coefficient of Variation (CV) based statistical confidence score calculation for auto-inserting policy atoms.
- **Fork Resolvers**: Enhanced `check_feasibility` to inject fork shadow scalars before evaluating `z3` constraints.
- **Serialization Reliability**: Custom schema-based recursive serialization for `SeriesPayload` nested dataclasses, fixing `asdict()` failures.
- **Unit Mismatch Guard**: Strict dimensional rejection in `query_range` to prevent silent conversion failures.
- **Cascade Rules & Triggers**: Added a trigger mechanism in `QuantitativeIndexManager` for reactive memory updates.
- **Unit & Uncertainty Propagation**: Implemented `UnitRegistry` and uncertainty tracking for quantitative payloads.

### Changed
- **Unified Memory Atom**: Extended the atom structure to support discriminated union payloads.
- **Recall API**: Added `query_range`, `query_temporal`, and `check_feasibility` to `RetrievalManager`.
- **Retrieval Pipeline**: Updated 4-way fusion to handle typed supersession (e.g., numeric conflicts and series merging).
- **Documentation Overhaul**: Updated `README.md`, `how_it_works.md`, and examples to reflect v0.6.0 capabilities and the new Quantitative logical engine.
- **Version Bump**: Official transition to v0.6.0.

## [0.5.0] - 2026-05-02
### Added
- **Forking & Memory Branching**: Introduced `db.fork(parent_epoch_id, new_epoch_id)` to create logical forks in the memory tree. This enables memory lineage and multi-agent branching without duplicating vector storage.
- **Improved Retrieval Precision**: Further hardening of the Topic Lock mechanism for multi-hop retrieval.

### Changed
- **Documentation Overhaul**: Updated `README.md`, `how_it_works.md`, and examples to reflect v0.5.0 capabilities and the new Forking API.
- **Version Bump**: Official transition to v0.5.0.

## [0.4.6] - 2026-04-22
### Changed
- **Documentation Overhaul**: Rewrote `how_it_works.md` from scratch to accurately reflect the v0.4.5+ retrieval pipeline, architectural tiers, and Topic Lock mechanisms.
- **Improved Retrieval Clarity**: Clarified the interaction between Semantic Bootstrapping, KG Seeding, and the 4-way RRF fusion process.


## [0.4.5] - 2026-04-13
### Fixed
- **Query Entity Isolation**: The relational expansion stage was mutating `query_entities` in-place with every KG-neighbour it encountered. By the time the Topic Lock scored candidates, the original clean query intent (e.g. `{'born_in'}`) had been replaced with every entity from every atom in the candidate pool. Fixed by snapshotting `original_query_entities` as a `frozenset` before expansion begins, and using only that frozen set for Topic Lock scoring.
- **Entity Extraction Over-Matching**: `extract_entities` used a token-level fuzzy match (checking if all words of a multi-word entity appear anywhere in the query word-bag) which allowed unrelated entities like `dog`, `cat`, `Japan` to match virtually any query. Replaced with strict substring presence — entities must now literally appear in the cleaned query text.
- **Blacklist Expanded**: Added `they`, `their`, `who`, `what`, `where`, `when`, `how` to the entity blacklist, preventing common interrogative words from triggering Topic Lock boosts.
- **Benchmark Isolation**: `run_all.py` now re-initialises the `EpochDB` instance and clears the Global KG between individual benchmark runs, eliminating cross-benchmark entity count contamination.

### Changed
- **Cold Tier Precision**: Removed INT8 scalar quantization from the Cold Tier. Embeddings are now stored as full `float32` in Parquet, eliminating the 1–2% precision noise that could cause ranking flukes in high-noise scenarios.
- **Signal-to-Noise Filter**: After RRF fusion, if any atom achieves a Topic Lock score (≥ 20.0), all non-signal atoms are aggressively demoted by `1e-7`, ensuring intent-matched corrections can never be outranked by semantically adjacent noise.
- **Topic Lock Boost raised to +20.0**: Ensures a Topic-Locked atom is mathematically unreachable by pure recency or semantic rank alone.
- **Deterministic Supersession**: Supersession detection and recency ranking now use `(created_at, atom.id)` as a composite sort key, preventing non-deterministic tie-breaking when atoms have identical timestamps.

### Benchmark Results
```
  Benchmark      Metric             Score    Pass
  LoCoMo         multi-hop recall   1.000    ✓
  ConvoMem       recall@3           1.000    ✓
  LongMemEval    recall@3           1.000    ✓
  NIAH           precision@3        1.000    ✓
```

## [0.4.4] - 2026-04-13
### Added
- **Cold Tier Entity Hook**: Step 1a of the retrieval pipeline now also seeds candidates directly from the Cold Tier for matching `query_entities`, in addition to the Hot Tier. Previously, historical corrections that scored below the semantic top-K cutoff were silently excluded from the candidate pool.

## [0.4.3] - 2026-04-13
### Fixed
- **Predicate Indexing in Global KG**: Predicates are now indexed as first-class entries in `global_kg` alongside Subjects and Objects, enabling the Entity Hook to pull atoms by action/relation (e.g. `born_in`) rather than only by named entity.

## [0.4.2] - 2026-04-13

### Added
- **Async Support for LangGraph Checkpointer**: Native async methods (`aget_tuple`, `alist`, `aput`, `aput_writes`) using `asyncio.to_thread` for non-blocking I/O in agentic workflows.

## [0.4.1] - 2026-04-13
### Added
- **Entity Seeding (Topic Hook)**: Architectural upgrade to retrieval logic that seeds candidates from `query_entities` via the Global KG, ensuring relevant facts outrank semantic noise.
- **Needle in a Haystack (NIAH)**: Achieved a validated **1.000 precision@3** and **1.000 recall@3** across high-noise benchmarks.
- **Improved RRF Depth**: Increased internal candidate pool from `2x` to `10x` top_k for better rank fusion coverage.

## [0.4.0] - 2026-04-12
### Added
- **Nuclear Topic Lock**: Implemented a discrete `+5.0` additive bonus for atoms matching the query's predicate domain, ensuring factual precision in retrieval.
- **Persistent HNSW Indexing (Cold Tier)**: All historical epochs are now indexed with HNSW, transitioning retrieval from $O(N)$ linear scans to $O(\log N)$ ANN search.
- **State-Aware Supersession Filter**: Implemented logic to detect and penalize stale factual states (Subject/Predicate updates), allowing newer facts to reliably outrank old ones.
- **Persistent Saliency**: Implemented `access_deltas.json` to preserve atom access counts and importance across process restarts.
- **Strict Monotonicity**: Engine-level timestamping ensures no RRF ties and predictable recency ranking.

### Fixed
- **Self-Supersession Bug**: Fixed issue where atoms with multiple values for the same predicate (e.g., itinerary) would incorrectly penalize themselves.
- **Punctuation Extraction**: Heuristic extractor now robustly handles tokens attached to punctuation (e.g., "work?").
- **ConvoMem Recall**: Improved ConvoMem benchmark score from 0.400 to **1.000**.
- **Cold Tier Performance**: Achieved 30x performance gain in historical retrieval.

## [0.3.0] - 2026-04-10
### Added
- **INT8 Quantization**: Reduced embedding footprint by 4x.
- **LangGraph Checkpointer**: Native support for agentic workflow persistence.
- **LoCoMo Benchmark**: Initial multi-hop relational reasoning validation.

## [0.2.0] - 2026-04-09
### Added
- **Tiered Storage**: Introduction of Hot Tier (RAM) and Cold Tier (Parquet).
- **ACID WAL**: Write-Ahead Log for crash recovery.
- **Knowledge Graph Integration**: First-class support for relational triples.
