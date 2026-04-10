# Changelog

All notable changes to **EpochDB** will be documented in this file.

## [0.2.2] - 2026-04-10
### Fixed
- **Storage Race Condition**: Epoch IDs now utilize UUID-based hashing (`epoch_uuid8`) instead of pure timestamps. This prevents critical data erasure (overwrites) during rapid-fire checkpoints in high-ingestion agentic workloads.
- **Longitudinal Recall Optimization**: Resolved a regression in the `LongMemEval` benchmark. Historical facts are no longer outranked by recency bias through expanded candidate depth and refined RRF constants.

### Changed
- **Retrieval Depth**: Increased internal semantic candidate hook depth to `top_k * 3` during the expansion phase to ensure robust retrieval of older memories.

## [0.2.1] - 2026-04-10
### Added
- **Zstandard Compression**: Integrated `Zstd` for Parquet archives. When combined with INT8 quantization, this provides an enterprise-grade compression ratio for vector-heavy datasets.

## [0.2.0] - 2026-04-09
### Added
- **Asynchronous Tiering**: Implemented daemon-threaded background flushes. The engine no longer blocks the main application loop during Hot-to-Cold tier transitions.
- **INT8 Scalar Quantization**: Native down-casting for embeddings in the Cold Tier, providing a 4x reduction in storage footprint with negligible recall loss.
- **Hybrid RRF Ranking**: Implemented Reciprocal Rank Fusion (RRF) to unify semantic similarity and access-based saliency scores.
- **Semantic Edge Filtering**: Configurable cosine-similarity thresholds for Knowledge Graph traversal (default `> 0.15`) to prevent "super-node" expansion noise.
- **LangGraph Checkpointer**: Native `EpochDBCheckpointer` for persisting agentic graph states and thread history.

## [0.1.3] - 2026-04-08
### Added
- **ACID Write-Ahead Log (WAL)**: Ensures memory atom safety across crashes.
- **Global Entity Index (GEI)**: Multi-epoch relational expansion bridging time-partitioned Parquet files.

## [0.1.0] - 2023-11-20
### Added
- Initial release with Tiered Vector Storage (HNSW + Parquet).
