# Changelog

All notable changes to EpochDB will be documented in this file.

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
