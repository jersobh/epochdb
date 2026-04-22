# Changelog

All notable changes to EpochDB will be documented in this file.

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
