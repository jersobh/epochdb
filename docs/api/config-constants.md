# Configuration & Mathematical Constants

This page documents the mathematical parameters and constants governing EpochDB's retrieval, scoring, and storage pipelines.

---

## Retrieval Pipeline Constants

EpochDB relies on mathematically derived constants to guarantee determinism in high-noise retrieval:

| Constant | Value | Purpose |
| :--- | :---: | :--- |
| **`TOPIC_LOCK_BOOST`** | **`+20.0`** | Additive boost awarded to atoms whose entity and predicate match confirmed query intent. Strictly bounds matched candidates far above the maximum theoretical RRF score (~0.12). |
| **`SUPERSEDED_PENALTY`** | **`0.0001x`** | Multiplicative demotion applied to stale atoms when a newer Subject-Predicate value has been committed. |
| **`NOISE_DEMOTION_FACTOR`** | **`1e-7`** | Attenuation multiplier applied to non-locked background candidates once a Topic Lock is confirmed. |
| **`RRF_K`** | **`60`** | Standard Reciprocal Rank Fusion denominator constant to prevent extreme rank sensitivity. |
| **`SEMANTIC_BOOTSTRAP_THRESHOLD`**| **`0.5`** | Minimum cosine similarity required to extract entities from semantic hits when no explicit query entities are passed. |
| **`WEIGHT_SEMANTIC`** | **`3.0`** | RRF weighting for vector cosine similarity rank. |
| **`WEIGHT_RECENCY`** | **`1.0`** | RRF weighting for chronological timestamp rank. |
| **`WEIGHT_ENTITY`** | **`1.0`** | RRF weighting for entity co-occurrence density. |
| **`WEIGHT_QUANTITATIVE`** | **`2.0`** | RRF weighting for scalar range / temporal proximity rank. |

---

## Engine Configuration Options

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `storage_dir` | `str` | `"./.epochdb_data"` | Base directory for WAL, Parquet, and HNSW persistence. |
| `dim` | `int` | Inferred | Dense embedding vector dimensionality. |
| `epoch_duration_secs` | `int` | `3600` | Duration in seconds before Hot Tier atoms are flushed to Cold Parquet. |
| `wal_sync_interval` | `float` | `0.0` | Seconds between `fsync` WAL flushes (`0.0` forces synchronous ACID safety). |
| `cold_search_mode` | `str` | `"probe"` | `"probe"` (selective probing) or `"exhaustive"` (full disk scan). |
| `recency_epochs` | `int` | `2` | Number of newest cold epochs always searched in probe mode. |
| `centroid_probes` | `int` | `2` | Number of additional cold epochs opened based on centroid similarity. |
| `parquet_compression` | `str` | `"zstd"` | Compression format (`"zstd"`, `"snappy"`, `"lz4"`, `"none"`). |
| `parquet_compression_level` | `int` | `3` | Zstandard compression level (1–22). |
| `tenant` | `str` | `None` | Physical subdirectory partition for multi-tenant isolation. |
