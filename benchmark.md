# EpochDB v0.4.1 — Benchmark Results

All benchmarks run end-to-end using **Gemini embedding-2-preview (3072D)**
via the Gemini API. No external corpus or cloud vector database required.

Run the benchmarks yourself:
```bash
# Named suite (LoCoMo · ConvoMem · LongMemEval · NIAH)
venv/bin/python -m benchmarks.run_all

# Capability suite (multi-hop · storage · WAL · NIAH)
venv/bin/python -m benchmarks.run_benchmark
```

---

## Named Benchmark Suite

> `gemini-embedding-2-preview` (3072D) · v0.4.1 Hardened Release

| Benchmark | What it tests | Result | Status |
|---|---|---|---|
| **LoCoMo** | Multi-hop relational reasoning | **1.000** | ✓ PASS |
| **ConvoMem** | Fact correction & preference recall | **1.000** | ✓ PASS |
| **LongMemEval** | Longitudinal recall (cross-epoch) | **1.000** | ✓ PASS |
| **NIAH** | Needle in a Haystack (High-noise) | **1.000** | ✓ PASS |

---

## Operation Latency & Capability Metrics

Detailed performance metrics for core engine operations on the v0.4.1 release.

### 1. Retrieval Latency

| Operation | Tier | Multiplier | Result (ms) |
|---|---|---|---|
| **Semantic Search** | Hot Tier | top_k × 10 | **0.4 ms** |
| **Relational Hop** | Global KG | 1-hop | **0.2 ms** |
| **Historical Recall**| Cold Tier | HNSW Index | **30.0 ms** |

> [!NOTE]
> Cold Tier latency is currently dominated by per-epoch overhead. Transitioning to a unified Global HNSW is planned for v0.5.0 to target < 5ms.

### 2. Stability & Recovery

| Metric | Target | Result |
|---|---|---|
| **WAL Replay** | 3 uncommitted atoms | **9.1 ms** |
| **Data Loss** | System Crash Simulation | **Zero** |
| **Ingest Speed** | 20 atoms (embeddings included) | **~3s** |

### 3. Storage Efficiency (INT8 + Zstd)

20 atoms × 3072D embeddings.

| Metric | Value |
|---|---|
| Raw float32 | 245,760 bytes (240 KB) |
| Compressed (INT8+Zstd) | 47,328 bytes (46 KB) |
| **Compression ratio** | **5.2×** |

---

## Technical Feature Matrix

| Feature | Mechanism | Result |
|---|---|---|
| **Topic Lock** | Factor P (+5.0 boost) | ✓ (Guaranteed Precision) |
| **Entity Seeding** | KG Seeding (Hook) | ✓ (Resolved NIAH Haystack) |
| **Supersession** | Temporal State Filter | ✓ (Corrected Facts Outrank Old) |
| **Tiered Storage** | Parquet + HNSW | ✓ (Hybrid Retrieval) |
| **Saliency** | access_deltas.json | ✓ (Persistent importance) |

---

## Final Verification - 2026-04-13 12:05 UTC

- **precision@3 (NIAH)**: 1.000
- **recall@3 (LongMemEval)**: 1.000
- **multi-hop (LoCoMo)**: 1.000
- **Status**: **Fully Hardened v0.4.1 Ready**
