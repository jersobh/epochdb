# Performance & Benchmarks

EpochDB has been benchmarked across both standardized relational evaluation suites and high-concurrency production load scenarios.

---

## 1. Official retrieval benchmarks (evidence-aware)

These numbers are from the **official** LoCoMo / ConvoMem / LongMemEval / NIAH datasets. The metric is whether gold answer or evidence text appears in the returned atoms (retrieval), **not** LLM-judged answer accuracy. Tiny synthetic smoke suites can still score 1.000 — do not confuse them with the table below.

### Full suite — 2026-11-13

`jersobh/epochdb-embed` (384D) · extraction=`local` · seed=`42` · `--official --limit 50`

| Benchmark | Focus Area | Metric | Score |
| :--- | :--- | :--- | :---: |
| **LoCoMo-10** | Multi-hop conversational QA | recall@20 | **0.863** |
| **ConvoMem** (Salesforce) | Preference / correction recall | recall@10 | **0.680** |
| **LongMemEval_S** | Longitudinal session haystacks | recall@10 | **0.860** |
| **NIAH** | Needle among distractors | precision@3 | **1.000** |

### Precision-first stack (v1.14)

Wide internal candidate pool → precision rerank → **small returned k** (token-efficient for LLM prompts):

| Benchmark | Metric | Score | Notes |
| :--- | :--- | :---: | :--- |
| **LoCoMo-10** | recall@50 | **0.955** | 1472/1542 QA · seed 42 |
| **ConvoMem** | recall@5 | **0.80** | n=50 · query decompose + elite seats |
| **ConvoMem** | recall@10 | **0.86** | same run |
| **ConvoMem** | recall@20 | **1.00** | coverage ceiling at k=20 |

What moved the needle: BM25 stemming, per-subquery normalized multi-query BM25, cold-tier BM25, precision fusion, and decoupling the internal pool from returned `top_k`.

*Run the suite locally:*
```bash
python -m benchmarks.run_all --official --limit 50 --extraction-model local --seed 42
```

---

## 2. Operational Latency

Measured across 10,000 active atoms with 768-dimensional float32 dense vectors on an AMD Ryzen 9 / PCIe Gen 4 NVMe:

| Operation | Tier / Medium | Latency | Speedup Factor |
| :--- | :--- | :---: | :--- |
| **Direct / Relational Lookup** | Hot Tier (RAM) | **0.2 ms – 0.4 ms** | Sub-millisecond |
| **Historical HNSW Query** | Cold Tier (Disk Index) | **~4.0 ms** | 30x faster than linear scan |
| **Scalar Range Search** | Interval Tree | **0.8 ms** | Logarithmic search |
| **Constraint Satisfaction** | Z3 SAT Solver | **2.5 ms** | Real-time guardrail |
| **DuckDB Columnar Aggregation** | Cold Tier Parquet | **12.0 ms** | Vectorized SIMD |
| **PyArrow Full Cold Scan** | Unindexed Parquet | **45.0 ms** | Cross-epoch baseline |
| **WAL Crash Recovery Replay** | Disk to RAM | **9.1 ms** | Full 10k atom state rebuild |

---

## 3. Multi-User Concurrency Benchmark

This benchmark evaluates end-to-end latency and token consumption under concurrent multi-user load, simulating **3 concurrent users** executing **10 conversation turns each** (30 turns total) over the live Gemini API:

### Configuration A: Cloud API Embeddings (`gemini-embedding-2` 3072D)

| Metric | Sync LangGraph | Async LangGraph | Async Aster + EpochBlackboard |
| :--- | :---: | :---: | :---: |
| **End-to-End Latency** | 352.060s | 113.027s | **39.869s** |
| **Average Turn Latency** | 11,735.3 ms | 3,767.6 ms | **1,329.0 ms** |
| **Throughput Speedup** | 1.00x (Baseline) | **3.11x** | **8.83x Faster** |
| **Total Input Tokens** | 28,385 | 24,145 | **21,932** |

> **Key Insight**: Cloud embedding requests introduce round-trip network delays. Aster's decoupled event pool runs asynchronously with EpochDB's non-blocking I/O, hiding remote network overhead to achieve an **8.83x speedup**.

---

### Configuration B: Local Offline Embeddings (`barisaydin/gte-base` 768D)

| Metric | Sync LangGraph | Async LangGraph | Async Aster + EpochBlackboard |
| :--- | :---: | :---: | :---: |
| **End-to-End Latency** | 189.905s | 61.752s | **62.276s** |
| **Average Turn Latency** | 6,330.2 ms | 2,058.4 ms | **2,075.9 ms** |
| **Throughput Speedup** | 1.00x (Baseline) | **3.08x** | **3.05x** |
| **Total Input Tokens** | 24,765 | 22,774 | **25,705** |

---

## 4. Token Efficiency: Linear $O(N)$ vs Quadratic $O(N^2)$

When used as a long-term memory backend for LangGraph or Aster, EpochDB saves **55% to 79%** of cumulative input tokens:

```
Token Consumption over 50 Conversation Turns:

Tokens
  ▲
  │                                    Standard Checkpointer O(N^2)
  │                                           /
  │                                         /
  │                                       /
  │                                     /
  │  ─────────────────────────────────/  EpochDB Selective Retrieval O(N)
  │
  └──────────────────────────────────────────────────────────►
  0                                                         50 Turns
```

- Standard checkpointers append message histories endlessly, re-submitting turns 1 to 49 on turn 50.
- EpochDB stores the history losslessly in its Parquet and HNSW tiers, but injects only the top-$k$ Topic-Locked facts into the context, keeping turn token payloads flat.
