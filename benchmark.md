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

> `gemini-embedding-2-preview` (3072D) · 91 API calls · 84.5s wall time

| Benchmark | What it tests | Score |
|---|---|---|
| **LoCoMo** | Multi-hop relational reasoning across KG chains | **1.000** |
| **ConvoMem** | Conversational recall with preference corrections | **1.000** |
| **LongMemEval** | Longitudinal session memory across 4 epoch checkpoints | **1.000** |
| **NIAH** | Needle in a Haystack (High-noise precision@3) | **1.000** |

### LoCoMo — Multi-Hop Relational Reasoning

3 chains of varying depth (2–4 hops). Queries are deliberately semantically
blank with respect to the terminal node — only KG traversal can resolve them.

| Chain | Target | Hops to resolve | Pass |
|---|---|---|---|
| Sam Altman → OpenAI → Helion | Helion Energy | 2 | ✓ |
| Alice → Aurora → Helios → Dr. Chen | Dr. Chen | 3 | ✓ |
| Marie → BioGen → CRISPR-X | CRISPR-X | 2 | ✓ |

**Aggregate recall: `1.000` (3/3)**

> Flat vector stores score `0.000` on these queries by design. They have no
> relational layer and are structurally incapable of multi-hop traversal.

### ConvoMem — Conversational Memory Recall

5 multi-turn conversations with preference corrections (e.g. Lisbon → Porto,
dog → cat, DataFlow → VectorAI). Each turn stored as a separate atom so the
RRF recency factor correctly ranks the most-recent correction above the
earlier superseded value. Flushed to Cold Tier before evaluation.

**recall@3: `1.000` (5/5)**

### LongMemEval — Longitudinal Session Memory

4 sessions, each flushed to Cold Tier before the next begins. All atoms
in Cold Tier at evaluation time. 2-hop KG expansion enabled.

**recall@3: `1.000` (4/4)**

### Needle in a Haystack — Retrieval Precision

3 signal facts hidden among 50 noise facts (the "Haystack").
Evaluation uses **Entity Hook** seeding to ensure the **Nuclear Topic Lock**
targets the correct memory subgraph.

**precision@3: `1.000` (3/3)**

---

## Capability Suite (Legacy Metrics)

> `gemini-embedding-2-preview` (3072D) · v0.4.1 Validation

### 1. Storage Efficiency — INT8 + Zstd

20 atoms × 3072D embeddings, serialized to Parquet with INT8 scalar
quantization and Zstandard compression.

| Metric | Value |
|---|---|
| Raw float32 | 245,760 bytes (240 KB) |
| INT8 + Zstd (Parquet) | 47,323 bytes (46 KB) |
| **Compression ratio** | **5.2×** |

### 2. WAL Crash Recovery

3 atoms written to WAL without committing. Database re-opened — WAL replay
fires automatically to restore Hot Tier state.

| Metric | Value |
|---|---|
| Atoms recovered | 3/3 |
| **Replay latency** | **11.8 ms** |
| Data loss | **zero** |

### 3. Feature Validation Matrix

| Feature | Test | Result |
|---|---|---|
| Cross-epoch semantic recall | LongMemEval | ✓ |
| Multi-hop KG reasoning | LoCoMo | ✓ |
| Recency-aware fact correction | ConvoMem | ✓ |
| High-noise suppression | NIAH | ✓ (1.000) |
| INT8 + Zstd compression | Storage Bench | ✓ (5.2×) |
| WAL crash recovery | WAL Bench | ✓ (3/3) |
| LangGraph thread persistence | EpochDBCheckpointer | ✓ |
| Persistent saliency deltas | access_deltas.json | ✓ |
