# EpochDB

**EpochDB** is an **ACID-compliant** agentic memory engine designed for lossless, tiered verbatim storage and multi-hop retrieval.

## Why
I had this idea while playing with LMDB. I wanted to create a memory system that could store conversations in a hybrid way, using in-memory for the most recent conversations and on-disk for older conversations. So, in order to have immutable data, I decided to use Parquet files for the on-disk storage. 

## Overview
Traditional AI memory systems compress conversations through destructive summarization. EpochDB bypasses this constraint by storing "Unified Memory Atoms"—the raw text intrinsically paired with dense embeddings.

EpochDB uses a tiered architecture reminiscent of CPU caching:
1. **L1: Working Memory**: Sub-millisecond HNSW vector index in RAM.
2. **L2: Historical Archive**: Cold storage in immutable, time-partitioned `.parquet` files via PyArrow.

It uniquely handles multi-hop retrieval over time-partitioned data using a **Global Entity Index**.

## Performance & Comparison

EpochDB is engineered specifically for **Agentic workflows** where logical continuity across long horizons is critical. In side-by-side benchmarks against industry standards, EpochDB remains the only local engine capable of complex multi-hop reasoning.

| Benchmark | Store | Metrics | Note |
| :--- | :--- | :--- | :--- |
| **LoCoMo** | **EpochDB** | **recall: 1.000** | **100% Multi-hop Accuracy** |
| | ChromaDB | recall: 0.000 | Failed to connect related events |
| | Qdrant | recall: 0.000 | Failed to connect related events |
| **ConvoMem**| EpochDB | recall@3: 1.000 | Perfect Semantic retrieval |
| | FAISS | recall@3: 1.000 | Perfect Semantic retrieval |

> [!IMPORTANT]
> **Relational Expansion**: While tools like FAISS and ChromaDB are excellent for single-turn semantic search, they act as "flat" stores. EpochDB leverages its integrated **Knowledge Graph** to bridge logical gaps, successfully navigating multi-hop queries where competitors fail completely.

## How It Works
See [`how_it_works.md`](how_it_works.md) for a detailed technical dive into the architecture.

## Benchmarks & Examples
See the full comparison in [`benchmark.md`](benchmark.md). Check out [`example_langgraph.py`](example_langgraph.py) for a real-world agent integration example.
