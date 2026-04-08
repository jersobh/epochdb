# EpochDB

**EpochDB** is an agentic memory engine designed for lossless, tiered verbatim storage and multi-hop retrieval.

## Why
I had this idea while playing with LMDB. I wanted to create a memory system that could store conversations in a hybrid way, using in-memory for the most recent conversations and on-disk for older conversations. So, in order to have immutable data, I decided to use Parquet files for the on-disk storage. 

## Overview
Traditional AI memory systems compress conversations through destructive summarization. EpochDB bypasses this constraint by storing "Unified Memory Atoms"—the raw text intrinsically paired with dense embeddings.

EpochDB uses a tiered architecture reminiscent of CPU caching:
1. **L1: Working Memory**: Sub-millisecond HNSW vector index in RAM.
2. **L2: Historical Archive**: Cold storage in immutable, time-partitioned `.parquet` files via PyArrow.

It uniquely handles multi-hop retrieval over time-partitioned data using a **Global Entity Index**.

## How It Works
See [`how_it_works.md`](how_it_works.md) for a detailed technical dive into the architecture.

## Benchmarks & Examples
See [`benchmark.md`](benchmark.md) for traces of EpochDB successfully integrated via `LangGraph`. Check out [`example_langgraph.py`](example_langgraph.py) for the source code.
