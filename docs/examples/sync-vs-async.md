# Sync vs. Async Concurrency Benchmark

This benchmark evaluates end-to-end latency and token consumption under concurrent multi-user load, comparing three real-world multi-agent configurations:

1. **Sync LangGraph + Sync EpochDB**: Sequential graph invocation with blocking synchronous I/O.
2. **Async LangGraph + Async EpochDB**: Concurrent graph execution (`ainvoke`) using async checkpointers.
3. **Async Aster + EpochBlackboard**: Decoupled event-driven reactive coordination running in parallel over EpochDB.

---

## Benchmark Scenario

- **Users Evaluated**: 3 concurrent users
- **Turns per User**: 10 conversation turns (30 turns total)
- **Live LLM Model**: Google Gemini API (`gemini-2.0-flash`)
- **Remote Embeddings**: `gemini-embedding-2` (3072 dimensions)

---

## Results Summary

| Metric | Sync LangGraph | Async LangGraph | Async Aster + EpochBlackboard |
| :--- | :---: | :---: | :---: |
| **Total Time (seconds)** | 352.06s | 113.02s | **39.87s** |
| **Average Turn Latency** | 11,735 ms | 3,767 ms | **1,329 ms** |
| **Throughput Speedup** | 1.00x (Baseline) | **3.11x** | **8.83x Faster** |
| **Input Tokens** | 28,385 | 24,145 | **21,932** |

---

## Running the Benchmark Locally

```bash
# Set your Gemini API Key
export GEMINI_API_KEY="your-gemini-key"

# Run the benchmark
python examples/sync_async_benchmark.py
```
