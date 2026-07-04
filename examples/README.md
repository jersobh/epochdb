# EpochDB Examples & Benchmarks

This directory contains practical examples and benchmarks showing how to run, configure, and evaluate EpochDB.

## Running Locally (Offline Mode) vs LLM Mode
EpochDB is fully functional **offline** and does not require external API keys. It defaults to a small local embedding model and heuristic-based entity/rule parsing. 
- **Offline / Local Mode**: No configuration needed.
- **LLM Mode (Optional)**: If you want to use advanced LLM fact extraction and routing, define your API keys in a `.env` file or environment:
  ```bash
  export GEMINI_API_KEY="your-key"
  # or
  export OPENAI_API_KEY="your-key"
  # or
  export ANTHROPIC_API_KEY="your-key"
  ```

---

## Code Examples

### 1. Basic & Advanced Engine Demos
- **[demo.py](demo.py)**: Showcases basic memory storage, retrieval, and updating.
- **[example_advanced.py](example_advanced.py)**: Explores state-aware supersession, discrete topic locking, and memory timeline queries.
- **[quantitative_demo.py](quantitative_demo.py)**: Demonstrates quantitative indexing, scalar range queries, interval trees, and Z3 constraint checks.

### 2. Integration with Agent Frameworks
- **[example_langchain.py](example_langchain.py)**: Uses EpochDB as a long-term vector store and KG memory inside a LangChain agent.
- **[example_langgraph.py](example_langgraph.py)**: Implements EpochDB as a native LangGraph checkpointer, thin-state compiler, and conversation restorer.

---

## Evaluation & Benchmarks

### 1. External System vs EpochDB Benchmark
- **[benchmark_external_vs_epochdb.py](benchmark_external_vs_epochdb.py)**: Compares EpochDB and a simulated external database-backed memory system across latency, token consumption, and retrieval accuracy:
  - **Latency**: Measures EpochDB's local sub-millisecond retrieval vs the external system's multi-database roundtrips.
  - **Token Scaling**: Measures how Contextualized Retrieval (turns expansion) affects token consumption over a conversation timeline.
  
  *Run the benchmark:*
  ```bash
  python3 examples/benchmark_external_vs_epochdb.py
  ```

### 2. General Verification & Performance
- **[sync_async_benchmark.py](sync_async_benchmark.py)**: Compares sync vs async client concurrency under load.
- **[benchmark_token_efficiency_accuracy.py](benchmark_token_efficiency_accuracy.py)**: Verifies recall accuracy and inputs/outputs token savings in production agent loops.
