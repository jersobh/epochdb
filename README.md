
<p align="center">
  <img
    src="https://raw.githubusercontent.com/jersobh/epochdb/main/logo-epoch.png"
    alt="EpochDB Logo"
    width="180"
  />
</p>

# EpochDB — Agentic Memory Engine 

[![Downloads](https://pepy.tech/badge/epochdb)](https://pepy.tech/project/epochdb)
[![PyPI](https://img.shields.io/pypi/v/epochdb.svg)](https://pypi.org/project/epochdb/)
[![Publish](https://img.shields.io/github/actions/workflow/status/jersobh/epochdb/publish.yml)](https://github.com/jersobh/epochdb/actions/workflows/publish.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/jersobh/epochdb)](https://github.com/jersobh/epochdb/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Versions](https://img.shields.io/pypi/pyversions/epochdb)



**EpochDB** is a high-performance, state-aware memory engine designed for lossless, tiered storage, atomic state management, and multi-hop relational reasoning. It is built specifically for AI agents that require perfect historical recall, long-term state persistence, and deterministic fact corrections.

> [!NOTE]
> **Looking for a distributed, sharded deployment?** Check out the [EpochDB Distributed Server](https://github.com/jersobh/epochdb-server) repository for multi-node clustering, consistent-hashing-based horizontal sharding, and high-concurrency coordinator routing.

---

## Why EpochDB?

Flat vector databases retrieve text based on semantic similarity but struggle to resolve conflicting facts (e.g. *"where does the user work now?"* vs *"where did they work last year?"*). EpochDB solves this through **Atomic State Management**:

- **Topic Lock & Entity Seeding**: Ensures retrieval stays within the target topic by seeding candidates directly from the Knowledge Graph.
- **State-Aware Supersession**: Automatically identifies and filters out stale facts when they are updated.
- **Adaptive Query Routing & Decomposition**: Dynamically routes incoming queries to optimal search engines (semantic, relational, temporal, or quantitative) or splits composite queries using LLMs (Gemini, OpenAI, Anthropic) or local offline rules.
- **Contextualized Retrieval (Temporal neighbor expansion)**: Retrieves chronological context turns immediately surrounding matched memories.
- **Tiered HNSW Hierarchy**: Sub-millisecond recall across working memory (L1 RAM) and historical archives (L2 Disk).
- **Memory Forking & Lineage**: Supports logical branches (`db.fork`) for multi-agent collaboration and hypothetical reasoning without copying data.
- **Pairwise Entity Graph Extraction**: Automatically generates co-occurrence relationship triples between extracted entities for connected graph visualizations.
- **Rich Domain Objects**: Returns structured `Memory`, `Entity`, and `Graph` abstractions rather than raw database tuples.

---

## Architecture

EpochDB uses a tiered hierarchy modeled after CPU caches to balance low latency and massive scale:

```mermaid
graph TD
    Agent([Agent / Application]) -->|remember / add_memory| Engine[EpochDB Engine]

    subgraph "Working Memory — RAM (Hot Tier)"
        Engine --> HNSW_H[HNSW Vector Index]
        Engine --> WAL[WAL: ACID Write-Ahead Log]
        Engine --> KG[Active Knowledge Graph]
    end

    subgraph "Historical Archive — Disk (Cold Tier)"
        HNSW_H -->|Async Flush| Parquet[(Parquet + F32 + Zstd)]
        Parquet --- HNSW_C[HNSW Index per Epoch]
        HNSW_C --- GEI[Global Entity Index]
    end

    subgraph "Retrieval Pipeline"
        HNSW_H --> Pool[Candidate Pool]
        HNSW_C --> Pool
        Pool --> KG_Exp[KG Expansion & Topic Lock]
        KG_Exp --> RRF[4-Way RRF Fusion + Supersession]
        RRF --> Context[Agentic Context]
    end
```

---

## Performance, Latency & Token Efficiency

### 1. The 1.000 Sweep Benchmark
EpochDB achieves a perfect **1.000** score across named benchmark suites designed to validate engine logic:

| Benchmark | What it tests | Metric | Score |
|---|---|---|---|
| **LoCoMo** | Multi-hop relational reasoning | Multi-hop recall | **1.000** |
| **ConvoMem** | Fact correction & preference recall | recall@3 | **1.000** |
| **LongMemEval** | Longitudinal recall (cross-epoch) | recall@3 | **1.000** |
| **NIAH** | Needle in a Haystack (High-noise) | precision@3 | **1.000** |

*Run the benchmark suite locally:*
```bash
venv/bin/python -m benchmarks.run_all
```

### 2. Operational Latency
Precision metrics across Hot and Cold tiers:
- **Direct/Multi-Hop Relational Retrieval (Hot Tier)**: **0.2 ms – 0.4 ms**
- **Historical HNSW Retrieval (Cold Tier)**: **~4.0 ms** (30x speedup from ~125 ms via persistent indexing)
- **Cold Tier Full Scan (`pyarrow.dataset`)**: **45.0 ms** (for cross-epoch scalar aggregations)
- **Scalar Range Query (B-tree)**: **0.8 ms**
- **Series Interpolation (IntervalTree)**: **1.2 ms**
- **Constraint Satisfaction (Z3 SAT Solver)**: **2.5 ms**
- **WAL Crash Recovery Replay**: **9.1 ms**

### 3. LangGraph Token Savings
When used as a checkpointer, EpochDB keeps LangGraph states "thin" by storing historical turns as Unified Memory Atoms and querying them selectively. This achieves **linear $O(N)$ token scaling** (saving **55% to 79%** of input tokens compared to standard checkpointers' quadratic $O(N^2)$ accumulation).

### 4. Sync vs. Async Concurrency Benchmark
This benchmark evaluates E2E latency and input token consumption under concurrent multi-user load, comparing three execution configurations:
1. **Sync LangGraph + Sync EpochDB**: Sequential graph invocation with blocking I/O.
2. **Async LangGraph + Async EpochDB**: Concurrent graph execution (`ainvoke`) using async checkpointers and DB facades.
3. **Async Aster + EpochBlackboard**: Decoupled event-driven reactive coordination running in parallel.

The scenario simulates **3 concurrent users** executing **10 conversation turns each** (30 turns total) over the live Gemini API:

#### Configuration A: Cloud API Embeddings (`gemini-embedding-2` 3072D)
Under this configuration, all text embedding vectors are generated remotely via the Gemini API, requiring sequential HTTP latency for embedding calls.

| Metric | Sync LangGraph | Async LangGraph | Async Aster |
| :--- | :---: | :---: | :---: |
| **E2E Latency (seconds)** | 352.060s | 113.027s | 39.869s |
| **Average Turn Latency** | 11,735.3ms | 3,767.6ms | 1,329.0ms |
| **Throughput Speedup** | 1.00x (Baseline) | **3.11x** | **8.83x** |
| **Total Input Tokens** | 28,385 | 24,145 | 21,932 |

*Key Insight*: Remote embedding requests introduce sequential round-trip latency. In the sync baseline and standard async, these network calls pile up. Aster runs a decoupled event pool which hides network round-trip overhead through maximum event-driven parallelism (yielding **8.83x** speedup).

#### Configuration B: Local Embeddings (`barisaydin/gte-base` 768D)
Under this configuration, all text embedding vectors are generated locally using the SentenceTransformer model loaded in RAM, removing all network latency for embedding calls.

| Metric | Sync LangGraph | Async LangGraph | Async Aster |
| :--- | :---: | :---: | :---: |
| **E2E Latency (seconds)** | 189.905s | 61.752s | 62.276s |
| **Average Turn Latency** | 6,330.2ms | 2,058.4ms | 2,075.9ms |
| **Throughput Speedup** | 1.00x (Baseline) | **3.08x** | **3.05x** |
| **Total Input Tokens** | 24,765 | 22,774 | 25,705 |

*Key Insight*: Eliminating remote embedding API calls slashes E2E latency across all configurations. The async pipeline collapses the blocking network time of LLM text generation down to a single user's duration (~60s), showing a clean **3.08x** speedup.

*Run the benchmark suite locally:*
```bash
.venv/bin/python examples/sync_async_benchmark.py
```

---

## Installation

```bash
pip install epochdb
```

---

## Quickstart

### 1. Synchronous API Facade
```python
from epochdb import EpochDB

# Initialize with auto-embedding
with EpochDB(storage_dir="./memory", embedding_model="all-MiniLM-L6-v2") as db:
    # Store a memory with KG triples
    db.remember("User works at DataFlow.", metadata={"triples": [("user", "works_at", "DataFlow")]})
    
    # Update facts (supersession resolves conflicts)
    db.remember("Actually, user now works at VectorAI.", metadata={"triples": [("user", "works_at", "VectorAI")]})
    
    # Query returns rich Memory objects
    results = db.query("Where does the user work?", k=1)
    print(results[0].text)  # "Actually, user now works at VectorAI."
```

### 2. Asynchronous API Facade
```python
import asyncio
from epochdb import AsyncEpochDB

async def main():
    # Async context manager for non-blocking I/O in agent loops
    async with AsyncEpochDB(storage_dir="./memory", embedding_model="all-MiniLM-L6-v2") as db:
        await db.remember("VectorAI develops CRISPR-X platform.", metadata={"triples": [("VectorAI", "develops", "CRISPR-X")]})
        
        results = await db.query("What does VectorAI build?", k=1)
        print(results[0].text)

asyncio.run(main())
```

### 3. MongoDB-Style Metadata Filtering
```python
# Filter retrieval using operators like $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte
results = db.query(
    "Query text", 
    k=5, 
    filters={
        "author": "Jeff", 
        "importance": {"$gt": 3},
        "category": {"$in": ["development", "production"]}
    }
)
```

### 4. Soft-Delete & Compaction
```python
# Mark memory as deleted (filtered out from queries by default)
db.delete(memory_id, hard=False)

# Reclaim space and deduplicate historical Parquet archives in the Cold Tier
db.compact()
```

### 5. Entity & Graph Traversal
```python
# Retrieve entity object
vector_ai = db.get_entity("VectorAI")

# Traverse relations in Global Entity Index
related = vector_ai.related()  # [Entity("user"), Entity("CRISPR-X")]

# Chronological timeline of the entity
timeline = vector_ai.timeline()

# Generate local graph segment
graph = db.entity_graph("VectorAI", depth=2)
print(graph.nodes)  # ['VectorAI', 'user', 'CRISPR-X']
print(graph.edges)  # List of edge dictionaries mapping sources and targets
```

---

## Client-Server Architecture

EpochDB supports remote deployments via a client-server architecture, allowing multiple agents or server processes to share a single, central database over HTTP.

> [!TIP]
> While the built-in server is ideal for single-node deployments, you can use the [EpochDB Distributed Server](https://github.com/jersobh/epochdb-server) for production environments that require horizontal sharding, multi-node clustering, gateway caching, and consistent hashing.

### 1. Starting the Server (`ThreadingEpochDBServer`)

Start the multi-threaded HTTP server on the host machine to serve an `EpochDB` instance:

```python
from epochdb import EpochDB
from epochdb.api.server import start_server

db = EpochDB(storage_dir="./shared_memory", embedding_model="all-MiniLM-L6-v2")
server = start_server(db, host="0.0.0.0", port=8080)

try:
    server.serve_forever()
finally:
    db.close()
```

### 2. Communicating via the Client (`RemoteEpochDB`)

Use the remote client to execute queries, store memories, and retrieve timelines over HTTP REST (optionally specifying consistency levels like `"one"`, `"quorum"`, or `"all"` for sharded/replicated environments):

```python
from epochdb import RemoteEpochDB

# Initialize the client
client = RemoteEpochDB(host="127.0.0.1", port=8080)

# Store a memory with explicit quorum consistency
client.remember("Pollyanna is married to Jefferson.", consistency="quorum")

# Query the remote database
results = client.query("Who is Pollyanna married to?", k=1)
print(results[0].text)  # "Pollyanna is married to Jefferson."

# Access database stats remotely
stats = client.stats()
print(stats)
```

---

## Multi-Tenant Partitioning & WAL Optimizations

### 1. Multi-Tenant Isolation
For multi-tenant SaaS platforms or isolated agent sessions, EpochDB can physically partition database files on disk using the `tenant` parameter:

```python
# Database files are physically isolated under the "tenants/tenant_alpha" subdirectory
db = EpochDB(storage_dir="./app_data", tenant="tenant_alpha")
```

### 2. Configurable WAL Sync Interval
By default, the Write-Ahead Log (WAL) synchronously forces an `fsync` call to disk on every transaction append, ensuring zero data loss but limiting write throughput. You can speed up writes dramatically by configuring asynchronous background syncing:

```python
# Sync the WAL file to disk asynchronously every 0.1 seconds in a background thread
db = EpochDB(storage_dir="./memory", wal_sync_interval=0.1)
```

### 3. Parquet Compression Configuration
When serializing working memory from the Hot Tier (RAM) to the Cold Tier (disk Parquet files), you can define the compression algorithm and level:

```python
# Configure Zstandard compression (level 3) for disk archives
db = EpochDB(storage_dir="./memory", parquet_compression="zstd", parquet_compression_level=3)
```

Supported methods include `"zstd"`, `"snappy"`, `"lz4"`, `"gzip"`, `"brotli"`, and `"none"` (defaulting to `"zstd"` with level `3`).

### 4. High-Performance io_uring WAL
On Linux hosts, EpochDB automatically compiles and loads a C shared library to write WAL appends through `io_uring` and Direct I/O (`O_DIRECT`), bypassing the kernel page cache and system call scheduling overhead to deliver up to **5x speedups** on synchronous operations with natural queue backpressure safety.

### 5. DuckDB SQL Analytics over Cold Tier Archives

EpochDB integrates with **DuckDB** to allow executing high-performance vectorized SQL queries over historical memory archives (`*.parquet`). The `cold_tier` view is registered automatically:

```python
# Execute vectorized SQL aggregations over all historical Parquet archives
results = db.query_sql("""
    SELECT 
        COUNT(*) as total_memories,
        AVG(scalar_value) as avg_value
    FROM cold_tier
    WHERE scalar_unit = 'degC'
""")
```

---

## LangGraph Integration

EpochDB provides native checkpointer support for both synchronous and asynchronous workflows:

```python
from epochdb.checkpointer import EpochDBCheckpointer
from epochdb import EpochDB

# Synchronous compile
with EpochDB(storage_dir="./agent_state") as db:
    checkpointer = EpochDBCheckpointer(db)
    app = workflow.compile(checkpointer=checkpointer)
```

For async runtimes:
```python
from epochdb import AsyncEpochDB
from epochdb.checkpointer import EpochDBCheckpointer

async def run_agent():
    async with AsyncEpochDB(storage_dir="./agent_state") as db:
        checkpointer = EpochDBCheckpointer(db)
        app = workflow.compile(checkpointer=checkpointer)
        # Uses aput, aget_tuple, and alist internally under the hood
```

---
## Configuring Embedding Providers

EpochDB supports multiple local and cloud embedding providers:

1. **Local Offline Models (Default)**: Pass any SentenceTransformer model name (e.g. `"all-MiniLM-L6-v2"`).
   ```python
   db = EpochDB(storage_dir="./memory", model="all-MiniLM-L6-v2")
   ```
2. **OpenAI & Compatible APIs**: Use the `openai:` prefix.
   - Requires `OPENAI_API_KEY` set in your environment.
   - You can optionally set `OPENAI_BASE_URL` to route requests to local proxies (e.g., vLLM, LM Studio, Ollama) or compatible cloud endpoints (e.g. Voyage AI, Cohere).
   - Ensure the `dim` parameter matches your target dimensions (e.g., `1536` for `text-embedding-3-small` or any custom dimension supported by the model).
   ```python
   db = EpochDB(storage_dir="./memory", model="openai:text-embedding-3-small", dim=1536)
   ```
3. **Google / Gemini API**: Use the `google:` prefix.
   - Requires `GEMINI_API_KEY` set in your environment.
   ```python
   db = EpochDB(storage_dir="./memory", model="google:text-embedding-004", dim=768)
   ```
4. **Ollama Local Service**: Use the `ollama:` prefix.
   - Requires a running local Ollama service.
   ```python
   db = EpochDB(storage_dir="./memory", model="ollama:all-minilm", dim=384)
   ```

## Repository Structure

The codebase is modularized to isolate engine subsystems:

* [`core/`](epochdb/core): Core transactions, checkpointers, and base units.
* [`storage/`](epochdb/storage): Hot Tier (RAM HNSW) and Cold Tier (Parquet storage).
* [`entities/`](epochdb/entities): Global KG manager, cascade updates, and reflection rules.
* [`retrieval/`](epochdb/retrieval): Multi-stage retrieval managers, quantitative indexes, and RRF fusion.
* [`api/`](epochdb/api): Public facade APIs (`EpochDB` and `AsyncEpochDB`) and domain objects (`Memory`, `Entity`, `Graph`).

---

## Technical Specifications & Constants

- **`+20.0` Topic Lock Boost**: Set mathematically larger than the maximum possible Reciprocal Rank Fusion (RRF) score sum (which caps at $\approx 0.05$ across semantic and recency ranks, using $K=60$). This acts as a "hard lock," ensuring query-intent-matched facts always outrank adjacent semantic noise.
- **`0.0001x` Supersession Penalty**: Multiplicatively demotes stale facts (e.g. older conflicting values for the same subject-predicate pair) to the bottom of the retrieval pool, resolving contradictions deterministically while preserving database history.
- **`1e-7` Signal-to-Noise Demotion**: Once a Topic-Locked fact is identified, all non-locked background noise is demoted by $10^{-7}$ to keep the LLM's context window clean and free from distractors.
- **Quantitative logic & Triggers**: Native support for Scalars, Time-Series, and Constraints. `IntervalTree` enables precise $O(\log n + k)$ range queries with base-unit normalization via persistent `schema_registry.json`.
- **Reactive Cascade Graphs**: `CascadeManager` automatically triggers downstream policy updates, while Coefficient of Variation (CV) reflections auto-generate constraint atoms from observed historical data trends.
- **Analytical Cold Tier**: Leveraging `pyarrow.dataset` for high-performance cross-epoch scanning and numeric aggregation directly over compressed Parquet archives.
- **ACID Crash Recovery**: Zero data loss for in-flight memories via the synchronous Write-Ahead Log.

---

## License

MIT — see [`LICENSE`](LICENSE).
