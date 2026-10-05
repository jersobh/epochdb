# Installation & Setup

EpochDB is published on PyPI and can be installed across Linux, macOS, and Windows. It supports Python 3.10 and newer.

---

## Basic Installation

For core runtime operations (Hot/Cold tiered storage, HNSW vector search, ACID WAL, Knowledge Graph, Interval Trees, and Z3 SAT solving):

::: code-group
```bash [pip]
pip install epochdb
```

```bash [poetry]
poetry add epochdb
```

```bash [uv]
uv add epochdb
```
:::

---

## Optional Feature Bundles

EpochDB uses modular optional dependencies to keep installation lightweight:

```bash
# Complete installation with all optional features
pip install "epochdb[all]"
```

### Specific Feature Extras

| Extra | Command | Included Libraries | Use Case |
| :--- | :--- | :--- | :--- |
| **`all`** | `pip install "epochdb[all]"` | Everything below | Full-featured autonomous agent setups |
| **`embeddings`** | `pip install "epochdb[embeddings]"` | `sentence-transformers` | Automated local embeddings on write/query |
| **`extraction`** | `pip install "epochdb[extraction]"` | `transformers`, `torch` | Local model-based KG triple extraction (REBEL) |
| **`langgraph`** | `pip install "epochdb[langgraph]"` | `langgraph` | Native `EpochDBCheckpointer` integration |
| **`google`** | `pip install "epochdb[google]"` | `google-genai` | Gemini cloud embeddings and LLM routing |
| **`duckdb`** | `pip install "epochdb[duckdb]"` | `duckdb` | Vectorized SQL analytics over historical archives |
| **`mcp`** | `pip install "epochdb[mcp]"` | `mcp[cli]` | Model Context Protocol server CLI (`epochdb-mcp`) |

---

## System Requirements & Native Libraries

EpochDB relies on optimized C/C++ backends for speed:
- **`hnswlib`**: Fast approximate nearest-neighbor search. Precompiled wheels are available for most platforms.
- **`pyarrow`**: High-performance Apache Arrow / Parquet serialization.
- **`z3-solver`**: Microsoft Z3 theorem prover for symbolic SAT constraints.
- **`Rtree`**: Spatial indexing for temporal series. (On Debian/Ubuntu systems, install `libspatialindex-dev` if building from source).

### High-Performance `io_uring` on Linux (Optional)
On Linux kernels 5.1+, EpochDB detects and automatically compiles a C shared library leveraging `io_uring` and Direct I/O (`O_DIRECT`). This bypasses kernel page cache overhead for up to **5x faster synchronous WAL appends**. If `io_uring` headers are not present, EpochDB falls back to POSIX `write()` with zero configuration needed.

---

## Verification

Run a quick inline Python verification:

```python
import epochdb
from epochdb import EpochDB

print(f"EpochDB version: {epochdb.__file__}")

with EpochDB(storage_dir="./.test_verify") as db:
    atom_id = db.remember("Verification memory atom test.")
    results = db.query("Verification", k=1)
    assert len(results) == 1
    print("EpochDB initialized and queried successfully!")
```

If you see `EpochDB initialized and queried successfully!`, your installation is complete and ready.
