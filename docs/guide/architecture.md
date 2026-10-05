# Architecture Overview

EpochDB operates as a two-tier hybrid engine that fuses dense vector indexing, graph relational reasoning, transactional logging, and symbolic rule verification into a coherent system.

---

## High-Level Architecture Diagram

```mermaid
graph TD
    Client([Autonomous Agent / LangGraph / Aster])
    
    subgraph "Ingestion & Transaction Layer"
        Client -->|remember / direct write| HotEngine[EpochDB Engine]
        Client -->|propose intent| Staging[Pending Staging Area]
        Staging --> Validator[Symbolic Validator Suite (LCAG)]
        Validator -->|APPROVED| HotEngine
        Validator -->|REJECTED| Reject([Raise ValidationError])
        HotEngine --> WAL[ACID Write-Ahead Log (WAL)]
    end

    subgraph "Hot Tier — L1 Working Memory (RAM)"
        HotEngine --> RAM_Atoms[Unified Memory Atoms Store]
        HotEngine --> Hot_HNSW[Hot HNSW Vector Index]
        HotEngine --> Active_KG[Active Knowledge Graph]
        HotEngine --> Quant_Index[IntervalTree / R-Tree]
    end

    subgraph "Cold Tier — L2 Historical Archive (Disk)"
        RAM_Atoms -->|Periodic / Size-Triggered Flush| Parquet[(Parquet Files: F32 + Zstd)]
        Parquet --- Cold_HNSW[HNSW Index per Epoch]
        Cold_HNSW --- GEI[Global Entity Index (GEI)]
        Cold_HNSW --- Centroids[Epoch Mean Centroids]
    end

    subgraph "5-Stage Retrieval Subsystem"
        Hot_HNSW --> Hook[Parallel Semantic Hook]
        Cold_HNSW -.->|Probed Epochs: Recency + GEI + Centroids| Hook
        Hook --> Boot[Semantic Bootstrapping]
        Boot --> Seed[Global KG Seeding (Topic Lock)]
        Seed --> Expansion[Relational Expansion (N-Hops)]
        Expansion --> Fusion[4-Way RRF Fusion & Supersession Engine]
        Fusion --> Output([Contextualized Memory Atoms])
    end
```

---

## System Subsystems

### 1. Unified Atom Store (`core/atom.py`)
The fundamental unit of persistence is the `UnifiedMemoryAtom`:
- **`atom_id`**: Universally unique monotonic identifier.
- **`text`**: Lossless, verbatim string.
- **`vector`**: Dense embedding representation (`float32` numpy array).
- **`metadata`**: Flexible JSON dictionary (authorship, tags, confidence, timestamps).
- **`triples`**: Structured relationship tuples `(Subject, Predicate, Object)`.
- **`superseded_by`**: Optional pointer to an atom that updated this fact.
- **`epoch_id`**: Partition identifier for cold serialization.

### 2. Transactional Write-Ahead Log (WAL) (`storage/wal.py`)
EpochDB maintains 100% crash durability:
- Every mutation is appended and synchronized to disk (`fsync` or `io_uring`) before being added to RAM.
- If the operating system or host machine crashes, EpochDB replays the WAL on startup, fully reconstructing the Hot Tier and Active KG in **< 10ms**.
- WAL sync intervals can be tuned (`wal_sync_interval=0.0` for synchronous safety, or `0.1` for high-throughput batching).

### 3. Hot Tier (`storage/hot_tier.py`)
Working memory is held in RAM:
- In-memory dict for $O(1)$ atom lookup by ID.
- `hnswlib.Index` for sub-millisecond approximate nearest neighbor search over active session vectors.
- Active Graph adjacency maps for instantaneous 1-hop and 2-hop traversals.

### 4. Cold Tier (`storage/cold_tier.py`)
Historical memories are flushed into structured disk archives:
- Stored as Apache Parquet partitions using **Float32 precision** and **Zstandard (`zstd`) level 3 compression**.
- Each epoch archive has its own dedicated `.hnsw` index file.
- The **Global Entity Index (GEI)** maintains a persistent inverted index mapping every entity name to the exact Parquet files containing its mentions.
- **Epoch Centroids**: Every epoch computes a vector centroid (mean embedding) to allow fast geometric pruning of irrelevant historical eras.

### 5. Neuro-Symbolic Staging Area (`validation/`)
When validators are registered, the write path bifurcates:
- `db.remember()` remains the zero-overhead fast path.
- `db.propose()` routes payloads into an in-memory `Pending` staging queue. Registered `SymbolicValidator` instances inspect the candidate text, metadata, and an immutable snapshot of the Active KG.
- Only upon unanimous `ValidationStatus.APPROVED` is the proposal committed atomically to the WAL, HNSW index, and Active KG.

---

## Write Path Lifecycle

```
[Agent Input]
     │
     ├── db.remember() ──────────────┐
     │                               │
     └── db.propose()                │
              │                      │
     [Stage in PENDING]              │
              │                      │
     [Run Symbolic Validators]       │
              │                      │
         [APPROVED?]                 │
          ├── NO  ──> Raise Error    │
          └── YES ───────────────────┤
                                     ▼
                        [Append to WAL (ACID sync)]
                                     │
                        [Insert into Hot HNSW]
                                     │
                        [Index Triples in Active KG]
                                     │
                        [Assign Atom ID & Return]
```

---

## Query Path Lifecycle

```
[Query Input]
     │
     ├─ 1. Parallel Semantic Hook (Hot RAM + Probed Cold Epochs)
     │
     ├─ 2. Semantic Bootstrapping (Extract top entities from semantic hits)
     │
     ├─ 3. Topic Lock & GEI Seeding (Fetch all atoms anchored to target entities)
     │
     ├─ 4. Relational Expansion (Traverse graph edges N-hops outward)
     │
     └─ 5. 4-Way RRF Fusion & Supersession (+20.0 Topic Lock, 0.0001x Staleness Demotion)
              │
              ▼
    [Top-K Ranked Verbatim Memory Atoms]
```
