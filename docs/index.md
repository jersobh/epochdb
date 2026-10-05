---
layout: home

hero:
  name: "EpochDB"
  text: "Agentic Memory Engine"
  tagline: "Lossless, tiered memory architecture with atomic state supersession, multi-hop knowledge graph reasoning, and neuro-symbolic verification."
  image:
    src: /logo.png
    alt: EpochDB Logo
  actions:
    - theme: brand
      text: Get Started
      link: /guide/introduction
    - theme: alt
      text: Architecture
      link: /guide/architecture
    - theme: alt
      text: View on GitHub
      link: https://github.com/jersobh/epochdb

features:
  - icon: 🧠
    title: Lossless Verbatim Atoms
    details: Replaces destructive LLM summarization with high-fidelity verbatim memory atoms paired with high-precision float32 dense embeddings.
  - icon: ⚡
    title: Two-Tier Memory Hierarchy
    details: Sub-millisecond working memory in RAM (Hot Tier HNSW + WAL) coupled with compressed Parquet archives and persistent epoch HNSW indexes on disk (Cold Tier).
  - icon: 🎯
    title: Atomic State & Topic Lock
    details: Eliminates hallucinations and temporal contradictions through deterministic Subject-Predicate supersession (+20.0 Topic Lock and 0.0001x staleness penalty).
  - icon: 🕸️
    title: Multi-Hop Knowledge Graph
    details: Native pairwise entity extraction, Global Entity Indexing (GEI), and BFS relationship traversal for complex multi-step reasoning.
  - icon: 🛡️
    title: Neuro-Symbolic Verification (LCAG)
    details: Opt-in two-phase propose commits where deterministic symbolic validators approve state transitions before atomic WAL/HNSW/KG persistence.
  - icon: 🚀
    title: First-Class Orchestration
    details: Seamlessly integrates as the memory substrate for Aster Framework (OABS) and provides LangGraph checkpointers with 55%–79% token savings.
---

## Quick Architecture Preview

```mermaid
graph TD
    Agent([Autonomous Agent / LLM]) -->|remember / propose| Engine[EpochDB Engine]
    
    subgraph "Hot Tier — RAM (Working Memory)"
        Engine --> HNSW_H[HNSW Vector Index (Sub-ms Recall)]
        Engine --> WAL[WAL: ACID Write-Ahead Log]
        Engine --> KG[Active Knowledge Graph]
    end

    subgraph "Cold Tier — Disk (Historical Archives)"
        HNSW_H -->|Async Flush| Parquet[(Parquet + Float32 + Zstd)]
        Parquet --- HNSW_C[Epoch HNSW Indexes]
        HNSW_C --- GEI[Global Entity Index]
        HNSW_C --- Centroids[Centroid Probing]
    end

    subgraph "Retrieval Engine"
        HNSW_H --> Pool[Candidate Pool]
        HNSW_C --> Probe[Epoch Probe: recency + GEI + centroids]
        Probe --> Pool
        Pool --> KG_Exp[KG Expansion & Topic Lock]
        KG_Exp --> RRF[4-Way RRF Fusion + Supersession]
        RRF --> Context[Precision Agent Context]
    end
```

## Quick Installation

::: code-group
```bash [pip]
pip install epochdb
```

```bash [pip (all extras)]
pip install "epochdb[all]"
```

```bash [poetry]
poetry add epochdb
```
:::

## 30-Second Example

```python
from epochdb import EpochDB

# Initialize with automated embeddings
with EpochDB(storage_dir="./memory", embedding_model="all-MiniLM-L6-v2") as db:
    # 1. Store structured facts
    db.remember("Alice joined VectorAI as Lead Engineer.", 
                metadata={"triples": [("Alice", "role", "Lead Engineer"), ("Alice", "works_at", "VectorAI")]})

    # 2. Update state — supersession automatically handles updates
    db.remember("Alice transitioned to Chief Architect at VectorAI.",
                metadata={"triples": [("Alice", "role", "Chief Architect")]})

    # 3. Retrieve ground truth
    memories = db.query("What is Alice's current role?", k=1)
    print(memories[0].text)
    # Output: "Alice transitioned to Chief Architect at VectorAI."
```
