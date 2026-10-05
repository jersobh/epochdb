# Quickstart Tutorial

This quickstart guides you through the foundational workflows of EpochDB: creating memories, automatic triple extraction, querying with supersession, and navigating the Knowledge Graph.

---

## 1. Synchronous API (`EpochDB`)

Use `EpochDB` as a context manager to ensure safe thread shutdown and WAL flushing:

```python
from epochdb import EpochDB

# 1. Initialize EpochDB
# embedding_model can be a local model (e.g. "all-MiniLM-L6-v2") or a provider
with EpochDB(storage_dir="./agent_memory", embedding_model="all-MiniLM-L6-v2") as db:

    # 2. Store memories with relational triples
    db.remember(
        "Dr. Marcus Vance founded NeuroLink Corp in 2021.",
        metadata={
            "triples": [
                ("Marcus Vance", "founded", "NeuroLink Corp"),
                ("NeuroLink Corp", "founded_in", "2021"),
            ],
            "category": "corporate",
        }
    )

    # 3. Add an updated fact (supersession in action)
    db.remember(
        "In 2025, Marcus Vance stepped down; Elena Rostova became CEO of NeuroLink Corp.",
        metadata={
            "triples": [
                ("Elena Rostova", "role", "CEO"),
                ("Elena Rostova", "leads", "NeuroLink Corp"),
            ],
            "category": "corporate",
        }
    )

    # 4. Perform a semantic query
    results = db.query("Who leads NeuroLink Corp?", k=1)
    
    for memory in results:
        print(f"Match: {memory.text}")
        print(f"Entities: {memory.entities}")
        print(f"Score: {memory.score:.4f}")
```

---

## 2. Asynchronous API (`AsyncEpochDB`)

For asynchronous agent runtimes (FastAPI, asyncio worker loops, Aster Framework), use `AsyncEpochDB`:

```python
import asyncio
from epochdb import AsyncEpochDB

async def run_agent():
    async with AsyncEpochDB(
        storage_dir="./agent_memory", 
        embedding_model="all-MiniLM-L6-v2"
    ) as db:
        # Asynchronously store an atom
        mem_id = await db.remember(
            "Quantum Processor X1 operates at 15 millikelvin.",
            metadata={"triples": [("Quantum Processor X1", "operates_at", "15mK")]}
        )

        # Non-blocking query
        memories = await db.query("What temperature does the quantum chip require?", k=1)
        print(memories[0].text)

asyncio.run(run_agent())
```

---

## 3. MongoDB-Style Metadata Filtering

Filter search candidates using standard relational operators (`$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, `$lte`):

```python
results = db.query(
    "production incidents",
    k=5,
    filters={
        "author": "Marcus",
        "severity": {"$gte": 3},
        "environment": {"$in": ["staging", "production"]},
    }
)
```

---

## 4. Entity & Graph Traversal

EpochDB models entities as rich Python domain objects:

```python
# Look up an entity directly from the Global Entity Index (GEI)
neurolink = db.get_entity("NeuroLink Corp")

print(f"Entity: {neurolink.name}")
print(f"Mentions across epochs: {len(neurolink.atom_ids)}")

# Traverse 1-hop connected neighbors
for neighbor in neurolink.related():
    print(f" - Connected to: {neighbor.name}")

# Retrieve the chronological timeline of events involving this entity
timeline = neurolink.timeline()
for event in timeline:
    print(f"[{event.created_at}] {event.text}")

# Extract a multi-hop visual graph subgraph (depth=2)
graph = db.entity_graph("NeuroLink Corp", depth=2)
print("Nodes:", graph.nodes)
print("Edges:", graph.edges)
```

---

## 5. Automated Background Triple Extraction

Instead of manually providing triples in metadata, configure EpochDB to extract them automatically:

```python
db = EpochDB(
    storage_dir="./memory",
    embedding_model="all-MiniLM-L6-v2",
    auto_extract=True,                 # Enable automated triple extraction
    extraction_model="hf:small",       # Local HuggingFace REBEL/T5 extractor
    async_extract=True,                # Run extraction in non-blocking worker thread
)

# remember() returns instantly; extraction runs in the background
atom_id = db.remember("Satya Nadella is the executive chairman and CEO of Microsoft.")

# Optional: block until background workers finish
db.wait_for_extractions()

memory = db.get(atom_id)
print("Extracted Triples:", memory.triples)
```

---

## 6. Soft Deletions and Space Compaction

```python
# Soft delete (retained in historical WAL for audits, but filtered from queries)
db.delete(atom_id, hard=False)

# Permanent hard purge & Cold Tier Parquet compaction
db.compact()
```
