# Knowledge Graph & Entity Traversal

EpochDB embeds an active Knowledge Graph (KG) subsystem directly into the storage engine. Instead of requiring a separate graph database (such as Neo4j) alongside a vector database, EpochDB co-locates relational triples and dense vectors in the same persistent atom records.

---

## The Dual Graph Layer

```
EpochDB Graph Subsystem
 ├── Active In-Memory KG (Hot Tier)
 │    ├── Adjacency mappings (source -> relation -> target)
 │    └── Entity-to-atom indices for instantaneous BFS
 │
 ├── Global Entity Index / GEI (Cold Tier)
 │    ├── Inverted index mapping entity names to Cold Parquet partition files
 │    └── Entity metadata and centroid associations
 │
 └── Persistent Relational Store
      └── SQLite relational tables for cross-epoch multi-hop durability
```

---

## Entity Domain Objects

When you look up an entity via `db.get_entity(name)`, EpochDB returns a rich `Entity` domain object:

```python
from epochdb import EpochDB

with EpochDB(storage_dir="./memory") as db:
    # Fetch entity abstraction
    dr_vance = db.get_entity("Marcus Vance")

    print(f"Name: {dr_vance.name}")
    print(f"Associated Atoms: {dr_vance.atom_ids}")

    # Traverse immediate relational neighbors
    neighbors = dr_vance.related()
    for entity in neighbors:
        print(f" -> Related: {entity.name}")

    # Chronological history of actions/events involving this entity
    events = dr_vance.timeline()
    for ev in events:
        print(f"[{ev.created_at}] {ev.text}")
```

---

## Multi-Hop Graph Traversal (`entity_graph`)

To explore complex relational paths across multiple hops, use `db.entity_graph()`:

```python
# Generate a local graph centered on "NeuroLink Corp" up to 2 hops away
graph = db.entity_graph("NeuroLink Corp", depth=2)

print("Nodes in Subgraph:")
for node in graph.nodes:
    print(f" - {node}")

print("\nDirected Edges:")
for edge in graph.edges:
    print(f" - {edge['source']} --[{edge['relation']}]--> {edge['target']}")
```

### JSON-Serialisable Structure for Frontend Visualizers
The `Graph` object returned can be converted directly to dictionaries formatted for visualization libraries like **vis.js**, Cytoscape, or D3:

```python
graph_dict = graph.to_dict()
# {
#   "nodes": [{"id": "NeuroLink Corp", "label": "NeuroLink Corp"}, ...],
#   "edges": [{"from": "Marcus Vance", "to": "NeuroLink Corp", "label": "founded"}, ...]
# }
```

---

## Pairwise Entity Extraction

When documents contain multiple entities without explicit triples, EpochDB's extraction subsystem can automatically extract **pairwise entity co-occurrence triples**:

```
Text: "Satya Nadella, Sam Altman, and Jensen Huang discussed AI data centers in San Jose."
                   │
                   ▼ (Automatic Triple Generation)
(Satya Nadella, discusses_with, Sam Altman)
(Sam Altman, discusses_with, Jensen Huang)
(Satya Nadella, located_in, San Jose)
(Sam Altman, located_in, San Jose)
(Jensen Huang, located_in, San Jose)
```

This creates a highly interconnected semantic mesh that empowers the 5-stage retrieval pipeline to execute multi-hop reasoning with high recall.
