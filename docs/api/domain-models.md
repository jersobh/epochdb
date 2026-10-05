# Domain Models (Memory, Entity, Graph)

EpochDB returns high-level, strongly-typed Python domain objects rather than unstructured tuples or raw database rows.

---

## 1. `Memory`

Represents an individual retrieved memory atom enriched with ranking signals and entity connections.

```python
class Memory:
    id: str                         # Unique atom identifier
    text: str                       # Verbatim text payload
    metadata: dict                  # Custom metadata dictionary
    score: float                    # Fused 5-Way RRF relevance score
    triples: list[tuple]            # Knowledge Graph triples (Subject, Predicate, Object)
    entities: list[str]             # Extracted entity names associated with this atom
    created_at: datetime            # UTC timestamp when atom was recorded
    superseded_by: str | None       # ID of atom that superseded this fact, if any
```

---

## 2. `Entity`

Represents a named concept, actor, or object in the Knowledge Graph.

```python
class Entity:
    name: str                       # Canonical name of the entity
    atom_ids: list[str]             # List of all memory atoms mentioning this entity

    def related(self) -> list[Entity]:
        """Returns 1-hop connected neighboring Entity objects."""
        ...

    def timeline(self) -> list[Memory]:
        """Returns chronological list of all Memory atoms involving this entity."""
        ...
```

---

## 3. `Graph`

Represents a multi-hop network neighborhood extracted from the Knowledge Graph.

```python
class Graph:
    nodes: list[str]                # List of entity node names
    edges: list[dict]               # List of directed edges: [{"source": ..., "target": ..., "relation": ...}]

    def to_dict(self) -> dict:
        """Serializes the graph to vis.js / Cytoscape compatible JSON format."""
        ...
```

---

## 4. `UnifiedMemoryAtom`

The internal low-level persistence struct stored in the Hot Tier and Cold Tier Parquet archives.

```python
class UnifiedMemoryAtom:
    atom_id: str
    text: str
    vector: np.ndarray              # Float32 dense embedding
    metadata: dict
    triples: list[tuple]
    superseded_by: str | None
    created_at: float               # Unix epoch float timestamp
    epoch_id: str                   # Partition key
```
