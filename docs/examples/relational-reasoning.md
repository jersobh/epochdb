# Multi-Hop Relational Reasoning

Flat vector databases struggle with multi-hop questions because the query keywords often have no direct overlap with the answer document. This example demonstrates how EpochDB uses its Knowledge Graph expansion to connect disparate facts across hops.

---

## The Scenario

We have three distinct documents ingested into the system:

1. **Document A**: *"Alice Vance is the Principal Scientist at BioHealth Labs."*
2. **Document B**: *"BioHealth Labs was acquired by Omnicorp for $2B."*
3. **Document C**: *"Omnicorp is headquartered in Geneva, Switzerland."*

Now an agent is asked:
> **"What city is the parent company of Alice Vance's employer located in?"**

- A standard vector search queries for *"city"*, *"parent company"*, and *"Alice Vance"*. It usually retrieves Document A, but completely misses Document C because Document C never mentions Alice or BioHealth.
- **EpochDB** follows the graph edges:
  `Alice Vance -> works_at -> BioHealth Labs -> subsidiary_of -> Omnicorp -> located_in -> Geneva`.

---

## Runnable Example

```python
from epochdb import EpochDB

def run_multihop_demo():
    with EpochDB(storage_dir="./multihop_demo", embedding_model="all-MiniLM-L6-v2") as db:
        # Ingest facts with structured triples
        db.remember(
            "Alice Vance serves as Principal Scientist at BioHealth Labs.",
            metadata={"triples": [("Alice Vance", "works_at", "BioHealth Labs")]}
        )

        db.remember(
            "BioHealth Labs is a wholly-owned subsidiary of Omnicorp.",
            metadata={"triples": [("BioHealth Labs", "subsidiary_of", "Omnicorp")]}
        )

        db.remember(
            "Omnicorp international headquarters are established in Geneva.",
            metadata={"triples": [("Omnicorp", "located_in", "Geneva")]}
        )

        # Query with multi-hop expansion (expand_hops=2)
        print("Executing Multi-Hop Query...")
        results = db.query(
            "What city is the parent company of Alice Vance's employer located in?",
            k=3,
            expand_hops=2
        )

        print("\nRetrieved Context Chain:")
        for idx, mem in enumerate(results, 1):
            print(f"{idx}. {mem.text}")

        # Extract complete relational subgraph
        subgraph = db.entity_graph("Alice Vance", depth=3)
        print("\nTraversed Knowledge Graph Edges:")
        for edge in subgraph.edges:
            print(f"  {edge['source']} --[{edge['relation']}]--> {edge['target']}")

if __name__ == "__main__":
    run_multihop_demo()
```
