# Basic Memory Operations

This example demonstrates the core CRUD lifecycle in EpochDB: remembering facts, updating existing information (supersession), querying with semantic search, and deleting atoms.

---

## Complete Runnable Script

```python
from epochdb import EpochDB

def main():
    print("=== Initializing EpochDB ===")
    with EpochDB(storage_dir="./demo_basic", embedding_model="all-MiniLM-L6-v2") as db:

        # 1. Insert initial facts
        id1 = db.remember(
            "Project Apollo was initiated by Dr. Sarah Lin in 2024.",
            metadata={"project": "Apollo", "status": "active"}
        )
        print(f"[Stored] Atom ID: {id1}")

        # 2. Querying
        print("\n=== Query 1: Who initiated Apollo? ===")
        results = db.query("Who founded or initiated Project Apollo?", k=1)
        for mem in results:
            print(f"Result: {mem.text} (Score: {mem.score:.4f})")

        # 3. State Update with Predicate
        print("\n=== Updating State: Lead Engineer Changed ===")
        db.remember(
            "In 2026, Dr. Sarah Lin transferred; Marcus Vance became lead of Project Apollo.",
            metadata={"triples": [("Marcus Vance", "leads", "Project Apollo")]}
        )

        # 4. Query with Supersession
        print("\n=== Query 2: Current Lead of Apollo ===")
        results = db.query("Who is currently leading Project Apollo?", k=1)
        for mem in results:
            print(f"Current Lead: {mem.text}")

        # 5. Soft Deletion
        print("\n=== Deleting Atom ===")
        deleted = db.delete(id1, hard=False)
        print(f"Atom soft-deleted: {deleted}")

        # Querying again
        results = db.query("Who founded Apollo?", k=1)
        print("After deletion, active top result:", results[0].text if results else "None")

if __name__ == "__main__":
    main()
```
