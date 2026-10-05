# AsyncEpochDB (Asynchronous API)

`AsyncEpochDB` provides a non-blocking facade designed for high-concurrency event loops, web servers (FastAPI/Starlette), and asynchronous multi-agent orchestration frameworks (Aster Framework).

---

## Async Context Manager

```python
import asyncio
from epochdb import AsyncEpochDB

async def main():
    async with AsyncEpochDB(
        storage_dir="./async_memory",
        embedding_model="all-MiniLM-L6-v2"
    ) as db:
        atom_id = await db.remember("Autonomous systems need non-blocking I/O.")
        results = await db.query("non-blocking I/O", k=1)
        print(results[0].text)

asyncio.run(main())
```

---

## Core Asynchronous Methods

All blocking disk and network I/O operations are offloaded to dedicated thread pools or native async coroutines.

### `await db.remember(text, vector, metadata)`
Asynchronously commits a memory atom to the WAL and Hot Tier.
- **Returns**: `str` (`atom_id`)

### `await db.propose(text, vector, metadata)`
Asynchronously stages a candidate atom and executes symbolic validators.
- **Returns**: `dict` (`{"status": "APPROVED", ...}`)

### `await db.query(query_text, query_vector, k, filters, ...)`
Executes the 5-stage retrieval pipeline asynchronously.
- **Returns**: `list[Memory]`

### `await db.get(atom_id)`
Asynchronously retrieves a memory atom by ID.
- **Returns**: `Memory | None`

### `await db.get_entity(name)`
Asynchronously resolves an entity from the Active KG and Global Entity Index.
- **Returns**: `Entity`

### `await db.entity_graph(name, depth=2)`
Asynchronously compiles a multi-hop subgraph.
- **Returns**: `Graph`

### `await db.delete(atom_id, hard=False)`
Asynchronously flags an atom as deleted or purges it.
- **Returns**: `bool`

### `await db.close()`
Asynchronously flushes all pending memory queues and releases pooled file handles.
