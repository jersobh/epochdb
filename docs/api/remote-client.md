# Remote Client & Server

For distributed deployments, microservice architectures, or multi-process agent workers, EpochDB includes a built-in multi-threaded HTTP server and lightweight client libraries.

---

## 1. Starting the Server

```python
from epochdb import EpochDB
from epochdb.api.server import start_server

db = EpochDB(storage_dir="./shared_cluster_memory", embedding_model="all-MiniLM-L6-v2")

# Launches a multi-threaded HTTP server
server = start_server(db, host="0.0.0.0", port=8080)

print("EpochDB Server listening on port 8080...")
try:
    server.serve_forever()
finally:
    db.close()
```

---

## 2. Synchronous Remote Client (`RemoteEpochDB`)

```python
from epochdb import RemoteEpochDB

# Connect to the remote instance
client = RemoteEpochDB(host="127.0.0.1", port=8080)

# Store a memory across the network
# (Supports distributed consistency levels: "one", "quorum", "all")
mem_id = client.remember(
    "Pollyanna is married to Jefferson.",
    consistency="quorum"
)

# Query the remote cluster
memories = client.query("Who is Pollyanna married to?", k=1)
print(memories[0].text)

# Inspect cluster health and partition statistics
stats = client.stats()
print("Cluster Statistics:", stats)
```

---

## 3. Asynchronous Remote Client (`AsyncRemoteEpochDB`)

```python
import asyncio
from epochdb import AsyncRemoteEpochDB

async def main():
    async with AsyncRemoteEpochDB(host="127.0.0.1", port=8080) as client:
        await client.remember("Async edge worker registered.")
        results = await client.query("edge worker status", k=1)
        print(results[0].text)

asyncio.run(main())
```

---

## Distributed Clustering & Horizontal Sharding

For multi-node deployments requiring consistent-hashing ring topologies, gateway caching, and multi-datacenter horizontal sharding, refer to the [EpochDB Distributed Server repository](https://github.com/jersobh/epochdb-server).
