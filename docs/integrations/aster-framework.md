# Aster Framework (OABS)

[Aster Framework](https://github.com/jersobh/aster-framework) is a Truth-Native multi-agent orchestration framework powered by the **Open-Agentic Blackboard Specification (OABS)**. 

Aster uses **EpochDB** as its primary shared memory substrate, delegating all vector search, knowledge graph traversal, and atomic state versioning to EpochDB's tiered engine.

---

## How Aster and EpochDB Collaborate

In Aster, agents never pass monolithic state objects directly to each other. Instead, agents interact via an asynchronous **Event Router** around a shared **Blackboard Surface** backed by `EpochDB`:

```mermaid
graph TD
    subgraph "Aster Agent Layer"
        AgentA[Agent Alpha]
        AgentB[Agent Beta]
        Router[Event Router (Pub/Sub)]
    end

    subgraph "OABS Blackboard Surface (aster.EpochBlackboard)"
        GlobalSpace["global:* (Lineage Enforced)"]
        PrivateSpace["agent:<name>:* (Scratchpads)"]
    end

    subgraph "EpochDB Tiered Engine"
        Hot[Hot Tier RAM: HNSW + WAL + Active KG]
        Cold[Cold Tier Disk: Parquet + Epoch HNSW]
    end

    AgentA -->|write_node / write_edge| GlobalSpace
    GlobalSpace -->|Emit BlackboardEvent| Router
    Router -->|Notify Subscribers| AgentB
    GlobalSpace --> Hot
    Hot -->|Periodic Flush| Cold
    AgentB -->|get_subgraph / query_vector| Hot
```

---

## The `EpochBlackboard` Wrapper

Aster provides `EpochBlackboard`, which sits directly on top of `AsyncEpochDB`:

```python
from aster import EpochBlackboard, LineageContext

# Initialized backed by EpochDB
bb = EpochBlackboard(
    storage_dir="./.aster_data",
    dim=384,
    model="all-MiniLM-L6-v2",
)

# Writes are stored as Unified Memory Atoms with lineage
event = await bb.write_node(
    node_id="global:fact:quarterly_profit",
    label="Fact",
    properties={"profit_margin": 0.24, "year": 2026},
    lineage=LineageContext(
        triggering_event_id="EV_102",
        agent_name="FinancialAnalyst",
        rationale="Computed Q3 net profit margin from audit reports.",
        sources_used=["global:raw:audit_2026_q3"],
    ),
)
```

---

## Why Aster + EpochDB is Superior to Traditional Frameworks

1. **State Footprint Reduction (94.5% less data in transit)**: Monolithic frameworks pass giant state dictionaries (~41KB) through every graph node. Aster writes state into EpochDB and routes only lightweight event notifications (~2.3KB).
2. **Cumulative Token Savings (3.3x lower API costs)**: Rather than stuffing complete conversation logs into prompt contexts, Aster agents query EpochDB for only the specific Topic-Locked facts they need.
3. **8.83x Multi-User Throughput Speedup**: Combining Aster's async event pool with EpochDB's non-blocking I/O enables massive concurrency that collapses network latency bottlenecks.
4. **Deterministic Auditing**: Every blackboard node is versioned and attributed through `LineageContext`.
