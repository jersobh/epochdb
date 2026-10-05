# EpochDB vs Mem0, Letta / MemGPT, and Graphiti / Zep

Agent memory systems solve overlapping problems with different bets. Some sit above a vector store and extract facts with an LLM. Some treat the context window like an OS and let the agent page its own state. Some build a temporal knowledge graph as the primary store. EpochDB is a **lossless, tiered memory engine**: verbatim atoms, hot/cold HNSW, an embedded knowledge graph, deterministic supersession, and optional symbolic write validation — without requiring an LLM on every write.

This page compares those approaches, then explains how EpochDB's `MemoryType.SKILL` fits into that design.

---

## At a glance

| Dimension | EpochDB | Mem0 | Letta / MemGPT | Graphiti / Zep |
| --- | --- | --- | --- | --- |
| Primary unit | Verbatim memory atom + embedding + triples | LLM-extracted fact / memory | Core memory blocks + archival / recall stores | Episode → entity nodes + temporal edges |
| Storage model | Embedded Hot Tier (RAM) + Cold Tier (Parquet + epoch HNSW) | Memory layer over vector DB + SQL + entity store | In-context blocks + external archival / recall | Temporal KG (Neo4j / FalkorDB / Neptune / Kuzu) |
| Write path | `remember` (direct) or `propose` (LCAG validators) | LLM extraction, dedup, embed | Agent tool calls rewrite core / archival memory | LLM extracts entities and edges with bi-temporal bounds |
| Conflict handling | Subject–Predicate supersession + Topic Lock | Dedup / update or ADD-only alongside older facts | Agent revises core blocks; archival search is semantic | Edge invalidation (`valid_at` / `invalid_at`, transaction time) |
| Graph | Co-located triples, Active KG, Global Entity Index | Entity linking; graph stronger on Platform | Not the primary model | First-class temporal context graph |
| Lexical search | BM25 hot index + cold lexical probes in RRF | Semantic + keyword (BM25) fusion | Typically semantic archival search | Vector + BM25 + graph hybrid |
| LLM required to persist? | No (optional extraction / routing) | Yes for fact distillation | Yes for agent-managed memory tools | Yes for episode → graph extraction |
| Best fit | Agents that need lossless history, state corrections, and local/self-hosted durability | Apps that want a managed memory API over existing vector backends | Long-running agents that self-manage what stays in the prompt | Systems that need bi-temporal “what was true when” graph queries |

---

## How each system thinks about memory

### EpochDB — lossless engine with state machinery

EpochDB stores every write as a **Unified Memory Atom**: raw text, `float32` embedding, metadata, and optional `(subject, predicate, object)` triples. History is not summarized away. Working memory lives in RAM (Hot Tier HNSW + WAL + Active KG). Historical epochs flush to Parquet with per-epoch HNSW indexes, a Global Entity Index, and centroid probing so cold search does not broadcast to every archive by default.

Retrieval is a multi-stage pipeline (semantic + keyword hook → KG seeding → relational expansion → 5-way RRF → supersession). When a newer atom updates the same Subject–Predicate, older candidates are demoted rather than silently deleted. `propose()` adds an opt-in neuro-symbolic gate: symbolic validators must approve before WAL / HNSW / KG commit.

### Mem0 — memory layer above a vector store

Mem0 is a **memory API** that sits between the application and storage. Conversation turns go through an LLM extraction pipeline that distills durable facts, deduplicates, embeds, and links entities. Persistence typically spans a vector database (Qdrant, Pinecone, pgvector, and many others), SQL for history, and an entity store for relationship-aware retrieval.

Mem0's strength is productized extraction and scoped search (`user_id`, `agent_id`, `run_id`) without building that pipeline yourself. The open-source path leans on the configured vector backend; richer graph memory is emphasized on the hosted platform. Persistence of *verbatim* dialogue is secondary to distilled memory records.

### Letta / MemGPT — OS-style context management

MemGPT (productized as Letta) treats the LLM context window like **main memory**. Core memory blocks (persona, human, custom) stay in every prompt. Conversation history and large corpora live out of context in recall and archival stores. The agent uses tools to search those stores and to rewrite core blocks when it decides something must stay pinned.

The innovation is **self-directed paging**: the model chooses what enters and leaves the prompt. That is an agent runtime / memory-management protocol more than a storage engine. Archival search is typically semantic (e.g. vector-backed). Temporal supersession, cold-tier probing, and symbolic commit gates are outside the core MemGPT thesis.

### Graphiti / Zep — temporal knowledge graphs

Graphiti (open source; Zep is the memory service built on it) turns messages and structured data into a **bi-temporal knowledge graph**. Episodes become entity nodes and relationship edges. Edges carry when a fact was valid in the world (`valid_at` / `invalid_at`) and when the system learned or expired it (`created_at` / `expired_at`). Contradictions invalidate old edges instead of deleting them, so point-in-time questions are first-class.

Retrieval combines vector similarity, full-text (BM25), and graph traversal. The graph is the center of gravity; verbatim episode text is an input to extraction rather than the default retrieval payload. Running Graphiti usually means operating a graph database backend.

---

## Side-by-side on the hard problems

### 1. Fidelity vs compression

| System | Approach |
| --- | --- |
| **EpochDB** | Persist verbatim atoms. Compress storage (Parquet + zstd), not meaning. Prompt size stays bounded by top-*k* retrieval. |
| **Mem0** | Compress *into* facts via LLM extraction. Good for preferences and durable summaries; original wording is not the primary artifact. |
| **Letta** | Core blocks are deliberately short and editable. Full history lives in external stores; the agent decides what to surface. |
| **Graphiti / Zep** | Episodes feed extracted entities and edges. History of *facts* is preserved temporally; free-text is secondary to the graph. |

### 2. Updating a wrong or outdated fact

| System | Approach |
| --- | --- |
| **EpochDB** | New atom with the same Subject–Predicate supersedes the old one in ranking (Topic Lock boost + staleness penalty). Optional `propose()` rejects illegal transitions before commit. |
| **Mem0** | Extraction-time dedup / conflict handling, or ADD-only retention of both versions depending on configuration and product path. |
| **Letta** | The agent edits core memory blocks or writes new archival entries. Correctness depends on tool use, not a DB-level supersession rule. |
| **Graphiti / Zep** | New edge invalidates the old edge's validity interval. Bi-temporal queries can reconstruct past belief and past world state. |

### 3. Relational / multi-hop reasoning

| System | Approach |
| --- | --- |
| **EpochDB** | Triples on atoms, Active KG in RAM, GEI for cold routing, `expand_hops` / entity domain objects. |
| **Mem0** | Entity linking boosts retrieval; full graph memory is stronger on Platform than in every OSS setup. |
| **Letta** | Not graph-native; multi-hop comes from the agent chaining tool calls and reasoning. |
| **Graphiti / Zep** | Native multi-hop over a temporal entity graph with hybrid search. |

### 4. Operational shape

| System | Approach |
| --- | --- |
| **EpochDB** | Embeddable Python library; local Hot/Cold directories; optional distributed [epochdb-server](https://github.com/jersobh/epochdb-server). No graph DB required. |
| **Mem0** | Library + many vector backends, or hosted platform. You (or Mem0) operate the underlying stores. |
| **Letta** | Agent server / runtime with memory blocks and tools as the integration surface. |
| **Graphiti / Zep** | Graphiti OSS against a graph DB, or Zep as a managed memory service. |

### 5. When an LLM is on the critical path

| System | Write path |
| --- | --- |
| **EpochDB** | Embedding encode always; LLM extraction optional (`auto_extract`). Symbolic validators are code, not prompts. |
| **Mem0** | LLM distillation is central to `add`. |
| **Letta** | LLM decides memory tool calls every turn that needs paging or core edits. |
| **Graphiti / Zep** | LLM extracts entities, edges, and often temporal bounds from each episode. |

EpochDB's design target is agents that cannot afford write-time LLM cost or non-deterministic extraction for every state change, but still need graph-aware, supersession-aware retrieval over a large archive.

---

## Choosing among them

Use **EpochDB** when you need:

- Verbatim, durable atoms with crash-safe WAL replay
- Deterministic correction of stale facts without deleting history
- Embedded vector + graph + tiered disk in one process
- Optional neuro-symbolic gates on state mutations
- First-class procedural skills (`MemoryType.SKILL`) stored as structured atoms

Use **Mem0** when you want a thin memory API, LLM-based fact extraction, and flexible backends (or a hosted service) without owning a custom engine.

Use **Letta / MemGPT** when the product *is* a long-lived agent that must program its own context: pinned persona/user blocks, tool-mediated archival search, and OS-like paging.

Use **Graphiti / Zep** when bi-temporal graph queries (“what was true about X at time T?”) and enterprise context graphs are the primary requirement, and you are willing to run (or buy) a graph-backed memory layer.

These are not mutually exclusive. An EpochDB store can back a Letta-style agent as archival memory; a Mem0-like extraction step can write *into* EpochDB atoms; Graphiti-style temporal edges can be modeled as triples plus metadata if you do not need a full bi-temporal graph DB.

---

## `MemoryType.SKILL` in EpochDB

`MemoryType` tags each atom for retrieval prioritization and filtered queries:

| Value | Role |
| --- | --- |
| `general` | Default facts and notes |
| `episodic` | Conversational context across sessions |
| `profile` | Long-term user facts and preferences |
| `working` | Short-term session context |
| `skill` | Procedural knowledge: named SOPs, tool playbooks, decision rules |

`MemoryType.SKILL` is for **how to do something**, not only **what is true**. Mem0 typically stores extracted preferences and facts. Letta pins persona/human text in core blocks and dumps procedures into archival blobs the agent must search. Graphiti models relationships over time, not executable step lists. EpochDB makes skills a typed, queryable atom with dedicated APIs.

### Writing a skill

```python
from epochdb import EpochDB

with EpochDB(storage_dir="./memory") as db:
    db.remember_skill(
        skill_name="refund_order",
        description="Refund a paid order after verifying it has not already been refunded.",
        steps=[
            {"step_num": 1, "action": "Look up the order", "details": "Must be paid, no prior refund."},
            {"step_num": 2, "action": "Call the refund tool", "details": "Pass order_id and captured amount."},
        ],
        decision_rules=["Never refund more than the captured amount."],
        tool_schema={
            "name": "refund_order",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
        skill_id="skill-refund-order",
    )
```

`remember_skill()`:

1. Builds a single searchable text payload (`SKILL [name]: … Steps: … Rules: …`).
2. Stores structured `metadata` (`skill_name`, `steps`, `decision_rules`, `tool_schema`, `type="skill"`).
3. Adds graph triples such as `(name, "is_skill", id)` and `(id, "has_step_n", action)` unless you pass custom `triples`.
4. Sets `memory_type="skill"` and optionally a stable `skill_id` as the atom id.

There is no separate `.md` skill file. The procedure is a normal atom: embedded, WAL-logged, flushed to Parquet, and supersedable like any other memory.

### Reading skills

- **`get_skill(name)`** — indexed exact match on name, title, skill id, process id, or atom id; semantic fallback scoped to `memory_type="skill"`.
- **`list_skills()`** — hot `SkillIndex` + cold skill refs (no full timeline scan).
- **`query(..., memory_type="skill")`** — hybrid (semantic + keyword) RRF retrieval limited to procedures.
- **`get_hot_summary_snapshot(user_id)`** — compact profile + top skills block for system-prompt injection.
- **MCP / LangChain tools** — `epochdb_remember_skill`, `epochdb_get_skill`, `epochdb_list_skills`, and profile helpers.

### Why skills are a separate type

Without a type tag, procedures drown in general episodic chatter. Filtering on `memory_type="skill"` keeps tool playbooks out of preference search and keeps profile facts out of SOP lookup. Agents can still use one engine for facts, episodes, profiles, working context, and skills — with one retrieval stack and one durability model.

For the full API surface and durability notes, see [Skill Memory](/guide/skill-memory).

---

## Further reading

- [Architecture Overview](/guide/architecture)
- [Atomic State & Supersession](/guide/atomic-state-and-supersession)
- [Knowledge Graph & GEI](/guide/knowledge-graph)
- [Skill Memory](/guide/skill-memory)
- [Neuro-Symbolic LCAG](/guide/neuro-symbolic-lcag)
