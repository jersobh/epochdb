# EpochDB Benchmarks

## LangGraph Integration & Workflow Execution

The following log demonstrates the successful execution of an agent workflow using `LangGraph` and `EpochDB` for context persistence across multiple turns of interaction. 

It leverages `gemini-embedding-2-preview` (detecting native 3072 dimensions) for indexing and `gemini-2.5-flash` for agentic generation. 

```text
python example_langgraph.py
Initializing EpochAgent with Tiered Memory...
Detected 3072 dimensions from Gemini.
Compiling Agent Graph...

================ SESSION 1: Establishing Reality =================

--- Turn: Hi, I'm Jeff. I'm building EpochDB. ---

[Node: Retrieve] Analyzing -> 'Hi, I'm Jeff. I'm building EpochDB.'
[Node: Retrieve] Found Context (incl. Relational Expansion):
No prior memory.

[Node: Generate] Reasoning with Gemini...
[Node: Generate] Reply: Hi Jeff, it's great to meet you. EpochDB sounds interesting! I'm EpochAgent. Building a database from the ground up is a significant undertaking. What are some of the key design goals or challenges you're currently focused on with EpochDB?


[Node: Extract & Store] Updating Knowledge Graph...
   -> Extracted Triples: [('user', 'has_name', 'Jeff'), ('user', 'works_on', 'epochdb'), ('epochdb', 'is_a', 'memory_engine')]

--- Turn: EpochDB is a tiered memory engine. ---

[Node: Retrieve] Analyzing -> 'EpochDB is a tiered memory engine.'
[Node: Retrieve] Found Context (incl. Relational Expansion):
- Interaction: Hi, I'm Jeff. I'm building EpochDB. -> Hi Jeff, it's great to meet you. EpochDB sounds interesting! I'm EpochAgent. Building a database from the ground up is a significant undertaking. What are some of the key design goals or challenges you're currently focused on with EpochDB?


[Node: Generate] Reasoning with Gemini...
[Node: Generate] Reply: Okay, that's a great starting point. So, EpochDB being a tiered memory engine is its core concept. To understand better, could you elaborate on the different tiers you're implementing? Are we talking about a RAM/SSD/HDD setup, or something more granular, perhaps incorporating NVMe or even cloud storage as a tier? Also, what are the key criteria for data movement between tiers? (e.g., access frequency, data age, data size) Understanding these aspects will help me grasp the specific challenges and design choices you're facing.


[Node: Extract & Store] Updating Knowledge Graph...
   -> Extracted Triples: [('user', 'works_on', 'epochdb'), ('epochdb', 'is_a', 'memory_engine')]

[Admin] Forcing cold-tier flush to Parquet...

================ SESSION 2: Multi-Hop Reasoning =================

--- Logic Test: What is the memory engine that Jeff is working on? ---

[Node: Retrieve] Analyzing -> 'What is the memory engine that Jeff is working on?'
[Node: Retrieve] Found Context (incl. Relational Expansion):
- Interaction: Hi, I'm Jeff. I'm building EpochDB. -> Hi Jeff, it's great to meet you. EpochDB sounds interesting! I'm EpochAgent. Building a database from the ground up is a significant undertaking. What are some of the key design goals or challenges you're currently focused on with EpochDB?

- Interaction: EpochDB is a tiered memory engine. -> Okay, that's a great starting point. So, EpochDB being a tiered memory engine is its core concept. To understand better, could you elaborate on the different tiers you're implementing? Are we talking about a RAM/SSD/HDD setup, or something more granular, perhaps incorporating NVMe or even cloud storage as a tier? Also, what are the key criteria for data movement between tiers? (e.g., access frequency, data age, data size) Understanding these aspects will help me grasp the specific challenges and design choices you're facing.


[Node: Generate] Reasoning with Gemini...
[Node: Generate] Reply: Jeff is working on a tiered memory engine called EpochDB.


[Node: Extract & Store] Updating Knowledge Graph...
```

The benchmark successfully proves that EpochDB correctly injects context into a stateful graph seamlessly, resulting in accurate multi-turn reasoning with no information compression.

---

## Comparative Analysis: EpochDB vs. Industry Standards

To validate the unique architecture of EpochDB, we ran a side-by-side comparison with **ChromaDB**, **LanceDB**, **FAISS**, and **Qdrant** using three key benchmarks: **ConvoMem** (conversational context), **LongMemEval** (longitudinal session retrieval), and **LoCoMo** (multi-hop relational reasoning).

### Performance Metrics

| Benchmark | Store | Ingest (s) | Eval (s) | Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **ConvoMem** | EpochDB | 10.97 | 0.07 | recall@3: 1.000 |
| | ChromaDB | 9.07 | 0.03 | recall@3: 1.000 |
| | LanceDB | 5.49 | 0.05 | recall@3: 1.000 |
| | FAISS | 4.66 | 0.03 | recall@3: 1.000 |
| | Qdrant | 4.58 | 0.03 | recall@3: 1.000 |
| **LongMemEval** | EpochDB | 0.09 | 0.04 | recall@3: 1.000 |
| | ChromaDB | 0.07 | 0.03 | recall@3: 1.000 |
| | LanceDB | 0.07 | 0.04 | recall@3: 1.000 |
| | FAISS | 0.05 | 0.02 | recall@3: 1.000 |
| | Qdrant | 0.05 | 0.03 | recall@3: 1.000 |
| **LoCoMo** | **EpochDB** | **0.05** | **0.02** | **multi_hop_recall: 1.000** |
| | ChromaDB | 0.06 | 0.02 | multi_hop_recall: 0.000 |
| | LanceDB | 0.05 | 0.02 | multi_hop_recall: 0.000 |
| | FAISS | 0.04 | 0.01 | multi_hop_recall: 0.000 |
| | Qdrant | 0.04 | 0.01 | multi_hop_recall: 0.000 |

### Key Takeaways

> [!IMPORTANT]
> **Relational Superiority Across the Board**: While industry giants like FAISS and Qdrant provide exceptional raw ingestion speed, they act as "flat" vector stores. EpochDB is the **only database tested** that successfully navigated multi-hop relational queries (LoCoMo), proving that an integrated Knowledge Graph is essential for advanced AI agent context.

> [!TIP]
> **Reliability vs. Speed**: EpochDB's slightly higher ingestion latency is a direct result of its **ACID-compliant WAL** and tiered storage management. For agentic workflows where memory persistence and multi-hop reasoning are critical, this small overhead is a negligible trade-off for the massive gain in retrieval quality.

> [!NOTE]
> **LanceDB Comparison**: As expected, LanceDB (Parquet-native) showed very strong performance similarity to EpochDB’s Cold Tier architecture, reaching near-FAISS speeds while maintaining local persistence. However, without a relational index, it remains a purely semantic search tool.
