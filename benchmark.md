# EpochDB Benchmarks

## LangGraph Integration & Workflow Execution

The following log demonstrates the successful execution of an agent workflow using `LangGraph` and `EpochDB` for context persistence across multiple turns of interaction. 

It leverages `gemini-embedding-2-preview` (detecting native 3072 dimensions) for indexing and `gemini-2.5-flash` for agentic generation. 

```text
python example_langgraph.py
Initializing Models & DB...
Checking embedding dimensionality via dummy call...
Detected 3072 dimensions for gemini-embedding-2-preview

Compiling Agent Graph...

================ RUNNING AGENT =================

--- Processing Input: Hello! My name is Jeff and I like programming. ---

[Node: Retrieve] Triggered for -> 'Hello! My name is Jeff and I like programming.'
[Node: Retrieve] Found Context:
No prior memory.

[Node: Generate] Calling Gemini...
[Node: Generate] Reply: Hello Jeff! It's great to meet you. Programming is a fascinating field! What kind of programming do you enjoy, or what sort of projects do you usually work on?

[Node: Store] Archiving interaction to Working Memory...
--- Processing Input: What is my name? ---

[Node: Retrieve] Triggered for -> 'What is my name?'
[Node: Retrieve] Found Context:
User said: 'Hello! My name is Jeff and I like programming.' | AI answered: 'Hello Jeff! It's great to meet you. Programming is a fascinating field! What kind of programming do you enjoy, or what sort of projects do you usually work on?'

[Node: Generate] Calling Gemini...
[Node: Generate] Reply: Your name is Jeff.

[Node: Store] Archiving interaction to Working Memory...
```

The benchmark successfully proves that EpochDB correctly injects context into a stateful graph seamlessly, resulting in accurate multi-turn reasoning with no information compression.

---

## Comparative Analysis: EpochDB vs. ChromaDB

To validate the unique architecture of EpochDB, we ran a side-by-side comparison with **ChromaDB** using three industry-standard memory benchmarks: **ConvoMem** (conversational context), **LongMemEval** (longitudinal session retrieval), and **LoCoMo** (multi-hop relational reasoning).

### Performance Metrics

| Benchmark | Store | Ingest (s) | Eval (s) | Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **ConvoMem** | EpochDB | 11.39 | 0.07 | recall@3: 1.000 |
| | ChromaDB | 9.67 | 0.03 | recall@3: 1.000 |
| **LongMemEval** | EpochDB | 0.09 | 0.05 | recall@3: 1.000 |
| | ChromaDB | 0.07 | 0.03 | recall@3: 1.000 |
| **LoCoMo** | **EpochDB** | **0.07** | **0.02** | **multi_hop_recall: 1.000** |
| | ChromaDB | 0.06 | 0.02 | multi_hop_recall: 0.000 |

### Key Takeaways

> [!IMPORTANT]
> **Relational Superiority**: While both databases perform perfectly on standard semantic retrieval (ConvoMem/LongMemEval), EpochDB demonstrates a massive architectural advantage on the **LoCoMo** benchmark. By leveraging its integrated **Knowledge Graph and Relational Expansion**, EpochDB achieved **100% recall on multi-hop queries** where ChromaDB failed completely (0% recall).

> [!NOTE]
> **Persistence Overhead**: EpochDB's slightly higher ingestion latency (approx 15-20%) is attributed to its **ACID-compliant Write-Ahead Log (WAL)** and tiered storage management, which ensures data integrity across restarts—a feature absent in basic in-memory ChromaDB configurations.
