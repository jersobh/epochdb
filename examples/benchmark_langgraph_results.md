# LangGraph vs. EpochDB vs. Juno Token Savings Benchmark

This benchmark compares the cumulative input token footprint of three conversational agent architectures:
1. **Standard LangGraph**: Flat, full-history message buffer ($O(N^2)$ cumulative token growth).
2. **LangGraph + EpochDB**: Thin state with selective semantic recall ($O(N)$ linear token growth).
3. **Juno (+ EpochDB)**: Dynamic blackboard workspace with targeted subgraph tiling ($O(N)$ linear token growth with structural context).

## Executive Summary

- **Execution Mode**: `Live (Gemini API)`
- **Standard LangGraph Input Tokens**: **25,386**
- **LangGraph + EpochDB Input Tokens**: **5,297** (Savings vs. Standard: **79.1%**)
- **Juno (+ EpochDB) Input Tokens**: **6,015** (Savings vs. Standard: **76.3%**)

## Token Cost Comparison Table

| Turn | User Query | Standard LangGraph (Tokens) | LangGraph + EpochDB (Tokens) | Juno (+ EpochDB) (Tokens) | Savings EpochDB (%) | Savings Juno (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | *"Hi, I'm Jeff. I'm starting a new project called Aegis."* | 51 | 71 | 51 | -39.2% | 0.0% |
| 2 | *"Project Aegis is a distributed graph database focused on sub-millisecond retrieval."* | 223 | 208 | 309 | 6.7% | -38.6% |
| 3 | *"We will build the Aegis database using the Rust programming language for performance and memory safety."* | 759 | 454 | 588 | 40.2% | 22.5% |
| 4 | *"The storage layer of Aegis is based on LSM-trees and uses a custom write-ahead log (WAL)."* | 1,615 | 542 | 850 | 66.4% | 47.4% |
| 5 | *"We are targeting a launch date of October 2026."* | 2,719 | 789 | 784 | 71.0% | 71.2% |
| 6 | *"What is the launch date we are targeting for our project?"* | 3,687 | 1,359 | 960 | 63.1% | 74.0% |
| 7 | *"What language are we using to build the Aegis database?"* | 3,723 | 824 | 814 | 77.9% | 78.1% |
| 8 | *"Can you explain what storage layer and architecture Aegis uses?"* | 3,774 | 196 | 505 | 94.8% | 86.6% |
| 9 | *"Who is starting the project and what is its name?"* | 4,399 | 377 | 537 | 91.4% | 87.8% |
| 10 | *"Summarize the database project we are working on."* | 4,436 | 477 | 617 | 89.2% | 86.1% |
| **TOTAL** | | **25,386** | **5,297** | **6,015** | **79.1%** | **76.3%** |

## Architectural Analysis

### 1. The Context Bloat Problem (Standard LangGraph)
Standard LangGraph persists the entire user-assistant thread history in the graph state. In every turn, the complete conversation history is sent to the LLM. This leads to **quadratic growth ($O(N^2)$)** in cumulative input token usage:

$$\text{Cumulative Tokens} \approx \sum_{i=1}^{N} (\text{System Prompt} + i \times \text{Turn Size})$$

### 2. The EpochDB Solution
By using EpochDB alongside `EpochDBCheckpointer`:
- **Thin State**: The LangGraph state only holds the immediate turn context, eliminating $O(N)$ historical growth in the state.
- **Precise Long-Term Recall**: The agent queries EpochDB using semantic and relational triple-hop logic, retrieving only the **top-2 relevant context items**.
- **Flat Growth ($O(N)$)**: Cumulative tokens grow linearly, keeping per-turn prompt size constant regardless of conversation depth.

### 3. The Juno Solution
Juno uses an `EpochBlackboard` surface and `ContextTiler` to represent dynamic agent state:
- **Workspace-scoped Graph**: Chat fragments and facts are stored as structured nodes/edges rather than raw text buffers.
- **Targeted Subgraph Tiling**: Instead of flat top-k vector search, Juno pulls a localized subgraph around active query concepts (depth=1) and appends only the last 2 chat turns.
- **Attributed Markdown**: Results are tiled into structured markdown blocks. This yields highly optimized context footprints and stable linear growth.
