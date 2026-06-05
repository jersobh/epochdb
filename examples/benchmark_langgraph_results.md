# LangGraph vs. EpochDB vs. Astraea Token Savings Benchmark

This benchmark compares the cumulative input token footprint of three conversational agent architectures:
1. **Standard LangGraph**: Flat, full-history message buffer ($O(N^2)$ cumulative token growth).
2. **LangGraph + EpochDB**: Thin state with selective semantic recall ($O(N)$ linear token growth).
3. **Astraea (+ EpochDB)**: Dynamic blackboard workspace with targeted subgraph tiling ($O(N)$ linear token growth with structural context).

## Executive Summary

- **Execution Mode**: `Live (Gemini API)`
- **Standard LangGraph Input Tokens**: **22,215**
- **LangGraph + EpochDB Input Tokens**: **4,413** (Savings vs. Standard: **80.1%**)
- **Astraea (+ EpochDB) Input Tokens**: **5,446** (Savings vs. Standard: **75.5%**)

## Token Cost Comparison Table

| Turn | User Query | Standard LangGraph (Tokens) | LangGraph + EpochDB (Tokens) | Astraea (+ EpochDB) (Tokens) | Savings EpochDB (%) | Savings Astraea (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | *"Hi, I'm Jeff. I'm starting a new project called Aegis."* | 51 | 71 | 52 | -39.2% | -2.0% |
| 2 | *"Project Aegis is a distributed graph database focused on sub-millisecond retrieval."* | 164 | 199 | 308 | -21.3% | -87.8% |
| 3 | *"We will build the Aegis database using the Rust programming language for performance and memory safety."* | 687 | 457 | 511 | 33.5% | 25.6% |
| 4 | *"The storage layer of Aegis is based on LSM-trees and uses a custom write-ahead log (WAL)."* | 1,449 | 520 | 682 | 64.1% | 52.9% |
| 5 | *"We are targeting a launch date of October 2026."* | 2,412 | 618 | 673 | 74.4% | 72.1% |
| 6 | *"What is the launch date we are targeting for our project?"* | 3,257 | 1,008 | 849 | 69.1% | 73.9% |
| 7 | *"What language are we using to build the Aegis database?"* | 3,293 | 665 | 719 | 79.8% | 78.2% |
| 8 | *"Can you explain what storage layer and architecture Aegis uses?"* | 3,347 | 209 | 511 | 93.8% | 84.7% |
| 9 | *"Who is starting the project and what is its name?"* | 3,757 | 473 | 549 | 87.4% | 85.4% |
| 10 | *"Summarize the database project we are working on."* | 3,798 | 193 | 592 | 94.9% | 84.4% |
| **TOTAL** | | **22,215** | **4,413** | **5,446** | **80.1%** | **75.5%** |

## Architectural Analysis

### 1. The Context Bloat Problem (Standard LangGraph)
Standard LangGraph persists the entire user-assistant thread history in the graph state. In every turn, the complete conversation history is sent to the LLM. This leads to **quadratic growth ($O(N^2)$)** in cumulative input token usage:

$$\text{Cumulative Tokens} \approx \sum_{i=1}^{N} (\text{System Prompt} + i \times \text{Turn Size})$$

### 2. The EpochDB Solution
By using EpochDB alongside `EpochDBCheckpointer`:
- **Thin State**: The LangGraph state only holds the immediate turn context, eliminating $O(N)$ historical growth in the state.
- **Precise Long-Term Recall**: The agent queries EpochDB using semantic and relational triple-hop logic, retrieving only the **top-2 relevant context items**.
- **Flat Growth ($O(N)$)**: Cumulative tokens grow linearly, keeping per-turn prompt size constant regardless of conversation depth.

### 3. The Astraea Solution
Astraea uses an `EpochBlackboard` surface and `ContextTiler` to represent dynamic agent state:
- **Workspace-scoped Graph**: Chat fragments and facts are stored as structured nodes/edges rather than raw text buffers.
- **Targeted Subgraph Tiling**: Instead of flat top-k vector search, Astraea pulls a localized subgraph around active query concepts (depth=1) and appends only the last 2 chat turns.
- **Attributed Markdown**: Results are tiled into structured markdown blocks. This yields highly optimized context footprints and stable linear growth.
