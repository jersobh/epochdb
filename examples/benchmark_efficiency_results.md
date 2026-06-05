# EpochDB vs. Standard LangGraph vs. Astraea Token & Accuracy Benchmark

This benchmark evaluates the input token efficiency and retrieval accuracy across three conversational agent architectures:
1. **Standard LangGraph (MemorySaver)**: Flat, full-history message buffer ($O(N^2)$ cumulative token growth).
2. **LangGraph + EpochDB**: Thin state with selective semantic recall ($O(N)$ linear token growth).
3. **Astraea (+ EpochDB)**: Dynamic blackboard workspace with targeted subgraph tiling ($O(N)$ linear token growth with structural context).

## Executive Summary

- **Execution Mode**: `Offline Mock (Local Keyword Embeddings)`
- **Standard LangGraph Input Tokens**: **7,319**
- **LangGraph + EpochDB Input Tokens**: **3,437** (Savings vs. Standard: **53.0%**)
- **Astraea (+ EpochDB) Input Tokens**: **2,733** (Savings vs. Standard: **62.7%**)

## Performance & Correctness Matrix

| Turn | User Query | Standard LangGraph (Tokens/Grade) | LangGraph + EpochDB (Tokens/Grade) | Astraea (+ EpochDB) (Tokens/Grade) |
| --- | --- | --- | --- | --- |
| 1 | *"Let's talk about Star Wars. Anakin Skywalker is a young slave on Tatooine who has a high concentration of midi-chlorians."* | 68 / ✓ | 93 / ✓ | 68 / ✓ |
| 2 | *"Anakin's master is Obi-Wan Kenobi, who trains him after Qui-Gon Jinn is killed by Darth Maul."* | 219 / ✓ | 244 / ✓ | 61 / ✓ |
| 3 | *"Anakin falls to the dark side and becomes Darth Vader, serving Emperor Palpatine."* | 363 / ✓ | 388 / ✓ | 58 / ✓ |
| 4 | *"Anakin has twins, Luke Skywalker and Leia Organa, who are hidden from him to protect them."* | 511 / ✓ | 378 / ✓ | 61 / ✓ |
| 5 | *"Luke trains with Yoda on Dagobah and eventually helps Anakin redeem himself, defeating Palpatine on the second Death Star."* | 670 / ✓ | 389 / ✓ | 69 / ✓ |
| 6 | *"Where did Luke Skywalker train, and who was his teacher?"* | 820 / ✓ | 421 / ✓ | 462 / ✓ |
| 7 | *"Who is Anakin Skywalker's master, and who killed Qui-Gon Jinn?"* | 949 / ✓ | 383 / ✓ | 478 / ✓ |
| 8 | *"What is the name of Anakin's twins, and who do they need protection from?"* | 1,089 / ✓ | 358 / ✓ | 434 / ✓ |
| 9 | *"Who did Anakin Skywalker become after turning to the dark side, and who does he serve?"* | 1,245 / ✓ | 389 / ✓ | 454 / ✓ |
| 10 | *"Summarize the story of Anakin Skywalker and his family from the first six movies."* | 1,385 / ✓ | 394 / ✓ | 588 / ✓ |
| **TOTAL** | | **7,319** | **3,437** | **2,733** |

## Findings & Insights

### 1. Cumulative Token Footprint
- **Standard LangGraph** experiences quadratic cost scaling. As conversation length grows, the input token volume per turn spirals upwards.
- **LangGraph + EpochDB** and **Astraea** maintain a flattened, linear cost scaling. Per-turn prompt size remains stable and flat, regardless of conversation depth.

### 2. Fact Correctness & Memory Retrieval
- All three architectures successfully retrieve the correct context and answer the queries correctly (100% accuracy).
- This demonstrates that **EpochDB** and **Astraea** achieve substantial token savings (typically >50% on longer conversations) **without any loss in retrieval quality or factual correctness**.