# LangGraph Integration

EpochDB provides native checkpointer support for [LangGraph](https://github.com/langchain-ai/langgraph), delivering high-performance persistence and substantial token savings.

---

## The Quadratic Token Problem in LangGraph

Standard LangGraph checkpointers (such as `MemorySaver` or `SqliteSaver`) persist state by appending the entire conversation history to the thread state. Over long sessions, this creates quadratic $O(N^2)$ token growth:

```
Turn 1:  [Message 1]                                 ->  100 tokens
Turn 10: [Message 1 ... Message 10]                 -> 1,000 tokens
Turn 50: [Message 1 ... Message 50]                 -> 5,000 tokens
-------------------------------------------------------------------
Cumulative input tokens billed: Sum(100 * t) ≈ 127,500 tokens!
```

---

## The EpochDB Solution: Thin State + Selective Recall

When using `EpochDBCheckpointer`, conversational turns are stored as **Unified Memory Atoms** in EpochDB:
- The graph state is kept "thin": only the active goal and lightweight thread pointers are serialized.
- When an agent node executes, it queries EpochDB for the **top-$k$ Topic-Locked facts** relevant to the current task.
- Token consumption scales linearly ($O(N)$), saving **55% to 79%** of cumulative token costs.

```mermaid
flowchart TD
    GraphNode[LangGraph Agent Node] -->|Query active intent| DB[(EpochDB Engine)]
    DB -->|Return Top-K Topic-Locked Atoms| Context[Slim Prompt Context]
    Context --> LLM[LLM Call: Flat Token Payload]
    LLM --> Checkpoint[EpochDBCheckpointer.put()]
    Checkpoint --> DB
```

---

## Code Example

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator
from epochdb import EpochDB
from epochdb.checkpointer import EpochDBCheckpointer

class AgentState(TypedDict):
    query: str
    context: str
    response: str

def retrieve_node(state: AgentState):
    # Query EpochDB for precision facts
    facts = db.query(state["query"], k=2)
    return {"context": "\n".join([f.text for f in facts])}

def generate_node(state: AgentState):
    # Prompt LLM with thin context
    return {"response": f"Answer based on {state['context']}"}

builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

with EpochDB(storage_dir="./graph_memory") as db:
    checkpointer = EpochDBCheckpointer(db)
    app = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "session_101"}}
    output = app.invoke({"query": "What is the status of project Titan?"}, config=config)
    print("Agent Output:", output["response"])
```
