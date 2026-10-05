# LangGraph Checkpointer (`EpochDBCheckpointer`)

The `EpochDBCheckpointer` implements the standard LangGraph `BaseCheckpointSaver` interface, enabling multi-agent graphs to persist execution state directly into EpochDB.

---

## Installation Extra

```bash
pip install "epochdb[langgraph]"
```

---

## Synchronous Usage

```python
from langgraph.graph import StateGraph
from epochdb import EpochDB
from epochdb.checkpointer import EpochDBCheckpointer

with EpochDB(storage_dir="./langgraph_state") as db:
    # Initialize checkpointer
    checkpointer = EpochDBCheckpointer(db)

    # Compile LangGraph workflow
    workflow = StateGraph(...)
    app = workflow.compile(checkpointer=checkpointer)

    # Invoke graph with thread session
    config = {"configurable": {"thread_id": "session_user_42"}}
    output = app.invoke({"messages": [...]}, config=config)
```

---

## Asynchronous Usage

```python
import asyncio
from epochdb import AsyncEpochDB
from epochdb.checkpointer import EpochDBCheckpointer

async def run():
    async with AsyncEpochDB(storage_dir="./langgraph_state") as db:
        checkpointer = EpochDBCheckpointer(db)
        app = workflow.compile(checkpointer=checkpointer)

        # Uses native aput, aget_tuple, and alist internally
        config = {"configurable": {"thread_id": "session_user_42"}}
        output = await app.ainvoke({"messages": [...]}, config=config)

asyncio.run(run())
```

---

## Why Use EpochDB as a Checkpointer?

1. **55% to 79% Token Savings**: Traditional checkpointers store flat conversation arrays that grow quadratically ($O(N^2)$). EpochDB stores turns as atomic memory units and retrieves only top-$k$ Topic-Locked facts, achieving linear $O(N)$ scaling.
2. **Unified Persistence**: One engine stores both short-term conversational execution graphs and long-term multi-hop knowledge.
3. **Point-in-Time Replays**: Inspect graph states at any historical epoch or step.
