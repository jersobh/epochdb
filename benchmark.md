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
