# API Reference Overview

EpochDB exposes clean, object-oriented Python APIs designed for both synchronous scripting and asynchronous multi-agent orchestration frameworks.

---

## Core Classes & Facades

| Module / Class | Type | Description |
| :--- | :--- | :--- |
| **`EpochDB`** | Class | Primary synchronous context manager and database client. |
| **`AsyncEpochDB`** | Class | Asynchronous client optimized for `asyncio` loops and agent eventbuses. |
| **`RemoteEpochDB`** | Class | HTTP REST client for communicating with remote or distributed EpochDB clusters. |
| **`AsyncRemoteEpochDB`** | Class | Asynchronous HTTP REST client. |
| **`Memory`** | Domain Object | Rich returned abstraction representing an individual retrieved memory atom. |
| **`Entity`** | Domain Object | Abstraction representing a real-world concept or node in the Knowledge Graph. |
| **`Graph`** | Domain Object | Subgraph network container with `.nodes`, `.edges`, and `.to_dict()`. |
| **`UnifiedMemoryAtom`** | Low-Level Type | Raw persistent atom record with dense vectors, triples, and timestamps. |
| **`SymbolicValidator`** | Abstract Class | Base class for defining deterministic validation rules for `propose()`. |
| **`EpochDBCheckpointer`** | Integration | LangGraph-compatible state checkpointer saving 55%–79% tokens. |
| **`EpochDBVectorStore`** | Integration | LangChain-compatible vectorstore provider. |
| **`EpochDBMultiHopRetriever`**| Integration | LangChain retriever combining vector similarity with graph traversal. |

---

## Quick Import Guide

```python
from epochdb import (
    EpochDB,
    AsyncEpochDB,
    RemoteEpochDB,
    AsyncRemoteEpochDB,
    Memory,
    Entity,
    Graph,
    UnifiedMemoryAtom,
)

# Neuro-symbolic validation
from epochdb.validation import (
    SymbolicValidator,
    ValidationResult,
    ValidationStatus,
)

# LangGraph checkpointer
from epochdb.checkpointer import EpochDBCheckpointer

# LangChain tools & vectorstore
from epochdb.vectorstore import EpochDBVectorStore, EpochDBMultiHopRetriever
from epochdb.tools import get_epochdb_tools
```
