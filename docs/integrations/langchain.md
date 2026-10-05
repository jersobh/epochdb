# LangChain & LlamaIndex Integration

EpochDB integrates with the LangChain and LlamaIndex ecosystems, enabling developers to replace standard vector stores with a state-aware, multi-hop memory engine.

---

## LangChain Integration

### 1. VectorStore Implementation
Use `EpochDBVectorStore` anywhere you would use a standard `VectorStore`:

```python
from epochdb.vectorstore import EpochDBVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = EpochDBVectorStore(
    storage_dir="./langchain_store",
    embedding=embeddings,
)

# Ingest documents with metadata
vectorstore.add_texts(
    texts=["Elena Rostova oversees the Quantum Computing initiative."],
    metadatas=[{"department": "R&D", "clearance": "TopSecret"}]
)
```

### 2. Multi-Hop Retriever with Topic Lock
Use `EpochDBMultiHopRetriever` in LCEL pipelines:

```python
from epochdb.vectorstore import EpochDBMultiHopRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

retriever = EpochDBMultiHopRetriever(
    vectorstore=vectorstore,
    expand_hops=2,
    topic_lock=True,
    k=3,
)

prompt = ChatPromptTemplate.from_template("""Answer the question using the context below:
Context: {context}
Question: {question}""")

chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | ChatOpenAI(model="gpt-4o")
)

response = chain.invoke("Who manages quantum computing?")
print(response.content)
```

---

## Agent Tool Calling

Expose EpochDB as executable tools to OpenAI / Anthropic / Gemini tool-calling agents:

```python
from epochdb.tools import get_epochdb_tools
from epochdb import EpochDB

db = EpochDB(storage_dir="./tools_memory")
tools = get_epochdb_tools(db)

# tools contains: [remember_tool, query_tool, entity_lookup_tool]
# Bind directly to your agent:
# agent = create_tool_calling_agent(llm, tools, prompt)
```
