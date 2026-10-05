# VectorStore & Multi-Hop Retriever

EpochDB provides native adapters for the **LangChain** ecosystem through `EpochDBVectorStore` and `EpochDBMultiHopRetriever`.

---

## 1. `EpochDBVectorStore`

Drop-in replacement for any LangChain `VectorStore`:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from epochdb.vectorstore import EpochDBVectorStore

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = EpochDBVectorStore(
    storage_dir="./langchain_epochdb",
    embedding=embeddings,
)

# Ingest texts
vectorstore.add_texts(
    texts=["VectorAI develops genomic models."],
    metadatas=[{"source": "press_release"}]
)

# Standard similarity search
docs = vectorstore.similarity_search("genomic modeling", k=2)
for doc in docs:
    print(doc.page_content)
```

---

## 2. `EpochDBMultiHopRetriever`

Combines dense vector similarity with automated graph traversal to retrieve multi-hop context:

```python
from epochdb.vectorstore import EpochDBMultiHopRetriever

retriever = EpochDBMultiHopRetriever(
    vectorstore=vectorstore,
    expand_hops=2,          # Follow relational graph edges up to 2 hops
    topic_lock=True,        # Enforce topic lock boost
    k=4,
)

# LangChain LCEL chain integration
chain = retriever | prompt | llm
response = chain.invoke("What projects does the team lead at VectorAI oversee?")
```
