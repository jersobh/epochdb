# Configuring Embedding Providers

EpochDB provides built-in model dispatchers supporting both local offline neural networks and cloud API embedding providers.

---

## 1. Local Offline Models (Default)

Requires `pip install "epochdb[embeddings]"`. No API keys or internet connection required:

```python
from epochdb import EpochDB

# Any HuggingFace sentence-transformers model
db = EpochDB(
    storage_dir="./offline_memory",
    embedding_model="all-MiniLM-L6-v2",   # Inferred dim=384
)
```

Recommended local models:
- `"all-MiniLM-L6-v2"` (384D — ultra-fast, low memory)
- `"BAAI/bge-small-en-v1.5"` (384D — high accuracy)
- `"BAAI/bge-large-en-v1.5"` (1024D — state of the art accuracy)

---

## 2. Google Gemini API

Requires `pip install "epochdb[google]"` and `GEMINI_API_KEY` set in your environment:

```python
import os
os.environ["GEMINI_API_KEY"] = "AIzaSy..."

db = EpochDB(
    storage_dir="./gemini_memory",
    embedding_model="google:text-embedding-004",
    dim=768,
)
```

---

## 3. OpenAI & OpenAI-Compatible Endpoints

Requires `OPENAI_API_KEY` set in your environment:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

db = EpochDB(
    storage_dir="./openai_memory",
    embedding_model="openai:text-embedding-3-small",
    dim=1536,
)
```

### Routing to Local Inference Servers (vLLM, LM Studio, Ollama)
You can point the OpenAI driver to any local proxy or cloud provider (e.g. Voyage AI, Cohere, Together AI):

```python
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "not-needed"

db = EpochDB(
    storage_dir="./vllm_memory",
    embedding_model="openai:custom-embedding-model",
    dim=1024,
)
```

---

## 4. Ollama Local Service

Connect directly to an Ollama daemon running on your workstation:

```python
db = EpochDB(
    storage_dir="./ollama_memory",
    embedding_model="ollama:all-minilm",
    dim=384,
)
```
