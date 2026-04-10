import os
import warnings
import numpy as np
import pyarrow.parquet as pq

# Suppress LangChain's Pydantic V1 warning
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

from typing import TypedDict
from langgraph.graph import StateGraph, END
from epochdb import EpochDB
from epochdb.checkpointer import EpochDBCheckpointer

# 1. State Definition
class AgentState(TypedDict):
    input: str
    context: str
    response: str
    extracted_triples: list[tuple]

# 2. Mock Embedder (Showcasing Semantic Filtering)
class KnowledgeMockEmbedder:
    def __init__(self, dim=384):
        self.dim = dim
        
    def encode(self, text: str) -> np.ndarray:
        text = text.lower()
        vec = np.zeros(self.dim, dtype=np.float32)
        
        # Determine vector space alignment
        if "weather" in text or "lunch" in text:
            # Irrelevant Fluff Clues: Orthogonal Dimension (Filtered out by RRF/Hops)
            vec[0] = 1.0
        else:
            # Core Project Data: Shared semantic subspace
            np.random.seed(sum(ord(c) for c in text))
            vec = np.random.rand(self.dim).astype(np.float32)
            vec[1:101] = 1.0  # Common signal for project-related info
            
        return vec / (np.linalg.norm(vec) + 1e-10)

def main():
    print("====================================================")
    print("=  Knowledge Base Assistant (EpochDB v0.3.0 Demo)  =")
    print("====================================================\n")

    db_dir = "./.epochdb_kb_demo"
    import shutil
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    print("[System] Initializing Tiered Engine & Background Persistence...")
    db = EpochDB(storage_dir=db_dir, dim=384)
    embedder = KnowledgeMockEmbedder(dim=384)

    # 3. LangGraph Node Definitions
    def retrieve_memory(state: AgentState):
        query_emb = embedder.encode(state["input"])
        
        # v0.3.0 Feature: Pass query entities to recall for prioritized ranking
        q_entities = ["Project Alpha"] if "alpha" in state["input"].lower() else []
        
        # Leveraging Semantic Edge Filtering + Entity-Centric RRF
        results = db.recall(query_emb, top_k=1, expand_hops=3, query_entities=q_entities)
        
        context_str = "\n".join([f" - {r.payload}" for r in results]) if results else "No related documents found."
        print(f"\n[Retriever] Context Re-Assembled:\n{context_str}")
        return {"context": context_str}

    def generate_response(state: AgentState):
        context = state["context"].lower()
        if "project alpha" in state["input"].lower():
            if "engineering dept" in context and "alice" in context:
                resp_text = "Project Alpha is overseen by Alice in the Engineering Department."
            else:
                resp_text = "I have some information on Project Alpha, but the department lead details are missing."
        else:
            resp_text = "Information archived."
            
        print(f"[Assistant] Response: {resp_text}")
        return {"response": resp_text}

    def extract_and_store(state: AgentState):
        extracted = []
        text = state["input"]
        
        # Simulate Relationship Extraction
        if "Alice" in text and "Engineering Dept" in text:
            extracted.append(("Alice", "leads", "Engineering Dept"))
        if "Engineering Dept" in text and "Project Alpha" in text:
            extracted.append(("Engineering Dept", "manages", "Project Alpha"))
        if "weather" in text:
            extracted.append(("Alice", "commented_on", "weather"))

        payload = f"KB Entry: {text}"
        emb = embedder.encode(payload)
        db.add_memory(payload=payload, embedding=emb, triples=extracted)
        return {"extracted_triples": extracted}

    # 4. Compilation
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_memory)
    workflow.add_node("generate", generate_response)
    workflow.add_node("extract_store", extract_and_store)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "extract_store")
    workflow.add_edge("extract_store", END)
    
    checkpointer = EpochDBCheckpointer(db)
    app = workflow.compile(checkpointer=checkpointer)
    
    thread_config = {"configurable": {"thread_id": "corporate_kb_001"}}

    # ================== DATA INGESTION ==================
    print("\n[Admin] Ingesting Corporate Records...")
    records = [
        "Alice is the primary lead for the Engineering Dept.",
        "The Engineering Dept manages the development of Project Alpha.",
        "Alice mentioned the weather was quite clear today." # Fluff entry
    ]
    
    for entry in records:
        app.invoke({"input": entry}, config=thread_config)
    
    # 5. Showcasing Asynchronous Flush to Cold Tier
    print("\n[Admin] Simulating Epoch Expiry (Moving data to Cold Tier)...")
    db.force_checkpoint() 
    
    # INT8 schema verification
    print("\n================ INT8 Parquet Validation =================")
    pq_files = [f for f in os.listdir(db_dir) if f.endswith(".parquet")]
    if pq_files:
        filepath = os.path.join(db_dir, pq_files[0])
        table = pq.read_table(filepath)
        print(f"[Storage] Target Archive: {pq_files[0]}")
        print(f"[Storage] Embedding DataType: {table.schema.field('embedding').type}")
        if 'int8' in str(table.schema.field('embedding').type):
             print(f"[Storage] >> SUCCESS: INT8 Quantization confirmed.")

    # 6. Cross-Epoch Multi-Hop Reasoning
    print("\n================ MULTI-HOP RESEARCH TEST =================")
    print("Goal: Retrieve information about 'Project Alpha'.")
    print("Expected: Connect 'Project Alpha' -> 'Engineering Dept' -> 'Alice' across Parquet files.")
    
    app.invoke({"input": "What is the status of Project Alpha and who leads it?"}, config=thread_config)
    
    db.close()
    print("\n[Done] Showcase complete.")

if __name__ == "__main__":
    main()
