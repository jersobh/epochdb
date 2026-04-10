import os
import time
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

# 2. Advanced Mock Embedder (Showcasing Semantic Filtering)
class AdvancedMockEmbedder:
    def __init__(self, dim=384):
        self.dim = dim
        
    def encode(self, text: str) -> np.ndarray:
        text = text.lower()
        vec = np.zeros(self.dim, dtype=np.float32)
        
        # Determine vector space alignment
        if "coffee" in text or "janitor" in text:
            # Fluff Clues: Orthogonal Dimension (Should be strictly filtered by RRF)
            vec[0] = 1.0
        else:
            # Genuine Investigatory Clues: Shared semantic subspace
            np.random.seed(sum(ord(c) for c in text))
            vec = np.random.rand(self.dim).astype(np.float32)
            vec[1:101] = 1.0  # Increase signal to ~100 dims to ensure score > 0.15 filter
            
        return vec / (np.linalg.norm(vec) + 1e-10)

def main():
    print("====================================================")
    print("=  Cyber-Detective Simulator (EpochDB v0.2.0 Demo) =")
    print("====================================================\n")

    db_dir = "./.epochdb_advanced_demo"
    import shutil
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    print("[System] Initializing Tiered Engine (384D) & Asynchronous Daemon Hooks...")
    db = EpochDB(storage_dir=db_dir, dim=384)
    embedder = AdvancedMockEmbedder(dim=384)

    # 3. LangGraph Node Definitions
    def retrieve_memory(state: AgentState):
        query_emb = embedder.encode(state["input"])
        # Leveraging Semantic Edge Filtering (< 0.15 cutoff in engine)
        # We explicitly limit top_k to 1 so that everything else MUST be fetched
        # dynamically via Knowledge Graph edge connections!
        results = db.recall(query_emb, top_k=1, expand_hops=3)
        
        context_str = "\n".join([f" - {r.payload}" for r in results]) if results else "No leads found."
        print(f"\n[Detective Retina] Relevant Context Re-Assembled:\n{context_str}")
        return {"context": context_str}

    def generate_response(state: AgentState):
        resp_text = ""
        context = state["context"].lower()
        
        if "project chimera" in state["input"].lower():
            if "neon club" in context and "dr. aris" in context:
                resp_text = "Analysis complete. Dr. Aris handed off the code outside the Neon Club."
            else:
                resp_text = "I don't have enough interconnected data yet."
        else:
            resp_text = "Evidence logged."
            
        print(f"[Detective Core] Analysis: {resp_text}")
        return {"response": resp_text}

    def extract_and_store(state: AgentState):
        extracted = []
        text = state["input"]
        
        # Simulating NLP Entity Extraction
        if "Dr. Aris" in text and "Neon Club" in text:
            extracted.append(("Dr. Aris", "spotted_at", "Neon Club"))
        if "Neon Club" in text and "Project Chimera" in text:
            extracted.append(("Neon Club", "linked_to", "Project Chimera"))
        if "coffee" in text:
            # Irrelevant Super-Node bait
            extracted.append(("Dr. Aris", "purchased", "coffee"))

        payload = f"Clue: {text}"
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
    
    thread_config = {"configurable": {"thread_id": "case_file_042"}}

    # ================== EVIDENCE INGESTION ==================
    print("\n[Admin] Bulk Ingesting Spaced Evidence...")
    evidence = [
        "Dr. Aris visited the Neon Club at 2 AM.",
        "A suspicious datapad related to Project Chimera was found near Neon Club.",
        "Dr. Aris bought a dark-roast coffee and tipped the barista $2.", # Fluff clue (orthogonal)
        "The janitor at the Neon Club was complaining about his low pay.",
        "Project Chimera was originally scheduled to launch next Tuesday.",
        "Dr. Aris hates the rain and always carries an umbrella."
    ]
    
    for clue in evidence:
        app.invoke({"input": clue}, config=thread_config)
    
    # 5. Showcasing Asynchronous Flush
    print("\n[Admin] Forcing Epoch Checkpoint (Simulating end-of-week memory decay)...")
    db.force_checkpoint() # Wait natively
    
    # Let's inspect the parquet file format on disk for INT8
    print("\n================ INT8 Parquet Validation =================")
    pq_files = [f for f in os.listdir(db_dir) if f.endswith(".parquet")]
    if pq_files:
        filepath = os.path.join(db_dir, pq_files[0])
        table = pq.read_table(filepath)
        schema = table.schema
        print(f"[Storage Inspector] Target Archive: {pq_files[0]}")
        print(f"[Storage Inspector] Detected Embedding DataType: {schema.field('embedding').type}")
        if 'list<item: int8>' in str(schema.field('embedding').type):
             print(f"[Storage Inspector] >> SUCCESS: Vector array successfully down-casted to Scalar int8 footprint.")
    else:
        print("[Storage Inspector] No Parquet files flushed.")

    # 6. Relational Graph Evaluation
    print("\n================ CROSS-EPOCH REASONING TEST =================")
    print("Testing multi-hop context retrieval. We are querying 'Project Chimera', and it should hop through 'Neon Club' to find 'Dr. Aris', WITHOUT dragging in the irrelevant 'coffee' purchase.")
    
    app.invoke({"input": "What do we know about Project Chimera?"}, config=thread_config)
    
    db.close()
    print("\n[Done] System Shutting Down.")

if __name__ == "__main__":
    main()
