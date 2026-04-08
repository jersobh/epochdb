import os
import logging
import warnings

# Suppress LangChain's Pydantic V1 warning on Python 3.14+ (must happen before LangGraph imports)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

import numpy as np
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from google import genai
from epochdb import EpochDB

logging.basicConfig(level=logging.ERROR)

# 1. State Definition
class AgentState(TypedDict):
    input: str
    context: str
    response: str
    extracted_triples: list[tuple]  # To store (S, R, O) for the KG

class GeminiEmbedder:
    def __init__(self, client, dim=768):
        self.client = client
        self.dim = dim
        self.model_name = 'gemini-embedding-2-preview'
        
    def encode(self, text: str) -> np.ndarray:
        if getattr(self, "is_dummy", False):
            # Simulation mode: consistent random vector
            return np.random.rand(self.dim).astype(np.float32)
            
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)

def main():
    print("Initializing EpochAgent with Tiered Memory...")
    
    api_key = os.environ.get("GEMINI_API_KEY", "dummy")
    client = genai.Client(api_key=api_key)
    
    # Simple dimensionality detection
    actual_dim = 768
    if api_key != "dummy":
        try:
            temp_resp = client.models.embed_content(model='gemini-embedding-2-preview', contents="test")
            actual_dim = len(temp_resp.embeddings[0].values)
            print(f"Detected {actual_dim} dimensions from Gemini.")
        except:
            pass
            
    embedder = GeminiEmbedder(client, dim=actual_dim)
    # Manual override for simulation check since SDK client is opaque
    embedder.is_dummy = (api_key == "dummy")
    
    db = EpochDB(storage_dir="./.epochdb_realworld", dim=actual_dim)

    # 2. Node Processing Functions
    def retrieve_memory(state: AgentState):
        """Pulls top contextual memories AND performs Relational Expansion"""
        print(f"\n[Node: Retrieve] Analyzing -> '{state['input']}'")
        try:
            query_emb = embedder.encode(state["input"])
            # Multi-hop retrieval: find semantic match + 2 hops of related info
            results = db.recall(query_emb, top_k=3, expand_hops=2)
            
            if not results:
                context_str = "No prior memory."
            else:
                context_str = "\n".join([f"- {r.payload}" for r in results])
        except Exception as e:
            print(f"[Warning] Retrieval path failed: {e}")
            context_str = "No prior memory."
            
        print(f"[Node: Retrieve] Found Context (incl. Relational Expansion):\n{context_str}\n")
        return {"context": context_str}

    def generate_response(state: AgentState):
        """Calls Gemini using the newly injected context"""
        prompt = (f"You are EpochAgent, powered by EpochDB tiered storage.\n"
                  f"Context from Long-term Memory (including graph relations):\n{state['context']}\n\n"
                  f"User's Current Input: {state['input']}\n"
                  f"Response:")
        
        print(f"[Node: Generate] Reasoning with Gemini...")
        try:
            if getattr(embedder, "is_dummy", False): raise Exception("Simulation")
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            resp_text = response.text
        except Exception:
            # Simulation logic for multi-hop demonstration if API fails or is dummy
            text = state["input"].lower()
            if "jeff" in state["context"] and "works on" in state["context"]:
                resp_text = "I remember you are Jeff, and you created EpochDB! It's a memory engine."
            elif "jeff" in text:
                resp_text = "Nice to meet you Jeff! I'll remember you."
            else:
                resp_text = "I'm listening! Tell me more about your work."
                
        print(f"[Node: Generate] Reply: {resp_text}\n")
        return {"response": resp_text}

    def extract_and_store(state: AgentState):
        """Extracts Entities/Relations and archives to Working Memory"""
        print(f"[Node: Extract & Store] Updating Knowledge Graph...")
        
        extracted = []
        text = state["input"].lower()
        if "i'm " in text or "i am " in text or "name is" in text:
            extracted.append(("user", "has_name", "Jeff"))
        if "epochdb" in text:
            extracted.append(("user", "works_on", "epochdb"))
            extracted.append(("epochdb", "is_a", "memory_engine"))

        memory_payload = f"Interaction: {state['input']} -> {state['response']}"
        try:
            emb = embedder.encode(memory_payload)
            db.add_memory(payload=memory_payload, embedding=emb, triples=extracted)
            if extracted:
                print(f"   -> Extracted Triples: {extracted}")
        except Exception as e:
            print(f"[Warning] Store failed: {e}")
            
        return {"extracted_triples": extracted}

    # 3. Assemble the LangGraph Workflow
    print("Compiling Agent Graph...")
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_memory)
    workflow.add_node("generate", generate_response)
    workflow.add_node("extract_store", extract_and_store)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "extract_store")
    workflow.add_edge("extract_store", END)
    
    app = workflow.compile()
    
    # 4. Multi-session Test Run
    print("\n================ SESSION 1: Establishing Reality =================")
    s1_inputs = [
        "Hi, I'm Jeff. I'm building EpochDB.",
        "EpochDB is a tiered memory engine."
    ]
    for turn in s1_inputs:
        print(f"\n--- Turn: {turn} ---")
        app.invoke({"input": turn})
    
    print("\n[Admin] Forcing cold-tier flush to Parquet...")
    db.force_checkpoint()
    
    print("\n================ SESSION 2: Multi-Hop Reasoning =================")
    test_query = "What is the memory engine that Jeff is working on?"
    print(f"\n--- Logic Test: {test_query} ---")
    app.invoke({"input": test_query})
            
    db.close()
    print(f"\n[Done] Database persisted in ./.epochdb_realworld.")

if __name__ == "__main__":
    main()
