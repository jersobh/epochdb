import os
import logging
import warnings

# Suppress LangChain's Pydantic V1 warning on Python 3.14+
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

class GeminiEmbedder:
    def __init__(self, client):
        self.client = client
        self.model_name = 'gemini-embedding-2-preview'
        
    def encode(self, text: str) -> np.ndarray:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)

def main():
    print("Initializing Models & DB...")
    
    # Initialize the new Gemini SDK using standard client structure
    # (Set export GEMINI_API_KEY=YOUR_KEY in terminal before running)
    api_key = os.environ.get("GEMINI_API_KEY", "DUMMY_KEY_FOR_TESTING")
    client = genai.Client(api_key=api_key)
    
    # Initialize Gemini Embedder instead of Local SentenceTransformers
    embedder = GeminiEmbedder(client)
    
    try:
        # Dynamically fetch the dimensions so we don't have to guess if it's 768 or 1024
        print("Checking embedding dimensionality via dummy call...")
        dummy_emb = embedder.encode("test initialization")
        actual_dim = len(dummy_emb)
        print(f"Detected {actual_dim} dimensions for gemini-embedding-2-preview")
    except Exception as e:
        print(f"[Warning] API call failed, defaulting to 768 dims. Error: {e}")
        actual_dim = 768
        
    db = EpochDB(storage_dir="./.epochdb_agent", dim=actual_dim)

    # 2. Define Node Processing Functions
    def retrieve_memory(state: AgentState):
        """Pulls top contextual memories from EpochDB"""
        print(f"\n[Node: Retrieve] Triggered for -> '{state['input']}'")
        try:
            # We must use gemini API for embedding
            query_emb = embedder.encode(state["input"])
            results = db.recall(query_emb, top_k=2)
            
            if not results:
                context_str = "No prior memory."
            else:
                context_str = "\n".join([str(r.payload) for r in results])
        except Exception as e:
            print(f"[Warning] Failed to embed query via API: {e}")
            context_str = "No prior memory (API failed)."
            
        print(f"[Node: Retrieve] Found Context:\n{context_str}\n")
        return {"context": context_str}

    def generate_response(state: AgentState):
        """Calls Gemini using the newly injected context"""
        prompt = (f"You are an AI assistant powered by EpochDB.\n"
                  f"Relevant memory context:\n{state['context']}\n"
                  f"User: {state['input']}\n"
                  f"AI:")
        
        print(f"[Node: Generate] Calling Gemini...")
        try:
            # Calls the Google GenAI SDK
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            resp_text = response.text
        except Exception as e:
            # gracefully simulate output if no real API key is set
            print(f"[Warning] Gemini API Exception: {e}")
            if "Jeff" in state["context"]:
                resp_text = "I remember you are Jeff!"
            else:
                resp_text = "Nice to meet you! Assuming API is down, I am simulating a response."
                
        print(f"[Node: Generate] Reply: {resp_text}\n")
        return {"response": resp_text}

    def store_memory(state: AgentState):
        """Saves the interaction atomically to EpochDB"""
        print(f"[Node: Store] Archiving interaction to Working Memory...")
        memory_payload = f"User said: '{state['input']}' | AI answered: '{state['response']}'"
        
        try:
            emb = embedder.encode(memory_payload)
            # Save to Working Memory
            db.add_memory(payload=memory_payload, embedding=emb)
        except Exception as e:
            print(f"[Warning] Could not extract embedding to save: {e}")
            
        return state

    # 3. Assemble the LangGraph Workflow
    print("\nCompiling Agent Graph...")
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_memory)
    workflow.add_node("generate", generate_response)
    workflow.add_node("store", store_memory)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "store")
    workflow.add_edge("store", END)
    
    app = workflow.compile()
    
    # 4. Agent Execution Run
    print("\n================ RUNNING AGENT =================\n")
    interactions = [
        "Hello! My name is Jeff and I like programming.",
        "What is my name?"
    ]
    
    for turn in interactions:
        print(f"--- Processing Input: {turn} ---")
        inputs = {"input": turn}
        for output in app.stream(inputs):
            # stream output yields the state at each completed node
            pass
            
    db.close()
    
    # Cleanup dummy data
    import shutil
    shutil.rmtree("./.epochdb_agent", ignore_errors=True)

if __name__ == "__main__":
    main()
