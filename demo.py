import os
import time
import shutil
import logging
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer
from epochdb import EpochDB

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
logging.basicConfig(level=logging.ERROR)

# ANSI Colors for a premium terminal experience
CLR_YEL = "\033[93m"
CLR_CYN = "\033[96m"
CLR_GRN = "\033[92m"
CLR_BLU = "\033[94m"
CLR_RED = "\033[91m"
CLR_END = "\033[0m"
BOLD = "\033[1m"

def print_banner(text):
    print(f"\n{BOLD}{CLR_BLU}{'='*60}{CLR_END}")
    print(f"{BOLD}{CLR_CYN} {text} {CLR_END}")
    print(f"{BOLD}{CLR_BLU}{'='*60}{CLR_END}")

def main():
    storage_dir = "./.epochdb_demo"
    if os.path.exists(storage_dir):
        try:
            shutil.rmtree(storage_dir, ignore_errors=True)
            # Short sleep to let the OS release handles
            time.sleep(0.5) 
        except Exception:
            pass
        
    print_banner("EPOCHDB: REASONING MEMORY ENGINE DEMO")
    
    # 1. Initialize Local Model
    print(f"{CLR_YEL}Loading Local Embedding Model (all-MiniLM-L6-v2)...{CLR_END}")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    dim = 384
    db = EpochDB(storage_dir=storage_dir, dim=dim)

    # 2. The 'Detective' Scenario
    print_banner("STEP 1: Establshing Disparate Facts")
    
    facts = [
        ("The suspect, Project Zero, uses Parquet files.", [("Project Zero", "uses", "Parquet")]),
        ("Parquet is a columnar storage format.", [("Parquet", "is_a", "Columnar Format")]),
        ("Jeff is the lead developer of Project Zero.", [("Jeff", "leads", "Project Zero")])
    ]
    
    for payload, triples in facts:
        print(f"{CLR_GRN}Ingesting: {CLR_END}{payload}")
        emb = embedder.encode(payload)
        db.add_memory(payload=payload, embedding=emb, triples=triples)
        time.sleep(0.2)
    
    # 3. Semantic Retrieval Test
    print_banner("STEP 2: Basic Semantic Retrieval")
    query = "Tell me about Jeff"
    print(f"{CLR_YEL}Query: '{query}'{CLR_END}")
    
    query_emb = embedder.encode(query)
    # Standard retrieval (0 hops)
    results = db.recall(query_emb, top_k=1, expand_hops=0)
    
    for r in results:
        print(f"{BOLD}[HIT]{CLR_END} {r.payload} (Saliency: {r.calculate_saliency():.3f})")

    # 4. Relational Expansion Test (Multi-Hop)
    print_banner("STEP 3: Multi-Hop Reasoning (Relational Expansion)")
    complex_query = "What file format does Jeff's project use?"
    print(f"{CLR_YEL}Complex Query: '{complex_query}'{CLR_END}")
    print(f"{CLR_BLU}Theory: Normal vector stores fail this because 'Jeff' and 'Parquet' share no keywords.{CLR_END}")
    
    query_emb = embedder.encode(complex_query)
    # Relational retrieval (2 hops)
    # Jeff -> leads -> Project Zero -> uses -> Parquet
    results = db.recall(query_emb, top_k=3, expand_hops=2)
    
    print(f"\n{BOLD}EpochDB results (Bridging the gap):{CLR_END}")
    for i, r in enumerate(results):
        color = CLR_GRN if "Parquet" in r.payload else CLR_END
        print(f"{i+1}. {color}{r.payload}{CLR_END}")

    # 5. Tiered Storage Lifecycle
    print_banner("STEP 4: Tiered Storage (The Hot -> Cold Flush)")
    print(f"{CLR_YEL}Flushing Working Memory to Parquet Archive...{CLR_END}")
    db.force_checkpoint()
    
    print(f"\n{BOLD}Hot Tier status:{CLR_END} {len(db.hot_tier.atoms)} atoms (Cleared)")
    
    print(f"\n{CLR_YEL}Querying the Archive (Long-term Memory):{CLR_END}")
    results = db.recall(query_emb, top_k=2, expand_hops=2)
    for r in results:
        print(f"{BOLD}[COLD HIT]{CLR_END} {r.payload} (Epoch: {r.epoch_id})")

    db.close()
    print_banner("DEMO COMPLETE")
    print(f"Data persisted in {CLR_CYN}{storage_dir}{CLR_END}. Run again to see instant recall.")

if __name__ == "__main__":
    main()
