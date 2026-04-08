import os
import shutil
import time
import psutil
import numpy as np
import logging
from epochdb import EpochDB
from .adapters import (
    EpochDBStoreAdapter, 
    ChromaDBStoreAdapter,
    LanceDBStoreAdapter,
    FAISSStoreAdapter,
    QdrantStoreAdapter
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ScalabilityTest")

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def run_scalability_test(n_items=10000, dim=384):
    logger.info(f"Starting 5-Way Scalability Test with {n_items} items (dim={dim})")
    
    # Setup Data Paths
    data_dirs = {
        "EpochDB": "./.epochdb_scalability",
        "ChromaDB": "./.chromadb_scalability",
        "LanceDB": "./.lancedb_scalability",
        "Qdrant": "./.qdrant_scalability" # Qdrant is :memory: but we list it for consistency
    }
    
    for name, d in data_dirs.items():
        if os.path.exists(d):
            shutil.rmtree(d)

    # Initialize Adapters
    epoch_engine = EpochDB(storage_dir=data_dirs["EpochDB"], dim=dim, epoch_duration_secs=5)
    
    stores = [
        ("EpochDB", EpochDBStoreAdapter(epoch_engine)),
        ("ChromaDB", ChromaDBStoreAdapter(collection_name="scalability_test")),
        ("LanceDB", LanceDBStoreAdapter(uri=data_dirs["LanceDB"])),
        ("FAISS", FAISSStoreAdapter(dim=dim)),
        ("Qdrant", QdrantStoreAdapter(collection_name="scalability_test"))
    ]

    all_results = {}
    batch_size = 100
    
    for name, store in stores:
        logger.info(f"--- Profiling {name} ---")
        store.clear()
        
        start_mem = get_process_memory()
        results = []
        
        start_time = time.time()
        for i in range(0, n_items, batch_size):
            batch_payloads = [f"Memory atom {j}" for j in range(i, i + batch_size)]
            batch_embeddings = [np.random.rand(dim).astype(np.float32) for _ in range(batch_size)]
            
            store.add_batch(batch_payloads, batch_embeddings)
            
            # Tracking at 1,000 item checkpoints
            current_count = i + batch_size
            if current_count % 1000 == 0:
                current_mem = get_process_memory() - start_mem
                logger.info(f"[{name}] Ingested {current_count} items. Memory Delta: {current_mem:.2f} MB")
                results.append((current_count, current_mem))
                
                if name == "EpochDB":
                    # Let the time-based epoch check trigger flushes if duration has passed
                    time.sleep(1.1) 

        end_time = time.time()
        logger.info(f"{name} took {end_time - start_time:.2f}s for {n_items} items.")
        all_results[name] = results

    # Print Comparison Table
    print("\n" + "="*90)
    header = f"{'Items':<10}"
    for name, _ in stores:
        header += f" | {name:<12}"
    print(header)
    print("-" * 90)
    
    for idx in range(n_items // 1000):
        row = f"{(idx+1)*1000:<10}"
        for name, res in all_results.items():
            row += f" | {res[idx][1]:<12.2f}"
        print(row)
    print("="*90 + "\n")

    epoch_engine.close()

if __name__ == "__main__":
    run_scalability_test()

if __name__ == "__main__":
    run_scalability_test()
