import os
import time
import shutil
import logging
from sentence_transformers import SentenceTransformer
from epochdb import EpochDB

from .adapters import (
    EpochDBStoreAdapter, 
    ChromaDBStoreAdapter, 
    LanceDBStoreAdapter, 
    FAISSStoreAdapter, 
    QdrantStoreAdapter
)
from .convomem import ConvoMemBenchmark
from .longmemeval import LongMemEvalBenchmark
from .locomo import LoCoMoBenchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComparisonRunner")

def run_comparison():
    logger.info("Initializing Embedder (all-MiniLM-L6-v2)")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    dim = 384

    # Setup Databases
    epoch_dir = "./.epochdb_benchmark_comp"
    if os.path.exists(epoch_dir):
        shutil.rmtree(epoch_dir)
    epoch_db = EpochDB(storage_dir=epoch_dir, dim=dim)
    
    # Initialize Adapters
    epoch_adapter = EpochDBStoreAdapter(epoch_db)
    chroma_adapter = ChromaDBStoreAdapter(collection_name="benchmark_comp")
    lance_adapter = LanceDBStoreAdapter(uri="./.lancedb_benchmark_comp")
    faiss_adapter = FAISSStoreAdapter(dim=dim)
    qdrant_adapter = QdrantStoreAdapter(collection_name="benchmark_comp")

    stores = [
        ("EpochDB", epoch_adapter),
        ("ChromaDB", chroma_adapter),
        ("LanceDB", lance_adapter),
        ("FAISS", faiss_adapter),
        ("Qdrant", qdrant_adapter)
    ]

    benchmarks = [
        ConvoMemBenchmark,
        LongMemEvalBenchmark,
        LoCoMoBenchmark
    ]

    all_results = {}

    for bench_cls in benchmarks:
        bench_name = bench_cls.name
        all_results[bench_name] = {}
        
        for store_name, store_adapter in stores:
            logger.info(f"--- Running {bench_name} on {store_name} ---")
            store_adapter.clear()
            
            benchmark = bench_cls(store_adapter, embedder)
            
            # Ingest
            start_ingest = time.time()
            benchmark.ingest()
            end_ingest = time.time()
            ingest_time = end_ingest - start_ingest
            
            # Evaluate
            start_eval = time.time()
            metrics = benchmark.evaluate()
            end_eval = time.time()
            eval_time = end_eval - start_eval
            
            # Store results
            result_data = {
                "ingest_latency_sec": ingest_time,
                "eval_latency_sec": eval_time,
                "metrics": metrics
            }
            all_results[bench_name][store_name] = result_data
            
            logger.info(f"Finished {bench_name} on {store_name}. Ingest: {ingest_time:.2f}s, Eval: {eval_time:.2f}s")

    # Final Summary Table
    print("\n" + "="*80)
    print(f"{'Benchmark':<15} | {'Store':<10} | {'Ingest (s)':<10} | {'Eval (s)':<10} | {'Metrics'}")
    print("-" * 80)
    for bench_name, stores_res in all_results.items():
        for store_name, res in stores_res.items():
            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in res['metrics'].items()])
            print(f"{bench_name:<15} | {store_name:<10} | {res['ingest_latency_sec']:<10.2f} | {res['eval_latency_sec']:<10.2f} | {metrics_str}")
    print("="*80 + "\n")

    epoch_db.close()
    return all_results

if __name__ == "__main__":
    run_comparison()
