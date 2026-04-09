import argparse
import logging
from sentence_transformers import SentenceTransformer
from epochdb import EpochDB

from .convomem import ConvoMemBenchmark
from .longmemeval import LongMemEvalBenchmark
from .locomo import LoCoMoBenchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BenchmarkRunner")

def run_benchmarks(run_convo=False, run_long=False, run_loco=False):
    logger.info("Initializing Embedder (all-MiniLM-L6-v2)")
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        dim = 384
    except Exception as e:
        logger.error(f"Failed to load sentence-transformers: {e}")
        return

    logger.info("Initializing EpochDB")
    # Clean DB for test
    import shutil
    import os
    if os.path.exists("./.epochdb_benchmark"):
        shutil.rmtree("./.epochdb_benchmark")
        
    db = EpochDB(storage_dir="./.epochdb_benchmark", dim=dim)
    
    from .adapters import EpochDBStoreAdapter
    store = EpochDBStoreAdapter(db)
    
    adapters = []
    if run_convo:
        adapters.append(ConvoMemBenchmark(store, embedder))
    if run_long:
        adapters.append(LongMemEvalBenchmark(store, embedder))
    if run_loco:
        adapters.append(LoCoMoBenchmark(store, embedder))
        
    if not adapters:
        adapters = [
            ConvoMemBenchmark(store, embedder),
            LongMemEvalBenchmark(store, embedder),
            LoCoMoBenchmark(store, embedder)
        ]

    for adapter in adapters:
        logger.info(f"=== Starting Benchmark {adapter.name} ===")
        db.hot_tier.clear()
        db.wal.clear()
        
        adapter.ingest()
        metrics = adapter.evaluate()
        
        logger.info(f"=== {adapter.name} Results ===")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.3f}")
        print("\n")

    db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--convo", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--loco", action="store_true")
    args = parser.parse_args()
    
    run_benchmarks(args.convo, args.long, args.loco)
