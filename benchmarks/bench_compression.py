import os
import shutil
import time
import numpy as np
from epochdb.engine import EpochDB

def run_single_benchmark(run_id, method, level, num_atoms=3000, dim=384):
    storage_dir = f"./.bench_comp_{run_id}_{method}_{level if level is not None else 'default'}"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
        
    try:
        # Initialize EpochDB
        db = EpochDB(
            storage_dir=storage_dir,
            dim=dim,
            parquet_compression=method,
            parquet_compression_level=level,
            model=None,  # Disable auto-embedding to focus on storage engine performance
            wal_sync_interval=0.5  # Prevent blocking on synchronous fsyncs
        )
        
        # Optimize SQLite for the benchmark to prevent physical disk commits from blocking
        try:
            db.kg_manager._conn.execute("PRAGMA synchronous = OFF")
            db.kg_manager._conn.execute("PRAGMA journal_mode = MEMORY")
        except Exception:
            pass
        
        # Pre-generate atoms using a reproducible seed
        rng = np.random.default_rng(42)
        embeddings = rng.random((num_atoms, dim)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)
        
        # Build memory payloads & KG triples
        items = []
        for i in range(num_atoms):
            payload = f"This is memory atom number {i} containing standard text data for benchmarking compression ratios. The text needs to be long enough to test character and string redundancy across row groups."
            triples = [("entity_a", f"relation_{i % 10}", "entity_b")]
            items.append({
                "payload": payload,
                "embedding": embeddings[i],
                "triples": triples
            })
            
        # Write to WAL & Hot Tier
        start_write = time.time()
        db.add_memory_batch(items)
        
        # Flush to Cold Tier (Parquet) synchronously
        db.force_checkpoint()
        write_duration = (time.time() - start_write) * 1000  # in ms
        
        # Measure size of the generated Parquet file
        parquet_files = [f for f in os.listdir(storage_dir) if f.endswith(".parquet")]
        if not parquet_files:
            raise RuntimeError("No parquet file was generated after checkpoint.")
            
        file_path = os.path.join(storage_dir, parquet_files[0])
        file_size_kb = os.path.getsize(file_path) / 1024.0
        
        # Read back from Cold Tier
        epoch_id = parquet_files[0].replace(".parquet", "")
        start_read = time.time()
        loaded_atoms = db.cold_tier.load_epoch(epoch_id)
        read_duration = (time.time() - start_read) * 1000  # in ms
        
        if len(loaded_atoms) != num_atoms:
            raise RuntimeError(f"Loaded atom count mismatch: {len(loaded_atoms)} vs {num_atoms}")
            
        db.close()
        return {
            "supported": True,
            "size_kb": file_size_kb,
            "write_ms": write_duration,
            "read_ms": read_duration
        }
    except Exception as e:
        return {
            "supported": False,
            "error": str(e)
        }
    finally:
        if os.path.exists(storage_dir):
            try:
                shutil.rmtree(storage_dir)
            except Exception:
                pass

def main():
    print("Initializing Parquet Compression Benchmark...\n")
    
    # Configurations to test
    configs = [
        {"method": "NONE", "level": None},
        {"method": "SNAPPY", "level": None},
        {"method": "LZ4", "level": None},
        
        # GZIP levels
        {"method": "GZIP", "level": None},
        {"method": "GZIP", "level": 1},
        {"method": "GZIP", "level": 6},
        {"method": "GZIP", "level": 9},
        
        # BROTLI levels
        {"method": "BROTLI", "level": None},
        {"method": "BROTLI", "level": 1},
        {"method": "BROTLI", "level": 6},
        {"method": "BROTLI", "level": 11},
    ]
    
    # ZSTD levels
    for zstd_level in [-3, 3, 4, 7, 9, 22]:
        configs.append({"method": "ZSTD", "level": zstd_level})
        
    import uuid
    run_id = uuid.uuid4().hex[:8]
    
    results = []
    for cfg in configs:
        method = cfg["method"]
        level = cfg["level"]
        level_str = f"level {level}" if level is not None else "default"
        print(f"Testing {method} ({level_str})...")
        
        res = run_single_benchmark(run_id, method, level)
        res["method"] = method
        res["level"] = level
        results.append(res)
        
    # Print the report
    print("\n" + "=" * 90)
    print("                 EPOCHDB PARQUET COMPRESSION BENCHMARK REPORT")
    print("=" * 90)
    print(f"{'Method':<12} | {'Level':<8} | {'File Size (KB)':<16} | {'Ratio (vs NONE)':<16} | {'Write (ms)':<14} | {'Read (ms)':<14}")
    print("-" * 90)
    
    # Find baseline NONE size
    none_size = None
    for r in results:
        if r["method"] == "NONE" and r["supported"]:
            none_size = r["size_kb"]
            break
            
    for r in results:
        method = r["method"]
        level = str(r["level"]) if r["level"] is not None else "default"
        if not r["supported"]:
            print(f"{method:<12} | {level:<8} | {'Not Supported':<16} | {'N/A':<16} | {'N/A':<14} | {'N/A':<14}")
        else:
            ratio_str = "1.00x"
            if none_size and none_size > 0:
                ratio = none_size / r["size_kb"]
                ratio_str = f"{ratio:.2f}x"
            size_str = f"{r['size_kb']:.2f} KB"
            write_str = f"{r['write_ms']:.1f} ms"
            read_str = f"{r['read_ms']:.1f} ms"
            print(f"{method:<12} | {level:<8} | {size_str:<16} | {ratio_str:<16} | {write_str:<14} | {read_str:<14}")
    print("=" * 90)

if __name__ == "__main__":
    main()
