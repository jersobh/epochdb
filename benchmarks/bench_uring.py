import os
import shutil
import time
import epochdb.core.transaction as tx

def run_wal_benchmark(name, use_helper, sync_interval, num_ops=5000):
    path = f"./.bench_wal_{name}.jsonl"
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
            
    # Set the helper toggle in transaction.py module
    tx._helper_loaded = use_helper
    
    try:
        # Initialize the WAL
        wal = tx.WriteAheadLog(path, sync_interval=sync_interval)
        
        # Prepare test payload
        data = {
            "id": "atom_12345",
            "payload": "This is a standard text log payload to benchmark serialization speed, direct memory copies, and fsync alignment performance.",
            "embedding": [0.1] * 384,
            "triples": [("a", "b", "c")]
        }
        
        # Ingestion loop
        start_time = time.time()
        for i in range(num_ops):
            wal.append("ADD", data)
        wal.append("COMMIT", {})
        write_duration = time.time() - start_time
        
        # Close the WAL to ensure all blocks are truncated and flushed
        wal.close()
        
        file_size_kb = os.path.getsize(path) / 1024.0
        
        # Replay performance
        # Force helper off for read, as read is always standard file read
        tx._helper_loaded = False
        wal_read = tx.WriteAheadLog(path)
        start_read = time.time()
        pending = wal_read.replay()
        read_duration = time.time() - start_read
        wal_read.close()
        
        # Verify correctness
        if len(pending) != num_ops:
            raise RuntimeError(f"Replay verified {len(pending)} records, expected {num_ops}")
            
        ops_sec = num_ops / write_duration
        
        return {
            "supported": True,
            "write_s": write_duration,
            "read_s": read_duration,
            "ops_sec": ops_sec,
            "size_kb": file_size_kb
        }
    except Exception as e:
        return {
            "supported": False,
            "error": str(e)
        }
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

def main():
    print("Initializing Write-Ahead Log Performance Benchmark...\n")
    print("Running append/sync test for 5,000 iterations per configuration...\n")
    
    configs = [
        {"name": "Standard (Sync fsync)", "helper": False, "sync_interval": 0.0},
        {"name": "Standard (Async 500ms)", "helper": False, "sync_interval": 0.5},
        {"name": "io_uring + O_DIRECT (Sync)", "helper": True, "sync_interval": 0.0},
        {"name": "io_uring + O_DIRECT (Async 500ms)", "helper": True, "sync_interval": 0.5},
    ]
    
    # Store original helper loaded state
    original_helper_loaded = tx._helper_loaded
    
    results = []
    for cfg in configs:
        name = cfg["name"]
        print(f"Benchmarking {name}...")
        res = run_wal_benchmark(cfg["name"].replace(" ", "_"), cfg["helper"], cfg["sync_interval"])
        res["name"] = name
        results.append(res)
        
    # Restore helper state
    tx._helper_loaded = original_helper_loaded
    
    print("\n" + "=" * 90)
    print("                     EPOCHDB WRITE-AHEAD LOG BENCHMARK REPORT")
    print("=" * 90)
    print(f"{'Configuration':<34} | {'Write (s)':<12} | {'Ops/Sec':<12} | {'Replay (ms)':<14} | {'File Size (KB)':<14}")
    print("-" * 90)
    
    for r in results:
        name = r["name"]
        if not r["supported"]:
            print(f"{name:<34} | {'Not Supported':<12} | {'N/A':<12} | {'N/A':<14} | {'N/A':<14}")
        else:
            write_str = f"{r['write_s']:.3f} s"
            ops_str = f"{r['ops_sec']:.1f}/s"
            read_str = f"{r['read_s']*1000:.1f} ms"
            size_str = f"{r['size_kb']:.1f} KB"
            print(f"{name:<34} | {write_str:<12} | {ops_str:<12} | {read_str:<14} | {size_str:<14}")
    print("=" * 90)

if __name__ == "__main__":
    main()
