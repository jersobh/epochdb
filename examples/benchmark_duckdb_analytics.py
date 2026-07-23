"""
benchmark_duckdb_analytics.py — DuckDB Cold Tier Analytics Benchmark
========================================================================
Measures query throughput, execution latency, and numerical accuracy
comparing PyArrow dataset scanning with DuckDB vectorized SQL analytics
over EpochDB historical Parquet archives.
"""

import os
import time
import shutil
import numpy as np
from epochdb import EpochDB
from epochdb.atom import ScalarPayload
from epochdb.storage.cold_tier import ColdTierAnalytics


def run_analytics_benchmark(num_epochs: int = 3, atoms_per_epoch: int = 500):
    storage_dir = "./.benchmark_duckdb_storage"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    total_atoms = num_epochs * atoms_per_epoch
    print("=" * 70)
    print(f"🚀 EPOCHDB DUCKDB ANALYTICS BENCHMARK ({total_atoms:,} Atoms Across {num_epochs} Parquet Files)")
    print("=" * 70)

    print(f"-> Generating & Flushing {total_atoms:,} atoms into Cold Tier Parquet archives...")
    db = EpochDB(storage_dir=storage_dir, dim=64)
    
    np.random.seed(42)
    units = ["degC", "degF", "K"]
    
    start_ingest = time.time()
    for ep in range(num_epochs):
        for i in range(atoms_per_epoch):
            val = float(ep * 100 + (i % 50))
            u = units[i % len(units)]
            s = ScalarPayload(value=val, unit=u)
            emb = np.random.rand(64).astype(np.float32)
            db.add_memory(
                payload=s,
                embedding=emb,
                triples=[(f"sensor_{i % 10}", "temperature", f"val_{val}")]
            )
        db.force_checkpoint()
    
    db.close()
    ingest_time = time.time() - start_ingest
    print(f"   ✓ Cold storage populated in {ingest_time:.2f}s.\n")

    analytics = ColdTierAnalytics(storage_dir)

    # 1. Benchmark PyArrow Dataset Scanning
    print("-> 1. Running PyArrow Linear Dataset Scan & Python Loop...")
    t0 = time.time()
    table = analytics._get_dataset_table("temperature", "sensor_0")
    if table is not None:
        pyarrow_count = len(table)
        pyarrow_avg = float(np.mean(table["scalar_value"].to_numpy()))
    else:
        pyarrow_count, pyarrow_avg = 0, 0.0
    t_pyarrow = (time.time() - t0) * 1000.0

    # 2. Benchmark DuckDB Vectorized SQL Querying (Cold vs Warm)
    print("-> 2. Running DuckDB Vectorized SQL Query...")
    # Warmup / Cold run
    _ = analytics.query_sql("SELECT COUNT(*) FROM cold_tier")

    t0 = time.time()
    sql_res = analytics.query_sql("""
        SELECT 
            COUNT(*) as count, 
            AVG(scalar_value) as avg_val,
            MIN(scalar_value) as min_val,
            MAX(scalar_value) as max_val
        FROM cold_tier
        WHERE triples LIKE '%sensor_0%' AND triples LIKE '%temperature%'
    """)
    t_duckdb = (time.time() - t0) * 1000.0

    duckdb_count = sql_res[0]["count"] if sql_res else 0
    duckdb_avg = sql_res[0]["avg_val"] if sql_res else 0.0

    # 3. Complex Aggregation & Group-By Benchmark
    print("-> 3. Running Complex Group-By Analytical Aggregation with DuckDB...")
    t0 = time.time()
    group_res = analytics.query_sql("""
        SELECT 
            scalar_unit,
            COUNT(*) as atom_count,
            AVG(scalar_value) as avg_val,
            STDDEV(scalar_value) as std_val
        FROM cold_tier
        GROUP BY scalar_unit
        ORDER BY atom_count DESC
    """)
    t_duckdb_group = (time.time() - t0) * 1000.0

    # Accuracy verification
    count_match = (pyarrow_count == duckdb_count)
    avg_match = np.isclose(pyarrow_avg, duckdb_avg, rtol=1e-4) if pyarrow_count > 0 else True
    accuracy_score = 100.0 if (count_match and avg_match) else 0.0

    speedup = t_pyarrow / t_duckdb if t_duckdb > 0 else 1.0

    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"• Dataset Size                 : {total_atoms:,} Atoms ({num_epochs} Parquet archives)")
    print(f"• PyArrow Dataset Scan Latency : {t_pyarrow:.2f} ms")
    print(f"• DuckDB Vectorized SQL Latency: {t_duckdb:.2f} ms")
    print(f"• DuckDB Speedup               : {speedup:.2f}x faster")
    print(f"• DuckDB Group-By Latency      : {t_duckdb_group:.2f} ms")
    print(f"• Numerical Accuracy Match     : {'PERFECT (100%)' if accuracy_score == 100.0 else 'FAILED'}")
    print(f"• Verified Row Count           : {duckdb_count:,} records matching criteria")
    print("=" * 70)

    # Cleanup
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    return {
        "accuracy_score": accuracy_score,
        "speedup": speedup,
        "t_pyarrow_ms": t_pyarrow,
        "t_duckdb_ms": t_duckdb,
        "t_duckdb_group_ms": t_duckdb_group,
    }


if __name__ == "__main__":
    run_analytics_benchmark()
