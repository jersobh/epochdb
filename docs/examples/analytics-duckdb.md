# DuckDB SQL Aggregations

This example illustrates how to perform cross-epoch statistical queries over historical Parquet archives using the integrated DuckDB SQL engine.

---

## Runnable Example

```python
from epochdb import EpochDB

def run_duckdb_demo():
    with EpochDB(storage_dir="./duckdb_demo") as db:
        # Populate memories across multiple sessions
        for i in range(1, 21):
            category = "infrastructure" if i % 2 == 0 else "security"
            latency = 10.0 + (i * 2.5)
            db.remember(
                f"Operational event #{i} completed successfully.",
                metadata={
                    "category": category,
                    "execution_ms": latency,
                    "priority": i % 3,
                }
            )

        # Force a flush from Hot Tier to Cold Tier Parquet files
        db.flush()

        print("=== Executing Vectorized SQL over Cold Archives ===")
        # Query the cold_tier view
        rows = db.query_sql("""
            SELECT 
                json_extract_string(metadata, '$.category') AS category,
                COUNT(*) AS total_events,
                AVG(CAST(json_extract_string(metadata, '$.execution_ms') AS DOUBLE)) AS avg_latency_ms,
                MAX(CAST(json_extract_string(metadata, '$.execution_ms') AS DOUBLE)) AS max_latency_ms
            FROM cold_tier
            GROUP BY 1
            ORDER BY total_events DESC
        """)

        print(f"{'Category':<16} | {'Total':<6} | {'Avg Latency (ms)':<18} | {'Max Latency (ms)'}")
        print("-" * 65)
        for row in rows:
            print(f"{row['category']:<16} | {row['total_events']:<6} | {row['avg_latency_ms']:<18.2f} | {row['max_latency_ms']:.2f}")

if __name__ == "__main__":
    run_duckdb_demo()
```
