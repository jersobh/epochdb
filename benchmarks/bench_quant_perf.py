import time
import numpy as np
import os
import shutil
from epochdb.engine import EpochDB
from epochdb.atom import ScalarPayload, SeriesPayload, SeriesPoint, ConstraintPayload

def benchmark_quant_perf():
    print("--- EpochDB Quantitative Performance Benchmark ---")
    storage_dir = "./.bench_quant_db"
    if os.path.exists(storage_dir): shutil.rmtree(storage_dir)
    
    with EpochDB(storage_dir=storage_dir, dim=128) as db:
        # 1. Scalar Range Query Latency (10k for speed, 1M target)
        print("\n1. Scalar Range Query (100k atoms)...")
        start_ingest = time.time()
        for i in range(100000):
            val = float(i)
            s = ScalarPayload(value=val, unit="C")
            db.add_memory(payload=s, embedding=np.random.rand(128).astype(np.float32), triples=[("room", "temp", str(val))])
        print(f"   Ingested 100k scalars in {time.time() - start_ingest:.2f}s")
        
        start_query = time.time()
        results = db.retriever.query_range("temp", 50000.0, 50100.0)
        query_time = (time.time() - start_query) * 1000
        print(f"   Range query [50000, 50100] found {len(results)} atoms in {query_time:.2f}ms")

        # 2. Series Interpolation Accuracy
        print("\n2. Series Interpolation Accuracy...")
        points = [SeriesPoint(timestamp=float(t), value=float(t*10), uncertainty_low=1.0, uncertainty_high=1.0) for t in range(0, 100, 10)]
        payload = SeriesPayload(points=points, unit="W")
        db.add_memory(payload=payload, embedding=np.zeros(128), triples=[("dev", "power", "series")])
        atom_id = [k for k,v in db.hot_tier.atoms.items() if v.payload_type.value == "series"][0]
        
        res = db.retriever.query_temporal(atom_id, 25.0)
        print(f"   Interpolated at 25.0: value={res['value']} (Expected 250.0), uncertainty={res['uncertainty']:.4f}")

        # 3. Constraint Feasibility Throughput
        print("\n3. Constraint Feasibility Throughput...")
        expr = {"op": "and", "left": {"op": ">", "field": "temp", "value": 20}, "right": {"op": "<", "field": "temp", "value": 30}}
        c_payload = ConstraintPayload(expression=expr)
        db.add_memory(payload=c_payload, embedding=np.zeros(128), triples=[("policy", "limit", "temp")])
        
        start_sat = time.time()
        for _ in range(100):
            db.retriever.check_feasibility([list(db.hot_tier.atoms.keys())[-1]], {"temp": 25.0})
        sat_time = (time.time() - start_sat) * 10
        print(f"   Constraint throughput: {sat_time:.2f}ms per 100 checks")

        # 4. Cascade Latency
        print("\n4. Cascade Latency...")
        cascade_count = [0]
        def on_update(atom): cascade_count[0] += 1
        db.hot_tier.quant_index.cascade_manager.register_dependency("temp", "bench_dep", on_update)
        
        start_cascade = time.time()
        db.add_memory(payload=ScalarPayload(value=30.0, unit="C"), embedding=np.zeros(128), triples=[("room", "temp", "30")])
        print(f"   Cascade propagation fired: {cascade_count[0]} times in {(time.time() - start_cascade)*1000:.2f}ms")

    if os.path.exists(storage_dir): shutil.rmtree(storage_dir)

if __name__ == "__main__":
    benchmark_quant_perf()
