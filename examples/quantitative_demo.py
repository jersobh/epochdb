import numpy as np
import time
from epochdb.engine import EpochDB
from epochdb.atom import ScalarPayload, SeriesPayload, SeriesPoint, ConstraintPayload, PayloadType

def quantitative_demo():
    print("--- EpochDB Quantitative Logic Demo ---")
    
    # Initialize DB (memory-only for demo if we use a temp dir)
    dim = 128
    storage_dir = "./.quant_demo_db"
    
    with EpochDB(storage_dir=storage_dir, dim=dim) as db:
        # 1. CASCADE TRIGGER: Set up a trigger for "temperature"
        def on_temp_update(atom):
            val = atom.payload.value
            print(f"  [TRIGGER] Temperature updated to {val}°C. Checking policy...")
            if val > 25.0:
                print("  [CASCADE] Alert: Temperature exceeds threshold! Triggering cooling policy.")
        
        db.hot_tier.quant_index.add_trigger("temperature", on_temp_update)

        # 2. SCALAR: Ingest a temperature reading
        print("\n1. Ingesting Scalar Data...")
        emb = np.random.rand(dim).astype(np.float32)
        s_payload = ScalarPayload(value=22.5, unit="C", uncertainty=0.1)
        db.add_memory(
            payload=s_payload, 
            embedding=emb, 
            triples=[("Room_101", "temperature", "22.5")]
        )
        
        # This update should trigger the cascade
        s_payload_hot = ScalarPayload(value=28.0, unit="C", uncertainty=0.1)
        db.add_memory(
            payload=s_payload_hot, 
            embedding=emb, 
            triples=[("Room_101", "temperature", "28.0")]
        )

        # 3. RANGE QUERY: Find all atoms with temperature between 20 and 30
        print("\n2. Executing Range Query...")
        results = db.retriever.query_range("temperature", 20.0, 30.0, unit="C")
        print(f"  Found {len(results)} atoms in range [20, 30]°C.")
        for r in results:
            print(f"  - Atom {r.id}: {r.payload.value}{r.payload.unit}")

        # 4. SERIES: Ingest a power consumption series
        print("\n3. Ingesting Time-Series Data...")
        points = [
            SeriesPoint(timestamp=1700000000.0, value=100.0),
            SeriesPoint(timestamp=1700003600.0, value=150.0),
            SeriesPoint(timestamp=1700007200.0, value=120.0),
        ]
        ser_payload = SeriesPayload(points=points, unit="W")
        ser_id = db.add_memory(
            payload=ser_payload, 
            embedding=emb, 
            triples=[("Building_A", "power_usage", "Series")]
        )

        # 5. INTERPOLATION: Query series value at a specific time
        query_ts = 1700001800.0 # Exactly halfway between point 0 and 1
        val = db.retriever.query_temporal(ser_id, query_ts)
        print(f"  Interpolated power usage at {query_ts}: {val}W")

        # 6. CONSTRAINTS: Check feasibility of a spending rule
        print("\n4. Checking Constraint Feasibility...")
        # Rule: budget > 1000 AND expense < budget
        expr = {
            "op": "and",
            "left": {"op": ">", "field": "budget", "value": 1000},
            "right": {"op": "<", "field": "expense", "value": 1500} # Reference to 'budget' var ideally, but simplified here
        }
        c_payload = ConstraintPayload(expression=expr)
        c_id = db.add_memory(payload=c_payload, embedding=emb, triples=[("Policy_V1", "rules", "spending")])

        # State 1: Healthy
        state_ok = {"budget": 2000, "expense": 1200}
        is_ok = db.retriever.check_feasibility([c_id], state_ok)
        print(f"  State {state_ok} feasible? {is_ok}")

        # State 2: Violation
        state_bad = {"budget": 800, "expense": 1200}
        is_bad = db.retriever.check_feasibility([c_id], state_bad)
        print(f"  State {state_bad} feasible? {is_bad}")

    print("\nDemo Complete.")

if __name__ == "__main__":
    import shutil
    import os
    if os.path.exists("./.quant_demo_db"):
        shutil.rmtree("./.quant_demo_db")
    
    quantitative_demo()
    
    if os.path.exists("./.quant_demo_db"):
        shutil.rmtree("./.quant_demo_db")
