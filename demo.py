import numpy as np
import logging
from epochdb import EpochDB

logging.basicConfig(level=logging.INFO)

def main():
    print("--- Initializing EpochDB ---")
    db = EpochDB(storage_dir="./.epochdb_test", dim=4)

    # 1. Create a dummy atom
    print("\n--- Adding Memory Atoms ---")
    emb1 = np.array([1.0, 0.0, 0.0, 0.0])
    id1 = db.add_memory(
        payload="User is named Jeff.",
        embedding=emb1,
        triples=[("Jeff", "is_a", "User")]
    )
    print(f"Added atom {id1}")

    emb2 = np.array([0.9, 0.1, 0.0, 0.0])
    id2 = db.add_memory(
        payload="Jeff works on Agentic Memory.",
        embedding=emb2,
        triples=[("Jeff", "works_on", "Agentic Memory")]
    )
    print(f"Added atom {id2}")
    
    emb3 = np.array([0.0, 0.0, 1.0, 0.0])
    id3 = db.add_memory(
        payload="The sky is blue.",
        embedding=emb3,
        triples=[("Sky", "has_color", "Blue")]
    )

    # 2. Querying (Hot Tier)
    print("\n--- Querying Working Memory ---")
    results = db.recall(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
    for i, r in enumerate(results):
        print(f"Rank {i+1}: {r.payload} | Saliency: {r.calculate_saliency():.3f} | Access: {r.access_count}")

    # Query again to see Saliency increase
    print("\n--- Querying Again (Observe Saliency Increase) ---")
    results = db.recall(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
    for i, r in enumerate(results):
        print(f"Rank {i+1}: {r.payload} | Saliency: {r.calculate_saliency():.3f} | Access: {r.access_count}")


    # 3. Force Epoch Checkpoint (Hot -> Cold)
    print("\n--- Forcing Epoch Checkpoint (Moving to Cold Tier) ---")
    db.force_checkpoint()
    
    print(f"Hot Tier size after checkpoint: {len(db.hot_tier.atoms)}")

    # 4. Querying (Cold Tier)
    print("\n--- Querying Historical Archive ---")
    results = db.recall(np.array([0.9, 0.1, 0.0, 0.0]), top_k=2)
    for i, r in enumerate(results):
        print(f"Rank {i+1}: {r.payload} | Epoch: {r.epoch_id}")

    db.close()
    
    # Cleanup for repeated tests
    import shutil
    shutil.rmtree("./.epochdb_test")

if __name__ == "__main__":
    main()
