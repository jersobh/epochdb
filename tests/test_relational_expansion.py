import os
import pytest
import numpy as np
from epochdb import EpochDB

@pytest.fixture
def test_db(tmp_path):
    storage_dir = str(tmp_path / "relational_db")
    db = EpochDB(storage_dir=storage_dir, dim=4)
    yield db
    db.close()

def test_multihop_chain(test_db):
    """
    Test a 3-hop chain: Alice -> Aurora -> Helios -> Quantum.
    Querying 'Alice' should reach 'Quantum' with expand_hops=3.
    """
    # Fact 1: Alice leads Aurora
    emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Alice leads Team Aurora", emb1, triples=[("Alice", "leads", "Team Aurora")])

    # Fact 2: Aurora develops Helios
    emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Team Aurora develops Project Helios", emb2, triples=[("Team Aurora", "develops", "Project Helios")])

    # Fact 3: Helios uses Quantum
    emb3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    test_db.add_memory("Project Helios uses Quantum Core", emb3, triples=[("Project Helios", "uses", "Quantum Core")])

    # Add noise to ensure Quantum is pushed out of top-5 semantic pool
    for i in range(10):
        noise_emb = np.array([0.0, 0.1, 0.1, 0.1], dtype=np.float32) # Orthogonal to emb1
        test_db.add_memory(f"Noise {i}", noise_emb)

    # Query with Alice's embedding.
    # Without expansion, only Fact 1 should be returned.
    results_0 = test_db.recall(emb1, top_k=2, expand_hops=0)
    payloads_0 = [r.payload for r in results_0]
    assert any("Alice" in p for p in payloads_0)
    assert not any("Quantum" in p for p in payloads_0)

    # With expansion=3, we should reach Quantum.
    results_3 = test_db.recall(emb1, top_k=10, expand_hops=3)
    payloads_3 = [r.payload for r in results_3]
    assert any("Quantum" in p for p in payloads_3), f"Quantum not found in {payloads_3}"

def test_cross_epoch_expansion(test_db):
    """
    Fact A (Cold Tier) -> Entity X -> Fact B (Hot Tier).
    Fact B (Hot Tier) -> Entity Y -> Fact C (Cold Tier).
    """
    # Fact A: Alice works at Acme (Cold)
    emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Alice works at Acme", emb_a, triples=[("Alice", "works_at", "Acme")])
    test_db.force_checkpoint() # Move to cold

    # Fact B: Acme builds Rockets (Hot)
    emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Acme builds Rockets", emb_b, triples=[("Acme", "builds", "Rockets")])

    # Fact C: Rockets use Fuel (Cold)
    emb_c = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    test_db.add_memory("Rockets use Fuel", emb_c, triples=[("Rockets", "use", "Fuel")])
    test_db.force_checkpoint() # Move to cold

    # Query Alice (Hot Tier empty for Alice, she's Cold)
    results = test_db.recall(emb_a, top_k=10, expand_hops=2)
    payloads = [r.payload for r in results]
    
    assert any("Alice" in p for p in payloads)
    assert any("Rockets" in p for p in payloads)
    assert any("Fuel" in p for p in payloads)

def test_topic_lock_boost(test_db):
    """
    Topic Lock should grant +20 bonus to facts matching ORIGINAL query entities,
    but NOT to facts discovered purely through expansion if they don't match intent.
    """
    # Signal: Jeff's project is EpochDB
    emb_signal = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Jeff's project is EpochDB", emb_signal, triples=[("Jeff", "leads", "EpochDB")])

    # Expansion link: EpochDB uses Parquet
    emb_link = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("EpochDB uses Parquet", emb_link, triples=[("EpochDB", "uses", "Parquet")])

    # Noise: Bob lives in Paris
    emb_noise = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Bob lives in Paris", emb_noise)

    # Query "Jeff's project"
    results = test_db.recall(emb_signal, top_k=10, expand_hops=1, query_entities=["Jeff"])
    
    # Jeff fact should be top because of Topic Lock (+20)
    assert results[0].payload == "Jeff's project is EpochDB"
    # Parquet fact should be present but lower score (no Topic Lock boost unless 'Parquet' was in query)
    assert any("Parquet" in r.payload for r in results)

def test_entity_seeding_stage1a(test_db):
    """
    If a query has an entity but the semantic score is low, Stage 1a should still pull it in.
    """
    # Fact: 'X-57 Maxwell is an electric plane'
    # Embedding is orthogonal to our query.
    emb_maxwell = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    test_db.add_memory("X-57 Maxwell is an electric plane", emb_maxwell, triples=[("X-57", "is", "electric plane")])
    test_db.force_checkpoint() # Move to cold

    # Query: 'Tell me about the X-57' (semantic score will be 0)
    query_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Add noise to fill semantic pool
    for i in range(10):
        noise_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # Matches query perfectly
        test_db.add_memory(f"Noise {i}", noise_emb)

    # 1. Without entity seeding
    results_no_seed = test_db.recall(query_emb, top_k=1, query_entities=[])
    assert "Maxwell" not in "".join([r.payload for r in results_no_seed])

    # 2. With entity seeding
    results_seed = test_db.recall(query_emb, top_k=5, query_entities=["X-57"])
    assert any("Maxwell" in r.payload for r in results_seed)
