import os
import shutil
import pytest
import numpy as np
from epochdb import EpochDB

@pytest.fixture
def test_db():
    storage_dir = "./.test_epochdb_retrieval"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    db = EpochDB(storage_dir=storage_dir, dim=4)
    yield db
    db.close()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

def test_semantic_recall(test_db):
    emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Apple", emb1)
    
    emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Banana", emb2)
    
    # Query for Apple
    results = test_db.recall(emb1, top_k=1)
    assert len(results) == 1
    assert "Apple" in results[0].payload

def test_relational_expansion(test_db):
    # Fact 1: Jeff leads Project Zero
    # Fact 2: Project Zero uses Parquet
    emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Jeff leads Project Zero", emb1, triples=[("Jeff", "leads", "Project Zero")])
    
    emb2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    test_db.add_memory("Project Zero uses Parquet", emb2, triples=[("Project Zero", "uses", "Parquet")])
    
    # Query for Jeff (expand_hops=2)
    results = test_db.recall(emb1, top_k=5, expand_hops=2)
    
    payloads = [r.payload for r in results]
    assert any("Jeff" in p for p in payloads)
    assert any("Parquet" in p for p in payloads)

def test_payload_deduplication(test_db):
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Add same payload twice with different IDs
    test_db.add_memory("Repeat fact", emb)
    test_db.add_memory("Repeat fact", emb)
    
    results = test_db.recall(emb, top_k=5)
    
    payloads = [r.payload for r in results]
    # Should only return one instance of the payload
    assert payloads.count("Repeat fact") == 1
