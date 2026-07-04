# tests/test_contextualized_retrieval.py
import pytest
import numpy as np
import shutil
import os
from epochdb import EpochDB

@pytest.fixture
def temp_db(tmp_path):
    storage_dir = str(tmp_path / "test_context_db")
    db = EpochDB(storage_dir=storage_dir, dim=4, embedding_model=None)
    yield db
    db.close()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

def test_contextualized_retrieval(temp_db):
    # Store a sequence of 3 memories (simulating turns of a conversation)
    # Using orthogonal vectors to verify exact matches
    emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    temp_db.remember("First turn: Hello, my name is Bob.", metadata={"session_id": "s1"}, triples=[("Bob", "name", "Bob")])
    
    emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    temp_db.remember("Second turn: I work in Boston.", metadata={"session_id": "s1"}, triples=[("Bob", "lives_in", "Boston")])
    
    emb3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    temp_db.remember("Third turn: Goodbye.", metadata={"session_id": "s1"}, triples=[("Bob", "status", "offline")])

    # Query for the middle turn (Second turn) with context_window=1
    results = temp_db.query("Boston", k=1, context_window=1)
    
    assert len(results) == 1
    nucleus = results[0]
    assert "Second turn" in nucleus.text
    
    # Check that context_neighbors contains all 3 turns
    assert "context_neighbors" in nucleus.metadata
    neighbors = nucleus.metadata["context_neighbors"]
    assert len(neighbors) == 3
    
    # Check chronological ordering of neighbors
    assert "First turn" in neighbors[0]["payload"]
    assert "Second turn" in neighbors[1]["payload"]
    assert "Third turn" in neighbors[2]["payload"]
