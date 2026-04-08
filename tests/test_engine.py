import os
import shutil
import pytest
import numpy as np
from epochdb import EpochDB

@pytest.fixture
def test_db():
    storage_dir = "./.test_epochdb_engine"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    db = EpochDB(storage_dir=storage_dir, dim=4)
    yield db
    db.close()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

def test_initialization(test_db):
    assert test_db.dim == 4
    assert os.path.exists(test_db.storage_dir)
    assert os.path.exists(os.path.join(test_db.storage_dir, "metadata.json"))

def test_add_memory(test_db):
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    id = test_db.add_memory("Test payload", emb)
    assert id is not None
    assert len(test_db.hot_tier.atoms) == 1

def test_dimensionality_enforcement(test_db):
    storage_dir = test_db.storage_dir
    test_db.close()
    
    # Try to re-open with wrong dimensions
    with pytest.raises(ValueError, match="Dimensionality mismatch"):
        db2 = EpochDB(storage_dir=storage_dir, dim=8)
        db2.close()

def test_tier_checkpoint(test_db):
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Fact 1", emb)
    assert len(test_db.hot_tier.atoms) == 1
    
    test_db.force_checkpoint()
    
    # Hot tier should be cleared
    assert len(test_db.hot_tier.atoms) == 0
    # Cold tier should have Parquet archives (verified by recall later)
    # Check if a parquet file was created
    epochs = [d for d in os.listdir(test_db.storage_dir) if d.startswith("epoch_")]
    assert len(epochs) >= 1
