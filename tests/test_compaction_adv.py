import os
import shutil
import tempfile
import pytest
import numpy as np
from epochdb.api.db import EpochDB

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)

def test_compaction_no_epochs(temp_dir):
    """Verify compaction does nothing if there are no historical epochs to compact."""
    db = EpochDB(storage_dir=temp_dir, dim=4)
    
    # Run compaction
    db.compact()
    
    # Verify no compacted files are generated
    files = os.listdir(temp_dir)
    compacted_files = [f for f in files if f.startswith("compacted_")]
    assert len(compacted_files) == 0
    db.close()

def test_compaction_pruning_and_supersession(temp_dir):
    """Verify compaction successfully removes soft-deleted and superseded memories, and updates global index."""
    db = EpochDB(storage_dir=temp_dir, dim=4)
    
    # 1. Ingest initial memories in Epoch 1
    # Standard memory that should remain
    id_keep = db.remember("Keep this memory", metadata={"triples": [("Keep", "status", "good")]})
    # Memory that will be soft-deleted
    id_delete = db.remember("Delete this memory", metadata={"triples": [("Delete", "status", "bad")]})
    # Memory that will be superseded (text update)
    id_supersede = db.remember("Old description of Alice", metadata={"triples": [("Alice", "description", "Old")]})
    
    # Force checkpoint to freeze Epoch 1 into Cold Tier
    db.force_checkpoint()
    
    # 2. Ingest updates in Epoch 2
    # Soft delete the delete-target memory
    db.delete(id_delete, hard=False)
    # Supersede Alice's description
    db.remember("New description of Alice", metadata={"triples": [("Alice", "description", "New")]})
    
    # Force checkpoint to freeze Epoch 2 into Cold Tier
    db.force_checkpoint()
    
    # At this point, we have 2 historical epochs in the Cold Tier.
    epochs_before = db.cold_tier.get_all_epochs()
    assert len(epochs_before) >= 2
    
    # Ensure all memories exist before compaction (logical view)
    m_keep = db.get(id_keep)
    assert m_keep is not None
    assert m_keep.text == "Keep this memory"
    
    m_delete = db.get(id_delete)
    assert m_delete is None  # Soft-deleted, get() returns None
    
    # Run compaction
    db.compact()
    
    # Verify historical epochs have been consolidated
    epochs_after = db.cold_tier.get_all_epochs()
    # Should only have the consolidated compacted epoch (and any active empty epoch if generated)
    compacted_epochs = [e for e in epochs_after if e.startswith("compacted_")]
    assert len(compacted_epochs) == 1
    
    # Verify non-compacted old files are cleaned up
    for old_epoch in epochs_before:
        assert not os.path.exists(os.path.join(temp_dir, f"{old_epoch}.parquet"))
        assert not os.path.exists(os.path.join(temp_dir, f"{old_epoch}.hnsw"))
        
    # Verify data integrity post-compaction
    # Standard kept memory should still be retrievable
    m_keep_after = db.get(id_keep)
    assert m_keep_after is not None
    assert m_keep_after.text == "Keep this memory"
    
    # Soft-deleted memory should be completely gone from the KG manager indices
    associations_deleted = db.kg_manager.get_associations("Delete")
    assert len(associations_deleted) == 0
    
    # Check that Alice's timeline contains only the active description
    history = db.get_entity_history("Alice")
    assert len(history) == 1
    assert history[0].payload == "New description of Alice"
    
    db.close()
