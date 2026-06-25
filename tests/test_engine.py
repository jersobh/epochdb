import os
import shutil
import time
import pytest
import numpy as np
from epochdb import EpochDB
from epochdb.transaction import WriteAheadLog


@pytest.fixture
def storage_dir(tmp_path):
    d = str(tmp_path / "test_engine_db")
    yield d
    if os.path.exists(d):
        shutil.rmtree(d)


@pytest.fixture
def test_db(storage_dir):
    db = EpochDB(storage_dir=storage_dir, dim=4)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------

def test_initialization(test_db, storage_dir):
    assert test_db.dim == 4
    assert os.path.exists(storage_dir)
    assert os.path.exists(os.path.join(storage_dir, "metadata.json"))


def test_add_memory(test_db):
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    atom_id = test_db.add_memory("Test payload", emb)
    assert atom_id is not None
    assert len(test_db.hot_tier.atoms) == 1


def test_dimensionality_enforcement(test_db, storage_dir):
    test_db.close()
    # Should automatically adjust dim to the stored dimension (4) instead of raising ValueError
    db2 = EpochDB(storage_dir=storage_dir, dim=8)
    assert db2.dim == 4
    db2.close()


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------

def test_context_manager(storage_dir):
    with EpochDB(storage_dir=storage_dir, dim=4) as db:
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        db.add_memory("context manager test", emb)
    # Lock file should be gone after __exit__.
    assert not os.path.exists(os.path.join(storage_dir, ".lock"))


# ---------------------------------------------------------------------------
# Checkpoint (sync)
# ---------------------------------------------------------------------------

def test_tier_checkpoint(test_db, storage_dir):
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Fact 1", emb)
    assert len(test_db.hot_tier.atoms) == 1

    test_db.force_checkpoint()

    assert len(test_db.hot_tier.atoms) == 0
    parquet_files = [f for f in os.listdir(storage_dir) if f.endswith(".parquet")]
    assert len(parquet_files) >= 1


# ---------------------------------------------------------------------------
# Dict payload round-trip through cold tier
# ---------------------------------------------------------------------------

def test_dict_payload_roundtrip(test_db):
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    payload = {"role": "user", "content": "Hello", "turn": 1}
    test_db.add_memory(payload, emb)
    test_db.force_checkpoint()

    results = test_db.recall(emb, top_k=1)
    assert len(results) == 1
    recovered = results[0].payload
    assert isinstance(recovered, dict), f"Expected dict, got {type(recovered)}"
    assert recovered["role"] == "user"
    assert recovered["content"] == "Hello"
    assert recovered["turn"] == 1


# ---------------------------------------------------------------------------
# WAL replay (crash recovery)
# ---------------------------------------------------------------------------

def test_wal_replay_on_startup(storage_dir):
    """
    Simulate a crash: verify committed transactions are recovered,
    while uncommitted ones are rolled back (ignored).
    """
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Step 1: Open DB.
    db = EpochDB(storage_dir=storage_dir, dim=4)
    from epochdb.atom import UnifiedMemoryAtom
    
    # 1. Committed atom
    committed_atom = UnifiedMemoryAtom(
        payload="Committed recovered atom",
        embedding=emb,
        triples=[("ghost", "survived", "crash")],
        epoch_id=db.current_epoch_id,
    )
    db.wal.append("ADD", committed_atom.to_dict())
    db.wal.append("COMMIT", {})
    
    # 2. Uncommitted atom (mid-transaction crash)
    uncommitted_atom = UnifiedMemoryAtom(
        payload="Uncommitted crashed atom",
        embedding=emb,
        triples=[("crashed", "died", "crash")],
        epoch_id=db.current_epoch_id,
    )
    db.wal.append("ADD", uncommitted_atom.to_dict())
    # Do NOT write COMMIT for this one
    
    db.wal._file.flush()
    # Close without cleanup (simulate crash by only releasing the lock).
    db.lock.release()
    db.wal.close()

    # Step 2: Re-open the DB — should replay only the committed one.
    db2 = EpochDB(storage_dir=storage_dir, dim=4)
    recovered_payloads = [a.payload for a in db2.hot_tier.atoms.values()]
    assert "Committed recovered atom" in recovered_payloads, "WAL replay failed: committed atom not recovered."
    assert "Uncommitted crashed atom" not in recovered_payloads, "WAL replay failed: uncommitted atom was recovered."
    db2.close()


# ---------------------------------------------------------------------------
# Stale lock auto-cleanup
# ---------------------------------------------------------------------------

def test_stale_lock_is_removed(storage_dir):
    """A lock file belonging to a dead PID should be removed automatically."""
    os.makedirs(storage_dir, exist_ok=True)
    lock_path = os.path.join(storage_dir, ".lock")
    # Write a lock with an impossibly-high PID that cannot be alive.
    with open(lock_path, "w") as f:
        f.write("9999999")

    # Opening the DB should succeed without raising.
    db = EpochDB(storage_dir=storage_dir, dim=4)
    db.close()
    assert not os.path.exists(lock_path)


# ---------------------------------------------------------------------------
# HotTier auto-resize
# ---------------------------------------------------------------------------

def test_hot_tier_auto_resize(storage_dir):
    """Adding more atoms than initial capacity should trigger an index resize."""
    initial_capacity = 10
    db = EpochDB(storage_dir=storage_dir, dim=4, hot_tier_capacity=initial_capacity)

    for i in range(12):  # Exceed the initial capacity.
        emb = np.random.rand(4).astype(np.float32)
        db.add_memory(f"fact {i}", emb)

    # All atoms should be in the hot tier, and max_elements should have grown.
    assert len(db.hot_tier.atoms) == 12
    assert db.hot_tier.max_elements > initial_capacity
    db.close()


# ---------------------------------------------------------------------------
# Forking / Branching
# ---------------------------------------------------------------------------

def test_fork(test_db):
    """Calling fork should record the parent->child relationship in the KG."""
    parent_id = "epoch_alpha"
    child_id = "epoch_beta"
    
    test_db.fork(parent_id, child_id)
    
    # Verify the association exists in the KG
    # The fork method uses (parent_id, "forked_to", child_id)
    associations = test_db.kg_manager.get_associations(parent_id)
    assert any(
        assoc[0] == "forked_to" and assoc[1] == child_id 
        for assoc in associations
    )


def test_custom_parquet_compression(storage_dir):
    # Initialize with SNAPPY and verify properties
    db = EpochDB(
        storage_dir=storage_dir,
        dim=4,
        parquet_compression="SNAPPY",
        parquet_compression_level=None
    )
    assert db.parquet_compression == "SNAPPY"
    assert db.parquet_compression_level is None
    
    # Change dynamically on the instance
    db.parquet_compression = "ZSTD"
    db.parquet_compression_level = 9
    assert db.parquet_compression == "ZSTD"
    assert db.parquet_compression_level == 9
    
    # Write to Parquet and verify round-trip
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    db.add_memory("Test compression payload", emb)
    db.force_checkpoint()
    
    # Verify we can recall it back correctly
    results = db.recall(emb, top_k=1)
    assert len(results) == 1
    assert results[0].payload == "Test compression payload"
    
    db.close()

