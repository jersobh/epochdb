import os
import shutil
import pytest
import numpy as np

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed (pip install epochdb[langgraph])")

from epochdb import EpochDB
from epochdb.checkpointer import EpochDBCheckpointer

@pytest.fixture
def test_db():
    storage_dir = "./.test_epochdb_checkpointer"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)
    db = EpochDB(storage_dir=storage_dir, dim=4)
    yield db
    db.close()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)

def test_checkpoint_save_load(test_db):
    checkpointer = EpochDBCheckpointer(test_db)
    
    thread_id = "test_thread"
    checkpoint_id = "cp_1"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    
    checkpoint = {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2024-01-01T00:00:00Z",
        "channel_values": {"key": "value"},
        "channel_versions": {"key": 1},
        "versions_seen": {},
        "pending_sends": [],
    }
    
    metadata = {"source": "test"}
    
    # Save
    checkpointer.put(config, checkpoint, metadata, {})
    
    # Load
    tup = checkpointer.get_tuple(config)
    
    assert tup is not None
    assert tup.checkpoint["id"] == checkpoint_id
    assert tup.metadata["source"] == "test"

def test_list_checkpoints(test_db):
    checkpointer = EpochDBCheckpointer(test_db)
    thread_id = "list_thread"
    
    for i in range(3):
        cp_id = f"cp_{i}"
        config = {"configurable": {"thread_id": thread_id, "checkpoint_id": cp_id}}
        checkpoint = {"id": cp_id, "v": 1}
        checkpointer.put(config, checkpoint, {"idx": i}, {})
        
    # List
    config = {"configurable": {"thread_id": thread_id}}
    checkpoints = list(checkpointer.list(config))
    
    assert len(checkpoints) == 3
    # Should be sorted reverse by ID/filename
    assert checkpoints[0].checkpoint["id"] == "cp_2"

def test_put_writes(test_db):
    checkpointer = EpochDBCheckpointer(test_db)
    thread_id = "write_thread"
    checkpoint_id = "cp_write"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    
    writes = [("channel1", "data1"), ("channel2", 42)]
    checkpointer.put_writes(config, writes, "task_1")
    
    # Verify file exists
    write_files = [f for f in os.listdir(os.path.join(test_db.storage_dir, "checkpoints", thread_id)) if "writes" in f]
    assert len(write_files) == 1
    assert "task_1" in write_files[0]

@pytest.mark.asyncio
async def test_async_save_load(test_db):
    checkpointer = EpochDBCheckpointer(test_db)
    
    thread_id = "test_async_thread"
    checkpoint_id = "cp_async"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    
    checkpoint = {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2024-01-01T00:00:00Z",
        "channel_values": {"key": "async_value"},
        "channel_versions": {"key": 1},
        "versions_seen": {},
        "pending_sends": [],
    }
    
    metadata = {"source": "async_test"}
    
    # Save async
    await checkpointer.aput(config, checkpoint, metadata, {})
    
    # Load async
    tup = await checkpointer.aget_tuple(config)
    
    assert tup is not None
    assert tup.checkpoint["id"] == checkpoint_id
    assert tup.checkpoint["channel_values"]["key"] == "async_value"
    
    # List async
    found = False
    async for t in checkpointer.alist(config):
        if t.checkpoint["id"] == checkpoint_id:
            found = True
            break
    assert found
