import os
import shutil
import tempfile
import threading
import time
import socket
import pytest
import numpy as np
from epochdb.api.db import EpochDB
from epochdb.api.client import RemoteEpochDB
from epochdb.api.server import start_server

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def test_thread_safety(temp_dir):
    """Verify that multiple concurrent threads can write and read without corrupting state or raising errors."""
    db = EpochDB(storage_dir=temp_dir, dim=4, wal_sync_interval=0.01)
    
    num_threads = 10
    writes_per_thread = 20
    errors = []
    
    def worker(thread_idx):
        try:
            for i in range(writes_per_thread):
                # Perform write
                text = f"Thread {thread_idx} writing turn {i}"
                db.remember(text, metadata={"triples": [("Thread", "wrote", text)]})
                
                # Perform read
                db.query(f"Thread {thread_idx}", k=2)
                
                # Sleep briefly to allow interleaving
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    db.close()
    
    assert not errors, f"Encountered concurrency errors: {[str(e) for e in errors]}"

def test_tenant_isolation(temp_dir):
    """Verify that specifying different tenants physically separates data and locks."""
    # Initialize two different tenants in the same base storage directory
    db_a = EpochDB(storage_dir=temp_dir, tenant="tenant_A", dim=4)
    db_b = EpochDB(storage_dir=temp_dir, tenant="tenant_B", dim=4)
    
    # Assert they use different physical subdirectories
    assert "tenants/tenant_A" in db_a.storage_dir
    assert "tenants/tenant_B" in db_b.storage_dir
    
    # Store unique memory in Tenant A
    db_a.remember("Tenant A secret memory", metadata={"triples": [("TenantA", "knows", "Secret")]})
    
    # Store unique memory in Tenant B
    db_b.remember("Tenant B secret memory", metadata={"triples": [("TenantB", "knows", "Secret")]})
    
    # Query tenant A
    res_a = db_a.query("Tenant A secret memory", k=5)
    assert len(res_a) == 1
    assert "Tenant A secret memory" in res_a[0].text
    
    # Query tenant B for tenant A's secret (should find nothing or B's secret)
    res_b_query_a = db_b.query("Tenant A secret memory", k=5)
    assert not any("Tenant A secret memory" in m.text for m in res_b_query_a)
    
    # Close databases
    db_a.close()
    db_b.close()

def test_wal_sync_interval(temp_dir):
    """Verify that using a sync_interval does not break basic operations and shuts down cleanly."""
    db = EpochDB(storage_dir=temp_dir, dim=4, wal_sync_interval=0.05)
    
    # Write a fact
    db.remember("Test periodic sync", metadata={"triples": [("WAL", "periodic", "test")]})
    
    # Retrieve it
    res = db.query("Test periodic sync", k=1)
    assert len(res) == 1
    assert res[0].text == "Test periodic sync"
    
    # Wait for background fsync thread to run at least once
    time.sleep(0.1)
    
    # Close should shut down sync thread and join cleanly
    db.close()

def test_client_server_communication(temp_dir):
    """Verify standard library HTTP server and RemoteEpochDB client communication."""
    db = EpochDB(storage_dir=temp_dir, dim=4)
    
    port = find_free_port()
    server = start_server(db, host="127.0.0.1", port=port)
    
    # Run server in background thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    # Allow server thread to spin up
    time.sleep(0.1)
    
    client = RemoteEpochDB(host="127.0.0.1", port=port)
    
    try:
        # Test remember
        mem_id = client.remember("Bob works at CERN", metadata={"triples": [("Bob", "works_at", "CERN")]})
        assert mem_id is not None
        
        # Test get
        mem = client.get(mem_id)
        assert mem is not None
        assert mem.text == "Bob works at CERN"
        assert mem.triples == [("Bob", "works_at", "CERN")]
        
        # Test query
        res = client.query("Where does Bob work?", k=1)
        assert len(res) == 1
        assert res[0].text == "Bob works at CERN"
        
        # Test update
        client.update(mem_id, text="Bob works at NASA")
        mem_updated = client.get(mem_id)
        assert mem_updated.text == "Bob works at NASA"
        
        # Test stats
        stats = client.stats()
        assert stats["l1_size"] == 1
        
        # Test delete
        client.delete(mem_id, hard=True)
        assert client.get(mem_id) is None
        
    finally:
        # Stop HTTP server gracefully
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)
        db.close()
