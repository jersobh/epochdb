import os
import shutil
import pytest
from epochdb.mcp_server import mcp

@pytest.fixture(autouse=True)
def clean_storage():
    storage_dir = "./.epochdb_mcp_test"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    os.environ["EPOCHDB_STORAGE_DIR"] = storage_dir
    yield
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

def test_mcp_tools_registration():
    # Verify that all 7 tools are registered on the FastMCP instance
    tool_names = list(mcp._tool_manager._tools.keys())
    assert "epochdb_remember" in tool_names
    assert "epochdb_query" in tool_names
    assert "epochdb_multi_hop" in tool_names
    assert "epochdb_get_timeline" in tool_names
    assert "epochdb_entity_graph" in tool_names
    assert "epochdb_update" in tool_names
    assert "epochdb_delete" in tool_names
    assert "epochdb_analyze" in tool_names

def test_mcp_tools_execution():
    # Direct execution check on the underlying tool functions
    remember_func = mcp._tool_manager._tools["epochdb_remember"].fn
    query_func = mcp._tool_manager._tools["epochdb_query"].fn
    analyze_func = mcp._tool_manager._tools["epochdb_analyze"].fn
    
    # 1. Store a memory
    mem_id = remember_func("The capital of France is Paris.", memory_type="profile")
    assert mem_id is not None
    
    # 2. Query it
    results = query_func("capital of France")
    assert len(results) > 0
    assert "Paris" in results[0]["text"]
    assert results[0]["memory_type"] == "profile"

    # 3. Analyze text
    triples = analyze_func("The capital of France is Paris.")
    assert len(triples) > 0
    # Our fallback LocalFactExtractor returns mentions for capitalized words, e.g. France, Paris
    has_france = any(t["subject"] == "France" or t["object"] == "France" for t in triples)
    assert has_france

