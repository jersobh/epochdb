import os
import shutil
import pytest
from datetime import datetime, timezone

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from epochdb import EpochDB, AsyncEpochDB
from epochdb.tools import get_epochdb_tools


@pytest.fixture
def sync_db():
    storage_dir = "./.test_epochdb_tools_sync"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)
    db = EpochDB(storage_dir=storage_dir, dim=4)
    yield db
    db.close()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)


@pytest.fixture
async def async_db():
    storage_dir = "./.test_epochdb_tools_async"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)
    db = AsyncEpochDB(storage_dir=storage_dir, dim=4)
    yield db
    await db.close()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)


def test_sync_tools_execution(sync_db):
    tools = get_epochdb_tools(sync_db)
    assert len(tools) == 7
    
    tool_map = {t.name: t for t in tools}
    assert "epochdb_remember" in tool_map
    assert "epochdb_query" in tool_map
    assert "epochdb_multi_hop" in tool_map
    assert "epochdb_get_timeline" in tool_map
    assert "epochdb_entity_graph" in tool_map
    assert "epochdb_update" in tool_map
    assert "epochdb_delete" in tool_map

    remember_tool = tool_map["epochdb_remember"]
    query_tool = tool_map["epochdb_query"]
    update_tool = tool_map["epochdb_update"]
    delete_tool = tool_map["epochdb_delete"]
    timeline_tool = tool_map["epochdb_get_timeline"]
    graph_tool = tool_map["epochdb_entity_graph"]

    # 1. Test remember
    mem_id = remember_tool.invoke({"text": "BMW has a headquarters in Munich.", "metadata": {"triples": [("BMW", "headquartered_in", "Munich")]}})
    assert isinstance(mem_id, str)
    assert len(mem_id) > 0

    # 2. Test query
    results = query_tool.invoke({"query": "Munich", "k": 2})
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["id"] == mem_id
    assert "Munich" in results[0]["text"]

    # 3. Test timeline
    timeline = timeline_tool.invoke({"entity_id": "Munich"})
    assert isinstance(timeline, list)
    assert len(timeline) >= 1
    assert timeline[0]["id"] == mem_id

    # 4. Test graph
    graph = graph_tool.invoke({"entity_id": "BMW", "depth": 1})
    assert isinstance(graph, dict)
    assert "BMW" in graph["nodes"]
    assert "Munich" in graph["nodes"]

    # 5. Test update
    update_res = update_tool.invoke({"memory_id": mem_id, "text": "BMW is headquartered in Munich, Germany."})
    assert "updated successfully" in update_res

    # Verify update
    results2 = query_tool.invoke({"query": "Germany", "k": 1})
    assert "Germany" in results2[0]["text"]

    # 6. Test delete (soft)
    delete_res = delete_tool.invoke({"memory_id": mem_id, "hard": False})
    assert "deleted successfully" in delete_res

    # Verify deleted
    results3 = query_tool.invoke({"query": "Germany", "k": 1})
    assert len(results3) == 0


@pytest.mark.anyio
async def test_async_tools_execution(async_db):
    # Initialize the underlying db by entering context
    async with async_db as db:
        tools = get_epochdb_tools(db)
        tool_map = {t.name: t for t in tools}

        remember_tool = tool_map["epochdb_remember"]
        query_tool = tool_map["epochdb_query"]
        multi_hop_tool = tool_map["epochdb_multi_hop"]
        update_tool = tool_map["epochdb_update"]
        delete_tool = tool_map["epochdb_delete"]

        # Test async execution (ainvoke)
        mem_id = await remember_tool.ainvoke({"text": "Juno is a smart agent framework.", "metadata": {"triples": [("Juno", "is_a", "framework")]}})
        assert isinstance(mem_id, str)

        results = await query_tool.ainvoke({"query": "Juno", "k": 1})
        assert len(results) >= 1
        assert results[0]["id"] == mem_id

        # Test multi-hop
        hop_results = await multi_hop_tool.ainvoke({"query": "Juno framework", "hops": 2})
        assert len(hop_results) >= 1

        update_res = await update_tool.ainvoke({"memory_id": mem_id, "text": "Juno is an advanced agentic framework."})
        assert "updated successfully" in update_res

        delete_res = await delete_tool.ainvoke({"memory_id": mem_id, "hard": True})
        assert "deleted successfully" in delete_res
