# tests/test_adaptive_router.py
import pytest
import shutil
import os
from epochdb import EpochDB
from epochdb.retrieval.router import AdaptiveRouter, QueryRouting

@pytest.fixture
def temp_db(tmp_path):
    storage_dir = str(tmp_path / "test_router_db")
    db = EpochDB(storage_dir=storage_dir, dim=4, embedding_model=None)
    yield db
    db.close()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

def test_local_fallback_routing(temp_db):
    router = AdaptiveRouter(temp_db)
    
    # 1. Quantitative
    q_route = router._local_fallback_routing("power_usage > 150.5")
    assert q_route.query_type == "quantitative"
    assert q_route.quantitative_field == "power_usage"
    assert q_route.quantitative_op == ">"
    assert q_route.quantitative_value == 150.5
    
    # 2. Temporal
    t_route = router._local_fallback_routing("show me the timeline of Jeff")
    assert t_route.query_type == "temporal"
    assert t_route.entity_id == "Jeff"
    
    # 3. Relational
    r_route = router._local_fallback_routing("Who is the manager of the engineering team?")
    assert r_route.query_type == "relational"
    
    # 4. Semantic
    s_route = router._local_fallback_routing("Explain quantum mechanics in simple terms.")
    assert s_route.query_type == "semantic"
