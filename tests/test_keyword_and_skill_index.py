import os
import shutil

import numpy as np
import pytest

from epochdb import EpochDB


@pytest.fixture
def db():
    storage_dir = "./.test_keyword_skill_index"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)
    database = EpochDB(storage_dir=storage_dir, dim=8)
    yield database
    database.close()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)


def test_keyword_channel_boosts_exact_id_match(db):
    unique = "ERR-404-NOT-FOUND-XYZ"
    db.remember(f"Incident ticket {unique} requires manual review.")
    db.remember("The weather in Lisbon is sunny today.")

    results = db.query(unique, k=3)
    assert results
    assert unique in results[0].text


def test_skill_index_get_and_list(db):
    skill_id = db.remember_skill(
        skill_name="refund_order",
        description="Refund a paid order.",
        steps=[{"step_num": 1, "action": "Verify payment status"}],
        skill_id="skill-refund-order",
    )
    assert skill_id == "skill-refund-order"

    fetched = db.get_skill("refund_order")
    assert fetched is not None
    assert fetched.id == skill_id
    assert fetched.memory_type == "skill"

    by_id = db.get_skill("skill-refund-order")
    assert by_id is not None
    assert by_id.id == skill_id

    skills = db.list_skills()
    assert len(skills) == 1
    assert skills[0].metadata["skill_name"] == "refund_order"


def test_skill_index_after_hot_tier_clear(db):
    db.remember_skill(
        skill_name="deploy_service",
        description="Deploy the API service.",
        steps=[{"action": "Run tests"}],
        skill_id="skill-deploy",
    )
    assert db.hot_tier.skill_index.resolve("deploy_service") == "skill-deploy"

    epoch_id = db.current_epoch_id
    db.flush()
    assert "skill-deploy" not in db.hot_tier.atoms

    cold_skills = db.cold_tier.list_skill_atom_refs()
    assert ("skill-deploy", epoch_id) in cold_skills

    fetched = db.get_skill("deploy_service")
    assert fetched is not None
    assert fetched.id == "skill-deploy"

    listed = db.list_skills()
    assert any(s.id == "skill-deploy" for s in listed)


def test_hot_summary_includes_skills(db):
    db.remember_user_profile("alice", "Prefers dark mode UI.")
    db.remember_skill(
        skill_name="onboard_user",
        description="Onboard a new user.",
        steps=[{"action": "Create account"}],
    )
    snapshot = db.get_hot_summary_snapshot("alice")
    assert "alice" in snapshot.lower() or "dark mode" in snapshot.lower()
    assert "onboard_user" in snapshot
