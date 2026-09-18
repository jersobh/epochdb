"""
Tiered retrieval: probed cold-tier search vs exhaustive broadcast.

These tests compare the pre-1.10 behaviour (search every epoch HNSW) with
the probe path (recency + GEI entity epochs + centroid neighbours).
"""

import time

import numpy as np
import pytest

from epochdb import EpochDB


DIM = 8
NEEDLE_ENTITY = "NeedleEntity"
NEEDLE_TEXT = "The needle lives in the oldest archive epoch."


def _unit(vec):
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return arr if n < 1e-10 else arr / n


def _query_vec():
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    return v


def _noise_vec(seed: int):
    rng = np.random.default_rng(seed)
    # Cluster noise in a subspace orthogonal to the query axis.
    v = rng.normal(0.0, 1.0, size=DIM).astype(np.float32)
    v[0] = 0.0
    return _unit(v)


def build_archive(tmp_path, n_epochs=24, atoms_per_epoch=8, recency_epochs=2, centroid_probes=2):
    """
    Oldest epoch holds the needle (aligned with e0). Remaining epochs are
    noise clusters so their centroids will not rank near the query.
    """
    db = EpochDB(
        storage_dir=str(tmp_path / "probe_archive"),
        dim=DIM,
        embedding_model=None,
        recency_epochs=recency_epochs,
        centroid_probes=centroid_probes,
        cold_search_mode="probe",
        topic_lock_fetch_cap=512,
    )
    q = _query_vec()
    db.add_memory(
        NEEDLE_TEXT,
        q,
        triples=[(NEEDLE_ENTITY, "is", "hidden")],
    )
    # Pad the needle epoch with orthogonal noise so its centroid is not the query.
    pad_cluster = _noise_vec(7)
    for j in range(atoms_per_epoch):
        db.add_memory(
            f"needle-pad {j}",
            _unit(pad_cluster + _noise_vec(70 + j) * 0.05),
        )
    db.force_checkpoint()
    time.sleep(0.02)

    for epoch_i in range(n_epochs - 1):
        cluster = _noise_vec(1000 + epoch_i)
        for j in range(atoms_per_epoch):
            jitter = _noise_vec(10_000 + epoch_i * 100 + j) * 0.05
            emb = _unit(cluster + jitter)
            db.add_memory(
                f"noise e{epoch_i} a{j}",
                emb,
                triples=[(f"NoiseHub{epoch_i}", "mentions", f"item{j}")],
            )
        db.force_checkpoint()
        time.sleep(0.01)

    return db


@pytest.fixture
def archive(tmp_path):
    db = build_archive(tmp_path)
    yield db
    db.close()


def _payloads(results):
    return [r.payload for r in results]


def test_small_store_is_auto_exhaustive(tmp_path):
    db = EpochDB(
        storage_dir=str(tmp_path / "tiny"),
        dim=DIM,
        embedding_model=None,
        recency_epochs=6,
        centroid_probes=12,
    )
    emb = _query_vec()
    db.add_memory("alpha", emb, triples=[("Alpha", "is", "letter")])
    db.force_checkpoint()
    db.add_memory("beta", _noise_vec(1), triples=[("Beta", "is", "letter")])
    db.force_checkpoint()

    db.recall(emb, top_k=2)
    plan = db.retriever.last_probe_plan
    assert plan is not None
    assert plan.n_total == 2
    assert plan.auto_exhaustive is True
    assert plan.n_searched == plan.n_total
    db.close()


def test_probe_opens_fewer_epochs_than_exhaustive(archive):
    q = _query_vec()
    n_epochs = len(archive.cold_tier.get_all_epochs())
    assert n_epochs >= 20

    archive.recall(q, top_k=3, query_entities=[NEEDLE_ENTITY], cold_search_mode="exhaustive")
    exhaustive = archive.retriever.last_probe_plan
    assert exhaustive.n_searched == n_epochs

    archive.recall(q, top_k=3, query_entities=[NEEDLE_ENTITY], cold_search_mode="probe")
    probe = archive.retriever.last_probe_plan
    assert probe.auto_exhaustive is False
    assert probe.n_searched < exhaustive.n_searched
    assert probe.from_entity, "GEI must include the needle epoch in the probe set"


def test_entity_query_recall_matches_exhaustive(archive):
    q = _query_vec()
    before = archive.recall(
        q, top_k=5, query_entities=[NEEDLE_ENTITY], cold_search_mode="exhaustive"
    )
    after = archive.recall(
        q, top_k=5, query_entities=[NEEDLE_ENTITY], cold_search_mode="probe"
    )
    assert NEEDLE_TEXT in _payloads(before)
    assert NEEDLE_TEXT in _payloads(after)


def test_centroid_written_on_checkpoint(tmp_path):
    db = EpochDB(storage_dir=str(tmp_path / "c"), dim=DIM, embedding_model=None)
    db.add_memory("fact", _query_vec())
    epoch_id = db.current_epoch_id
    db.force_checkpoint()
    centroid = db.cold_tier.get_centroid(epoch_id)
    assert centroid is not None
    assert centroid.shape == (DIM,)
    db.close()


def test_cold_bootstrap_after_flush(tmp_path):
    """Vector-only recall after checkpoint should still discover KG entities."""
    db = EpochDB(
        storage_dir=str(tmp_path / "boot"),
        dim=DIM,
        embedding_model=None,
        recency_epochs=6,
        centroid_probes=12,
    )
    db.add_memory(
        "Maya works at Acme",
        _query_vec(),
        triples=[("Maya", "works_at", "Acme")],
    )
    db.force_checkpoint()
    assert len(db.hot_tier.atoms) == 0

    results = db.recall(_query_vec(), top_k=3, expand_hops=0, query_entities=None)
    assert any("Maya" in r.payload for r in results)
    db.close()


def test_topic_lock_cap_keeps_newest_association(tmp_path):
    db = EpochDB(
        storage_dir=str(tmp_path / "cap"),
        dim=DIM,
        embedding_model=None,
        topic_lock_fetch_cap=1,
    )
    older = _noise_vec(3)
    newer = _query_vec()
    db.add_memory(
        "Maya works at Globex",
        older,
        triples=[("Maya", "works_at", "Globex")],
    )
    db.force_checkpoint()
    db.add_memory(
        "Maya works at Acme",
        newer,
        triples=[("Maya", "works_at", "Acme")],
    )
    db.force_checkpoint()

    newest = db.kg_manager.get_associations("Maya", limit=1)
    assert len(newest) == 1
    results = db.recall(newer, top_k=3, query_entities=["Maya"])
    assert any("Acme" in r.payload for r in results)
    db.close()


def test_plan_includes_entity_epoch_even_when_old(archive):
    q = _query_vec()
    plan = archive.retriever.plan_cold_search(
        q, query_entities={NEEDLE_ENTITY}, cold_search_mode="probe"
    )
    assert plan.auto_exhaustive is False
    assert plan.from_entity, "GEI must route the needle's epoch into the probe set"
    assert plan.n_searched < plan.n_total
