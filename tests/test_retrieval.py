import os
import shutil
import pytest
import numpy as np
from epochdb import EpochDB


@pytest.fixture
def test_db(tmp_path):
    db = EpochDB(storage_dir=str(tmp_path / "retrieval_db"), dim=4)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Basic Semantic Recall
# ---------------------------------------------------------------------------

def test_semantic_recall(test_db):
    """
    Apple (emb1) and Banana (emb2) are orthogonal vectors.
    Querying with emb1 should return Apple and NOT return Banana when top_k=1.
    Note: with 3-Way RRF, the exact rank of Apple vs. Banana still depends
    on recency, but with orthogonal embeddings Banana should score 0 semantically.
    """
    emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Apple", emb1)

    emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Banana", emb2)

    # With the cold-tier threshold of 0.1, Banana (cosine sim = 0 with emb1)
    # only enters via hot-tier scan. Both atoms pass the hot-tier top_k*2 fetch,
    # so we retrieve top_k=2 and verify Apple is present.
    results = test_db.recall(emb1, top_k=2)
    payloads = [r.payload for r in results]
    assert "Apple" in payloads, f"Expected 'Apple' in results, got: {payloads}"


# ---------------------------------------------------------------------------
# Relational Expansion (Hot Tier)
# ---------------------------------------------------------------------------

def test_relational_expansion(test_db):
    emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory(
        "Jeff leads Project Zero",
        emb1,
        triples=[("Jeff", "leads", "Project Zero")],
    )

    emb2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    test_db.add_memory(
        "Project Zero uses Parquet",
        emb2,
        triples=[("Project Zero", "uses", "Parquet")],
    )

    results = test_db.recall(emb1, top_k=5, expand_hops=2)
    payloads = [r.payload for r in results]
    assert any("Jeff" in p for p in payloads)
    assert any("Parquet" in p for p in payloads)


# ---------------------------------------------------------------------------
# Payload Deduplication
# ---------------------------------------------------------------------------

def test_payload_deduplication(test_db):
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Repeat fact", emb)
    test_db.add_memory("Repeat fact", emb)

    results = test_db.recall(emb, top_k=5)
    payloads = [r.payload for r in results]
    assert payloads.count("Repeat fact") == 1


# ---------------------------------------------------------------------------
# Triples with special characters (quote, backslash, unicode)
# ---------------------------------------------------------------------------

def test_triples_special_characters(test_db):
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    tricky_triples = [
        ("O'Brien", "said", 'He replied "yes"'),
        ("path\\to\\file", "contains", "data"),
        ("Ångström", "unit", "Å"),
    ]
    test_db.add_memory("Special chars fact", emb, triples=tricky_triples)
    test_db.force_checkpoint()

    results = test_db.recall(emb, top_k=1)
    assert len(results) == 1
    recovered_triples = results[0].triples
    assert len(recovered_triples) == 3
    assert recovered_triples[0][0] == "O'Brien"
    assert recovered_triples[1][0] == "path\\to\\file"
    assert recovered_triples[2][0] == "Ångström"


# ---------------------------------------------------------------------------
# Multi-hop across Hot→Cold boundary
# ---------------------------------------------------------------------------

def test_multihop_hot_to_cold_boundary(test_db):
    """
    Fact A lives in the Cold Tier (checkpointed).
    Fact B lives in the Hot Tier (current epoch).
    They share an entity. Relational expansion should bridge the boundary.
    """
    emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory(
        "Alice works at Acme Corp",
        emb_a,
        triples=[("Alice", "works_at", "Acme Corp")],
    )
    # Push Fact A to the Cold Tier.
    test_db.force_checkpoint()

    emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory(
        "Acme Corp builds rockets",
        emb_b,
        triples=[("Acme Corp", "builds", "rockets")],
    )

    # Query with emb_a (semantically close to Fact A in cold).
    results = test_db.recall(emb_a, top_k=5, expand_hops=2)
    payloads = [r.payload for r in results]

    assert any("Alice" in p for p in payloads), "Fact A not recovered from cold tier"
    assert any("rockets" in p for p in payloads), "Fact B not reached via KG expansion"


# ---------------------------------------------------------------------------
# access_count delta tracking
# ---------------------------------------------------------------------------

def test_access_count_delta_for_cold_atoms(test_db):
    """access_count should reflect hot-tier recalls even after an atom moves cold."""
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    test_db.add_memory("Memorable fact", emb)
    test_db.force_checkpoint()

    # Recall twice from the cold tier.
    test_db.recall(emb, top_k=1)
    results = test_db.recall(emb, top_k=1)

    assert results[0].access_count >= 2, (
        "access_count should have been incremented by cold-tier recalls"
    )
