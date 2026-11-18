"""Regression coverage for hard-delete compaction, filters, cold update,
context_window isolation, and BM25 facet-winner eligibility.
"""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from epochdb.api.db import EpochDB
from epochdb.retrieval.retrieval import RetrievalManager, _validate_filters


class TestHardDeleteCompaction(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="epochdb_hard_delete_")
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=4,
            embedding_model=None,
            wal_use_uring=False,
        )

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_hard_delete_purged_by_compact_and_hidden_from_sql(self):
        mid = self.db.remember("sensitive text", metadata={"k": "v"})
        self.db.force_checkpoint()

        # Soft then hard delete (tombstone removed from hot; id stays deleted).
        self.db.delete(mid, hard=False)
        self.db.delete(mid, hard=True)

        # Logical hide before compact; SQL also excludes hard-deleted IDs.
        self.assertIsNone(self.db.get(mid))
        rows = self.db.query_sql(f"SELECT payload FROM cold_tier WHERE id = '{mid}'")
        self.assertEqual(rows, [])

        self.db.compact()
        rows = self.db.query_sql(f"SELECT payload FROM cold_tier WHERE id = '{mid}'")
        self.assertEqual(rows, [])
        self.assertIsNone(self.db.get(mid))

        # Survives restart as logical hide; still absent from SQL.
        self.db.close()
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=4,
            embedding_model=None,
            wal_use_uring=False,
        )
        self.assertIsNone(self.db.get(mid))
        rows = self.db.query_sql(f"SELECT payload FROM cold_tier WHERE id = '{mid}'")
        self.assertEqual(rows, [])

    def test_hard_delete_rewrites_existing_compacted_archive(self):
        keep = self.db.remember("keep me")
        doomed = self.db.remember("erase me later")
        self.db.force_checkpoint()
        self.db.compact()

        compacted = [
            f
            for f in os.listdir(self.test_dir)
            if f.startswith("compacted_") and f.endswith(".parquet")
        ]
        self.assertEqual(len(compacted), 1)

        self.db.delete(doomed, hard=True)
        self.db.compact()

        rows = self.db.query_sql(f"SELECT id FROM cold_tier WHERE id = '{doomed}'")
        self.assertEqual(rows, [])
        kept = self.db.query_sql(f"SELECT id FROM cold_tier WHERE id = '{keep}'")
        self.assertEqual(len(kept), 1)
        self.assertIsNotNone(self.db.get(keep))


class TestMatchFilters(unittest.TestCase):
    def test_validate_unknown_operator(self):
        with self.assertRaises(ValueError):
            _validate_filters({"tag": {"$regex": "x"}})
        with self.assertRaises(ValueError):
            _validate_filters({"n": {"$gte ": 3}})

    def test_validate_in_requires_collection(self):
        with self.assertRaises(ValueError):
            _validate_filters({"tag": {"$in": "abc"}})

    def test_type_mismatch_is_non_match(self):
        rm = RetrievalManager(hot_tier=None, cold_tier=None, kg_manager=None)
        self.assertFalse(rm.match_filters({"level": "high"}, {"level": {"$gt": 1}}))
        self.assertTrue(rm.match_filters({"n": 5}, {"n": {"$gte": 3}}))

    def test_search_rejects_unknown_operator_even_without_candidates(self):
        test_dir = tempfile.mkdtemp(prefix="epochdb_filters_")
        try:
            db = EpochDB(
                storage_dir=test_dir,
                dim=4,
                embedding_model=None,
                wal_use_uring=False,
            )
            with self.assertRaises(ValueError):
                db.query("anything", k=1, filters={"tag": {"$regex": "x"}})
            db.close()
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


class TestColdUpdateNewestVersion(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="epochdb_cold_update_")
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=4,
            embedding_model=None,
            wal_use_uring=False,
        )

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_metadata_only_update_uses_newest_cold_version(self):
        mid = self.db.remember("version one text", metadata={"ver": 1})
        self.db.force_checkpoint()

        self.db.update(mid, text="version two text", metadata={"ver": 2, "kept": True})
        self.db.force_checkpoint()
        # Ensure cold-only: clear hot so update must load from cold.
        self.db.hot_tier.atoms.clear()

        # Force epoch iteration to prefer the older file first.
        epochs = self.db.cold_tier.get_all_epochs()
        self.assertGreaterEqual(len(epochs), 2)
        # Monkeypatch get_all_epochs to return oldest-first explicitly.
        oldest_first = sorted(
            epochs,
            key=lambda e: os.path.getmtime(os.path.join(self.test_dir, f"{e}.parquet")),
        )
        self.db.cold_tier.get_all_epochs = lambda: oldest_first

        mem = self.db.get(mid)
        self.assertEqual(mem.text, "version two text")
        self.assertEqual(mem.metadata.get("ver"), 2)

        self.db.update(mid, metadata={"patched": True})
        updated = self.db.get(mid)
        self.assertEqual(updated.text, "version two text")
        self.assertEqual(updated.metadata.get("ver"), 2)
        self.assertTrue(updated.metadata.get("kept"))
        self.assertTrue(updated.metadata.get("patched"))


class TestContextWindowIsolation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="epochdb_context_")
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=4,
            embedding_model=None,
            wal_use_uring=False,
        )

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_context_neighbors_do_not_mutate_store_or_wal(self):
        self.db.remember("First turn hello", metadata={"session_id": "s1"})
        target = self.db.remember("Second turn boston", metadata={"session_id": "s1"})
        self.db.remember("Other secret turn", metadata={"session_id": "s2", "scope": "other"})

        results = self.db.query("boston", k=1, context_window=1)
        self.assertEqual(len(results), 1)
        nucleus = results[0]
        self.assertIn("context_neighbors", nucleus.metadata)

        stored = self.db.hot_tier.atoms[nucleus.id]
        self.assertNotIn("context_neighbors", stored.metadata)

        later = self.db.query("boston", k=1, context_window=0)
        self.assertEqual(len(later), 1)
        self.assertNotIn("context_neighbors", later[0].metadata)

        self.db.update(nucleus.id, metadata={"touched": True})
        with open(os.path.join(self.test_dir, "wal.jsonl")) as f:
            records = [json.loads(line) for line in f if line.strip()]
        adds = [
            r
            for r in records
            if r.get("op") == "ADD" and r.get("data", {}).get("id") == nucleus.id
        ]
        self.assertTrue(adds)
        self.assertNotIn("context_neighbors", adds[-1]["data"]["metadata"])
        self.assertEqual(target, nucleus.id)

    def test_context_neighbors_honor_filters(self):
        self.db.remember("alpha turn", metadata={"project": "a"})
        self.db.remember("boston turn", metadata={"project": "a"})
        self.db.remember("gamma turn", metadata={"project": "b"})

        results = self.db.query(
            "boston", k=1, context_window=2, filters={"project": "a"}
        )
        self.assertEqual(len(results), 1)
        neighbors = results[0].metadata.get("context_neighbors", [])
        self.assertTrue(neighbors)
        for n in neighbors:
            self.assertEqual(n["metadata"].get("project"), "a")


class TestFacetWinnerEligibility(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="epochdb_facet_")
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=4,
            embedding_model=None,
            wal_use_uring=False,
        )

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_facet_winner_respects_filters(self):
        bad = self.db.remember(
            "deploy uses zorblax-7 tokens", metadata={"project": "b"}
        )
        good = self.db.remember(
            "zorblax-7 rotation runbook for project a", metadata={"project": "a"}
        )
        results = self.db.query("zorblax-7", k=10, filters={"project": "a"})
        ids = [m.id for m in results]
        self.assertNotIn(bad, ids)
        self.assertIn(good, ids)

    def test_facet_winner_respects_soft_delete(self):
        mid = self.db.remember("unique-keyword-xyz exact match payload")
        self.db.delete(mid, hard=False)
        results = self.db.query("unique-keyword-xyz", k=5)
        ids = [m.id for m in results]
        self.assertNotIn(mid, ids)


if __name__ == "__main__":
    unittest.main()
