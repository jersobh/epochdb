"""Tests for configurable / async triple extraction."""

import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from epochdb import EpochDB
from epochdb.core.fact_extractor import (
    LocalFactExtractor,
    resolve_extraction_backend,
    _parse_rebel_output,
    _parse_jsonish_triples,
)


class TestExtractionRouting(unittest.TestCase):
    def test_resolve_backends(self):
        self.assertEqual(resolve_extraction_backend("local")[0], "local")
        self.assertEqual(resolve_extraction_backend("hf")[0], "hf")
        self.assertEqual(resolve_extraction_backend("hf:small")[1], "google/flan-t5-small")
        self.assertEqual(resolve_extraction_backend("google:gemini-2.5-flash")[0], "google")
        self.assertEqual(resolve_extraction_backend("openai:gpt-4o-mini")[0], "openai")
        self.assertEqual(resolve_extraction_backend("Babelscape/rebel-large")[0], "hf")
        self.assertEqual(resolve_extraction_backend("gemini-2.5-flash")[0], "google")

    def test_rebel_parser(self):
        decoded = "<triplet> Alice <subj> Acme <obj> works at"
        triples = _parse_rebel_output(decoded)
        self.assertEqual(triples, [("Alice", "works at", "Acme")])

    def test_jsonish_parser(self):
        text = '{"triples": [{"subject": "Bob", "predicate": "likes", "object": "Paris"}]}'
        self.assertEqual(_parse_jsonish_triples(text), [("Bob", "likes", "Paris")])


class TestAsyncExtraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="epochdb_extract_")
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=32,
            embedding_model=None,
            auto_extract=True,
            extraction_model="local",
            async_extract=True,
        )

    def tearDown(self):
        try:
            self.db.close()
        except Exception:
            pass
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_remember_returns_immediately_and_merges(self):
        gate = threading.Event()
        mock_extractor = MagicMock()
        mock_extractor.extract_local.return_value = [("Alice", "co_occurs_with", "Paris")]
        mock_extractor.backend = "mock"
        mock_extractor.resolved_model_id = "mock-model"

        def slow_extract(_text):
            gate.wait(timeout=5.0)
            return [("Alice", "visited", "Paris")]

        mock_extractor.extract.side_effect = slow_extract

        with patch.object(self.db, "_get_fact_extractor", return_value=mock_extractor):
            mid = self.db.remember("Alice visited Paris last summer.")
            mem = self.db.get(mid)
            self.assertIsNotNone(mem)
            self.assertEqual(mem.metadata.get("extraction_status"), "pending")
            self.assertGreaterEqual(self.db.pending_extractions(), 1)
            # Seed triples present immediately
            self.assertTrue(any(t[1] == "co_occurs_with" for t in mem.triples))

            gate.set()
            self.db.wait_for_extractions(timeout=5.0)
            mem2 = self.db.get(mid)
            self.assertEqual(mem2.metadata.get("extraction_status"), "done")
            preds = {t[1] for t in mem2.triples}
            self.assertIn("visited", preds)
            self.assertIn("co_occurs_with", preds)
            mock_extractor.extract.assert_called()

    def test_sync_extract_when_async_disabled(self):
        self.db.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir = tempfile.mkdtemp(prefix="epochdb_extract_")
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=32,
            embedding_model=None,
            auto_extract=True,
            extraction_model="local",
            async_extract=False,
        )
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [("User", "works_at", "VectorAI")]
        mock_extractor.extract_local.return_value = []
        mock_extractor.backend = "local"
        mock_extractor.resolved_model_id = None

        with patch.object(self.db, "_get_fact_extractor", return_value=mock_extractor):
            mid = self.db.remember("User works at VectorAI.")
            mem = self.db.get(mid)
            self.assertEqual(mem.metadata.get("extraction_status"), "done")
            self.assertEqual(mem.triples, [("User", "works_at", "VectorAI")])
            self.assertEqual(self.db.pending_extractions(), 0)

    def test_set_extraction_model_resets_cache(self):
        first = self.db._get_fact_extractor()
        self.db.set_extraction_model("hf:small")
        second = self.db._get_fact_extractor()
        self.assertIsNot(first, second)
        self.assertEqual(second.backend, "hf")
        self.assertEqual(second.resolved_model_id, "google/flan-t5-small")

    def test_analyze_uses_configured_extractor(self):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [("A", "rel", "B")]
        with patch.object(self.db, "_get_fact_extractor", return_value=mock_extractor):
            triples = self.db.analyze("A rel B")
            self.assertEqual(triples, [("A", "rel", "B")])


class TestLocalExtractorStillWorks(unittest.TestCase):
    def test_local_cooccurrence(self):
        ext = LocalFactExtractor(engine=None)
        triples = ext.extract("Alice and Bob met in Paris.")
        ents = {t[0] for t in triples} | {t[2] for t in triples}
        self.assertTrue({"Alice", "Bob", "Paris"} & ents)


if __name__ == "__main__":
    unittest.main()
