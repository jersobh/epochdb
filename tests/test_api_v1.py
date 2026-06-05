import unittest
import os
import shutil
import numpy as np
import datetime
from epochdb import EpochDB, Memory, Entity, Graph

class TestAPIv1(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./.test_api_v1"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.db = EpochDB(storage_dir=self.test_dir, dim=128)

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_remember_and_get(self):
        # 1. remember a basic memory with metadata
        meta = {"author": "Jeff", "project": "Aegis", "category": "development"}
        mid = self.db.remember("Implemented custom WAL in Rust", metadata=meta)
        
        # 2. get the memory and verify wrapping
        mem = self.db.get(mid)
        self.assertIsNotNone(mem)
        self.assertIsInstance(mem, Memory)
        self.assertEqual(mem.text, "Implemented custom WAL in Rust")
        self.assertEqual(mem.metadata["author"], "Jeff")
        self.assertEqual(mem.payload_type, "text")

    def test_remember_batch(self):
        items = [
            "Setup dev server",
            {"text": "Refactored hot tier HNSW resize threshold", "metadata": {"priority": "high"}},
            {"text": "Wrote unit tests for new facade", "metadata": {"priority": "medium"}}
        ]
        ids = self.db.remember_batch(items)
        self.assertEqual(len(ids), 3)
        
        m1 = self.db.get(ids[0])
        m2 = self.db.get(ids[1])
        m3 = self.db.get(ids[2])
        
        self.assertEqual(m1.text, "Setup dev server")
        self.assertEqual(m2.metadata["priority"], "high")
        self.assertEqual(m3.metadata["priority"], "medium")

    def test_update_and_delete(self):
        mid = self.db.remember("Original memory text", metadata={"ver": 1})
        
        # Update text and metadata
        self.db.update(mid, text="Updated memory text", metadata={"ver": 2, "edited": True})
        
        mem = self.db.get(mid)
        self.assertEqual(mem.text, "Updated memory text")
        self.assertEqual(mem.metadata["ver"], 2)
        self.assertTrue(mem.metadata["edited"])
        
        # Soft delete
        self.db.delete(mid, hard=False)
        self.assertIsNone(self.db.get(mid)) # should be filtered out by default
        
        # Hard delete
        self.db.delete(mid, hard=True)
        self.assertIsNone(self.db.get(mid))

    def test_query_and_filtering(self):
        # Insert memories with metadata
        self.db.remember("The project Aegis uses Rust for performance.", metadata={"project": "Aegis", "lang": "rust"})
        self.db.remember("The project EpochDB uses Python for facade.", metadata={"project": "EpochDB", "lang": "python"})
        
        # Semantic query
        results = self.db.query("Which project is built with Rust?", k=2)
        self.assertGreater(len(results), 0)
        
        # Query with exact metadata filtering
        res_aegis = self.db.query("database", k=2, filters={"project": "Aegis"})
        self.assertEqual(len(res_aegis), 1)
        self.assertEqual(res_aegis[0].metadata["lang"], "rust")
        
        # Query with operator metadata filtering ($in)
        res_ops = self.db.query("database", k=2, filters={"lang": {"$in": ["rust", "python"]}})
        self.assertEqual(len(res_ops), 2)

    def test_timeline_and_entities(self):
        # Remember memories mentioning entities
        mid1 = self.db.remember("Marie heads the research team at BioGen.", metadata={"triples": [("Marie", "leads", "BioGen")]})
        mid2 = self.db.remember("BioGen develops CRISPR-X platform.", metadata={"triples": [("BioGen", "develops", "CRISPR-X")]})
        
        # Get entity objects
        entities = self.db.extract_entities("Marie's organisation BioGen")
        self.assertGreater(len(entities), 0)
        self.assertTrue(any(isinstance(e, Entity) for e in entities))
        
        biogen = self.db.get_entity("BioGen")
        self.assertEqual(biogen.id, "BioGen")
        
        # Entity related neighbors
        related = biogen.related()
        self.assertIn("Marie", related)
        self.assertIn("CRISPR-X", related)
        
        # Entity timeline
        timeline = biogen.timeline()
        self.assertEqual(len(timeline), 2)
        
        # Graph construction
        graph = self.db.entity_graph("Marie", depth=2)
        self.assertIsInstance(graph, Graph)
        self.assertIn("Marie", graph.nodes)
        self.assertIn("BioGen", graph.nodes)
        self.assertIn("CRISPR-X", graph.nodes)
        self.assertEqual(len(graph.edges), 2)

    def test_compaction(self):
        # Write some scalar updates that supersede each other
        self.db.remember("Temperature is 22.0 C", metadata={"triples": [("room1", "temperature", "22.0")]})
        self.db.remember("Temperature is 25.0 C", metadata={"triples": [("room1", "temperature", "25.0")]})
        
        # Write soft deleted memory
        mid = self.db.remember("To be deleted", metadata={"category": "temporary"})
        self.db.delete(mid, hard=False)
        
        # Force flush to cold tier so it has historical files
        self.db.force_checkpoint()
        
        # Compact
        self.db.compact()
        
        # Verify stats after compaction
        stats = self.db.stats()
        # L2 compacted database shouldn't have soft deleted and superseded atoms
        self.assertEqual(stats["l2_size"], 1)

if __name__ == "__main__":
    unittest.main()
