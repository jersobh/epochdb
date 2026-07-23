import unittest
import numpy as np
import os
import shutil
from epochdb.engine import EpochDB
from epochdb.atom import ScalarPayload

class TestColdTierAnalytics(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./.test_cold_analytics"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.db = EpochDB(storage_dir=self.test_dir, dim=128)

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_unit_aware_trend(self):
        """Test that ColdTierAnalytics handles mixed units correctly."""
        # Add 0°C
        s1 = ScalarPayload(value=0.0, unit="degC")
        self.db.add_memory(payload=s1, embedding=np.zeros(128), triples=[("sensor1", "temp", "val")])
        
        # Add 32°F (which is 0°C)
        s2 = ScalarPayload(value=32.0, unit="degF")
        self.db.add_memory(payload=s2, embedding=np.zeros(128), triples=[("sensor1", "temp", "val")])
        
        # Add 100°C
        s3 = ScalarPayload(value=100.0, unit="degC")
        self.db.add_memory(payload=s3, embedding=np.zeros(128), triples=[("sensor1", "temp", "val")])

        # Flush to cold tier
        self.db.force_checkpoint()
        
        # Run analytics
        from epochdb.cold_tier import ColdTierAnalytics
        analytics = ColdTierAnalytics(self.test_dir)
        trend = analytics.query_trend("temp", "sensor1")
        
        self.assertIn("mean", trend)
        # Mean of [0, 0, 100] is 33.33...
        self.assertAlmostEqual(trend["mean"], 33.33333333333, places=5)
        self.assertEqual(trend["unit"], "degC")

    def test_anomalies_normalized(self):
        """Test anomaly detection with unit normalization."""
        # Mean will be ~10
        for _ in range(10):
            s = ScalarPayload(value=10.0, unit="m")
            self.db.add_memory(payload=s, embedding=np.zeros(128), triples=[("track1", "dist", "val")])
        
        # Add an anomaly: 1km (1000m)
        s_anomaly = ScalarPayload(value=1.0, unit="km")
        self.db.add_memory(payload=s_anomaly, embedding=np.zeros(128), triples=[("track1", "dist", "val")])
        
        self.db.force_checkpoint()
        
        from epochdb.cold_tier import ColdTierAnalytics
        analytics = ColdTierAnalytics(self.test_dir)
        anomalies = analytics.detect_anomalies("dist", "track1", sigma=3.0)
        
        self.assertEqual(len(anomalies), 1)
        self.assertAlmostEqual(anomalies[0]["value"], 1000.0)
        self.assertEqual(anomalies[0]["unit"], "m")

    def test_duckdb_sql_query(self):
        """Test DuckDB SQL execution over Cold Tier Parquet archives."""
        for i in range(5):
            s = ScalarPayload(value=float(i * 10), unit="degC")
            self.db.add_memory(payload=s, embedding=np.zeros(128), triples=[("sensor2", "temp", str(i))])
        
        self.db.force_checkpoint()

        # Execute DuckDB query via engine
        res = self.db.query_sql("SELECT COUNT(*) as count, AVG(scalar_value) as avg_val FROM cold_tier WHERE scalar_unit = 'degC'")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["count"], 5)
        self.assertEqual(res[0]["avg_val"], 20.0)

if __name__ == "__main__":
    unittest.main()
