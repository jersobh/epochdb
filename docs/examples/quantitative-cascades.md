# Quantitative Cascades & Interval Trees

This example demonstrates how EpochDB combines continuous scalar intervals with automated reactive policy cascades.

---

## Runnable Example

```python
from epochdb import EpochDB

def run_quantitative_demo():
    with EpochDB(storage_dir="./quant_demo") as db:
        print("=== Recording Sensor Measurements ===")
        # Record scalar temperatures with physical units and measurement uncertainty
        db.remember(
            "Primary coolant loop temperature reading at 14:00.",
            metadata={
                "scalar_name": "coolant_temp",
                "scalar_value": 72.4,
                "scalar_unit": "degC",
                "uncertainty": 0.4,
            }
        )

        db.remember(
            "Primary coolant loop temperature spike at 14:05.",
            metadata={
                "scalar_name": "coolant_temp",
                "scalar_value": 96.8,
                "scalar_unit": "degC",
                "uncertainty": 0.8,
            }
        )

        # 1. Querying scalar range using IntervalTree (bypasses vector similarity)
        print("\n=== Range Query: Temperatures between 70°C and 80°C ===")
        normal_readings = db.query_scalar_range("coolant_temp", min_val=70.0, max_val=80.0)
        for r in normal_readings:
            print("Normal reading:", r.text)

        # 2. Detecting threshold excursions
        print("\n=== Excursion Query: Temperatures > 90°C ===")
        spikes = db.query_scalar_range("coolant_temp", min_val=90.0, max_val=150.0)
        for s in spikes:
            print("ALERT: Excursion detected ->", s.text)

if __name__ == "__main__":
    run_quantitative_demo()
```
