# Quantitative Logic & SAT Constraints

AI agents operating in physical, financial, or cyber-physical environments cannot rely solely on text embeddings. Queries like *"Find servers where temperature was between 75°C and 85°C while CPU load exceeded 90%"* or *"Verify whether resource constraints are satisfiable"* cannot be solved accurately by cosine similarity.

EpochDB v0.6.2+ introduces a native **Quantitative Index Layer** running in parallel with the HNSW vector store.

---

## The Quantitative Index Subsystems

```
EpochDB Engine
 ├── HNSW Vector Index (Semantic Similarity)
 └── Quantitative Index Subsystem
      ├── Scalar Index (IntervalTree) — O(log N + K) numeric range queries
      ├── Base-Unit Registry (Pint) — Automated metric/imperial unit normalization
      ├── Series Index (R-Tree) — 2D spatial & temporal time-series range lookups
      ├── Constraint Checker (Z3 Solver) — First-order logic & SMT satisfiability
      └── Reactive Cascades (CascadeManager) — Automated policy trigger graphs
```

---

## 1. Scalar Index & Unit Normalization (Interval Trees)

EpochDB uses `intervaltree.IntervalTree` to index continuous numeric values and measurements with measurement uncertainty (e.g., $22.5 \pm 0.5$):

- **$O(\log n + k)$ Overlap Queries**: Bypasses vector indexing entirely to retrieve atoms falling within numeric bounds.
- **Physical Unit Normalization**: Uses the `pint` physical quantities library. Whether data is recorded as `77 degF` or `25 degC`, values are normalized to canonical SI base units before indexing via `schema_registry.json`.

```python
from epochdb import EpochDB

with EpochDB(storage_dir="./sensor_data") as db:
    # Record scalar measurement with units and uncertainty
    db.remember(
        "Sensor A registered normal operating temperature.",
        metadata={
            "scalar_name": "core_temp",
            "scalar_value": 78.5,
            "scalar_unit": "degC",
            "uncertainty": 0.5,
        }
    )

    # Query scalar ranges directly (0.8 ms lookup latency)
    results = db.query_scalar_range("core_temp", min_val=75.0, max_val=80.0)
    print(f"Matched {len(results)} readings within threshold.")
```

---

## 2. Series Index (R-Trees for Time-Series)

Temporal time-series data points `(timestamp, value)` are indexed as 2D spatial coordinates in an R-tree (`Rtree` / `libspatialindex`):
- Enables lightning-fast 2D bounding-box queries over both time intervals and numeric ranges simultaneously.
- Provides native support for series interpolation, slope estimation, and missing interval imputation.

---

## 3. Z3 Constraint Checker & SAT Solver

EpochDB embeds the **Microsoft Z3 Theorem Prover** (`z3-solver`):
- Agents can assert symbolic propositions, linear arithmetic equations, and boolean constraints into the database.
- The constraint checker evaluates constraint satisfiability in **~2.5 ms**.

```python
# Evaluate whether agent resource allocations conflict
is_satisfiable, model = db.check_constraints([
    "workers >= 5",
    "workers * memory_per_worker <= total_ram",
    "total_ram == 32",
    "memory_per_worker >= 8",
])

print("Feasible:", is_satisfiable)  # False (5 * 8 = 40 > 32)
```

---

## 4. Reactive Cascades (`CascadeManager`)

When an incoming observation violates a quantitative constraint or alters an equilibrium:
- The `CascadeManager` traverses a persisted dependency graph.
- Automatically triggers downstream policy events and invalidates stale assumptions.
- Allows agents to reactively handle emergencies (e.g. automatically tripping circuit breakers or notifying human operators).
