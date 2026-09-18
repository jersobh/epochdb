"""
benchmark_tiered_retrieval.py — Exhaustive vs probed cold-tier search
=====================================================================

Before (exhaustive): open every cold-epoch HNSW on each query.
After  (probe):      recency window + GEI entity epochs + centroid neighbours.

Runs fully offline with synthetic embeddings (no API keys, no HF models).

Usage:
    python examples/benchmark_tiered_retrieval.py
    python examples/benchmark_tiered_retrieval.py --epochs 40 --repeats 21
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time

import numpy as np

from epochdb import EpochDB


DIM = 16
NEEDLE_ENTITY = "NeedleEntity"
NEEDLE_TEXT = "The needle lives in the oldest archive epoch."


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return arr if n < 1e-10 else arr / n


def query_vec() -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    return v


def noise_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 1.0, size=DIM).astype(np.float32)
    v[0] = 0.0
    return _unit(v)


def build_corpus(storage_dir: str, n_epochs: int, atoms_per_epoch: int) -> EpochDB:
    db = EpochDB(
        storage_dir=storage_dir,
        dim=DIM,
        embedding_model=None,
        recency_epochs=4,
        centroid_probes=6,
        cold_search_mode="probe",
    )
    q = query_vec()
    db.add_memory(
        NEEDLE_TEXT,
        q,
        triples=[(NEEDLE_ENTITY, "is", "hidden")],
    )
    pad_cluster = noise_vec(7)
    for j in range(atoms_per_epoch):
        db.add_memory(
            f"needle-pad {j}",
            _unit(pad_cluster + noise_vec(70 + j) * 0.05),
        )
    db.force_checkpoint()

    for epoch_i in range(n_epochs - 1):
        cluster = noise_vec(1000 + epoch_i)
        for j in range(atoms_per_epoch):
            jitter = noise_vec(50_000 + epoch_i * 200 + j) * 0.08
            db.add_memory(
                f"noise e{epoch_i} a{j}",
                _unit(cluster + jitter),
                triples=[(f"NoiseHub{epoch_i}", "mentions", f"item{j}")],
            )
        db.force_checkpoint()
    return db


def timed_recall(db: EpochDB, query, *, repeats: int, **kwargs):
    # Warmup so HNSW mmap / thread pool are not in the timed window.
    db.recall(query, **kwargs)
    samples = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = db.recall(query, **kwargs)
        samples.append((time.perf_counter() - t0) * 1000.0)
    plan = db.retriever.last_probe_plan
    payloads = [a.payload for a in last]
    hit = NEEDLE_TEXT in payloads
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p95_ms": statistics.quantiles(samples, n=20)[-1] if len(samples) >= 20 else max(samples),
        "epochs_total": plan.n_total if plan else 0,
        "epochs_searched": plan.n_searched if plan else 0,
        "auto_exhaustive": bool(plan.auto_exhaustive) if plan else False,
        "hit": hit,
        "from_entity": list(plan.from_entity) if plan else [],
        "from_recency": list(plan.from_recency) if plan else [],
        "from_centroid": list(plan.from_centroid) if plan else [],
    }


def fmt(row: dict) -> str:
    return (
        f"{row['median_ms']:8.2f} ms median | "
        f"{row['mean_ms']:8.2f} ms mean | "
        f"epochs {row['epochs_searched']}/{row['epochs_total']} | "
        f"needle {'HIT' if row['hit'] else 'MISS'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exhaustive vs probed cold-tier retrieval")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--atoms-per-epoch", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    storage = tempfile.mkdtemp(prefix="epochdb_tiered_bench_")
    print(f"Building archive: {args.epochs} epochs × {args.atoms_per_epoch} atoms")
    print(f"{storage}\n")
    db = build_corpus(storage, args.epochs, args.atoms_per_epoch)
    q = query_vec()
    try:
        n_epochs = len(db.cold_tier.get_all_epochs())
        print(f"Cold epochs on disk: {n_epochs}\n")

        print("1. Entity query  (GEI can route the needle epoch)")
        exhaustive_ent = timed_recall(
            db,
            q,
            repeats=args.repeats,
            top_k=args.k,
            query_entities=[NEEDLE_ENTITY],
            expand_hops=0,
            cold_search_mode="exhaustive",
        )
        probe_ent = timed_recall(
            db,
            q,
            repeats=args.repeats,
            top_k=args.k,
            query_entities=[NEEDLE_ENTITY],
            expand_hops=0,
            cold_search_mode="probe",
        )
        print(f"  exhaustive  {fmt(exhaustive_ent)}")
        print(f"  probe       {fmt(probe_ent)}")
        if exhaustive_ent["median_ms"] > 0:
            speedup = exhaustive_ent["median_ms"] / max(probe_ent["median_ms"], 1e-9)
            print(f"  speedup     {speedup:.2f}x  (median)")

        print("\n2. Semantic-only query  (no query_entities; centroid/recency only)")
        exhaustive_sem = timed_recall(
            db,
            q,
            repeats=args.repeats,
            top_k=args.k,
            query_entities=[],
            expand_hops=0,
            cold_search_mode="exhaustive",
        )
        probe_sem = timed_recall(
            db,
            q,
            repeats=args.repeats,
            top_k=args.k,
            query_entities=[],
            expand_hops=0,
            cold_search_mode="probe",
        )
        print(f"  exhaustive  {fmt(exhaustive_sem)}")
        print(f"  probe       {fmt(probe_sem)}")

        print("\nTakeaway")
        print("  exhaustive = pre-1.10 broadcast (every epoch HNSW)")
        print("  probe      = recency + entity epochs + centroid neighbours")
        if probe_ent["hit"] and exhaustive_ent["hit"]:
            print("  Entity queries keep needle recall while opening fewer files.")
        if exhaustive_sem["hit"] and not probe_sem["hit"]:
            print(
                "  Semantic-only needle in an old orthogonal epoch may miss under "
                "probe — pass query_entities or raise centroid_probes."
            )
        elif probe_sem["hit"]:
            print("  Centroid routing also recovered the semantic-only needle.")
    finally:
        db.close()
        shutil.rmtree(storage, ignore_errors=True)


if __name__ == "__main__":
    main()
