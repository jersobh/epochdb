"""
demo.py — EpochDB v0.4.0 Self-Contained Walkthrough
=====================================================
Demonstrates all core features using Gemini embeddings (gemini-embedding-2-preview)
and the new auto-embedding convenience API.

Usage:
    export GEMINI_API_KEY=your_key
    python demo.py
"""

import os
import time
import shutil
import logging
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)

from utils.shared import load_dotenv
load_dotenv(os.path.dirname(os.path.abspath(__file__)))

from google import genai

from epochdb import EpochDB

# ── ANSI terminal colours ──────────────────────────────────────────────────────
R  = "\033[0m"
YL = "\033[93m"
CY = "\033[96m"
GN = "\033[92m"
BL = "\033[94m"
RD = "\033[91m"
BD = "\033[1m"

def hr(title=""):
    line = "─" * 60
    if title:
        pad = (58 - len(title)) // 2
        print(f"\n╔{line}╗")
        print(f"║{' ' * pad}{BD}{title}{R}{' ' * (58 - pad - len(title))}║")
        print(f"╚{line}╝\n")
    else:
        print(f"\n{line}\n")


# ── Gemini Embedder ─────────────────────────────────────────────────────────────
class GeminiEmbedder:
    MODEL = "gemini-embedding-2-preview"
    DIM   = 3072

    def __init__(self, client):
        self.client = client

    def encode(self, text: str) -> np.ndarray:
        resp = self.client.models.embed_content(model=self.MODEL, contents=text)
        return np.array(resp.embeddings[0].values, dtype=np.float32)


def main():
    storage_dir = "./.epochdb_demo"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)
        time.sleep(0.3)

    hr("EpochDB v0.4.0 — Feature Walkthrough")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"{RD}Error: GEMINI_API_KEY is not set.{R}")
        return

    client  = genai.Client(api_key=api_key)
    embedder = GeminiEmbedder(client)
    DIM      = GeminiEmbedder.DIM

    print(f"{YL}Connecting to Gemini embedding model ({embedder.MODEL}, {DIM}D)...{R}")
    db = EpochDB(storage_dir=storage_dir, dim=DIM)

    # ── STEP 1: Ingesting Disparate Facts ─────────────────────────────────────
    hr("Step 1 — Ingesting Disparate Facts")

    facts = [
        (
            "Jeff is the lead developer of Project Zero.",
            [("Jeff", "leads", "Project Zero")],
        ),
        (
            "Project Zero uses Parquet files for cold storage.",
            [("Project Zero", "uses", "Parquet")],
        ),
        (
            "Parquet is a columnar storage format optimised for analytics.",
            [("Parquet", "is_a", "Columnar Format")],
        ),
        (
            "The weather in Lisbon is warm and sunny today.",
            [],  # Irrelevant fluff — should not surface in targeted queries.
        ),
    ]

    for payload, triples in facts:
        print(f"  {GN}→{R} {payload}")
        emb = embedder.encode(payload)
        db.add_memory(payload=payload, embedding=emb, triples=triples)
        time.sleep(0.2)  # Avoid rate-limit bursts.

    # ── STEP 2: Basic Semantic Recall ─────────────────────────────────────────
    hr("Step 2 — Semantic Recall (0 hops)")

    query = "Tell me about Jeff"
    print(f"  {YL}Query:{R} \"{query}\"\n")

    q_emb   = embedder.encode(query)
    results = db.recall(q_emb, top_k=1, expand_hops=0, query_entities=["Jeff"])

    for r in results:
        print(f"  {BD}[HIT]{R} {r.payload}")
        print(f"        epoch={r.epoch_id}  access_count={r.access_count}  saliency={r.calculate_saliency():.4f}")

    # ── STEP 3: Multi-Hop Relational Reasoning ────────────────────────────────
    hr("Step 3 — Multi-Hop Reasoning (2 hops)")

    complex_query = "What file format does Jeff's project use?"
    print(f"  {YL}Query:{R} \"{complex_query}\"")
    print(f"  {BL}Note:{R} 'Jeff' and 'Parquet' share no keywords — only the KG bridges them.\n")

    q_emb   = embedder.encode(complex_query)
    results = db.recall(
        q_emb,
        top_k=4,
        expand_hops=2,
        query_entities=["Jeff", "Project Zero", "Parquet"],
    )

    print(f"  {BD}Results (3-Way RRF ranked):{R}")
    for i, r in enumerate(results, 1):
        highlight = GN if "Parquet" in r.payload else R
        print(f"  {i}. {highlight}{r.payload}{R}")

    # ── STEP 4: Hot → Cold Tier Lifecycle ─────────────────────────────────────
    hr("Step 4 — Epoch Checkpoint (Hot → Cold)")

    print(f"  {YL}Flushing Working Memory to Parquet archive...{R}\n")
    db.force_checkpoint()

    hot_count = len(db.hot_tier.atoms)
    parquet_files = [f for f in os.listdir(storage_dir) if f.endswith(".parquet")]
    print(f"  Hot Tier atoms:    {BD}{hot_count}{R}  (cleared)")
    print(f"  Parquet archives:  {BD}{len(parquet_files)}{R}  ({', '.join(parquet_files)})\n")

    # ── STEP 5: Long-Term Memory Recall from Cold Tier ────────────────────────
    hr("Step 5 — Long-Term Recall from Cold Tier")

    print(f"  {YL}Re-querying after checkpoint — all data now lives on disk.{R}\n")
    results = db.recall(q_emb, top_k=3, expand_hops=2)

    for r in results:
        print(f"  {BD}[COLD HIT]{R} {r.payload}")
        print(f"             epoch={r.epoch_id}  access_count={r.access_count}")

    # ── STEP 6: Crash-Recovery (WAL Replay) ───────────────────────────────────
    hr("Step 6 — Crash Recovery (WAL Replay)")

    print(f"  {YL}Simulating a mid-transaction crash...{R}")

    # Write an uncommitted ADD directly into the WAL.
    from epochdb.atom import UnifiedMemoryAtom
    ghost_emb  = embedder.encode("This atom was never committed before the crash.")
    ghost_atom = UnifiedMemoryAtom(
        payload="Crash-recovered atom",
        embedding=ghost_emb,
        triples=[("crash", "recovered_by", "WAL")],
        epoch_id=db.current_epoch_id,
    )
    db.wal.append("ADD", ghost_atom.to_dict())
    db.wal._file.flush()

    # Simulate crash: release lock without calling close().
    db.lock.release()
    db.wal.close()

    print(f"  {GN}Re-opening database — WAL replay should restore the atom...{R}")
    db2 = EpochDB(storage_dir=storage_dir, dim=DIM)

    recovered = [a for a in db2.hot_tier.atoms.values() if a.payload == "Crash-recovered atom"]
    if recovered:
        print(f"  {GN}{BD}✓ WAL replay succeeded:{R} \"{recovered[0].payload}\"")
    else:
        print(f"  {RD}✗ WAL replay did not recover the atom.{R}")

    db2.close()

    hr("Demo Complete")
    print(f"  All data persisted in {CY}{storage_dir}{R}")
    print(f"  Run again to observe instant recall from the Cold Tier archive.\n")


if __name__ == "__main__":
    main()
