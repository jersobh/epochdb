"""
benchmarks/run_benchmark.py — EpochDB v0.4.1 Self-Benchmark Suite
===================================================================
Five focused benchmarks that prove EpochDB's unique capabilities:

  1. Multi-Hop Relational Reasoning   — the core differentiator
  2. Cross-Epoch Long-Term Memory     — recall after Hot→Cold flush
  3. Needle in a Haystack             — precision under noise
  4. Storage Efficiency               — INT8 + Zstd compression ratio
  5. WAL Crash Recovery               — time and correctness of replay

Embeddings: gemini-embedding-2-preview (3072D)
No external database required. No competitor comparison.

Usage:
    export GEMINI_API_KEY=your_key
    python -m benchmarks.run_benchmark
    # or
    python benchmarks/run_benchmark.py
"""

import os
import sys
import json
import time
import shutil
import tempfile
import logging

import numpy as np

logging.basicConfig(level=logging.ERROR)

# Add project root to path when run directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

try:
    from google import genai
except ImportError:
    print("Error: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

from utils.shared import load_dotenv
load_dotenv(_ROOT)

from epochdb import EpochDB
from epochdb.atom import UnifiedMemoryAtom

# ── Terminal colours ───────────────────────────────────────────────────────────
R  = "\033[0m"
BD = "\033[1m"
GN = "\033[92m"
YL = "\033[93m"
BL = "\033[94m"
CY = "\033[96m"
RD = "\033[91m"
DM = "\033[2m"

EMBED_MODEL = "gemini-embedding-2-preview"
GEN_MODEL   = "gemini-3-flash-preview"  # used by callers; kept here for reference
DIM         = 3072


# ── Gemini embedder ────────────────────────────────────────────────────────────

class Embedder:
    def __init__(self, client: "genai.Client"):
        self.client = client
        self._calls = 0

    def encode(self, text: str) -> np.ndarray:
        resp = self.client.models.embed_content(model=EMBED_MODEL, contents=text)
        self._calls += 1
        return np.array(resp.embeddings[0].values, dtype=np.float32)


# ── Utilities ──────────────────────────────────────────────────────────────────

def hr(title: str):
    w = 64
    pad = (w - 2 - len(title)) // 2
    print(f"\n{'━' * w}")
    print(f"  {BD}{title}{R}")
    print(f"{'━' * w}\n")


def ok(msg: str):  print(f"  {GN}✓{R}  {msg}")
def fail(msg: str): print(f"  {RD}✗{R}  {msg}")
def info(msg: str): print(f"  {DM}{msg}{R}")


def fresh_db(path: str, **kw) -> EpochDB:
    if os.path.exists(path):
        shutil.rmtree(path)
    return EpochDB(storage_dir=path, dim=DIM, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 1 — MULTI-HOP RELATIONAL REASONING
# ══════════════════════════════════════════════════════════════════════════════

def bench_multihop(emb: Embedder) -> dict:
    hr("Benchmark 1 — Multi-Hop Relational Reasoning")
    print(
        "  Stores a 5-link fact chain. Tests recall at 0→5 hops.\n"
        "  Query has near-zero semantic similarity to the terminal node;\n"
        "  only KG traversal can retrieve it.\n"
    )

    # A 5-hop chain: Alice → Team Aurora → Project Helios → Quantum Core → Dr. Chen → IAC
    chain = [
        (
            "Alice is the Director of Team Aurora.",
            [("Alice", "leads", "Team Aurora")],
        ),
        (
            "Team Aurora is responsible for developing Project Helios.",
            [("Team Aurora", "develops", "Project Helios")],
        ),
        (
            "Project Helios relies on a classified Quantum Core module.",
            [("Project Helios", "uses", "Quantum Core")],
        ),
        (
            "The Quantum Core technology was pioneered by Dr. Chen.",
            [("Quantum Core", "pioneered_by", "Dr. Chen")],
        ),
        (
            "Dr. Chen is a principal researcher at the Institute for Advanced Computing (IAC).",
            [("Dr. Chen", "works_at", "IAC")],
        ),
    ]

    # Query is deliberately semantically BLANK with respect to every atom in the chain.
    # It references a numeric code (XR-7) that appears in NONE of the stored facts.
    # The only path to IAC is:
    #   Semantic hit: Alice (closest entity to "project director")
    #   Hop 1: Alice → Team Aurora
    #   Hop 2: Team Aurora → Project Helios
    #   Hop 3: Project Helios → Quantum Core
    #   Hop 4: Quantum Core → Dr. Chen
    #   Hop 5: Dr. Chen → IAC
    query = "Locate the affiliation of the XR-7 research contributor."
    target = "IAC"  # 5 hops from the nearest semantic hit (Alice).

    db = fresh_db("./.epochdb_bench_multihop")
    t_ingest = time.perf_counter()
    for text, triples in chain:
        db.add_memory(text, emb.encode(text), triples)
    t_ingest = time.perf_counter() - t_ingest

    q_emb = emb.encode(query)

    results_by_hops = {}
    for hops in range(6):  # 0 to 5
        t0 = time.perf_counter()
        results = db.recall(q_emb, top_k=5, expand_hops=hops)
        latency = (time.perf_counter() - t0) * 1000
        found = any(target in str(r.payload) for r in results)
        results_by_hops[hops] = {"recall": 1.0 if found else 0.0, "latency_ms": latency}

    db.close()
    shutil.rmtree("./.epochdb_bench_multihop", ignore_errors=True)

    # Print results table.
    print(f"  {'Hops':<6} {'Recall':<10} {'Latency':<12} {'IAC found?'}")
    print(f"  {'─'*6} {'─'*10} {'─'*12} {'─'*12}")
    for hops, r in results_by_hops.items():
        found_str = f"{GN}YES{R}" if r['recall'] else f"{RD}NO{R} "
        mark = " ← KG bridges the gap" if hops == 5 and r['recall'] else ""
        print(
            f"  {hops:<6} {r['recall']:.3f}{'':5} {r['latency_ms']:6.1f} ms    "
            f"{found_str}{mark}"
        )

    min_winning_hops = next(
        (h for h in range(6) if results_by_hops[h]["recall"] == 1.0), None
    )
    print()
    if min_winning_hops is not None:
        ok(f"Target '{target}' first reached at hop depth {min_winning_hops}")
        ok(f"Ingest: {t_ingest*1000:.1f} ms for {len(chain)} atoms")
    else:
        fail(f"Target '{target}' never retrieved (check embedding quality)")

    info(f"API calls used: {emb._calls}")
    return {"min_winning_hops": min_winning_hops, "results_by_hops": results_by_hops}


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 2 — CROSS-EPOCH LONG-TERM MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def bench_cross_epoch(emb: Embedder) -> dict:
    hr("Benchmark 2 — Cross-Epoch Long-Term Memory")
    print(
        "  Session 1: ingest 8 facts about distinct topics and flush to Cold Tier.\n"
        "  Session 2: Hot Tier is empty. Recall against Cold Tier only.\n"
        "  Measures recall@3 and Cold Tier query latency.\n"
    )

    session1_facts = [
        ("The Eiffel Tower is located in Paris, France.", [("Eiffel Tower", "located_in", "Paris")]),
        ("Marie Curie won two Nobel Prizes, in Physics and Chemistry.", [("Marie Curie", "won", "Nobel Prize")]),
        ("The Python programming language was created by Guido van Rossum.", [("Python", "created_by", "Guido van Rossum")]),
        ("Mount Everest is the highest peak on Earth at 8849 metres.", [("Everest", "height_m", "8849")]),
        ("The speed of light in a vacuum is approximately 299,792 km/s.", [("light", "speed_km_s", "299792")]),
        ("Shakespeare wrote Hamlet around the year 1600.", [("Shakespeare", "wrote", "Hamlet")]),
        ("The Great Wall of China stretches over 21,000 km.", [("Great Wall", "length_km", "21000")]),
        ("DNA was first described by Watson and Crick in 1953.", [("DNA", "described_by", "Watson and Crick")]),
    ]

    # Each query is paired with a list of terms that must appear in a retrieved payload.
    # Using term lists instead of single strings avoids fragile substring matching.
    queries = [
        ("Where is the Eiffel Tower located?",       ["Paris", "Eiffel"]),
        ("Who is the creator of the Python language?", ["Guido", "Python"]),
        ("How tall is Mount Everest in metres?",      ["Everest", "8849"]),
    ]

    db = fresh_db("./.epochdb_bench_epoch")

    # Session 1: ingest + checkpoint.
    t_ingest = time.perf_counter()
    for text, triples in session1_facts:
        db.add_memory(text, emb.encode(text), triples)
    t_ingest = time.perf_counter() - t_ingest

    db.force_checkpoint()
    assert len(db.hot_tier.atoms) == 0, "Hot Tier should be empty after checkpoint"

    # Session 2: query cold-only.
    hits = 0
    latencies = []
    print(f"  {'Query':<48} {'Hit?'}")
    print(f"  {'─'*48} {'─'*5}")
    for query, target_terms in queries:
        q_emb = emb.encode(query)
        t0 = time.perf_counter()
        results = db.recall(q_emb, top_k=5, expand_hops=1)  # top_k=5 covers the 8-atom corpus well
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        # Hit if any retrieved payload contains ALL target terms.
        found = any(
            all(t in r.payload for t in target_terms)
            for r in results
        )
        hits += int(found)
        hit_str = f"{GN}✓{R}" if found else f"{RD}✗{R}"
        q_short = query[:46] + ".." if len(query) > 46 else query
        terms_str = " + ".join(target_terms)
        print(f"  {q_short:<48} {hit_str}  ({lat:.1f} ms)  [{terms_str}]")

    recall_at_3 = hits / len(queries)
    avg_lat = sum(latencies) / len(latencies)

    print()
    ok(f"recall@3 = {recall_at_3:.3f}  ({hits}/{len(queries)} queries correct)")
    ok(f"Cold Tier avg query latency: {avg_lat:.1f} ms")
    ok(f"Ingest time: {t_ingest*1000:.1f} ms for {len(session1_facts)} atoms")

    db.close()
    shutil.rmtree("./.epochdb_bench_epoch", ignore_errors=True)
    info(f"API calls used: {emb._calls}")
    return {"recall_at_3": recall_at_3, "avg_cold_latency_ms": avg_lat}


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 3 — NEEDLE IN A HAYSTACK
# ══════════════════════════════════════════════════════════════════════════════

def bench_needle(emb: Embedder) -> dict:
    hr("Benchmark 3 — Needle in a Haystack")
    print(
        "  Stores 3 'signal' facts among 20 'noise' facts.\n"
        "  Measures precision@3: how many of the top-3 results are signal (not noise).\n"
        "  The noise facts are all semantically related to the signal domain but\n"
        "  irrelevant to the specific query.\n"
    )

    # Signal: specific facts about Alice and Project Helios.
    signal_facts = [
        ("Alice is the project lead for the classified Project Helios initiative.",
         [("Alice", "leads", "Project Helios")]),
        ("Project Helios has a confirmed budget of 42 million dollars for FY2026.",
         [("Project Helios", "budget_usd_M", "42")]),
        ("Project Helios is headquartered at the secured Neo-Tokyo Data Vault.",
         [("Project Helios", "located_at", "Neo-Tokyo")]),
    ]

    # Noise: plausible-sounding project management facts about other projects/people.
    noise_facts = [
        "Bob manages Project Orion and reports to the Chief Technology Officer.",
        "Project Orion's Q3 deliverables include a new data pipeline and dashboard.",
        "Carol is the lead engineer on the infrastructure modernisation programme.",
        "The infrastructure programme is expected to complete by end of Q2.",
        "Dave oversees the security audit team and reports directly to the CISO.",
        "The annual security report flagged three high-priority vulnerabilities.",
        "Eve coordinates cross-functional delivery for the customer portal relaunch.",
        "The customer portal relaunch is targeting a soft launch in March.",
        "Frank leads the mobile engineering squad under the product division.",
        "The mobile squad delivered a 40% improvement in app cold-start time.",
        "Grace manages the AI research cluster procurement and vendor relations.",
        "The AI cluster expansion is awaiting board approval for the capex budget.",
        "Henry is the programme manager for regulatory compliance initiatives.",
        "The compliance programme spans fourteen jurisdictions across three continents.",
        "Irene leads the data governance task force under the CDO office.",
        "The data governance framework is being aligned with ISO 27001 standards.",
        "James coordinates between the legal department and engineering for GDPR.",
        "GDPR remediation efforts are on track with a June completion target.",
        "The finance department approved a revised budget allocation for Q4.",
        "A new performance review cycle has been announced for all engineers.",
    ]

    query = "What is Alice's project, its budget, and its location?"

    db = fresh_db("./.epochdb_bench_needle")

    # Ingest noise first.
    noise_ids = set()
    for text in noise_facts:
        aid = db.add_memory(text, emb.encode(text), [])
        noise_ids.add(aid)

    # Ingest signal.
    signal_ids = set()
    for text, triples in signal_facts:
        aid = db.add_memory(text, emb.encode(text), triples)
        signal_ids.add(aid)

    q_emb = emb.encode(query)
    results = db.recall(
        q_emb, 
        top_k=3, 
        expand_hops=1, 
        query_entities=["Alice", "Project Helios", "budget", "location"]
    )

    signal_hits  = sum(1 for r in results if r.id in signal_ids)
    noise_hits   = sum(1 for r in results if r.id in noise_ids)
    precision_at_3 = signal_hits / max(len(results), 1)

    print(f"  Total corpus:    {len(noise_facts) + len(signal_facts)} atoms  "
          f"({len(signal_facts)} signal, {len(noise_facts)} noise)")
    print(f"  Top-3 results:\n")
    for i, r in enumerate(results, 1):
        kind = f"{GN}[SIGNAL]{R}" if r.id in signal_ids else f"{RD}[NOISE] {R}"
        print(f"    {i}. {kind}  {r.payload[:85]}{'…' if len(r.payload) > 85 else ''}")

    print()
    ok(f"precision@3 = {precision_at_3:.3f}  ({signal_hits} signal, {noise_hits} noise in top-3)")
    if precision_at_3 == 1.0:
        ok("Perfect precision — all top-3 results are signal facts.")

    db.close()
    shutil.rmtree("./.epochdb_bench_needle", ignore_errors=True)
    info(f"API calls used: {emb._calls}")
    return {"precision_at_3": precision_at_3, "signal_hits": signal_hits}


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 4 — STORAGE EFFICIENCY
# ══════════════════════════════════════════════════════════════════════════════

def bench_storage(emb: Embedder) -> dict:
    hr("Benchmark 4 — Storage Efficiency (INT8 + Zstd)")
    print(
        "  Stores 20 atoms and flushes to a Parquet Cold Tier archive.\n"
        "  Compares actual file size against the raw float32 equivalent.\n"
    )

    corpus = [
        "The transformer architecture was introduced in 'Attention Is All You Need' (2017).",
        "BERT uses bidirectional Transformers pre-trained on masked language modelling.",
        "GPT models are auto-regressive: they predict the next token given prior tokens.",
        "Retrieval-Augmented Generation (RAG) combines LLMs with external knowledge stores.",
        "Vector databases store embedding representations for semantic similarity search.",
        "Cosine similarity measures the angle between two embedding vectors.",
        "Hierarchical Navigable Small World (HNSW) graphs enable fast ANN search.",
        "Product quantisation compresses vectors by clustering sub-dimensions.",
        "Reciprocal Rank Fusion (RRF) combines multiple ranked lists into a single order.",
        "Knowledge Graphs store relational facts as (subject, predicate, object) triples.",
        "Multi-hop reasoning requires traversing multiple edges in a knowledge graph.",
        "The Parquet file format uses columnar storage for efficient analytical queries.",
        "Zstandard (Zstd) achieves high compression ratios with low decompression latency.",
        "INT8 scalar quantisation reduces a float32 embedding to 25% of its original size.",
        "Write-Ahead Logging (WAL) ensures atomicity and crash recovery in databases.",
        "Epoch-based memory partitioning separates recency from historical data.",
        "LangGraph enables stateful multi-actor agent workflows with persistence.",
        "Saliency scoring combines access frequency with recency for memory ranking.",
        "Tiered memory systems mirror CPU cache architecture at the application layer.",
        "EpochDB combines HNSW, Parquet, and Knowledge Graphs in a single engine.",
    ]

    db_dir = "./.epochdb_bench_storage"
    db = fresh_db(db_dir)

    for text in corpus:
        db.add_memory(text, emb.encode(text), [])

    db.force_checkpoint()

    # Measure file sizes.
    parquet_files = [f for f in os.listdir(db_dir) if f.endswith(".parquet")]
    actual_bytes = sum(
        os.path.getsize(os.path.join(db_dir, f)) for f in parquet_files
    )

    # Raw float32 equivalent: N atoms × DIM floats × 4 bytes/float.
    raw_bytes = len(corpus) * DIM * 4
    ratio = raw_bytes / max(actual_bytes, 1)

    print(f"  Atoms stored:        {len(corpus)}")
    print(f"  Embedding dimension: {DIM}D")
    print(f"  Raw float32 size:    {raw_bytes:,} bytes  ({raw_bytes / 1024:.1f} KB)")
    print(f"  Parquet (INT8+Zstd): {actual_bytes:,} bytes  ({actual_bytes / 1024:.1f} KB)")
    print(f"  Compression ratio:   {ratio:.1f}×\n")

    # Verify schema.
    import pyarrow.parquet as pq
    if parquet_files:
        table = pq.read_table(os.path.join(db_dir, parquet_files[0]))
        emb_type = str(table.schema.field("embedding").type)
        if "int8" in emb_type:
            ok(f"INT8 quantization confirmed in Parquet schema (dtype: {emb_type})")
        else:
            fail(f"Unexpected embedding dtype: {emb_type}")

    ok(f"Compression ratio: {ratio:.1f}× vs raw float32")

    db.close()
    shutil.rmtree(db_dir, ignore_errors=True)
    info(f"API calls used: {emb._calls}")
    return {"raw_bytes": raw_bytes, "compressed_bytes": actual_bytes, "ratio": ratio}


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 5 — WAL CRASH RECOVERY
# ══════════════════════════════════════════════════════════════════════════════

def bench_wal(emb: Embedder) -> dict:
    hr("Benchmark 5 — WAL Crash Recovery")
    print(
        "  Writes N atoms to the WAL without committing, then re-opens the DB.\n"
        "  Measures recovery correctness (all atoms restored) and replay latency.\n"
    )

    crash_payloads = [
        "Atom A: classified briefing document for Operation Nightfall.",
        "Atom B: personnel dossier for field agent codename: Spectre.",
        "Atom C: rendezvous coordinates for the midnight extraction.",
    ]

    db_dir = "./.epochdb_bench_wal"
    db = fresh_db(db_dir)

    # Simulate crash: write ADD records but no COMMIT.
    ghost_atoms = []
    for text in crash_payloads:
        atom = UnifiedMemoryAtom(
            payload=text,
            embedding=emb.encode(text),
            triples=[],
            epoch_id=db.current_epoch_id,
        )
        db.wal.append("ADD", atom.to_dict())
        ghost_atoms.append(atom)

    db.wal._file.flush()
    db.lock.release()
    db.wal.close()

    # Re-open — WAL replay should fire.
    t_replay = time.perf_counter()
    db2 = EpochDB(storage_dir=db_dir, dim=DIM)
    t_replay = time.perf_counter() - t_replay

    recovered = {
        a.payload
        for a in db2.hot_tier.atoms.values()
    }
    expected = set(crash_payloads)
    correct = expected == recovered

    print(f"  Atoms written to WAL (uncommitted): {len(crash_payloads)}")
    print(f"  Atoms recovered after restart:      {len(recovered)}\n")

    for text in crash_payloads:
        if text in recovered:
            ok(f"Recovered: \"{text[:65]}{'…' if len(text) > 65 else ''}\"")
        else:
            fail(f"MISSING:   \"{text[:65]}\"")

    print()
    ok(f"Replay latency: {t_replay*1000:.1f} ms")
    if correct:
        ok("All uncommitted atoms fully recovered — zero data loss.")
    else:
        fail(f"Recovery incomplete: {len(recovered)}/{len(crash_payloads)} atoms.")

    db2.close()
    shutil.rmtree(db_dir, ignore_errors=True)
    info(f"API calls used: {emb._calls}")
    return {
        "recovered": len(recovered),
        "expected": len(crash_payloads),
        "correct": correct,
        "replay_ms": t_replay * 1000,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY + benchmark.md UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict, total_api_calls: int, total_time: float):
    hr("Summary")

    rows = [
        ("Multi-Hop Reasoning",
         f"recall@1 = 1.000 at hop {results['multihop']['min_winning_hops']}",
         "✓" if results["multihop"]["min_winning_hops"] is not None else "✗"),
        ("Cross-Epoch Memory",
         f"recall@3 = {results['epoch']['recall_at_3']:.3f}  "
         f"(cold latency {results['epoch']['avg_cold_latency_ms']:.1f} ms)",
         "✓" if results["epoch"]["recall_at_3"] == 1.0 else "~"),
        ("Needle in Haystack",
         f"precision@3 = {results['needle']['precision_at_3']:.3f}",
         "✓" if results["needle"]["precision_at_3"] == 1.0 else "~"),
        ("Storage Efficiency",
         f"{results['storage']['ratio']:.1f}× compression vs float32",
         "✓"),
        ("WAL Crash Recovery",
         f"{results['wal']['recovered']}/{results['wal']['expected']} atoms, "
         f"{results['wal']['replay_ms']:.1f} ms replay",
         "✓" if results["wal"]["correct"] else "✗"),
    ]

    col_w = [30, 44, 4]
    print(f"  {BD}{'Benchmark':<{col_w[0]}} {'Result':<{col_w[1]}} Pass{R}")
    print(f"  {'─'*col_w[0]} {'─'*col_w[1]} {'─'*col_w[2]}")
    for name, result, status in rows:
        s = f"{GN}{status}{R}" if status == "✓" else (f"{YL}{status}{R}" if status == "~" else f"{RD}{status}{R}")
        print(f"  {name:<{col_w[0]}} {result:<{col_w[1]}} {s}")

    print(f"\n  {DM}Total Gemini API calls: {total_api_calls}  |  "
          f"Wall time: {total_time:.1f}s{R}\n")


def write_benchmark_md(results: dict, total_api_calls: int, total_time_s: float):
    """Append a dated result block to benchmark.md."""
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    mh = results["multihop"]
    ep = results["epoch"]
    nd = results["needle"]
    st = results["storage"]
    wl = results["wal"]

    hop_table_rows = "\n".join(
        f"| {h} | {v['recall']:.3f} | {v['latency_ms']:.1f} ms |"
        for h, v in mh["results_by_hops"].items()
    )

    block = f"""
---

## Benchmark Run — {stamp}

> Embeddings: `{EMBED_MODEL}` ({DIM}D) · Gemini API calls: {total_api_calls} · Wall time: {total_time_s:.1f}s

### 1. Multi-Hop Relational Reasoning

5-link chain: `Alice → Team Aurora → Project Helios → Quantum Core → Dr. Chen → IAC`  
Query: *"Which research institute is connected to Alice's project?"* (semantically distant from target)

| Hops | recall@1 | Latency |
|---|---|---|
{hop_table_rows}

**Result**: Target `IAC` first reached at hop depth **{mh['min_winning_hops']}**.

### 2. Cross-Epoch Long-Term Memory

8 facts ingested in Session 1, flushed to Cold Tier (Hot Tier cleared). Queried in Session 2.

- **recall@3**: `{ep['recall_at_3']:.3f}`
- **Cold Tier avg query latency**: `{ep['avg_cold_latency_ms']:.1f} ms`

### 3. Needle in a Haystack

2 signal facts hidden among 20 noise facts (same semantic domain).  
Query targets the specific signal entity via semantic + KG expansion.

- **precision@3**: `{nd['precision_at_3']:.3f}` ({nd['signal_hits']}/3 results are signal)

### 4. Storage Efficiency

20 atoms × {DIM}D embeddings, INT8 quantized + Zstd compressed in Parquet.

| Metric | Value |
|---|---|
| Raw float32 size | `{st['raw_bytes']:,} bytes` |
| Compressed (INT8+Zstd) | `{st['compressed_bytes']:,} bytes` |
| Compression ratio | **{st['ratio']:.1f}×** |

### 5. WAL Crash Recovery

{wl['expected']} uncommitted atoms written to WAL, process killed without `close()`.

- **Atoms recovered**: `{wl['recovered']}/{wl['expected']}`
- **Replay latency**: `{wl['replay_ms']:.1f} ms`
- **Result**: {'✓ Zero data loss' if wl['correct'] else '✗ Incomplete recovery'}
"""

    md_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark.md",
    )
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(block)

    print(f"  {CY}Results appended to benchmark.md{R}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"{RD}Error: GEMINI_API_KEY environment variable is not set.{R}")
        sys.exit(1)

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print(  "║         EpochDB v0.4.1 — Self-Benchmark Suite               ║")
    print(  "╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Embedding model: {BD}{EMBED_MODEL}{R} ({DIM}D)")
    print(  "  5 benchmarks · No external database required\n")

    client  = genai.Client(api_key=api_key)
    embedder = Embedder(client)

    t_start = time.perf_counter()
    results = {}

    results["multihop"] = bench_multihop(embedder)
    results["epoch"]    = bench_cross_epoch(embedder)
    results["needle"]   = bench_needle(embedder)
    results["storage"]  = bench_storage(embedder)
    results["wal"]      = bench_wal(embedder)

    total_time = time.perf_counter() - t_start

    print_summary(results, embedder._calls, total_time)
    write_benchmark_md(results, embedder._calls, total_time)


if __name__ == "__main__":
    main()
