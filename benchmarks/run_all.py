"""
benchmarks/run_all.py — EpochDB v0.4.0 Named Benchmark Suite
=============================================================
Runs the three named benchmarks (LoCoMo, ConvoMem, LongMemEval) against EpochDB
using Gemini embeddings, then appends dated results to benchmark.md.

Usage:
    # From project root:
    venv/bin/python -m benchmarks.run_all

    # Or directly:
    venv/bin/python benchmarks/run_all.py
"""

import os
import sys
import time
import shutil
import logging

import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from utils.shared import load_dotenv
load_dotenv(_ROOT)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="  %(message)s",
)
logger = logging.getLogger("BenchmarkRunner")

# ── Imports ────────────────────────────────────────────────────────────────────
try:
    from google import genai
except ImportError:
    print("Error: google-genai is not installed. Run: pip install google-genai")
    sys.exit(1)

from epochdb import EpochDB
from benchmarks import locomo, convomem, longmemeval

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_MODEL = "gemini-embedding-2-preview"
DIM         = 3072
STORAGE_DIR = "./.epochdb_run_all"

R  = "\033[0m"
BD = "\033[1m"
GN = "\033[92m"
YL = "\033[93m"
RD = "\033[91m"
CY = "\033[96m"
DM = "\033[2m"


# ── Gemini Embedder ────────────────────────────────────────────────────────────

class GeminiEmbedder:
    def __init__(self, client: genai.Client):
        self.client = client
        self._calls = 0

    def encode(self, text: str) -> np.ndarray:
        resp = self.client.models.embed_content(model=EMBED_MODEL, contents=text)
        self._calls += 1
        return np.array(resp.embeddings[0].values, dtype=np.float32)


# ── Helpers ────────────────────────────────────────────────────────────────────

def hr(title: str):
    w = 64
    print(f"\n{'━' * w}")
    print(f"  {BD}{title}{R}")
    print(f"{'━' * w}\n")


def status_icon(val: float, threshold: float = 1.0) -> str:
    if val >= threshold:
        return f"{GN}✓{R}"
    if val >= 0.5:
        return f"{YL}~{R}"
    return f"{RD}✗{R}"


# ── Results Writer ─────────────────────────────────────────────────────────────

def append_to_benchmark_md(results: dict, api_calls: int, wall_time: float):
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lo = results["locomo"]
    cv = results["convomem"]
    lm = results["longmemeval"]

    chain_rows = "\n".join(
        f"| Chain {c['chain']} ({c['target']}) | {c['found_at_hop'] if c['recall'] else 'not found'} | {'✓' if c['recall'] else '✗'} |"
        for c in lo["chains"]
    )

    block = f"""
---

## Named Benchmark Suite — {stamp}

> Embeddings: `{EMBED_MODEL}` ({DIM}D)  
> Gemini API calls: {api_calls}  ·  Wall time: {wall_time:.1f}s  
> All data self-contained (no external HuggingFace datasets required)

---

### LoCoMo — Multi-Hop Relational Reasoning

**Aggregate recall**: `{lo['recall@chains']:.3f}` ({int(lo['recall@chains'] * lo['total_chains'])}/{lo['total_chains']} chains)

| Chain (target) | Found at hop | Pass |
|---|---|---|
{chain_rows}

> LoCoMo queries are deliberately semantically distant from their targets.
> Only Knowledge Graph traversal can retrieve the answer — flat vector stores
> return 0 by design on these queries.

---

### ConvoMem — Conversational Memory Recall

**recall@3**: `{cv['recall@3']:.3f}` ({cv['correct']}/{cv['total']} conversations correct)

5 multi-turn conversations ingested and flushed to Cold Tier before evaluation.
Includes preference updates and corrections (tests most-recent-fact recall).

---

### LongMemEval — Longitudinal Session Memory

**recall@3**: `{lm['recall@3']:.3f}` ({lm['correct']}/{lm['total']} QA pairs correct)

{lm['sessions']} sessions ingested with epoch checkpoints between each.
All data in Cold Tier at evaluation time. 2-hop KG expansion enabled.

---

### Summary

| Benchmark | Metric | Result |
|---|---|---|
| LoCoMo | Multi-hop recall | `{lo['recall@chains']:.3f}` |
| ConvoMem | recall@3 | `{cv['recall@3']:.3f}` |
| LongMemEval | recall@3 | `{lm['recall@3']:.3f}` |
"""

    md_path = os.path.join(_ROOT, "benchmark.md")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"\n  {CY}Results appended to benchmark.md{R}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"{RD}Error: GEMINI_API_KEY not set (checked .env and environment).{R}")
        sys.exit(1)

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print(  "║    EpochDB v0.4.0 — Named Benchmark Suite                   ║")
    print(  "║    LoCoMo  ·  ConvoMem  ·  LongMemEval                      ║")
    print(  "╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Embedding:  {BD}{EMBED_MODEL}{R} ({DIM}D)")
    print(  "  No external datasets required\n")

    client   = genai.Client(api_key=api_key)
    embedder = GeminiEmbedder(client)

    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
    db = EpochDB(storage_dir=STORAGE_DIR, dim=DIM)

    t_start  = time.perf_counter()
    results  = {}

    # ── LoCoMo ──────────────────────────────────────────────────────────────
    hr("1 / 3 — LoCoMo (Multi-Hop Relational Reasoning)")
    t0 = time.perf_counter()
    results["locomo"] = locomo.run(db, embedder)
    lo = results["locomo"]
    print(f"\n  Aggregate recall:  {BD}{lo['recall@chains']:.3f}{R}  "
          f"({int(lo['recall@chains'] * lo['total_chains'])}/{lo['total_chains']} chains)")
    for c in lo["chains"]:
        icon = f"{GN}✓{R}" if c["recall"] else f"{RD}✗{R}"
        hop_info = f"hop {c['found_at_hop']}" if c["recall"] else "not found"
        print(f"  {icon}  Chain {c['chain']} — target: '{c['target']}'  ({hop_info})")
    print(f"\n  {DM}Time: {time.perf_counter()-t0:.1f}s  ·  API calls: {embedder._calls}{R}")

    # ── ConvoMem ─────────────────────────────────────────────────────────────
    hr("2 / 3 — ConvoMem (Conversational Memory Recall)")
    t0 = time.perf_counter()
    results["convomem"] = convomem.run(db, embedder)
    cv = results["convomem"]
    print(f"\n  recall@3:  {BD}{cv['recall@3']:.3f}{R}  "
          f"({cv['correct']}/{cv['total']} correct)")
    print(f"  {DM}Time: {time.perf_counter()-t0:.1f}s  ·  API calls: {embedder._calls}{R}")

    # ── LongMemEval ───────────────────────────────────────────────────────────
    hr("3 / 3 — LongMemEval (Longitudinal Session Memory)")
    t0 = time.perf_counter()
    results["longmemeval"] = longmemeval.run(db, embedder)
    lm = results["longmemeval"]
    print(f"\n  recall@3:  {BD}{lm['recall@3']:.3f}{R}  "
          f"({lm['correct']}/{lm['total']} correct, {lm['sessions']} sessions)")
    print(f"  {DM}Time: {time.perf_counter()-t0:.1f}s  ·  API calls: {embedder._calls}{R}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    wall_time = time.perf_counter() - t_start
    hr("Summary")

    rows = [
        ("LoCoMo",      "multi-hop recall", f"{lo['recall@chains']:.3f}",  lo["recall@chains"]),
        ("ConvoMem",    "recall@3",         f"{cv['recall@3']:.3f}",       cv["recall@3"]),
        ("LongMemEval", "recall@3",         f"{lm['recall@3']:.3f}",       lm["recall@3"]),
    ]
    col = [14, 18, 8]
    print(f"  {BD}{'Benchmark':<{col[0]}} {'Metric':<{col[1]}} {'Score':<{col[2]}} Pass{R}")
    print(f"  {'─'*col[0]} {'─'*col[1]} {'─'*col[2]} {'─'*4}")
    for name, metric, score, val in rows:
        icon = status_icon(val, threshold=0.8)
        print(f"  {name:<{col[0]}} {metric:<{col[1]}} {score:<{col[2]}} {icon}")

    print(f"\n  {DM}Total API calls: {embedder._calls}  |  Wall time: {wall_time:.1f}s{R}")

    db.close()
    shutil.rmtree(STORAGE_DIR, ignore_errors=True)

    append_to_benchmark_md(results, embedder._calls, wall_time)


if __name__ == "__main__":
    main()
