"""
benchmarks/run_all.py — EpochDB v0.4.1 Named Benchmark Suite
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
import asyncio
try:
    from google import genai
except ImportError:
    print("Error: google-genai is not installed. Run: pip install google-genai")
    sys.exit(1)

from epochdb import EpochDB, AsyncEpochDB
from benchmarks import locomo, convomem, longmemeval, needle


class AsyncEpochDBAdapter:
    def __init__(self, async_db: AsyncEpochDB, loop: asyncio.AbstractEventLoop):
        self.async_db = async_db
        self.loop = loop

    @property
    def hot_tier(self):
        return self.async_db._db.hot_tier

    @property
    def wal(self):
        return self.async_db._db.wal

    @property
    def kg_manager(self):
        return self.async_db._db.kg_manager

    @property
    def extraction_model(self):
        return self.async_db._db.extraction_model

    def _get_fact_extractor(self):
        return self.async_db._db._get_fact_extractor()

    def add_memory(self, payload, embedding, triples=None, metadata=None):
        return self.loop.run_until_complete(
            asyncio.to_thread(
                self.async_db._db.add_memory, payload, embedding, triples, None, metadata
            )
        )

    def recall(
        self,
        query_emb,
        top_k=5,
        expand_hops=1,
        query_entities=None,
        query_text=None,
        cold_search_mode=None,
        **kwargs,
    ):
        return self.loop.run_until_complete(
            asyncio.to_thread(
                self.async_db._db.recall,
                query_emb,
                top_k,
                expand_hops,
                query_entities,
                None,
                None,
                cold_search_mode or kwargs.get("cold_search_mode"),
                query_text,
            )
        )

    def force_checkpoint(self):
        return self.loop.run_until_complete(
            asyncio.to_thread(self.async_db._db.force_checkpoint)
        )

    def extract_entities(self, text):
        return self.loop.run_until_complete(
            asyncio.to_thread(self.async_db._db.extract_entities, text)
        )


class AsyncDbWrapper:
    def __init__(self, storage_dir, dim, extraction_model=None):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        kwargs = {"storage_dir": storage_dir, "dim": dim}
        if extraction_model is not None:
            kwargs["extraction_model"] = extraction_model
        self.async_db = AsyncEpochDB(**kwargs)
        self.db = self.loop.run_until_complete(self.async_db.__aenter__())
        self.adapter = AsyncEpochDBAdapter(self.db, self.loop)

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def close(self):
        if self.loop is not None and not self.loop.is_closed():
            try:
                self.loop.run_until_complete(self.async_db.__aexit__(None, None, None))
            except Exception:
                pass
            try:
                self.loop.close()
            except Exception:
                pass
            self.loop = None


def init_db(storage_dir: str, dim: int, async_mode: bool, extraction_model: str = "local"):
    if async_mode:
        return AsyncDbWrapper(storage_dir, dim, extraction_model=extraction_model)
    else:
        return EpochDB(
            storage_dir=storage_dir,
            dim=dim,
            extraction_model=extraction_model,
        )

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_MODEL = "jersobh/epochdb-embed"
DIM         = 384
STORAGE_DIR = "./.epochdb_run_all"

R  = "\033[0m"
BD = "\033[1m"
GN = "\033[92m"
YL = "\033[93m"
RD = "\033[91m"
CY = "\033[96m"
DM = "\033[2m"


# ── Local Embedder ────────────────────────────────────────────────────────────

class LocalEmbedder:
    def __init__(self, client=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBED_MODEL)
        self._calls = 0

    def encode(self, text: str) -> np.ndarray:
        self._calls += 1
        return self.model.encode(text, normalize_embeddings=True).astype(np.float32)


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

def append_to_benchmark_md(
    results: dict,
    api_calls: int,
    wall_time: float,
    *,
    extraction_model: str = "local",
    seed: int = 42,
):
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lo = results["locomo"]
    cv = results["convomem"]
    lm = results["longmemeval"]
    nd = results["needle"]

    chain_rows = "\n".join(
        f"| Chain {c['chain']} ({c['target']}) | {c['found_at_hop'] if c.get('found_at_hop') is not None else 'n/a'} | {'✓' if c['recall'] else '✗'} |"
        for c in lo["chains"]
    )
    lo_k = lo.get("top_k", 5)
    cv_k = cv.get("top_k", 3)
    lm_k = lm.get("top_k", 5)

    block = f"""
---

## Named Benchmark Suite — {stamp}

> Embeddings: `{EMBED_MODEL}` ({DIM}D)  
> Extraction: `{extraction_model}`  ·  Seed: `{seed}`  
> Gemini API calls: {api_calls}  ·  Wall time: {wall_time:.1f}s  
> Data: `{results['locomo'].get('source','synthetic')}` / `{results['convomem'].get('source','synthetic')}` / `{results['longmemeval'].get('source','synthetic')}`

---

### LoCoMo — Multi-Hop Relational Reasoning

**Aggregate recall**: `{lo['recall@chains']:.3f}` ({lo.get('correct', int(lo['recall@chains'] * lo['total_chains']))}/{lo.get('total', lo['total_chains'])} QA · {lo['total_chains']} conversations · recall@{lo_k})

| Chain (target) | Found at hop | Pass |
|---|---|---|
{chain_rows}

> Official LoCoMo uses evidence-aware retrieval recall (answer or evidence turns in top-k).

---

### ConvoMem — Conversational Memory Recall

**recall@{cv_k}**: `{cv['recall@3']:.3f}` ({cv['correct']}/{cv['total']} conversations correct)

Conversations ingested with local/heuristic KG triples; keyword+semantic recall.

---

### LongMemEval — Longitudinal Session Memory

**recall@{lm_k}**: `{lm['recall@3']:.3f}` ({lm['correct']}/{lm['total']} QA pairs correct)

{lm['sessions']} sessions ingested with epoch checkpoints between each.
Evidence-aware retrieval with KG expansion.

---

### Needle in a Haystack — Retrieval Precision

**precision@3**: `{nd['precision_at_3']:.3f}` ({nd['signal_hits']}/3 results are signal)

3 signal facts hidden among **{nd['total_noise']}** noise facts.  
Evaluation uses Entity Hook seeding to ensure Topic Lock.

---

### Summary

| Benchmark | Metric | Result |
|---|---|---|
| LoCoMo | multi-hop recall@{lo_k} | `{lo['recall@chains']:.3f}` |
| ConvoMem | recall@{cv_k} | `{cv['recall@3']:.3f}` |
| LongMemEval | recall@{lm_k} | `{lm['recall@3']:.3f}` |
| NIAH | precision@3 | `{nd['precision_at_3']:.3f}` |
"""

    md_path = os.path.join(_ROOT, "benchmark.md")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"\n  {CY}Results appended to benchmark.md{R}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run EpochDB benchmarks")
    parser.add_argument("--async-mode", "-a", action="store_true", help="Run benchmarks in async mode using AsyncEpochDB facade")
    parser.add_argument(
        "--huge",
        action="store_true",
        help="Huge dataset mode: ~10k NIAH noise facts + 500 LongMemEval sessions",
    )
    parser.add_argument(
        "--niah-noise",
        type=int,
        default=None,
        help="Override NIAH haystack size (implies expanded noise corpus)",
    )
    parser.add_argument(
        "--longmem-extra-sessions",
        type=int,
        default=None,
        help="Extra LongMemEval distractor sessions (core QA unchanged)",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="Use official LoCoMo / ConvoMem / LongMemEval corpora from benchmarks/data/",
    )
    parser.add_argument(
        "--longmem-variant",
        choices=["s", "m"],
        default="s",
        help="Official LongMemEval variant (s≈115k tok, m≈500 sessions). Default: s",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap official LongMemEval questions / ConvoMem cases (default: 50 / 200)",
    )
    parser.add_argument(
        "--extraction-model",
        default="local",
        help="Triple extraction backend: local | hf:... | google:... | openai:... "
        "(default: local; overrides EPOCHDB_EXTRACTION_MODEL)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle official case/QA order with this seed (default: 42)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"{RD}Error: GEMINI_API_KEY not set (checked .env and environment).{R}")
        sys.exit(1)

    extraction_model = args.extraction_model
    niah_noise = args.niah_noise
    longmem_extra = args.longmem_extra_sessions
    if args.huge and not args.official:
        if niah_noise is None:
            niah_noise = 10_000
        if longmem_extra is None:
            # 4 core sessions + 496 distractors ≈ LongMemEval_M scale (~500)
            longmem_extra = 496

    mode_str = "Async" if args.async_mode else "Sync"
    if args.official:
        scale_str = "OFFL"
    elif args.huge or niah_noise or longmem_extra:
        scale_str = "HUGE"
    else:
        scale_str = "std"
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║    EpochDB — Named Benchmark Suite ({mode_str:5} / {scale_str:4})         ║")
    print(  "║    LoCoMo · ConvoMem · LongMemEval · NIAH                   ║")
    print(  "╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Embedding:  {BD}{EMBED_MODEL}{R} ({DIM}D)")
    print(f"  Extraction: {BD}{extraction_model}{R}")
    print(f"  Seed:       {BD}{args.seed}{R}")
    if args.official:
        lm_limit = args.limit if args.limit is not None else 50
        cv_limit = args.limit if args.limit is not None else 200
        print(
            f"  Official:   LoCoMo-10 · ConvoMem(limit={cv_limit}) · "
            f"LongMemEval_{args.longmem_variant.upper()}(limit={lm_limit}) · NIAH"
        )
        print(
            "  Metric note: retrieval recall (answer/evidence in top-k), "
            "not LLM-judged QA accuracy\n"
        )
    else:
        if niah_noise or longmem_extra:
            print(
                f"  Scale:      NIAH noise={niah_noise or 'default'}  ·  "
                f"LongMemEval extra sessions={longmem_extra or 0}"
            )
        print(  "  4 benchmarks · synthetic built-ins (use --official for real corpora)\n")

    client   = genai.Client(api_key=api_key)
    embedder = LocalEmbedder(client)

    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
    db = init_db(STORAGE_DIR, DIM, args.async_mode, extraction_model=extraction_model)
    db.close()

    t_start  = time.perf_counter()
    results  = {}

    lm_limit = args.limit if args.limit is not None else 50
    cv_limit = args.limit if args.limit is not None else 200

    # ── LoCoMo ──────────────────────────────────────────────────────────────
    hr("1 / 3 — LoCoMo (Multi-Hop Relational Reasoning)")
    if os.path.exists(STORAGE_DIR): shutil.rmtree(STORAGE_DIR)
    db = init_db(STORAGE_DIR, DIM, args.async_mode, extraction_model=extraction_model)
    t0 = time.perf_counter()
    if args.official:
        results["locomo"] = locomo.run_official(db, embedder, seed=args.seed)
    else:
        results["locomo"] = locomo.run(db, embedder)
    lo = results["locomo"]
    qa_bit = ""
    if "total" in lo:
        qa_bit = f", {lo.get('correct', 0)}/{lo['total']} QA"
    print(
        f"\n  Aggregate recall:  {BD}{lo['recall@chains']:.3f}{R}  "
        f"({int(lo['recall@chains'] * lo['total_chains'])}/{lo['total_chains']} chains"
        f"{qa_bit})  [{lo.get('source', 'synthetic')}]"
    )
    for c in lo["chains"][:12]:
        icon = f"{GN}✓{R}" if c["recall"] else f"{RD}✗{R}"
        if args.official:
            print(f"  {icon}  {c['chain']} — {c['target']}")
        else:
            hop_info = f"hop {c['found_at_hop']}" if c["recall"] else "not found"
            print(f"  {icon}  Chain {c['chain']} — target: '{c['target']}'  ({hop_info})")
    print(f"\n  {DM}Time: {time.perf_counter()-t0:.1f}s  ·  API calls: {embedder._calls}{R}")
    db.close()

    # ── ConvoMem ─────────────────────────────────────────────────────────────
    hr("2 / 3 — ConvoMem (Conversational Memory Recall)")
    if os.path.exists(STORAGE_DIR): shutil.rmtree(STORAGE_DIR)
    db = init_db(STORAGE_DIR, DIM, args.async_mode, extraction_model=extraction_model)
    t0 = time.perf_counter()
    if args.official:
        results["convomem"] = convomem.run_official(
            db, embedder, limit=cv_limit, seed=args.seed
        )
    else:
        results["convomem"] = convomem.run(db, embedder)
    cv = results["convomem"]
    print(f"\n  recall@3:  {BD}{cv['recall@3']:.3f}{R}  "
          f"({cv['correct']}/{cv['total']} correct)  [{cv.get('source', 'synthetic')}]")
    print(f"  {DM}Time: {time.perf_counter()-t0:.1f}s  ·  API calls: {embedder._calls}{R}")
    db.close()

    # ── LongMemEval ───────────────────────────────────────────────────────────
    hr("3 / 3 — LongMemEval (Longitudinal Session Memory)")
    if os.path.exists(STORAGE_DIR): shutil.rmtree(STORAGE_DIR)
    db = init_db(STORAGE_DIR, DIM, args.async_mode, extraction_model=extraction_model)
    t0 = time.perf_counter()
    if args.official:
        results["longmemeval"] = longmemeval.run_official(
            db,
            embedder,
            variant=args.longmem_variant,
            limit=lm_limit,
            seed=args.seed,
        )
    else:
        results["longmemeval"] = longmemeval.run(
            db, embedder, extra_sessions=longmem_extra or 0
        )
    lm = results["longmemeval"]
    print(f"\n  recall@3:  {BD}{lm['recall@3']:.3f}{R}  "
          f"({lm['correct']}/{lm['total']} correct, {lm['sessions']} sessions)"
          f"  [{lm.get('source', 'synthetic')}]")
    print(f"  {DM}Time: {time.perf_counter()-t0:.1f}s  ·  API calls: {embedder._calls}{R}")
    db.close()

    # ── NIAH ──────────────────────────────────────────────────────────────────
    hr("4 / 4 — Needle in a Haystack (Retrieval Precision)")
    if os.path.exists(STORAGE_DIR): shutil.rmtree(STORAGE_DIR)
    db = init_db(STORAGE_DIR, DIM, args.async_mode, extraction_model=extraction_model)
    t0 = time.perf_counter()
    results["needle"] = needle.run(db, embedder, noise_count=niah_noise)
    nd = results["needle"]
    print(f"\n  precision@3:  {BD}{nd['precision_at_3']:.3f}{R}  "
          f"({nd['signal_hits']}/{nd['total_signal']} signal hits, "
          f"{nd['total_noise']} noise)")
    print(f"  {DM}Time: {time.perf_counter()-t0:.1f}s  ·  API calls: {embedder._calls}{R}")
    db.close()

    # ── Final Summary ─────────────────────────────────────────────────────────
    wall_time = time.perf_counter() - t_start
    hr("Summary")

    rows = [
        ("LoCoMo",      "multi-hop recall", f"{lo['recall@chains']:.3f}",  lo["recall@chains"]),
        ("ConvoMem",    "recall@3",         f"{cv['recall@3']:.3f}",       cv["recall@3"]),
        ("LongMemEval", "recall@3",         f"{lm['recall@3']:.3f}",       lm["recall@3"]),
        ("NIAH",        "precision@3",      f"{nd['precision_at_3']:.3f}", nd["precision_at_3"]),
    ]
    col = [14, 18, 8]
    print(f"  {BD}{'Benchmark':<{col[0]}} {'Metric':<{col[1]}} {'Score':<{col[2]}} Pass{R}")
    print(f"  {'─'*col[0]} {'─'*col[1]} {'─'*col[2]} {'─'*4}")
    for name, metric, score, val in rows:
        icon = status_icon(val, threshold=0.9)
        print(f"  {name:<{col[0]}} {metric:<{col[1]}} {score:<{col[2]}} {icon}")

    print(f"\n  {DM}Total API calls: {embedder._calls}  |  Wall time: {wall_time:.1f}s{R}")
    print(f"  {DM}Extraction: {extraction_model}{R}")

    db.close()
    shutil.rmtree(STORAGE_DIR, ignore_errors=True)

    append_to_benchmark_md(
        results,
        embedder._calls,
        wall_time,
        extraction_model=extraction_model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
