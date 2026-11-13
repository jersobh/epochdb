"""
benchmarks/locomo.py — LoCoMo (Longitudinal Conversational Memory)
===================================================================
Tests multi-hop relational reasoning across long conversational chains.

LoCoMo is the key EpochDB differentiator: queries that require bridging
multiple disconnected facts via the Knowledge Graph — a task structurally
impossible for flat vector stores.

Dataset: built-in, modelled on the LoCoMo benchmark's relational chain structure.
Embeddings: Gemini embedding-2-preview (3072D)
"""

import time
import logging
from typing import Dict, List, Tuple

from benchmarks.extract_helpers import (
    evidence_hit,
    extract_triples,
    reset_case_state,
    surface_query_entities,
)

logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────
# Each chain tests a different multi-hop depth.
# Chain A: 3-hop  (Sam Altman → OpenAI → Helion)
# Chain B: 4-hop  (Alice → Team Aurora → Project Helios → Quantum Core → Dr. Chen)
# Chain C: 2-hop  (Marie → BioGen → CRISPR-X)

CHAINS: List[Tuple[List[Tuple[str, List[tuple]]], str, str, int]] = [
    # (facts_with_triples, query, target_substring, expected_min_hops)
    (
        [
            ("The CEO of OpenAI is Sam Altman.",
             [("OpenAI", "has_CEO", "Sam Altman")]),
            ("Sam Altman made a significant personal investment in a fusion energy company.",
             [("Sam Altman", "invested_in", "fusion_company")]),
            ("The fusion energy company backed by Sam Altman is called Helion Energy.",
             [("fusion_company", "is_named", "Helion Energy")]),
        ],
        # Query has near-zero similarity to "Helion Energy" — KG must bridge it.
        "Name the clean energy firm the OpenAI chief executive has backed financially.",
        "Helion",
        2,
    ),
    (
        [
            ("Alice is the Director of Team Aurora.",
             [("Alice", "leads", "Team Aurora")]),
            ("Team Aurora is developing Project Helios.",
             [("Team Aurora", "develops", "Project Helios")]),
            ("Project Helios relies on the Quantum Core module.",
             [("Project Helios", "uses", "Quantum Core")]),
            ("The Quantum Core was engineered by Dr. Chen.",
             [("Quantum Core", "engineered_by", "Dr. Chen")]),
        ],
        # Query mentions neither "Dr. Chen" nor "Quantum Core".
        "Who built the technology that Alice's team's flagship project depends on?",
        "Dr. Chen",
        3,
    ),
    (
        [
            ("Marie is the head of research at BioGen.",
             [("Marie", "leads", "BioGen")]),
            ("BioGen is developing a gene-editing platform called CRISPR-X.",
             [("BioGen", "develops", "CRISPR-X")]),
        ],
        # Query mentions neither "BioGen" nor "CRISPR-X".
        "What gene-editing platform is Marie's organisation building?",
        "CRISPR-X",
        1,
    ),
]


def run(db, embedder) -> Dict[str, float]:
    """Run the LoCoMo benchmark. Returns per-chain recall and aggregate."""
    logger.info("[LoCoMo] Running multi-hop relational reasoning benchmark.")

    chain_results = []
    total_chains = len(CHAINS)

    for idx, (facts, query, target, min_hops) in enumerate(CHAINS, 1):
        db.hot_tier.clear()
        db.wal.clear()

        # Ingest.
        for text, triples in facts:
            db.add_memory(text, embedder.encode(text), triples)

        # Evaluate at increasing hop depths.
        q_emb = embedder.encode(query)
        found_at = None
        for hops in range(6):
            results = db.recall(q_emb, top_k=5, expand_hops=hops, query_text=query)
            if any(target.lower() in str(r.payload).lower() for r in results):
                found_at = hops
                break

        recall = 1.0 if found_at is not None else 0.0
        chain_results.append({
            "chain": idx,
            "target": target,
            "found_at_hop": found_at,
            "expected_min_hops": min_hops,
            "recall": recall,
        })

        status = f"✓  found at hop {found_at}" if recall else "✗  not found"
        logger.info(f"  Chain {idx}: target='{target}'  {status}")

    aggregate_recall = sum(r["recall"] for r in chain_results) / total_chains
    return {
        "recall@chains": aggregate_recall,
        "chains": chain_results,
        "total_chains": total_chains,
        "source": "synthetic",
    }


def _answer_match(answer, payloads: List[str]) -> bool:
    """True if the LoCoMo answer (str/int) is evidenced in retrieved payloads."""
    return evidence_hit(answer, None, payloads)


def run_official(
    db, embedder, *, limit_qa: int | None = None, seed: int | None = 42
) -> Dict[str, float]:
    """Run official LoCoMo-10: ingest each conversation, score QA via recall@20."""
    from benchmarks.official_data import LOCOMO_PATH, download_locomo, iter_locomo_cases

    path = download_locomo()
    logger.info("[LoCoMo] Official dataset: %s (seed=%s)", path, seed)

    chain_results = []
    total_correct = 0
    total_qa = 0
    # Precision-first: return ≤10 atoms to the judge / LLM context.
    top_k = 10
    expand_hops = 3

    for sample_id, turns, qa in iter_locomo_cases(path, seed=seed):
        reset_case_state(db)

        texts = [turn["text"] for turn in turns]
        for j, turn in enumerate(turns):
            # Pair adjacent dialogue turns — better multi-hop evidence packing.
            text = f"{texts[j - 1]} {texts[j]}" if j > 0 else texts[j]
            triples = extract_triples(db, text)
            meta = {"dia_id": turn.get("dia_id")} if turn.get("dia_id") else None
            db.add_memory(text, embedder.encode(text), triples, metadata=meta)

        sample_correct = 0
        sample_total = 0
        for qi, q in enumerate(qa):
            if limit_qa is not None and total_qa >= limit_qa:
                break
            question = q["question"]
            answer = q["answer"]
            evidence_texts = q.get("evidence_texts") or []
            entities = []
            if hasattr(db, "extract_entities"):
                try:
                    entities = list(db.extract_entities(question) or [])
                except Exception:
                    entities = []
            for e in surface_query_entities(question):
                if e not in entities:
                    entities.append(e)
            q_emb = embedder.encode(question)
            results = db.recall(
                q_emb,
                top_k=top_k,
                expand_hops=expand_hops,
                query_entities=entities or None,
                query_text=question,
                cold_search_mode="exhaustive",
            )
            payloads = [str(r.payload) for r in results]
            ok = evidence_hit(answer, evidence_texts, payloads)
            sample_correct += int(ok)
            sample_total += 1
            total_correct += int(ok)
            total_qa += 1

        recall = sample_correct / sample_total if sample_total else 0.0
        chain_results.append(
            {
                "chain": sample_id,
                "target": f"{sample_correct}/{sample_total}",
                "found_at_hop": None,
                "expected_min_hops": 0,
                "recall": recall,
                "correct": sample_correct,
                "total": sample_total,
            }
        )
        logger.info(
            "  %s: recall@%d=%.3f (%d/%d)",
            sample_id,
            top_k,
            recall,
            sample_correct,
            sample_total,
        )
        if limit_qa is not None and total_qa >= limit_qa:
            break

    aggregate = total_correct / total_qa if total_qa else 0.0
    return {
        "recall@chains": aggregate,
        "chains": chain_results,
        "total_chains": len(chain_results),
        "correct": total_correct,
        "total": total_qa,
        "top_k": top_k,
        "source": "official:locomo10",
    }
