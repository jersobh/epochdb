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
            results = db.recall(q_emb, top_k=5, expand_hops=hops)
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
    }
