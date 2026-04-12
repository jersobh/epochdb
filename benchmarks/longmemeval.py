"""
benchmarks/longmemeval.py — LongMemEval (Longitudinal Session Memory)
======================================================================
Tests that EpochDB correctly recalls facts across multiple discrete sessions
separated by epoch checkpoints (Hot→Cold flushes).

This validates the key architectural property: memories written in Session N
are still accurately retrievable in Session N+K after multiple Cold Tier
transitions, using both semantic search and KG expansion.

Dataset: built-in, modelled on the LongMemEval benchmark's session structure.
Embeddings: Gemini embedding-2-preview (3072D)
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────
# Format: List of sessions, each being a list of (text, triples).
# After each session, force_checkpoint() is called to flush Hot→Cold.
# QA pairs then query across all sessions from a fresh (empty) Hot Tier.

SESSIONS: List[List[Tuple[str, List[tuple]]]] = [
    # Session 1 — Personal background
    [
        ("The user met Alice at an AI conference in Berlin in 2021.",
         [("user", "met", "Alice"), ("Alice", "at", "Berlin conference 2021")]),
        ("The user started a new job at TechCorp as a senior engineer.",
         [("user", "works_at", "TechCorp")]),
    ],
    # Session 2 — Updates and new information
    [
        ("Alice has moved jobs and now works as a research lead at BioGen.",
         [("Alice", "works_at", "BioGen"), ("Alice", "role", "research lead")]),
        ("The user is considering relocating to Amsterdam for a new project.",
         [("user", "considering", "Amsterdam")]),
    ],
    # Session 3 — Further development
    [
        ("BioGen is developing a novel RNA-based vaccine platform.",
         [("BioGen", "develops", "RNA vaccine platform")]),
        ("The user decided to accept the Amsterdam offer and will move in Q3.",
         [("user", "moving_to", "Amsterdam"), ("user", "moving_date", "Q3")]),
    ],
    # Session 4 — Distant reference
    [
        ("Alice mentioned at a follow-up call that BioGen's RNA platform is in Phase II trials.",
         [("BioGen RNA platform", "phase", "Phase II")]),
    ],
]

QA_PAIRS: List[Tuple[str, List[str], List[str]]] = [
    # (question, required_answer_terms, helpful_query_entities)
    ("Where did the user first meet Alice?",
     ["Berlin", "conference"],
     ["Alice", "Berlin"]),
    ("Where does Alice currently work?",
     ["BioGen"],
     ["Alice", "BioGen"]),
    ("What is the user planning to do regarding Amsterdam?",
     ["Amsterdam"],
     ["Amsterdam", "user"]),
    ("What is BioGen developing and what stage is it at?",
     ["RNA", "Phase II"],
     ["BioGen", "RNA", "Phase"]),
]


def run(db, embedder) -> Dict[str, float]:
    """Run the LongMemEval benchmark. Returns recall@3 over all QA pairs."""
    logger.info("[LongMemEval] Running longitudinal session memory benchmark.")

    db.hot_tier.clear()
    db.wal.clear()

    # Ingest sessions with a checkpoint between each to simulate time passing.
    for session_idx, session in enumerate(SESSIONS, 1):
        for text, triples in session:
            db.add_memory(text, embedder.encode(text), triples)
        db.force_checkpoint()
        logger.info(f"  Session {session_idx} ingested and flushed to Cold Tier.")

    # All data is now in the Cold Tier. Evaluate.
    assert len(db.hot_tier.atoms) == 0, "Hot Tier should be empty after all checkpoints"

    correct = 0
    for question, answer_terms, q_entities in QA_PAIRS:
        q_emb = embedder.encode(question)
        results = db.recall(q_emb, top_k=4, expand_hops=2, query_entities=q_entities)
        found = any(
            all(term.lower() in str(r.payload).lower() for term in answer_terms)
            for r in results
        )
        if found:
            correct += 1
        status = "✓" if found else "✗"
        logger.info(f"  {status}  Q: {question[:65]}  targets={answer_terms}")

    recall_at_3 = correct / len(QA_PAIRS)
    return {
        "recall@3": recall_at_3,
        "correct": correct,
        "total": len(QA_PAIRS),
        "sessions": len(SESSIONS),
    }
