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

from benchmarks.extract_helpers import (
    evidence_hit,
    extract_triples,
    reset_case_state,
    surface_query_entities,
)

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


def expand_sessions(extra_sessions: int) -> List[List[Tuple[str, List[tuple]]]]:
    """Pad the longitudinal corpus with distractor sessions (huge-dataset mode)."""
    if extra_sessions <= 0:
        return list(SESSIONS)
    out = list(SESSIONS)
    cities = [
        "Lisbon", "Oslo", "Seoul", "Toronto", "Nairobi", "Santiago", "Helsinki", "Dubai",
    ]
    orgs = [
        "NovaLabs", "QuarkForge", "CedarSystems", "PulseAI", "Northwind", "HelixWorks",
    ]
    for i in range(extra_sessions):
        city = cities[i % len(cities)]
        org = orgs[i % len(orgs)]
        person = f"Contact{i}"
        out.append(
            [
                (
                    f"{person} joined {org} and briefly visited {city} for an offsite.",
                    [(person, "works_at", org), (person, "visited", city)],
                ),
                (
                    f"{org} filed progress report #{i} covering routine platform maintenance.",
                    [(org, "report_id", str(i))],
                ),
            ]
        )
    return out


def run(db, embedder, extra_sessions: int = 0) -> Dict[str, float]:
    """Run the LongMemEval benchmark. Returns recall@3 over all QA pairs.

    ``extra_sessions`` adds distractor sessions flushed to Cold Tier (huge mode).
    Core QA pairs stay the same so recall remains comparable.
    """
    sessions = expand_sessions(extra_sessions)
    logger.info(
        "[LongMemEval] Running longitudinal session memory benchmark "
        f"({len(sessions)} sessions)."
    )

    db.hot_tier.clear()
    db.wal.clear()

    # Ingest sessions with a checkpoint between each to simulate time passing.
    for session_idx, session in enumerate(sessions, 1):
        for text, triples in session:
            db.add_memory(text, embedder.encode(text), triples)
        db.force_checkpoint()
        if session_idx <= 4 or session_idx % 25 == 0 or session_idx == len(sessions):
            logger.info(f"  Session {session_idx}/{len(sessions)} ingested and flushed.")

    # All data is now in the Cold Tier. Evaluate.
    assert len(db.hot_tier.atoms) == 0, "Hot Tier should be empty after all checkpoints"

    correct = 0
    for question, answer_terms, q_entities in QA_PAIRS:
        q_emb = embedder.encode(question)
        results = db.recall(
            q_emb, top_k=4, expand_hops=2, query_entities=q_entities, query_text=question
        )
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
        "sessions": len(sessions),
        "source": "synthetic",
    }


def _longmem_hit(answer: str, evidence_texts: List[str], payloads: List[str]) -> bool:
    return evidence_hit(answer, evidence_texts, payloads)


def run_official(
    db,
    embedder,
    *,
    variant: str = "s",
    limit: int | None = 50,
    seed: int | None = 42,
) -> Dict[str, float]:
    """Run official LongMemEval (cleaned). Default: S, first ``limit`` questions.

    Metric: retrieval recall@10 — whether answer/evidence appears in top-k atoms
    after ingesting that question's haystack (not LLM-judged QA accuracy).
    """
    from benchmarks.official_data import (
        LONGMEM_M,
        LONGMEM_S,
        download_longmemeval,
        iter_longmemeval_cases,
    )

    path = download_longmemeval(variant)
    logger.info(
        "[LongMemEval] Official %s dataset: %s (limit=%s, seed=%s)",
        variant.upper(),
        path,
        limit,
        seed,
    )

    # Small returned k; engine retrieves a wide pool then precision-reranks.
    top_k = 10
    expand_hops = 2
    window = 2
    correct = 0
    total = 0
    total_sessions = 0
    for case in iter_longmemeval_cases(path, limit=limit, seed=seed):
        reset_case_state(db)

        # Checkpoint mid-stream to bound Hot Tier; leave the tail warm for recall.
        turns = case["turns"]
        for i, turn in enumerate(turns, 1):
            chunk = " ".join(turns[max(0, i - window) : i])
            triples = extract_triples(db, chunk)
            db.add_memory(chunk, embedder.encode(chunk), triples=triples)
            if i % 40 == 0 and i < len(turns):
                db.force_checkpoint()
        total_sessions += case["n_sessions"]

        entities = []
        if hasattr(db, "extract_entities"):
            try:
                entities = list(db.extract_entities(case["question"]) or [])
            except Exception:
                entities = []
        for e in surface_query_entities(case["question"]):
            if e not in entities:
                entities.append(e)
        q_emb = embedder.encode(case["question"])
        results = db.recall(
            q_emb,
            top_k=top_k,
            expand_hops=expand_hops,
            query_entities=entities or None,
            query_text=case["question"],
            cold_search_mode="exhaustive",
        )
        payloads = [str(r.payload) for r in results]
        ok = evidence_hit(case["answer"], case["evidence_texts"], payloads)
        correct += int(ok)
        total += 1
        status = "✓" if ok else "✗"
        logger.info(
            "  %s [%d] %s (%s, %d sessions, %d turns)",
            status,
            total,
            (case["question"] or "")[:55],
            case.get("question_type"),
            case["n_sessions"],
            len(case["turns"]),
        )

    return {
        "recall@3": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "sessions": total_sessions,
        "top_k": top_k,
        "source": f"official:longmemeval_{variant}_cleaned",
        "variant": variant,
        "path": str(path if variant == "s" else LONGMEM_M),
    }
