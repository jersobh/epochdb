"""
benchmarks/convomem.py — ConvoMem (Conversational Memory Recall)
================================================================
Tests verbatim recall of specific facts stated in multi-turn conversations,
including preference updates and corrections.

Each conversation turn is stored as its own atom so that the most-recent
value of a fact has a fresher `created_at` timestamp and wins in the RRF
recency ranking — correctly modelling how an agent should track state.

Dataset: built-in, modelled on the ConvoMem benchmark's QA structure.
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
# Format: (turns: List[(text, triples)], question: str, expected_answer_terms: List[str])
# Each turn is stored as a SEPARATE atom so recency ranking resolves corrections.

CONVERSATIONS: List[Tuple[List[Tuple[str, List[tuple]]], str, List[str]]] = [
    (
        [
            ("User said they were born in Lisbon, Portugal.",
             [("user", "born_in", "Lisbon")]),
            ("User corrected themselves: they were actually born in Porto, not Lisbon.",
             [("user", "born_in", "Porto")]),
        ],
        "Where was the user born?",
        ["Porto"],
    ),
    (
        [
            ("User mentioned their pet is a dog named Max.",
             [("Max", "is_a", "dog")]),
            ("User clarified: Max is actually a cat, not a dog.",
             [("Max", "is_a", "cat")]),
        ],
        "What type of pet is Max?",
        ["cat"],
    ),
    (
        [
            ("User works as a software engineer at DataFlow Inc.",
             [("user", "works_at", "DataFlow Inc")]),
            ("User announced they accepted a new role as senior ML engineer at VectorAI.",
             [("user", "works_at", "VectorAI"), ("user", "role", "senior ML engineer")]),
        ],
        "Where does the user currently work?",
        ["VectorAI"],
    ),
    (
        [
            ("User's favourite programming language is Python.",
             [("user", "favourite_language", "Python")]),
        ],
        "What is the user's favourite programming language?",
        ["Python"],
    ),
    (
        [
            ("User is planning to visit Japan next month.",
             [("user", "visiting", "Japan")]),
            ("User confirmed their Japan itinerary: Kyoto and Osaka, possibly adding Tokyo.",
             [("user", "visiting_cities", "Kyoto"), ("user", "visiting_cities", "Osaka")]),
        ],
        "Which cities is the user planning to visit in Japan?",
        ["Kyoto", "Osaka"],
    ),
]


def run(db, embedder) -> Dict[str, float]:
    """Run the ConvoMem benchmark. Returns recall@3 over all QA pairs."""
    logger.info("[ConvoMem] Running conversational memory recall benchmark.")

    db.hot_tier.clear()
    db.wal.clear()

    # Ingest each turn as its own atom (critical: preserves recency ordering).
    qa_cases: List[Tuple[str, List[str]]] = []
    for turns, question, answer_terms in CONVERSATIONS:
        for text, triples in turns:
            db.add_memory(text, embedder.encode(text), triples=triples)
        qa_cases.append((question, answer_terms))

    # Flush to Cold Tier to test cross-epoch recall too.
    db.force_checkpoint()

    correct = 0
    for question, answer_terms in qa_cases:
        # Manually extract relevant KG entities from the query string
        entities = db.extract_entities(question)
        q_emb = embedder.encode(question)
        
        # Use low-level recall but with high-level entity context boost
        results = db.recall(
            q_emb, top_k=3, expand_hops=1, query_entities=entities, query_text=question
        )

        found = any(
            all(term.lower() in str(r.payload).lower() for term in answer_terms)
            for r in results
        )
        if found:
            correct += 1
        status = "✓" if found else "✗"
        logger.info(f"  {status}  Q: {question[:60]}  targets={answer_terms}")

    recall_at_3 = correct / len(qa_cases)
    return {
        "recall@3": recall_at_3,
        "correct": correct,
        "total": len(qa_cases),
        "source": "synthetic",
    }


def _answer_in_results(answer: str, payloads: List[str]) -> bool:
    return evidence_hit(answer, None, payloads)


def run_official(
    db, embedder, *, limit: int | None = 200, seed: int | None = 42
) -> Dict[str, float]:
    """Run official Salesforce ConvoMem slice (retrieval recall@10)."""
    from benchmarks.official_data import download_convomem_sample, iter_convomem_cases

    root = download_convomem_sample()
    logger.info(
        "[ConvoMem] Official dataset under %s (limit=%s, seed=%s)", root, limit, seed
    )

    cases = list(iter_convomem_cases(root, limit=limit, seed=seed))
    if not cases:
        raise RuntimeError(
            "No ConvoMem cases parsed — check benchmarks/data/convomem download."
        )

    # Precision-first: return k=5. Engine: multi-query BM25 + precision rerank
    # + BM25 elite seats. Dual ingest (turn + bigram window) improves both
    # exact BM25 hits and local dialogue context without raising returned k.
    top_k = 5
    expand_hops = 2
    correct = 0
    for i, case in enumerate(cases, 1):
        reset_case_state(db)
        turns = case["turns"]
        for j, turn in enumerate(turns):
            triples = extract_triples(db, turn)
            db.add_memory(turn, embedder.encode(turn), triples=triples)
            if j > 0:
                window = f"{turns[j - 1]} {turn}"
                db.add_memory(
                    window, embedder.encode(window), triples=extract_triples(db, window)
                )

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
        ok = evidence_hit(case["answer"], case.get("evidence_texts"), payloads)
        correct += int(ok)
        if i <= 20 or i % 50 == 0 or i == len(cases):
            status = "✓" if ok else "✗"
            logger.info(
                "  %s [%d/%d] %s → %s",
                status,
                i,
                len(cases),
                case["question"][:55],
                str(case["answer"])[:40],
            )

    return {
        "recall@3": correct / len(cases),
        "correct": correct,
        "total": len(cases),
        "top_k": top_k,
        "source": "official:Salesforce/ConvoMem",
    }
