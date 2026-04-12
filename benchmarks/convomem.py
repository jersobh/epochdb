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
        results = db.recall(q_emb, top_k=3, expand_hops=1, query_entities=entities)
        
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
    }
