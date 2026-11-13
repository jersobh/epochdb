"""
Precision reranker for memory recall.

Design goals
------------
* Return a *small* top-k (token-cheap for LLM context).
* Rank so the evidence atom is in that small set — not by dumping k=100.
* Fuse dense similarity, BM25, entity overlap, and lexical/phrase signals
  into one calibrated score without a heavy cross-encoder dependency.

Stage model
-----------
1. Candidate generation (wide pool) happens in RetrievalManager.search.
2. This module re-scores those candidates for *precision* against query_text.
3. Only the top ``top_k`` survive to the caller.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple

from epochdb.retrieval.keyword_index import tokenize as _kw_tokenize

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?", re.I)
_TIME_RE = re.compile(
    r"\b(?:[01]?\d|2[0-3])(?::[0-5]\d)?\s*(?:am|pm)\b|\b(?:[01]?\d|2[0-3]):[0-5]\d\b",
    re.I,
)
_NUM_RE = re.compile(r"\b\d+(?:[.:]\d+)?\b")
# Content words that rarely distinguish evidence on their own.
_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "my",
        "me",
        "i",
        "you",
        "your",
        "we",
        "our",
        "they",
        "them",
        "their",
        "with",
        "from",
        "as",
        "at",
        "by",
        "about",
        "into",
        "over",
        "after",
        "before",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "how",
        "why",
        "did",
        "do",
        "does",
        "have",
        "has",
        "had",
        "can",
        "could",
        "would",
        "should",
        "will",
        "just",
        "also",
        "very",
        "really",
        "said",
        "mention",
        "mentioned",
        "tell",
        "told",
        "remind",
        "remember",
        "please",
        "hey",
        "hi",
        "user",
        "assistant",
    }
)


def tokenize(text: str) -> List[str]:
    # Share stemming with BM25 so Monday/Mondays align.
    return list(_kw_tokenize(text or ""))


def content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in _STOP and len(t) > 2]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    if not inter:
        return 0.0
    return inter / len(sa | sb)


def _coverage(query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
    if not query_tokens:
        return 0.0
    ds = set(doc_tokens)
    return sum(1 for t in query_tokens if t in ds) / len(query_tokens)


def _phrase_hits(query: str, doc: str) -> float:
    """Reward distinctive bigrams/trigrams from the query found in the doc."""
    q = content_tokens(query)
    if len(q) < 2:
        return 0.0
    d = " ".join(tokenize(doc))
    hits = 0
    total = 0
    for n in (3, 2):
        for i in range(len(q) - n + 1):
            gram = " ".join(q[i : i + n])
            total += 1
            if gram in d:
                hits += 1
    return hits / total if total else 0.0


def _time_number_alignment(query: str, doc: str) -> float:
    """Times and numbers in the query that also appear in the document."""
    q_times = {m.group(0).lower().replace(" ", "") for m in _TIME_RE.finditer(query)}
    d_times = {m.group(0).lower().replace(" ", "") for m in _TIME_RE.finditer(doc)}
    q_nums = set(_NUM_RE.findall(query))
    d_nums = set(_NUM_RE.findall(doc))
    score = 0.0
    if q_times:
        score += 0.6 * (len(q_times & d_times) / len(q_times))
    if q_nums:
        score += 0.4 * (len(q_nums & d_nums) / len(q_nums))
    return score


def _proper_noun_overlap(query: str, doc: str) -> float:
    """Capitalized / brand-like tokens from the query present in the doc."""
    props = []
    for raw in re.findall(r"\b[A-Z][A-Za-z0-9+'’.-]{1,}\b", query or ""):
        w = raw.lower().strip(".'")
        if w not in _STOP and len(w) > 2:
            props.append(w)
    if not props:
        return 0.0
    d = set(tokenize(doc))
    return sum(1 for p in props if p in d) / len(props)


def _sentence_max_lexical(query: str, document: str) -> float:
    """Score each sentence / dialogue turn; take the max (anti-dilution)."""
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?=\b(?:User|Assistant)\s*:)", document or "")
    parts = [p.strip() for p in parts if p and len(p.strip()) > 5]
    if not parts:
        parts = [document or ""]
    return max(_lexical_once(query, p) for p in parts)


def _lexical_once(query: str, document: str) -> float:
    q_tok = content_tokens(query)
    d_tok = content_tokens(document)
    cov = _coverage(q_tok, d_tok)
    jac = _jaccard(q_tok, d_tok)
    phrase = _phrase_hits(query, document)
    tn = _time_number_alignment(query, document)
    prop = _proper_noun_overlap(query, document)
    return (
        0.30 * cov
        + 0.15 * jac
        + 0.25 * phrase
        + 0.20 * tn
        + 0.25 * prop
    )


def lexical_precision(query: str, document: str) -> float:
    """
    Single lexical precision score in ~[0, 1+].

    Combines content-token coverage, Jaccard, phrase hits, times/numbers,
    and proper-noun overlap. Uses sentence-max so long windows do not dilute
    a strong evidence clause.
    """
    if not query or not document:
        return 0.0
    return _sentence_max_lexical(query, document)


_PREF_Q = re.compile(
    r"\b(prefer|preference|mention(?:ed)?|said|told|remind|favorite|favourite)\b",
    re.I,
)
_PREF_D = re.compile(
    r"\b(i (?:prefer|like|love|hate|find|always|usually|switched|chose)|"
    r"my (?:preference|favorite|favourite))\b",
    re.I,
)


def autobiographical_boost(query: str, document: str) -> float:
    """When the user asks what *they* said/prefer, boost first-person evidence."""
    if not _PREF_Q.search(query or ""):
        return 0.0
    return 0.35 if _PREF_D.search(document or "") else 0.0


def fuse_precision(
    *,
    dense_sim: float,
    bm25: float,
    bm25_max: float,
    entity_overlap: int,
    lexical: float,
    rrf_prior: float,
    rrf_max: float,
    auto_boost: float = 0.0,
) -> float:
    """
    Calibrated fusion → higher is better.

    Dense and BM25 are min-max normalized against the candidate pool.
    Lexical precision is already ~[0, 1]. Entity overlap is saturating.
    """
    d = max(0.0, float(dense_sim))
    b = (float(bm25) / bm25_max) if bm25_max > 1e-9 else 0.0
    e = 1.0 - math.exp(-0.7 * max(0, int(entity_overlap)))
    r = (float(rrf_prior) / rrf_max) if rrf_max > 1e-9 else 0.0
    lex = max(0.0, float(lexical))

    # Max-pool lets one strong channel (dense / BM25 / lexical) promote
    # evidence that would otherwise lose a weighted average — lifts
    # ConvoMem-style preference hits into small returned k.
    peak = max(d, b, lex)
    return (
        0.40 * peak
        + 0.20 * d
        + 0.16 * b
        + 0.12 * lex
        + 0.07 * e
        + 0.05 * r
        + float(auto_boost)
    )


def rerank(
    query: str,
    candidates: Sequence[Tuple[object, float, float, int, float]],
    top_k: int,
) -> List[Tuple[object, float]]:
    """
    ``candidates``: (atom, dense_sim, bm25, entity_overlap, rrf_prior)
    Returns ``[(atom, precision_score), ...]`` length ≤ top_k.
    """
    if not candidates:
        return []
    if not (query or "").strip():
        # Fall back to RRF prior order.
        ordered = sorted(candidates, key=lambda x: x[4], reverse=True)
        return [(a, float(p)) for a, _, _, _, p in ordered[:top_k]]

    bm25_max = max((c[2] for c in candidates), default=0.0)
    rrf_max = max((c[4] for c in candidates), default=0.0) or 1.0

    scored: List[Tuple[object, float]] = []
    for atom, dense_sim, bm25, ent_ov, rrf_prior in candidates:
        payload = str(getattr(atom, "payload", atom))
        lex = lexical_precision(query, payload)
        score = fuse_precision(
            dense_sim=dense_sim,
            bm25=bm25,
            bm25_max=bm25_max,
            entity_overlap=ent_ov,
            lexical=lex,
            rrf_prior=rrf_prior,
            rrf_max=rrf_max,
            auto_boost=autobiographical_boost(query, payload),
        )
        scored.append((atom, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Guarantee BM25 elite seats: the best lexical matches often *are* the
    # evidence, even when dense neighbors outrank them in RRF.
    bm25_order = sorted(candidates, key=lambda x: x[2], reverse=True)
    elite_n = min(2, top_k // 2 + 1, len(bm25_order))
    elite_ids = {id(c[0]) for c in bm25_order[:elite_n]}
    merged: List[Tuple[object, float]] = []
    seen = set()
    score_map = {id(a): s for a, s in scored}
    for atom, _, bm25, _, _ in bm25_order[:elite_n]:
        aid = id(atom)
        if aid in seen:
            continue
        merged.append((atom, score_map.get(aid, float(bm25))))
        seen.add(aid)
    for atom, score in scored:
        if id(atom) in seen:
            continue
        merged.append((atom, score))
        seen.add(id(atom))
        if len(merged) >= top_k:
            break
    return merged[:top_k]
