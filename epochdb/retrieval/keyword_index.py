"""Lightweight BM25 keyword index for hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Set, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Okapi BM25 parameters (standard defaults).
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def atom_text(payload) -> str:
    if isinstance(payload, str):
        return payload
    if hasattr(payload, "description") and payload.description:
        return str(payload.description)
    return str(payload)


class KeywordIndex:
    """Inverted index with BM25 scoring over hot-tier atom text."""

    def __init__(self) -> None:
        self._doc_tokens: Dict[str, List[str]] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._postings: Dict[str, Set[str]] = defaultdict(set)
        self._avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self._doc_tokens)

    def index(self, atom_id: str, text: str) -> None:
        self.remove(atom_id)
        tokens = tokenize(text)
        if not tokens:
            return
        self._doc_tokens[atom_id] = tokens
        self._doc_lengths[atom_id] = len(tokens)
        for term in set(tokens):
            self._postings[term].add(atom_id)
        self._recompute_avgdl()

    def remove(self, atom_id: str) -> None:
        tokens = self._doc_tokens.pop(atom_id, None)
        self._doc_lengths.pop(atom_id, None)
        if not tokens:
            return
        for term in set(tokens):
            posting = self._postings.get(term)
            if posting is None:
                continue
            posting.discard(atom_id)
            if not posting:
                self._postings.pop(term, None)
        self._recompute_avgdl()

    def clear(self) -> None:
        self._doc_tokens.clear()
        self._doc_lengths.clear()
        self._postings.clear()
        self._avgdl = 0.0

    def _recompute_avgdl(self) -> None:
        if not self._doc_lengths:
            self._avgdl = 0.0
            return
        self._avgdl = sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        query_tokens = tokenize(query)
        if not query_tokens or not self._doc_tokens:
            return []

        n_docs = len(self._doc_tokens)
        scores: Dict[str, float] = defaultdict(float)

        for term in query_tokens:
            posting = self._postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for atom_id in posting:
                tokens = self._doc_tokens[atom_id]
                tf = tokens.count(term)
                dl = self._doc_lengths[atom_id]
                denom = tf + _K1 * (1.0 - _B + _B * (dl / (self._avgdl or 1.0)))
                scores[atom_id] += idf * (tf * (_K1 + 1.0)) / (denom or 1.0)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]


def score_text_overlap(query: str, document: str) -> float:
    """Fast lexical score for cold-tier row scans (no index required)."""
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    d_tokens = tokenize(document)
    if not d_tokens:
        return 0.0

    q_set = set(q_tokens)
    matches = sum(1 for token in d_tokens if token in q_set)
    score = matches / len(q_set)

    q_lower = query.lower().strip()
    doc_lower = document.lower()
    if q_lower and q_lower in doc_lower:
        score += 1.0

    return score
