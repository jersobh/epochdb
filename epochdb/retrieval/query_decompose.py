"""
Lightweight query decomposition for conversational memory recall.

No LLM required: heuristic splits that surface *multiple intents* in a
single user question (preference + ask, given-X-what-Y, etc.) so candidate
generation is not dominated by one facet.
"""

from __future__ import annotations

import re
from typing import List


_GIVEN = re.compile(
    r"^\s*(?:given|considering|based on)\s+(.+?)[,;]\s*(.+?)\s*$",
    re.I | re.S,
)
_REMIND = re.compile(
    r"^\s*(?:remind me[, ]+|can you remind me[, ]+|what did i (?:mention|say)|"
    r"what was|what is|what's|where did i|when did i|who did i)\s+",
    re.I,
)


def focus_query(query: str) -> str:
    """Strip interrogative wrappers; keep the topical core."""
    q = (query or "").strip()
    if not q:
        return q
    q2 = _REMIND.sub("", q).strip()
    return q2 or q


_STOP_LIGHT = frozenset(
    "the a an and or to of in on for is are was were be what when where "
    "which who whom how why did do does have has had can could would should "
    "will just also very really some any my me i you your we our they them "
    "their with from as at by about into over after before please".split()
)


def content_keyword_query(query: str) -> str:
    """Bag of content words — strong BM25 signal for sparse evidence turns."""
    words = re.findall(r"[A-Za-z0-9']+", query or "")
    keep = []
    for w in words:
        wl = w.lower()
        if wl in _STOP_LIGHT or len(wl) < 3:
            continue
        keep.append(w)
    return " ".join(keep[:14])


def decompose_query(query: str) -> List[str]:
    """
    Return 1–4 focused sub-queries (original first).

    Examples
    --------
    "Given my preference for X, what Y should we do?"
        → [full, "my preference for X", "what Y should we do", content-keywords]
    """
    q = (query or "").strip()
    if not q:
        return []
    out: List[str] = [q]
    focused = focus_query(q)
    if focused.lower() != q.lower():
        out.append(focused)

    m = _GIVEN.match(q)
    if m:
        left, right = m.group(1).strip(), m.group(2).strip()
        if len(left) > 8:
            out.append(left)
        if len(right) > 8:
            out.append(right)

    ck = content_keyword_query(focused if focused else q)
    if ck and ck.lower() not in {x.lower() for x in out}:
        out.append(ck)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for item in out:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq[:5]
