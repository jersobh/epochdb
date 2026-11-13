"""Shared helpers for official benchmark ingest / recall."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


def unwrap_engine(db: Any) -> Any:
    """Return the underlying EpochDB from sync/async wrappers."""
    if hasattr(db, "async_db") and getattr(db.async_db, "_db", None) is not None:
        return db.async_db._db
    if hasattr(db, "_db") and db._db is not None:
        return db._db
    return db


def reset_case_state(db: Any) -> None:
    """Clear hot tier, WAL, and KG between isolated official cases."""
    engine = unwrap_engine(db)
    if hasattr(db, "hot_tier"):
        db.hot_tier.clear()
    if hasattr(db, "wal"):
        db.wal.clear()
    kg = getattr(db, "kg_manager", None) or getattr(engine, "kg_manager", None)
    if kg is not None and hasattr(kg, "clear"):
        try:
            kg.clear()
        except Exception:
            pass
    preds = getattr(engine, "predicates", None)
    if preds is not None and hasattr(preds, "clear"):
        preds.clear()


def extract_triples(db: Any, text: str) -> List[Tuple[str, str, str]]:
    """Run the configured FactExtractor (local / hf / cloud) on ``text``."""
    engine = unwrap_engine(db)
    getter = getattr(engine, "_get_fact_extractor", None)
    if getter is None:
        return []
    try:
        triples = getter().extract(text) or []
        return [(str(s), str(p), str(o)) for s, p, o in triples]
    except Exception:
        return []


def surface_query_entities(text: str) -> List[str]:
    """Capitalized / quoted tokens to seed recall even before they exist in the KG."""
    if not text:
        return []
    out: List[str] = []
    seen = set()
    for raw in text.replace("?", " ").replace("'", " ").split():
        w = raw.strip(".,!?;:()\"'").strip()
        if len(w) < 3:
            continue
        if w[0].isupper() or w.lower() in {"monday", "tuesday", "wednesday", "thursday", "friday"}:
            key = w.lower()
            if key not in seen:
                seen.add(key)
                out.append(w)
    return out[:12]


def evidence_hit(
    answer: Any,
    evidence_texts: Optional[List[str]],
    payloads: List[str],
) -> bool:
    """True if answer or evidence-turn text is supported by retrieved payloads."""
    blob = " ".join(payloads).lower()
    ans = str(answer or "").strip().lower()
    if ans and ans in blob:
        return True
    # Short answers / times / names
    if ans:
        compact = ans.replace(" ", "")
        if len(ans) <= 40 and (ans in blob or compact in blob.replace(" ", "")):
            return True
    for ev in evidence_texts or []:
        ev_l = (ev or "").lower().strip()
        if len(ev_l) < 8:
            continue
        if ev_l[:100] in blob or ev_l in blob:
            return True
        # Distinctive window from evidence (skip leading pronouns)
        words = [w for w in ev_l.split() if len(w) > 2]
        for window in (8, 6, 4):
            if len(words) >= window:
                span = " ".join(words[:window])
                if span in blob:
                    return True
                span = " ".join(words[-window:])
                if span in blob:
                    return True
        tokens = [t for t in words if len(t) > 3][:16]
        need = max(2, (len(tokens) + 1) // 2)
        if tokens and sum(1 for t in tokens if t in blob) >= need:
            return True
    if ans:
        tokens = [t for t in ans.replace(",", " ").replace(".", " ").split() if len(t) > 2]
        if tokens and sum(1 for t in tokens if t in blob) >= max(1, (len(tokens) + 1) // 2):
            return True
    return False
