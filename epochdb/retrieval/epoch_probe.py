"""
Cold-tier epoch probing.

Instead of opening every historical HNSW index, select a probe set from:

1. Recency — newest epochs by parquet mtime (working-set analogue).
2. Entity routing — epochs listed in the Global Entity Index for query entities.
3. Centroid neighbours — cosine(query, epoch mean embedding) among the rest.

Small stores fall back to exhaustive search automatically so existing
recall tests and tiny deployments do not change behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

import numpy as np


DEFAULT_COLD_SEARCH_MODE = "probe"
DEFAULT_RECENCY_EPOCHS = 6
DEFAULT_CENTROID_PROBES = 12
DEFAULT_TOPIC_LOCK_FETCH_CAP = 512


@dataclass
class EpochProbePlan:
    """Which cold-tier epochs a query will open."""

    mode: str
    epochs: List[str]
    n_total: int
    from_recency: List[str] = field(default_factory=list)
    from_entity: List[str] = field(default_factory=list)
    from_centroid: List[str] = field(default_factory=list)
    auto_exhaustive: bool = False

    @property
    def n_searched(self) -> int:
        return len(self.epochs)


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-10:
        return arr
    return arr / norm


def plan_epoch_search(
    cold_tier,
    query_emb: np.ndarray,
    entity_epochs: Optional[Iterable[str]] = None,
    mode: str = DEFAULT_COLD_SEARCH_MODE,
    recency_k: int = DEFAULT_RECENCY_EPOCHS,
    centroid_k: int = DEFAULT_CENTROID_PROBES,
) -> EpochProbePlan:
    """
    Return the cold-tier epochs to search for ``query_emb``.

    ``mode="exhaustive"`` always returns every epoch. ``mode="probe"``
    returns recency + entity + centroid neighbours, unless the archive is
    small enough that probing cannot save work.
    """
    all_epochs = list(cold_tier.get_all_epochs())
    n_total = len(all_epochs)
    entity_set: Set[str] = {e for e in (entity_epochs or []) if e in set(all_epochs)}

    if n_total == 0:
        return EpochProbePlan(mode=mode, epochs=[], n_total=0)

    resolved_mode = (mode or DEFAULT_COLD_SEARCH_MODE).strip().lower()
    if resolved_mode not in ("probe", "exhaustive"):
        resolved_mode = DEFAULT_COLD_SEARCH_MODE

    recency_k = max(0, int(recency_k))
    centroid_k = max(0, int(centroid_k))
    budget = recency_k + centroid_k

    if resolved_mode == "exhaustive" or n_total <= max(budget, 1):
        return EpochProbePlan(
            mode="exhaustive" if resolved_mode == "exhaustive" else "probe",
            epochs=list(all_epochs),
            n_total=n_total,
            from_recency=list(all_epochs) if resolved_mode != "exhaustive" else [],
            from_entity=sorted(entity_set),
            from_centroid=[],
            auto_exhaustive=resolved_mode != "exhaustive",
        )

    by_mtime = cold_tier.get_epochs_by_mtime(descending=True)
    from_recency = [e for e in by_mtime[:recency_k]]
    selected: Set[str] = set(from_recency)
    selected.update(entity_set)

    remaining = [e for e in all_epochs if e not in selected]
    from_centroid: List[str] = []
    q = _normalize(query_emb)
    if centroid_k > 0 and remaining and float(np.linalg.norm(q)) > 1e-10:
        scored = []
        for epoch_id in remaining:
            centroid = cold_tier.get_centroid(epoch_id)
            if centroid is None or centroid.size != q.size:
                continue
            scored.append((float(np.dot(_normalize(centroid), q)), epoch_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        from_centroid = [epoch_id for _, epoch_id in scored[:centroid_k]]
        selected.update(from_centroid)

    # Preserve a stable order: recency first, then the rest in directory order.
    ordered = [e for e in from_recency]
    seen = set(ordered)
    for epoch_id in all_epochs:
        if epoch_id in selected and epoch_id not in seen:
            ordered.append(epoch_id)
            seen.add(epoch_id)

    return EpochProbePlan(
        mode="probe",
        epochs=ordered,
        n_total=n_total,
        from_recency=from_recency,
        from_entity=sorted(entity_set),
        from_centroid=from_centroid,
        auto_exhaustive=False,
    )
