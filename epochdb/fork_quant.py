import logging
from typing import Dict, List, Optional, Any, Tuple
from .atom import UnifiedMemoryAtom, PayloadType, ScalarPayload, SeriesPayload, SeriesPoint

logger = logging.getLogger(__name__)

class QuantitativeForkManager:
    """Handles quantitative state within memory forks (shadow values and merge strategies)."""
    def __init__(self):
        # fork_id -> { field -> { atom_id: (val, low, high, timestamp) } }
        self.shadow_scalars: Dict[str, Dict[str, Dict[str, Tuple[float, float, float, float]]]] = {}
        # fork_id -> { (entity, relation) -> [point_overlays] }
        self.series_overlays: Dict[str, Dict[Tuple[str, str], List[SeriesPoint]]] = {}

    def add_shadow_scalar(self, fork_id: str, field: str, atom_id: str, val: float, low: float, high: float, timestamp: float = 0.0):
        if fork_id not in self.shadow_scalars:
            self.shadow_scalars[fork_id] = {}
        if field not in self.shadow_scalars[fork_id]:
            self.shadow_scalars[fork_id][field] = {}
        self.shadow_scalars[fork_id][field][atom_id] = (val, low, high, timestamp)

    def query_overlap(self, fork_id: str, field: str, q_low: float, q_high: float) -> List[str]:
        """Checks fork shadow scalars for overlaps."""
        if fork_id not in self.shadow_scalars or field not in self.shadow_scalars[fork_id]:
            return []
        
        results = []
        for aid, (val, low, high, ts) in self.shadow_scalars[fork_id][field].items():
            # Interval overlap: atom.low <= query.high and atom.high >= query.low
            if low <= q_high and high >= q_low:
                results.append(aid)
        return results

    def add_series_overlay(self, fork_id: str, entity: str, relation: str, points: List[SeriesPoint], strategy: str = "append"):
        if fork_id not in self.series_overlays:
            self.series_overlays[fork_id] = {}
        
        key = (entity.lower(), relation.lower())
        if strategy == "replace":
            self.series_overlays[fork_id][key] = points
        elif strategy == "append":
            if key not in self.series_overlays[fork_id]:
                self.series_overlays[fork_id][key] = []
            self.series_overlays[fork_id][key].extend(points)
        elif strategy == "interleave":
            if key not in self.series_overlays[fork_id]:
                self.series_overlays[fork_id][key] = []
            combined = self.series_overlays[fork_id][key] + points
            self.series_overlays[fork_id][key] = sorted(combined, key=lambda p: p.timestamp)

    def resolve_series_points(self, fork_id: str, entity: str, relation: str, canonical_points: List[SeriesPoint]) -> List[SeriesPoint]:
        key = (entity.lower(), relation.lower())
        if fork_id in self.series_overlays and key in self.series_overlays[fork_id]:
            return sorted(canonical_points + self.series_overlays[fork_id][key], key=lambda p: p.timestamp)
        return canonical_points

    def reconcile_conflicts(self, fork_id: str, canonical_scalars: Dict[str, Any], policy: str = "timestamp_wins") -> Dict[str, Any]:
        """Merges fork scalars into canonical state based on policy."""
        if fork_id not in self.shadow_scalars:
            return canonical_scalars
        
        merged = canonical_scalars.copy()
        for field, shadow_atoms in self.shadow_scalars[fork_id].items():
            if not shadow_atoms:
                continue

            if policy == "timestamp_wins":
                # Find latest shadow atom by timestamp
                latest = max(shadow_atoms.values(), key=lambda x: x[3])
                merged[field] = latest
            elif policy == "confidence_wins":
                # Requires confidence metadata, fallback to timestamp for now
                latest = max(shadow_atoms.values(), key=lambda x: x[3])
                merged[field] = latest
            elif policy == "explicit":
                logger.warning(f"Conflict in fork {fork_id} for field {field}: requires explicit resolution")
                continue
                
        return merged
