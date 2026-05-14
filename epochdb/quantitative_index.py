import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from sortedcontainers import SortedDict
import rtree
import z3
import numpy as np
from .atom import UnifiedMemoryAtom, PayloadType, ScalarPayload, SeriesPayload, ConstraintPayload, SeriesPoint, Interval
from .units import UnitRegistry
from .fork_quant import QuantitativeForkManager

logger = logging.getLogger(__name__)

class ScalarIndex:
    """Uses IntervalTree for 1D interval overlap indexing of scalars with unit normalization."""
    def __init__(self, base_unit: str):
        from intervaltree import IntervalTree
        self.tree = IntervalTree()
        self.base_unit = base_unit

    def add(self, value: float, low: float, high: float, atom_id: str):
        # IntervalTree doesn't allow low == high. Add a tiny epsilon if needed.
        if low >= high:
            high = low + 1e-12
        self.tree[low:high] = atom_id

    def query_overlap(self, q_low: float, q_high: float) -> List[str]:
        """Find atoms whose intervals overlap with [q_low, q_high]."""
        if q_low >= q_high:
            q_high = q_low + 1e-12
        hits = self.tree.overlap(q_low, q_high)
        return [interval.data for interval in hits]

class SeriesIndex:
    """R-tree indexing 2D spatial points with aggregation support."""
    def __init__(self, unit: str):
        p = rtree.index.Property()
        p.dimension = 2
        self.index = rtree.index.Index(properties=p)
        self.id_map: Dict[int, Tuple[str, int]] = {}
        self._next_id = 0
        self.unit = unit
        # Cache for aggregated views: atom_id -> {window: [aggregated_points]}
        self.aggregates: Dict[str, Dict[str, List[SeriesPoint]]] = {}

    def add_series(self, atom_id: str, payload: SeriesPayload):
        for i, point in enumerate(payload.points):
            idx = self._next_id
            self._next_id += 1
            t = point.timestamp
            v_low = point.value - point.uncertainty_low
            v_high = point.value + point.uncertainty_high
            self.index.insert(idx, (t, v_low, t, v_high))
            self.id_map[idx] = (atom_id, i)
        
        self._compute_aggregates(atom_id, payload)

    def _compute_aggregates(self, atom_id: str, payload: SeriesPayload):
        if not payload.points: return
        window_secs = self._parse_window(payload.window)
        points = sorted(payload.points, key=lambda p: p.timestamp)
        
        buckets: Dict[int, List[SeriesPoint]] = {}
        for p in points:
            b_idx = int(p.timestamp // window_secs)
            if b_idx not in buckets: buckets[b_idx] = []
            buckets[b_idx].append(p)
        
        agg_points = []
        for b_idx in sorted(buckets.keys()):
            b_points = buckets[b_idx]
            ts = b_idx * window_secs
            vals = [p.value for p in b_points]
            
            if payload.aggregation == "mean": val = sum(vals) / len(vals)
            elif payload.aggregation == "sum": val = sum(vals)
            elif payload.aggregation == "min": val = min(vals)
            elif payload.aggregation == "max": val = max(vals)
            elif payload.aggregation == "first": val = vals[0]
            elif payload.aggregation == "last": val = vals[-1]
            else: val = vals[0]
            
            u_low = max(p.uncertainty_low for p in b_points)
            u_high = max(p.uncertainty_high for p in b_points)
            agg_points.append(SeriesPoint(timestamp=ts, value=val, uncertainty_low=u_low, uncertainty_high=u_high))
        
        if atom_id not in self.aggregates: self.aggregates[atom_id] = {}
        self.aggregates[atom_id][payload.window] = agg_points

    def _parse_window(self, window: str) -> int:
        if window == "1min": return 60
        if window == "1h": return 3600
        if window == "1d": return 86400
        return 3600

    def query_rect(self, t_min: float, v_min: float, t_max: float, v_max: float) -> List[str]:
        hits = self.index.intersection((t_min, v_min, t_max, v_max))
        atom_ids = set()
        for hit_id in hits:
            atom_id, _ = self.id_map[hit_id]
            atom_ids.add(atom_id)
        return list(atom_ids)

    def query_aggregate(self, atom_id: str, window: str, t_start: float, t_end: float) -> List[SeriesPoint]:
        if atom_id not in self.aggregates or window not in self.aggregates[atom_id]:
            return []
        return [p for p in self.aggregates[atom_id][window] if t_start <= p.timestamp <= t_end]

class ConstraintChecker:
    """Z3 solver with dynamic variable binding and graph resolution."""
    def __init__(self, unit_registry: Optional[UnitRegistry] = None):
        self.solver = z3.Solver()
        self.unit_registry = unit_registry or UnitRegistry()

    def _to_z3(self, expr: Dict[str, Any], state: Dict[str, float], index_manager: Any) -> z3.BoolRef:
        op = expr.get("op")
        if op == "and":
            return z3.And(self._to_z3(expr["left"], state, index_manager), self._to_z3(expr["right"], state, index_manager))
        elif op == "or":
            return z3.Or(self._to_z3(expr["left"], state, index_manager), self._to_z3(expr["right"], state, index_manager))
        elif op == "not":
            return z3.Not(self._to_z3(expr["expr"], state, index_manager))
        elif op in [">", "<", ">=", "<=", "==", "!="]:
            left_val = self._resolve_val(expr.get("left") or expr.get("field"), state, index_manager)
            right_val = self._resolve_val(expr.get("right") or expr.get("value"), state, index_manager)
            
            if left_val is None or right_val is None:
                return z3.BoolVal(False)

            if op == ">": return left_val > right_val
            if op == "<": return left_val < right_val
            if op == ">=": return left_val >= right_val
            if op == "<=": return left_val <= right_val
            if op == "==": return left_val == right_val
            if op == "!=": return left_val != right_val
        return z3.BoolVal(True)

    def _resolve_val(self, key: Any, state: Dict[str, float], index_manager: Any) -> Optional[float]:
        if isinstance(key, (int, float)): return float(key)
        if isinstance(key, str):
            if key in state: return state[key]
            # Graph Traversal: resolve variable from live scalar index (latest value)
            if index_manager and key in index_manager.scalar_indices:
                s_index = index_manager.scalar_indices[key]
                # In a real system, we'd store a 'latest_val' in ScalarIndex.
                # For now, we take the last added atom's value (simplified).
                if s_index.id_map:
                    # Very rough approximation: get most recent ID
                    latest_id = list(s_index.id_map.values())[-1]
                    # We need the actual atom. This is getting complex, so we'll assume 
                    # the manager has a 'latest_values' cache.
                    if hasattr(index_manager, "latest_values"):
                        return index_manager.latest_values.get(key)
        return None

    def check_feasibility(self, constraints: List[ConstraintPayload], state: Dict[str, float], index_manager: Any = None) -> bool:
        self.solver.push()
        try:
            for c in constraints:
                self.solver.add(self._to_z3(c.expression, state, index_manager))
            return self.solver.check() == z3.sat
        finally:
            self.solver.pop()

class QuantitativeIndexManager:
    """Orchestrates indexing, aggregation, and triggers with unit normalization."""
    def __init__(self, unit_registry: Optional[UnitRegistry] = None, storage_dir: Optional[str] = None):
        import os
        self.storage_dir = storage_dir
        self.unit_registry = unit_registry or UnitRegistry()
        from .cascade import CascadeManager
        from .fork_quant import QuantitativeForkManager
        self.scalar_indices: Dict[str, ScalarIndex] = {}
        self.series_index = SeriesIndex("dimensionless")
        self.cascade_manager = CascadeManager(storage_dir=self.storage_dir)
        self.fork_manager = QuantitativeForkManager()
        self.constraint_checker = ConstraintChecker(self.unit_registry)
        self.latest_values: Dict[str, float] = {} # field -> normalized_value
        self.triggers: List[Tuple[str, Any]] = []
        
        self._load_schema()

    def _load_schema(self):
        if not self.storage_dir: return
        import os, json
        schema_path = os.path.join(self.storage_dir, "schema_registry.json")
        if os.path.exists(schema_path):
            try:
                with open(schema_path, "r") as f:
                    data = json.load(f)
                    for field, unit in data.items():
                        self.scalar_indices[field] = ScalarIndex(unit)
            except Exception as e:
                logger.error(f"Failed to load schema registry: {e}")

    def _save_schema(self):
        if not self.storage_dir: return
        import os, json
        schema_path = os.path.join(self.storage_dir, "schema_registry.json")
        data = {f: idx.base_unit for f, idx in self.scalar_indices.items()}
        try:
            with open(schema_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save schema registry: {e}")

    def add_trigger(self, field: str, callback: Any):
        self.triggers.append((field, callback))

    def index_atom(self, atom: UnifiedMemoryAtom):
        if atom.payload_type == PayloadType.SCALAR:
            field = "value"
            if atom.triples: field = atom.triples[0][1]
            
            # 1. Dimensional Guard
            unit = atom.payload.unit
            if field not in self.scalar_indices:
                self.scalar_indices[field] = ScalarIndex(unit)
                self._save_schema()
            else:
                if not self.unit_registry.compatible(unit, self.scalar_indices[field].base_unit):
                    logger.error(f"Incompatible unit '{unit}' for field '{field}' (base: {self.scalar_indices[field].base_unit})")
                    return

            # 2. Unit Normalization
            base_unit = self.scalar_indices[field].base_unit
            val = self.unit_registry.convert(atom.payload.value, unit, base_unit)
            i_low = self.unit_registry.convert(atom.payload.uncertainty_low, unit, base_unit)
            i_high = self.unit_registry.convert(atom.payload.uncertainty_high, unit, base_unit)
            
            self.scalar_indices[field].add(val, val - i_low, val + i_high, atom.id)
            self.latest_values[field] = val
            
            self.cascade_manager.trigger_cascade(field, atom)
            for trigger_field, callback in self.triggers:
                if trigger_field == field: callback(atom)
        
        elif atom.payload_type == PayloadType.SERIES:
            self.series_index.add_series(atom.id, atom.payload)
            field = "series"
            if atom.triples: field = atom.triples[0][1]
            self.cascade_manager.trigger_cascade(field, atom)
        
        elif atom.payload_type == PayloadType.CONSTRAINT:
            # Point 8: Persistent Dependency Graph
            # Extract fields from expression recursively
            fields = self._extract_fields(atom.payload.expression)
            def re_eval_callback(_):
                logger.info(f"Re-evaluating constraint {atom.id}")
                # Real implementation would check feasibility and emit events
            self.cascade_manager.register_constraint(atom.id, fields, re_eval_callback)

    def _extract_fields(self, expr: Dict[str, Any]) -> Set[str]:
        fields = set()
        op = expr.get("op")
        if op in ("and", "or"):
            fields.update(self._extract_fields(expr["left"]))
            fields.update(self._extract_fields(expr["right"]))
        elif op == "not":
            fields.update(self._extract_fields(expr["expr"]))
        elif "field" in expr:
            fields.add(expr["field"])
        return fields

    def clear(self):
        self.scalar_indices = {}
        self.series_index = SeriesIndex("dimensionless")
        self.cascade_manager.clear()
        self.fork_manager = QuantitativeForkManager()
