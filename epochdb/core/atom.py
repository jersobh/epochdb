import uuid
import time
from dataclasses import dataclass, field, asdict
from typing import Any, List, Tuple, Optional, Dict, Union
from enum import Enum
import numpy as np

class PayloadType(Enum):
    TEXT = "text"
    SCALAR = "scalar"
    SERIES = "series"
    CONSTRAINT = "constraint"


class MemoryType(Enum):
    """Semantic memory categories for retrieval prioritization."""
    GENERAL = "general"
    EPISODIC = "episodic"      # Conversational context that persists across sessions
    PROFILE = "profile"        # Long-term user facts and preferences
    WORKING = "working"        # Short-term context for the current session
    SKILL = "skill"            # Synthesized procedural skills & executable tools


@dataclass
class Interval:
    low: float
    high: float

    def __add__(self, other):
        return Interval(self.low + other.low, self.high + other.high)
    
    def __sub__(self, other):
        return Interval(self.low - other.high, self.high - other.low)

    def contains(self, val: float) -> bool:
        return self.low <= val <= self.high

    def overlaps(self, other) -> bool:
        return not (self.high < other.low or self.low > other.high)

@dataclass
class ScalarPayload:
    value: float
    unit: str = "dimensionless"
    uncertainty_low: float = 0.0
    uncertainty_high: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def interval(self) -> Interval:
        return Interval(self.value - self.uncertainty_low, self.value + self.uncertainty_high)

@dataclass
class SeriesPoint:
    timestamp: float
    value: float
    uncertainty_low: float = 0.0
    uncertainty_high: float = 0.0

@dataclass
class SeriesPayload:
    points: List[SeriesPoint]
    unit: str = "dimensionless"
    interpolation: str = "linear"  # linear, step, cubic
    aggregation: str = "mean"      # mean, sum, min, max, last, first
    window: str = "1h"             # 1min, 1h, 1d
    supersession_policy: str = "append" # append, replace, patch

@dataclass
class ConstraintPayload:
    expression: Dict[str, Any]  # Recursive tree: {"op": "and", "left": {...}, "right": {...}}
                                # or {"op": ">", "field": "price", "value": 100}
    description: Optional[str] = None

@dataclass
class UnifiedMemoryAtom:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Union[str, ScalarPayload, SeriesPayload, ConstraintPayload] = None
    payload_type: PayloadType = PayloadType.TEXT
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    triples: List[Tuple[str, str, str]] = field(default_factory=list)
    created_at: float = 0.0
    access_count: int = 0
    epoch_id: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    namespace: Optional[str] = None
    memory_type: MemoryType = MemoryType.GENERAL

    def __post_init__(self):
        if self.payload_type == PayloadType.TEXT:
            if isinstance(self.payload, ScalarPayload):
                self.payload_type = PayloadType.SCALAR
            elif isinstance(self.payload, SeriesPayload):
                self.payload_type = PayloadType.SERIES
            elif isinstance(self.payload, ConstraintPayload):
                self.payload_type = PayloadType.CONSTRAINT

    def calculate_saliency(self) -> float:
        """
        S = R / (T + 1)
        where R is access_count and T is time since creation in hours.
        """
        T_hours = (time.time() - self.created_at) / 3600.0
        return self.access_count / (T_hours + 1.0)

    def to_dict(self):
        payload_data = self.payload
        if isinstance(self.payload, ScalarPayload):
            payload_data = {
                "value": self.payload.value,
                "unit": self.payload.unit,
                "uncertainty_low": self.payload.uncertainty_low,
                "uncertainty_high": self.payload.uncertainty_high,
                "timestamp": self.payload.timestamp,
            }
        elif isinstance(self.payload, SeriesPayload):
            payload_data = {
                "points": [
                    {
                        "timestamp": p.timestamp,
                        "value": p.value,
                        "uncertainty_low": p.uncertainty_low,
                        "uncertainty_high": p.uncertainty_high,
                    }
                    for p in self.payload.points
                ],
                "unit": self.payload.unit,
                "interpolation": self.payload.interpolation,
                "aggregation": self.payload.aggregation,
                "window": self.payload.window,
                "supersession_policy": self.payload.supersession_policy
            }
        elif isinstance(self.payload, ConstraintPayload):
            payload_data = {
                "expression": self.payload.expression,
                "description": self.payload.description,
            }
        
        return {
            "id": self.id,
            "payload": payload_data,
            "payload_type": self.payload_type.value,
            "embedding": self.embedding.tolist(),
            "triples": self.triples,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "epoch_id": self.epoch_id,
            "metadata": self.metadata,
            "namespace": self.namespace,
            "memory_type": self.memory_type.value,
        }

    @classmethod
    def from_dict(cls, data):
        ptype_str = data.get("payload_type", "text")
        ptype = PayloadType(ptype_str)
        payload = data["payload"]

        if ptype == PayloadType.SCALAR:
            payload = ScalarPayload(
                value=payload["value"],
                unit=payload.get("unit", "dimensionless"),
                uncertainty_low=payload.get("uncertainty_low", 0.0),
                uncertainty_high=payload.get("uncertainty_high", 0.0),
                timestamp=payload.get("timestamp", time.time()),
            )
        elif ptype == PayloadType.SERIES:
            points = [
                SeriesPoint(
                    timestamp=p["timestamp"],
                    value=p["value"],
                    uncertainty_low=p.get("uncertainty_low", 0.0),
                    uncertainty_high=p.get("uncertainty_high", 0.0),
                )
                for p in payload.get("points", [])
            ]
            payload = SeriesPayload(
                points=points,
                unit=payload.get("unit", "dimensionless"),
                interpolation=payload.get("interpolation", "linear"),
                aggregation=payload.get("aggregation", "mean"),
                window=payload.get("window", "1h"),
                supersession_policy=payload.get("supersession_policy", "append")
            )
        elif ptype == PayloadType.CONSTRAINT:
            payload = ConstraintPayload(**payload)

        # Parse memory_type with backward compatibility
        mt_str = data.get("memory_type", "general")
        try:
            mt = MemoryType(mt_str)
        except ValueError:
            mt = MemoryType.GENERAL

        return cls(
            id=data["id"],
            payload=payload,
            payload_type=ptype,
            embedding=np.array(data["embedding"], dtype=np.float32),
            triples=[tuple(t) for t in data["triples"]],
            created_at=data["created_at"],
            access_count=data["access_count"],
            epoch_id=data["epoch_id"],
            metadata=data.get("metadata", {}),
            namespace=data.get("namespace"),
            memory_type=mt,
        )
