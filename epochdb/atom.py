import uuid
import time
from dataclasses import dataclass, field
from typing import Any, List, Tuple
import numpy as np

@dataclass
class UnifiedMemoryAtom:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Any = None
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    triples: List[Tuple[str, str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    epoch_id: str = "active"

    def calculate_saliency(self) -> float:
        """
        S = R / (T + 1)
        where R is access_count and T is time since creation in hours.
        """
        T_hours = (time.time() - self.created_at) / 3600.0
        return self.access_count / (T_hours + 1.0)

    def to_dict(self):
        return {
            "id": self.id,
            "payload": self.payload,
            "embedding": self.embedding.tolist(),
            "triples": self.triples,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "epoch_id": self.epoch_id
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            payload=data["payload"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            triples=[tuple(t) for t in data["triples"]],
            created_at=data["created_at"],
            access_count=data["access_count"],
            epoch_id=data["epoch_id"]
        )
