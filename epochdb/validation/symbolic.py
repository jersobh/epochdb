"""
Symbolic validator interface for Logical Constraint-Augmented Generation (LCAG).

Validators inspect a proposed write against a Knowledge Graph snapshot and
return a deterministic APPROVED / REJECTED decision. Implementations may be
pure Python (e.g. Pydantic rules) or bridge to external SMT solvers (Z3, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ValidationStatus(Enum):
    """Lifecycle / decision status for a staged state change."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ValidationResult:
    """Deterministic outcome of a symbolic verification pass."""

    status: ValidationStatus
    reason: Optional[str] = None


class SymbolicValidator(ABC):
    """
    Base class for symbolic (deterministic) validators.

    Subclass and implement :meth:`verify`. Inject instances via
    ``EpochDB(validators=[...])`` to enable the two-phase ``propose`` path.

    Example::

        class NoForbiddenEntity(SymbolicValidator):
            def verify(self, text, metadata, kg_snapshot):
                if "FORBIDDEN" in (metadata or {}).get("tags", []):
                    return ValidationResult(
                        ValidationStatus.REJECTED,
                        reason="tag FORBIDDEN is not allowed",
                    )
                return ValidationResult(ValidationStatus.APPROVED)
    """

    @abstractmethod
    def verify(
        self,
        text: str,
        metadata: Dict[str, Any],
        kg_snapshot: Dict[str, Any],
    ) -> ValidationResult:
        """
        Verify a proposed memory against the current KG snapshot.

        Parameters
        ----------
        text:
            Proposed memory payload.
        metadata:
            Proposed metadata (read-only by convention; do not mutate).
        kg_snapshot:
            Lightweight, read-oriented view of the active Knowledge Graph.
            Prefer treating this as immutable; avoid deep copies in hot paths.
        """
        raise NotImplementedError
