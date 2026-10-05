# Symbolic Validators (LCAG)

The `epochdb.validation` module defines the interfaces for **Logical Constraint-Augmented Generation (LCAG)**.

---

## `ValidationStatus` (Enum)

```python
from enum import Enum

class ValidationStatus(str, Enum):
    APPROVED = "APPROVED"               # Proposal meets all deterministic constraints
    REJECTED = "REJECTED"               # Proposal violates one or more rules
    PENDING_REVIEW = "PENDING_REVIEW"   # Requires Human-In-The-Loop (HITL) approval
```

---

## `ValidationResult` (Dataclass)

```python
from dataclasses import dataclass

@dataclass
class ValidationResult:
    status: ValidationStatus
    reason: str | None = None
    metadata: dict | None = None
```

---

## `SymbolicValidator` (Abstract Base Class)

To create a custom guardrail, inherit from `SymbolicValidator` and implement the `verify` method:

```python
from abc import ABC, abstractmethod
from epochdb.validation import ValidationResult

class SymbolicValidator(ABC):
    @abstractmethod
    def verify(
        self,
        text: str,
        metadata: dict | None,
        kg_snapshot: dict,
    ) -> ValidationResult:
        """
        Evaluate candidate write before durable commit.

        :param text: Candidate verbatim string.
        :param metadata: Accompanying metadata dictionary.
        :param kg_snapshot: Immutable snapshot containing active entities,
                            predicates, and active epoch IDs.
        :return: ValidationResult indicating approval or rejection.
        """
        raise NotImplementedError
```
