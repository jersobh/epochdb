# Neuro-Symbolic Guardrails with LCAG

This tutorial demonstrates how to implement deterministic security policies using EpochDB's **Logical Constraint-Augmented Generation (LCAG)** two-phase commit interface (`propose`).

---

## Scenario: Financial Portfolio Guardrail

In an automated investment platform, an LLM agent generates rebalancing trades. However, SEC and risk regulations enforce hard constraints:
1. No single asset can exceed 30% of total portfolio value.
2. High-risk cryptocurrency assets cannot be purchased if the account risk profile is `"conservative"`.

If an agent hallucinates a trade violating these rules, the write must be rejected **before** entering durable state.

---

## Complete Implementation

```python
from epochdb import (
    EpochDB,
    SymbolicValidator,
    ValidationResult,
    ValidationStatus,
)

class PortfolioRiskValidator(SymbolicValidator):
    def verify(self, text: str, metadata: dict | None, kg_snapshot: dict) -> ValidationResult:
        meta = metadata or {}
        
        asset = meta.get("asset")
        allocation = meta.get("allocation_pct", 0)
        risk_profile = meta.get("risk_profile", "conservative")
        asset_class = meta.get("asset_class", "equity")

        # Rule 1: Max 30% allocation per asset
        if allocation > 30.0:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                reason=f"Asset allocation for {asset} ({allocation}%) exceeds 30% risk ceiling."
            )

        # Rule 2: Conservative profiles cannot buy crypto
        if risk_profile == "conservative" and asset_class == "crypto":
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                reason="Cryptocurrency allocations prohibited under conservative risk profiles."
            )

        return ValidationResult(ValidationStatus.APPROVED)

def main():
    validators = [PortfolioRiskValidator()]

    with EpochDB(storage_dir="./portfolio_demo", validators=validators) as db:
        print("=== Test 1: Compliant Equity Trade ===")
        res1 = db.propose(
            "Allocate 20% to Microsoft Corp (MSFT).",
            metadata={
                "asset": "MSFT",
                "allocation_pct": 20.0,
                "asset_class": "equity",
                "risk_profile": "conservative"
            }
        )
        print("Status:", res1["status"])
        print("Committed Atom ID:", res1.get("memory_id"))

        print("\n=== Test 2: Over-allocation Trade (Violation) ===")
        res2 = db.propose(
            "Allocate 45% to Apple Inc (AAPL).",
            metadata={
                "asset": "AAPL",
                "allocation_pct": 45.0,
                "asset_class": "equity",
                "risk_profile": "conservative"
            }
        )
        print("Status:", res2["status"])
        print("Rejection Error:", res2.get("error"))

        print("\n=== Test 3: Unauthorized Asset Class (Violation) ===")
        res3 = db.propose(
            "Allocate 10% to Bitcoin (BTC).",
            metadata={
                "asset": "BTC",
                "allocation_pct": 10.0,
                "asset_class": "crypto",
                "risk_profile": "conservative"
            }
        )
        print("Status:", res3["status"])
        print("Rejection Error:", res3.get("error"))

if __name__ == "__main__":
    main()
```
