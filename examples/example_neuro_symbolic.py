"""
example_neuro_symbolic.py — Everyday LCAG (two-phase memory commits)
===================================================================
Story: a personal AI assistant for Maya keeps durable facts in EpochDB
(workplace, allergies, calendar). The LLM can *propose* changes, but
deterministic rules decide what actually gets saved.

  propose(intent) → stage → check rules → save  OR  reject with a reason

Why this matters: LLMs guess. Symbolic validators make sure bad guesses
never become long-term memory.

Runs fully offline (no API keys / embedding model required).

Usage:
    python examples/example_neuro_symbolic.py
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile

from epochdb import (
    EpochDB,
    SymbolicValidator,
    ValidationResult,
    ValidationStatus,
)

# ── pretty printing ────────────────────────────────────────────────────────────

R, BD, GN, RD, YL, CY, DIM = (
    "\033[0m", "\033[1m", "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[2m",
)


def section(title: str) -> None:
    print(f"\n{BD}{CY}▸ {title}{R}")


def show(who: str, result: dict) -> None:
    if result.get("status") == "APPROVED":
        print(f"  {GN}APPROVED{R}  {who}")
        print(f"           saved as {result['memory_id'][:8]}…")
    else:
        print(f"  {RD}REJECTED{R}  {who}")
        print(f"           reason: {result.get('error')}")


# ── Real-world rules (simple to read) ──────────────────────────────────────────


class OnlyTrustedWriters(SymbolicValidator):
    """
    Profile facts may only come from trusted channels.

    Think: the user themselves, HR sync, or the calendar app —
    not a random web scrape or an unverified chat rumor.
    """

    TRUSTED = {"user", "hr-system", "calendar"}

    def verify(self, text, metadata, kg_snapshot):
        source = (metadata or {}).get("source")
        if source not in self.TRUSTED:
            return ValidationResult(
                ValidationStatus.REJECTED,
                reason=(
                    f"'{source}' is not allowed to change profile facts "
                    "(use user / hr-system / calendar)"
                ),
            )
        return ValidationResult(ValidationStatus.APPROVED)


class NoSecretsInMemory(SymbolicValidator):
    """Never store passwords, API keys, or card numbers as durable memory."""

    PATTERNS = [
        (re.compile(r"\bpassword\b\s*(is\s*)?[:=]", re.I), "looks like a password"),
        (re.compile(r"\bapi[_ ]?key\b", re.I), "looks like an API key"),
        (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "looks like a card number"),
    ]

    def verify(self, text, metadata, kg_snapshot):
        for pattern, label in self.PATTERNS:
            if pattern.search(text or ""):
                return ValidationResult(
                    ValidationStatus.REJECTED,
                    reason=f"refusing to store secrets ({label})",
                )
        return ValidationResult(ValidationStatus.APPROVED)


class OneCurrentEmployer(SymbolicValidator):
    """
    Maya can only have one *current* workplace in the knowledge graph.

    If she already `works_at` somewhere, reject a second employer fact
    instead of silently accumulating contradictions the agent invented.
    """

    def verify(self, text, metadata, kg_snapshot):
        triples = (metadata or {}).get("triples") or []
        entities = set(kg_snapshot.get("entities") or [])
        predicates = set(kg_snapshot.get("predicates") or [])

        for triple in triples:
            if len(triple) < 3:
                continue
            person, rel, company = str(triple[0]), str(triple[1]), str(triple[2])
            if rel != "works_at":
                continue
            # Person already known AND we already track works_at somewhere → conflict risk
            if person in entities and "works_at" in predicates:
                return ValidationResult(
                    ValidationStatus.REJECTED,
                    reason=(
                        f"{person} already has a workplace on file — "
                        f"refuse inventing a second job at {company}. "
                        "Update the existing fact instead of proposing a duplicate."
                    ),
                )
        return ValidationResult(ValidationStatus.APPROVED)


class ReasonableExpense(SymbolicValidator):
    """
    Expense memories must include a sane dollar amount.

    Catches LLM hallucinations like "team lunch cost $50,000".
    """

    def verify(self, text, metadata, kg_snapshot):
        amount = (metadata or {}).get("amount_usd")
        if amount is None:
            return ValidationResult(ValidationStatus.APPROVED)
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return ValidationResult(
                ValidationStatus.REJECTED,
                reason="amount_usd must be a number",
            )
        if amount < 0 or amount > 5_000:
            return ValidationResult(
                ValidationStatus.REJECTED,
                reason=f"${amount:,.0f} is outside the allowed expense range ($0–$5,000)",
            )
        return ValidationResult(ValidationStatus.APPROVED)


# ── Walkthrough ────────────────────────────────────────────────────────────────


def main() -> None:
    storage = tempfile.mkdtemp(prefix="epochdb_maya_")

    print(f"""
{BD}EpochDB — Neuro-symbolic memory for a personal assistant{R}
{DIM}{storage}{R}

Imagine an agent that talks to Maya and wants to *remember* things.
We use {BD}propose(){R} so rules can approve or reject before anything is saved.
{BD}remember(){R} stays available for raw notes that do not need those checks.
""")

    db = EpochDB(
        storage_dir=storage,
        dim=32,
        embedding_model=None,  # offline demo
        validators=[
            OnlyTrustedWriters(),
            NoSecretsInMemory(),
            OneCurrentEmployer(),
            ReasonableExpense(),
        ],
    )

    try:
        # ── Setup: known facts (bootstrapped without LCAG) ────────────────────
        section("Setup — seed Maya's profile with remember()")
        db.remember(
            "Maya works at Acme Corp.",
            metadata={
                "source": "hr-system",
                "triples": [("Maya", "works_at", "Acme Corp")],
            },
        )
        db.remember(
            "Maya is allergic to peanuts.",
            metadata={
                "source": "user",
                "triples": [("Maya", "allergic_to", "peanuts")],
            },
        )
        print("  Known facts: Maya works at Acme Corp · allergic to peanuts")
        print(f"  Graph entities: {sorted(db.get_kg_snapshot()['entities'])}")

        # ── 1. Good update from the calendar ──────────────────────────────────
        section("1. Calendar sync proposes a meeting — should APPROVE")
        print("  Agent: “Maya has a 1:1 with Jordan on Friday.”")
        show(
            "calendar → manager link",
            db.propose(
                "Maya has a weekly 1:1 with Jordan.",
                metadata={
                    "source": "calendar",
                    "triples": [("Maya", "reports_to", "Jordan")],
                },
            ),
        )

        # ── 2. Rumor from the open web ────────────────────────────────────────
        section("2. Unverified web rumor tries to change her job — should REJECT")
        print("  Agent (from a blog): “Maya just joined Globex as VP.”")
        show(
            "web-scrape → new employer",
            db.propose(
                "Maya just joined Globex as VP of Engineering.",
                metadata={
                    "source": "web-scrape",
                    "triples": [("Maya", "works_at", "Globex")],
                },
            ),
        )

        # ── 3. Same rumor, but pretending to be HR ────────────────────────────
        section("3. Even from HR, a second workplace conflicts with the graph — REJECT")
        print("  Agent: “HR says Maya works at Globex now.”")
        print("  Rule: one current employer — don't invent a parallel job.")
        show(
            "hr-system → second works_at",
            db.propose(
                "Maya works at Globex.",
                metadata={
                    "source": "hr-system",
                    "triples": [("Maya", "works_at", "Globex")],
                },
            ),
        )

        # ── 4. Secrets must never be remembered ───────────────────────────────
        section("4. Agent tries to store a password — should REJECT")
        print("  Agent: “I'll remember Maya's Wi‑Fi password for next time.”")
        show(
            "user → password text",
            db.propose(
                "Maya's home Wi-Fi password is: hunter2",
                metadata={"source": "user"},
            ),
        )

        # ── 5. Hallucinated expense ───────────────────────────────────────────
        section("5. LLM invents a huge lunch bill — should REJECT")
        print("  Agent: “Log $50,000 for the team lunch.”")
        show(
            "user → absurd expense",
            db.propose(
                "Team lunch at Noon Diner cost $50000.",
                metadata={"source": "user", "amount_usd": 50_000},
            ),
        )

        # ── 6. Normal expense ─────────────────────────────────────────────────
        section("6. Normal coffee expense — should APPROVE")
        print("  Agent: “Maya spent $6.50 on coffee.”")
        show(
            "user → coffee $6.50",
            db.propose(
                "Maya spent $6.50 on coffee at Beacon Roasters.",
                metadata={"source": "user", "amount_usd": 6.50},
            ),
        )

        # ── 7. New person is fine ─────────────────────────────────────────────
        section("7. Add a new colleague (no conflict) — should APPROVE")
        print("  Agent: “Priya joined Acme on Maya's team.”")
        show(
            "hr-system → Priya",
            db.propose(
                "Priya joined Acme Corp on Maya's team.",
                metadata={
                    "source": "hr-system",
                    "triples": [("Priya", "works_at", "Acme Corp")],
                },
            ),
        )

        # ── 8. remember() for chat logs (no LCAG) ─────────────────────────────
        section("8. Raw chat log via remember() — no validators (by design)")
        mid = db.remember(
            "User said: 'lol maybe I should quit and join Globex' (joke, not a fact)",
            metadata={"source": "web-scrape"},  # would fail propose(); remember ignores rules
        )
        print(f"  {GN}saved chat note{R} without validation → {mid[:8]}…")
        print("  Use remember() for transcripts; use propose() for durable profile facts.")

        # ── Recap ─────────────────────────────────────────────────────────────
        section("What stuck in long-term memory?")
        snap = db.get_kg_snapshot()
        print(f"  People / places in the graph: {sorted(snap['entities'])}")
        print(f"  Relationships tracked:        {sorted(snap['predicates'])}")
        print(f"  Staging leftovers:            {len(db.hot_tier.staging)} (always 0 after propose)")
        print(f"""
  {YL}Takeaway{R}
    • {BD}propose(){R}  = “ask to change the world model” (rules can say no)
    • {BD}remember(){R} = “just store this” (fast path, no LCAG)
    • Rejected intents never touch WAL, HNSW, or the knowledge graph
""")
    finally:
        db.close()
        shutil.rmtree(storage, ignore_errors=True)
        print(f"{DIM}cleaned up demo storage{R}\n")


if __name__ == "__main__":
    main()
