"""Unit tests for neuro-symbolic two-phase propose() / staging validation."""

import asyncio
import os
import shutil
import unittest

from epochdb import (
    AsyncEpochDB,
    EpochDB,
    SymbolicValidator,
    ValidationResult,
    ValidationStatus,
)
from epochdb.storage.hot_tier import HotTier
from epochdb.validation.symbolic import ValidationResult as VR


# ---------------------------------------------------------------------------
# Mock / fixture validators
# ---------------------------------------------------------------------------


class RequireProjectTag(SymbolicValidator):
    """Mock business rule: metadata must include project == 'EpochDB'."""

    def verify(self, text, metadata, kg_snapshot):
        if (metadata or {}).get("project") != "EpochDB":
            return ValidationResult(
                ValidationStatus.REJECTED,
                reason="metadata.project must equal 'EpochDB'",
            )
        return ValidationResult(ValidationStatus.APPROVED)


class RejectForbiddenSubstring(SymbolicValidator):
    """Mock business rule: reject payloads containing a banned token."""

    def verify(self, text, metadata, kg_snapshot):
        if "DROP_ALL" in (text or ""):
            return ValidationResult(
                ValidationStatus.REJECTED,
                reason="payload contains forbidden token DROP_ALL",
            )
        _ = kg_snapshot.get("entities")
        return ValidationResult(ValidationStatus.APPROVED)


class TrackingValidator(SymbolicValidator):
    """Records calls so tests can assert short-circuit / invocation order."""

    def __init__(self, name: str, decision: ValidationStatus, reason: str = None):
        self.name = name
        self.decision = decision
        self.reason = reason
        self.calls = []

    def verify(self, text, metadata, kg_snapshot):
        self.calls.append({"text": text, "metadata": dict(metadata or {}), "kg": kg_snapshot})
        return ValidationResult(self.decision, reason=self.reason)


class UniqueEntityValidator(SymbolicValidator):
    """Reject if subject entity already exists in the KG snapshot."""

    def verify(self, text, metadata, kg_snapshot):
        triples = (metadata or {}).get("triples") or []
        entities = set(kg_snapshot.get("entities") or [])
        for triple in triples:
            if not triple:
                continue
            subj = str(triple[0])
            if subj in entities:
                return ValidationResult(
                    ValidationStatus.REJECTED,
                    reason=f"entity '{subj}' already exists in KG",
                )
        return ValidationResult(ValidationStatus.APPROVED)


class RequireNonEmptyText(SymbolicValidator):
    def verify(self, text, metadata, kg_snapshot):
        if not (text or "").strip():
            return ValidationResult(
                ValidationStatus.REJECTED,
                reason="text must be non-empty",
            )
        return ValidationResult(ValidationStatus.APPROVED)


class RejectWithoutReason(SymbolicValidator):
    """REJECTED with reason=None → propose should still return a structured error."""

    def verify(self, text, metadata, kg_snapshot):
        return ValidationResult(ValidationStatus.REJECTED)


class PredicateAllowlist(SymbolicValidator):
    """Only allow predicates present in an allowlist (uses kg_snapshot + metadata)."""

    def __init__(self, allowed):
        self.allowed = set(allowed)

    def verify(self, text, metadata, kg_snapshot):
        triples = (metadata or {}).get("triples") or []
        for t in triples:
            if len(t) >= 2 and t[1] not in self.allowed:
                return ValidationResult(
                    ValidationStatus.REJECTED,
                    reason=f"predicate '{t[1]}' not in allowlist",
                )
        return ValidationResult(ValidationStatus.APPROVED)


# ---------------------------------------------------------------------------
# Core propose scenarios
# ---------------------------------------------------------------------------


class TestProposeTwoPhaseCommit(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./.test_propose_lcag"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.db = None

    def tearDown(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _db(self, validators=None, **kwargs):
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=32,
            embedding_model=None,
            validators=validators,
            **kwargs,
        )
        return self.db

    def test_propose_approved(self):
        db = self._db(validators=[RequireProjectTag(), RejectForbiddenSubstring()])
        result = db.propose(
            "Implemented symbolic staging area",
            metadata={"project": "EpochDB", "author": "test"},
        )
        self.assertEqual(result["status"], "APPROVED")
        self.assertIn("memory_id", result)

        mid = result["memory_id"]
        self.assertNotIn(mid, db.hot_tier.staging)
        self.assertIn(mid, db.hot_tier.atoms)

        mem = db.get(mid)
        self.assertIsNotNone(mem)
        self.assertEqual(mem.text, "Implemented symbolic staging area")
        self.assertEqual(mem.metadata["project"], "EpochDB")

    def test_propose_rejected_discards_staging(self):
        db = self._db(validators=[RequireProjectTag(), RejectForbiddenSubstring()])
        before_atoms = len(db.hot_tier.atoms)

        result = db.propose(
            "Please DROP_ALL production state",
            metadata={"project": "EpochDB"},
        )
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("DROP_ALL", result["error"])
        self.assertNotIn("memory_id", result)

        self.assertEqual(len(db.hot_tier.staging), 0)
        self.assertEqual(len(db.hot_tier.atoms), before_atoms)

    def test_propose_rejected_on_metadata_rule(self):
        db = self._db(validators=[RequireProjectTag()])
        result = db.propose("Valid text", metadata={"project": "Other"})
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("project", result["error"])
        self.assertEqual(len(db.hot_tier.staging), 0)
        self.assertEqual(len(db.hot_tier.atoms), 0)

    def test_propose_without_validators_bypasses_validation(self):
        db = self._db()
        self.assertEqual(db.validators, [])

        result = db.propose(
            "Unvalidated but committed via propose",
            metadata={"project": "Anything"},
        )
        self.assertEqual(result["status"], "APPROVED")
        mid = result["memory_id"]
        self.assertIn(mid, db.hot_tier.atoms)
        self.assertIsNotNone(db.get(mid))

    def test_remember_unaffected_by_validators(self):
        db = self._db(validators=[RejectForbiddenSubstring()])
        mid = db.remember(
            "Contains DROP_ALL but remember bypasses validators",
            metadata={"project": "x"},
        )
        self.assertIsNotNone(db.get(mid))
        self.assertIn(mid, db.hot_tier.atoms)

    def test_validators_none_same_as_empty(self):
        db = self._db(validators=None)
        self.assertEqual(db.validators, [])
        result = db.propose("ok")
        self.assertEqual(result["status"], "APPROVED")

    def test_short_circuit_skips_later_validators(self):
        first = TrackingValidator("first", ValidationStatus.REJECTED, reason="first failed")
        second = TrackingValidator("second", ValidationStatus.APPROVED)
        db = self._db(validators=[first, second])

        result = db.propose("anything", metadata={"project": "EpochDB"})
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["error"], "first failed")
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 0)
        self.assertEqual(len(db.hot_tier.staging), 0)
        self.assertEqual(len(db.hot_tier.atoms), 0)

    def test_all_validators_invoked_on_approval(self):
        a = TrackingValidator("a", ValidationStatus.APPROVED)
        b = TrackingValidator("b", ValidationStatus.APPROVED)
        c = TrackingValidator("c", ValidationStatus.APPROVED)
        db = self._db(validators=[a, b, c])

        result = db.propose("chain ok", metadata={"k": 1})
        self.assertEqual(result["status"], "APPROVED")
        self.assertEqual(len(a.calls), 1)
        self.assertEqual(len(b.calls), 1)
        self.assertEqual(len(c.calls), 1)

    def test_second_validator_rejects_after_first_approves(self):
        a = TrackingValidator("a", ValidationStatus.APPROVED)
        b = TrackingValidator("b", ValidationStatus.REJECTED, reason="second says no")
        db = self._db(validators=[a, b])

        result = db.propose("partial", metadata={})
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["error"], "second says no")
        self.assertEqual(len(a.calls), 1)
        self.assertEqual(len(b.calls), 1)
        self.assertEqual(len(db.hot_tier.atoms), 0)

    def test_rejected_without_reason_uses_fallback_error(self):
        db = self._db(validators=[RejectWithoutReason()])
        result = db.propose("x")
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["error"], "Validation failed")

    def test_empty_text_rejected(self):
        db = self._db(validators=[RequireNonEmptyText()])
        result = db.propose("   ", metadata={})
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("non-empty", result["error"])

    def test_sequential_approve_reject_approve(self):
        db = self._db(validators=[RequireProjectTag()])

        r1 = db.propose("one", metadata={"project": "EpochDB"})
        self.assertEqual(r1["status"], "APPROVED")

        r2 = db.propose("two", metadata={"project": "Wrong"})
        self.assertEqual(r2["status"], "REJECTED")

        r3 = db.propose("three", metadata={"project": "EpochDB"})
        self.assertEqual(r3["status"], "APPROVED")

        self.assertEqual(len(db.hot_tier.atoms), 2)
        self.assertEqual(len(db.hot_tier.staging), 0)
        self.assertIsNotNone(db.get(r1["memory_id"]))
        self.assertIsNotNone(db.get(r3["memory_id"]))

    def test_propose_metadata_dict_as_triples_compat(self):
        """Same compat as remember: triples=dict + metadata=None → treat as metadata."""
        db = self._db(validators=[RequireProjectTag()])
        result = db.propose("compat", triples={"project": "EpochDB", "note": 1})
        self.assertEqual(result["status"], "APPROVED")
        mem = db.get(result["memory_id"])
        self.assertEqual(mem.metadata["project"], "EpochDB")
        self.assertEqual(mem.metadata["note"], 1)

    def test_propose_empty_metadata_defaults(self):
        db = self._db()
        result = db.propose("no meta")
        self.assertEqual(result["status"], "APPROVED")
        mem = db.get(result["memory_id"])
        self.assertIsInstance(mem.metadata, dict)


# ---------------------------------------------------------------------------
# KG-aware validation
# ---------------------------------------------------------------------------


class TestProposeKGIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./.test_propose_kg"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.db = None

    def tearDown(self):
        if self.db is not None:
            self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_kg_snapshot_includes_seeded_entities(self):
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=32,
            embedding_model=None,
            validators=[UniqueEntityValidator()],
        )
        self.db.remember(
            "Alice works at Acme",
            triples=[("Alice", "works_at", "Acme")],
        )
        snap = self.db.get_kg_snapshot()
        self.assertIn("Alice", snap["entities"])
        self.assertIn("Acme", snap["entities"])
        self.assertIn("works_at", snap["predicates"])
        self.assertEqual(snap["epoch_id"], self.db.current_epoch_id)

    def test_unique_entity_rule_rejects_duplicate(self):
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=32,
            embedding_model=None,
            validators=[UniqueEntityValidator()],
        )
        self.db.remember("seed", triples=[("Alice", "knows", "Bob")])

        rejected = self.db.propose(
            "Alice again",
            metadata={"triples": [("Alice", "likes", "Coffee")]},
        )
        self.assertEqual(rejected["status"], "REJECTED")
        self.assertIn("Alice", rejected["error"])
        self.assertEqual(len(self.db.hot_tier.staging), 0)

        # New entity should still be allowed
        approved = self.db.propose(
            "Carol intro",
            metadata={"triples": [("Carol", "likes", "Tea")]},
        )
        self.assertEqual(approved["status"], "APPROVED")
        self.assertIn("Carol", self.db.get_kg_snapshot()["entities"])

    def test_rejected_propose_does_not_mutate_kg(self):
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=32,
            embedding_model=None,
            validators=[RejectForbiddenSubstring()],
        )
        before = set(self.db.get_kg_snapshot()["entities"])
        result = self.db.propose(
            "DROP_ALL now",
            metadata={"triples": [("Evil", "destroys", "World")]},
        )
        self.assertEqual(result["status"], "REJECTED")
        after = set(self.db.get_kg_snapshot()["entities"])
        self.assertEqual(before, after)
        self.assertNotIn("Evil", after)

    def test_approved_propose_updates_kg_associations(self):
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=32,
            embedding_model=None,
            validators=[PredicateAllowlist({"reports_to", "co_occurs_with", "mentions"})],
        )
        result = self.db.propose(
            "Dana reports to Eve",
            metadata={"triples": [("Dana", "reports_to", "Eve")]},
        )
        self.assertEqual(result["status"], "APPROVED")
        mid = result["memory_id"]
        assoc = self.db.kg_manager.get_associations("Dana")
        atom_ids = {row[0] for row in assoc}
        self.assertIn(mid, atom_ids)
        self.assertIn("reports_to", self.db.predicates)

    def test_predicate_allowlist_rejects_unknown_edge(self):
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=32,
            embedding_model=None,
            validators=[PredicateAllowlist({"knows"})],
        )
        result = self.db.propose(
            "bad edge",
            metadata={"triples": [("A", "hacks", "B")]},
        )
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("hacks", result["error"])
        self.assertEqual(self.db.kg_manager.get_associations("A"), [])

    def test_kg_snapshot_pending_ids_empty_after_commit(self):
        tracker = TrackingValidator("t", ValidationStatus.APPROVED)
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=32,
            embedding_model=None,
            validators=[tracker],
        )
        self.db.propose("snap check")
        # During verify, pending_ids should have included the staged id
        self.assertEqual(len(tracker.calls), 1)
        pending_during = list(tracker.calls[0]["kg"]["pending_ids"])
        self.assertEqual(len(pending_during), 1)
        self.assertEqual(len(self.db.hot_tier.staging), 0)


# ---------------------------------------------------------------------------
# Hot-tier staging API (direct)
# ---------------------------------------------------------------------------


class TestHotTierStagingAPI(unittest.TestCase):
    def setUp(self):
        self.hot = HotTier(dim=8, max_elements=100)

    def test_stage_does_not_touch_atoms_or_hnsw(self):
        mid = self.hot.stage_memory("m1", "hello", metadata={"a": 1})
        self.assertEqual(mid, "m1")
        self.assertIn("m1", self.hot.staging)
        self.assertNotIn("m1", self.hot.atoms)
        self.assertEqual(self.hot.vector_index.get_current_count(), 0)
        staged = self.hot.get_staged_memory("m1")
        self.assertEqual(staged.status, "PENDING")
        self.assertEqual(staged.text, "hello")

    def test_discard_staged_memory(self):
        self.hot.stage_memory("m1", "bye")
        self.assertTrue(self.hot.discard_staged_memory("m1"))
        self.assertFalse(self.hot.discard_staged_memory("m1"))
        self.assertIsNone(self.hot.get_staged_memory("m1"))

    def test_commit_staged_promotes_to_atoms(self):
        import numpy as np

        emb = np.zeros(8, dtype=np.float32)
        emb[0] = 1.0
        self.hot.stage_memory("m1", "promoted", embedding=emb, triples=[("X", "y", "Z")])
        committed = self.hot.commit_staged_memory("m1")
        self.assertEqual(committed, "m1")
        self.assertNotIn("m1", self.hot.staging)
        self.assertIn("m1", self.hot.atoms)
        self.assertEqual(self.hot.atoms["m1"].payload, "promoted")
        self.assertGreaterEqual(self.hot.vector_index.get_current_count(), 1)

    def test_commit_missing_returns_none(self):
        self.assertIsNone(self.hot.commit_staged_memory("does-not-exist"))

    def test_clear_empties_staging(self):
        self.hot.stage_memory("a", "1")
        self.hot.stage_memory("b", "2")
        self.hot.clear()
        self.assertEqual(len(self.hot.staging), 0)
        self.assertEqual(len(self.hot.atoms), 0)

    def test_engine_commit_staged_missing_returns_none(self):
        path = "./.test_propose_commit_missing"
        if os.path.exists(path):
            shutil.rmtree(path)
        db = EpochDB(storage_dir=path, dim=8, embedding_model=None)
        try:
            self.assertIsNone(db.commit_staged_memory("nope"))
        finally:
            db.close()
            shutil.rmtree(path)


# ---------------------------------------------------------------------------
# Isolation: staged content not queryable until commit
# ---------------------------------------------------------------------------


class TestStagingIsolation(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./.test_propose_isolation"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_rejected_memory_not_retrievable(self):
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=32,
            embedding_model=None,
            validators=[RejectForbiddenSubstring()],
        )
        self.db.propose("DROP_ALL secret payload xyzzy", metadata={"project": "EpochDB"})
        # Direct get by scanning atoms — nothing should hold the secret text
        for atom in self.db.hot_tier.atoms.values():
            self.assertNotIn("xyzzy", str(atom.payload))
        self.assertIsNone(self.db.get("not-a-real-id"))

    def test_approved_memory_queryable_via_get(self):
        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=32,
            embedding_model=None,
            validators=[RequireProjectTag()],
        )
        r = self.db.propose("visible after commit", metadata={"project": "EpochDB"})
        mem = self.db.get(r["memory_id"])
        self.assertEqual(mem.text, "visible after commit")


# ---------------------------------------------------------------------------
# Async facade
# ---------------------------------------------------------------------------


class TestAsyncPropose(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./.test_propose_async"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_async_propose_approve_and_reject(self):
        async def _run():
            db = AsyncEpochDB(
                storage_dir=self.test_dir,
                dim=32,
                embedding_model=None,
                validators=[RequireProjectTag()],
            )
            try:
                ok = await db.propose("async ok", metadata={"project": "EpochDB"})
                self.assertEqual(ok["status"], "APPROVED")
                bad = await db.propose("async bad", metadata={"project": "nope"})
                self.assertEqual(bad["status"], "REJECTED")
            finally:
                await db.__aexit__(None, None, None)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# ValidationResult / interface contract
# ---------------------------------------------------------------------------


class TestValidationInterface(unittest.TestCase):
    def test_validation_result_frozen(self):
        r = ValidationResult(ValidationStatus.APPROVED, reason="ok")
        with self.assertRaises(Exception):
            r.status = ValidationStatus.REJECTED  # type: ignore[misc]

    def test_symbolic_validator_is_abstract(self):
        with self.assertRaises(TypeError):
            SymbolicValidator()  # type: ignore[abstract]

    def test_status_enum_values(self):
        self.assertEqual(ValidationStatus.PENDING.value, "PENDING")
        self.assertEqual(ValidationStatus.APPROVED.value, "APPROVED")
        self.assertEqual(ValidationStatus.REJECTED.value, "REJECTED")

    def test_reexport_matches_module(self):
        self.assertIs(ValidationResult, VR)


# ---------------------------------------------------------------------------
# Validator receives expected kg_snapshot shape
# ---------------------------------------------------------------------------


class TestKgSnapshotShape(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./.test_propose_snap_shape"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_snapshot_keys_and_types(self):
        seen = {}

        class Capture(SymbolicValidator):
            def verify(self, text, metadata, kg_snapshot):
                seen.update(kg_snapshot)
                return ValidationResult(ValidationStatus.APPROVED)

        self.db = EpochDB(
            storage_dir=self.test_dir,
            dim=16,
            embedding_model=None,
            validators=[Capture()],
        )
        self.db.propose("capture snap")
        self.assertIn("entities", seen)
        self.assertIn("predicates", seen)
        self.assertIn("epoch_id", seen)
        self.assertIn("hot_atom_ids", seen)
        self.assertIn("pending_ids", seen)
        self.assertIsInstance(seen["entities"], list)
        self.assertIsInstance(seen["predicates"], frozenset)
        self.assertIsInstance(seen["epoch_id"], str)
        self.assertIsInstance(seen["hot_atom_ids"], tuple)
        self.assertIsInstance(seen["pending_ids"], tuple)


if __name__ == "__main__":
    unittest.main()
