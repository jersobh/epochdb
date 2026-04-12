import numpy as np
from typing import List, Dict, Set
from .atom import UnifiedMemoryAtom
from .hot_tier import HotTier
from .cold_tier import ColdTier
import logging

logger = logging.getLogger(__name__)


class RetrievalManager:
    """Multi-stage retrieval with Global KG and persistent access-count tracking."""

    def __init__(
        self,
        hot_tier: HotTier,
        cold_tier: ColdTier,
        global_kg: Dict[str, List[List[str]]],
    ):
        self.hot_tier = hot_tier
        self.cold_tier = cold_tier
        self.global_kg = global_kg

        # Track access-count increments for cold-tier atoms in memory.
        # These are applied on top of the stored access_count when an atom
        # is loaded from Parquet. Cleared on engine close (not persisted across
        # process restarts — acceptable since counts are a ranking signal only).
        self._access_deltas: Dict[str, int] = {}

    def _fetch_atom_by_id(self, atom_id: str, epoch_id: str) -> UnifiedMemoryAtom:
        # Check Hot Tier first.
        if atom_id in self.hot_tier.atoms:
            return self.hot_tier.atoms[atom_id]

        # Check Cold Tier, targeted by epoch.
        atoms = self.cold_tier.load_epoch(epoch_id)
        for a in atoms:
            if a.id == atom_id:
                # Apply any in-memory access delta accumulated since last checkpoint.
                a.access_count += self._access_deltas.get(atom_id, 0)
                return a
        return None

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        expand_hops: int = 1,
        query_entities: List[str] = None,
    ) -> List[UnifiedMemoryAtom]:
        query_entities = set(query_entities) if query_entities else set()
        # Candidates: {atom_id: (atom, semantic_similarity)}
        candidates: Dict[str, tuple] = {}

        # --- 1. Semantic Hook: Hot Tier ---
        hot_hits = self.hot_tier.query_vector(query_emb, top_k=top_k * 2)
        for atom in hot_hits:
            if len(atom.embedding) == len(query_emb):
                score = np.dot(atom.embedding, query_emb) / (
                    np.linalg.norm(atom.embedding) * np.linalg.norm(query_emb) + 1e-10
                )
            else:
                score = 0.0
            candidates[atom.id] = (atom, float(score))

        # --- 1b. Semantic Hook: Cold Tier ---
        epochs = self.cold_tier.get_all_epochs()
        for epoch in epochs:
            cold_atoms = self.cold_tier.load_epoch(epoch)
            if not cold_atoms:
                continue

            embeddings = np.array([a.embedding for a in cold_atoms])
            norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
            norms = np.where(norms == 0, 1e-10, norms)
            sims = np.dot(embeddings, query_emb) / norms

            best_idx = np.argsort(sims)[-(top_k * 5):][::-1]
            for idx in best_idx:
                if sims[idx] > 0.1:
                    atom = cold_atoms[idx]
                    if len(atom.embedding) != len(query_emb):
                        continue
                    # Apply accumulated access-count delta for this cold atom.
                    atom.access_count += self._access_deltas.get(atom.id, 0)
                    candidates[atom.id] = (atom, float(sims[idx]))

        # --- 2. Relational Expansion via Global KG ---
        if expand_hops > 0:
            expansion_set = set(candidates.keys())
            for _ in range(expand_hops):
                new_neighbors: Set[str] = set()
                for a_id in expansion_set:
                    atom_data = candidates.get(a_id)
                    if not atom_data:
                        continue
                    atom = atom_data[0]
                    entities = set()
                    for subj, pred, obj in atom.triples:
                        entities.add(subj)
                        entities.add(obj)

                    for ent in entities:
                        if ent in self.global_kg:
                            for neighbor_atom_id, epoch_id in self.global_kg[ent]:
                                if neighbor_atom_id not in candidates:
                                    n_atom = self._fetch_atom_by_id(
                                        neighbor_atom_id, epoch_id
                                    )
                                    if n_atom and len(n_atom.embedding) == len(query_emb):
                                        score = np.dot(
                                            n_atom.embedding, query_emb
                                        ) / (
                                            np.linalg.norm(n_atom.embedding)
                                            * np.linalg.norm(query_emb)
                                            + 1e-10
                                        )
                                        new_neighbors.add(n_atom.id)
                                        candidates[n_atom.id] = (n_atom, float(score))
                expansion_set = new_neighbors

        # --- 3. Payload Deduplication ---
        all_candidates = list(candidates.values())
        unique_results = []
        seen_payloads: Set[str] = set()

        for atom, sim in all_candidates:
            payload_key = str(atom.payload)
            if payload_key not in seen_payloads:
                unique_results.append((atom, sim))
                seen_payloads.add(payload_key)

        if not unique_results:
            return []

        # --- 4. 3-Way Reciprocal Rank Fusion ---

        # Factor A: Semantic Similarity
        unique_results.sort(key=lambda x: x[1], reverse=True)
        sem_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        # Factor B: Recency (Freshness)
        unique_results.sort(key=lambda x: x[0].created_at, reverse=True)
        recency_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        # Factor C: Entity Overlap with Query
        def get_overlap(atom: UnifiedMemoryAtom) -> int:
            atom_entities = set()
            for s, _, o in atom.triples:
                atom_entities.add(s)
                atom_entities.add(o)
            return len(query_entities.intersection(atom_entities))

        unique_results.sort(key=lambda x: get_overlap(x[0]), reverse=True)
        entity_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        K = 60

        def multi_rrf_score(atom_id: str) -> float:
            return (
                1.0 / (K + sem_ranks[atom_id])
                + 1.0 / (K + recency_ranks[atom_id])
                + 1.0 / (K + entity_ranks[atom_id])
            )

        unique_results.sort(key=lambda x: multi_rrf_score(x[0].id), reverse=True)

        # --- 5. Materialise Results & Update Access Counts ---
        final_atoms = []
        for atom, _ in unique_results[:top_k]:
            atom.access_count += 1
            # Track delta for cold-tier atoms so subsequent loads reflect it.
            if atom.id not in self.hot_tier.atoms:
                self._access_deltas[atom.id] = (
                    self._access_deltas.get(atom.id, 0) + 1
                )
            final_atoms.append(atom)

        return final_atoms
