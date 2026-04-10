import numpy as np
from typing import List, Dict, Set
from .atom import UnifiedMemoryAtom
from .hot_tier import HotTier
from .cold_tier import ColdTier
import logging

logger = logging.getLogger(__name__)

class RetrievalManager:
    """Multi-stage retrieval process with Global KG."""
    def __init__(self, hot_tier: HotTier, cold_tier: ColdTier, global_kg: Dict[str, List[List[str]]]):
        self.hot_tier = hot_tier
        self.cold_tier = cold_tier
        self.global_kg = global_kg
        
    def _fetch_atom_by_id(self, atom_id: str, epoch_id: str) -> UnifiedMemoryAtom:
        # Check Hot Tier
        if atom_id in self.hot_tier.atoms:
            return self.hot_tier.atoms[atom_id]
        
        # Check Cold Tier targeted by epoch
        atoms = self.cold_tier.load_epoch(epoch_id)
        for a in atoms:
            if a.id == atom_id:
                return a
        return None

    def search(self, query_emb: np.ndarray, top_k: int = 5, expand_hops: int = 1, query_entities: List[str] = None) -> List[UnifiedMemoryAtom]:
        query_entities = set(query_entities) if query_entities else set()
        # Use Tuples to track (atom, semantic_similarity) for RRF
        candidates: Dict[str, tuple] = {}

        # 1. Semantic Hook (Hot Tier)
        hot_hits = self.hot_tier.query_vector(query_emb, top_k=top_k * 2)
        for atom in hot_hits:
            if len(atom.embedding) == len(query_emb):
                score = np.dot(atom.embedding, query_emb) / (np.linalg.norm(atom.embedding) * np.linalg.norm(query_emb) + 1e-10)
            else:
                score = 0.0
            candidates[atom.id] = (atom, float(score))

        # Cold tier Semantic Hook
        epochs = self.cold_tier.get_all_epochs()
        for epoch in epochs:
            cold_atoms = self.cold_tier.load_epoch(epoch)
            if cold_atoms:
                embeddings = np.array([a.embedding for a in cold_atoms])
                norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
                norms = np.where(norms == 0, 1e-10, norms)
                dots = np.dot(embeddings, query_emb)
                sims = dots / norms
                
                best_idx = np.argsort(sims)[-top_k * 5:][::-1]
                for idx in best_idx:
                    if sims[idx] > 0.1: # Minimum threshold for cold tier
                        atom = cold_atoms[idx]
                        if len(atom.embedding) != len(query_emb):
                            continue
                        candidates[atom.id] = (atom, float(sims[idx]))

        # 2. Relational Expansion (Global KG)
        if expand_hops > 0:
            expansion_set = set(candidates.keys())
            for _ in range(expand_hops):
                new_neighbors = set()
                for a_id in expansion_set:
                    atom_data = candidates.get(a_id)
                    if not atom_data: continue
                    atom = atom_data[0]
                    entities = set()
                    for subj, pred, obj in atom.triples:
                        entities.add(subj)
                        entities.add(obj)
                    
                    for ent in entities:
                        if ent in self.global_kg:
                            for neighbor_atom_id, epoch_id in self.global_kg[ent]:
                                if neighbor_atom_id not in candidates:
                                    n_atom = self._fetch_atom_by_id(neighbor_atom_id, epoch_id)
                                    if n_atom and len(n_atom.embedding) == len(query_emb):
                                        score = np.dot(n_atom.embedding, query_emb) / (np.linalg.norm(n_atom.embedding) * np.linalg.norm(query_emb) + 1e-10)
                                        new_neighbors.add(n_atom.id)
                                        candidates[n_atom.id] = (n_atom, float(score))
                expansion_set = new_neighbors

        # 3. Hybrid Ranking (3-Way RRF) and Payload Deduplication
        all_candidates = list(candidates.values())
        unique_results = []
        seen_payloads = set()
        
        for atom, sim in all_candidates:
            payload_key = str(atom.payload)
            if payload_key not in seen_payloads:
                unique_results.append((atom, sim))
                seen_payloads.add(payload_key)

        if not unique_results:
            return []

        # -- RRF FACTORS --
        
        # Factor A: Semantic Similarity Rank
        unique_results.sort(key=lambda x: x[1], reverse=True)
        sem_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        # Factor B: Recency Rank (Freshness)
        unique_results.sort(key=lambda x: x[0].created_at, reverse=True)
        recency_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        # Factor C: Entity Overlap Rank
        def get_overlap(atom):
            atom_entities = set()
            for s, p, o in atom.triples:
                atom_entities.add(s); atom_entities.add(o)
            return len(query_entities.intersection(atom_entities))
        
        unique_results.sort(key=lambda x: get_overlap(x[0]), reverse=True)
        entity_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        # Execute 3-Way Reciprocal Rank Fusion
        K = 60
        def multi_rrf_score(atom_id):
            score = (1.0 / (K + sem_ranks[atom_id]))
            score += (1.0 / (K + recency_ranks[atom_id]))
            score += (1.0 / (K + entity_ranks[atom_id]))
            return score

        unique_results.sort(key=lambda x: multi_rrf_score(x[0].id), reverse=True)

        final_atoms = []
        for atom, _ in unique_results[:top_k]:
            atom.access_count += 1
            final_atoms.append(atom)

        return final_atoms
