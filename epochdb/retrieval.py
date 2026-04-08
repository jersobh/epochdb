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
        # epoch_id might be "epoch_X" but load_epoch expects X
        epoch_str = epoch_id.replace("epoch_", "")
        atoms = self.cold_tier.load_epoch(epoch_str)
        for a in atoms:
            if a.id == atom_id:
                return a
        return None

    def search(self, query_emb: np.ndarray, top_k: int = 5, expand_hops: int = 1) -> List[UnifiedMemoryAtom]:
        candidates: Dict[str, UnifiedMemoryAtom] = {}

        # 1. Semantic Hook (Hot Tier)
        hot_hits = self.hot_tier.query_vector(query_emb, top_k=top_k)
        for atom in hot_hits:
            candidates[atom.id] = atom

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
                
                best_idx = np.argsort(sims)[-top_k:][::-1]
                for idx in best_idx:
                    if sims[idx] > 0.0:
                        atom = cold_atoms[idx]
                        if len(atom.embedding) != len(query_emb):
                            logger.warning(f"Skipping atom {atom.id} due to dim mismatch ({len(atom.embedding)} vs {len(query_emb)})")
                            continue
                        candidates[atom.id] = atom

        # 2. Relational Expansion (Global KG)
        if expand_hops > 0:
            expansion_set = set(candidates.keys())
            for _ in range(expand_hops):
                new_neighbors = set()
                # Step B: Identify Entities
                for a_id in expansion_set:
                    atom = candidates.get(a_id)
                    if not atom: continue
                    entities = set()
                    for subj, pred, obj in atom.triples:
                        entities.add(subj)
                        entities.add(obj)
                    
                    # Step C: Query KG for neighbors
                    for ent in entities:
                        if ent in self.global_kg:
                            for neighbor_atom_id, epoch_id in self.global_kg[ent]:
                                if neighbor_atom_id not in candidates:
                                    # Step D: Targeted Fetch
                                    n_atom = self._fetch_atom_by_id(neighbor_atom_id, epoch_id)
                                    if n_atom:
                                        new_neighbors.add(n_atom.id)
                                        candidates[n_atom.id] = n_atom
                expansion_set = new_neighbors

        # 3. Temporal Re-ranking and Payload Deduplication
        all_atoms = list(candidates.values())
        all_atoms.sort(key=lambda x: x.calculate_saliency(), reverse=True)

        unique_results: List[UnifiedMemoryAtom] = []
        seen_payloads = set()
        
        for atom in all_atoms:
            # We use a simple string representation for deduplication
            payload_key = str(atom.payload)
            if payload_key not in seen_payloads:
                atom.access_count += 1
                unique_results.append(atom)
                seen_payloads.add(payload_key)

        return unique_results[:top_k * 2]
