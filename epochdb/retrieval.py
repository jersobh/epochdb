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
        # Use Tuples to track (atom, semantic_similarity) for RRF
        candidates: Dict[str, tuple] = {}

        # 1. Semantic Hook (Hot Tier)
        hot_hits = self.hot_tier.query_vector(query_emb, top_k=top_k)
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
                
                best_idx = np.argsort(sims)[-top_k:][::-1]
                for idx in best_idx:
                    if sims[idx] > 0.0:
                        atom = cold_atoms[idx]
                        if len(atom.embedding) != len(query_emb):
                            logger.warning(f"Skipping atom {atom.id} due to dim mismatch ({len(atom.embedding)} vs {len(query_emb)})")
                            continue
                        candidates[atom.id] = (atom, float(sims[idx]))

        # 2. Relational Expansion (Global KG)
        if expand_hops > 0:
            expansion_set = set(candidates.keys())
            for _ in range(expand_hops):
                new_neighbors = set()
                # Step B: Identify Entities
                for a_id in expansion_set:
                    atom_data = candidates.get(a_id)
                    if not atom_data: continue
                    atom = atom_data[0]
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
                                        if len(n_atom.embedding) == len(query_emb):
                                            score = np.dot(n_atom.embedding, query_emb) / (np.linalg.norm(n_atom.embedding) * np.linalg.norm(query_emb) + 1e-10)
                                        else:
                                            score = 0.0
                                        
                                        # SMART FILTERING: Only expand to this neighbor if semantic score > threshold
                                        if score > 0.15:
                                            new_neighbors.add(n_atom.id)
                                            candidates[n_atom.id] = (n_atom, float(score))
                expansion_set = new_neighbors

        # 3. Hybrid Ranking (RRF) and Payload Deduplication
        all_atoms_with_sim = list(candidates.values())
        unique_results = []
        seen_payloads = set()
        
        for atom, sim in all_atoms_with_sim:
            payload_key = str(atom.payload)
            if payload_key not in seen_payloads:
                unique_results.append((atom, sim))
                seen_payloads.add(payload_key)

        # Rank by Semantic Similarity
        unique_results.sort(key=lambda x: x[1], reverse=True)
        sim_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        # Rank by Saliency
        unique_results.sort(key=lambda x: x[0].calculate_saliency(), reverse=True)
        saliency_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        # Execute Reciprocal Rank Fusion
        K = 60
        def rrf_score(atom_id):
            return (1.0 / (K + sim_ranks[atom_id])) + (1.0 / (K + saliency_ranks[atom_id]))

        unique_results.sort(key=lambda x: rrf_score(x[0].id), reverse=True)

        final_atoms = []
        for atom, _ in unique_results[:top_k * 2]:
            atom.access_count += 1
            final_atoms.append(atom)

        return final_atoms
