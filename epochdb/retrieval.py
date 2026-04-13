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

        # Check Cold Tier using targeted metadata lookup
        atoms = self.cold_tier.load_atom_metadata(epoch_id, [atom_id])
        if atoms:
            a = atoms[0]
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
        # Freeze the original query intent before graph expansion contaminates it.
        # get_topic_boost MUST score against this, not the expanded set.
        original_query_entities = frozenset(query_entities)
        # Candidates: {atom_id: (atom, semantic_similarity)}
        candidates: Dict[str, tuple] = {}

        # --- 1. Semantic Hook: Hot Tier ---
        # We fetch a larger pool to allow RRF and Topic Locking to function.
        hot_hits = self.hot_tier.query_vector(query_emb, top_k=top_k * 10)
        for atom in hot_hits:
            if len(atom.embedding) == len(query_emb):
                score = np.dot(atom.embedding, query_emb) / (
                    np.linalg.norm(atom.embedding) * np.linalg.norm(query_emb) + 1e-10
                )
            else:
                score = 0.0
            candidates[atom.id] = (atom, float(score))

        # --- Semantic Bootstrapping ---
        # If no query_entities provided, we 'bootstrap' them from the top semantic matches
        # in the Hot Tier. This allows vector-only queries to still leverage the KG 
        # locking mechanisms.
        if not query_entities and hot_hits:
            for atom in hot_hits[:2]:  # Use only high-confidence hits
                for s, p, o in atom.triples:
                    query_entities.add(s)
                    query_entities.add(o)

        # --- 1a. Entity Hook: Global KG Seeding ---
        # If query entities match Global KG entries, we pull them in as candidates 
        # even if their semantic score was too low to make the initial pool.
        for qe in query_entities:
            if qe in self.global_kg:
                # Group neighbor atoms by epoch for optimized loading
                epoch_to_atom_ids: Dict[str, List[str]] = {}
                for a_id, ep_id in self.global_kg[qe]:
                    if a_id not in candidates:
                        if ep_id not in epoch_to_atom_ids:
                            epoch_to_atom_ids[ep_id] = []
                        epoch_to_atom_ids[ep_id].append(a_id)
                
                # Fetch from Cold Tier if needed
                for ep_id, a_ids in epoch_to_atom_ids.items():
                    atoms = self.cold_tier.load_atom_metadata(ep_id, a_ids)
                    for a in atoms:
                        sim = np.dot(a.embedding, query_emb) / (
                            np.linalg.norm(a.embedding) * np.linalg.norm(query_emb) + 1e-10
                        )
                        candidates[a.id] = (a, float(sim))
                
                # Also check Hot Tier
                for a_id, _ in self.global_kg[qe]:
                    if a_id in self.hot_tier.atoms and a_id not in candidates:
                        a = self.hot_tier.atoms[a_id]
                        sim = np.dot(a.embedding, query_emb) / (
                            np.linalg.norm(a.embedding) * np.linalg.norm(query_emb) + 1e-10
                        )
                        candidates[a.id] = (a, float(sim))

        # --- Keyword-based Entity Extraction (Auto-Expansion) ---
        # If no explicit entities are passed, we scan the query embedding surface 
        # (or payload keywords) for matches in the Global KG to boost Factor C.
        if not query_entities:
            # We don't have the raw query text here, but we can use the Global KG 
            # keys as a candidate set for heuristic matching if the user passed 
            # an unpopulated set. For now, we rely on the expansion set below.
            pass

        # --- 1b. Semantic Hook: Cold Tier (Fast Indexed Search) ---
        epochs = self.cold_tier.get_all_epochs()
        for epoch in epochs:
            # We fetch a larger pool to ensure corrections are captured for RRF fusion.
            cold_hits = self.cold_tier.search_epoch(epoch, query_emb, top_k=top_k * 10)
            for atom in cold_hits:
                if len(atom.embedding) != len(query_emb):
                    continue
                    
                sim = np.dot(atom.embedding, query_emb) / (
                    np.linalg.norm(atom.embedding) * np.linalg.norm(query_emb) + 1e-10
                )
                atom.access_count += self._access_deltas.get(atom.id, 0)
                candidates[atom.id] = (atom, float(sim))

        # --- 2. Relational Expansion via Global KG ---
        # This doubles as our Entity Extraction: if a candidate atom mentions an entity 
        # that was also in our query (heuristically), it gets a Factor C boost.
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
                            # Process neighbors
                            epoch_to_atom_ids: Dict[str, List[str]] = {}
                            for neighbor_atom_id, epoch_id in self.global_kg[ent]:
                                if neighbor_atom_id not in candidates:
                                    if epoch_id not in epoch_to_atom_ids:
                                        epoch_to_atom_ids[epoch_id] = []
                                    epoch_to_atom_ids[epoch_id].append(neighbor_atom_id)
                                else:
                                    # If already in candidates, we still verify it's a valid 
                                    # entity match for the current query context.
                                    query_entities.add(ent)
                            
                            for epoch_id, atom_ids in epoch_to_atom_ids.items():
                                n_atoms = self.cold_tier.load_atom_metadata(epoch_id, atom_ids)
                                for n_atom in n_atoms:
                                    if len(n_atom.embedding) == len(query_emb):
                                         sim = np.dot(n_atom.embedding, query_emb) / (
                                            np.linalg.norm(n_atom.embedding) * np.linalg.norm(query_emb) + 1e-10
                                        )
                                         # AUTO-BOOST: if we reached this atom via a KG hop, 
                                         # it should count as an entity match for Factor C.
                                         query_entities.add(ent)
                                         new_neighbors.add(n_atom.id)
                                         candidates[n_atom.id] = (n_atom, float(sim))
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

        # --- 4. 4-Way Fusion with Topic Locking & Supersession ---
        K = 60  # Industry standard for RRF stability

        # 1. Supersession Detection (State-Aware)
        superseded_ids: Set[str] = set()
        # Sort by (created_at, id) for deterministic recency resolution even on time collisions
        recency_sorted = sorted(unique_results, key=lambda x: (x[0].created_at, x[0].id), reverse=True)
        active_states: Dict[Tuple[str, str], str] = {}
        for atom, _ in recency_sorted:
            for s, p, o in atom.triples:
                state_key = (str(s).lower(), str(p).lower())
                if state_key in active_states:
                    if active_states[state_key] != atom.id:
                        superseded_ids.add(atom.id)
                else:
                    active_states[state_key] = atom.id

        # 2. Base RRF Ranks
        unique_results.sort(key=lambda x: x[1], reverse=True)
        semantic_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        unique_results.sort(key=lambda x: (x[0].created_at, x[0].id), reverse=True)
        recency_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        def get_overlap(atom: UnifiedMemoryAtom) -> int:
            atom_entities = set()
            for s, _, o in atom.triples:
                atom_entities.add(s)
                atom_entities.add(o)
            return len(query_entities.intersection(atom_entities))

        # If no query entities were found/passed, neutralization is required to prevent 
        # recency-based tie-breaking in RRF.
        if not query_entities:
            entity_ranks = {x[0].id: 0 for x in unique_results}
        else:
            unique_results.sort(key=lambda x: get_overlap(x[0]), reverse=True)
            entity_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        # 3. Discrete Topic Lock (Consolidated Saliency)
        def get_topic_boost(atom: UnifiedMemoryAtom) -> float:
            atom_elements = set()
            for s, p, o in atom.triples:
                atom_elements.add(str(s).lower())
                atom_elements.add(str(o).lower())
                atom_elements.add(p.lower())
            
            boost = 0.0
            # Use original_query_entities — NOT the expansion-contaminated query_entities
            for qe in original_query_entities:
                qe_l = qe.lower()
                is_broad = len(self.global_kg.get(qe, [])) > 5
                
                # We reward matches on THE PRECISE INTENT (The Predicate)
                # or Narrow Entities (Subject/Object).
                for s, p, o in atom.triples:
                    p_l = p.lower()
                    # 1. Exact Predicate Match (Intent Alignment)
                    if qe_l == p_l:
                        boost = max(boost, 20.0) # Nuclear Boost
                        break
                    
                    # 2. Fuzzy Predicate/Entity Match (Prefix/Narrow)
                    qe_root = qe_l[:4]
                    if qe_l in (str(s).lower(), str(o).lower()) and not is_broad:
                        boost = max(boost, 20.0)
                        break
                    
                    if qe_l in p_l or (len(qe_root) >= 4 and qe_root in p_l):
                        boost = max(boost, 20.0)
                        break
            
            return boost

        def multi_rrf_score(atom_id: str, atom: UnifiedMemoryAtom) -> float:
            # 3-Way RRF Fusion
            s_rank = semantic_ranks.get(atom_id, 1000)
            r_rank = recency_ranks.get(atom_id, 1000)
            e_rank = entity_ranks.get(atom_id, 1000)
            
            # Weighted reciprocal ranks
            score = (
                2.0 / (60 + s_rank)    # Semantic (2x weight)
                + 1.0 / (60 + r_rank)  # Recency
                + 1.0 / (60 + e_rank)  # Multi-hop Context
            )
            
            # Topic Lock (Precision Booster)
            boost = get_topic_boost(atom)
            score += boost
            
            # Supersession (Conflict resolution)
            if atom_id in superseded_ids:
                score *= 0.0001
                
            return score

        # Pre-calculate scores to detect 'Signal' presence
        scored_results = []
        has_signal = False
        for atom, sim in unique_results:
            score = multi_rrf_score(atom.id, atom)
            if score >= 20.0:
                has_signal = True
            scored_results.append((atom, score))
        
        # Signal-to-Noise Filter: if Signal exists, demote Noise aggressively
        if has_signal:
            final_scored = []
            for atom, score in scored_results:
                if score < 20.0:
                    score *= 0.0000001
                final_scored.append((atom, score))
            scored_results = final_scored

        scored_results.sort(key=lambda x: x[1], reverse=True)
        unique_results = scored_results

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
