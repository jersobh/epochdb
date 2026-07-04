import numpy as np
from typing import List, Dict, Set, Optional, Any, Tuple
from epochdb.core.atom import UnifiedMemoryAtom, PayloadType, SeriesPoint
from epochdb.core.units import UnitRegistry
from epochdb.retrieval.quantitative_index import ScalarIndex
import logging
import re

logger = logging.getLogger(__name__)


class RetrievalManager:
    """Multi-stage retrieval with Global KG and persistent access-count tracking."""

    def __init__(
        self,
        hot_tier: "HotTier",
        cold_tier: "ColdTier",
        kg_manager: "KGManager",
    ):
        self.hot_tier = hot_tier
        self.cold_tier = cold_tier
        self.kg_manager = kg_manager
        self.unit_registry = UnitRegistry()

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

    def query_range(self, field: str, min_val: float, max_val: float, unit: Optional[str] = None, fork_id: Optional[str] = None) -> List[UnifiedMemoryAtom]:
        """Range query over scalars with interval arithmetic and unit normalization."""
        q_min, q_max = min_val, max_val
        if unit:
            # Normalize query bounds to base unit if necessary
            if field in self.hot_tier.quant_index.scalar_indices:
                base_unit = self.hot_tier.quant_index.scalar_indices[field].base_unit
                if not self.unit_registry.compatible(unit, base_unit):
                    raise ValueError(
                        f"Dimensional mismatch: query unit '{unit}' is incompatible "
                        f"with field '{field}' base unit '{base_unit}'"
                    )
                q_min = self.unit_registry.convert(min_val, unit, base_unit)
                q_max = self.unit_registry.convert(max_val, unit, base_unit)
        
        # 1. Hot Tier Query
        atom_ids = self.hot_tier.quant_index.scalar_indices.get(field, ScalarIndex("")).query_overlap(q_min, q_max)
        
        # 2. Fork Overlay (Shadow Values)
        if fork_id:
            fork_hits = self.hot_tier.quant_index.fork_manager.query_overlap(fork_id, field, q_min, q_max)
            atom_ids = list(set(atom_ids + fork_hits))

        results = []
        for aid in atom_ids:
            atom = self.hot_tier.atoms.get(aid)
            if not atom:
                # Fallback to cold tier search (simplified)
                pass
            
            if atom:
                if unit and atom.payload_type == PayloadType.SCALAR:
                    if not self.unit_registry.compatible(atom.payload.unit, unit):
                        continue
                results.append(atom)
        return results

    def query_temporal(self, atom_id: str, timestamp: float, fork_id: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Interpolate series value and propagate uncertainty via RSS."""
        atom = self._fetch_atom_by_id(atom_id, "active")
        if not atom or atom.payload_type != PayloadType.SERIES:
            return None
        
        points = atom.payload.points
        if fork_id:
            points = self.hot_tier.quant_index.fork_manager.resolve_series_points(
                fork_id, atom.triples[0][0], atom.triples[0][1], points
            )
        
        points = sorted(points, key=lambda p: p.timestamp)
        if not points: return None

        if timestamp <= points[0].timestamp: 
            return {"value": points[0].value, "uncertainty": max(points[0].uncertainty_low, points[0].uncertainty_high)}
        if timestamp >= points[-1].timestamp: 
            return {"value": points[-1].value, "uncertainty": max(points[-1].uncertainty_low, points[-1].uncertainty_high)}

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            if p1.timestamp <= timestamp <= p2.timestamp:
                t_total = p2.timestamp - p1.timestamp
                w2 = (timestamp - p1.timestamp) / t_total
                w1 = 1.0 - w2
                val = w1 * p1.value + w2 * p2.value
                u1 = max(p1.uncertainty_low, p1.uncertainty_high)
                u2 = max(p2.uncertainty_low, p2.uncertainty_high)
                uncertainty = np.sqrt((w1 * u1)**2 + (w2 * u2)**2)
                return {"value": float(val), "uncertainty": float(uncertainty)}
        return None

    def query_aggregate(self, atom_id: str, window: str, t_start: float, t_end: float) -> List[SeriesPoint]:
        """Hit pre-materialized aggregated views."""
        return self.hot_tier.quant_index.series_index.query_aggregate(atom_id, window, t_start, t_end)

    def check_feasibility(self, constraint_ids: List[str], state: Dict[str, float], fork_id: Optional[str] = None) -> bool:
        """Check if a set of constraints is feasible against a given state, including fork state."""
        constraints = []
        merged_state = state.copy()
        
        # Inject fork shadow values into state
        if fork_id and fork_id in self.hot_tier.quant_index.fork_manager.shadow_scalars:
            for field, shadow_atoms in self.hot_tier.quant_index.fork_manager.shadow_scalars[fork_id].items():
                if shadow_atoms:
                    # Take the latest shadow value by timestamp
                    latest = max(shadow_atoms.items(), key=lambda x: x[1][3])
                    merged_state[field] = latest[1][0]  # the value

        for cid in constraint_ids:
            atom = self._fetch_atom_by_id(cid, "active")
            if atom and atom.payload_type == PayloadType.CONSTRAINT:
                constraints.append(atom.payload)
        
        return self.hot_tier.quant_index.constraint_checker.check_feasibility(
            constraints, merged_state, index_manager=self.hot_tier.quant_index
        )

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        expand_hops: int = 1,
        query_entities: List[str] = None,
        payload_type: Optional[PayloadType] = None,
        fork_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        context_window: int = 0,
    ) -> List[UnifiedMemoryAtom]:
        query_entities = set(query_entities) if query_entities else set()
        # Freeze the original query intent before graph expansion contaminates it.
        # get_topic_boost MUST score against this, not the expanded set.
        original_query_entities = frozenset(query_entities)
        # Candidates: {atom_id: (atom, semantic_similarity)}
        candidates: Dict[str, tuple] = {}

        # --- Quantitative Intent Extraction ---
        # "high power usage" -> (field="power_usage", op=">", threshold=learned/heuristic)
        quant_intent = self._extract_quant_intent(original_query_entities)

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
                score = np.dot(atom.embedding, query_emb) / (
                    np.linalg.norm(atom.embedding) * np.linalg.norm(query_emb) + 1e-10
                )
                if score > 0.5:
                    for s, p, o in atom.triples:
                        query_entities.add(s)
                        query_entities.add(o)

        # --- 1a. Entity Hook: Global KG Seeding ---
        # If query entities match Global KG entries, we pull them in as candidates 
        # even if their semantic score was too low to make the initial pool.
        for qe in query_entities:
            associations = self.kg_manager.get_associations(qe)
            if associations:
                # Group neighbor atoms by epoch for optimized loading
                epoch_to_atom_ids: Dict[str, List[str]] = {}
                for a_id, ep_id in associations:
                    if a_id not in candidates:
                        if ep_id not in epoch_to_atom_ids:
                            epoch_to_atom_ids[ep_id] = []
                        epoch_to_atom_ids[ep_id].append(a_id)
                
                # Fetch from Cold Tier if needed
                for ep_id, a_ids in epoch_to_atom_ids.items():
                    atoms = self.cold_tier.load_atom_metadata(ep_id, a_ids)
                    for a in atoms:
                        sim = 0.0
                        if query_emb.any() and a.embedding.any():
                            sim = np.dot(a.embedding, query_emb) / (
                                np.linalg.norm(a.embedding) * np.linalg.norm(query_emb) + 1e-10
                            )
                        # Entity match gets a baseline boost to ensure retrieval
                        candidates[a.id] = (a, float(sim) + 0.5)
                
                # Also check Hot Tier
                for a_id, _ in associations:
                    if a_id in self.hot_tier.atoms and a_id not in candidates:
                        a = self.hot_tier.atoms[a_id]
                        sim = 0.0
                        if query_emb.any() and a.embedding.any():
                            sim = np.dot(a.embedding, query_emb) / (
                                np.linalg.norm(a.embedding) * np.linalg.norm(query_emb) + 1e-10
                            )
                        candidates[a.id] = (a, float(sim) + 0.5)

        # --- Keyword-based Entity Extraction (Auto-Expansion) ---
        # If no explicit entities are passed, we scan the query embedding surface 
        # (or payload keywords) for matches in the Global KG to boost Factor C.
        if not query_entities:
            # We don't have the raw query text here, but we can use the Global KG 
            # keys as a candidate set for heuristic matching if the user passed 
            # an unpopulated set. For now, we rely on the expansion set below.
            pass

        # --- 1b. Semantic Hook: Cold Tier (Parallel Indexed Search) ---
        epochs = self.cold_tier.get_all_epochs()
        if epochs:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(epochs))) as executor:
                # Launch all epoch searches in parallel
                future_to_epoch = {
                    executor.submit(self.cold_tier.search_epoch, epoch, query_emb, top_k * 10): epoch 
                    for epoch in epochs
                }
                
                for future in concurrent.futures.as_completed(future_to_epoch):
                    try:
                        cold_hits = future.result()
                        for atom in cold_hits:
                            if len(atom.embedding) != len(query_emb):
                                continue
                                
                            sim = np.dot(atom.embedding, query_emb) / (
                                np.linalg.norm(atom.embedding) * np.linalg.norm(query_emb) + 1e-10
                            )
                            atom.access_count += self._access_deltas.get(atom.id, 0)
                            candidates[atom.id] = (atom, float(sim))
                    except Exception as e:
                        logger.error(f"Search failed for epoch {future_to_epoch[future]}: {e}")

        # --- 2. Relational Expansion via Global KG ---
        # This doubles as our Entity Extraction: if a candidate atom mentions an entity 
        # that was also in our query (heuristically), it gets a Factor C boost.
        if expand_hops > 0:
            # Sort candidates by similarity score and expand only the most promising ones
            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1][1], reverse=True)
            expansion_limit = max(top_k, 50)
            expansion_set = {k for k, _ in sorted_candidates[:expansion_limit]}
            
            for _ in range(expand_hops):
                new_neighbors: Set[str] = set()
                
                # Gather all unique entities to query associations for in a single batch
                all_entities_to_expand = set()
                atom_entities_map = {}
                for a_id in expansion_set:
                    atom_data = candidates.get(a_id)
                    if not atom_data:
                        continue
                    atom = atom_data[0]
                    entities = set()
                    for subj, pred, obj in atom.triples:
                        entities.add(subj)
                        entities.add(obj)
                    atom_entities_map[a_id] = entities
                    all_entities_to_expand.update(entities)
                
                # Batch fetch associations for all entities at once
                associations_batch = self.kg_manager.get_associations_batch(list(all_entities_to_expand))
                
                for a_id in expansion_set:
                    entities = atom_entities_map.get(a_id, set())
                    for ent in entities:
                        associations = associations_batch.get(ent)
                        if associations:
                            # Process neighbors
                            epoch_to_atom_ids: Dict[str, List[str]] = {}
                            for neighbor_atom_id, epoch_id in associations:
                                if neighbor_atom_id not in candidates:
                                    if epoch_id not in epoch_to_atom_ids:
                                        epoch_to_atom_ids[epoch_id] = []
                                    epoch_to_atom_ids[epoch_id].append(neighbor_atom_id)
                                else:
                                    # If already in candidates, we still verify it's a valid 
                                    # entity match for the current query context.
                                    query_entities.add(ent)
                            
                            for epoch_id, atom_ids in epoch_to_atom_ids.items():
                                n_atoms = []
                                missing_atom_ids = []
                                for aid in atom_ids:
                                    if aid in self.hot_tier.atoms:
                                        n_atoms.append(self.hot_tier.atoms[aid])
                                    else:
                                        missing_atom_ids.append(aid)
                                
                                if missing_atom_ids:
                                    n_atoms.extend(self.cold_tier.load_atom_metadata(epoch_id, missing_atom_ids))

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
                
                # For next hop, only expand the newly found neighbors up to expansion_limit
                if len(new_neighbors) > expansion_limit:
                    sorted_neighbors = sorted(
                        new_neighbors,
                        key=lambda x: candidates[x][1] if x in candidates else 0.0,
                        reverse=True
                    )
                    expansion_set = set(sorted_neighbors[:expansion_limit])
                else:
                    expansion_set = new_neighbors

        # --- 3. Payload Deduplication ---
        all_candidates = list(candidates.values())
        unique_results = []
        seen_payloads: Set[str] = set()

        for atom, sim in all_candidates:
            if atom.metadata.get("_deleted"):
                continue
            if filters and not self.match_filters(atom.metadata, filters):
                continue
            payload_key = str(atom.payload)
            if payload_key not in seen_payloads:
                unique_results.append((atom, sim))
                seen_payloads.add(payload_key)

        if not unique_results:
            return []

        # --- 4. 4-Way Fusion with Topic Locking & Supersession ---
        K = 60  # Industry standard for RRF stability

        # 1. Supersession Detection (State-Aware & Typed)
        superseded_ids: Set[str] = set()
        # Sort by (created_at, id) for deterministic recency resolution
        recency_sorted = sorted(unique_results, key=lambda x: (x[0].created_at, x[0].id), reverse=True)
        
        # Track active states per (subject, predicate)
        active_states: Dict[Tuple[str, str], UnifiedMemoryAtom] = {}
        
        for atom, _ in recency_sorted:
            for s, p, o in atom.triples:
                state_key = (str(s).lower(), str(p).lower())
                
                if state_key in active_states:
                    active_atom = active_states[state_key]
                    
                    # RULE: Scalars supersede if values differ or if newer
                    if atom.payload_type == PayloadType.SCALAR and active_atom.payload_type == PayloadType.SCALAR:
                        if atom.payload.value != active_atom.payload.value:
                            superseded_ids.add(atom.id)
                        else:
                            # Same value, newer one wins
                            pass # active_atom is already newer due to sort order
                    
                    # RULE: Series APPEND (No supersession here, they merge during materialization)
                    elif atom.payload_type == PayloadType.SERIES and active_atom.payload_type == PayloadType.SERIES:
                        # Series atoms complement each other
                        pass
                    
                    # RULE: Constraints evaluate subsumption
                    elif atom.payload_type == PayloadType.CONSTRAINT and active_atom.payload_type == PayloadType.CONSTRAINT:
                        # If active_atom (newer) is broader than atom (older), atom is superseded
                        # For now, a simple 'same expression' check or placeholder for Z3 subsumption
                        if atom.payload.expression == active_atom.payload.expression:
                            superseded_ids.add(atom.id)
                    
                    # DEFAULT: Recency-based supersession for Text or mixed types
                    else:
                        if active_atom.id != atom.id:
                            superseded_ids.add(atom.id)
                else:
                    active_states[state_key] = atom

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
        # 3. Quantitative Relevance Rank (The 5th Signal)
        def get_quant_score(atom: UnifiedMemoryAtom) -> float:
            if not quant_intent or atom.payload_type not in [PayloadType.SCALAR, PayloadType.SERIES]:
                return 0.0
            
            field, op, val = quant_intent
            # Check if atom has this field in triples
            has_field = any(p.lower() == field.lower() for _, p, _ in atom.triples)
            if not has_field: return 0.0
            
            atom_val = 0.0
            if atom.payload_type == PayloadType.SCALAR:
                atom_val = atom.payload.value
            elif atom.payload_type == PayloadType.SERIES:
                # Use last value for series relevance
                if atom.payload.points: atom_val = atom.payload.points[-1].value
            
            if op == ">" and atom_val > val: return 1.0
            if op == "<" and atom_val < val: return 1.0
            return 0.0

        unique_results.sort(key=lambda x: get_quant_score(x[0]), reverse=True)
        quant_ranks = {x[0].id: i for i, x in enumerate(unique_results)}

        # 3. Discrete Topic Lock (Consolidated Saliency)
        def get_topic_boost(atom: UnifiedMemoryAtom) -> float:
            boost = 0.0
            # Use original_query_entities — NOT the expansion-contaminated query_entities
            for qe in original_query_entities:
                qe_l = qe.lower()
                is_broad = len(self.kg_manager.get_associations(qe)) > 10 # Increased threshold for 'broad'
                
                for s, p, o in atom.triples:
                    p_l = p.lower()
                    # 1. Exact Predicate Match (Intent Alignment) - High Signal
                    if qe_l == p_l:
                        boost += 15.0
                        break
                    
                    # 2. Narrow Entity Match (Subject/Object)
                    if qe_l in (str(s).lower(), str(o).lower()) and not is_broad:
                        boost += 10.0
                        break
                    
                    # 3. Fuzzy/Partial Match - Low Signal
                    qe_root = qe_l[:4]
                    if len(qe_root) >= 4 and qe_root in p_l:
                        boost += 5.0
                        break
            
            return boost

        def multi_rrf_score(atom_id: str, atom: UnifiedMemoryAtom) -> float:
            # 3-Way RRF Fusion
            s_rank = semantic_ranks.get(atom_id, 1000)
            r_rank = recency_ranks.get(atom_id, 1000)
            e_rank = entity_ranks.get(atom_id, 1000)
            
            # Weighted reciprocal ranks
            score = (
                3.0 / (K + s_rank)    # Semantic (3x weight)
                + 1.0 / (K + r_rank)  # Recency
                + 1.0 / (K + e_rank)  # Multi-hop Context
                + 2.0 / (K + quant_ranks.get(atom_id, 1000)) # Quantitative (2x weight)
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
        for atom, sim in unique_results:
            score = multi_rrf_score(atom.id, atom)
            scored_results.append((atom, score))
        
        # Dynamic Signal-to-Noise Filter: demote background noise ONLY when a clear signal is found.
        if scored_results:
            max_score = max(s[1] for s in scored_results)
            # Signal threshold: requires at least one strong boost or multiple weak ones
            threshold = max(15.0, max_score * 0.8)
            
            final_scored = []
            has_strong_signal = max_score >= threshold
            
            for atom, score in scored_results:
                if has_strong_signal and score < threshold:
                    # Soft demotion (0.1x) to keep context available but deprioritized
                    score *= 0.1
                final_scored.append((atom, score))
            scored_results = final_scored

        scored_results.sort(key=lambda x: x[1], reverse=True)
        unique_results = scored_results

        # --- 5. Materialise Results & Update Access Counts ---
        final_atoms = []
        # Merge series points if multiple series atoms for the same entity are present
        merged_series: Dict[Tuple[str, str], UnifiedMemoryAtom] = {}

        for atom, score in unique_results[:top_k]:
            if atom.id in superseded_ids:
                continue

            if atom.payload_type == PayloadType.SERIES:
                for s, p, o in atom.triples:
                    key = (str(s).lower(), str(p).lower())
                    if key in merged_series:
                        # Merge points into the existing 'master' atom for this recall result
                        existing = merged_series[key]
                        # Create a new payload with combined points
                        all_points = existing.payload.points + atom.payload.points
                        # Deduplicate by timestamp
                        unique_points = {p.timestamp: p for p in all_points}.values()
                        existing.payload.points = sorted(list(unique_points), key=lambda x: x.timestamp)
                        # We don't add this atom to final_atoms again
                        continue
                    else:
                        merged_series[key] = atom
            
            atom.access_count += 1
            # Track delta for cold-tier atoms so subsequent loads reflect it.
            if atom.id not in self.hot_tier.atoms:
                self._access_deltas[atom.id] = (
                    self._access_deltas.get(atom.id, 0) + 1
                )
            final_atoms.append(atom)

        # --- 6. Context Window (Temporal Context Expansion) ---
        if context_window > 0 and final_atoms:
            chrono_cache = {}
            for atom in final_atoms:
                ns = atom.namespace
                if ns not in chrono_cache:
                    all_atoms = list(self.hot_tier.atoms.values())
                    for epoch_id in self.cold_tier.get_all_epochs():
                        try:
                            all_atoms.extend(self.cold_tier.load_epoch(epoch_id))
                        except Exception as e:
                            logger.error(f"Failed to load epoch {epoch_id} for context expansion: {e}")
                    
                    filtered = []
                    seen_ids = set()
                    for a in all_atoms:
                        if a.id in seen_ids:
                            continue
                        if a.metadata.get("_deleted"):
                            continue
                        if a.namespace == ns:
                            filtered.append(a)
                            seen_ids.add(a.id)
                    filtered.sort(key=lambda x: x.created_at)
                    chrono_cache[ns] = filtered
                
                ns_atoms = chrono_cache[ns]
                try:
                    idx = next(i for i, a in enumerate(ns_atoms) if a.id == atom.id)
                    start_idx = max(0, idx - context_window)
                    end_idx = min(len(ns_atoms), idx + context_window + 1)
                    neighbors = ns_atoms[start_idx:end_idx]
                    
                    neighbor_dicts = []
                    for neighbor in neighbors:
                        payload_str = neighbor.payload if isinstance(neighbor.payload, str) else str(neighbor.payload)
                        meta = {k: v for k, v in neighbor.metadata.items() if k != "context_neighbors"}
                        neighbor_dicts.append({
                            "id": neighbor.id,
                            "payload": payload_str,
                            "created_at": neighbor.created_at,
                            "memory_type": neighbor.memory_type.value if neighbor.memory_type else "general",
                            "metadata": meta
                        })
                    atom.metadata["context_neighbors"] = neighbor_dicts
                except StopIteration:
                    pass

        return final_atoms

    def match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        if not filters:
            return True
        if not metadata:
            return False
        for k, v in filters.items():
            if k not in metadata:
                return False
            meta_val = metadata[k]
            if isinstance(v, dict):
                for op, val in v.items():
                    if op == "$eq":
                        if meta_val != val: return False
                    elif op == "$ne":
                        if meta_val == val: return False
                    elif op == "$in":
                        if meta_val not in val: return False
                    elif op == "$nin":
                        if meta_val in val: return False
                    elif op == "$gt":
                        if not (meta_val > val): return False
                    elif op == "$gte":
                        if not (meta_val >= val): return False
                    elif op == "$lt":
                        if not (meta_val < val): return False
                    elif op == "$lte":
                        if not (meta_val <= val): return False
                    else:
                        if meta_val != v: return False
            else:
                if meta_val != v:
                    return False
        return True

    def _extract_quant_intent(self, entities: Set[str]) -> Optional[Tuple[str, str, float]]:
        """Heuristic to detect quantitative intent from query entities."""
        # Example: if "temperature" and "high" are in entities
        # This is a placeholder for a more sophisticated LLM-based extraction.
        ent_l = {e.lower() for e in entities}
        if "temperature" in ent_l or "temp" in ent_l:
            field = "temperature"
            if "high" in ent_l: return (field, ">", 25.0)
            if "low" in ent_l: return (field, "<", 15.0)
        
        if "power" in ent_l or "usage" in ent_l:
            field = "power_usage"
            if "high" in ent_l: return (field, ">", 200.0)
        
        return None
