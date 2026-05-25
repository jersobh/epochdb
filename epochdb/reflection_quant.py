import logging
from typing import Dict, List, Any
from .atom import UnifiedMemoryAtom, PayloadType, ConstraintPayload
from .cold_tier import ColdTierAnalytics

logger = logging.getLogger(__name__)

class QuantitativeReflectionManager:
    """Auto-generates constraints and policies by reflecting on Cold Tier trends."""
    def __init__(self, analytics: ColdTierAnalytics, hot_tier: Any):
        self.analytics = analytics
        self.hot_tier = hot_tier

    def reflect_on_entity(self, entity: str, field: str, confidence_threshold: float = 0.95) -> List[UnifiedMemoryAtom]:
        """Analyzes historical data and suggests policy constraints."""
        trend = self.analytics.query_trend(field, entity)
        if not trend: return []

        # Compute Coefficient of Variation (CV) for confidence
        mean_val = trend.get("mean", 0.0)
        std_val = trend.get("std", 0.0)
        
        if abs(mean_val) > 1e-9:
            cv = abs(std_val / mean_val)
        else:
            cv = min(std_val, 1.0) # Fallback if mean is near 0
            
        # Confidence is inversely proportional to variation
        confidence = max(0.0, 1.0 - cv)
        
        if confidence < confidence_threshold:
            return []

        suggested_atoms = []
        if field == "temperature" and trend["max"] is not None:
            limit = trend["max"] + 1.0
            expr = {"op": "<", "field": field, "value": limit}
            desc = f"Observed policy: {entity} {field} should stay below {limit}"
            
            payload = ConstraintPayload(expression=expr, description=desc)
            atom = UnifiedMemoryAtom(
                payload=payload,
                payload_type=PayloadType.CONSTRAINT,
                triples=[(entity, "policy", field), ("reflection", "derived_from", "trend")]
            )
            # AUTO-INSERT into hot tier
            if hasattr(self.hot_tier, "_add_atom"):
                self.hot_tier._add_atom(atom)
            suggested_atoms.append(atom)
            logger.info(f"Reflected and inserted policy for {entity}.{field}: {desc}")

        return suggested_atoms
