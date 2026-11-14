"""Shared predicate vocab used across retrieval and compaction."""

# Associative / typing / biographical edges must never overwrite earlier
# episodic TEXT memories as if they were single-valued scalar facts.
# GLiNER and HF extractors emit many of these on every remember().
NON_SUPERSEDING_PREDICATES = frozenset(
    {
        # Co-occurrence / weak association
        "co_occurs_with",
        "mentions",
        "associated_with",
        "related_to",
        "tagged_with",
        "has_tag",
        "contains",
        "part_of",
        "member_of",
        # Typing (extractors label entities; many memories share the same type edge)
        "is_a",
        "instance_of",
        "type_of",
        "has_type",
        "rdf:type",
        # Biographical / employment / creation
        "founded",
        "co_founded",
        "created",
        "invented",
        "designed",
        "mentored",
        "worked_at",
        "works_at",
        "attended",
        "developed",
        "pioneered",
        "authored",
        "discovered",
        "produced",
        "competes_with",
    }
)
