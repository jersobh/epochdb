"""
benchmarks/needle.py — Needle in a Haystack (NIAH)
==================================================
Tests precision@3 in a high-noise environment.
A 'needle' (signal fact) is hidden among a 'haystack' of noise facts.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── Dataset ────────────────────────────────────────────────────────────────────

SIGNAL_FACTS: List[Tuple[str, List[tuple]]] = [
    ("Alice is the project lead for the classified Project Helios initiative.",
     [("Alice", "leads", "Project Helios")]),
    ("Project Helios has a confirmed budget of 42 million dollars for FY2026.",
     [("Project Helios", "budget_usd_M", "42")]),
    ("Project Helios is headquartered at the secured Neo-Tokyo Data Vault.",
     [("Project Helios", "located_at", "Neo-Tokyo")]),
]

NOISE_FACTS: List[str] = [
    "The engineering team is focusing on refactoring the core API for better performance.",
    "A new security protocol has been implemented across the primary data centers.",
    "Quarterly results show a significant increase in user engagement for the mobile app.",
    "The marketing department is planning a rebranding campaign for the next fiscal year.",
    "Infrastructure maintenance is scheduled for the last weekend of this month.",
    "The human resources department announced new training programmes for all staff.",
    "A new project management tool has been adopted by the product development teams.",
    "Customer feedback suggests that the latest UI update is generally well-received.",
    "The research team published a paper on the applications of graph neural networks.",
    "Legal review of the new service level agreements is currently underway.",
    "The finance team updated the expenditure reporting process for project budgets.",
    "A cross-functional task force is addressing the recent delivery delays.",
    "The annual company retreat will be held in a mountain resort this September.",
    "Technical documentation for the v0.4.1 release is nearing completion.",
    "The board approved the expansion into three new international markets.",
    "Developer surveys indicate a strong preference for the new CI/CD pipeline.",
    "The customer support team is expanding to provide 24/7 global coverage.",
    "A prototype for the next-generation hardware module is being tested in the lab.",
    "Internal audits confirmed compliance with the latest data privacy regulations.",
    "The innovation lab is exploring sustainable energy solutions for the hardware stack.",
    "Project Orion reports a 15% improvement in latency for the indexing service.",
    "The legal department is vetting several potential acquisition targets for Q4.",
    "Project Apollo's heat shield test was successful across all parameters.",
    "The data engineering team is optimizing the Parquet compression ratios.",
    "Project Artemis budget was revised to reflect the increased hardware costs.",
    "The security team flagged a potential vulnerability in the legacy auth system.",
    "A new performance benchmarking suite has been developed for the vector store.",
    "The recruiting team is looking for senior engineers with HNSW expertise.",
    "Project Aurora team is relocating to the newer Houston facility.",
    "The global supply chain issue has slightly delayed the AI cluster expansion.",
    "The customer portal relaunch is now targeted for a soft launch in early April.",
    "Project Zenith is investigating a new type of scalar quantization for INT8.",
    "The DevOps team is migrating the staging environment to a new cloud provider.",
    "Project Hermes has successfully integrated the new triple extraction pipeline.",
    "The documentation team is standardizing the API reference format.",
    "Project Titan is focusing on cold-tier storage efficiency and Zstd tuning.",
    "The executive team is reviewing the strategic roadmap for the next two years.",
    "Project Icarus successfully completed its first high-altitude flight test.",
    "The compliance team is auditing the new KYC process for international users.",
    "Project Odyssey is exploring decentralized storage backends for user data.",
    "The front-end squad is implementing a new design system based on Outfit fonts.",
    "Project Vulcan has achieved a 30% reduction in power consumption for idling nodes.",
    "The data science team is training a new embedding model with 4096 dimensions.",
    "Project Chronos is implementing a more granular Write-Ahead Logging system.",
    "The sales department exceeded the targets for the primary software subscription.",
    "Project Atlas is mapping the relationship between semantic noise and recall.",
    "The intern team developed a visualization dashboard for memory saliency.",
    "Project Minerva is applying large language models to automated SQL optimization.",
    "The operations team improved the cold-start time of the container registry.",
    "Project Phoenix is rebuilding the recovery logic for multi-node clusters.",
]

def run(db, embedder) -> Dict[str, float]:
    """Run the NIAH benchmark. Returns precision@3 and related metrics."""
    logger.info("[NIAH] Running Needle in a Haystack (precision@3) benchmark.")

    db.hot_tier.clear()
    db.wal.clear()

    # Ingest haystack (noise)
    noise_ids = set()
    for text in NOISE_FACTS:
        aid = db.add_memory(text, embedder.encode(text), triples=[])
        noise_ids.add(aid)
    
    # Ingest needles (signal)
    signal_ids = set()
    for text, triples in SIGNAL_FACTS:
        aid = db.add_memory(text, embedder.encode(text), triples=triples)
        signal_ids.add(aid)
    
    # Evaluate
    query = "What is Alice's project, its budget, and its location?"
    # Entities to trigger the Factor P (Topic Lock)
    query_entities = ["Alice", "Project Helios", "budget", "location"]
    
    q_emb = embedder.encode(query)
    results = db.recall(q_emb, top_k=3, expand_hops=1, query_entities=query_entities)
    
    signal_hits = sum(1 for r in results if r.id in signal_ids)
    noise_hits = sum(1 for r in results if r.id in noise_ids)
    precision_at_3 = signal_hits / max(len(results), 1)
    
    # Also calculate recall@3 — out of our 3 signal facts, how many did we get?
    recall_at_3 = signal_hits / len(SIGNAL_FACTS)

    logger.info(f"  precision@3: {precision_at_3:.3f} ({signal_hits} signal, {noise_hits} noise)")
    logger.info(f"  recall@3:    {recall_at_3:.3f} ({signal_hits}/{len(SIGNAL_FACTS)} signal facts)")

    return {
        "precision_at_3": precision_at_3,
        "recall_at_3": recall_at_3,
        "signal_hits": signal_hits,
        "noise_hits": noise_hits,
        "total_signal": len(SIGNAL_FACTS),
        "total_noise": len(NOISE_FACTS),
    }
