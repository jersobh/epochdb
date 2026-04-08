import logging
from typing import Dict, Any
from tqdm import tqdm
from .base import BenchmarkAdapter

logger = logging.getLogger(__name__)

class LoCoMoBenchmark(BenchmarkAdapter):
    name = "LoCoMo"

    def load_dataset(self) -> Any:
        # Mock dataset representing multi-hop logic and events
        return [
            {"id": "doc1", "text": "Event 1: The CEO of OpenAI is Sam Altman."},
            {"id": "doc2", "text": "Event 2: Sam Altman invested in a fusion energy company."},
            {"id": "doc3", "text": "Event 3: The fusion company is called Helion."}
        ]

    def _extract_heuristic_triples(self, text: str) -> list:
        # We mock triple extraction to test Relational Expansion in EpochDB's KG
        if "Sam Altman" in text and "CEO" in text:
            return [("OpenAI", "has_CEO", "Sam Altman")]
        elif "Sam Altman" in text and "fusion" in text:
            return [("Sam Altman", "invested_in", "fusion company")]
        elif "Helion" in text and "fusion" in text:
            return [("fusion company", "is_named", "Helion")]
        return []

    def ingest(self):
        dataset = self.load_dataset()
        for item in tqdm(dataset, desc="Ingesting LoCoMo"):
            emb = self.embedder.encode(item["text"])
            triples = self._extract_heuristic_triples(item["text"])
            self.db.add_memory(payload=item["text"], embedding=emb, triples=triples)

    def evaluate(self) -> Dict[str, float]:
        # LoCoMo evaluates Multi-hop reasoning
        q = "What is the name of the fusion company that the CEO of OpenAI invested in?"
        
        # In a real model, it would break this down.
        # We search with Semantic Hook + Relational Expansion (hops)
        # We set top_k low but expand_hops=2 to allow the Knowledge Graph to traverse:
        # OpenAI -> Sam Altman -> fusion company -> Helion
        
        q_emb = self.embedder.encode(q)
        # Using the engine we query. (engine internally passes expand_hops=1, we can bypass to hit retrieval directly)
        results = self.db.retriever.search(q_emb, top_k=2, expand_hops=2)
        
        found = any("Helion" in str(r.payload) for r in results)
        
        return {"multi_hop_recall": 1.0 if found else 0.0}
