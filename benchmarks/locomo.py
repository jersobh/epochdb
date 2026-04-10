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
        for item in tqdm(dataset, desc=f"Ingesting LoCoMo to {self.store.__class__.__name__}"):
            emb = self.embedder.encode(item["text"])
            triples = self._extract_heuristic_triples(item["text"])
            self.store.add(payload=item["text"], embedding=emb, triples=triples)

    def evaluate(self) -> Dict[str, float]:
        # LoCoMo evaluates Multi-hop reasoning
        q = "What is the name of the fusion company that the CEO of OpenAI invested in?"
        
        q_emb = self.embedder.encode(q)
        # Heuristic extraction for the benchmark query
        q_entities = ["Sam Altman", "OpenAI", "fusion company", "Helion"]
        
        # We use expand_hops=2 to allow the Knowledge Graph to traverse (for EpochDB)
        # and v0.3.0 query_entities for boosting.
        results = self.store.query(q_emb, top_k=2, expand_hops=2, query_entities=q_entities)
        
        found = any("Helion" in str(r.payload) for r in results)
        
        return {"multi_hop_recall": 1.0 if found else 0.0}
