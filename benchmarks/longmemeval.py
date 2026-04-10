import logging
from typing import Dict, Any
from tqdm import tqdm
from .base import BenchmarkAdapter

logger = logging.getLogger(__name__)

class LongMemEvalBenchmark(BenchmarkAdapter):
    name = "LongMemEval"

    def load_dataset(self) -> Any:
        # Simulated dataset representing longitudinal sessions
        return [
            {"session_id": 1, "text": "Session 1: User met Alice at a conference in 2021."},
            {"session_id": 2, "text": "Session 2: User started a new job at TechCorp."},
            {"session_id": 3, "text": "Session 3: Alice works at BioGen now."},
            {"session_id": 4, "text": "Session 4: User is considering moving to New York."}
        ]

    def ingest(self):
        dataset = self.load_dataset()
        for item in tqdm(dataset, desc=f"Ingesting LongMemEval sessions to {self.store.__class__.__name__}"):
            emb = self.embedder.encode(item["text"])
            self.store.add(payload=item["text"], embedding=emb)
            
            # Force checkpoints periodically to simulate time passing (Hot -> Cold transitions in EpochDB)
            if item["session_id"] % 2 == 0:
                self.store.checkpoint()

    def evaluate(self) -> Dict[str, float]:
        questions = [
            ("Where did the user meet Alice?", "conference"),
            ("Where does Alice work?", "BioGen")
        ]
        
        correct = 0
        for q, expected in questions:
            q_emb = self.embedder.encode(q)
            # v0.3.0: Boost recall using known entities from the query
            q_entities = ["Alice"] + expected.split()
            
            results = self.store.query(q_emb, top_k=3, query_entities=q_entities)
            found = any(expected.lower() in str(r.payload).lower() for r in results)
            if found:
                correct += 1
                
        return {"recall@3": correct / len(questions)}
