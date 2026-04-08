import logging
from typing import Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from .base import BenchmarkAdapter

logger = logging.getLogger(__name__)

class ConvoMemBenchmark(BenchmarkAdapter):
    name = "ConvoMem"

    def load_dataset(self) -> Any:
        try:
            logger.info("Loading ConvoMem dataset from huggingface (Salesforce/ConvoMem)...")
            dataset = load_dataset("Salesforce/ConvoMem", split="test[:100]", trust_remote_code=True) 
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load ConvoMem from HF: {e}. Falling back to simulated mock dataset.")
            return [
                {
                    "context": ["User: My favorite color is blue.", "Assistant: Got it."],
                    "question": "What is the user's favorite color?",
                    "answer": "blue"
                },
                {
                    "context": ["User: Actually, my favorite color is now red.", "Assistant: Updated."],
                    "question": "What is the user's favorite color?",
                    "answer": "red"
                }
            ]

    def ingest(self):
        dataset = self.load_dataset()
        logger.info(f"Ingesting {len(dataset)} conversation chunks...")
        
        self.test_cases = []
        for i, item in enumerate(tqdm(dataset, desc=f"Ingesting ConvoMem to {self.store.__class__.__name__}")):
            if isinstance(item, dict) and "context" in item:
                context = "\n".join(item["context"]) if isinstance(item["context"], list) else str(item["context"])
                q = item.get("question", "")
                a = item.get("answer", "")
            else:
                context = str(item)
                q = "Mock question"
                a = "Mock answer"

            emb = self.embedder.encode(context)
            triples = self._extract_heuristic_triples(context)
            self.store.add(payload=context, embedding=emb, triples=triples)

            self.test_cases.append((q, a))

        self.store.checkpoint()

    def evaluate(self) -> Dict[str, float]:
        logger.info(f"Evaluating ConvoMem benchmark on {self.store.__class__.__name__}...")
        correct = 0
        total = len(self.test_cases)
        
        for q, expected_answer in tqdm(self.test_cases, desc="Evaluating ConvoMem"):
            query_emb = self.embedder.encode(q)
            results = self.store.query(query_emb, top_k=3)
            
            found = False
            for r in results:
                if expected_answer.lower() in str(r.payload).lower():
                    found = True
                    break
            
            if found:
                correct += 1
                
        accuracy = correct / max(1, total)
        return {"recall@3": accuracy}
