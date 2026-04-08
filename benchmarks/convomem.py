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
            # We attempt to load a standard subset or the full HF dataset.
            # E.g. Salesforce/ConvoMem - we use the validation/test split.
            logger.info("Loading ConvoMem dataset from huggingface (Salesforce/ConvoMem)...")
            dataset = load_dataset("Salesforce/ConvoMem", split="test[:100]") # use 100 samples for swift bench
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
        for i, item in enumerate(tqdm(dataset, desc="Ingesting ConvoMem to EpochDB")):
            if isinstance(item, dict) and "context" in item:
                # Mock structure or HF structure
                context = "\n".join(item["context"]) if isinstance(item["context"], list) else str(item["context"])
                q = item.get("question", "")
                a = item.get("answer", "")
            else:
                context = str(item)
                q = "Mock question"
                a = "Mock answer"

            # 1. Embed and push to DB
            emb = self.embedder.encode(context)
            triples = self._extract_heuristic_triples(context)
            self.db.add_memory(payload=context, embedding=emb, triples=triples)

            # 2. Store question for evaluate
            self.test_cases.append((q, a))

        # Checkpoint to push some to cold tier for testing
        self.db.force_checkpoint()

    def evaluate(self) -> Dict[str, float]:
        logger.info("Evaluating ConvoMem benchmark...")
        correct = 0
        total = len(self.test_cases)
        
        for q, expected_answer in tqdm(self.test_cases, desc="Evaluating ConvoMem"):
            query_emb = self.embedder.encode(q)
            results = self.db.recall(query_emb, top_k=3)
            
            # Recall@3 Check (Is the expected answer in the retrieved payload strings?)
            found = False
            for r in results:
                if expected_answer.lower() in str(r.payload).lower():
                    found = True
                    break
            
            if found:
                correct += 1
                
        accuracy = correct / max(1, total)
        return {"recall@3": accuracy}
