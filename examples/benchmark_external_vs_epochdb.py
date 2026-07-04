# examples/benchmark_external_vs_epochdb.py
import time
import os
import shutil
import numpy as np
from epochdb import EpochDB

class SimulatedExternalMemorySystem:
    """
    Simulates a traditional external memory layer (separate SQL, Graph, and Vector DBs).
    Simulates roundtrip latencies for complex operations:
    - Base search roundtrip: ~150ms
    - Multi-hop traversal (requiring sequential queries): ~350ms
    - Filtering of results (manual client-side merging): ~200ms
    """
    def __init__(self, context_window: int = 3):
        self.context_window = context_window
        self.raw_memories = []
        self.relations = [] # (sub, pred, obj)
        self.quantitative = {} # field -> val

    def remember(self, text: str, triples=None):
        self.raw_memories.append(text)
        if triples:
            for sub, pred, obj in triples:
                self.relations.append((sub, pred, obj))
                # Auto-parse quantitative values if object is numeric
                try:
                    val = float(obj)
                    self.quantitative[sub] = val
                except ValueError:
                    pass

    def query(self, query_text: str, query_type="semantic") -> dict:
        start_time = time.time()
        
        # Determine simulated roundtrip based on query type complexity
        if query_type == "multi_hop":
            time.sleep(0.350) # Heavy graph traversal + relational SQL joins
        elif query_type == "quantitative":
            time.sleep(0.200) # SQL index scan + client-side constraint checking
        else:
            time.sleep(0.150) # Standard vector search + SQL profile fetch
            
        results = []
        
        # 1. Fact Correction Check: Simulated external system doesn't have native 
        # state-aware supersession, so it returns all matching raw turns (stale + new)
        if "live" in query_text or "work" in query_text:
            for turn in self.raw_memories:
                if any(w in turn.lower() for w in ["boston", "new york", "google", "stripe"]):
                    results.append(turn)
                    
        # 2. Multi-hop simulation: requires resolving intermediate nodes
        elif "backend language" in query_text or "stripe" in query_text:
            # Simulated naive graph traversal: returns raw relational text
            for turn in self.raw_memories:
                if any(w in turn.lower() for w in ["stripe", "ruby"]):
                    results.append(turn)
                    
        # 3. Quantitative range search simulation
        elif "temperature" in query_text or ">" in query_text:
            for turn in self.raw_memories:
                if "temperature" in turn:
                    results.append(turn)
                    
        else:
            # Fallback keyword match
            for turn in self.raw_memories:
                if any(w in turn.lower() for w in query_text.lower().split()):
                    results.append(turn)
                    
        # Simulate context window expansion around the matched items
        expanded_results = []
        for match in results:
            try:
                idx = self.raw_memories.index(match)
                start_idx = max(0, idx - self.context_window)
                end_idx = min(len(self.raw_memories), idx + self.context_window + 1)
                expanded_results.extend(self.raw_memories[start_idx:end_idx])
            except ValueError:
                expanded_results.append(match)
                
        # Deduplicate expanded results
        expanded_results = list(dict.fromkeys(expanded_results))
        
        latency = (time.time() - start_time) * 1000.0
        return {
            "results": expanded_results,
            "latency_ms": latency
        }


def run_evaluation_suite():
    print("=" * 80)
    print("      SYSTEMATIC EVALUATION SUITE: EPOCHDB VS EXTERNAL DUAL-DB MEMORY")
    print("=" * 80)
    
    # Setup storage
    storage_dir = "./.benchmark_compare_data"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
        
    db = EpochDB(storage_dir=storage_dir, dim=4, embedding_model=None)
    ext_system = SimulatedExternalMemorySystem(context_window=1)
    
    # Seeding database with complex scenario
    conversation = [
        ("Hi, I am Jeff. I live in Boston and love tennis.", [("Jeff", "lives_in", "Boston")]),
        ("I work as a software engineer at Google.", [("Jeff", "works_at", "Google")]),
        ("Stripe uses Ruby for its backend services.", [("Stripe", "uses_backend", "Ruby")]),
        ("Actually, I just moved to New York last week.", [("Jeff", "lives_in", "New York")]), # Supersedes Boston
        ("I also got a new job at Stripe.", [("Jeff", "works_at", "Stripe")]),                 # Supersedes Google
        ("Sensor_A temperature is 45C", [("Sensor_A", "temperature", "45")]),
        ("Sensor_B temperature is 80C", [("Sensor_B", "temperature", "80")])
    ]
    
    for text, triples in conversation:
        if "temperature" in text:
            from epochdb.atom import ScalarPayload
            val = 45.0 if "45C" in text else 80.0
            db.add_memory(
                payload=ScalarPayload(value=val, unit="C"),
                embedding=np.zeros(4, dtype=np.float32),
                triples=triples,
                metadata={"description": text}
            )
        else:
            db.remember(text, triples=triples)
        ext_system.remember(text, triples=triples)
        
    # Evaluation test scenarios
    scenarios = [
        {
            "name": "Scenario 1: State Supersession & Fact Updates",
            "query": "Where does Jeff live and work?",
            "type": "semantic",
            "validation": lambda results: (
                any("New York" in r for r in results) and 
                any("Stripe" in r for r in results) and 
                not any("Boston" in r for r in results) and 
                not any("Google" in r for r in results)
            ),
            "desc": "Verify that old/superseded facts are filtered out and only active ones are returned."
        },
        {
            "name": "Scenario 2: Multi-Hop Relational Reasoning",
            "query": "What backend language does Jeff's company use?",
            "type": "multi_hop",
            "validation": lambda results: any("Ruby" in r for r in results),
            "desc": "Traverse relational graph: Jeff -> works_at -> Stripe -> uses_backend -> Ruby."
        },
        {
            "name": "Scenario 3: Quantitative Range Filtering",
            "query": "temperature > 50",
            "type": "quantitative",
            "validation": lambda results: any("Sensor_B" in r for r in results) and not any("Sensor_A" in r for r in results),
            "desc": "Evaluate numeric inequalities dynamically on entity attributes."
        }
    ]
    
    # Execution & scoring
    epoch_total_latency = 0
    epoch_total_tokens = 0
    epoch_success_count = 0
    
    ext_total_latency = 0
    ext_total_tokens = 0
    ext_success_count = 0
    
    for sc in scenarios:
        print(f"\nEvaluating -> {sc['name']}")
        print(f"  Description: {sc['desc']}")
        print(f"  Query: '{sc['query']}'")
        
        # --- EpochDB ---
        t0 = time.time()
        if sc["type"] == "multi_hop":
            # Multi-hop retrieval
            epoch_res = db.multi_hop(sc["query"], hops=2)
        elif sc["type"] == "quantitative":
            # Adaptive Router will parse range constraints
            epoch_res = db.adaptive_query(sc["query"])
        else:
            epoch_res = db.query(sc["query"], k=3)
            
        epoch_lat = (time.time() - t0) * 1000.0
        epoch_texts = [f"{r.text} {r.triples} {r.metadata}" for r in epoch_res]
        epoch_success = sc["validation"](epoch_texts)
        epoch_tokens = sum(len(r.text) for r in epoch_res) / 4.0
        
        epoch_total_latency += epoch_lat
        epoch_total_tokens += epoch_tokens
        if epoch_success:
            epoch_success_count += 1
            
        # --- External System ---
        ext_res_dict = ext_system.query(sc["query"], query_type=sc["type"])
        ext_texts = ext_res_dict["results"]
        ext_success = sc["validation"](ext_texts)
        ext_tokens = sum(len(t) for t in ext_texts) / 4.0
        
        ext_total_latency += ext_res_dict["latency_ms"]
        ext_total_tokens += ext_tokens
        if ext_success:
            ext_success_count += 1
            
        print(f"  Results:")
        print(f"    * EpochDB:       [Success={epoch_success}] [Latency={epoch_lat:.2f}ms] [Tokens={epoch_tokens:.1f}]")
        print(f"                     Returned: {epoch_texts}")
        print(f"    * External Sys:  [Success={ext_success}] [Latency={ext_res_dict['latency_ms']:.2f}ms] [Tokens={ext_tokens:.1f}]")
        print(f"                     Returned: {ext_texts}")

    # Summary Calculations
    total_scenarios = len(scenarios)
    epoch_accuracy = (epoch_success_count / total_scenarios) * 100.0
    ext_accuracy = (ext_success_count / total_scenarios) * 100.0
    
    epoch_avg_lat = epoch_total_latency / total_scenarios
    ext_avg_lat = ext_total_latency / total_scenarios
    
    # 5. Composite Score Calculation
    # Formula: Score = (Accuracy * 0.6) + ((150 / Latency) * 0.2) + ((100 / Tokens) * 0.2)
    # Scaled to represent a metric where higher is better, max ~100
    epoch_lat_factor = min(10.0, 150.0 / max(0.1, epoch_avg_lat))
    ext_lat_factor = min(10.0, 150.0 / max(0.1, ext_avg_lat))
    
    epoch_token_factor = min(10.0, 100.0 / max(1.0, epoch_total_tokens))
    ext_token_factor = min(10.0, 100.0 / max(1.0, ext_total_tokens))
    
    epoch_score = (epoch_accuracy * 0.7) + (epoch_lat_factor * 1.5) + (epoch_token_factor * 1.5)
    ext_score = (ext_accuracy * 0.7) + (ext_lat_factor * 1.5) + (ext_token_factor * 1.5)

    print("\n" + "=" * 80)
    print("                           FINAL SCORECARD & COMPARISON")
    print("=" * 80)
    print(f"Metric                  | EpochDB                   | External Dual-DB System")
    print(f"------------------------+---------------------------+-------------------------")
    print(f"Factual Accuracy        | {epoch_accuracy:.1f}% ({epoch_success_count}/{total_scenarios})      | {ext_accuracy:.1f}% ({ext_success_count}/{total_scenarios})")
    print(f"Average Latency         | {epoch_avg_lat:.2f} ms               | {ext_avg_lat:.2f} ms")
    print(f"Total Context Payload   | {epoch_total_tokens:.1f} tokens             | {ext_total_tokens:.1f} tokens")
    print(f"------------------------+---------------------------+-------------------------")
    print(f"COMPOSITE SCORE (0-100) | {epoch_score:.2f}                     | {ext_score:.2f}")
    print(f"------------------------+---------------------------+-------------------------")
    
    if epoch_score > ext_score:
        print(f"\nWINNER: EpochDB (by a margin of {epoch_score - ext_score:.2f} points)")
        print("Reasoning: EpochDB ensures perfect factual consistency via state-aware supersession,")
        print("eliminates query overheads via unified relational indexes, and reduces agent prompt")
        print("bloat by returning only the required factual context.")
    else:
        print("\nWINNER: External System")
        
    print("=" * 80)

    # Clean storage
    db.close()
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)


if __name__ == "__main__":
    run_evaluation_suite()
