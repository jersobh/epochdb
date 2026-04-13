"""
example_advanced.py — EpochDB v0.4.1 Advanced Features Showcase
================================================================
Demonstrates:
  • Cross-epoch multi-hop reasoning over a corporate Knowledge Base
  • INT8 + Zstd Cold Tier compression validation
  • access_count / saliency tracking across Hot and Cold tiers
  • EpochDBCheckpointer for LangGraph thread persistence

Embeddings: Gemini embedding-2-preview (3072D)
Generation: rule-based (showcases retrieval quality without LLM costs)

Usage:
    export GEMINI_API_KEY=your_key
    pip install epochdb[langgraph] google-genai
    python example_advanced.py
"""

import os
import shutil
import logging
import warnings
import numpy as np
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)

from utils.shared import load_dotenv
load_dotenv(os.path.dirname(os.path.abspath(__file__)))

from typing import TypedDict, List, Tuple
from langgraph.graph import StateGraph, END
from google import genai

from epochdb import EpochDB
from epochdb.checkpointer import EpochDBCheckpointer

# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    input: str
    context: str
    response: str
    extracted_triples: List[Tuple[str, str, str]]


# ── Gemini Embedder ────────────────────────────────────────────────────────────

EMBED_MODEL = "gemini-embedding-2-preview"
GEN_MODEL   = "gemini-3-flash-preview"  # not used here; rule-based responder saves tokens
DIM         = 3072


def embed(client: genai.Client, text: str) -> np.ndarray:
    resp = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return np.array(resp.embeddings[0].values, dtype=np.float32)


# ── Rule-based Responder (no LLM cost) ────────────────────────────────────────

def rule_based_response(query: str, context: str) -> str:
    """
    Derives an answer from the retrieved context using keyword matching.
    Demonstrates that retrieval quality — not generation — is EpochDB's value-add.
    """
    ctx = context.lower()
    q   = query.lower()

    if "project alpha" in q:
        if "alice" in ctx and "engineering" in ctx:
            return "Project Alpha is managed by the Engineering Dept, which is led by Alice."
        elif "alice" in ctx:
            return "Alice is connected to Project Alpha through the Engineering Department."
        else:
            return "I found some context for Project Alpha but the lead information is missing."

    if "who leads" in q or "who runs" in q:
        if "alice" in ctx and "engineering" in ctx:
            return "Alice leads the Engineering Department."
        return "Leadership information not found in current context."

    if "budget" in q:
        if "500" in ctx or "budget" in ctx:
            return "Project Alpha has a budget of $500,000 for Q2."
        return "Budget information not found."

    return f"Based on context: {context[:200]}..."


# ── Triple Extraction (heuristic) ─────────────────────────────────────────────

def extract_triples(text: str) -> List[Tuple[str, str, str]]:
    triples = []
    if "Alice" in text and "Engineering" in text:
        triples.append(("Alice", "leads", "Engineering Dept"))
    if "Engineering" in text and "Project Alpha" in text:
        triples.append(("Engineering Dept", "manages", "Project Alpha"))
    if "Bob" in text and "Project Alpha" in text:
        triples.append(("Bob", "reports_to", "Project Alpha"))
    if "budget" in text.lower() and "Project Alpha" in text:
        triples.append(("Project Alpha", "has_budget", "500000_Q2"))
    return triples


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is not set.")
        return

    client = genai.Client(api_key=api_key)

    db_dir = "./.epochdb_kb_demo"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     EpochDB v0.4.1 — Advanced Corporate KB Demo          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    db = EpochDB(storage_dir=db_dir, dim=DIM)

    # ── LangGraph Nodes ───────────────────────────────────────────────────────

    def retrieve_memory(state: AgentState) -> dict:
        q_emb = embed(client, state["input"])
        q_ent = [w for w in state["input"].split() if w[0].isupper()]

        results = db.recall(
            q_emb,
            top_k=3,
            expand_hops=3,
            query_entities=q_ent,
        )
        context = "\n".join(f"· {r.payload}" for r in results) if results else "No context."
        print(f"\n  [Retrieve] {len(results)} atom(s) for: \"{state['input'][:60]}\"")
        for r in results:
            print(f"             → {r.payload[:90]}{'…' if len(r.payload) > 90 else ''}")
        return {"context": context}

    def generate_response(state: AgentState) -> dict:
        reply = rule_based_response(state["input"], state["context"])
        print(f"  [Response] {reply}")
        return {"response": reply}

    def extract_and_store(state: AgentState) -> dict:
        triples = extract_triples(state["input"])
        emb = embed(client, state["input"])
        db.add_memory(payload=f"KB: {state['input']}", embedding=emb, triples=triples)
        if triples:
            print(f"  [Store]    Triples: {triples}")
        return {"extracted_triples": triples}

    # ── Graph Compilation ─────────────────────────────────────────────────────

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_memory)
    workflow.add_node("generate", generate_response)
    workflow.add_node("extract_store", extract_and_store)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "extract_store")
    workflow.add_edge("extract_store", END)

    checkpointer = EpochDBCheckpointer(db)
    app = workflow.compile(checkpointer=checkpointer)
    thread = {"configurable": {"thread_id": "corporate_kb_001"}}

    # ── Phase 1: Data Ingestion ───────────────────────────────────────────────

    print("━" * 62)
    print("  PHASE 1 — Knowledge Base Ingestion")
    print("━" * 62)

    records = [
        # Core organisational facts
        "Alice is the primary lead for the Engineering Department.",
        "The Engineering Department manages the development of Project Alpha.",
        # Financial context (stored in a different semantic region)
        "Project Alpha has been allocated a budget of $500,000 for Q2.",
        # Personnel
        "Bob is a senior engineer who reports to the Project Alpha team.",
        # Noise (semantically distant — should NOT surface in project queries)
        "The cafeteria menu changed this week. Tuesday is pasta day.",
        "The office wifi password was updated on Monday.",
    ]

    for record in records:
        print(f"\n  Ingesting: \"{record[:70]}{'…' if len(record) > 70 else ''}\"")
        app.invoke({"input": record}, config=thread)

    # ── Phase 2: Cold Tier Flush ───────────────────────────────────────────────

    print("\n" + "━" * 62)
    print("  PHASE 2 — Epoch Checkpoint (Hot → Cold)")
    print("━" * 62)

    print("\n  Forcing epoch checkpoint...")
    db.force_checkpoint()

    pq_files = [f for f in os.listdir(db_dir) if f.endswith(".parquet")]
    print(f"  {len(pq_files)} Parquet archive(s) written.\n")

    # Validate INT8 quantization in the archive.
    if pq_files:
        target = os.path.join(db_dir, pq_files[0])
        table  = pq.read_table(target)
        emb_type = str(table.schema.field("embedding").type)
        status = "✓ INT8 confirmed" if "int8" in emb_type else f"✗ Unexpected type: {emb_type}"
        print(f"  Archive: {pq_files[0]}")
        print(f"  Embedding dtype:  {emb_type}")
        print(f"  Quantization:     {status}")
        print(f"  Rows:             {table.num_rows}")

    # ── Phase 3: Cross-Epoch Multi-Hop Queries ────────────────────────────────

    print("\n" + "━" * 62)
    print("  PHASE 3 — Cross-Epoch Multi-Hop Reasoning")
    print("━" * 62)
    print("  (Hot Tier is empty — all data lives in Parquet archives.)\n")

    queries = [
        (
            "What is the status of Project Alpha and who leads it?",
            "Expects: Alice → Engineering Dept → Project Alpha (3 hops)",
        ),
        (
            "What is the Q2 budget for Project Alpha?",
            "Expects: Project Alpha → 500k budget fact",
        ),
        (
            "Who is Bob and what project is he on?",
            "Expects: Bob → Project Alpha → Engineering Dept → Alice",
        ),
    ]

    for query, expectation in queries:
        print(f"  Query: \"{query}\"")
        print(f"  Expect: {expectation}")
        app.invoke({"input": query}, config=thread)
        print()

    # ── Phase 4: Saliency Tracking ────────────────────────────────────────────

    print("━" * 62)
    print("  PHASE 4 — Saliency & access_count Tracking")
    print("━" * 62)
    print("\n  Re-querying cold atoms and checking access_count deltas...\n")

    q_emb = embed(client, "Tell me about Project Alpha")
    hits  = db.recall(q_emb, top_k=3, expand_hops=2)

    for r in hits:
        print(f"  · \"{r.payload[:70]}\"")
        print(f"    access_count={r.access_count}  saliency={r.calculate_saliency():.5f}  epoch={r.epoch_id}")
        print()

    db.close()
    print("━" * 62)
    print(f"  Done. Data persisted in: {db_dir}")
    print("━" * 62)


if __name__ == "__main__":
    main()
