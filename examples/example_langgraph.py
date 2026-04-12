"""
example_langgraph.py — EpochDB v0.4.0 + LangGraph Multi-Session Agent
=======================================================================
Demonstrates a stateful conversational agent that persists both:
  - Long-term associative memory  (EpochDB atoms + Knowledge Graph)
  - Short-term agentic state      (EpochDBCheckpointer → LangGraph threads)

Embeddings: Gemini embedding-2-preview (3072D, local to your machine via API)
Generation: gemini-3-flash-preview
Triple extraction: heuristic (replace with an LLM call in production)

Usage:
    export GEMINI_API_KEY=your_key
    pip install epochdb[langgraph] google-genai
    python example_langgraph.py
"""

import os
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)

from utils.shared import load_dotenv
load_dotenv(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
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


# ── Gemini Helpers ─────────────────────────────────────────────────────────────

EMBED_MODEL = "gemini-embedding-2-preview"
GEN_MODEL   = "gemini-3-flash-preview"
DIM         = 3072


def embed(client: genai.Client, text: str) -> np.ndarray:
    resp = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return np.array(resp.embeddings[0].values, dtype=np.float32)


def generate(client: genai.Client, prompt: str) -> str:
    resp = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return resp.text.strip()


def extract_triples_heuristic(text: str) -> List[Tuple[str, str, str]]:
    """
    Lightweight heuristic triple extractor.
    In production, replace this with an LLM structured-output call, e.g.:
        response = client.models.generate_content(
            model=GEN_MODEL,
            contents=f"Extract (subject, relation, object) triples from: {text}",
            ...
        )
    """
    triples = []
    tl = text.lower()
    if "i'm " in tl or "i am " in tl or "my name is" in tl:
        triples.append(("user", "has_name", "Jeff"))
    if "epochdb" in tl:
        triples.append(("user", "works_on", "EpochDB"))
        triples.append(("EpochDB", "is_a", "memory_engine"))
    if "tiered" in tl or "tier" in tl:
        triples.append(("EpochDB", "architecture", "tiered_storage"))
    if "langgraph" in tl:
        triples.append(("EpochDB", "integrates_with", "LangGraph"))
    return triples


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is not set.")
        return

    client = genai.Client(api_key=api_key)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          EpochDB v0.4.0 — LangGraph Agent Demo           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    print(f"  Embedding:  {EMBED_MODEL} ({DIM}D)")
    print(f"  Generation: {GEN_MODEL}")
    print()

    storage_dir = "./.epochdb_realworld"
    db = EpochDB(storage_dir=storage_dir, dim=DIM)

    # ── Node Definitions ───────────────────────────────────────────────────────

    def retrieve_memory(state: AgentState) -> dict:
        """Semantic + relational recall from EpochDB."""
        print(f"\n  [Retrieve] Query: \"{state['input']}\"")
        try:
            q_emb = embed(client, state["input"])

            # Extract candidate entities from the query for RRF boosting.
            # (Capitalized words are a cheap heuristic — use NER in production.)
            q_entities = [
                w.strip(".,?!")
                for w in state["input"].split()
                if w[0].isupper() or w.lower() in ("epochdb",)
            ]

            results = db.recall(
                q_emb,
                top_k=4,
                expand_hops=2,
                query_entities=q_entities,
            )

            if results:
                context = "\n".join(f"- {r.payload}" for r in results)
                print(f"  [Retrieve] {len(results)} atom(s) recalled (incl. KG expansion):")
                for r in results:
                    print(f"             · {r.payload[:80]}{'…' if len(r.payload) > 80 else ''}")
            else:
                context = "No prior memory."
                print("  [Retrieve] No prior memory found.")

        except Exception as e:
            context = "No prior memory."
            print(f"  [Retrieve] Warning: {e}")

        return {"context": context}

    def generate_response(state: AgentState) -> dict:
        """Call Gemini with injected long-term context."""
        print(f"\n  [Generate] Calling {GEN_MODEL}...")
        prompt = (
            "You are EpochAgent, an AI assistant with perfect long-term memory "
            "powered by EpochDB's tiered storage and Knowledge Graph.\n\n"
            f"Long-term memory context (retrieved via semantic + relational search):\n"
            f"{state['context']}\n\n"
            f"User: {state['input']}\n"
            "Agent:"
        )
        try:
            reply = generate(client, prompt)
        except Exception as e:
            reply = f"[Generation failed: {e}]"

        print(f"  [Generate] Reply: {reply[:120]}{'…' if len(reply) > 120 else ''}")
        return {"response": reply}

    def extract_and_store(state: AgentState) -> dict:
        """Extract KG triples and archive the full interaction as a memory atom."""
        interaction = f"User: {state['input']}\nAgent: {state['response']}"
        triples = extract_triples_heuristic(state["input"])

        try:
            emb = embed(client, interaction)
            db.add_memory(payload=interaction, embedding=emb, triples=triples)
            if triples:
                print(f"\n  [Store] Extracted {len(triples)} triple(s): {triples}")
            else:
                print(f"\n  [Store] Atom stored (no triples extracted).")
        except Exception as e:
            print(f"\n  [Store] Warning: {e}")

        return {"extracted_triples": triples}

    # ── Graph Assembly ─────────────────────────────────────────────────────────

    print("  Compiling LangGraph workflow with EpochDBCheckpointer...\n")
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

    thread = {"configurable": {"thread_id": "demo_thread_001"}}

    # ── Session 1: Establish Facts ─────────────────────────────────────────────

    print("━" * 62)
    print("  SESSION 1 — Establishing Facts")
    print("━" * 62)

    for turn in [
        "Hi, I'm Jeff. I'm building EpochDB.",
        "EpochDB is a tiered memory engine. It stores verbatim atoms.",
        "It also integrates natively with LangGraph for agentic workflows.",
    ]:
        print(f"\n  ┌─ Turn: {turn}")
        app.invoke({"input": turn}, config=thread)
        print(f"  └─ Done.")

    # Flush Hot Tier to Parquet (simulates time passing between sessions).
    print("\n  [Admin] Flushing Hot Tier to Cold (Parquet) archive...")
    db.force_checkpoint()
    cold_files = [f for f in os.listdir(storage_dir) if f.endswith(".parquet")]
    print(f"  [Admin] {len(cold_files)} archive file(s) written to disk.")

    # ── Session 2: Multi-Hop Recall ────────────────────────────────────────────

    print("\n" + "━" * 62)
    print("  SESSION 2 — Cross-Epoch Multi-Hop Reasoning")
    print("━" * 62)
    print("  (All facts now live in the Cold Tier — RAM has been cleared.)\n")

    for turn in [
        "What is the memory engine that Jeff is working on?",
        "How does EpochDB handle data across long time horizons?",
        "Does EpochDB work with LangGraph? How?",
    ]:
        print(f"\n  ┌─ Query: {turn}")
        app.invoke({"input": turn}, config=thread)
        print(f"  └─ Done.")

    db.close()
    print(f"\n  [Done] All data persisted in: {storage_dir}\n")


if __name__ == "__main__":
    import sys
    import shutil

    storage_dir = "./.epochdb_realworld"
    if "--keep" not in sys.argv and os.path.exists(storage_dir):
        print(f"  (Wiping previous demo data in {storage_dir}. Pass --keep to retain.)")
        shutil.rmtree(storage_dir, ignore_errors=True)

    main()
