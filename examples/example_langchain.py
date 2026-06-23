"""
example_langchain.py — EpochDB LangChain & LangGraph Integrations Walkthrough
=============================================================================
Demonstrates the custom LangChain integrations:
1. get_epochdb_tools: A list of 7 tools for LangGraph agents.
2. EpochDBVectorStore: A standard LangChain vector store wrapper.
3. EpochDBMultiHopRetriever: A custom LangChain retriever wrapping multi-hop queries.

Usage:
    python examples/example_langchain.py
"""

import os
import shutil
import time
from typing import Optional

from epochdb import EpochDB, AsyncEpochDB
from epochdb.tools import get_epochdb_tools
from epochdb.vectorstore import EpochDBVectorStore, EpochDBMultiHopRetriever

# ── ANSI Terminal Colors ──────────────────────────────────────────────────────
R  = "\033[0m"
YL = "\033[93m"
CY = "\033[96m"
GN = "\033[92m"
BL = "\033[94m"
BD = "\033[1m"


def hr(title: str):
    line = "═" * 60
    pad = (58 - len(title)) // 2
    print(f"\n╔{line}╗")
    print(f"║{' ' * pad}{BD}{title}{R}{' ' * (58 - pad - len(title))}║")
    print(f"╚{line}╝\n")


def run_vectorstore_demo(db):
    hr("1. EpochDBVectorStore Integration")
    
    # Wrap database in a LangChain VectorStore
    store = EpochDBVectorStore(db)
    
    print(f"{YL}Adding documents to VectorStore...{R}")
    ids = store.add_texts(
        texts=[
            "BMW is headquartered in Munich, Germany.",
            "Mercedes-Benz is headquartered in Stuttgart, Germany.",
            "Porsche is headquartered in Stuttgart, Germany.",
        ],
        metadatas=[
            {"brand": "BMW", "city": "Munich"},
            {"brand": "Mercedes", "city": "Stuttgart"},
            {"brand": "Porsche", "city": "Stuttgart"},
        ]
    )
    print(f"{GN}Added {len(ids)} documents. Generated IDs: {ids}{R}\n")
    
    print(f"{YL}Executing similarity search for 'Stuttgart'...{R}")
    docs = store.similarity_search("Stuttgart", k=2)
    for idx, doc in enumerate(docs, 1):
        print(f"  {idx}. Content: {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")
        
    print(f"\n{YL}Executing similarity search with scores...{R}")
    docs_and_scores = store.similarity_search_with_score("Munich", k=1)
    for doc, score in docs_and_scores:
        print(f"  Content: {doc.page_content}")
        print(f"  Score:   {score}")


def run_retriever_demo(db):
    hr("2. EpochDBMultiHopRetriever Integration")
    
    # Add relational facts
    print(f"{YL}Adding related facts to build a Knowledge Graph...{R}")
    db.remember(
        "Jeff is the lead engineer of Project Hera.",
        metadata={"triples": [("Jeff", "leads", "Project Hera")]}
    )
    db.remember(
        "Project Hera utilizes Rust for performance.",
        metadata={"triples": [("Project Hera", "uses", "Rust")]}
    )
    
    # Wrap database in a Multi-Hop Retriever (2 hops)
    retriever = EpochDBMultiHopRetriever(db=db, k=3, hops=2)
    
    print(f"{YL}Querying retriever: 'What language does Jeff's project use?'{R}")
    docs = retriever.invoke("Jeff Rust project")
    
    print(f"{GN}Retrieved Documents (connected via multi-hop graph paths):{R}")
    for idx, doc in enumerate(docs, 1):
        print(f"  {idx}. Content: {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")


def run_tools_demo(db):
    hr("3. LangGraph / LangChain Tools Integration")
    
    # Generate tools list
    tools = get_epochdb_tools(db)
    print(f"{GN}Generated {len(tools)} StructuredTools for LangGraph agent:{R}")
    for t in tools:
        print(f"  - {BD}{t.name}{R}: {t.description[:80]}...")
        
    # Let's test calling tools programmatically
    tool_map = {t.name: t for t in tools}
    
    # Store a memory using epochdb_remember
    print(f"\n{YL}Invoking 'epochdb_remember' tool...{R}")
    mem_id = tool_map["epochdb_remember"].invoke({
        "text": "The BMW 5 Series has an electric variant called the i5.",
        "metadata": {"triples": [("i5", "is_electric_variant_of", "BMW 5 Series")]}
    })
    print(f"  {GN}Tool Returned Memory ID: {mem_id}{R}")
    
    # Query it back using epochdb_query
    print(f"\n{YL}Invoking 'epochdb_query' tool...{R}")
    query_res = tool_map["epochdb_query"].invoke({
        "query": "electric 5 series",
        "k": 1
    })
    print(f"  {GN}Tool Query Results: {query_res}{R}")


def main():
    storage_dir = "./.epochdb_langchain_demo"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)
        time.sleep(0.1)
        
    print(f"{YL}Initializing local EpochDB instance...{R}")
    # In practice, you can supply a sentence-transformers or Gemini model name:
    # db = EpochDB(storage_dir=storage_dir, dim=384, model="sentence-transformers/all-MiniLM-L6-v2")
    db = EpochDB(storage_dir=storage_dir, dim=4)
    
    try:
        run_vectorstore_demo(db)
        run_retriever_demo(db)
        run_tools_demo(db)
    finally:
        db.close()
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir, ignore_errors=True)
            
    hr("Walkthrough Complete")
    print(f"All LangChain and LangGraph integrations completed successfully!\n")


if __name__ == "__main__":
    main()
