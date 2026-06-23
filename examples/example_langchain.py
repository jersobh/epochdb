"""
example_langchain.py — EpochDB LangChain & LangGraph Integrations Walkthrough
=============================================================================
Demonstrates the custom LangChain integrations for both Sync and Async facades:
1. get_epochdb_tools: A list of 7 tools for LangGraph agents.
2. EpochDBVectorStore: A standard LangChain vector store wrapper.
3. EpochDBMultiHopRetriever: A custom LangChain retriever wrapping multi-hop queries.

Usage:
    python examples/example_langchain.py
"""

import asyncio
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


# ── SYNCHRONOUS DEMONSTRATIONS ────────────────────────────────────────────────

def run_sync_vectorstore_demo(db):
    print(f"{YL}Adding documents to VectorStore (Sync)...{R}")
    store = EpochDBVectorStore(db)
    ids = store.add_texts(
        texts=[
            "BMW is headquartered in Munich, Germany.",
            "Mercedes-Benz is headquartered in Stuttgart, Germany.",
        ],
        metadatas=[
            {"brand": "BMW", "city": "Munich"},
            {"brand": "Mercedes", "city": "Stuttgart"},
        ]
    )
    print(f"{GN}Added {len(ids)} documents. Generated IDs: {ids}{R}\n")
    
    print(f"{YL}Executing similarity search for 'Munich' (Sync)...{R}")
    docs = store.similarity_search("Munich", k=1)
    for idx, doc in enumerate(docs, 1):
        print(f"  {idx}. Content: {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")


def run_sync_retriever_demo(db):
    print(f"{YL}Adding related facts to build a Knowledge Graph (Sync)...{R}")
    db.remember(
        "Jeff is the lead engineer of Project Hera.",
        metadata={"triples": [("Jeff", "leads", "Project Hera")]}
    )
    db.remember(
        "Project Hera utilizes Rust for performance.",
        metadata={"triples": [("Project Hera", "uses", "Rust")]}
    )
    
    retriever = EpochDBMultiHopRetriever(db=db, k=3, hops=2)
    
    print(f"{YL}Querying retriever: 'What language does Jeff's project use?' (Sync)...{R}")
    docs = retriever.invoke("Jeff Rust project")
    
    print(f"{GN}Retrieved Documents (connected via multi-hop paths):{R}")
    for idx, doc in enumerate(docs, 1):
        print(f"  {idx}. Content: {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")


def run_sync_tools_demo(db):
    tools = get_epochdb_tools(db)
    tool_map = {t.name: t for t in tools}
    
    print(f"{YL}Invoking 'epochdb_remember' tool (Sync)...{R}")
    mem_id = tool_map["epochdb_remember"].invoke({
        "text": "The BMW 5 Series has an electric variant called the i5.",
        "metadata": {"triples": [("i5", "is_electric_variant_of", "BMW 5 Series")]}
    })
    print(f"  {GN}Tool Returned Memory ID: {mem_id}{R}")
    
    print(f"\n{YL}Invoking 'epochdb_query' tool (Sync)...{R}")
    query_res = tool_map["epochdb_query"].invoke({
        "query": "electric 5 series",
        "k": 1
    })
    print(f"  {GN}Tool Query Results: {query_res}{R}")


# ── ASYNCHRONOUS DEMONSTRATIONS ───────────────────────────────────────────────

async def run_async_vectorstore_demo(db):
    print(f"{YL}Adding documents to VectorStore (Async)...{R}")
    store = EpochDBVectorStore(db)
    ids = await store.aadd_texts(
        texts=[
            "Audi is headquartered in Ingolstadt, Germany.",
            "Porsche is headquartered in Stuttgart, Germany.",
        ],
        metadatas=[
            {"brand": "Audi", "city": "Ingolstadt"},
            {"brand": "Porsche", "city": "Stuttgart"},
        ]
    )
    print(f"{GN}Added {len(ids)} documents. Generated IDs: {ids}{R}\n")
    
    print(f"{YL}Executing similarity search for 'Ingolstadt' (Async)...{R}")
    docs = await store.asimilarity_search("Ingolstadt", k=1)
    for idx, doc in enumerate(docs, 1):
        print(f"  {idx}. Content: {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")


async def run_async_retriever_demo(db):
    print(f"{YL}Adding related facts to build a Knowledge Graph (Async)...{R}")
    await db.remember(
        "Sarah is the lead designer of Project Zeus.",
        metadata={"triples": [("Sarah", "designs", "Project Zeus")]}
    )
    await db.remember(
        "Project Zeus utilizes React for the UI.",
        metadata={"triples": [("Project Zeus", "uses", "React")]}
    )
    
    retriever = EpochDBMultiHopRetriever(db=db, k=3, hops=2)
    
    print(f"{YL}Querying retriever: 'What UI library does Sarah's project use?' (Async)...{R}")
    docs = await retriever.ainvoke("Sarah React project")
    
    print(f"{GN}Retrieved Documents (connected via multi-hop paths):{R}")
    for idx, doc in enumerate(docs, 1):
        print(f"  {idx}. Content: {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")


async def run_async_tools_demo(db):
    tools = get_epochdb_tools(db)
    tool_map = {t.name: t for t in tools}
    
    print(f"{YL}Invoking 'epochdb_remember' tool (Async)...{R}")
    mem_id = await tool_map["epochdb_remember"].ainvoke({
        "text": "The Audi e-tron is a fully electric SUV.",
        "metadata": {"triples": [("e-tron", "is_electric_variant_of", "Audi")]}
    })
    print(f"  {GN}Tool Returned Memory ID: {mem_id}{R}")
    
    print(f"\n{YL}Invoking 'epochdb_query' tool (Async)...{R}")
    query_res = await tool_map["epochdb_query"].ainvoke({
        "query": "electric e-tron",
        "k": 1
    })
    print(f"  {GN}Tool Query Results: {query_res}{R}")


# ── MAIN RUNNER ───────────────────────────────────────────────────────────────

async def async_main(storage_dir):
    print(f"{YL}Initializing local AsyncEpochDB instance...{R}")
    async_db = AsyncEpochDB(storage_dir=storage_dir, dim=4)
    async with async_db as db:
        await run_async_vectorstore_demo(db)
        await run_async_retriever_demo(db)
        await run_async_tools_demo(db)


def main():
    storage_dir = "./.epochdb_langchain_demo"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)
        time.sleep(0.1)
        
    # --- 1. Run Sync Facade Demo ---
    hr("Running Synchronous Facade Walkthrough (EpochDB)")
    print(f"{YL}Initializing local EpochDB instance...{R}")
    db = EpochDB(storage_dir=storage_dir, dim=4)
    try:
        run_sync_vectorstore_demo(db)
        run_sync_retriever_demo(db)
        run_sync_tools_demo(db)
    finally:
        db.close()
        
    # Clear directory between runs
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)
        time.sleep(0.1)

    # --- 2. Run Async Facade Demo ---
    hr("Running Asynchronous Facade Walkthrough (AsyncEpochDB)")
    asyncio.run(async_main(storage_dir))
    
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir, ignore_errors=True)
            
    hr("Walkthrough Complete")
    print(f"All Sync and Async LangChain and LangGraph integrations completed successfully!\n")


if __name__ == "__main__":
    main()
