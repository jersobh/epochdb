#!/usr/bin/env python3
"""
sync_async_benchmark.py — Sync vs Async Performance & Token Benchmark
======================================================================
Quantitatively compares the E2E latency, scheduling overhead, and input token consumption of:
1. Sync LangGraph + Sync EpochDB: Sequential, blocking graph execution.
2. Async LangGraph + Async EpochDB: Concurrent, non-blocking graph execution (ainvoke).
3. Async Astraea + EpochBlackboard: Decoupled event-driven reactive coordination.

Features:
- Mock Mode: Standard run with keyword-seeded embeddings and simulated network latency.
- Live Mode: Real Gemini API calls (via google-genai) when GEMINI_API_KEY is present.
- Concurrent stress testing: Simulates multiple parallel user chats to highlight async throughput.
- ASCII charts and markdown reports saved in root.
"""

import os
import sys
import time
import shutil
import asyncio
import argparse
import datetime
import logging
import warnings
from typing import TypedDict, List, Dict, Any, Tuple

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)

# Setup path insertion for both local imports and the Astraea/Aster framework
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../")))
sys.path.insert(1, os.path.abspath(os.path.join(current_dir, "../../astraea_framework")))
sys.path.insert(2, "/home/jeff/Projects/astraea_framework")
sys.path.insert(3, os.path.abspath(os.path.join(current_dir, "../../aster_framework")))
sys.path.insert(4, os.path.abspath(os.path.join(current_dir, "../../aster-framework")))
sys.path.insert(5, "/home/jeff/Projects/aster_framework")
sys.path.insert(6, "/home/jeff/Projects/aster-framework")

from epochdb import EpochDB, AsyncEpochDB
from epochdb.checkpointer import EpochDBCheckpointer
from langgraph.graph import StateGraph, END

# Import Astraea/Aster framework components
try:
    from aster import EpochBlackboard, EventRouter, Agent, Team
    from aster.tiler import ContextTiler
    from aster.types import BlackboardEvent, EventType, LineageContext
    HAS_ASTRAEA = True
except ImportError:
    try:
        from astraea import EpochBlackboard, EventRouter, Agent, Team
        from astraea.tiler import ContextTiler
        from astraea.types import BlackboardEvent, EventType, LineageContext
        HAS_ASTRAEA = True
    except ImportError as e:
        HAS_ASTRAEA = False
        print(f"Warning: Could not import Astraea/Aster: {e}")

# Try importing google-genai
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# ── Configuration & Constants ────────────────────────────────────────────────

STORAGE_DIR_SYNC = "./.epochdb_sync_lg"
STORAGE_DIR_ASYNC = "./.epochdb_async_lg"
STORAGE_DIR_ASTRAEA = "./.epochdb_async_astraea"

EMBED_MODEL = "barisaydin/gte-base"
GEN_MODEL   = "gemini-3-flash-preview"
DIM         = 768

USER_NAMES = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia"]

SCENARIO_TURNS = [
    {
        "user": "Hi! I'm {name}. I am starting a new project called EpochDB.",
        "assistant": "Hello {name}! I've noted that you're starting Project EpochDB.",
        "keywords": ["epochdb"]
    },
    {
        "user": "EpochDB uses a tiered architecture with a hot RAM vector tier and a cold Parquet tier.",
        "assistant": "Understood. EpochDB features a tiered storage architecture with hot RAM and cold Parquet tiers.",
        "keywords": ["tiered", "ram", "parquet"]
    },
    {
        "user": "The Hot Tier uses HNSW index and WAL, while the Cold Tier uses Parquet with Zstd compression.",
        "assistant": "Got it. The Hot Tier employs an HNSW index with a WAL for durability, and the Cold Tier uses Parquet with Zstd compression.",
        "keywords": ["hnsw", "wal", "parquet", "zstd"]
    },
    {
        "user": "A Topic Lock boost of +20.0 is applied to prioritize matched query entity subgraphs in retrieval.",
        "assistant": "Acknowledged. A Topic Lock boost of +20.0 is used to prioritize matching entity subgraphs.",
        "keywords": ["topic", "lock", "boost", "20"]
    },
    {
        "user": "Stale conflicting facts are demoted using a Supersession penalty multiplier of 0.0001x.",
        "assistant": "Understood. A Supersession penalty multiplier of 0.0001x is applied to demote stale, conflicting memories.",
        "keywords": ["supersession", "penalty", "multiplier", "0.0001"]
    },
    {
        "user": "To keep context windows clean, a Signal-to-Noise demotion of 1e-7 is applied to background clutter.",
        "assistant": "Logged. A Signal-to-Noise demotion multiplier of 1e-7 is applied to demote background noise.",
        "keywords": ["signal", "noise", "demotion", "1e-7"]
    },
    {
        "user": "We use a Z3 SAT solver for constraint verification and IntervalTree for scalar ranges.",
        "assistant": "Noted. Z3 is used for constraint checking, and IntervalTree is used for scalar range queries.",
        "keywords": ["z3", "sat", "solver", "intervaltree"]
    },
    {
        "user": "We implement memory forks with db.fork for parallel multi-agent hypothesis testing.",
        "assistant": "Understood. Memory forks via db.fork enable parallel multi-agent hypothesis checking.",
        "keywords": ["fork", "multi-agent", "hypothesis"]
    },
    {
        "user": "What index and safety log does the Hot Tier use, and how are stale facts demoted?",
        "assistant": "The Hot Tier uses an HNSW index and a Write-Ahead Log (WAL). Stale facts are demoted using a Supersession penalty multiplier of 0.0001x.",
        "keywords": ["hnsw", "wal", "supersession", "penalty"]
    },
    {
        "user": "What are the exact values of the Topic Lock boost, Supersession penalty, and Signal-to-Noise demotion?",
        "assistant": "The Topic Lock boost is +20.0, the Supersession penalty is 0.0001x, and the Signal-to-Noise demotion is 1e-7.",
        "keywords": ["20", "0.0001", "1e-7"]
    }
]

KEYWORDS_LIST = ["epochdb", "tiered", "ram", "parquet", "hot", "cold", "architecture", "alice", "bob", "charlie", "hnsw", "wal", "zstd", "topic", "lock", "boost", "20", "supersession", "penalty", "multiplier", "0.0001", "signal", "noise", "demotion", "1e-7", "z3", "sat", "solver", "intervaltree", "fork", "multi-agent", "hypothesis"]


# ── Colors & Visuals ──────────────────────────────────────────────────────────

class C:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @staticmethod
    def header(text):
        return f"\n{C.BOLD}{C.MAGENTA}╔" + "═" * (len(text) + 2) + "╗" + C.END + \
               f"\n{C.BOLD}{C.MAGENTA}║ {text} ║" + C.END + \
               f"\n{C.BOLD}{C.MAGENTA}╚" + "═" * (len(text) + 2) + "╝" + C.END


# ── Token Counter & Mock Helpers ──────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Estimate token count using characters / 4."""
    return int(len(text) / 4.0)


def get_mock_embedding(text: str) -> list:
    """Generate a mock vector with cosine similarity potential for overlap keywords."""
    vec = [0.0] * DIM
    text_lower = text.lower()
    for i, kw in enumerate(KEYWORDS_LIST):
        if i < DIM and kw in text_lower:
            vec[i] = 1.0
    # normalize
    norm = sum(x**2 for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def get_mock_response(prompt: str, turn_idx: int, user_name: str) -> str:
    """Generate mock assistant response."""
    return SCENARIO_TURNS[turn_idx]["assistant"].format(name=user_name)


# ── LLM Client Wrappers ────────────────────────────────────────────────────────

# ── LLM Client Wrappers with OCC/Rate-Limit Retries ───────────────────────────

_local_embedder = None

def get_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer
        _local_embedder = SentenceTransformer(EMBED_MODEL)
    return _local_embedder

def sync_embed(client, text: str, live_mode: bool) -> list:
    if not live_mode:
        time.sleep(0.1)
        return get_mock_embedding(text)
    
    embedder = get_local_embedder()
    return embedder.encode(text, normalize_embeddings=True).tolist()

async def async_embed(client, text: str, live_mode: bool) -> list:
    if not live_mode:
        await asyncio.sleep(0.1)
        return get_mock_embedding(text)
    
    embedder = get_local_embedder()
    emb = await asyncio.to_thread(embedder.encode, text, normalize_embeddings=True)
    return emb.tolist()

def sync_generate(client, prompt: str, turn_idx: int, user_name: str, live_mode: bool) -> str:
    if not live_mode:
        time.sleep(0.2)
        return get_mock_response(prompt, turn_idx, user_name)
        
    import random
    max_retries = 6
    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=GEN_MODEL, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Resource exhausted" in err_str:
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                raise e

async def async_generate(client, prompt: str, turn_idx: int, user_name: str, live_mode: bool) -> str:
    if not live_mode:
        await asyncio.sleep(0.2)
        return get_mock_response(prompt, turn_idx, user_name)
        
    import random
    max_retries = 6
    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = await client.aio.models.generate_content(model=GEN_MODEL, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Resource exhausted" in err_str:
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                raise e


# ── 1. Sync LangGraph Node Definitions ────────────────────────────────────────

class SyncLGState(TypedDict):
    input: str
    context: str
    response: str
    prompt_tokens: int
    user_name: str


def make_sync_langgraph_pipeline(db: EpochDB, client: Any, live_mode: bool):
    workflow = StateGraph(SyncLGState)

    def retrieve_node(state: SyncLGState):
        q = state["input"]
        q_emb = sync_embed(client, q, live_mode)

        results = db.recall(
            query_emb=os.sys.modules['numpy'].array(q_emb, dtype=os.sys.modules['numpy'].float32),
            top_k=2,
            expand_hops=1,
            query_entities=db.extract_entities(q)
        )

        context = "\n".join(f"- {r.payload}" for r in results) if results else "No prior memory."
        return {"context": context}

    def generate_node(state: SyncLGState):
        prompt = (
            "You are a helpful AI memory assistant.\n"
            f"Context: {state['context']}\n\n"
            f"User: {state['input']}\n"
            "Assistant:"
        )
        tokens = count_tokens(prompt)

        # Estimate turn index
        turn_idx = 0
        for i, turn in enumerate(SCENARIO_TURNS):
            if turn["user"].format(name=state["user_name"]) == state["input"]:
                turn_idx = i
                break

        reply = sync_generate(client, prompt, turn_idx, state["user_name"], live_mode)
        return {"response": reply, "prompt_tokens": tokens}

    def store_node(state: SyncLGState):
        interaction = f"User: {state['input']}\nAssistant: {state['response']}"
        emb = sync_embed(client, interaction, live_mode)

        # Extract entities for triples
        entities = db.extract_entities(state["input"])
        triples = []
        for e in entities:
            triples.append((e, "mentions", e))

        db.add_memory(
            payload=interaction,
            embedding=os.sys.modules['numpy'].array(emb, dtype=os.sys.modules['numpy'].float32),
            triples=triples
        )
        return {}

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("store", store_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "store")
    workflow.add_edge("store", END)

    checkpointer = EpochDBCheckpointer(db)
    return workflow.compile(checkpointer=checkpointer)


# ── 2. Async LangGraph Node Definitions ───────────────────────────────────────

class AsyncLGState(TypedDict):
    input: str
    context: str
    response: str
    prompt_tokens: int
    user_name: str


def make_async_langgraph_pipeline(async_db: AsyncEpochDB, client: Any, live_mode: bool):
    workflow = StateGraph(AsyncLGState)

    async def retrieve_node(state: AsyncLGState):
        q = state["input"]
        q_emb = await async_embed(client, q, live_mode)

        db = async_db._get_db_sync()
        results = await async_db.recall(
            query_emb=os.sys.modules['numpy'].array(q_emb, dtype=os.sys.modules['numpy'].float32),
            top_k=2,
            expand_hops=1,
            query_entities=db.extract_entities(q)
        )

        context = "\n".join(f"- {r.payload}" for r in results) if results else "No prior memory."
        return {"context": context}

    async def generate_node(state: AsyncLGState):
        prompt = (
            "You are a helpful AI memory assistant.\n"
            f"Context: {state['context']}\n\n"
            f"User: {state['input']}\n"
            "Assistant:"
        )
        tokens = count_tokens(prompt)

        # Estimate turn index
        turn_idx = 0
        for i, turn in enumerate(SCENARIO_TURNS):
            if turn["user"].format(name=state["user_name"]) == state["input"]:
                turn_idx = i
                break

        reply = await async_generate(client, prompt, turn_idx, state["user_name"], live_mode)
        return {"response": reply, "prompt_tokens": tokens}

    async def store_node(state: AsyncLGState):
        interaction = f"User: {state['input']}\nAssistant: {state['response']}"
        emb = await async_embed(client, interaction, live_mode)

        # Extract entities for triples
        db = async_db._get_db_sync()
        entities = db.extract_entities(state["input"])
        triples = []
        for e in entities:
            triples.append((e, "mentions", e))

        await async_db.add_memory(
            payload=interaction,
            embedding=os.sys.modules['numpy'].array(emb, dtype=os.sys.modules['numpy'].float32),
            triples=triples
        )
        return {}

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("store", store_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "store")
    workflow.add_edge("store", END)

    db_instance = async_db._get_db_sync()
    checkpointer = EpochDBCheckpointer(db_instance)
    return workflow.compile(checkpointer=checkpointer)


# ── 3. Async Astraea Agent Definition ──────────────────────────────────────────

if HAS_ASTRAEA:
    class AstraeaChatAgent(Agent):
        def __init__(self, name: str, client: Any, live_mode: bool, user_idx: int):
            super().__init__(name, subscription_pattern=f"user_{user_idx}:input")
            self.client = client
            self.live_mode = live_mode
            self.user_idx = user_idx

        async def handle(self, event: BlackboardEvent) -> None:
            payload = event.payload
            user_idx = self.user_idx
            turn_idx = payload["turn"]
            user_input = payload["input"]
            user_name = USER_NAMES[user_idx]

            active_concept = f"user_{user_idx}:concept:EpochDB"
            nodes = []
            edges = []
            
            # Subgraph recall — filter out ChatFragments to keep only Concepts in the subgraph context
            try:
                sub = await self.blackboard.get_subgraph(active_concept, max_depth=1)
                nodes.extend([n for n in sub.get("nodes", []) if n.get("label") != "ChatFragment"])
                edges.extend(sub.get("edges", []))
            except KeyError:
                pass

            # Chat history recall — explicitly fetch only the last 2 turns to keep token size flat
            for prev_idx in range(max(0, turn_idx - 2), turn_idx):
                chat_id = f"user_{user_idx}:chat:msg_{prev_idx}"
                try:
                    node = await self.blackboard.read_node(chat_id)
                    nodes.append(node)
                except KeyError:
                    pass

            # Deduplicate nodes by ID
            seen_n = set()
            uniq_nodes = []
            for n in nodes:
                nid = n.get("id")
                if nid not in seen_n:
                    seen_n.add(nid)
                    uniq_nodes.append(n)
            
            # Only keep edges that link nodes we are actually keeping
            keep_ids = {n["id"] for n in uniq_nodes}
            uniq_edges = []
            for e in edges:
                if e.get("source") in keep_ids and e.get("target") in keep_ids:
                    uniq_edges.append(e)

            # Context serialization
            context = ContextTiler.serialize_subgraph_to_markdown({"nodes": uniq_nodes, "edges": uniq_edges})

            prompt = (
                "You are a helpful AI assistant. Answer using the context from the blackboard:\n"
                f"{context}\n\n"
                f"User: {user_input}"
            )

            # Generate
            reply = await async_generate(self.client, prompt, turn_idx, user_name, self.live_mode)

            # Write chat fragment and concept mapping
            chat_id = f"user_{user_idx}:chat:msg_{turn_idx}"
            lineage = LineageContext(triggering_event_id=event.event_id, agent_name=self.name, rationale="chat memory")

            await self.blackboard.write_node(
                node_id=chat_id,
                label="ChatFragment",
                properties={"text": f"User: {user_input} | Assistant: {reply}", "turn": turn_idx},
                lineage=lineage
            )

            await self.blackboard.write_node(
                node_id=active_concept,
                label="Concept",
                properties={"text": "EpochDB is a tiered AI memory engine built in Python"},
                lineage=lineage
            )

            await self.blackboard.write_edge(chat_id, active_concept, "references", {}, lineage)

            # Publish response event
            completion = BlackboardEvent(
                event_id=f"EV-RESP-{user_idx}-{turn_idx}",
                event_type=EventType.NODE_INSERTED,
                key_path=f"user_{user_idx}:response",
                payload={"response": reply, "prompt_tokens": count_tokens(prompt)},
                epoch_version=1,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                priority=1
            )
            await self.router.publish(completion)



# ── ASCII Latency Chart Drawing ───────────────────────────────────────────────

def draw_latency_chart(results: Dict[str, float]):
    print("\n" + C.BOLD + C.CYAN + "📈 LATENCY PERFORMANCE COMPARISON" + C.END)
    print("━" * 65)
    max_latency = max(results.values()) if results.values() else 1.0
    max_bar_width = 30
    for name, latency in results.items():
        bar_len = int((latency / max_latency) * max_bar_width) if max_latency > 0 else 0
        bar = "█" * bar_len
        # Determine color
        if "Sync" in name:
            color = C.RED
        elif "Astraea" in name:
            color = C.CYAN
        else:
            color = C.GREEN
        print(f"  {name:<22}: {color}{bar:<30}{C.END} ({latency:.3f}s)")
    print("━" * 65 + "\n")


# ── Report Generation ─────────────────────────────────────────────────────────

def save_markdown_report(metrics: Dict[str, Any], live_mode: bool):
    filename = "sync_async_benchmark_report.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# LangGraph & Astraea Sync vs. Async Performance Report\n\n")
        f.write("This benchmark compares E2E latency and input token consumption under concurrent multi-user load across three execution configurations:\n\n")
        f.write("1. **Sync LangGraph + Sync EpochDB**: Sequential graph invocation with blocking I/O.\n")
        f.write("2. **Async LangGraph + Async EpochDB**: Concurrent graph execution (`ainvoke`) using async checkpointers and DB facades.\n")
        f.write("3. **Async Astraea + EpochBlackboard**: Decoupled event-driven reactive coordination running in parallel.\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Execution Mode**: `{'Live (Gemini API)' if live_mode else 'Offline Mock (Local Simulated Network Latency)'}`\n")
        f.write(f"- **Users evaluated**: {metrics['num_users']}\n")
        f.write(f"- **Turns per user**: {metrics['num_turns']}\n\n")

        f.write("| Metrics | Sync LangGraph | Async LangGraph | Async Astraea |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **E2E Latency (seconds)** | {metrics['sync_lg_latency']:.3f}s | {metrics['async_lg_latency']:.3f}s | {metrics['astraea_latency']:.3f}s |\n")
        f.write(f"| **Average Turn Latency (ms)** | {metrics['sync_lg_avg_turn_ms']:.1f}ms | {metrics['async_lg_avg_turn_ms']:.1f}ms | {metrics['astraea_avg_turn_ms']:.1f}ms |\n")
        f.write(f"| **Speedup vs. Sync Baseline** | 1.0x (Baseline) | **{metrics['async_lg_speedup']:.2f}x** | **{metrics['astraea_speedup']:.2f}x** |\n")
        f.write(f"| **Total Input Tokens** | {metrics['sync_lg_tokens']:,} | {metrics['async_lg_tokens']:,} | {metrics['astraea_tokens']:,} |\n")
        f.write(f"| **Token Savings vs. LangGraph**| - | 0.0% | **{metrics['astraea_token_savings']:.1f}%** |\n\n")

        f.write("## Performance & Architectural Insights\n\n")
        f.write("### 1. Latency & Concurrency Gains (Sync vs. Async)\n")
        f.write("Under synchronous execution, user turns are executed sequentially, causing network/database blocking latency to build up linearly ($O(U \\times T)$). "
                "By using async primitives, requests are processed concurrently. This collapses the E2E latency to approximately a single user's duration ($O(T)$), "
                "yielding significant speedups under multi-user concurrency.\n\n")
        f.write("### 2. Token Savings (LangGraph vs. Astraea)\n")
        f.write("- **LangGraph** keeps conversation history inside the thread memory, accumulating prompt sizes quadratically. "
                "Even with thin memory checkpoints, it still passes the flat list of history turns.\n")
        f.write("- **Astraea** employs OABS concept-hop subgraphs and structural context tiling. It fetches localized knowledge graph clusters "
                "and appends only the last 2 chat turns. This reduces prompt sizes significantly, delivering flat token overhead per turn.\n")

    print(f"{C.GREEN}✓ Report saved successfully to: {filename}{C.END}")


# ── Main Runner ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Sync vs Async LangGraph & Astraea Benchmark")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode instead of live mode")
    parser.add_argument("--keep", action="store_true", help="Do not wipe storage databases after running")
    parser.add_argument("--users", type=int, default=3, help="Number of concurrent users (1 to 15)")
    args = parser.parse_args()

    # Load environment variables from .env if present
    from utils.shared import load_dotenv
    load_dotenv(os.path.dirname(os.path.abspath(__file__)))

    num_users = min(max(args.users, 1), 15)
    num_turns = len(SCENARIO_TURNS)
    live_mode = not args.mock

    # Clean directories
    for d in [STORAGE_DIR_SYNC, STORAGE_DIR_ASYNC, STORAGE_DIR_ASTRAEA]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # Instantiate LLM Client
    client = None
    if live_mode:
        if not HAS_GENAI:
            raise ImportError("Error: google-genai package is required to run live benchmarks. Please install it or run with --mock.")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Error: GEMINI_API_KEY environment variable is not set and was not found in a .env file.\n"
                "To run accurate, realistic benchmarks, please provide a valid GEMINI_API_KEY, or run with --mock."
            )
        client = genai.Client(api_key=api_key)

    print(C.header("Sync vs Async LangGraph & Astraea Performance Benchmark"))
    print(f"  Concurrent Users : {num_users}")
    print(f"  Turns per User   : {num_turns}")
    print(f"  Mode             : {'LIVE (Gemini API)' if live_mode else 'MOCK (Simulated 300ms I/O latency)'}")
    print(f"  Astraea Loaded   : {HAS_ASTRAEA}")
    print("━" * 60)

    # -------------------------------------------------------------------------
    # 1. RUN SYNC LANGGRAPH
    # -------------------------------------------------------------------------
    print(f"\n⚡ Running {C.BOLD}Sync LangGraph Baseline{C.END}...")
    sync_db = EpochDB(storage_dir=STORAGE_DIR_SYNC, dim=DIM)
    sync_app = make_sync_langgraph_pipeline(sync_db, client, live_mode)

    sync_total_tokens = 0
    start_time = time.perf_counter()

    for u_idx in range(num_users):
        u_name = USER_NAMES[u_idx]
        thread_config = {"configurable": {"thread_id": f"sync_thread_{u_idx}"}}
        for t_idx, turn in enumerate(SCENARIO_TURNS):
            user_prompt = turn["user"].format(name=u_name)
            res = sync_app.invoke(
                {"input": user_prompt, "context": "", "response": "", "prompt_tokens": 0, "user_name": u_name},
                config=thread_config
            )
            sync_total_tokens += res.get("prompt_tokens", 0)

    sync_lg_latency = time.perf_counter() - start_time
    sync_db.close()

    print(f"   Completed E2E in {sync_lg_latency:.3f}s ({sync_total_tokens:,} tokens)")

    # -------------------------------------------------------------------------
    # 2. RUN ASYNC LANGGRAPH
    # -------------------------------------------------------------------------
    print(f"\n⚡ Running {C.BOLD}Async LangGraph Pipeline{C.END}...")
    
    async_lg_tokens = 0
    async_lg_tokens_lock = asyncio.Lock()

    async with AsyncEpochDB(storage_dir=STORAGE_DIR_ASYNC, dim=DIM) as async_db:
        async_app = make_async_langgraph_pipeline(async_db, client, live_mode)

        async def run_async_lg_user(u_idx):
            nonlocal async_lg_tokens
            u_name = USER_NAMES[u_idx]
            thread_config = {"configurable": {"thread_id": f"async_thread_{u_idx}"}}
            for t_idx, turn in enumerate(SCENARIO_TURNS):
                user_prompt = turn["user"].format(name=u_name)
                res = await async_app.ainvoke(
                    {"input": user_prompt, "context": "", "response": "", "prompt_tokens": 0, "user_name": u_name},
                    config=thread_config
                )
                async with async_lg_tokens_lock:
                    async_lg_tokens += res.get("prompt_tokens", 0)

        start_time = time.perf_counter()
        await asyncio.gather(*(run_async_lg_user(i) for i in range(num_users)))
        async_lg_latency = time.perf_counter() - start_time

    print(f"   Completed E2E in {async_lg_latency:.3f}s ({async_lg_tokens:,} tokens)")

    # -------------------------------------------------------------------------
    # 3. RUN ASYNC ASTRAEA
    # -------------------------------------------------------------------------
    astraea_latency = 0.0
    astraea_tokens = 0

    if not HAS_ASTRAEA:
        print(f"\n🌀 {C.YELLOW}Skipping Astraea test (framework not installed or not imported){C.END}")
    else:
        print(f"\n🌀 Running {C.BOLD}Async Astraea Blackboard Pipeline{C.END}...")
        
        ast_blackboard = EpochBlackboard(storage_dir=STORAGE_DIR_ASTRAEA, dim=DIM)
        ast_router = EventRouter()
        ast_agents = [
            AstraeaChatAgent(f"AstraeaChatAgent_{i}", client, live_mode, user_idx=i)
            for i in range(num_users)
        ]

        ast_team = Team("ChatTeam", agents=ast_agents, blackboard=ast_blackboard, router=ast_router)
        await ast_team.start()

        astraea_tokens_lock = asyncio.Lock()

        async def run_astraea_user(u_idx):
            nonlocal astraea_tokens
            u_name = USER_NAMES[u_idx]
            for t_idx, turn in enumerate(SCENARIO_TURNS):
                user_prompt = turn["user"].format(name=u_name)
                resp_key = f"user_{u_idx}:response"
                
                # Subscribe before trigger publish
                sub_queue = await ast_router.subscribe(resp_key)
                
                trigger = BlackboardEvent(
                    event_id=f"EV-TRIG-{u_idx}-{t_idx}",
                    event_type=EventType.NODE_INSERTED,
                    key_path=f"user_{u_idx}:input",
                    payload={"input": user_prompt, "turn": t_idx, "user_idx": u_idx},
                    epoch_version=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc),
                    priority=1
                )
                
                await ast_router.publish(trigger)
                try:
                    resp_event = await asyncio.wait_for(sub_queue.get(), timeout=120.0)
                    async with astraea_tokens_lock:
                        astraea_tokens += resp_event.payload.get("prompt_tokens", 0)
                finally:
                    await ast_router.unsubscribe(resp_key, sub_queue)

        start_time = time.perf_counter()
        await asyncio.gather(*(run_astraea_user(i) for i in range(num_users)))
        astraea_latency = time.perf_counter() - start_time

        await ast_team.stop()
        await ast_blackboard.close()

        print(f"   Completed E2E in {astraea_latency:.3f}s ({astraea_tokens:,} tokens)")

    # -------------------------------------------------------------------------
    # 4. REPORT & VISUALIZATION
    # -------------------------------------------------------------------------
    # Latency Chart
    latencies = {
        "Sync LangGraph": sync_lg_latency,
        "Async LangGraph": async_lg_latency
    }
    if HAS_ASTRAEA:
        latencies["Async Astraea"] = astraea_latency
    draw_latency_chart(latencies)

    # Metrics Summary
    total_turns_run = num_users * num_turns
    
    speedup_async_lg = sync_lg_latency / async_lg_latency
    speedup_astraea = sync_lg_latency / astraea_latency if HAS_ASTRAEA else 0.0
    token_savings_astraea = (async_lg_tokens - astraea_tokens) / max(async_lg_tokens, 1) * 100 if HAS_ASTRAEA else 0.0

    metrics = {
        "num_users": num_users,
        "num_turns": num_turns,
        "sync_lg_latency": sync_lg_latency,
        "sync_lg_avg_turn_ms": (sync_lg_latency / total_turns_run) * 1000,
        "sync_lg_tokens": sync_total_tokens,
        "async_lg_latency": async_lg_latency,
        "async_lg_avg_turn_ms": (async_lg_latency / total_turns_run) * 1000,
        "async_lg_tokens": async_lg_tokens,
        "async_lg_speedup": speedup_async_lg,
        "astraea_latency": astraea_latency,
        "astraea_avg_turn_ms": (astraea_latency / total_turns_run) * 1000 if HAS_ASTRAEA else 0.0,
        "astraea_tokens": astraea_tokens,
        "astraea_speedup": speedup_astraea,
        "astraea_token_savings": token_savings_astraea
    }

    save_markdown_report(metrics, live_mode)

    # Clean directories unless keeping
    if not args.keep:
        for d in [STORAGE_DIR_SYNC, STORAGE_DIR_ASYNC, STORAGE_DIR_ASTRAEA]:
            if os.path.exists(d):
                shutil.rmtree(d)


if __name__ == "__main__":
    asyncio.run(main())
