#!/usr/bin/env python3
"""
benchmark_langgraph_tokens.py — LangGraph Token Savings Benchmark
==================================================================
Quantitatively compares the input token consumption of:
1. Standard LangGraph (Full History Message Buffer)
2. LangGraph + EpochDB (Thin State Graph + Relational/Semantic Recall)

Features:
- Mock Mode: Keyword-based deterministic mock embeddings and generation (runs out-of-the-box).
- Live Mode: Real Gemini API calls (via google-genai) when GEMINI_API_KEY is present.
- ASCII Growth Chart: Text-based line chart comparing token usage over turns.
- Report Generation: Saves detailed breakdown to benchmark_langgraph_results.md.

Usage:
    python examples/benchmark_langgraph_tokens.py [--live] [--keep]
"""

import os
import sys
import time
import shutil
import argparse
import numpy as np
from typing import TypedDict, List, Dict, Any, Tuple

# Suppress noisy warnings
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)

# Ensure absolute imports work from workspace root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epochdb import EpochDB
from epochdb.checkpointer import EpochDBCheckpointer
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Juno framework imports (will be dynamically imported in main using the parsed --juno-path)
EpochBlackboard = None
ContextTiler = None
LineageContext = None
HAS_JUNO = False

def import_juno(juno_path: str):
    global EpochBlackboard, ContextTiler, LineageContext, HAS_JUNO
    if juno_path and os.path.exists(juno_path):
        sys.path.insert(0, os.path.abspath(juno_path))
    try:
        from juno import EpochBlackboard
        from juno.tiler import ContextTiler
        from juno.types import LineageContext
        HAS_JUNO = True
    except ImportError as e:
        HAS_JUNO = False
        print(f"Warning: Could not import Juno framework from {juno_path}: {e}")

# Try importing tabulate for formatted tables
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Try importing tiktoken for token counting
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# Try importing google-genai for live model tests
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# ── Configuration & Colors ───────────────────────────────────────────────────

STORAGE_DIR = "./.epochdb_benchmark_langgraph"
JUNO_DIR    = "./.juno_benchmark_temp"
EMBED_MODEL = "gemini-embedding-2-preview"
GEN_MODEL   = "gemini-3-flash-preview"
DIM         = 3072

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

    @staticmethod
    def subheader(text):
        return f"\n{C.BOLD}{C.BLUE}▶ {text}{C.END}"


# ── Conversational Scenario Data ──────────────────────────────────────────────

SCENARIO_TURNS = [
    {
        "user": "Hi, I'm Jeff. I'm starting a new project called Aegis.",
        "assistant": "Hello Jeff! I've noted that you're starting Project Aegis. Let me know what we are building!",
    },
    {
        "user": "Project Aegis is a distributed graph database focused on sub-millisecond retrieval.",
        "assistant": "Got it. Project Aegis is a distributed graph database designed for sub-millisecond retrieval performance.",
    },
    {
        "user": "We will build the Aegis database using the Rust programming language for performance and memory safety.",
        "assistant": "Excellent choice. Building the Aegis database in Rust ensures high performance and memory safety.",
    },
    {
        "user": "The storage layer of Aegis is based on LSM-trees and uses a custom write-ahead log (WAL).",
        "assistant": "Understood. The Aegis storage layer uses LSM-trees and a custom write-ahead log (WAL) for durability and fast write throughput.",
    },
    {
        "user": "We are targeting a launch date of October 2026.",
        "assistant": "Recorded. The launch target for Project Aegis is October 2026.",
    },
    {
        "user": "What is the launch date we are targeting for our project?",
        "assistant": "We are targeting October 2026 as the launch date for Project Aegis.",
    },
    {
        "user": "What language are we using to build the Aegis database?",
        "assistant": "We are building the Aegis database using the Rust programming language for performance and memory safety.",
    },
    {
        "user": "Can you explain what storage layer and architecture Aegis uses?",
        "assistant": "Aegis uses a storage layer based on LSM-trees and a custom write-ahead log (WAL) for write efficiency and durability.",
    },
    {
        "user": "Who is starting the project and what is its name?",
        "assistant": "Jeff is starting the project, and its name is Project Aegis.",
    },
    {
        "user": "Summarize the database project we are working on.",
        "assistant": "Project Aegis is a distributed graph database started by Jeff, built in Rust for performance and memory safety. It features a storage layer with LSM-trees, a custom WAL, and is scheduled for launch in October 2026.",
    }
]

# Keyword list for mock vector simulation
KEYWORDS_LIST = [
    "jeff", "aegis", "database", "performance", "rust", "storage", "lsm", "wal",
    "launch", "october", "2026", "language", "who", "starting", "name", "summarize"
]


# ── Token Counter & Mock Helpers ──────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Estimate token count using tiktoken (cl100k_base) or character count fallback."""
    if HAS_TIKTOKEN:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    return int(len(text) / 4.0)


def get_mock_embedding(text: str) -> np.ndarray:
    """Generate a mock vector with higher cosine similarity for overlapping keywords."""
    vec = np.zeros(DIM, dtype=np.float32)
    text_lower = text.lower()
    for i, kw in enumerate(KEYWORDS_LIST):
        if kw in text_lower:
            vec[i] = 1.0
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def get_client(live_mode: bool):
    """Retrieve Gemini client if in live mode."""
    if live_mode:
        if not HAS_GENAI:
            print(f"{C.RED}Error: google-genai is not installed. Running in Mock Mode.{C.END}")
            return None
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print(f"{C.RED}Error: GEMINI_API_KEY is not set. Running in Mock Mode.{C.END}")
            return None
        return genai.Client(api_key=api_key)
    return None


def embed(client, text: str, live_mode: bool) -> np.ndarray:
    """Return embedding vector (real or mock)."""
    if live_mode and client:
        resp = client.models.embed_content(model=EMBED_MODEL, contents=text)
        return np.array(resp.embeddings[0].values, dtype=np.float32)
    return get_mock_embedding(text)


def get_response(client, prompt: str, turn_idx: int, live_mode: bool) -> str:
    """Generate assistant response (real or mock)."""
    if live_mode and client:
        try:
            resp = client.models.generate_content(model=GEN_MODEL, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            print(f"{C.RED}Live Generation failed: {e}. Using mock fallback.{C.END}")
    return SCENARIO_TURNS[turn_idx]["assistant"]


# ── Baseline Graph (Without EpochDB) ──────────────────────────────────────────

class BaselineState(TypedDict):
    input: str
    messages: List[dict]
    response: str
    prompt_tokens: int


def make_baseline_graph(client, live_mode: bool):
    workflow = StateGraph(BaselineState)

    def generate_node(state: BaselineState):
        system_prompt = "You are a helpful AI assistant. Answer the user's latest query using the context from the conversation history."
        history_str = ""
        for msg in state.get("messages", []):
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        prompt = f"{system_prompt}\n\nConversation History:\n{history_str}User: {state['input']}\nAssistant:"
        tokens = count_tokens(prompt)

        # Estimate turn index
        turn_idx = len(state.get("messages", [])) // 2
        response = get_response(client, prompt, turn_idx, live_mode)

        new_messages = list(state.get("messages", []))
        new_messages.append({"role": "user", "content": state["input"]})
        new_messages.append({"role": "assistant", "content": response})

        return {
            "response": response,
            "messages": new_messages,
            "prompt_tokens": tokens
        }

    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ── Optimized Graph (With EpochDB) ────────────────────────────────────────────

class EpochDBState(TypedDict):
    input: str
    context: str
    response: str
    prompt_tokens: int


def make_epochdb_graph(db: EpochDB, client, live_mode: bool):
    workflow = StateGraph(EpochDBState)

    def retrieve_node(state: EpochDBState):
        latest_input = state["input"]
        q_emb = embed(client, latest_input, live_mode)

        q_entities = [
            w.strip(".,?!")
            for w in latest_input.split()
            if w[0].isupper() or w.lower() in ("aegis", "lapis", "rust", "wal", "lsm")
        ]

        results = db.recall(
            q_emb,
            top_k=2,
            expand_hops=2,
            query_entities=q_entities
        )

        if results:
            context = "\n".join(f"- {r.payload}" for r in results)
        else:
            context = "No prior memory."

        return {"context": context}

    def generate_node(state: EpochDBState):
        latest_input = state["input"]
        context = state["context"]

        prompt = (
            "You are a helpful AI assistant with perfect long-term memory powered by EpochDB.\n"
            "Answer the user's query using the retrieved context from long-term memory if relevant.\n\n"
            "Retrieved memory context:\n"
            f"{context}\n\n"
            f"User: {latest_input}\n"
            "Assistant:"
        )
        tokens = count_tokens(prompt)

        # Estimate turn index
        turn_idx = 0
        for i, turn in enumerate(SCENARIO_TURNS):
            if turn["user"] == latest_input:
                turn_idx = i
                break

        response = get_response(client, prompt, turn_idx, live_mode)
        return {"response": response, "prompt_tokens": tokens}

    def store_node(state: EpochDBState):
        latest_input = state["input"]
        response = state["response"]
        interaction = f"User: {latest_input}\nAgent: {response}"
        emb = embed(client, interaction, live_mode)

        # Heuristic triples
        triples = []
        tl = latest_input.lower()
        if "jeff" in tl:
            triples.append(("user", "has_name", "Jeff"))
        if "aegis" in tl:
            triples.append(("user", "works_on", "Project Aegis"))
            triples.append(("Project Aegis", "is_a", "database"))
        if "rust" in tl:
            triples.append(("Project Aegis", "built_with", "Rust"))
        if "lsm" in tl:
            triples.append(("Project Aegis", "uses_storage", "LSM-trees"))
        if "lapis" in tl:
            triples.append(("Project Aegis", "has_storage_engine", "Lapis"))
            triples.append(("Lapis", "uses", "LSM-trees"))
        if "wal" in tl:
            triples.append(("Project Aegis", "uses_log", "WAL"))
        if "launch" in tl or "2026" in tl:
            triples.append(("Project Aegis", "launch_target", "October 2026"))

        db.add_memory(payload=interaction, embedding=emb, triples=triples)
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


# ── Juno Blackboard Scenario Turn ─────────────────────────────────────────────

async def run_juno_turn(board: Any, client: Any, query: str, turn_idx: int, live_mode: bool) -> int:
    """Simulate a Juno agent turn using EpochBlackboard and ContextTiler, returning prompt tokens."""
    # 1. Identify active concepts based on query keywords
    active_concepts = []
    ql = query.lower()
    if "jeff" in ql or "aegis" in ql:
        active_concepts.append("global:concept:Aegis")
    if "rust" in ql:
        active_concepts.append("global:concept:Rust")
    if "lsm" in ql:
        active_concepts.append("global:concept:LSM")
    if "wal" in ql:
        active_concepts.append("global:concept:WAL")
    if "launch" in ql or "2026" in ql:
        active_concepts.append("global:concept:launch")

    # Fallback default
    if not active_concepts:
        active_concepts.append("global:concept:Aegis")

    # 2. Retrieve subgraphs for active concepts
    nodes = []
    edges = []
    visited_nodes = set()
    lineage = LineageContext(
        triggering_event_id=f"EV-turn-{turn_idx}",
        agent_name="JunoAgent",
        rationale="Retrieval for chat context",
    )

    for concept_id in active_concepts:
        try:
            sub = await board.get_subgraph(concept_id, max_depth=1)
            for n in sub.get("nodes", []):
                nid = n["id"]
                # Avoid retrieving chat fragments via the concept subgraph
                if n.get("label") == "ChatFragment":
                    continue
                if nid not in visited_nodes:
                    visited_nodes.add(nid)
                    nodes.append(n)
            for e in sub.get("edges", []):
                # Avoid retrieving chat-to-concept relations via the concept subgraph
                if "chat" in e.get("source", "").lower() or "chat" in e.get("target", "").lower():
                    continue
                edges.append(e)
        except KeyError:
            # Concept node not created yet (expected in Turn 1 prior to write)
            pass

    # 3. Retrieve last 2 chat turns from blackboard node registry
    recent_chats = []
    for offset in [2, 1]:
        prev_idx = turn_idx - offset
        if prev_idx >= 0:
            cid = f"global:chat:msg_{prev_idx}"
            node_info = board._node_registry.get(cid)
            if node_info:
                recent_chats.append(node_info)

    for c_node in recent_chats:
        if c_node["id"] not in visited_nodes:
            visited_nodes.add(c_node["id"])
            nodes.append(c_node)

    # 4. Serialize using ContextTiler
    serialized_context = ContextTiler.serialize_subgraph_to_markdown({
        "nodes": nodes,
        "edges": edges
    })

    # 5. Build prompt & estimate tokens
    prompt = (
        "You are a helpful AI assistant with perfect memory powered by Juno.\n"
        "Answer the user's query using the context from the blackboard:\n\n"
        f"{serialized_context}\n\n"
        f"User: {query}\n"
        "Assistant:"
    )
    tokens = count_tokens(prompt)

    # 6. Generate Response
    response = get_response(client, prompt, turn_idx, live_mode)

    # 7. Write interaction node and concept connections back to Blackboard
    vec = embed(client, query, live_mode)
    juno_vec = list(vec)  # EpochBlackboard requires list type for vector

    # Ensure concept nodes are written
    for concept_id in active_concepts:
        concept_name = concept_id.split(":")[-1]
        if concept_id not in board._node_registry:
            await board.write_node(
                node_id=concept_id,
                label="Concept",
                properties={"text": f"Concept definition for {concept_name}"},
                vector=juno_vec,
                lineage=lineage
            )

    # Write ChatFragment
    chat_id = f"global:chat:msg_{turn_idx}"
    interaction = f"User: {query} | Assistant: {response}"
    await board.write_node(
        node_id=chat_id,
        label="ChatFragment",
        properties={"text": interaction, "turn": turn_idx},
        vector=juno_vec,
        lineage=lineage
    )

    # Connect ChatFragment to concepts
    for concept_id in active_concepts:
        await board.write_edge(chat_id, concept_id, "references", {}, lineage)

    # Connect sequential chat messages
    if turn_idx > 0:
        prev_chat_id = f"global:chat:msg_{turn_idx-1}"
        await board.write_edge(prev_chat_id, chat_id, "next_turn", {}, lineage)

    # Write structural learning edges between concepts
    if "rust" in ql:
        await board.write_edge("global:concept:Aegis", "global:concept:Rust", "built_with", {}, lineage)
    if "lsm" in ql:
        await board.write_edge("global:concept:Aegis", "global:concept:LSM", "uses_storage", {}, lineage)
    if "wal" in ql:
        await board.write_edge("global:concept:Aegis", "global:concept:WAL", "uses_log", {}, lineage)
    if "launch" in ql or "2026" in ql:
        await board.write_edge("global:concept:Aegis", "global:concept:launch", "launch_target", {}, lineage)

    return tokens


# ── ASCII Chart Drawing ───────────────────────────────────────────────────────

def draw_ascii_chart(turns: List[int], baseline: List[int], epochdb: List[int], juno: List[int]):
    """Draw a beautiful terminal ASCII chart comparing token growth."""
    print(f"\n{C.BOLD}{C.CYAN}📈 Input Token Growth Chart (over turns){C.END}")

    max_val = max(max(baseline), max(epochdb), max(juno))
    height = 10
    width = 50

    # Scale data to dimensions
    scaled_base = [int((val / max_val) * (height - 1)) for val in baseline]
    scaled_ep = [int((val / max_val) * (height - 1)) for val in epochdb]
    scaled_jn = [int((val / max_val) * (height - 1)) for val in juno]

    grid = [[" " for _ in range(width)] for _ in range(height)]
    num_points = len(turns)

    for col in range(width):
        idx = int((col / (width - 1)) * (num_points - 1))
        b_row = height - 1 - scaled_base[idx]
        e_row = height - 1 - scaled_ep[idx]
        j_row = height - 1 - scaled_jn[idx]

        grid[b_row][col] = "B"

        if grid[e_row][col] == "B":
            grid[e_row][col] = "X"
        else:
            grid[e_row][col] = "E"

        if grid[j_row][col] in ("B", "E", "X"):
            grid[j_row][col] = "X"
        else:
            grid[j_row][col] = "J"

    print("      ┌" + "─" * width + "┐")
    for r in range(height):
        val = max_val - (r / (height - 1)) * max_val
        print(f"{int(val):5d} │", end="")
        for c in range(width):
            char = grid[r][c]
            if char == "B":
                print(f"{C.RED}B{C.END}", end="")
            elif char == "E":
                print(f"{C.GREEN}E{C.END}", end="")
            elif char == "J":
                print(f"{C.CYAN}J{C.END}", end="")
            elif char == "X":
                print(f"{C.YELLOW}X{C.END}", end="")
            else:
                print(char, end="")
        print("│")
    print("      └" + "─" * width + "┘")
    print("        " + " ".join([f"Turn {turns[int(p * (num_points-1))]}" for p in [0.0, 0.25, 0.5, 0.75, 1.0]]))
    print(f"        (Legend: {C.RED}B{C.END} = Standard LangGraph, {C.GREEN}E{C.END} = LangGraph + EpochDB, {C.CYAN}J{C.END} = Juno (+ EpochDB), {C.YELLOW}X{C.END} = Overlap)")


# ── Report Generation ─────────────────────────────────────────────────────────

def save_report(results: List[dict], total_base: int, total_ep: int, total_jn: int, live_mode: bool):
    filename = "examples/benchmark_langgraph_results.md"
    savings_ep = (total_base - total_ep) / total_base * 100
    savings_jn = (total_base - total_jn) / total_base * 100

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# LangGraph vs. EpochDB vs. Juno Token Savings Benchmark\n\n")
        f.write("This benchmark compares the cumulative input token footprint of three conversational agent architectures:\n")
        f.write("1. **Standard LangGraph**: Flat, full-history message buffer ($O(N^2)$ cumulative token growth).\n")
        f.write("2. **LangGraph + EpochDB**: Thin state with selective semantic recall ($O(N)$ linear token growth).\n")
        f.write("3. **Juno (+ EpochDB)**: Dynamic blackboard workspace with targeted subgraph tiling ($O(N)$ linear token growth with structural context).\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Execution Mode**: `{'Live (Gemini API)' if live_mode else 'Offline Mock (Local Keyword Embeddings)'}`\n")
        f.write(f"- **Standard LangGraph Input Tokens**: **{total_base:,}**\n")
        f.write(f"- **LangGraph + EpochDB Input Tokens**: **{total_ep:,}** (Savings vs. Standard: **{savings_ep:.1f}%**)\n")
        f.write(f"- **Juno (+ EpochDB) Input Tokens**: **{total_jn:,}** (Savings vs. Standard: **{savings_jn:.1f}%**)\n\n")

        f.write("## Token Cost Comparison Table\n\n")
        f.write("| Turn | User Query | Standard LangGraph (Tokens) | LangGraph + EpochDB (Tokens) | Juno (+ EpochDB) (Tokens) | Savings EpochDB (%) | Savings Juno (%) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")

        for r in results:
            sav_ep = (r["baseline_tokens"] - r["epochdb_tokens"]) / r["baseline_tokens"] * 100
            sav_jn = (r["baseline_tokens"] - r["juno_tokens"]) / r["baseline_tokens"] * 100
            f.write(f"| {r['turn']} | *\"{r['query']}\"* | {r['baseline_tokens']:,} | {r['epochdb_tokens']:,} | {r['juno_tokens']:,} | {sav_ep:.1f}% | {sav_jn:.1f}% |\n")

        f.write(f"| **TOTAL** | | **{total_base:,}** | **{total_ep:,}** | **{total_jn:,}** | **{savings_ep:.1f}%** | **{savings_jn:.1f}%** |\n\n")

        f.write("## Architectural Analysis\n\n")
        f.write("### 1. The Context Bloat Problem (Standard LangGraph)\n")
        f.write("Standard LangGraph persists the entire user-assistant thread history in the graph state. In every turn, the complete conversation history is sent to the LLM. This leads to **quadratic growth ($O(N^2)$)** in cumulative input token usage:\n\n")
        f.write("$$\\text{Cumulative Tokens} \\approx \\sum_{i=1}^{N} (\\text{System Prompt} + i \\times \\text{Turn Size})$$\n\n")

        f.write("### 2. The EpochDB Solution\n")
        f.write("By using EpochDB alongside `EpochDBCheckpointer`:\n")
        f.write("- **Thin State**: The LangGraph state only holds the immediate turn context, eliminating $O(N)$ historical growth in the state.\n")
        f.write("- **Precise Long-Term Recall**: The agent queries EpochDB using semantic and relational triple-hop logic, retrieving only the **top-2 relevant context items**.\n")
        f.write("- **Flat Growth ($O(N)$)**: Cumulative tokens grow linearly, keeping per-turn prompt size constant regardless of conversation depth.\n\n")

        f.write("### 3. The Juno Solution\n")
        f.write("Juno uses an `EpochBlackboard` surface and `ContextTiler` to represent dynamic agent state:\n")
        f.write("- **Workspace-scoped Graph**: Chat fragments and facts are stored as structured nodes/edges rather than raw text buffers.\n")
        f.write("- **Targeted Subgraph Tiling**: Instead of flat top-k vector search, Juno pulls a localized subgraph around active query concepts (depth=1) and appends only the last 2 chat turns.\n")
        f.write("- **Attributed Markdown**: Results are tiled into structured markdown blocks. This yields highly optimized context footprints and stable linear growth.\n")

    print(f"\n{C.GREEN}✓ Benchmark results report successfully saved to: {filename}{C.END}")


# ── Main Runner ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="LangGraph vs EpochDB vs Juno Token Benchmark")
    parser.add_argument("--live", action="store_true", help="Run with real Gemini API embeddings & generation")
    parser.add_argument("--keep", action="store_true", help="Keep the benchmark storage directories after execution")
    
    # Resolve default juno path dynamically relative to this script
    default_juno_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "juno_framework")
    )
    parser.add_argument(
        "--juno-path",
        type=str,
        default=default_juno_path,
        help="Path to the juno_framework directory"
    )
    args = parser.parse_args()

    # Import Juno framework from the specified path
    import_juno(args.juno_path)
    if not HAS_JUNO:
        print(f"{C.RED}Error: Juno framework is required to run this benchmark. Please check --juno-path.{C.END}")
        sys.exit(1)

    # Clean storage directories
    for p in [STORAGE_DIR, JUNO_DIR]:
        if os.path.exists(p):
            shutil.rmtree(p)

    live_mode = args.live
    client = None
    if live_mode:
        client = get_client(live_mode)
        if not client:
            live_mode = False
            print(f"{C.YELLOW}Falling back to Mock Mode...{C.END}")

    print(C.header("EpochDB vs. LangGraph vs. Juno Token Savings Benchmark"))
    print(f"  Mode:            {'LIVE (Gemini API)' if live_mode else 'MOCK (Keyword-Seeded Vectors)'}")
    print(f"  Token Estimator: {'tiktoken (cl100k_base)' if HAS_TIKTOKEN else 'Character count / 4'}")
    print(f"  Storage Dir:     {STORAGE_DIR}")
    print(f"  Juno Dir:        {JUNO_DIR}")
    print("━" * 60)

    # Initialize EpochDB for LangGraph+EpochDB
    db = EpochDB(storage_dir=STORAGE_DIR, dim=DIM, model=f"google:{EMBED_MODEL}" if live_mode else None)

    # Initialize EpochBlackboard for Juno
    juno_board = EpochBlackboard(
        storage_dir=JUNO_DIR,
        dim=DIM,
        model=f"google:{EMBED_MODEL}" if live_mode else None
    )

    # Compile Workflows
    baseline_app = make_baseline_graph(client, live_mode)
    epochdb_app = make_epochdb_graph(db, client, live_mode)

    baseline_thread = {"configurable": {"thread_id": "baseline_bench"}}
    epochdb_thread = {"configurable": {"thread_id": "epochdb_bench"}}

    results = []
    total_base = 0
    total_ep = 0
    total_jn = 0

    print(f"\n{C.BOLD}Executing simulated conversation turns...{C.END}\n")

    for idx, turn in enumerate(SCENARIO_TURNS):
        turn_num = idx + 1
        query = turn["user"]

        print(f" {C.CYAN}[Turn {turn_num:02d}]{C.END} User: \"{query[:45]}...\"")

        # 1. Run Standard LangGraph
        res_base = baseline_app.invoke({"input": query}, config=baseline_thread)
        base_tokens = res_base["prompt_tokens"]

        # 2. Run LangGraph + EpochDB
        res_ep = epochdb_app.invoke({"input": query}, config=epochdb_thread)
        ep_tokens = res_ep["prompt_tokens"]

        # 3. Run Juno (+ EpochDB)
        jn_tokens = await run_juno_turn(juno_board, client, query, idx, live_mode)

        sav_ep = (base_tokens - ep_tokens) / base_tokens * 100
        sav_jn = (base_tokens - jn_tokens) / base_tokens * 100

        print(f"         ├─ Standard: {base_tokens:5,} tokens")
        print(f"         ├─ EpochDB:  {ep_tokens:5,} tokens  ({C.GREEN}-{sav_ep:.1f}%{C.END})")
        print(f"         ├─ Juno:     {jn_tokens:5,} tokens  ({C.CYAN}-{sav_jn:.1f}%{C.END})")

        results.append({
            "turn": turn_num,
            "query": query,
            "baseline_tokens": base_tokens,
            "epochdb_tokens": ep_tokens,
            "juno_tokens": jn_tokens
        })

        total_base += base_tokens
        total_ep += ep_tokens
        total_jn += jn_tokens

    # Close databases
    db.close()
    await juno_board.close()

    # Clean up storage if not keeping it
    if not args.keep:
        for p in [STORAGE_DIR, JUNO_DIR]:
            if os.path.exists(p):
                shutil.rmtree(p)

    # Report results in console
    print("\n" + "━" * 60)
    print(f"{C.BOLD}🏆 BENCHMARK RESULTS SUMMARY{C.END}")
    print("━" * 60)
    print(f"  Standard LangGraph Input Tokens:  {total_base:8,}")
    print(f"  LangGraph + EpochDB Input Tokens: {total_ep:8,}")
    print(f"  Juno (+ EpochDB) Input Tokens:    {total_jn:8,}")
    savings_ep = (total_base - total_ep) / total_base * 100
    savings_jn = (total_base - total_jn) / total_base * 100
    print(f"  {C.BOLD}EpochDB Input Token Savings:      {C.GREEN}{savings_ep:.1f}%{C.END}")
    print(f"  {C.BOLD}Juno Input Token Savings:         {C.CYAN}{savings_jn:.1f}%{C.END}")
    print("━" * 60)

    # Draw Chart
    turns_list = [r["turn"] for r in results]
    base_list = [r["baseline_tokens"] for r in results]
    ep_list = [r["epochdb_tokens"] for r in results]
    jn_list = [r["juno_tokens"] for r in results]
    draw_ascii_chart(turns_list, base_list, ep_list, jn_list)

    # Save Markdown Report
    save_report(results, total_base, total_ep, total_jn, live_mode)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
