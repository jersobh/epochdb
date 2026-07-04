import os
import sys
import asyncio
from typing import Optional, Dict, Any, List
from mcp.server.fastmcp import FastMCP
from epochdb.api.db import EpochDB

# Initialize FastMCP server
mcp = FastMCP("EpochDB")

# Global reference for lazily loaded DB instance
_db: Optional[EpochDB] = None

def get_db() -> EpochDB:
    global _db
    if _db is None:
        storage_dir = os.getenv("EPOCHDB_STORAGE_DIR", "./.epochdb_data")
        model = os.getenv("EPOCHDB_MODEL", "all-MiniLM-L6-v2")
        tenant = os.getenv("EPOCHDB_TENANT")
        namespace = os.getenv("EPOCHDB_NAMESPACE")
        _db = EpochDB(
            storage_dir=storage_dir,
            embedding_model=model,
            tenant=tenant,
            namespace=namespace
        )
    return _db

@mcp.tool()
def epochdb_remember(text: str, metadata: Optional[dict] = None, memory_type: Optional[str] = None) -> str:
    """
    Store a new memory, knowledge atom, or fact in EpochDB for long-term retrieval.
    
    Args:
        text: The text content of the memory or fact.
        metadata: Optional metadata dict.
        memory_type: Optional memory type ('general', 'episodic', 'profile', 'working').
    """
    db = get_db()
    return db.remember(text, metadata=metadata, memory_type=memory_type)

@mcp.tool()
def epochdb_query(query: str, k: int = 5, min_score: float = 0.0, memory_type: Optional[str] = None, context_window: int = 0) -> list:
    """
    Search and retrieve semantically relevant memories from EpochDB based on a search query.
    
    Args:
        query: The semantic search query.
        k: The number of relevant memories to retrieve.
        min_score: Minimum similarity score threshold (0.0 to 1.0).
        memory_type: Optional filter by memory type ('general', 'episodic', 'profile', 'working').
        context_window: Optional number of context turns to retrieve around the matched memory.
    """
    db = get_db()
    memories = db.query(query, k=k, min_score=min_score, memory_type=memory_type, context_window=context_window)
    return [
        {
            "id": m.id,
            "text": m.text,
            "metadata": m.metadata,
            "created_at": m.created_at,
            "access_count": m.access_count,
            "triples": m.triples,
            "payload_type": m.payload_type,
            "memory_type": m.memory_type,
            "namespace": m.namespace,
        }
        for m in memories
    ]

@mcp.tool()
def epochdb_multi_hop(query: str, hops: int = 2, k: int = 5, context_window: int = 0) -> list:
    """
    Retrieve memories related to a query using multi-hop relational search across the global entity graph.
    Use this for complex questions that require connecting multiple facts.
    
    Args:
        query: The multi-hop relational search query.
        hops: Number of relationship hops to traverse in the entity graph.
        k: Number of relevant memories to return.
        context_window: Optional number of context turns to retrieve around the matched memory.
    """
    db = get_db()
    memories = db.multi_hop(query, hops=hops, k=k, context_window=context_window)
    return [
        {
            "id": m.id,
            "text": m.text,
            "metadata": m.metadata,
            "created_at": m.created_at,
            "access_count": m.access_count,
            "triples": m.triples,
            "payload_type": m.payload_type,
            "memory_type": m.memory_type,
            "namespace": m.namespace,
        }
        for m in memories
    ]

@mcp.tool()
def epochdb_adaptive_query(query: str, k: int = 5, context_window: int = 0) -> list:
    """
    Intelligently routes a query to the optimal engine(s) (semantic, relational, temporal, or quantitative) 
    using LLM-orchestrated routing (or local rule fallbacks if offline) and retrieves relevant memories.
    
    Args:
        query: The natural language search query.
        k: The number of relevant memories to retrieve.
        context_window: Number of chronological context turns to retrieve before and after the matched memory.
    """
    db = get_db()
    memories = db.adaptive_query(query, k=k, context_window=context_window)
    return [
        {
            "id": m.id,
            "text": m.text,
            "metadata": m.metadata,
            "created_at": m.created_at,
            "access_count": m.access_count,
            "triples": m.triples,
            "payload_type": m.payload_type,
            "memory_type": m.memory_type,
            "namespace": m.namespace,
        }
        for m in memories
    ]

@mcp.tool()
def epochdb_get_timeline(entity_id: Optional[str] = None, start: Optional[float] = None, end: Optional[float] = None) -> list:
    """
    Retrieve the chronological history and timeline of memories associated with a specific entity ID.
    
    Args:
        entity_id: The unique ID of the entity. If omitted, returns timeline of all memories.
        start: Optional start timestamp.
        end: Optional end timestamp.
    """
    db = get_db()
    memories = db.get_timeline(entity_id=entity_id, start=start, end=end)
    return [
        {
            "id": m.id,
            "text": m.text,
            "metadata": m.metadata,
            "created_at": m.created_at,
            "access_count": m.access_count,
            "triples": m.triples,
            "payload_type": m.payload_type,
            "memory_type": m.memory_type,
            "namespace": m.namespace,
        }
        for m in memories
    ]

@mcp.tool()
def epochdb_entity_graph(entity_id: str, depth: int = 2) -> dict:
    """
    Construct and retrieve the local entity graph representation (nodes and edges) around a specific entity ID.
    
    Args:
        entity_id: The unique identifier of the central entity.
        depth: Relational depth to traverse in the graph.
    """
    db = get_db()
    graph_obj = db.entity_graph(entity_id, depth=depth)
    return {
        "nodes": getattr(graph_obj, "nodes", []),
        "edges": getattr(graph_obj, "edges", [])
    }

@mcp.tool()
def epochdb_update(memory_id: str, text: Optional[str] = None, metadata: Optional[dict] = None) -> str:
    """
    Update the text or metadata of an existing memory in EpochDB by its ID.
    
    Args:
        memory_id: The unique ID of the memory to update.
        text: Optional new text content for the memory.
        metadata: Optional metadata dict to merge with the existing metadata.
    """
    db = get_db()
    db.update(memory_id, text=text, metadata=metadata)
    return f"Memory {memory_id} updated successfully."

@mcp.tool()
def epochdb_delete(memory_id: str, hard: bool = False) -> str:
    """
    Delete a memory from EpochDB by its ID.
    
    Args:
        memory_id: The unique ID of the memory to delete.
        hard: If True, permanently delete the memory. If False, soft-deletes.
    """
    db = get_db()
    db.delete(memory_id, hard=hard)
    return f"Memory {memory_id} deleted successfully."

@mcp.tool()
def epochdb_analyze(text: str) -> list:
    """
    Extract entity relationship triples (subject, predicate, object) from a given text.
    
    Args:
        text: The text content to analyze.
    """
    db = get_db()
    triples = db.analyze(text)
    return [{"subject": t[0], "predicate": t[1], "object": t[2]} for t in triples]

def main():
    mcp.run()

if __name__ == "__main__":
    main()
