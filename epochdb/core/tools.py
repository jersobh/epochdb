import asyncio
import inspect
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

# Gracefully handle langchain_core dependency.
# Since langgraph/langchain are optional, we raise an ImportError with guidance if they are not installed.
try:
    from langchain_core.tools import StructuredTool
except ImportError:
    raise ImportError(
        "langchain-core is required to use epochdb tools. "
        "Please install it via `pip install langchain-core` or `pip install epochdb[langgraph]`."
    )


# --- Schemas ---

class RememberInput(BaseModel):
    text: str = Field(description="The text content of the memory or fact to store.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata associated with the memory, e.g. custom tags or entity triples."
    )


class QueryInput(BaseModel):
    query: str = Field(description="The semantic search query.")
    k: int = Field(default=5, description="Number of relevant memories to retrieve.")
    min_score: float = Field(default=0.0, description="Minimum similarity score threshold (0.0 to 1.0).")


class MultiHopInput(BaseModel):
    query: str = Field(description="The multi-hop relational search query.")
    hops: int = Field(default=2, description="Number of relationship hops to traverse in the entity graph.")
    k: int = Field(default=5, description="Number of relevant memories to return.")


class TimelineInput(BaseModel):
    entity_id: Optional[str] = Field(
        default=None,
        description="The unique identifier of the entity. If omitted, returns timeline of all memories."
    )
    start: Optional[Union[float, str]] = Field(default=None, description="Start timestamp (float) or ISO string.")
    end: Optional[Union[float, str]] = Field(default=None, description="End timestamp (float) or ISO string.")


class GraphInput(BaseModel):
    entity_id: str = Field(description="The unique identifier of the central entity.")
    depth: int = Field(default=2, description="Relational depth to traverse in the graph.")


class UpdateInput(BaseModel):
    memory_id: str = Field(description="The unique ID of the memory to update.")
    text: Optional[str] = Field(default=None, description="Optional new text content for the memory.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata dict to merge with the existing metadata."
    )


class DeleteInput(BaseModel):
    memory_id: str = Field(description="The unique ID of the memory to delete.")
    hard: bool = Field(default=False, description="If True, permanently delete the memory. If False, soft-deletes.")


# --- Helpers ---

def _serialize_memory(m: Any) -> Dict[str, Any]:
    return {
        "id": getattr(m, "id", None),
        "text": getattr(m, "text", ""),
        "metadata": getattr(m, "metadata", {}),
        "created_at": getattr(m, "created_at", None),
        "access_count": getattr(m, "access_count", 0),
        "triples": getattr(m, "triples", []),
        "payload_type": getattr(m, "payload_type", "text"),
    }


def _parse_time(t: Optional[Union[float, str]]) -> Optional[Any]:
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return datetime.fromtimestamp(t, tz=timezone.utc)
    if isinstance(t, str):
        try:
            return datetime.fromtimestamp(float(t), tz=timezone.utc)
        except ValueError:
            pass
        try:
            import dateutil.parser
            return dateutil.parser.parse(t)
        except Exception:
            raise ValueError(f"Could not parse time format: {t}")
    return t


def get_epochdb_tools(db: Any) -> List[StructuredTool]:
    """
    Generate a list of LangChain StructuredTools for the given EpochDB or AsyncEpochDB instance.
    
    Args:
        db: An instance of EpochDB or AsyncEpochDB.
        
    Returns:
        A list of 7 tools configured for sync and async execution.
    """
    # Detect if we are using AsyncEpochDB or synchronous EpochDB
    is_async = hasattr(db, "_get_db_sync") or (
        hasattr(db, "remember") and inspect.iscoroutinefunction(db.remember)
    )

    # 1. epochdb_remember
    def remember(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if is_async:
            return db._get_db_sync().remember(text, metadata)
        return db.remember(text, metadata)

    async def aremember(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if is_async:
            return await db.remember(text, metadata)
        return await asyncio.to_thread(db.remember, text, metadata)

    remember_tool = StructuredTool.from_function(
        func=remember,
        coroutine=aremember,
        name="epochdb_remember",
        description="Store a new memory, knowledge atom, or fact in EpochDB for long-term retrieval.",
        args_schema=RememberInput
    )

    # 2. epochdb_query
    def query(query: str, k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        if is_async:
            memories = db._get_db_sync().query(query, k=k, min_score=min_score)
        else:
            memories = db.query(query, k=k, min_score=min_score)
        return [_serialize_memory(m) for m in memories]

    async def aquery(query: str, k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        if is_async:
            memories = await db.query(query, k=k, min_score=min_score)
        else:
            memories = await asyncio.to_thread(db.query, query, k=k, min_score=min_score)
        return [_serialize_memory(m) for m in memories]

    query_tool = StructuredTool.from_function(
        func=query,
        coroutine=aquery,
        name="epochdb_query",
        description="Search and retrieve semantically relevant memories from EpochDB based on a search query.",
        args_schema=QueryInput
    )

    # 3. epochdb_multi_hop
    def multi_hop(query: str, hops: int = 2, k: int = 5) -> List[Dict[str, Any]]:
        if is_async:
            memories = db._get_db_sync().multi_hop(query, hops=hops, k=k)
        else:
            memories = db.multi_hop(query, hops=hops, k=k)
        return [_serialize_memory(m) for m in memories]

    async def amulti_hop(query: str, hops: int = 2, k: int = 5) -> List[Dict[str, Any]]:
        if is_async:
            memories = await db.multi_hop(query, hops=hops, k=k)
        else:
            memories = await asyncio.to_thread(db.multi_hop, query, hops=hops, k=k)
        return [_serialize_memory(m) for m in memories]

    multi_hop_tool = StructuredTool.from_function(
        func=multi_hop,
        coroutine=amulti_hop,
        name="epochdb_multi_hop",
        description=(
            "Retrieve memories related to a query using multi-hop relational search across the global entity graph. "
            "Use this for complex questions that require connecting multiple facts."
        ),
        args_schema=MultiHopInput
    )

    # 4. epochdb_get_timeline
    def get_timeline(
        entity_id: Optional[str] = None,
        start: Optional[Union[float, str]] = None,
        end: Optional[Union[float, str]] = None
    ) -> List[Dict[str, Any]]:
        start_val = _parse_time(start)
        end_val = _parse_time(end)
        if is_async:
            memories = db._get_db_sync().get_timeline(entity_id=entity_id, start=start_val, end=end_val)
        else:
            memories = db.get_timeline(entity_id=entity_id, start=start_val, end=end_val)
        return [_serialize_memory(m) for m in memories]

    async def aget_timeline(
        entity_id: Optional[str] = None,
        start: Optional[Union[float, str]] = None,
        end: Optional[Union[float, str]] = None
    ) -> List[Dict[str, Any]]:
        start_val = _parse_time(start)
        end_val = _parse_time(end)
        if is_async:
            memories = await db.get_timeline(entity_id=entity_id, start=start_val, end=end_val)
        else:
            memories = await asyncio.to_thread(
                db.get_timeline, entity_id=entity_id, start=start_val, end=end_val
            )
        return [_serialize_memory(m) for m in memories]

    timeline_tool = StructuredTool.from_function(
        func=get_timeline,
        coroutine=aget_timeline,
        name="epochdb_get_timeline",
        description="Retrieve the chronological history and timeline of memories associated with a specific entity ID (or all memories if entity_id is omitted).",
        args_schema=TimelineInput
    )

    # 5. epochdb_entity_graph
    def entity_graph(entity_id: str, depth: int = 2) -> Dict[str, Any]:
        if is_async:
            graph_obj = db._get_db_sync().entity_graph(entity_id, depth=depth)
        else:
            graph_obj = db.entity_graph(entity_id, depth=depth)
        return {
            "nodes": getattr(graph_obj, "nodes", []),
            "edges": getattr(graph_obj, "edges", [])
        }

    async def aentity_graph(entity_id: str, depth: int = 2) -> Dict[str, Any]:
        if is_async:
            graph_obj = await db.entity_graph(entity_id, depth=depth)
        else:
            graph_obj = await asyncio.to_thread(db.entity_graph, entity_id, depth=depth)
        return {
            "nodes": getattr(graph_obj, "nodes", []),
            "edges": getattr(graph_obj, "edges", [])
        }

    entity_graph_tool = StructuredTool.from_function(
        func=entity_graph,
        coroutine=aentity_graph,
        name="epochdb_entity_graph",
        description="Construct and retrieve the local entity graph representation (nodes and edges) around a specific entity ID.",
        args_schema=GraphInput
    )

    # 6. epochdb_update
    def update(memory_id: str, text: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        if is_async:
            db._get_db_sync().update(memory_id, text=text, metadata=metadata)
        else:
            db.update(memory_id, text=text, metadata=metadata)
        return f"Memory {memory_id} updated successfully."

    async def aupdate(
        memory_id: str, text: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if is_async:
            await db.update(memory_id, text=text, metadata=metadata)
        else:
            await asyncio.to_thread(db.update, memory_id, text=text, metadata=metadata)
        return f"Memory {memory_id} updated successfully."

    update_tool = StructuredTool.from_function(
        func=update,
        coroutine=aupdate,
        name="epochdb_update",
        description="Update the text or metadata of an existing memory in EpochDB by its ID.",
        args_schema=UpdateInput
    )

    # 7. epochdb_delete
    def delete(memory_id: str, hard: bool = False) -> str:
        if is_async:
            db._get_db_sync().delete(memory_id, hard=hard)
        else:
            db.delete(memory_id, hard=hard)
        return f"Memory {memory_id} deleted successfully."

    async def adelete(memory_id: str, hard: bool = False) -> str:
        if is_async:
            await db.delete(memory_id, hard=hard)
        else:
            await asyncio.to_thread(db.delete, memory_id, hard=hard)
        return f"Memory {memory_id} deleted successfully."

    delete_tool = StructuredTool.from_function(
        func=delete,
        coroutine=adelete,
        name="epochdb_delete",
        description="Delete a memory from EpochDB by its ID.",
        args_schema=DeleteInput
    )

    return [
        remember_tool,
        query_tool,
        multi_hop_tool,
        timeline_tool,
        entity_graph_tool,
        update_tool,
        delete_tool
    ]
