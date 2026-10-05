import asyncio
import inspect
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
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
    memory_type: Optional[str] = Field(
        default=None,
        description="Optional memory type: 'general', 'episodic', 'profile', 'working', or 'skill'."
    )


class QueryInput(BaseModel):
    query: str = Field(description="The semantic search query.")
    k: int = Field(default=5, description="Number of relevant memories to retrieve.")
    min_score: float = Field(default=0.0, description="Minimum similarity score threshold (0.0 to 1.0).")
    memory_type: Optional[str] = Field(
        default=None,
        description="Optional filter by memory type: 'general', 'episodic', 'profile', 'working', or 'skill'."
    )


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

class AnalyzeInput(BaseModel):
    text: str = Field(description="The text content to analyze for relationship triples.")
class DeleteInput(BaseModel):
    memory_id: str = Field(description="The unique ID of the memory to delete.")
    hard: bool = Field(default=False, description="If True, permanently delete the memory. If False, soft-deletes.")


class RememberSkillInput(BaseModel):
    skill_name: str = Field(description="Canonical name for the procedural skill.")
    description: str = Field(description="What the skill does.")
    steps: List[Dict[str, Any]] = Field(
        description="Ordered steps, each with optional step_num, action, and details."
    )
    tool_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional JSON-schema style tool definition for this skill.",
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional extra metadata.")
    skill_id: Optional[str] = Field(default=None, description="Optional stable atom id for upserts.")
    decision_rules: Optional[List[str]] = Field(
        default=None,
        description="Optional guardrails the agent should follow when executing the skill.",
    )


class GetSkillInput(BaseModel):
    skill_name: str = Field(description="Skill name, title, skill_id, or atom id.")


class RememberProfileInput(BaseModel):
    user_id: str = Field(description="User identifier for the profile fact.")
    fact_text: str = Field(description="Long-term user preference or identity fact.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional extra metadata.")


class GetProfileInput(BaseModel):
    user_id: str = Field(description="User identifier whose profile facts should be retrieved.")


class HotSummaryInput(BaseModel):
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user id to include profile facts in the snapshot.",
    )


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
        "memory_type": getattr(m, "memory_type", "general"),
        "namespace": getattr(m, "namespace", None),
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
        A list of LangChain tools configured for sync and async execution.
    """
    # Detect if we are using AsyncEpochDB or synchronous EpochDB
    is_async = hasattr(db, "_get_db_sync") or (
        hasattr(db, "remember") and inspect.iscoroutinefunction(db.remember)
    )

    # 1. epochdb_remember
    def remember(text: str, metadata: Optional[Dict[str, Any]] = None, memory_type: Optional[str] = None) -> str:
        if is_async:
            return db._get_db_sync().remember(text, metadata, memory_type=memory_type)
        return db.remember(text, metadata, memory_type=memory_type)

    async def aremember(text: str, metadata: Optional[Dict[str, Any]] = None, memory_type: Optional[str] = None) -> str:
        if is_async:
            return await db.remember(text, metadata, memory_type=memory_type)
        return await asyncio.to_thread(db.remember, text, metadata, memory_type=memory_type)

    remember_tool = StructuredTool.from_function(
        func=remember,
        coroutine=aremember,
        name="epochdb_remember",
        description="Store a new memory, knowledge atom, or fact in EpochDB for long-term retrieval.",
        args_schema=RememberInput
    )

    # 2. epochdb_query
    def query(query: str, k: int = 5, min_score: float = 0.0, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if is_async:
            memories = db._get_db_sync().query(query, k=k, min_score=min_score, memory_type=memory_type)
        else:
            memories = db.query(query, k=k, min_score=min_score, memory_type=memory_type)
        return [_serialize_memory(m) for m in memories]

    async def aquery(query: str, k: int = 5, min_score: float = 0.0, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if is_async:
            memories = await db.query(query, k=k, min_score=min_score, memory_type=memory_type)
        else:
            memories = await asyncio.to_thread(db.query, query, k=k, min_score=min_score, memory_type=memory_type)
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

    # 8. epochdb_analyze
    def analyze(text: str) -> List[Tuple[str, str, str]]:
        if is_async:
            return db._get_db_sync().analyze(text)
        return db.analyze(text)

    async def aanalyze(text: str) -> List[Tuple[str, str, str]]:
        if is_async:
            return await db.analyze(text)
        return await asyncio.to_thread(db.analyze, text)

    analyze_tool = StructuredTool.from_function(
        func=analyze,
        coroutine=aanalyze,
        name="epochdb_analyze",
        description="Extract entity relationship triples (subject, predicate, object) from a given text.",
        args_schema=AnalyzeInput
    )

    # 9. epochdb_remember_skill
    def remember_skill(
        skill_name: str,
        description: str,
        steps: List[Dict[str, Any]],
        tool_schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skill_id: Optional[str] = None,
        decision_rules: Optional[List[str]] = None,
    ) -> str:
        if is_async:
            return db._get_db_sync().remember_skill(
                skill_name, description, steps, tool_schema, metadata, skill_id, decision_rules
            )
        return db.remember_skill(
            skill_name, description, steps, tool_schema, metadata, skill_id, decision_rules
        )

    async def aremember_skill(
        skill_name: str,
        description: str,
        steps: List[Dict[str, Any]],
        tool_schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skill_id: Optional[str] = None,
        decision_rules: Optional[List[str]] = None,
    ) -> str:
        if is_async:
            return await db.remember_skill(
                skill_name, description, steps, tool_schema, metadata, skill_id, decision_rules
            )
        return await asyncio.to_thread(
            db.remember_skill,
            skill_name,
            description,
            steps,
            tool_schema,
            metadata,
            skill_id,
            decision_rules,
        )

    remember_skill_tool = StructuredTool.from_function(
        func=remember_skill,
        coroutine=aremember_skill,
        name="epochdb_remember_skill",
        description="Store a procedural skill (SOP / tool playbook) as a typed MemoryType.SKILL atom.",
        args_schema=RememberSkillInput,
    )

    # 10. epochdb_get_skill
    def get_skill(skill_name: str) -> Optional[Dict[str, Any]]:
        if is_async:
            mem = db._get_db_sync().get_skill(skill_name)
        else:
            mem = db.get_skill(skill_name)
        return _serialize_memory(mem) if mem else None

    async def aget_skill(skill_name: str) -> Optional[Dict[str, Any]]:
        if is_async:
            mem = await db.get_skill(skill_name)
        else:
            mem = await asyncio.to_thread(db.get_skill, skill_name)
        return _serialize_memory(mem) if mem else None

    get_skill_tool = StructuredTool.from_function(
        func=get_skill,
        coroutine=aget_skill,
        name="epochdb_get_skill",
        description="Retrieve a procedural skill by name, title, skill_id, or atom id.",
        args_schema=GetSkillInput,
    )

    # 11. epochdb_list_skills
    def list_skills() -> List[Dict[str, Any]]:
        if is_async:
            skills = db._get_db_sync().list_skills()
        else:
            skills = db.list_skills()
        return [_serialize_memory(s) for s in skills]

    async def alist_skills() -> List[Dict[str, Any]]:
        if is_async:
            skills = await db.list_skills()
        else:
            skills = await asyncio.to_thread(db.list_skills)
        return [_serialize_memory(s) for s in skills]

    list_skills_tool = StructuredTool.from_function(
        func=list_skills,
        coroutine=alist_skills,
        name="epochdb_list_skills",
        description="List all stored procedural skills (MemoryType.SKILL).",
    )

    # 12. epochdb_remember_user_profile
    def remember_user_profile(
        user_id: str,
        fact_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if is_async:
            return db._get_db_sync().remember_user_profile(user_id, fact_text, metadata)
        return db.remember_user_profile(user_id, fact_text, metadata)

    async def aremember_user_profile(
        user_id: str,
        fact_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if is_async:
            return await db.remember_user_profile(user_id, fact_text, metadata)
        return await asyncio.to_thread(db.remember_user_profile, user_id, fact_text, metadata)

    remember_profile_tool = StructuredTool.from_function(
        func=remember_user_profile,
        coroutine=aremember_user_profile,
        name="epochdb_remember_user_profile",
        description="Store a long-term user profile fact (MemoryType.PROFILE).",
        args_schema=RememberProfileInput,
    )

    # 13. epochdb_get_user_profile
    def get_user_profile(user_id: str) -> List[Dict[str, Any]]:
        if is_async:
            profiles = db._get_db_sync().get_user_profile(user_id)
        else:
            profiles = db.get_user_profile(user_id)
        return [_serialize_memory(p) for p in profiles]

    async def aget_user_profile(user_id: str) -> List[Dict[str, Any]]:
        if is_async:
            profiles = await db.get_user_profile(user_id)
        else:
            profiles = await asyncio.to_thread(db.get_user_profile, user_id)
        return [_serialize_memory(p) for p in profiles]

    get_profile_tool = StructuredTool.from_function(
        func=get_user_profile,
        coroutine=aget_user_profile,
        name="epochdb_get_user_profile",
        description="Retrieve stored profile facts for a user.",
        args_schema=GetProfileInput,
    )

    # 14. epochdb_get_hot_summary_snapshot
    def get_hot_summary_snapshot(user_id: Optional[str] = None) -> str:
        if is_async:
            return db._get_db_sync().get_hot_summary_snapshot(user_id)
        return db.get_hot_summary_snapshot(user_id)

    async def aget_hot_summary_snapshot(user_id: Optional[str] = None) -> str:
        if is_async:
            return await db.get_hot_summary_snapshot(user_id)
        return await asyncio.to_thread(db.get_hot_summary_snapshot, user_id)

    hot_summary_tool = StructuredTool.from_function(
        func=get_hot_summary_snapshot,
        coroutine=aget_hot_summary_snapshot,
        name="epochdb_get_hot_summary_snapshot",
        description="Build a compact profile + skills block for system-prompt injection.",
        args_schema=HotSummaryInput,
    )

    return [
        remember_tool,
        query_tool,
        multi_hop_tool,
        timeline_tool,
        entity_graph_tool,
        update_tool,
        delete_tool,
        analyze_tool,
        remember_skill_tool,
        get_skill_tool,
        list_skills_tool,
        remember_profile_tool,
        get_profile_tool,
        hot_summary_tool,
    ]
