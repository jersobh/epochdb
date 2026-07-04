# epochdb/retrieval/router.py
import os
import json
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class QueryRouting(BaseModel):
    query_type: str = Field(description="One of: 'semantic', 'relational', 'temporal', 'quantitative', 'composite'")
    sub_queries: List[str] = Field(description="For composite queries, the sub-queries to execute. For others, the list containing the single original query.")
    entity_id: Optional[str] = Field(None, description="The specific entity ID if the query is relational or temporal (e.g. 'Jeff', 'temperature')")
    quantitative_field: Optional[str] = Field(None, description="The quantitative field to query (e.g. 'power_usage', 'price')")
    quantitative_op: Optional[str] = Field(None, description="The comparison operator (e.g. '>', '<', '==')")
    quantitative_value: Optional[float] = Field(None, description="The numeric value to compare against")
    reasoning: str = Field(description="Brief explanation of the routing decisions")

class AdaptiveRouter:
    def __init__(self, db: "EpochDB", model_id: Optional[str] = None):
        self.db = db
        self.model_id = model_id

    def route_and_query(self, query: str, k: int = 5, context_window: int = 0) -> List["Memory"]:
        """
        Intelligently routes a query to the optimal engine(s), falls back to local rules if offline.
        """
        routing = self._get_routing(query)
        logger.info(f"Adaptive Query Routing: {routing.query_type} (Reasoning: {routing.reasoning})")

        if routing.query_type == "composite" and len(routing.sub_queries) > 1:
            # Decompose and query in parallel
            merged_memories = []
            seen_ids = set()
            for sub_q in routing.sub_queries:
                sub_results = self.route_and_query(sub_q, k=k, context_window=context_window)
                for mem in sub_results:
                    if mem.id not in seen_ids:
                        merged_memories.append(mem)
                        seen_ids.add(mem.id)
            return merged_memories[:k]

        elif routing.query_type == "quantitative" and routing.quantitative_field:
            field = routing.quantitative_field
            op = routing.quantitative_op or "=="
            val = routing.quantitative_value if routing.quantitative_value is not None else 0.0
            
            # Translate to query_range
            min_val = -float("inf")
            max_val = float("inf")
            if op == ">":
                min_val = val
            elif op == "<":
                max_val = val
            elif op == "==":
                min_val = val
                max_val = val
            elif op == ">=":
                min_val = val
            elif op == "<=":
                max_val = val

            atoms = self.db.retriever.query_range(field=field, min_val=min_val, max_val=max_val)
            from epochdb.api.db import Memory
            return [Memory(a) for a in atoms][:k]

        elif routing.query_type == "temporal":
            # Chronological lookup
            results = self.db.get_timeline(entity_id=routing.entity_id)
            return results[:k]

        elif routing.query_type == "relational":
            # Multi-hop retrieval
            return self.db.multi_hop(query, hops=2, k=k, context_window=context_window)

        else:
            # Default to standard semantic query
            return self.db.query(query, k=k, context_window=context_window)

    def _get_routing(self, query: str) -> QueryRouting:
        # 1. Try Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key, http_options={'api_version': 'v1beta'})
                model = self.model_id or "gemini-2.5-flash"
                
                prompt = (
                    "Analyze the query and determine the most appropriate retrieval strategy.\n"
                    f"Query: \"{query}\"\n"
                )
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': QueryRouting,
                    }
                )
                data = json.loads(response.text)
                return QueryRouting(**data)
            except Exception as e:
                logger.warning(f"Gemini routing failed, trying next provider: {e}")

        # 2. Try OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                model = self.model_id or "gpt-4o-mini"
                
                response = client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Analyze the query and determine the retrieval strategy."},
                        {"role": "user", "content": query}
                    ],
                    response_format=QueryRouting
                )
                return response.choices[0].message.parsed
            except Exception as e:
                logger.warning(f"OpenAI routing failed, trying next provider: {e}")

        # 3. Try Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key)
                model = self.model_id or "claude-3-5-haiku-20241022"
                
                # Use tool use to enforce structure
                response = client.messages.create(
                    model=model,
                    max_tokens=1000,
                    system="You are a routing agent. You must invoke the route_query tool to output structured query routing details.",
                    messages=[{"role": "user", "content": query}],
                    tools=[
                        {
                            "name": "route_query",
                            "description": "Route the user query to the database",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "query_type": {"type": "string", "enum": ["semantic", "relational", "temporal", "quantitative", "composite"]},
                                    "sub_queries": {"type": "array", "items": {"type": "string"}},
                                    "entity_id": {"type": "string"},
                                    "quantitative_field": {"type": "string"},
                                    "quantitative_op": {"type": "string"},
                                    "quantitative_value": {"type": "number"},
                                    "reasoning": {"type": "string"}
                                },
                                "required": ["query_type", "sub_queries", "reasoning"]
                            }
                        }
                    ],
                    tool_choice={"type": "tool", "name": "route_query"}
                )
                # Parse tool inputs
                for content in response.content:
                    if content.type == "tool_use" and content.name == "route_query":
                        return QueryRouting(**content.input)
            except Exception as e:
                logger.warning(f"Anthropic routing failed, falling back to local: {e}")

        # 4. Strict Local Fallback Heuristics
        return self._local_fallback_routing(query)

    def _local_fallback_routing(self, query: str) -> QueryRouting:
        q_lower = query.lower()
        
        # Check quantitative (e.g. "temperature > 100", "price < 20")
        match = re.search(r"(\w+)\s*(>=|<=|>|<|==)\s*([+-]?\d+(?:\.\d+)?)", q_lower)
        if match:
            field = match.group(1)
            op = match.group(2)
            val = float(match.group(3))
            return QueryRouting(
                query_type="quantitative",
                sub_queries=[query],
                quantitative_field=field,
                quantitative_op=op,
                quantitative_value=val,
                reasoning="Local fallback: parsed comparison operator and field."
            )

        # Check temporal keywords
        temporal_keywords = ["timeline", "history", "chronological", "when", "yesterday", "today", "ago", "chronology", "sequence"]
        if any(tk in q_lower for tk in temporal_keywords):
            # Try to extract the first capitalized word as entity name, or use query keywords
            entity = None
            words = [w.strip(".,!?;:()\"'") for w in query.split() if w.strip()]
            for w in words:
                if w and w[0].isupper() and w.lower() not in ["what", "when", "who", "where", "how", "why", "the", "yesterday"]:
                    entity = w
                    break
            return QueryRouting(
                query_type="temporal",
                sub_queries=[query],
                entity_id=entity,
                reasoning="Local fallback: detected temporal keywords in query."
            )

        # Check relational keywords
        relational_keywords = ["manager", "related", "connected", "leads", "uses", "friend", "parent", "child", "spouse", "location", "works at"]
        if any(rk in q_lower for rk in relational_keywords):
            return QueryRouting(
                query_type="relational",
                sub_queries=[query],
                reasoning="Local fallback: detected relational keyword in query."
            )

        # Default to semantic
        return QueryRouting(
            query_type="semantic",
            sub_queries=[query],
            reasoning="Local fallback: query defaulted to standard semantic search."
        )
