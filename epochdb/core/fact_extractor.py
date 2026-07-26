# epochdb/core/fact_extractor.py
import os
import json
import logging
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RelationshipTriple(BaseModel):
    subject: str = Field(description="The subject entity of the relation (noun or concept)")
    predicate: str = Field(description="The relationship or action connecting subject and object (verb or prepositional phrase)")
    object: str = Field(description="The object entity or property (noun, value, or concept)")

class ExtractedFacts(BaseModel):
    triples: List[RelationshipTriple] = Field(description="List of relationship triples extracted from the text")

class GoogleFactExtractor:
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or "gemini-2.5-flash"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai is required to use GoogleFactExtractor. "
                    "Install it with: pip install epochdb[google]"
                )
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment.")
            self._client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
        return self._client

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        try:
            client = self._get_client()
            prompt = (
                "Extract all significant semantic entity relationships and facts from the following text "
                "as a list of (subject, predicate, object) triples. "
                "Focus on key properties, user preferences, events, and relations.\n\n"
                f"Text: \"{text}\""
            )
            response = client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': ExtractedFacts,
                }
            )
            data = json.loads(response.text)
            res = []
            for t in data.get("triples", []):
                res.append((t["subject"], t["predicate"], t["object"]))
            return res
        except Exception as e:
            logger.error(f"GoogleFactExtractor failed: {e}")
            raise e

class LocalFactExtractor:
    def __init__(self, engine=None):
        self.engine = engine

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        raw_entities = []
        if self.engine is not None:
            try:
                raw_entities = [str(e) for e in self.engine.extract_entities(text) if str(e)]
            except Exception:
                pass
        
        # Fallback to simple heuristic if engine returns no entities (e.g. brand new DB)
        if not raw_entities:
            words = [w.strip(".,!?;:()\"'") for w in text.split() if w.strip()]
            nouns = [w for w in words if w and w[0].isupper()]
            if not nouns:
                nouns = [w for w in words if len(w) > 3][:3]
            raw_entities = [str(n) for n in nouns if n]

        seen = set()
        entities = []
        predicates = set()
        if self.engine is not None:
            predicates = {str(p) for p in getattr(self.engine, "predicates", set()) or set()}
        for e in raw_entities:
            # Predicates matched by extract_entities Pass-2 must not become graph nodes.
            if e in predicates or e.lower().replace(" ", "_") in {p.lower() for p in predicates}:
                continue
            if e not in seen:
                seen.add(e)
                entities.append(e)

        res = []
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                res.append((entities[i], "co_occurs_with", entities[i + 1]))
            if len(entities) > 2:
                res.append((entities[0], "co_occurs_with", entities[-1]))
        elif len(entities) == 1:
            res.append((entities[0], "mentions", entities[0]))

        return res

class FactExtractor:
    def __init__(self, engine=None, model_id: Optional[str] = None):
        self.engine = engine
        self.model_id = model_id
        self._google_extractor = None
        self._local_extractor = LocalFactExtractor(engine)

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        # Use Google LLM if GEMINI_API_KEY is available
        if os.getenv("GEMINI_API_KEY"):
            try:
                if self._google_extractor is None:
                    self._google_extractor = GoogleFactExtractor(self.model_id)
                return self._google_extractor.extract(text)
            except Exception as e:
                logger.warning(f"Falling back to LocalFactExtractor due to error: {e}")
                
        return self._local_extractor.extract(text)
