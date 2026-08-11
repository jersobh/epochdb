"""
Fact / triple extraction backends for EpochDB.

Model IDs follow the same prefix style as embedding models:

- ``local``                         — fast heuristic co-occurrence (no ML)
- ``google:gemini-2.5-flash``        — Gemini structured extraction
- ``openai:gpt-4o-mini``             — OpenAI JSON extraction
- ``hf:Babelscape/rebel-large``      — Hugging Face seq2seq (REBEL / T5 / …)
- ``hf``                            — default lightweight HF model
- bare ``org/name`` paths           — treated as Hugging Face models

Environment overrides: ``EPOCHDB_EXTRACTION_MODEL``, ``GEMINI_MODEL``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Small open IE model (~1.5GB); override with hf:<other> for lighter T5s etc.
DEFAULT_HF_EXTRACTION_MODEL = "Babelscape/rebel-large"
# Ultralight prompt-based fallback when users ask for "small" / "mini"
DEFAULT_HF_LIGHT_MODEL = "google/flan-t5-small"


class RelationshipTriple(BaseModel):
    subject: str = Field(description="The subject entity of the relation (noun or concept)")
    predicate: str = Field(
        description="The relationship or action connecting subject and object (verb or prepositional phrase)"
    )
    object: str = Field(description="The object entity or property (noun, value, or concept)")


class ExtractedFacts(BaseModel):
    triples: List[RelationshipTriple] = Field(
        description="List of relationship triples extracted from the text"
    )


def _normalize_triples(raw: Any) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    seen = set()
    for t in raw or []:
        if isinstance(t, RelationshipTriple):
            s, p, o = t.subject, t.predicate, t.object
        elif isinstance(t, dict):
            s = t.get("subject") or t.get("head") or t.get("subj")
            p = t.get("predicate") or t.get("type") or t.get("relation") or t.get("pred")
            o = t.get("object") or t.get("tail") or t.get("obj")
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            s, p, o = t[0], t[1], t[2]
        else:
            continue
        s, p, o = str(s or "").strip(), str(p or "").strip(), str(o or "").strip()
        if not s or not p or not o:
            continue
        key = (s, p, o)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def resolve_extraction_backend(model_id: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Returns ``(backend, resolved_model_id)`` where backend is one of
    ``local | google | openai | hf``.
    """
    raw = (model_id or "").strip()
    if not raw:
        # Prefer configured cloud keys when no explicit model was set.
        if os.getenv("GEMINI_API_KEY"):
            return "google", os.getenv("GEMINI_MODEL") or os.getenv("EPOCHDB_EXTRACTION_MODEL") or "gemini-2.5-flash"
        if os.getenv("OPENAI_API_KEY"):
            return "openai", os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        return "local", None

    lower = raw.lower()
    if lower in ("local", "heuristic", "none", "off"):
        return "local", None

    if lower in ("hf", "huggingface", "transformers"):
        return "hf", DEFAULT_HF_EXTRACTION_MODEL
    if lower in ("hf:small", "hf:light", "hf:mini", "huggingface:small"):
        return "hf", DEFAULT_HF_LIGHT_MODEL

    if lower.startswith("google:"):
        return "google", raw.split(":", 1)[1].strip() or "gemini-2.5-flash"
    if lower.startswith("openai:"):
        return "openai", raw.split(":", 1)[1].strip() or "gpt-4o-mini"
    if lower.startswith("hf:") or lower.startswith("huggingface:") or lower.startswith("transformers:"):
        name = raw.split(":", 1)[1].strip()
        if not name or name.lower() in ("default", "rebel"):
            return "hf", DEFAULT_HF_EXTRACTION_MODEL
        if name.lower() in ("small", "light", "mini"):
            return "hf", DEFAULT_HF_LIGHT_MODEL
        return "hf", name

    # Bare Gemini-style ids without prefix
    if lower.startswith("gemini"):
        return "google", raw

    # org/name → Hugging Face
    if "/" in raw:
        return "hf", raw

    # Unknown bare token → treat as local alias if "local", else HF hub id
    return "hf", raw


class GoogleFactExtractor:
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = (
            model_id
            or os.getenv("GEMINI_MODEL")
            or os.getenv("EPOCHDB_EXTRACTION_MODEL")
            or "gemini-2.5-flash"
        )
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
            self._client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
        return self._client

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        try:
            client = self._get_client()
            prompt = (
                "Extract all significant semantic entity relationships and facts from the following text "
                "as a list of (subject, predicate, object) triples. "
                "Focus on key properties, user preferences, events, and relations.\n\n"
                f'Text: "{text}"'
            )
            response = client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ExtractedFacts,
                },
            )
            data = json.loads(response.text)
            return _normalize_triples(data.get("triples", []))
        except Exception as e:
            logger.error(f"GoogleFactExtractor failed: {e}")
            raise


class OpenAIFactExtractor:
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai is required to use OpenAIFactExtractor. "
                    "Install it with: pip install openai"
                )
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        try:
            client = self._get_client()
            prompt = (
                "Extract significant (subject, predicate, object) triples from the text. "
                'Return JSON: {"triples": [{"subject": "...", "predicate": "...", "object": "..."}]}.\n\n'
                f"Text: {text}"
            )
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(response.choices[0].message.content or "{}")
            return _normalize_triples(data.get("triples", []))
        except Exception as e:
            logger.error(f"OpenAIFactExtractor failed: {e}")
            raise


class LocalFactExtractor:
    """Heuristic co-occurrence extractor — always available, no ML deps."""

    def __init__(self, engine=None):
        self.engine = engine

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        raw_entities = []
        if self.engine is not None:
            try:
                raw_entities = [str(e) for e in self.engine.extract_entities(text) if str(e)]
            except Exception:
                pass

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
            if e in predicates or e.lower().replace(" ", "_") in {p.lower() for p in predicates}:
                continue
            if e not in seen:
                seen.add(e)
                entities.append(e)

        res: List[Tuple[str, str, str]] = []
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                res.append((entities[i], "co_occurs_with", entities[i + 1]))
            if len(entities) > 2:
                res.append((entities[0], "co_occurs_with", entities[-1]))
        elif len(entities) == 1:
            res.append((entities[0], "mentions", entities[0]))

        return res


def _parse_rebel_output(decoded: str) -> List[Tuple[str, str, str]]:
    """Parse REBEL / mREBEL special-token generations into triples."""
    triplets: List[Tuple[str, str, str]] = []
    relation = subject = object_ = ""
    current = "x"
    cleaned = (
        decoded.replace("<s>", "")
        .replace("<pad>", "")
        .replace("</s>", "")
        .replace("<unk>", "")
        .strip()
    )
    for token in cleaned.split():
        if token in ("<triplet>", "<triple>"):
            current = "t"
            if subject and relation and object_:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if subject and relation and object_:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject = f"{subject} {token}".strip()
            elif current == "s":
                object_ = f"{object_} {token}".strip()
            elif current == "o":
                relation = f"{relation} {token}".strip()
    if subject and relation and object_:
        triplets.append((subject.strip(), relation.strip(), object_.strip()))
    return _normalize_triples(triplets)


def _parse_jsonish_triples(text: str) -> List[Tuple[str, str, str]]:
    """Best-effort parse of JSON or line-oriented triples from generative models."""
    text = (text or "").strip()
    if not text:
        return []

    # Fenced JSON
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else text

    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return _normalize_triples(data.get("triples") or data.get("relations") or [])
        if isinstance(data, list):
            return _normalize_triples(data)
    except json.JSONDecodeError:
        pass

    # Find embedded JSON object/array
    for match in re.finditer(r"(\{.*\}|\[.*\])", text, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                got = _normalize_triples(data.get("triples") or data.get("relations") or [])
            else:
                got = _normalize_triples(data)
            if got:
                return got
        except json.JSONDecodeError:
            continue

    # Lines like: (Alice, works_at, Acme) or Alice | works_at | Acme
    out = []
    for line in text.splitlines():
        line = line.strip().strip("-•*").strip()
        m = re.match(r"^\(?\s*([^,|]+)\s*[,|]\s*([^,|]+)\s*[,|]\s*([^)|]+)\s*\)?$", line)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    return _normalize_triples(out)


class HuggingFaceFactExtractor:
    """
    Local Hugging Face seq2seq extractor (REBEL-family or prompt-based T5/BART).

    Lazy-loads ``transformers`` + torch weights on first ``extract`` call,
    with optional on-disk cache under ``~/.cache/epochdb_models`` (same pattern
    as SentenceTransformer embeddings).
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_cache_path: Optional[str] = None,
        max_length: int = 256,
        max_input_length: int = 256,
    ):
        self.model_id = model_id or DEFAULT_HF_EXTRACTION_MODEL
        self.model_cache_path = model_cache_path
        self.max_length = max_length
        self.max_input_length = max_input_length
        self._tokenizer = None
        self._model = None
        self._is_rebel = "rebel" in self.model_id.lower()

    def _cache_dir(self) -> Optional[str]:
        safe = self.model_id.replace("/", "_")
        if self.model_cache_path:
            return os.path.join(self.model_cache_path, safe)
        return os.path.join(os.path.expanduser("~"), ".cache", "epochdb_models", safe)

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for Hugging Face triple extraction. "
                "Install it with: pip install epochdb[extraction]"
            )

        cache_dir = self._cache_dir()
        load_id = cache_dir if cache_dir and os.path.isdir(cache_dir) and os.listdir(cache_dir) else self.model_id
        logger.info(f"Loading HuggingFace extraction model: {load_id}")
        self._tokenizer = AutoTokenizer.from_pretrained(load_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(load_id)
        self._model.eval()

        # Persist a local copy for faster subsequent startups (best-effort)
        if cache_dir and load_id == self.model_id:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                self._tokenizer.save_pretrained(cache_dir)
                self._model.save_pretrained(cache_dir)
            except Exception as e:
                logger.warning(f"Could not cache HF extraction model to {cache_dir}: {e}")

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for Hugging Face triple extraction. "
                "Install it with: pip install epochdb[extraction]"
            )

        self._load()
        assert self._tokenizer is not None and self._model is not None

        if self._is_rebel:
            inputs = self._tokenizer(
                text,
                max_length=self.max_input_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            prompt = (
                "Extract relationship triples from the text as JSON with key "
                '"triples" as a list of {subject, predicate, object} objects.\n'
                f"Text: {text}\nJSON:"
            )
            inputs = self._tokenizer(
                prompt,
                max_length=self.max_input_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=3,
                num_return_sequences=1,
            )
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=not self._is_rebel)[0]

        if self._is_rebel:
            # Keep special tokens for REBEL parse
            decoded_special = self._tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
            return _parse_rebel_output(decoded_special)

        triples = _parse_jsonish_triples(decoded)
        if triples:
            return triples
        # Last resort: local-style nothing — return empty rather than junk
        logger.warning(f"HF extractor produced no parseable triples for model={self.model_id}")
        return []


class FactExtractor:
    """Facade that routes to the configured extraction backend with local fallback."""

    def __init__(
        self,
        engine=None,
        model_id: Optional[str] = None,
        model_cache_path: Optional[str] = None,
    ):
        self.engine = engine
        self.model_id = model_id
        self.model_cache_path = model_cache_path or getattr(engine, "_model_cache_path", None)
        self._backend_name, self._resolved_id = resolve_extraction_backend(model_id)
        self._impl = None
        self._local_extractor = LocalFactExtractor(engine)

    @property
    def backend(self) -> str:
        return self._backend_name

    @property
    def resolved_model_id(self) -> Optional[str]:
        return self._resolved_id

    def _get_impl(self):
        if self._impl is not None:
            return self._impl
        if self._backend_name == "google":
            self._impl = GoogleFactExtractor(self._resolved_id)
        elif self._backend_name == "openai":
            self._impl = OpenAIFactExtractor(self._resolved_id)
        elif self._backend_name == "hf":
            self._impl = HuggingFaceFactExtractor(
                self._resolved_id,
                model_cache_path=self.model_cache_path,
            )
        else:
            self._impl = self._local_extractor
        return self._impl

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        if self._backend_name == "local":
            return self._local_extractor.extract(text)
        try:
            return self._get_impl().extract(text)
        except Exception as e:
            logger.warning(f"Falling back to LocalFactExtractor due to error ({self._backend_name}): {e}")
            return self._local_extractor.extract(text)

    def extract_local(self, text: str) -> List[Tuple[str, str, str]]:
        """Always-on heuristic seed triples (safe for sync ingest path)."""
        return self._local_extractor.extract(text)
