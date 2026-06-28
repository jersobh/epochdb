import json
import urllib.request
import urllib.error
from typing import List, Optional, Any, Dict
import httpx
from epochdb.api.db import Memory
from epochdb.core.atom import UnifiedMemoryAtom


class RemoteEpochDB:
    """
    Remote HTTP client facade for the EpochDB Agentic Memory Engine.
    Communicates with a running ThreadingEpochDBServer instance.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.url = f"http://{host}:{port}"

    def _post(self, path: str, data: dict) -> dict:
        url = self.url + path
        req_data = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = response.read()
                return json.loads(res_data.decode('utf-8'))
        except urllib.error.HTTPError as e:
            res_data = e.read()
            try:
                err_json = json.loads(res_data.decode('utf-8'))
                if "error" in err_json:
                    raise RuntimeError(f"Server error: {err_json['error']}")
            except Exception:
                pass
            raise RuntimeError(f"HTTP Error {e.code}: {e.reason}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to EpochDB server at {self.url}: {e}")

    def remember(
        self,
        text: str,
        triples: Optional[Any] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        if isinstance(triples, dict) and metadata is None:
            metadata = triples
            triples = None
        if triples is not None:
            metadata = metadata or {}
            metadata["triples"] = triples
        res = self._post("/remember", {"text": text, "metadata": metadata})
        return res["atom_id"]

    def remember_batch(self, items: list) -> List[str]:
        res = self._post("/remember_batch", {"items": items})
        return res["atom_ids"]

    def get(self, memory_id: str) -> Optional[Memory]:
        res = self._post("/get", {"memory_id": memory_id})
        if res.get("id"):
            atom = UnifiedMemoryAtom.from_dict(res)
            return Memory(atom)
        return None

    def update(self, memory_id: str, text: Optional[str] = None, metadata: Optional[dict] = None) -> None:
        self._post("/update", {"memory_id": memory_id, "text": text, "metadata": metadata})

    def delete(self, memory_id: str, hard: bool = False) -> None:
        self._post("/delete", {"memory_id": memory_id, "hard": hard})

    def query(self, text: str, k: int = 5, filters: Optional[dict] = None, min_score: float = 0.0) -> List[Memory]:
        res = self._post("/query", {"text": text, "k": k, "filters": filters, "min_score": min_score})
        memories = []
        for m_dict in res.get("memories", []):
            atom = UnifiedMemoryAtom.from_dict(m_dict)
            memories.append(Memory(atom))
        return memories

    def multi_hop(self, text: str, hops: int = 2, k: int = 5, filters: Optional[dict] = None) -> List[Memory]:
        res = self._post("/multi_hop", {"text": text, "hops": hops, "k": k, "filters": filters})
        memories = []
        for m_dict in res.get("memories", []):
            atom = UnifiedMemoryAtom.from_dict(m_dict)
            memories.append(Memory(atom))
        return memories

    def get_timeline(self, entity_id: Optional[str] = None, start: Optional[Any] = None, end: Optional[Any] = None) -> List[Memory]:
        start_val = start.timestamp() if hasattr(start, "timestamp") else start
        end_val = end.timestamp() if hasattr(end, "timestamp") else end
        res = self._post("/get_timeline", {"entity_id": entity_id, "start": start_val, "end": end_val})
        memories = []
        for m_dict in res.get("memories", []):
            atom = UnifiedMemoryAtom.from_dict(m_dict)
            memories.append(Memory(atom))
        return memories

    def stats(self) -> dict:
        return self._post("/stats", {})

    def flush(self) -> None:
        self._post("/flush", {})

    def compact(self) -> None:
        self._post("/compact", {})

    def close(self) -> None:
        self._post("/close", {})


class AsyncRemoteEpochDB:
    """
    Asynchronous remote HTTP client facade for the EpochDB Agentic Memory Engine.
    Communicates with a running FastAPI coordinator server, shard server, or ThreadingEpochDBServer.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8080, api_key: Optional[str] = None):
        self.base_url = f"http://{host}:{port}"
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers["X-API-Key"] = api_key
            headers["X-Internal-Token"] = api_key
            
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=30.0)

    async def _post(self, path: str, data: dict) -> dict:
        try:
            response = await self.client.post(path, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            try:
                err_json = e.response.json()
                if "error" in err_json:
                    raise RuntimeError(f"Server error: {err_json['error']}")
                if "detail" in err_json:
                    raise RuntimeError(f"Server error: {err_json['detail']}")
            except Exception:
                pass
            raise RuntimeError(f"HTTP Error {e.response.status_code}: {e.response.reason_phrase}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to EpochDB server at {self.base_url}: {e}")

    async def remember(
        self,
        text: str,
        triples: Optional[Any] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        if isinstance(triples, dict) and metadata is None:
            metadata = triples
            triples = None
        if triples is not None:
            metadata = metadata or {}
            metadata["triples"] = triples
        res = await self._post("/remember", {"text": text, "metadata": metadata})
        return res.get("id") or res.get("atom_id")

    async def remember_batch(self, items: list) -> List[str]:
        res = await self._post("/remember_batch", {"items": items})
        return res.get("atom_ids", [])

    def _normalize_atom_dict(self, m: dict) -> dict:
        if "payload" not in m and "text" in m:
            m["payload"] = m["text"]
        if "payload_type" not in m:
            m["payload_type"] = "text"
        if "embedding" not in m:
            m["embedding"] = []
        if "triples" not in m:
            m["triples"] = m.get("metadata", {}).get("triples", []) if isinstance(m.get("metadata"), dict) else []
        if "created_at" not in m:
            m["created_at"] = 0.0
        if "access_count" not in m:
            m["access_count"] = 0
        if "epoch_id" not in m:
            m["epoch_id"] = "active"
        return m

    async def get(self, memory_id: str) -> Optional[Memory]:
        res = await self._post("/get", {"memory_id": memory_id})
        # Handle cases where memory structure is returned as 'memory' attribute or nested directly
        m_dict = res.get("memory") if "memory" in res else res
        if m_dict and m_dict.get("id"):
            m_dict = self._normalize_atom_dict(m_dict)
            atom = UnifiedMemoryAtom.from_dict(m_dict)
            return Memory(atom)
        return None

    async def update(self, memory_id: str, text: Optional[str] = None, metadata: Optional[dict] = None) -> None:
        await self._post("/update", {"memory_id": memory_id, "text": text, "metadata": metadata})

    async def delete(self, memory_id: str, hard: bool = False) -> None:
        await self._post("/delete", {"memory_id": memory_id, "hard": hard})

    async def query(self, text: str, k: int = 5, filters: Optional[dict] = None, min_score: float = 0.0) -> List[Memory]:
        # Formulate query payload compatible with both Coordinator ("query") and Standard Shard ("text")
        payload = {
            "query": text,
            "text": text,
            "k": k,
            "filters": filters,
            "min_score": min_score
        }
        res = await self._post("/query", payload)
        raw_results = res.get("results") or res.get("memories") or []
        
        memories = []
        for m in raw_results:
            m = self._normalize_atom_dict(m)
            atom = UnifiedMemoryAtom.from_dict(m)
            mem_obj = Memory(atom)
            if "score" in m:
                mem_obj.score = float(m["score"])
            memories.append(mem_obj)
        return memories

    async def get_timeline(self, entity_id: Optional[str] = None, start: Optional[Any] = None, end: Optional[Any] = None) -> List[Memory]:
        start_val = start.timestamp() if hasattr(start, "timestamp") else start
        end_val = end.timestamp() if hasattr(end, "timestamp") else end
        res = await self._post("/get_timeline", {"entity_id": entity_id, "start": start_val, "end": end_val})
        
        raw_results = res if isinstance(res, list) else (res.get("memories") or res.get("results") or [])
        memories = []
        for m in raw_results:
            m = self._normalize_atom_dict(m)
            atom = UnifiedMemoryAtom.from_dict(m)
            memories.append(Memory(atom))
        return memories


    async def entity_graph(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        try:
            response = await self.client.get(f"/entity_graph?entity_id={entity_id}&depth={depth}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to query entity graph: {e}")

    async def stats(self) -> dict:
        try:
            # Try GET first (compatible with Coordinator Mode)
            response = await self.client.get("/stats")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            # Fallback to POST (compatible with core ThreadingEpochDBServer)
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 405:
                return await self._post("/stats", {})
            raise

    async def flush(self) -> None:
        await self._post("/flush", {})

    async def compact(self) -> None:
        await self._post("/compact", {})

    async def close(self) -> None:
        await self.client.aclose()

