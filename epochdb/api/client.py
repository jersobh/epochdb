import json
import urllib.request
import urllib.error
from typing import List, Optional, Any, Dict
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

    def remember(self, text: str, metadata: Optional[dict] = None) -> str:
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
