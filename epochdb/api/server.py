import json
import logging
import http.server
import socketserver
from urllib.parse import urlparse
from typing import Optional
from epochdb.api.db import EpochDB

logger = logging.getLogger(__name__)

class EpochDBRequestHandler(http.server.BaseHTTPRequestHandler):
    db: Optional[EpochDB] = None

    def log_message(self, format, *args):
        # Suppress standard logging to prevent cluttering stdout during concurrent requests
        pass

    def do_POST(self):
        parsed_path = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            req_data = json.loads(post_data) if post_data else {}
        except Exception:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid JSON")
            return

        if not self.db:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Database not initialized")
            return

        try:
            response_body = self.handle_route(parsed_path.path, req_data)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_body).encode('utf-8'))
        except Exception as e:
            logger.error(f"Error handling path {parsed_path.path}: {e}", exc_info=True)
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

    def handle_route(self, path: str, req_data: dict) -> dict:
        if path == "/remember":
            text = req_data.get("text")
            metadata = req_data.get("metadata")
            atom_id = self.db.remember(text, metadata)
            return {"atom_id": atom_id}
            
        elif path == "/remember_batch":
            items = req_data.get("items")
            atom_ids = self.db.remember_batch(items)
            return {"atom_ids": atom_ids}
            
        elif path == "/get":
            memory_id = req_data.get("memory_id")
            memory = self.db.get(memory_id)
            if memory:
                return memory._atom.to_dict()
            return {"memory": None}
            
        elif path == "/update":
            memory_id = req_data.get("memory_id")
            text = req_data.get("text")
            metadata = req_data.get("metadata")
            self.db.update(memory_id, text, metadata)
            return {"status": "ok"}
            
        elif path == "/delete":
            memory_id = req_data.get("memory_id")
            hard = req_data.get("hard", False)
            self.db.delete(memory_id, hard)
            return {"status": "ok"}
            
        elif path == "/query":
            text = req_data.get("text")
            k = req_data.get("k", 5)
            filters = req_data.get("filters")
            min_score = req_data.get("min_score", 0.0)
            memories = self.db.query(text, k=k, filters=filters, min_score=min_score)
            return {"memories": [self._serialize_memory(m) for m in memories]}
            
        elif path == "/multi_hop":
            text = req_data.get("text")
            hops = req_data.get("hops", 2)
            k = req_data.get("k", 5)
            filters = req_data.get("filters")
            memories = self.db.multi_hop(text, hops=hops, k=k, filters=filters)
            return {"memories": [self._serialize_memory(m) for m in memories]}
            
        elif path == "/get_timeline":
            entity_id = req_data.get("entity_id")
            start = req_data.get("start")
            end = req_data.get("end")
            memories = self.db.get_timeline(entity_id=entity_id, start=start, end=end)
            return {"memories": [self._serialize_memory(m) for m in memories]}
            
        elif path == "/stats":
            return self.db.stats()
            
        elif path == "/flush":
            self.db.flush()
            return {"status": "ok"}
            
        elif path == "/compact":
            self.db.compact()
            return {"status": "ok"}
            
        elif path == "/close":
            self.db.close()
            return {"status": "ok"}

        else:
            raise ValueError(f"Unknown route: {path}")

    def _serialize_memory(self, memory) -> dict:
        return memory._atom.to_dict()

class ThreadingEpochDBServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

def start_server(db: EpochDB, host: str = "127.0.0.1", port: int = 8080) -> ThreadingEpochDBServer:
    class CustomHandler(EpochDBRequestHandler):
        pass
    CustomHandler.db = db
    server = ThreadingEpochDBServer((host, port), CustomHandler)
    logger.info(f"Initialized EpochDB HTTP server on {host}:{port}")
    return server
