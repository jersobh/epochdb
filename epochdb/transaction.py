import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FileLock:
    """A simple file-based lock to prevent concurrent epochdb intances from writing."""
    def __init__(self, lock_path: str):
        self.lock_path = lock_path

    def acquire(self):
        if os.path.exists(self.lock_path):
            raise RuntimeError(f"Database is locked by another process: {self.lock_path}")
        with open(self.lock_path, "w") as f:
            f.write(str(os.getpid()))

    def release(self):
        if os.path.exists(self.lock_path):
            os.remove(self.lock_path)


class WriteAheadLog:
    """Append-only JSONL log for crash recovery."""
    def __init__(self, wal_path: str):
        self.wal_path = wal_path
        self._file = open(self.wal_path, "a")

    def append(self, operation: str, data: Dict[str, Any]):
        record = json.dumps({"op": operation, "data": data})
        self._file.write(record + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self):
        self._file.close()

    def clear(self):
        """Called upon successful Epoch Checkpoint."""
        self._file.close()
        open(self.wal_path, "w").close()
        self._file = open(self.wal_path, "a")


class MultiIndexTransaction:
    """
    Context manager to ensure an atom is written to the WAL,
    the Vector Index, and the Knowledge Graph atomically.
    """
    def __init__(self, wal: WriteAheadLog, hot_tier):
        self.wal = wal
        self.hot_tier = hot_tier
        self.pending_atoms = []
    
    def __enter__(self):
        self.pending_atoms = []
        return self

    def add(self, atom):
        self.pending_atoms.append(atom)
        self.wal.append("ADD", atom.to_dict())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Transaction failed, rolling back. Reason: {exc_val}")
            self.wal.append("ROLLBACK", {})
            return False 
        
        for atom in self.pending_atoms:
            self.hot_tier._add_atom(atom)
        self.wal.append("COMMIT", {})
        return True
