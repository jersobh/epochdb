import os
import json
import logging
import threading
import time
from typing import Dict, Any
from datetime import datetime, date

logger = logging.getLogger(__name__)


def _pid_is_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)  # Signal 0: no-op, just checks existence
        return True
    except (ProcessLookupError, PermissionError):
        # ProcessLookupError → process is gone
        # PermissionError → process exists but we can't signal it (still alive)
        return isinstance(
            Exception(), PermissionError
        )  # PermissionError means alive
    except Exception:
        return False


def _pid_is_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # We can't signal it, but the process exists.
        return True
    except Exception:
        return False


class FileLock:
    """
    A simple file-based lock to prevent concurrent EpochDB instances
    from writing to the same storage directory.

    Uses an atomic O_CREAT|O_EXCL open to eliminate the TOCTOU race condition.
    Automatically removes stale locks left behind by crashed processes.
    """

    def __init__(self, lock_path: str):
        self.lock_path = lock_path

    def acquire(self):
        pid = os.getpid()
        try:
            # Atomic exclusive create — raises FileExistsError if lock exists.
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(str(pid))
        except FileExistsError:
            # Lock file already exists — check if the owning process is alive.
            try:
                with open(self.lock_path, "r") as f:
                    existing_pid = int(f.read().strip())
            except (ValueError, OSError):
                # Unreadable lock file → treat as stale.
                existing_pid = None

            if existing_pid and _pid_is_alive(existing_pid):
                raise RuntimeError(
                    f"Database is locked by another process (PID {existing_pid}): "
                    f"{self.lock_path}"
                )
            else:
                # Stale lock — remove it and retry once.
                logger.warning(
                    f"Removing stale lock file (PID {existing_pid} no longer alive): "
                    f"{self.lock_path}"
                )
                try:
                    os.remove(self.lock_path)
                except OSError:
                    pass
                self.acquire()  # One retry after removing the stale lock.

    def release(self):
        try:
            os.remove(self.lock_path)
        except OSError:
            pass


class WriteAheadLog:
    """Append-only JSONL log for crash recovery."""

    def __init__(self, wal_path: str, sync_interval: float = 0.0):
        self.wal_path = wal_path
        self.sync_interval = sync_interval
        self._file = open(self.wal_path, "a")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sync_thread = None

        if self.sync_interval > 0.0:
            self._sync_thread = threading.Thread(target=self._periodic_sync, daemon=True)
            self._sync_thread.start()

    def _periodic_sync(self):
        while not self._stop_event.is_set():
            time.sleep(self.sync_interval)
            with self._lock:
                if not self._file.closed:
                    try:
                        self._file.flush()
                        os.fsync(self._file.fileno())
                    except Exception as e:
                        logger.error(f"Error during periodic WAL fsync: {e}")

    def append(self, operation: str, data: Dict[str, Any]):
        def json_serial(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        record = json.dumps({"op": operation, "data": data}, default=json_serial)
        with self._lock:
            self._file.write(record + "\n")
            self._file.flush()
            if self.sync_interval <= 0.0:
                try:
                    os.fsync(self._file.fileno())
                except OSError as e:
                    logger.error(f"Failed to fsync WAL: {e}")

    def log_delete(self, atom_id: str):
        """Log a persistent delete operation."""
        self.append("DELETE", {"id": atom_id})

    def replay(self) -> list:
        """
        Read uncommitted ADD records from the WAL for crash recovery.
        Returns a list of atom dicts that were written but never committed.
        """
        if not os.path.exists(self.wal_path):
            return []

        pending = []
        try:
            with open(self.wal_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        op = record.get("op")
                        if op == "ADD":
                            pending.append(record["data"])
                        elif op == "COMMIT":
                            # COMMIT means the atoms were successfully committed, so we don't need to replay them.
                            pending = []
                        elif op == "ROLLBACK":
                            # ROLLBACK means discard the pending atoms.
                            pending = []
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.error(f"Failed to read WAL for replay: {e}")
            return []

        return pending

    def close(self):
        if self._sync_thread:
            self._stop_event.set()
            self._sync_thread.join(timeout=1.0)
        with self._lock:
            if not self._file.closed:
                try:
                    self._file.flush()
                    os.fsync(self._file.fileno())
                except Exception:
                    pass
                self._file.close()

    def clear(self):
        """Called upon successful Epoch Checkpoint."""
        with self._lock:
            self._file.close()
            open(self.wal_path, "w").close()
            self._file = open(self.wal_path, "a")


class MultiIndexTransaction:
    """
    Context manager to ensure an atom is written to the WAL
    and the Vector Index atomically.
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
