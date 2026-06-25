import os
import json
import logging
import threading
import time
import subprocess
import ctypes
from typing import Dict, Any
from datetime import datetime, date

logger = logging.getLogger(__name__)

# Compile and Load io_uring C Helper dynamically
HELPER_SO = os.path.join(os.path.dirname(__file__), "uring_helper.so")
HELPER_C = os.path.join(os.path.dirname(__file__), "uring_helper.c")

_helper_loaded = False
_lib = None

def _compile_helper():
    if not os.path.exists(HELPER_C):
        return False
    try:
        cmd = [
            "gcc", "-O3", "-shared", "-fPIC",
            "-o", HELPER_SO, HELPER_C,
            "-luring"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0
    except Exception:
        return False

if not os.path.exists(HELPER_SO):
    _compile_helper()

if os.path.exists(HELPER_SO):
    try:
        _lib = ctypes.CDLL(HELPER_SO)
        _lib.uring_writer_open.argtypes = [ctypes.c_char_p]
        _lib.uring_writer_open.restype = ctypes.c_void_p
        
        _lib.uring_writer_write.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int]
        _lib.uring_writer_write.restype = ctypes.c_int
        
        _lib.uring_writer_reap_completions.argtypes = [ctypes.c_void_p]
        _lib.uring_writer_reap_completions.restype = None
        
        _lib.uring_writer_close.argtypes = [ctypes.c_void_p]
        _lib.uring_writer_close.restype = None
        
        _helper_loaded = True
    except Exception as e:
        logger.warning(f"Could not load uring_helper.so: {e}")


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


class WALFileWrapper:
    def __init__(self, wal):
        self._wal = wal

    def flush(self):
        with self._wal._lock:
            if self._wal.uring_writer is None and self._wal._real_file:
                try:
                    self._wal._real_file.flush()
                except Exception:
                    pass

    @property
    def closed(self) -> bool:
        if self._wal.uring_writer is not None:
            return False
        return self._wal._real_file.closed if self._wal._real_file else True

    def write(self, data):
        if self._wal.uring_writer is None and self._wal._real_file:
            self._wal._real_file.write(data)

    def close(self):
        if self._wal.uring_writer is None and self._wal._real_file:
            self._wal._real_file.close()

    def fileno(self):
        if self._wal.uring_writer is None and self._wal._real_file:
            return self._wal._real_file.fileno()
        return -1


class WriteAheadLog:
    """Append-only JSONL log for crash recovery."""

    def __init__(self, wal_path: str, sync_interval: float = 0.0, use_uring: bool = True):
        self.wal_path = wal_path
        self.sync_interval = sync_interval
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sync_thread = None
        
        self.uring_writer = None
        if use_uring and _helper_loaded:
            self.uring_writer = _lib.uring_writer_open(self.wal_path.encode('utf-8'))
            
        if self.uring_writer is None:
            self._real_file = open(self.wal_path, "a")
        else:
            self._real_file = None

        self._file_wrapper = WALFileWrapper(self)

        if self.sync_interval > 0.0:
            self._sync_thread = threading.Thread(target=self._periodic_sync, daemon=True)
            self._sync_thread.start()

    @property
    def _file(self):
        return self._file_wrapper

    def _periodic_sync(self):
        while not self._stop_event.is_set():
            time.sleep(self.sync_interval)
            with self._lock:
                if self.uring_writer is not None:
                    _lib.uring_writer_reap_completions(self.uring_writer)
                elif self._real_file and not self._real_file.closed:
                    try:
                        self._real_file.flush()
                        os.fsync(self._real_file.fileno())
                    except Exception as e:
                        logger.error(f"Error during periodic WAL fsync: {e}")

    def append(self, operation: str, data: Dict[str, Any]):
        def json_serial(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        record = json.dumps({"op": operation, "data": data}, default=json_serial)
        record_bytes = (record + "\n").encode('utf-8')
        
        with self._lock:
            if self.uring_writer is not None:
                sync = 1 if self.sync_interval <= 0.0 else 0
                _lib.uring_writer_write(self.uring_writer, record_bytes, len(record_bytes), sync)
                if not sync:
                    _lib.uring_writer_reap_completions(self.uring_writer)
            else:
                self._real_file.write(record + "\n")
                self._real_file.flush()
                if self.sync_interval <= 0.0:
                    try:
                        os.fsync(self._real_file.fileno())
                    except OSError as e:
                        logger.error(f"Failed to fsync WAL: {e}")

    def log_delete(self, atom_id: str):
        """Log a persistent delete operation."""
        self.append("DELETE", {"id": atom_id})

    def replay(self) -> list:
        """
        Read committed ADD records from the WAL for crash recovery.
        Returns a list of atom dicts that were successfully committed.
        """
        if not os.path.exists(self.wal_path):
            return []

        committed = []
        current_tx = []
        try:
            with open(self.wal_path, "r", errors="ignore") as f:
                for line in f:
                    line = line.strip('\0 \n\r')
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        op = record.get("op")
                        if op == "ADD":
                            current_tx.append(record["data"])
                        elif op == "COMMIT":
                            committed.extend(current_tx)
                            current_tx = []
                        elif op == "ROLLBACK":
                            current_tx = []
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.error(f"Failed to read WAL for replay: {e}")
            return []

        # Any uncommitted additions left in current_tx at EOF are discarded (rolled back).
        return committed

    def close(self):
        if self._sync_thread:
            self._stop_event.set()
            self._sync_thread.join(timeout=1.0)
        with self._lock:
            if self.uring_writer is not None:
                _lib.uring_writer_close(self.uring_writer)
                self.uring_writer = None
            elif self._real_file and not self._real_file.closed:
                try:
                    self._real_file.flush()
                    os.fsync(self._real_file.fileno())
                except Exception:
                    pass
                self._real_file.close()

    def clear(self):
        """Called upon successful Epoch Checkpoint."""
        with self._lock:
            if self.uring_writer is not None:
                _lib.uring_writer_close(self.uring_writer)
                open(self.wal_path, "w").close()
                self.uring_writer = _lib.uring_writer_open(self.wal_path.encode('utf-8'))
            else:
                self._real_file.close()
                open(self.wal_path, "w").close()
                self._real_file = open(self.wal_path, "a")


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
