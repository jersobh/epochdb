import os
import json
import logging
from typing import Dict, Any

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

    def __init__(self, wal_path: str):
        self.wal_path = wal_path
        self._file = open(self.wal_path, "a")

    def append(self, operation: str, data: Dict[str, Any]):
        record = json.dumps({"op": operation, "data": data})
        self._file.write(record + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

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
                        elif op in ("COMMIT", "ROLLBACK"):
                            # COMMIT → these atoms are safe; clear pending set.
                            # ROLLBACK → discard pending atoms.
                            pending = []
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.error(f"Failed to read WAL for replay: {e}")
            return []

        return pending

    def close(self):
        self._file.close()

    def clear(self):
        """Called upon successful Epoch Checkpoint."""
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
