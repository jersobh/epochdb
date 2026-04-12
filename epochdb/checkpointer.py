import os
import json
import base64
import logging
from datetime import datetime, timezone
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


def _typed_to_json(typed: Tuple[str, bytes]) -> dict:
    """
    Convert the (type_str, bytes) tuple returned by serde.dumps_typed()
    into a JSON-safe dict by base64-encoding the bytes payload.
    """
    type_str, data_bytes = typed
    return {
        "__type__": type_str,
        "__data__": base64.b64encode(data_bytes).decode("ascii"),
    }


def _json_to_typed(obj: dict) -> Tuple[str, bytes]:
    """Restore a (type_str, bytes) tuple from the JSON-safe dict."""
    return (obj["__type__"], base64.b64decode(obj["__data__"]))


class EpochDBCheckpointer(BaseCheckpointSaver):
    """
    A LangGraph checkpointer that uses EpochDB's storage directory for
    persistence.

    Checkpoints are stored as UTF-8 JSON files (.ckpt.json).  The binary
    payload produced by serde.dumps_typed() is base64-encoded so it survives
    the JSON round-trip cleanly.

    Backward compatibility: if a legacy pickle (.ckpt) file exists for a
    given checkpoint ID it is loaded transparently and re-written as JSON on
    the first read.
    """

    def __init__(
        self,
        db: Any,  # EpochDB instance — provides storage_dir.
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde or JsonPlusSerializer())
        self.db = db
        self.checkpoint_dir = os.path.join(self.db.storage_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------

    def _thread_path(self, thread_id: str) -> str:
        path = os.path.join(self.checkpoint_dir, thread_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _ckpt_path(self, thread_id: str, checkpoint_id: str) -> str:
        return os.path.join(self._thread_path(thread_id), f"{checkpoint_id}.ckpt.json")

    def _legacy_ckpt_path(self, thread_id: str, checkpoint_id: str) -> str:
        return os.path.join(self._thread_path(thread_id), f"{checkpoint_id}.ckpt")

    def _writes_path(self, thread_id: str, checkpoint_id: str, task_id: str) -> str:
        return os.path.join(
            self._thread_path(thread_id),
            f"{checkpoint_id}.{task_id}.writes.json",
        )

    # -------------------------------------------------------------------------
    # Checkpoint listing
    # -------------------------------------------------------------------------

    def _list_checkpoint_ids(self, thread_id: str) -> List[str]:
        thread_path = self._thread_path(thread_id)
        ids: set = set()
        for fname in os.listdir(thread_path):
            if fname.endswith(".ckpt.json"):
                ids.add(fname[: -len(".ckpt.json")])
            elif fname.endswith(".ckpt"):
                # Legacy pickle — include for migration on first read.
                ids.add(fname[: -len(".ckpt")])
        return sorted(ids, reverse=True)

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def _read_checkpoint_file(
        self, thread_id: str, checkpoint_id: str
    ) -> Optional[dict]:
        """
        Load a checkpoint file.  Prefers the JSON format; falls back to legacy
        pickle and migrates it to JSON on the first successful read.
        """
        json_path = self._ckpt_path(thread_id, checkpoint_id)
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f"Corrupt or unreadable checkpoint file {json_path!r}: {e}. "
                    "Treating as missing and falling through to pickle fallback."
                )
                # Rename the bad file so it doesn't block future reads.
                try:
                    os.rename(json_path, json_path + ".corrupt")
                except OSError:
                    pass

        # ── Legacy pickle fallback ──────────────────────────────────────────
        legacy_path = self._legacy_ckpt_path(thread_id, checkpoint_id)
        if os.path.exists(legacy_path):
            try:
                import pickle

                with open(legacy_path, "rb") as f:
                    raw = pickle.load(f)

                logger.info(
                    f"Migrating legacy pickle checkpoint {checkpoint_id!r} to JSON."
                )

                # raw["checkpoint"] is already a (type_str, bytes) tuple from
                # the old pickle format — convert it to our JSON-safe envelope.
                migrated = {
                    "checkpoint": _typed_to_json(raw["checkpoint"]),
                    "metadata": raw.get("metadata", {}),
                    "parent_config": raw.get("parent_config"),
                    "timestamp": raw.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(migrated, f)

                return migrated
            except Exception as e:
                logger.error(f"Failed to load/migrate legacy pickle checkpoint: {e}")

        return None

    # -------------------------------------------------------------------------
    # BaseCheckpointSaver interface
    # -------------------------------------------------------------------------

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """Retrieve a checkpoint tuple from the store."""
        thread_id = config["configurable"].get("thread_id")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if not thread_id:
            return None

        if not checkpoint_id:
            ids = self._list_checkpoint_ids(thread_id)
            if not ids:
                return None
            checkpoint_id = ids[0]

        data = self._read_checkpoint_file(thread_id, checkpoint_id)
        if data is None:
            return None

        return CheckpointTuple(
            config=config,
            checkpoint=self.serde.loads_typed(_json_to_typed(data["checkpoint"])),
            metadata=data["metadata"],
            parent_config=data.get("parent_config"),
        )

    def list(
        self,
        config: dict,
        *,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread, newest first."""
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return

        ids = self._list_checkpoint_ids(thread_id)
        count = 0
        for cp_id in ids:
            if limit and count >= limit:
                break
            cp_config = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": cp_id}
            }
            tup = self.get_tuple(cp_config)
            if tup:
                yield tup
                count += 1

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> dict:
        """Serialise and store a checkpoint as a JSON file."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]

        # serde.dumps_typed() returns (type_str, bytes) — not JSON-safe.
        # Wrap it in a base64 envelope before dumping.
        data = {
            "checkpoint": _typed_to_json(self.serde.dumps_typed(checkpoint)),
            "metadata": metadata,
            "parent_config": config.get("parent_config"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        path = self._ckpt_path(thread_id, checkpoint_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: dict,
        writes: Sequence[tuple],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        # Each write is (channel: str, value: Any).
        # serde.dumps_typed(value) also returns (type_str, bytes) — same treatment.
        serialized = [
            [channel, _typed_to_json(self.serde.dumps_typed(value))]
            for channel, value in writes
        ]

        path = self._writes_path(thread_id, checkpoint_id, task_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialized, f)
