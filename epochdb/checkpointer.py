import os
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Union

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


class EpochDBCheckpointer(BaseCheckpointSaver):
    """
    A LangGraph checkpointer that uses EpochDB's storage directory for persistence.

    Checkpoints are stored as JSON files (via the `serde` serializer already
    attached to the base class) rather than pickle, making them forward-compatible
    across Python and LangGraph versions.

    Backward compatibility: on read, if a `.ckpt.json` file is not found but a
    legacy `.ckpt` (pickle) file exists, it is loaded and transparently migrated
    to the JSON format.
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
        """Path for legacy pickle-format checkpoints."""
        return os.path.join(self._thread_path(thread_id), f"{checkpoint_id}.ckpt")

    def _writes_path(self, thread_id: str, checkpoint_id: str, task_id: str) -> str:
        return os.path.join(
            self._thread_path(thread_id),
            f"{checkpoint_id}.{task_id}.writes.json",
        )

    # -------------------------------------------------------------------------
    # Sorted checkpoint listing (newest first)
    # -------------------------------------------------------------------------

    def _list_checkpoint_ids(self, thread_id: str) -> List[str]:
        thread_path = self._thread_path(thread_id)
        ids = set()
        for f in os.listdir(thread_path):
            if f.endswith(".ckpt.json"):
                ids.add(f[: -len(".ckpt.json")])
            elif f.endswith(".ckpt"):
                # Legacy pickle files — include so they can be migrated on read.
                ids.add(f[: -len(".ckpt")])
        return sorted(ids, reverse=True)

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------

    def _dump(self, data: dict) -> str:
        return json.dumps(data)

    def _load(self, raw: str) -> dict:
        return json.loads(raw)

    def _read_checkpoint_file(
        self, thread_id: str, checkpoint_id: str
    ) -> Optional[dict]:
        """
        Read a checkpoint file, preferring the JSON format.
        Falls back to legacy pickle format and migrates it to JSON on success.
        """
        json_path = self._ckpt_path(thread_id, checkpoint_id)
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Legacy pickle fallback.
        legacy_path = self._legacy_ckpt_path(thread_id, checkpoint_id)
        if os.path.exists(legacy_path):
            try:
                import pickle
                with open(legacy_path, "rb") as f:
                    data = pickle.load(f)
                logger.info(
                    f"Migrating legacy pickle checkpoint {checkpoint_id} to JSON."
                )
                # Write JSON version alongside the pickle so future reads are fast.
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return data
            except Exception as e:
                logger.error(f"Failed to load legacy pickle checkpoint: {e}")

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
            checkpoint=self.serde.loads_typed(data["checkpoint"]),
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
        """Store a checkpoint as a JSON file."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]

        data = {
            "checkpoint": self.serde.dumps_typed(checkpoint),
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
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        serialized = [
            (channel, self.serde.dumps_typed(value)) for channel, value in writes
        ]

        path = self._writes_path(thread_id, checkpoint_id, task_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialized, f)
