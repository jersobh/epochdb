import os
import json
import pickle
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
    """A LangGraph checkpointer that uses EpochDB's storage directory for persistence."""

    def __init__(
        self,
        db: Any, # We take the EpochDB instance to share the storage_dir
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde or JsonPlusSerializer())
        self.db = db
        self.checkpoint_dir = os.path.join(self.db.storage_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_path(self, thread_id: str, checkpoint_id: str = None) -> str:
        thread_path = os.path.join(self.checkpoint_dir, thread_id)
        os.makedirs(thread_path, exist_ok=True)
        if checkpoint_id:
            return os.path.join(thread_path, f"{checkpoint_id}.ckpt")
        return thread_path

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the store."""
        thread_id = config["configurable"].get("thread_id")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        if not thread_id:
            return None

        if not checkpoint_id:
            # Get latest checkpoint for this thread
            thread_path = self._get_path(thread_id)
            checkpoints = sorted(
                [f.replace(".ckpt", "") for f in os.listdir(thread_path) if f.endswith(".ckpt")],
                reverse=True
            )
            if not checkpoints:
                return None
            checkpoint_id = checkpoints[0]

        path = self._get_path(thread_id, checkpoint_id)
        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            data = pickle.load(f)
            
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
        """List checkpoints from the store."""
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return
            
        thread_path = self._get_path(thread_id)
        checkpoints = sorted(
            [f.replace(".ckpt", "") for f in os.listdir(thread_path) if f.endswith(".ckpt")],
            reverse=True
        )
        
        count = 0
        for cp_id in checkpoints:
            if limit and count >= limit:
                break
            
            cp_config = {"configurable": {"thread_id": thread_id, "checkpoint_id": cp_id}}
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
        """Store a checkpoint in the store."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        
        path = self._get_path(thread_id, checkpoint_id)
        
        data = {
            "checkpoint": self.serde.dumps_typed(checkpoint),
            "metadata": metadata,
            "parent_config": config.get("parent_config"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
            
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
        
        path = os.path.join(self._get_path(thread_id), f"{checkpoint_id}.{task_id}.writes")
        
        data = [
            (channel, self.serde.dumps_typed(value))
            for channel, value in writes
        ]
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
