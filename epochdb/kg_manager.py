import sqlite3
import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class KGManager:
    """
    Manages the Global Entity Index (Knowledge Graph) using SQLite.
    Provides 'Disk-Direct' relational lookups without loading the whole graph into RAM.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._setup_db()

    def _setup_db(self):
        """Initialize schema and indices."""
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_index (
                entity TEXT,
                atom_id TEXT,
                epoch_id TEXT,
                UNIQUE(entity, atom_id, epoch_id)
            )
        """)
        # Index for fast entity lookups (Stage 1a and Stage 3)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity ON kg_index (entity)")
        self._conn.commit()

    def add_association(self, entity: str, atom_id: str, epoch_id: str):
        """Add an entity -> [atom_id, epoch_id] association."""
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO kg_index (entity, atom_id, epoch_id) VALUES (?, ?, ?)",
                (entity, atom_id, epoch_id)
            )
            # We don't commit on every write for performance; we rely on explicit flush or close.
            # But the user requested "batching" in engine.py, so we can commit periodically.
        except Exception as e:
            logger.error(f"Failed to add association to SQLite KG: {e}")

    def add_associations_batch(self, associations: List[Tuple[str, str, str]]):
        """Bulk add associations."""
        if not associations:
            return
        try:
            cursor = self._conn.cursor()
            cursor.executemany(
                "INSERT OR IGNORE INTO kg_index (entity, atom_id, epoch_id) VALUES (?, ?, ?)",
                associations
            )
            self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to batch add associations to SQLite KG: {e}")

    def get_associations(self, entity: str) -> List[List[str]]:
        """Retrieve all [atom_id, epoch_id] pairs for a given entity."""
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT atom_id, epoch_id FROM kg_index WHERE entity = ?",
                (entity,)
            )
            return [list(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to query SQLite KG for entity '{entity}': {e}")
            return []

    def get_all_entities(self) -> List[str]:
        """Returns all distinct entities in the KG."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT DISTINCT entity FROM kg_index")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to fetch all entities: {e}")
            return []

    def commit(self):
        """Force a commit to disk."""
        self._conn.commit()

    def close(self):
        """Flush and close the database connection."""
        try:
            self._conn.commit()
            self._conn.close()
        except Exception:
            pass
