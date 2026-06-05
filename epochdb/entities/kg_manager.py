import sqlite3
import os
import logging
import time
import threading
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

class KGManager:
    """
    Manages the Global Entity Index (Knowledge Graph) using SQLite.
    Provides 'Disk-Direct' relational lookups without loading the whole graph into RAM.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._setup_db()

    def _setup_db(self):
        """Initialize schema and indices."""
        with self._lock:
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_lower ON kg_index (LOWER(entity))")
            self._conn.commit()

    def add_association(self, entity: str, atom_id: str, epoch_id: str):
        """Add an entity -> [atom_id, epoch_id] association."""
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute(
                    "INSERT OR IGNORE INTO kg_index (entity, atom_id, epoch_id) VALUES (?, ?, ?)",
                    (entity, atom_id, epoch_id)
                )
            except Exception as e:
                logger.error(f"Failed to add association to SQLite KG: {e}")

    def add_associations_batch(self, associations: List[Tuple[str, str, str]]):
        """Bulk add associations."""
        if not associations:
            return
        with self._lock:
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
        with self._lock:
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

    def get_associations_batch(self, entities: List[str]) -> Dict[str, List[List[str]]]:
        """Retrieve all [atom_id, epoch_id] pairs for a batch of entities."""
        if not entities:
            return {}
        with self._lock:
            try:
                cursor = self._conn.cursor()
                results = {}
                for i in range(0, len(entities), 500):
                    chunk = entities[i:i+500]
                    placeholders = ",".join("?" for _ in chunk)
                    cursor.execute(
                        f"SELECT entity, atom_id, epoch_id FROM kg_index WHERE entity IN ({placeholders})",
                        chunk
                    )
                    for ent, atom_id, epoch_id in cursor.fetchall():
                        if ent not in results:
                            results[ent] = []
                        results[ent].append([atom_id, epoch_id])
                for ent in entities:
                    if ent not in results:
                        results[ent] = []
                return results
            except Exception as e:
                logger.error(f"Failed to query SQLite KG for batch of entities: {e}")
                return {ent: [] for ent in entities}

    def get_all_entities(self) -> List[str]:
        """Returns all distinct entities in the KG."""
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute("SELECT DISTINCT entity FROM kg_index")
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to fetch all entities: {e}")
                return []

    def get_entities_by_prefix(self, prefix: str) -> List[str]:
        """Returns all distinct entities starting with the given prefix."""
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT entity FROM kg_index WHERE entity LIKE ?",
                    (f"{prefix}%",)
                )
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to fetch entities by prefix '{prefix}': {e}")
                return []

    def claim_entity(self, entity: str, claimer: str, expiry_secs: int = 30) -> bool:
        """
        Atomically attempt to claim an entity using a special CLAIM entity.
        Returns True if successful, False if already claimed and not expired.
        """
        claim_key = f"CLAIM:{entity}"
        now = time.time()
        with self._lock:
            try:
                cursor = self._conn.cursor()
                # 1. Check if a valid claim already exists
                cursor.execute(
                    "SELECT atom_id FROM kg_index WHERE entity = ?",
                    (claim_key,)
                )
                existing = cursor.fetchone()
                if existing:
                    try:
                        expiry = float(existing[0])
                        if now < expiry:
                            return False # Still claimed
                    except ValueError:
                        pass # Invalid expiry, allow overwrite

                # 2. Set new claim (atom_id used as expiry timestamp)
                expiry = now + expiry_secs
                cursor.execute(
                    "INSERT OR REPLACE INTO kg_index (entity, atom_id, epoch_id) VALUES (?, ?, ?)",
                    (claim_key, str(expiry), claimer)
                )
                self._conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to claim entity '{entity}': {e}")
                return False

    def unclaim_entity(self, entity: str, claimer: str) -> bool:
        """Release a claim if held by the given claimer."""
        claim_key = f"CLAIM:{entity}"
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute(
                    "DELETE FROM kg_index WHERE entity = ? AND epoch_id = ?",
                    (claim_key, claimer)
                )
                self._conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Failed to unclaim entity '{entity}': {e}")
                return False

    def get_lineage(self, entity: str) -> List[Tuple[str, str, float]]:
        """
        Returns chronological lineage of atom associations for an entity.
        """
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute(
                    "SELECT atom_id, epoch_id FROM kg_index WHERE entity = ?",
                    (entity,)
                )
                return cursor.fetchall()
            except Exception as e:
                logger.error(f"Failed to fetch lineage for '{entity}': {e}")
                return []

    def get_candidate_entities(self, sig_words: List[str]) -> List[str]:
        """Fetch candidate entities that match any of the significant words or contain acronym parentheses."""
        if not sig_words:
            # If no sig words, we can't do LIKE match, return empty to be safe
            return []
        with self._lock:
            try:
                cursor = self._conn.cursor()
                conditions = []
                params = []
                for w in sig_words:
                    conditions.append("entity LIKE ?")
                    params.append(f"%{w}%")
                
                # Also always pull entities with parentheses for acronym matches
                conditions.append("entity LIKE '%(%)%'")
                
                query = "SELECT DISTINCT entity FROM kg_index WHERE " + " OR ".join(conditions)
                cursor.execute(query, params)
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to fetch candidate entities for {sig_words}: {e}")
                return []

    def get_neighbors(self, entity: str) -> List[str]:
        """Retrieve all neighbor entities that share at least one atom_id with the given entity."""
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute(
                    """
                    SELECT DISTINCT entity FROM kg_index 
                    WHERE atom_id IN (SELECT atom_id FROM kg_index WHERE entity = ?)
                    AND entity != ?
                    """,
                    (entity, entity)
                )
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to query neighbors for '{entity}': {e}")
                return []

    def remove_associations(self, atom_ids: List[str]):
        """Remove associations for deleted/superseded atoms."""
        if not atom_ids:
            return
        with self._lock:
            try:
                cursor = self._conn.cursor()
                for i in range(0, len(atom_ids), 500):
                    chunk = atom_ids[i:i+500]
                    placeholders = ",".join("?" for _ in chunk)
                    cursor.execute(
                        f"DELETE FROM kg_index WHERE atom_id IN ({placeholders})",
                        chunk
                    )
                self._conn.commit()
            except Exception as e:
                logger.error(f"Failed to remove associations from SQLite KG: {e}")

    def update_epoch_ids(self, old_epochs: List[str], new_epoch_id: str):
        """Update epoch_id for relocated atoms during compaction."""
        if not old_epochs:
            return
        with self._lock:
            try:
                cursor = self._conn.cursor()
                placeholders = ",".join("?" for _ in old_epochs)
                cursor.execute(
                    f"UPDATE kg_index SET epoch_id = ? WHERE epoch_id IN ({placeholders})",
                    [new_epoch_id] + old_epochs
                )
                self._conn.commit()
            except Exception as e:
                logger.error(f"Failed to update epoch IDs in SQLite KG: {e}")

    def commit(self):
        """Force a commit to disk."""
        with self._lock:
            self._conn.commit()

    def close(self):
        """Flush and close the database connection."""
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass
