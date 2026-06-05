import logging
from typing import Dict, List, Set, Tuple, Any, Callable
from epochdb.core.atom import UnifiedMemoryAtom, PayloadType

logger = logging.getLogger(__name__)

class CascadeManager:
    """Manages reactive re-evaluation of dependencies on quantitative updates."""
    def __init__(self, storage_dir: str = None):
        # field -> list of (dependent_atom_id, callback)
        self.dependency_graph: Dict[str, List[Tuple[str, Callable]]] = {}
        # Persistent dependencies: constraint_id -> set of fields it depends on
        self.constraint_dependencies: Dict[str, Set[str]] = {}
        self.storage_dir = storage_dir
        if self.storage_dir:
            self.load_from_disk()

    def register_constraint(self, constraint_id: str, fields: Set[str], callback: Callable):
        """Registers a constraint and its field dependencies."""
        self.constraint_dependencies[constraint_id] = fields
        for field in fields:
            self.register_dependency(field, constraint_id, callback)
        self.save_to_disk()

    def register_dependency(self, field: str, dependent_id: str, callback: Callable):
        if field not in self.dependency_graph:
            self.dependency_graph[field] = []
        self.dependency_graph[field].append((dependent_id, callback))

    def trigger_cascade(self, field: str, atom: UnifiedMemoryAtom):
        if field not in self.dependency_graph:
            return

        logger.info(f"Triggering cascade for field: {field}")
        for dep_id, callback in self.dependency_graph[field]:
            try:
                # Re-evaluate dependent (e.g. constraint)
                # In a real system, we'd emit an event if the constraint becomes infeasible
                callback(atom)
            except Exception as e:
                logger.error(f"Cascade failed for {dep_id}: {e}")

    def save_to_disk(self):
        if not self.storage_dir:
            return
        import os, json
        path = os.path.join(self.storage_dir, "cascade_graph.json")
        try:
            data = {k: list(v) for k, v in self.constraint_dependencies.items()}
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save cascade graph: {e}")

    def load_from_disk(self):
        if not self.storage_dir:
            return
        import os, json
        path = os.path.join(self.storage_dir, "cascade_graph.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    self.constraint_dependencies = {k: set(v) for k, v in data.items()}
                    # Note: callbacks are not persisted, they must be re-registered 
                    # during index_atom replay.
            except Exception as e:
                logger.error(f"Failed to load cascade graph: {e}")

    def clear(self):
        self.dependency_graph = {}
        self.constraint_dependencies = {}
        self.save_to_disk()
