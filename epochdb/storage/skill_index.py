"""Hot-tier indexes for MemoryType.SKILL lookups."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from epochdb.core.atom import MemoryType, UnifiedMemoryAtom
from epochdb.retrieval.keyword_index import atom_text


def is_skill_atom(atom: UnifiedMemoryAtom) -> bool:
    meta = atom.metadata or {}
    mt = atom.memory_type.value if atom.memory_type else "general"
    return (
        mt == MemoryType.SKILL.value
        or meta.get("type") in ("skill", "process_summary")
        or bool(meta.get("skill_name"))
    )


def skill_lookup_keys(atom: UnifiedMemoryAtom) -> List[str]:
    meta = atom.metadata or {}
    keys: List[str] = []
    for raw in (
        meta.get("skill_name"),
        meta.get("title"),
        meta.get("skill_id"),
        meta.get("process_id"),
        atom.id,
    ):
        if raw is None:
            continue
        key = str(raw).strip().lower()
        if key and key not in keys:
            keys.append(key)
    return keys


class SkillIndex:
    """Maps skill names/ids and memory_type tags to hot-tier atom ids."""

    def __init__(self) -> None:
        self._skill_ids: Set[str] = set()
        self._name_to_id: Dict[str, str] = {}
        self._memory_type_ids: Dict[str, Set[str]] = {}

    def index_atom(self, atom: UnifiedMemoryAtom) -> None:
        self.unindex_atom(atom.id, atom)

        mt = atom.memory_type.value if atom.memory_type else MemoryType.GENERAL.value
        self._memory_type_ids.setdefault(mt, set()).add(atom.id)

        if not is_skill_atom(atom):
            return

        self._skill_ids.add(atom.id)
        for key in skill_lookup_keys(atom):
            self._name_to_id[key] = atom.id

    def unindex_atom(self, atom_id: str, atom: Optional[UnifiedMemoryAtom] = None) -> None:
        self._skill_ids.discard(atom_id)

        for mt, ids in list(self._memory_type_ids.items()):
            ids.discard(atom_id)
            if not ids:
                self._memory_type_ids.pop(mt, None)

        if atom is not None:
            for key in skill_lookup_keys(atom):
                if self._name_to_id.get(key) == atom_id:
                    self._name_to_id.pop(key, None)
        else:
            for key, mapped in list(self._name_to_id.items()):
                if mapped == atom_id:
                    self._name_to_id.pop(key, None)

    def clear(self) -> None:
        self._skill_ids.clear()
        self._name_to_id.clear()
        self._memory_type_ids.clear()

    def list_skill_ids(self) -> List[str]:
        return list(self._skill_ids)

    def resolve(self, needle: str) -> Optional[str]:
        key = (needle or "").strip().lower()
        if not key:
            return None
        return self._name_to_id.get(key)

    def ids_for_memory_type(self, memory_type: str) -> Set[str]:
        return set(self._memory_type_ids.get(memory_type, set()))
