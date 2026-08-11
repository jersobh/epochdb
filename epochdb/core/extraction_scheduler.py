"""
Background (non-blocking) triple extraction scheduler.

``remember`` / ``remember_batch`` return immediately with seed triples; a
worker thread runs the configured extraction model and merges results into
the atom + knowledge graph via ``replace_memory``.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def merge_triples(*groups) -> List[Tuple[str, str, str]]:
    merged: List[Tuple[str, str, str]] = []
    seen = set()
    for group in groups:
        for t in group or []:
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                key = (str(t[0]), str(t[1]), str(t[2]))
            else:
                continue
            if not key[0] or not key[1] or not key[2] or key in seen:
                continue
            seen.add(key)
            merged.append(key)
    return merged


class ExtractionScheduler:
    def __init__(self, engine, max_workers: int = 1):
        self._engine = engine
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(max_workers or 1)),
            thread_name_prefix="epochdb-extract",
        )
        self._futures: Set[concurrent.futures.Future] = set()
        self._lock = threading.Lock()
        self._closed = False

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._futures)

    def submit(
        self,
        atom_id: str,
        text: str,
        seed_triples: Optional[List[Tuple[str, str, str]]] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[concurrent.futures.Future]:
        with self._lock:
            if self._closed:
                return None
            fut = self._executor.submit(
                self._run,
                atom_id,
                text,
                list(seed_triples or []),
                dict(metadata or {}),
            )
            self._futures.add(fut)
            fut.add_done_callback(self._on_done)
            return fut

    def _on_done(self, fut: concurrent.futures.Future) -> None:
        with self._lock:
            self._futures.discard(fut)
        try:
            fut.result()
        except Exception as e:
            logger.error(f"Background triple extraction failed: {e}")

    def _run(
        self,
        atom_id: str,
        text: str,
        seed_triples: List[Tuple[str, str, str]],
        metadata: dict,
    ) -> None:
        engine = self._engine
        try:
            extractor = engine._get_fact_extractor()
            discovered = extractor.extract(text) or []
        except Exception as e:
            logger.error(f"Extraction model error for atom {atom_id}: {e}")
            discovered = []

        merged = merge_triples(seed_triples, discovered)

        with engine._internal_lock:
            if atom_id in getattr(engine, "deleted_atom_ids", set()):
                return

            atom = engine.hot_tier.atoms.get(atom_id)
            if atom is None:
                for epoch_id in engine.cold_tier.get_all_epochs():
                    loaded = engine.cold_tier.load_atom_metadata(epoch_id, [atom_id])
                    if loaded:
                        atom = loaded[0]
                        break

            if atom is None:
                logger.warning(f"Skipping extraction merge; atom {atom_id} no longer exists")
                return

            # Preserve any triples written after the seed (e.g. concurrent updates)
            final = merge_triples(atom.triples, merged)
            meta = dict(atom.metadata or {})
            meta.update(metadata or {})
            meta["triples"] = final
            meta["extraction_status"] = "done"
            try:
                extractor = engine._get_fact_extractor()
                meta["extraction_backend"] = getattr(extractor, "backend", None)
                if getattr(extractor, "resolved_model_id", None):
                    meta["extraction_model"] = extractor.resolved_model_id
            except Exception:
                pass

            embedding = atom.embedding
            if embedding is None or (hasattr(embedding, "size") and embedding.size == 0):
                embedding = np.zeros(engine.dim, dtype=np.float32)

            memory_type = getattr(atom, "memory_type", None)
            engine.replace_memory(
                atom_id=atom_id,
                payload=atom.payload,
                embedding=np.asarray(embedding, dtype=np.float32),
                triples=final,
                metadata=meta,
            )
            # Preserve memory_type across replace
            if memory_type is not None:
                hot = engine.hot_tier.atoms.get(atom_id)
                if hot is not None:
                    hot.memory_type = memory_type
                    try:
                        engine.wal.append("ADD", hot.to_dict())
                    except Exception:
                        pass

            logger.debug(
                f"Async extraction merged {len(discovered)} triples into atom {atom_id} "
                f"(total={len(final)})"
            )

    def wait(self, timeout: Optional[float] = None) -> None:
        with self._lock:
            futs = list(self._futures)
        if not futs:
            return
        done, not_done = concurrent.futures.wait(
            futs,
            timeout=timeout,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        for fut in done:
            try:
                fut.result()
            except Exception:
                pass
        if not_done:
            raise TimeoutError(
                f"{len(not_done)} extraction job(s) still pending after timeout={timeout}"
            )

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        try:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
        except TypeError:
            # Python < 3.9 compatibility for cancel_futures kw
            self._executor.shutdown(wait=wait)
