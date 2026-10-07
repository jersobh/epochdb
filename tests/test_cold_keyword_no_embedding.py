"""Keyword cold search must work without loading the embedding column."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from epochdb.storage.cold_tier import ColdTier


def test_search_keyword_without_embedding_column():
    with tempfile.TemporaryDirectory() as tmp:
        cold = ColdTier(tmp)
        epoch_id = "epoch_kw"
        table = pa.table(
            {
                "id": ["a1", "a2"],
                "payload": ['"needle in haystack"', '"unrelated text"'],
                "payload_type": ["text", "text"],
                "metadata": ["{}", "{}"],
                "created_at": [1.0, 2.0],
                "access_count": [0, 0],
                "epoch_id": [epoch_id, epoch_id],
                "embedding": [np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)],
                "embedding_max": [1.0, 1.0],
            }
        )
        path = os.path.join(tmp, f"{epoch_id}.parquet")
        pq.write_table(table, path)

        hits = cold.search_keyword("needle", epochs=[epoch_id], top_k=5)
        assert len(hits) == 1
        atom, score = hits[0]
        assert atom.id == "a1"
        assert score > 0
        assert atom.embedding is not None
        assert len(atom.embedding) == 0


def test_row_to_atom_missing_embedding():
    cold = ColdTier(tempfile.gettempdir())
    atom = cold._row_to_atom(
        {
            "id": "x",
            "payload": '"hello"',
            "payload_type": "text",
            "metadata": "{}",
        }
    )
    assert atom.id == "x"
    assert atom.payload == "hello"
    assert len(atom.embedding) == 0
    assert atom.created_at == 0.0
    assert atom.access_count == 0
