import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from epochdb.engine import OpenAIEmbedder, EpochDB

class TestOpenAIEmbedder(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_value_error(self):
        embedder = OpenAIEmbedder(model_id="text-embedding-3-small", dim=384)
        with self.assertRaises(ValueError) as context:
            embedder.encode("Hello world")
        self.assertIn("OPENAI_API_KEY not found in environment", str(context.exception))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test12345"})
    @patch("requests.post")
    def test_encode_single_text(self, mock_post):
        # Mock API Response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [1.0, 2.0, 3.0]
                }
            ],
            "model": "text-embedding-3-small"
        }
        mock_post.return_value = mock_response

        embedder = OpenAIEmbedder(model_id="text-embedding-3-small", dim=3)
        vector = embedder.encode("Hello world", normalize_embeddings=True)

        # Assert post parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["model"], "text-embedding-3-small")
        self.assertEqual(kwargs["json"]["input"], ["Hello world"])
        self.assertEqual(kwargs["json"]["dimensions"], 3)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-test12345")

        # Verify normalization: [1, 2, 3] / sqrt(14)
        norm = np.linalg.norm(np.array([1.0, 2.0, 3.0]))
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32) / norm
        np.testing.assert_allclose(vector, expected, rtol=1e-5)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test12345"})
    @patch("requests.post")
    def test_encode_batch(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]}
            ],
            "model": "text-embedding-3-small"
        }
        mock_post.return_value = mock_response

        embedder = OpenAIEmbedder(model_id="text-embedding-3-small", dim=2)
        vectors = embedder.encode_batch(["text1", "text2"])

        self.assertEqual(vectors.shape, (2, 2))
        np.testing.assert_allclose(vectors[0], np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(vectors[1], np.array([0.0, 1.0], dtype=np.float32))

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-custom",
        "OPENAI_BASE_URL": "https://custom-gateway.local/v1/"
    })
    @patch("requests.post")
    def test_custom_base_url_resolution(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
            "model": "text-embedding-3-small"
        }
        mock_post.return_value = mock_response

        embedder = OpenAIEmbedder(model_id="text-embedding-3-small", dim=2)
        embedder.encode("test")

        # Verify URL normalization from https://custom-gateway.local/v1/ -> https://custom-gateway.local/v1/embeddings
        mock_post.assert_called_once()
        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "https://custom-gateway.local/v1/embeddings")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test12345"})
    @patch("requests.post")
    def test_dimensions_not_passed_to_older_models(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
            "model": "text-embedding-ada-002"
        }
        mock_post.return_value = mock_response

        embedder = OpenAIEmbedder(model_id="text-embedding-ada-002", dim=2)
        embedder.encode("test")

        kwargs = mock_post.call_args[1]
        # Older models do not accept the dimensions payload field
        self.assertNotIn("dimensions", kwargs["json"])

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test12345"})
    def test_engine_lazy_resolution(self):
        # Verify that we can instantiate EpochDB with "openai:*" model configuration without exceptions
        # (It shouldn't try to call the network until an actual remember/recall occurs)
        d = tempfile_dir = os.path.join(os.path.dirname(__file__), "..", "scratch", "test_openai_engine_db")
        os.makedirs(os.path.dirname(d), exist_ok=True)
        try:
            db = EpochDB(storage_dir=d, model="openai:text-embedding-3-small", dim=1536)
            self.assertEqual(db._model_name, "openai:text-embedding-3-small")
            # Lazy resolution check
            self.assertIsNone(db._embedder)
            embedder = db._get_embedder()
            self.assertIsInstance(embedder, OpenAIEmbedder)
            self.assertEqual(embedder.model_id, "text-embedding-3-small")
            self.assertEqual(embedder.dim, 1536)
            db.close()
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
