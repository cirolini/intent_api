import os
import importlib
import unittest


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Backup existing environment variables
        self._old_model_name = os.environ.pop("MODEL_NAME", None)
        self._old_data_dir = os.environ.pop("DATA_DIR", None)
        self._old_threshold = os.environ.pop("THRESHOLD", None)

    def tearDown(self):
        # Restore environment variables
        if self._old_model_name is not None:
            os.environ["MODEL_NAME"] = self._old_model_name
        if self._old_data_dir is not None:
            os.environ["DATA_DIR"] = self._old_data_dir
        if self._old_threshold is not None:
            os.environ["THRESHOLD"] = self._old_threshold
        importlib.reload(importlib.import_module("app.config"))

    def test_default_config(self):
        config = importlib.reload(importlib.import_module("app.config"))
        self.assertEqual(config.MODEL_NAME, "all-MiniLM-L6-v2")
        self.assertEqual(config.DATA_DIR, "faiss_indices")
        self.assertEqual(config.DEFAULT_THRESHOLD, 0.7)

    def test_env_config(self):
        os.environ["MODEL_NAME"] = "test-model"
        os.environ["DATA_DIR"] = "test-data"
        os.environ["THRESHOLD"] = "0.5"
        config = importlib.reload(importlib.import_module("app.config"))
        self.assertEqual(config.MODEL_NAME, "test-model")
        self.assertEqual(config.DATA_DIR, "test-data")
        self.assertEqual(config.DEFAULT_THRESHOLD, 0.5)