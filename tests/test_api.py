import unittest
from unittest.mock import patch

try:
    import click
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

from app import app as flask_app


@unittest.skipUnless(HAS_CLICK, "click library is required for API tests")
class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = flask_app.test_client()

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {"status": "ok"})

    def test_predict_missing_text(self):
        resp = self.client.post("/predict", json={})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_predict_success(self):
        with patch("app.api.preprocess", lambda text: text + "_pp"), \
             patch("app.api.predict", lambda text, top_k, threshold: {"query": text, "predicted_intent": "i", "candidates": ["i"], "scores": [1.0]}):
            resp = self.client.post("/predict", json={"text": "test", "top_k": 3, "threshold": 0.6})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["query"], "test_pp")
        self.assertEqual(data["predicted_intent"], "i")

    def test_predict_internal_error(self):
        def raise_error(text, top_k, threshold):
            raise RuntimeError("fail")

        with patch("app.api.predict", raise_error):
            resp = self.client.post("/predict", json={"text": "test"})
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data.get("error"), "Internal server error")

    def test_build_index_success(self):
        with patch("app.api.build_index", lambda: ("idx", "lbl")):
            resp = self.client.post("/build_index")
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        self.assertEqual(data.get("index_path"), "idx")
        self.assertEqual(data.get("labels_path"), "lbl")

    def test_build_index_failure(self):
        def raise_error():
            raise RuntimeError("fail")

        with patch("app.api.build_index", raise_error):
            resp = self.client.post("/build_index")
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data.get("error"), "Falha ao gerar índice")