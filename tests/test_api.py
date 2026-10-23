"""
Unit Tests for Flask API Endpoints and Page Routing
"""

import unittest
from app import app


class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "online")

    def test_page_routes(self):
        routes = ["/", "/dashboard", "/dataset", "/preprocessing", "/rnn", "/lstm", "/analyzer", "/comparison", "/dependency", "/evaluation", "/documentation"]
        for r in routes:
            res = self.client.get(r)
            self.assertEqual(res.status_code, 200, f"Route {r} failed with status {res.status_code}")

    def test_preprocessing_endpoint(self):
        payload = {"text": "This film was not bad at all!", "max_length": 50}
        res = self.client.post("/api/preprocess", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("not", data["tokens"])
        self.assertEqual(len(data["padded_sequence"]), 50)

    def test_models_status_endpoint(self):
        res = self.client.get("/api/models/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("rnn", data)
        self.assertIn("lstm", data)


if __name__ == "__main__":
    unittest.main()
