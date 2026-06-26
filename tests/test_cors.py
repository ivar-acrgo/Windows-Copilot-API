"""CORS preflight (OPTIONS) for browser and cross-origin API clients."""

import unittest

from fastapi.testclient import TestClient

from server.api import app


class CorsPreflightTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_options_chat_completions_returns_200(self):
        resp = self.client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("access-control-allow-origin"), "*")
        self.assertIn("POST", resp.headers.get("access-control-allow-methods", ""))

    def test_options_models_returns_200(self):
        resp = self.client.options(
            "/v1/models",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
