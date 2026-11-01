import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app import app


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_example_and_validation(self):
        model = self.client.get("/api/example").json()
        self.assertEqual(self.client.post("/api/validate", json=model).status_code, 200)
        model.update(core_index=1, cladding_index=1.5)
        self.assertEqual(self.client.post("/api/validate", json=model).status_code, 422)

    def test_unavailable_gpu_is_rejected_without_starting_cpu(self):
        config = self.client.get("/api/example").json()
        config.update(solver="fdtdx", device="gpu")
        with patch("backend.app.capabilities", return_value={"targets": [{"solver": "fdtdx", "device": "gpu", "available": False, "reason": "CUDA not available"}]}), patch("backend.app.manager.submit") as submit:
            response = self.client.post("/api/jobs", json=config)
        self.assertEqual(response.status_code, 409)
        self.assertIn("CUDA", response.json()["detail"])
        submit.assert_not_called()

    def test_unknown_jobs_and_remote_origin(self):
        self.assertEqual(self.client.get("/api/jobs/not-a-run").status_code, 404)
        self.assertEqual(self.client.get("/api/health", headers={"Origin": "https://unrelated.example"}).status_code, 403)
