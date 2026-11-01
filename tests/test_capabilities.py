import unittest
from unittest.mock import patch
from backend.capabilities import capabilities


class RecommendationTruthTests(unittest.TestCase):
    def test_missing_fdtdx_is_not_called_available(self):
        def fake_probe(target):
            solver, device = target
            return {"solver": solver, "device": device, "available": solver == "meep", "reason": "not installed"}
        with patch("backend.capabilities._probe", side_effect=fake_probe), patch("backend.capabilities._cache", None):
            result = capabilities(refresh=True)
        self.assertIn("FDTDX is currently unavailable", result["recommendation"]["reason"])
        self.assertNotIn("available alternative", result["recommendation"]["reason"])

    def test_no_ready_target_means_no_recommendation(self):
        with patch("backend.capabilities._probe", side_effect=lambda t: {"solver": t[0], "device": t[1], "available": False}), patch("backend.capabilities._cache", None):
            self.assertIsNone(capabilities(refresh=True)["recommendation"])
