import unittest
from unittest.mock import patch
from backend.capabilities import capabilities


class RecommendationTruthTests(unittest.TestCase):
    def test_missing_fdtdx_is_not_called_available(self):
        def fake_probe(target):
            solver, device = target
            return {"solver": solver, "device": device, "available": solver == "meep", "reason": "not installed"}
        with patch("backend.capabilities._probe", side_effect=fake_probe), patch("backend.capabilities._cache", None), patch("backend.capabilities._target_cache", {}):
            result = capabilities(refresh=True)
        self.assertIn("FDTDX is currently unavailable", result["recommendation"]["reason"])
        self.assertNotIn("available alternative", result["recommendation"]["reason"])

    def test_no_ready_target_means_no_recommendation(self):
        with patch("backend.capabilities._probe", side_effect=lambda t: {"solver": t[0], "device": t[1], "available": False}), patch("backend.capabilities._cache", None), patch("backend.capabilities._target_cache", {}):
            self.assertIsNone(capabilities(refresh=True)["recommendation"])


    def test_submitting_one_target_probes_only_that_target(self):
        row={"solver":"meep","device":"cpu","available":True}
        with patch("backend.capabilities._target_cache",{}),patch("backend.capabilities._probe",return_value=row) as probe:
            for _ in range(2):
                self.assertEqual(capabilities(target=("meep","cpu"))["targets"],[row])
            probe.assert_called_once_with(("meep","cpu"))
