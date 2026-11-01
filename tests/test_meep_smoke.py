"""Real solver checks. Run with the isolated Meep interpreter (no mocks)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from backend.adapters.meep_adapter import run


@unittest.skipUnless(importlib.util.find_spec("meep"), "Requires official PyMeep")
class MeepSmokeTest(unittest.TestCase):
    def test_real_waveguide_and_parameter_response(self):
        import numpy as np
        config = {
            "schema_version": 1, "wavelength_um": 1.55, "core_width_um": 0.6,
            "core_index": 2.0, "cladding_index": 1.0, "length_um": 10.0,
            "height_um": 5.0, "resolution": 20, "duration_fs": 80.0,
            "pml_um": 1.0, "solver": "meep", "device": "cpu",
        }
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first"
            first = run(config, first_path)
            with np.load(first_path / "arrays.npz") as saved:
                field = saved["field"].copy()
                epsilon = saved["epsilon"].copy()
                self.assertTrue(np.isfinite(field).all())
                self.assertGreater(float(np.max(np.abs(field))), 1e-6)
                self.assertGreater(float(epsilon.max()), float(epsilon.min()))
                self.assertEqual(field.shape, (len(saved["y_um"]), len(saved["x_um"])))
                self.assertTrue(np.all(np.diff(saved["times_fs"]) >= 0))
                self.assertGreater(saved["frames"].shape[0], 5)
                # A real propagated signal must have reached the far half.
                self.assertGreater(float(np.max(np.abs(field[:, saved["x_um"] > 0]))), 1e-6)
            self.assertEqual(first["device"], "cpu")
            self.assertAlmostEqual(first["actual_duration_fs"], 80, delta=0.2)
            changed = dict(config, core_width_um=0.9)
            second_path = Path(directory) / "wider"
            run(changed, second_path)
            with np.load(second_path / "arrays.npz") as saved:
                self.assertFalse(np.array_equal(epsilon, saved["epsilon"]))
                self.assertFalse(np.allclose(field, saved["field"]))


class MeepDeviceGuardTest(unittest.TestCase):
    def test_explicit_gpu_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CPU only"):
            run({"device": "gpu"}, Path("/unused"))


if __name__ == "__main__":
    unittest.main()
