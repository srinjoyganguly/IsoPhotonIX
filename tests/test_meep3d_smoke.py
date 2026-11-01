"""Real finite-height Meep checks; use the isolated PyMeep interpreter."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

from backend.adapters.meep3d_adapter import run


def example_config():
    def box(object_id, size, position, index):
        return {"id": object_id, "name": object_id.title(), "kind": "box", "visible": True,
                "size": dict(zip("xyz", size)), "position": dict(zip("xyz", position)),
                "rotation_deg": 0, "radius": 1, "inner_radius": .6, "refractive_index": index}
    return {"schema_version": 2, "study_type": "waveguide-3d", "solver": "meep", "device": "cpu",
            "wavelength_um": 1.55, "resolution": 16, "duration_fs": 60, "pml_um": .75, "cladding_index": 1,
            "geometry": {"schema": "photonics-workbench-geometry", "schema_version": 1, "units": "um",
                         "mode": "draft-3d", "name": "Finite-height guide", "domain": {"x": 8, "y": 4, "z": 4},
                         "objects": [box("substrate", (8, 4, 1.75), (0, 0, -1.125), 1.44),
                                     box("core", (8, .75, .5), (0, 0, 0), 2)]}}


class Meep3DGuardTests(unittest.TestCase):
    def test_gpu_is_rejected_before_import(self):
        with self.assertRaisesRegex(ValueError, "CPU only"):
            run({"device": "gpu"}, Path("/unused"))


@unittest.skipUnless(importlib.util.find_spec("meep"), "Requires official PyMeep")
class Meep3DNumericalTests(unittest.TestCase):
    def test_full_volume_materials_propagation_and_thickness_response(self):
        import numpy as np
        config = example_config()
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "default"
            metadata = run(config, first)
            with np.load(first / "arrays.npz") as arrays:
                field, fields, epsilon = (arrays[key].copy() for key in ("field", "fields", "epsilon"))
                x, y, z = (arrays[f"{axis}_um"] for axis in "xyz")
                self.assertEqual(field.shape, (len(z), len(y), len(x)))
                self.assertEqual(fields.shape, (3, *field.shape))
                self.assertTrue(np.isfinite(fields).all())
                self.assertTrue(np.array_equal(field, fields[1]))
                self.assertGreater(float(np.max(np.abs(field[:, :, x > 0]))), 1e-6)
                self.assertGreater(float(np.max(np.ptp(field, axis=0))), 1e-6)
                at = lambda xyz: epsilon[tuple(int(np.argmin(abs(axis - value))) for axis, value in zip((z, y, x), xyz[::-1]))]
                self.assertAlmostEqual(float(at((0, 0, 0))), 4, places=4)
                self.assertAlmostEqual(float(at((0, 0, -.8))), 1.44**2, places=4)
                self.assertAlmostEqual(float(at((0, 0, .8))), 1, places=4)
                self.assertAlmostEqual(float(at((0, .8, 0))), 1, places=4)
            self.assertEqual(metadata["dimensions"], "3D")
            self.assertAlmostEqual(metadata["actual_duration_fs"], 60, delta=.2)
            changed = copy.deepcopy(config)
            changed["geometry"]["objects"][1]["size"]["z"] = .75
            changed["geometry"]["objects"][0]["size"]["z"] = 1.625
            changed["geometry"]["objects"][0]["position"]["z"] = -1.1875
            second = Path(folder) / "thicker"
            run(changed, second)
            with np.load(second / "arrays.npz") as arrays:
                self.assertFalse(np.array_equal(epsilon, arrays["epsilon"]))
                self.assertFalse(np.allclose(field, arrays["field"]))


if __name__ == "__main__":
    unittest.main()
