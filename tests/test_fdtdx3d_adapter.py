"""Pure placement and Yee-sampling contract tests; real solves have separate evidence."""
import copy
import unittest

import numpy as np

from backend.adapters.fdtdx3d_adapter import cell_center_fields, grid_layout


def study():
    def box(ident, size, position, index):
        return {"id": ident, "kind": "box", "size": dict(zip("xyz", size)),
                "position": dict(zip("xyz", position)), "rotation_deg": 0,
                "refractive_index": index, "visible": True}
    return {"schema_version": 2, "study_type": "waveguide-3d", "resolution": 16,
            "pml_um": .75, "geometry": {"domain": {"x": 8, "y": 4, "z": 4},
            "objects": [box("core", (8, .75, .5), (0, 0, 0), 2),
                        box("substrate", (8, 4, 1.75), (0, 0, -1.125), 1.44)]}}


class Fdtdx3DPlacementTests(unittest.TestCase):
    def test_finite_height_source_and_touching_substrate(self):
        layout = grid_layout(study())
        self.assertEqual(layout["shape"], [128, 64, 64])
        self.assertEqual(layout["placement"]["core"], {"start": [0, 26, 28], "shape": [128, 12, 8]})
        self.assertEqual(layout["placement"]["substrate"], {"start": [0, 0, 0], "shape": [128, 64, 28]})
        self.assertEqual(layout["source_shape"], [1, 12, 8])
        self.assertEqual(layout["source_start"], [20, 26, 28])
        self.assertEqual(layout["pml_cells"], 12)

    def test_fractional_dimensions_preserve_touching_interface_and_ignore_visibility(self):
        config = study()
        core, substrate = config["geometry"]["objects"]
        core["size"]["z"] = .4
        core["visible"] = False
        substrate["size"]["z"] = 1.8
        substrate["position"]["z"] = -1.1
        layout = grid_layout(config)
        self.assertEqual(layout["placement"]["substrate"]["shape"][2], layout["placement"]["core"]["start"][2])
        self.assertEqual(layout["placement"]["core"]["shape"][2], 6)

    def test_unsupported_geometry_cannot_be_silently_projected(self):
        for change in [lambda c: c["geometry"]["objects"][0].update(kind="ring"),
                       lambda c: c["geometry"]["objects"][0].update(rotation_deg=45),
                       lambda c: c["geometry"]["objects"][0]["position"].update(y=.2),
                       lambda c: c["geometry"]["objects"][1]["position"].update(z=-1),
                       lambda c: c["geometry"]["objects"].append(copy.deepcopy(c["geometry"]["objects"][0]))]:
            config = study(); change(config)
            with self.assertRaises(ValueError):
                grid_layout(config)


class YeeSamplingTests(unittest.TestCase):
    def test_linear_analytic_fields_recover_common_cell_centre_coordinates(self):
        z, y, x = np.meshgrid(np.arange(5), np.arange(6), np.arange(7), indexing="ij")
        # Known affine functions sampled at each component's physical Yee offset.
        raw = np.stack([2 * (x + .5) + 3 * y - z,
                        -x + 4 * (y + .5) + 2 * z,
                        5 * x - y + 3 * (z + .5)]).astype(np.float32)
        centered = cell_center_fields(raw)
        cx, cy, cz = x[:-1, :-1, :-1] + .5, y[:-1, :-1, :-1] + .5, z[:-1, :-1, :-1] + .5
        expected = np.stack([2 * cx + 3 * cy - cz, -cx + 4 * cy + 2 * cz, 5 * cx - cy + 3 * cz])
        np.testing.assert_array_equal(centered, expected)
        self.assertEqual(centered.shape, (3, 4, 5, 6))

    def test_no_extrapolation_or_input_mutation(self):
        raw = np.arange(3 * 2 * 2 * 2, dtype=np.float32).reshape(3, 2, 2, 2)
        original = raw.copy()
        result = cell_center_fields(raw)
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(result.shape, (3, 1, 1, 1))
        self.assertTrue(np.isfinite(result).all())
        with self.assertRaises(ValueError):
            cell_center_fields(raw[:, :1])


if __name__ == "__main__":
    unittest.main()
