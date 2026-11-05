"""Grid translation checks. Real device comparisons are recorded separately."""
import unittest

import numpy as np

from backend.adapters.fdtdx_taper import raster_layout, run
from backend.component_library import build_component
from backend.taper_study import TaperStudy


def profile(width_in, width_out, height=.22):
    geometry = build_component('taper', {
        'input_width_um': width_in, 'output_width_um': width_out, 'height_um': height,
    })['geometry']
    return TaperStudy(geometry=geometry, solver='fdtdx', device='cpu').layout()


class TaperMaterialGrid(unittest.TestCase):
    def test_native_blocks_tile_a_finite_trapezoid_and_both_leads(self):
        layout = raster_layout(profile(.4, 1.2), 20, .75)
        x, y, z = layout['centres']
        counts = np.zeros(layout['shape'], dtype=np.uint8)
        for start, size in layout['runs']:
            counts[tuple(slice(s, s+n) for s, n in zip(start, size))] += 1
        self.assertTrue(np.array_equal(counts, layout['occupied']))
        # End faces are at +/-3 µm. Each native block's cross-section must
        # follow the analytic sidewall between them through the full domain.
        widths = np.clip(.8 + x * .8 / 6, .4, 1.2)
        expected = ((abs(y[None, :, None]) <= widths[:, None, None]/2 + 1e-12) &
                    (abs(z[None, None, :]) <= .11 + 1e-12))
        np.testing.assert_array_equal(counts, expected)
        self.assertEqual(layout['source_shape'], (1, 8, 4))
        self.assertLess(layout['source_x'], -3)
        self.assertTrue((counts[0] == counts[x < -3]).all())
        self.assertTrue((counts[-1] == counts[x > 3]).all())

    def test_equal_width_limit_is_one_uniform_block_and_reverse_taper_narrows(self):
        straight = raster_layout(profile(.4, .4), 20, .75)
        self.assertEqual(len(straight['runs']), 1)
        taper = raster_layout(profile(1.2, .4), 20, .75)
        occupied = taper['occupied'].sum(axis=(1, 2))
        self.assertTrue((np.diff(occupied) <= 0).all())
        self.assertEqual(occupied[0], 24*4)
        self.assertEqual(occupied[-1], 8*4)

    def test_wrong_study_rejected_before_loading_jax(self):
        with self.assertRaisesRegex(ValueError, 'version 5'):
            run({'solver': 'fdtdx', 'study_type': 'waveguide-3d', 'schema_version': 2}, None)


if __name__ == '__main__':
    unittest.main()
