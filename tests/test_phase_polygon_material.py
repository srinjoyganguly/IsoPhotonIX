import unittest
import numpy as np

from backend.calibrated_geometry import multiport_model, straight_polygon
from scripts.calibrate_fdtdx_multiport import grid
from scripts.phase_polygon_material import extruded_yee, polygon_fraction


class PolygonMaterialTests(unittest.TestCase):
    def test_oblique_triangle_and_polygon_reversal(self):
        triangle = [(0, 0), (2, 0), (0, 2)]
        expected = np.array([[1, .5], [.5, 0]])
        for shape in (triangle, triangle[::-1]):
            np.testing.assert_allclose(polygon_fraction([shape], [.5,1.5], [.5,1.5], 1), expected)


    def test_concave_fork_union_and_mirror(self):
        for case in ('coupler','splitter'):
            model = multiport_model(case)
            _, _, (x,y,_), _ = grid(model,16)
            fraction = polygon_fraction(model['polygons'],x,y,1/16)
            np.testing.assert_allclose(fraction, fraction[:,::-1],atol=1e-11)
            # Equal rail width and opposite centerlines do not overlap in the coupler.
            if case == 'coupler':
                np.testing.assert_allclose(fraction.sum(axis=1)/16, 1., atol=1e-11)
            else:
                np.testing.assert_allclose(fraction[x < -2].sum(axis=1)/16, .5, atol=1e-11)
                np.testing.assert_allclose(fraction[x > 2].sum(axis=1)/16, 1., atol=1e-11)


    def test_straight_vertical_and_sidewall_mixing(self):
        model = multiport_model('splitter')
        _, _, axes, _ = grid(model,16)
        eps = extruded_yee(model,16,axes,[straight_polygon(model,0)])
        clad, core = model['cladding_index']**2, model['core_index']**2
        assert eps.dtype == np.float32 and eps.shape == (3,224,112,64)
        assert eps.min() >= clad-1e-6 and eps.max() <= core+1e-6
        # Face through upper surface z=+0.125 for Ex: half air-cladding, half core.
        np.testing.assert_allclose(eps[0,112,56,34], (clad+core)/2, rtol=1e-6)
        # Ez has y-face at the outer straight sidewall. This field is tangential there.
        np.testing.assert_allclose(eps[2,112,60,32], (clad+core)/2, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
