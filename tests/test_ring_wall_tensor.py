"""Physical sidewall limits and staggered tensor checks independent of GPU evolution."""
import unittest
from importlib.util import find_spec

import numpy as np
from scripts.ring_wall_tensor import in_plane_inverse, wall_inverse


class WallTensor(unittest.TestCase):
    def test_half_filled_wall_has_harmonic_normal_and_arithmetic_tangent(self):
        f, silicon, oxide, bg = .5, 12., 2., 1.
        expected_normal = f/silicon+(1-f)/bg
        expected_tangent = 1/(f*silicon+(1-f)*bg)
        for angle in (0., np.pi/4, np.pi/2, -np.pi/4):
            n = np.array([np.cos(angle), np.sin(angle)])
            t = np.array([-n[1], n[0]])
            xx, xy, yy = in_plane_inverse(f, *n, 1., 0., silicon, oxide, bg)
            matrix = np.array([[xx, xy], [xy, yy]])
            np.testing.assert_allclose(matrix @ n, expected_normal * n, atol=1e-14)
            np.testing.assert_allclose(matrix @ t, expected_tangent * t, atol=1e-14)
        for fraction, index in ((0., bg), (1., silicon)):
            np.testing.assert_allclose(in_plane_inverse(fraction,.6,.8,1.,0.,silicon,oxide,bg), [1/index,0,1/index])
        np.testing.assert_allclose(in_plane_inverse(f,.6,.8,0.,1.,silicon,oxide,bg),[1/oxide,0,1/oxide])

    @unittest.skipUnless(find_spec('scipy'), 'Geometry and mode worker need SciPy')
    def test_off_diagonals_required_at_curved_walls_and_zero_on_ports(self):
        from scripts.calibrate_loaded_ring3d import geometry, disk_fractions, lateral_fraction, permittivity
        from scripts.probe_ring_wall_mixing import plane
        for vendor, mesh in [('lumerical',16),('lumerical',20),('tidy3d',16)]:
            g=geometry(vendor,mesh,.65,1.8,'wall-tensor')
            args=(disk_fractions,lateral_fraction,permittivity)
            device=wall_inverse(g,False,*args)
            reference=wall_inverse(g,True,*args)
            self.assertFalse(np.any(reference[[1,3]]))
            self.assertFalse(np.any(device[[2,5,6,7]]))
            for component in (1,3):
                self.assertGreater(device[component].max(),.01)
                self.assertLess(device[component].min(),-.01)
            np.testing.assert_array_equal(device[:,g['source_x']],reference[:,g['source_x']])
            for x,y,z in g['ports']:
                np.testing.assert_array_equal(device[:,x],reference[:,x])
            self.assertTrue(np.all(device[[0,4,8]]>0))
            self.assertTrue(np.all(device[[0,4,8]]<=1.000001))
            if vendor=='lumerical':
                expected,_=plane(dict(g,ring_interface='exact-area'),'normal-wall')
                x,y,z=g['ports'][0]; h=2*g['half_ap']
                np.testing.assert_allclose(1/reference[[0,4,8],x:x+1,y:y+h,z:z+h], expected, rtol=3e-7)


if __name__=='__main__':
    unittest.main()
