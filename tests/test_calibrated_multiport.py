"""Physical layout contracts shared by the solver adapters."""
import unittest

from backend.calibrated_geometry import multiport_model, straight_polygon


def occupied(point, polygon):
    x, y = point
    crossings = sum((a[1] > y) != (b[1] > y) and
                    x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]
                    for a,b in zip(polygon,polygon[1:]+polygon[:1]))
    return crossings % 2 == 1


class CalibratedLayouts(unittest.TestCase):
    def test_disjoint_ports_match_isolated_straight_guides_at_both_planes(self):
        for name,count in [('coupler',4),('splitter',3)]:
            m=multiport_model(name)
            self.assertEqual(len(m['ports']),count)
            for i,p in enumerate(m['ports']):
                ref=straight_polygon(m,i)
                for x in (p['x'],p['source_x']):
                    for dy in (-.95,-.26,-.24,0,.24,.26,.95):
                        y=p['y']+dy
                        self.assertEqual(any(occupied((x,y),poly) for poly in m['polygons']),occupied((x,y),ref))
                self.assertGreater(p['inward_sign']*(p['x']-p['source_x']),0.)

    def test_central_coupler_and_y_branch_topology(self):
        coupler=multiport_model('coupler')
        self.assertFalse(any(occupied((0,0),p) for p in coupler['polygons']))
        for y in (-.3125,.3125):
            self.assertTrue(any(occupied((0,y),p) for p in coupler['polygons']))
        shape=multiport_model('splitter')['polygons'][0]
        for x in (-4,-2): self.assertTrue(occupied((x,0),shape))
        self.assertFalse(occupied((3.,0.),shape))
        self.assertTrue(occupied((3.,1.25),shape))

    def test_pml_does_not_move_ports_or_interior(self):
        baseline=multiport_model('coupler')
        wide=multiport_model('coupler',1.,2.25)
        for key in ('polygons','ports','frequency_per_um'):
            self.assertEqual(baseline[key],wide[key])
        for a,b in zip(baseline['cell'],wide['cell']): self.assertAlmostEqual(a-1.5,b-2)
        for pml,aperture in ((.3,2.),(.75,3.)):
            with self.assertRaises(ValueError): multiport_model('coupler',pml,aperture)

    def test_coherent_passivity_and_complex_reciprocity_are_distinct_checks(self):
        import numpy as np
        from scripts.calibrate_multiport import matrix_diagnostics
        values=np.array([[[.7, .7j], [.7,.7j]]])
        power,singular,reciprocity=matrix_diagnostics(values,True)
        np.testing.assert_allclose(power,[[.98,.98]])
        np.testing.assert_allclose(singular,[1.4])
        np.testing.assert_allclose(reciprocity,[np.sqrt(.98)])
        self.assertIsNone(matrix_diagnostics(values[:,:,:1],False)[2])
        with self.assertRaises(ValueError):
            matrix_diagnostics(values[:,:,:1],True)
