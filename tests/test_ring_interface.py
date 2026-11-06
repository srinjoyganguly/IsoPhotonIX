import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'Ring material quadrature requires the scientific worker.')
class RingInterface(unittest.TestCase):
    def test_disk_and_strip_areas_are_conserved_for_displaced_centers(self):
        from scripts.calibrate_loaded_ring3d import disk_fractions, lateral_fraction
        for mesh in (12,16,20,24):
            dx=1/mesh
            rad,width,bus=3.1,.4,3.6
            x=np.arange(-5*mesh,5*mesh)*dx+.23*dx
            y=np.arange(-5*mesh,5*mesh)*dx+.48*dx
            disk=disk_fractions(x,y,dx,rad)
            self.assertAlmostEqual(disk.sum()*dx**2,np.pi*rad**2,places=10)
            straight=lateral_fraction(x,y,dx,rad,width,bus,True,'exact-area')
            np.testing.assert_allclose(straight.sum(axis=1)*dx,2*width,atol=2e-14)
            device=lateral_fraction(x,y,dx,rad,width,bus,False,'exact-area')
            self.assertAlmostEqual((device-straight).sum()*dx**2,2*np.pi*rad*width,places=10)

    def test_partial_disk_cells_against_independent_quadrature(self):
        from scipy.integrate import quad
        from scripts.calibrate_loaded_ring3d import disk_fractions
        r,dx=3.1,1/16
        for x,y in [(0.,r),(r,0.),(2.2,-2.18),(-2.18,-2.2),(5.,1.)]:
            def height(a):
                h=np.sqrt(max(0., r*r-a*a))
                return max(0.,min(y+dx/2,h)-max(y-dx/2,-h))
            critical=[-r,r]
            for b in (y-dx/2,y+dx/2):
                if abs(b)<r:
                    t=np.sqrt(r*r-b*b)
                    critical.extend([-t,t])
            v=quad(height,x-dx/2,x+dx/2,points=[t for t in critical if x-dx/2<t<x+dx/2],
                   epsabs=1e-13)[0]/dx**2
            self.assertAlmostEqual(v,disk_fractions(np.array([x]),np.array([y]),dx,r)[0,0],places=10)

    def test_legacy_is_default_and_exact_preserves_separate_port_reference(self):
        from scripts.calibrate_loaded_ring3d import geometry,permittivity
        for vendor in ('lumerical','tidy3d'):
            g=geometry(vendor,16,.65,1.8)
            self.assertEqual(g['ring_interface'],'legacy')
            default=permittivity(g,False)
            explicit=permittivity(geometry(vendor,16,.65,1.8,'legacy'),False)
            np.testing.assert_array_equal(default,explicit)
            g=geometry(vendor,16,.65,1.8,'exact-area')
            reference,device=permittivity(g,True),permittivity(g,False)
            self.assertGreater(np.max(abs(device-default)),.05)
            for px,py,pz in g['ports']:
                section=(slice(None),px,slice(py,py+2*g['half_ap']),slice(pz,pz+2*g['half_ap']))
                np.testing.assert_array_equal(reference[section],device[section])
