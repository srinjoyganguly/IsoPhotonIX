"""Validate the bounded material model and dispersive projection without running a device."""
import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('scipy'),'The ring worker needs SciPy.')
class RingMaterialTests(unittest.TestCase):
    def test_ade_fits_passive_mixed_cells_and_preserves_port_cross_sections(self):
        from scripts.calibrate_loaded_ring3d import geometry,permittivity,LIGHT
        from scripts.ring_ade_stage import build
        g=geometry('lumerical',12,.65,1.8,'exact-area')
        dt=.99/np.sqrt(3)/g['resolution']*1e-6/LIGHT
        a,ref,m=build(g,True,dt,permittivity)
        b,dev,n=build(g,False,dt,permittivity)
        self.assertGreaterEqual(a['inf'].min(),1.)
        self.assertGreaterEqual(a['c3'].min(),0.)
        self.assertLess(max(n['maximum_cell_epsilon_errors']),.01)
        self.assertEqual(m['held_out_indices'],[1,3])
        self.assertNotEqual(m['applied_c3_sha256'],n['applied_c3_sha256'])
        for px,py,pz in g['ports']:
            aperture=2*g['half_ap']
            section=(slice(None),slice(px,px+1),slice(py,py+aperture),slice(pz,pz+aperture))
            for key in ('inf','c3'):
                np.testing.assert_array_equal(a[key][section],b[key][section])
            np.testing.assert_array_equal(ref[section],dev[section])

    def test_projection_uses_material_basis_wavelengths(self):
        from scripts.calibrate_loaded_ring3d import WAVELENGTHS,directional_spectrum
        from scripts.ring_palik_dispersion import WAVELENGTHS as basis
        time=np.arange(20)/4
        trace=np.zeros((len(time),4,len(basis),2))
        # Each of five independent basis coefficients carries a known wavelength label.
        trace[0,:,:,:]=basis[None,:,None]
        raw={'time_um_c':time,'projection_wavelengths_um':basis,'projections':trace}
        cols=directional_spectrum(raw,1.)
        actual=WAVELENGTHS*np.exp(2j*np.pi*time[0]/WAVELENGTHS)/4
        np.testing.assert_allclose(cols[:,0,0],actual,rtol=1e-13)
