import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from scripts import calibrate_finite_grating as grating
from scripts import lumerical_grating3d as ansys
from scripts.grating_angular_spectrum import air_angular_flux


class FiniteGrating3D(unittest.TestCase):
    def test_narrow_width_changes_teeth_but_preserves_incident_domain(self):
        wide,narrow=(grating.settings(x,12) for x in ('compact','compact-narrow'))
        self.assertEqual(grating.grid(wide)[0],grating.grid(narrow)[0])
        self.assertEqual(wide['waist'],narrow['waist'])
        _, _, (xs,ys,zs)=grating.grid(wide)
        row=(np.argmin(abs(xs-.2)),slice(None),np.argmin(abs(zs-.18)))
        and_wide, and_narrow = [grating.material(m,False)[row] > 8 for m in (wide,narrow)]
        self.assertLess(and_narrow.sum(),and_wide.sum())
        self.assertFalse(np.any(and_narrow & ~and_wide))

    def test_published_uniform_grating_residual_film_extends_beyond_teeth(self):
        m=grating.settings('tidy3d',12)
        shape,p,axes=grating.grid(m)
        self.assertGreaterEqual(p,6);self.assertEqual(len(shape),3)
        xs,ys,zs=axes
        idx=tuple(np.argmin(abs(a-v)) for a,v in zip(axes,(.2,5.9,.04)))
        idx2=(*idx[:2],np.argmin(abs(zs-.2)))
        mat=grating.material(m,False)
        self.assertGreater(mat[idx],8.)
        self.assertLess(mat[idx2],2.1)
        self.assertAlmostEqual(m['taper'],50.)

    def test_project_material_and_focusing_sector(self):
        m=grating.settings('lumerical',12)
        self.assertEqual(m['teeth'],38)
        xs=np.array([-.2,.2,.54]); ys=np.array([0.,7.9,18.])
        low,top=ansys.xy_coverage(m,xs,ys)
        self.assertEqual(tuple(low[:,0]),(1.,1.,1.))
        self.assertEqual(tuple(top[:,0]),(1.,0.,1.))
        self.assertFalse(np.any(top[:,-1]))
        r=json.loads((Path(__file__).resolve().parents[1]/'docs/verification/v012-grating-commercial-references.json').read_text())
        self.assertAlmostEqual(r['lumerical']['material_epsilon']['Si (Silicon) - Palik']['real'][2],m['n_si']**2)
        self.assertAlmostEqual(r['lumerical']['port_centers_um']['x'][1],m['output_x'])

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'The native fiber mode uses SciPy.')
    def test_lp01_boundary_and_confinement(self):
        m=grating.settings('lumerical',12)
        u,w,n=ansys.lp01(m['fiber_radius'],m['fiber_core_n'],m['fiber_clad_n'],1.55)
        self.assertTrue(m['fiber_clad_n']<n<m['fiber_core_n'])
        delta=1e-5
        f=ansys.fiber_radial_field(np.array([0.,1.-delta,1.,1.+delta,2.]),u,w)
        self.assertTrue(np.all(np.diff(f)<0.))
        self.assertLess(abs(f[1]+f[3]-2*f[2]),1e-7)
        self.assertLess(f[-1],.12)

    def test_angular_projection_recovers_tilted_plane_wave(self):
        # Choose the plane size to put kx exactly on an FFT bin.
        n=64; dx=1/16; wavelength=1.; sx=wavelength/(n*dx); cz=np.sqrt(1-sx*sx)
        phase=np.exp(2j*np.pi*np.arange(n)/n)[:,None]*np.ones((1,n))
        plane=np.zeros((1,6,n,n),complex)
        plane[0,1]=phase;plane[0,3]=-cz*phase;plane[0,5]=sx*phase
        up=air_angular_flux(plane,dx,[wavelength])
        down=air_angular_flux(plane,dx,[wavelength],upward=False)
        expected=.5*n*n*dx**2*cz
        np.testing.assert_allclose(up['power'].sum(),expected)
        self.assertLess(down['power'].sum(),1e-12)
        self.assertEqual(int(np.argmax(up['polar_bin_power'][0])),1)
        self.assertEqual(np.unravel_index(up['power'][0].argmax(),(n,n)),(1,0))

    def test_polar_power_and_fiber_return_have_consistent_normalization(self):
        plane=np.zeros((1,6,8,8),complex)
        plane[:,1]=1; plane[:,3]=1
        inc=-grating.normal_flux(plane,.2)
        np.testing.assert_allclose(inc,[.5*8*8*.2**2])
        reverse=plane.conj().copy()
        reverse[:,3:]*=-1
        s=ansys.reflected_fiber_amplitude(reverse,plane,inc,.2)
        np.testing.assert_allclose(s,[1.])
        np.testing.assert_allclose(ansys.reflected_fiber_amplitude(plane,plane,inc,.2),[0.])
