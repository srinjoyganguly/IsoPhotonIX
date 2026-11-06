"""Finite two-bus ring geometry, native data and broadband modal reconstruction."""
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np

HAS_SCIPY = find_spec('scipy') is not None


@unittest.skipUnless(HAS_SCIPY, 'Loaded-ring numerical tests run in the solver environment with SciPy.')
class LoadedRing(unittest.TestCase):
    def test_worker_calls_stage_identity_before_calculating(self):
        from scripts import calibrate_loaded_ring3d as worker
        class ReachedIdentity(Exception):
            pass
        runtime=(object(),object(),object(),object())
        with TemporaryDirectory() as tmp, \
             patch('sys.argv',['ring','--vendor','lumerical','--device','gpu','--output',str(Path(tmp)/'run')]), \
             patch.object(worker,'_ports_runtime',return_value=runtime), \
             patch.object(worker.importlib.metadata,'version',return_value='tested'), \
             patch.object(worker,'stage_fingerprint',side_effect=ReachedIdentity) as identify:
            with self.assertRaises(ReachedIdentity):
                worker.main()
        self.assertEqual(identify.call_args.args[:2],(worker.ROOT,runtime))

    def test_both_finite_stacks_and_measured_leads(self):
        from scripts.calibrate_loaded_ring3d import geometry, permittivity, OXIDE
        for name,top in (('lumerical',1.),('tidy3d',OXIDE)):
            g=geometry(name,16,.65,1.8)
            ref=permittivity(g,True)
            device=permittivity(g,False)
            nx,ny,nz=g['shape']
            h=g['half_ap']
            for x,y,z in g['ports']:
                np.testing.assert_array_equal(ref[:,x,y:y+2*h,z:z+2*h],device[:,x,y:y+2*h,z:z+2*h])
            np.testing.assert_allclose(device[:,:,0,-1],top**2,rtol=1e-6)
            np.testing.assert_allclose(device[:,:,0,0],OXIDE**2,rtol=1e-6)
            self.assertGreater(device[1,nx//2,ny//2+round(g['radius_um']*16),nz//2],8.)
            self.assertLess(ref[1,nx//2,ny//2+round(g['radius_um']*16),nz//2],3.)
            self.assertLess(device[1,nx//2,ny//2,nz//2],3.)
            self.assertTrue(np.all(device>=ref-1e-5))
            self.assertEqual(round(16*g['edge_gap_um'],3),1.6)

    def test_time_projection_retains_band_dependent_mode_gauge(self):
        from scripts.calibrate_loaded_ring3d import directional_spectrum,BASIS_WAVELENGTHS,WAVELENGTHS
        ts=np.arange(50)*.1
        w=1.55
        signal=np.sin(2*np.pi*ts/w)
        basis=BASIS_WAVELENGTHS-.5
        traces=signal[:,None,None,None]*basis[None,None,:,None]*np.ones((1,4,1,2))
        result=directional_spectrum({'projections':traces,'time_um_c':ts},1)
        target=(np.exp(2j*np.pi*ts[None,:]/WAVELENGTHS[:,None]) @ signal)*.1*(WAVELENGTHS-.5)
        np.testing.assert_allclose(result[:,2,1],target,rtol=1e-12,atol=1e-12)

    def test_spacing_cannot_jump_an_unqualified_resonance(self):
        from scripts.calibrate_loaded_ring3d import WAVELENGTHS,qualified_spacing
        mask=np.ones(len(WAVELENGTHS),bool)
        roots=[{'wavelength_um':WAVELENGTHS[i]} for i in (40,80,120)]
        mask[80]=False
        self.assertIsNone(qualified_spacing(roots,mask))
        mask[80]=True
        mask[70]=False
        self.assertEqual(qualified_spacing(roots,mask),tuple(roots[1:]))

    def test_decay_and_equal_bus_fit_have_the_same_known_loaded_q(self):
        from scripts.measure_loaded_ring_losses import ring_decay, fit_symmetric_peak
        wave=np.arange(1.53,1.571,.0002)
        pole,q,internal=1.55,520.,.17
        shape=1/(1+(2*q*(pole/wave-1))**2)
        power=np.zeros((len(wave),4))
        power[:,1]=1-(1-internal**2)*shape
        power[:,2]=(1-internal)**2*shape
        fit=fit_symmetric_peak(wave,power,pole,.026)
        self.assertAlmostEqual(fit['loaded_q'],q,places=4)
        self.assertAlmostEqual(fit['internal_fraction'],internal,places=6)
        ts=np.arange(0,3000,.2)
        decay=np.pi/(pole*q)
        field=np.exp(-decay*ts[:,None])*np.cos(2*np.pi*ts[:,None]/pole+[0,.3,1.1])
        measure=ring_decay(ts,field,pole,.026)
        self.assertTrue(measure['qualified'])
        self.assertLess(abs(measure['loaded_q']/q-1),.002)
        too_short=ring_decay(ts[:1200],field[:1200],pole,.026)
        self.assertFalse(too_short['qualified'])


if __name__=='__main__':
    unittest.main()
