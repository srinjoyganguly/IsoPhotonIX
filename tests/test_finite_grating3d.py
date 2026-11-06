import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

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

    def test_guided_emission_uses_separate_guide_and_beam_power(self):
        from scripts.grating_emission import observables
        field=np.zeros((5,6,8,8),dtype=complex)
        field[:,1]=1;field[:,3]=1
        reverse=field.copy();reverse[:,3:]*=-1
        r={'upper':field*0,'output_amplitudes':np.tile([2.+0j,0j],(5,1))}
        d={'upper':.5*reverse,'output_amplitudes':np.tile([2.+0j,.5j],(5,1))}
        measured=observables(r,d,{'upper':field},5)
        # Incoming guide carries 4 units; upper receive plane's unit wave carries 1.28.
        np.testing.assert_allclose(measured['incident_guide'],4.)
        np.testing.assert_allclose(measured['incident_beam'],1.28)
        np.testing.assert_allclose(abs(measured['gaussian_channel'])**2,.08)
        np.testing.assert_allclose(measured['upgoing_fraction'],.08)
        np.testing.assert_allclose(abs(measured['guide_reflection'])**2,.0625)

    def test_cache_never_reuses_another_gpu_or_scipy_version(self):
        hardware={'name':'NVIDIA test device','ordinal':0,'driver_version':12090,
                  'compute_capability':'8.6','per_cache_device_sha256':'gpu-a'}
        versions={'scipy':'1'}
        def version(name): return versions.get(name,'test')
        def stage(cmd, check, env):
            target=Path(cmd[cmd.index('--output')+1])
            np.savez(target/'arrays.npz',measured=np.array([1.]))
            info={'cuda_identity':hardware.copy(),'native_worker_sha256':grating.file_hash(grating.__file__)}
            (target/'receipt.json').write_text(json.dumps(info))
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ',{'PHOTONICS_GPU_LOCK':str(Path(temp)/'gpu.lock')}), patch.object(grating.cuda_identity,'fingerprint',side_effect=lambda salt:hardware.copy()), patch('importlib.metadata.version',side_effect=version), patch('subprocess.run',side_effect=stage) as native:
            cache=Path(temp); model=grating.settings('compact',12)
            first,_=grating.stage_archive(model,'reference',cache)
            reused,_=grating.stage_archive(model,'reference',cache)
            self.assertEqual(first,reused);self.assertEqual(native.call_count,1)
            hardware['per_cache_device_sha256']='gpu-b'
            other,_=grating.stage_archive(model,'reference',cache)
            self.assertNotEqual(first,other);self.assertEqual(native.call_count,2)
            versions['scipy']='2'
            third,_=grating.stage_archive(model,'reference',cache)
            self.assertNotEqual(third,other);self.assertEqual(native.call_count,3)
            report=json.loads((third/'receipt.json').read_text())
            report['cuda_identity']['per_cache_device_sha256']='gpu-a'
            (third/'receipt.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'device'):
                grating.stage_archive(model,'reference',cache)

    def test_raw_gpu_uuid_stays_private_and_salt_is_bound_to_cache(self):
        from scripts.grating_cuda_identity import cache_salt,fingerprint
        raw=b'0123456789abcdef'
        with tempfile.TemporaryDirectory() as temp, patch('scripts.grating_cuda_identity.driver_device',return_value=(raw,{'name':'test'})):
            first=cache_salt(Path(temp)/'a')
            self.assertEqual(first,cache_salt(Path(temp)/'a'))
            self.assertNotEqual(fingerprint(first),fingerprint(cache_salt(Path(temp)/'b')))
            self.assertNotIn(raw.decode(),json.dumps(fingerprint(first)))

    def test_memory_budget_combines_observations_and_an_actual_cuda_plan(self):
        from scripts.grating_memory_budget import MIB,estimate
        base={'gpu':'device','geometry':'tidy3d','resolution':12,'cells':1_000_000,
              'cpu_field_allocation_verified':True,'executed_fdtd_steps':0,'compiled_live_bytes':100*MIB}
        actual={'gpu':'device','geometry':'tidy3d','resolution':12,'executed_fdtd_steps':1,
                'global_used_mib':700,'allocator_bytes':{},'actual_compiled_live_bytes':100*MIB}
        refined=dict(base,resolution=16,cells=2_000_000,compiled_live_bytes=240*MIB)
        sample=estimate(base,actual,refined,idle_used_mib=500,free_before_mib=1800)
        self.assertEqual(sample['baseline_native_peak_mib'],200.)
        self.assertEqual(sample['estimated_runtime_mib'],480.)
        self.assertTrue(sample['fits_envelope'])
        self.assertFalse(estimate(base,actual,refined,500,1400)['fits_envelope'])
        with self.assertRaisesRegex(ValueError,'disagree'):
            estimate(base,dict(actual,actual_compiled_live_bytes=110*MIB),refined,500,1800)
        with self.assertRaisesRegex(ValueError,'host'):
            estimate(base,actual,dict(refined,cpu_field_allocation_verified=False),500,1800)

    def test_full_geometry_free_memory_preflight(self):
        import json
        budget=json.loads((grating.ROOT/'docs/verification/v012-grating-memory-budget.json').read_text())
        free=budget['measured_free_mib']
        for name in ('lumerical','tidy3d'):
            row=grating.memory_admission(grating.settings(name,14),free)
            self.assertLess(row['required_free_with_reserve_mib'],free)
            with self.assertRaisesRegex(ValueError,'requires'):
                grating.memory_admission(grating.settings(name,16),free)
        with self.assertRaises(ValueError):
            grating.settings('compact',14)
