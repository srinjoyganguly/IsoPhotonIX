import unittest
from importlib.util import find_spec
import numpy as np
from scripts.reciprocal_ring_wall import OFFSETS,reciprocal_links


class ReciprocalRingWall(unittest.TestCase):
    def test_shared_edges_reproduce_uniform_tensor_and_exclude_wrap_at_open_faces(self):
        inv=np.zeros((9,4,5,2),np.float32)
        inv[[0,4,8]]=.5
        inv[[1,3]]=.18
        periodic,info=reciprocal_links(inv,.1,1,periodic=True)
        np.testing.assert_allclose(periodic,.18/4)
        self.assertEqual(info['links_limited'],0)
        open_links,_=reciprocal_links(inv,.1,1)
        for i,(sx,sy) in enumerate(OFFSETS):
            if sx: np.testing.assert_array_equal(open_links[i,-1],0)
            if sy: np.testing.assert_array_equal(open_links[i,:,0],0)

    @unittest.skipUnless(find_spec('scipy'),'Numerical worker tests use SciPy.')
    def test_adversarial_spatial_variation_is_symmetric_and_bounded(self):
        from scripts.validate_reciprocal_ring_wall import matrix
        rng=np.random.default_rng(237)
        shape=(5,6,2);lower=.1;upper=1.
        inv=np.zeros((9,*shape),np.float32)
        inv[[0,4,8]]=rng.uniform(lower,upper,(3,*shape))
        inv[[1,3]]=rng.uniform(-1,1,(2,*shape))
        links,info=reciprocal_links(inv,lower,upper,periodic=True)
        dense=matrix(inv,links).toarray()
        vals=np.linalg.eigvalsh(dense)
        self.assertGreater(info['links_limited'],0)
        np.testing.assert_array_equal(dense,dense.T)
        self.assertGreaterEqual(vals[0],lower-1e-7)
        self.assertLessEqual(vals[-1],upper+1e-7)

    @unittest.skipUnless(find_spec('scipy'),'Numerical worker tests use SciPy.')
    def test_widespread_passivity_and_severe_late_growth_reject_every_channel(self):
        from scripts.calibrate_loaded_ring3d import field_stability
        decay=np.stack([np.exp(-np.arange(400)/80)]*3,axis=1)
        healthy=field_stability(decay,np.ones(241),3200)
        self.assertTrue(healthy['usable_spectrum'])
        growth=decay.copy()
        growth[200:300]=5
        growth[300:]=20
        self.assertTrue(field_stability(growth,np.ones(241),3200)['usable_spectrum'])
        self.assertFalse(field_stability(growth,np.full(241,1.4),3200)['usable_spectrum'])


class FreshReciprocalAdmission(unittest.TestCase):
    @unittest.skipUnless(find_spec('scipy'),'Numerical worker tests use SciPy.')
    def test_live_device_free_memory_and_matching_profile_are_required(self):
        from pathlib import Path
        import json
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.reciprocal_ring_wall import memory_identity, fresh_admission
        from scripts.calibrate_loaded_ring3d import geometry
        g=geometry('lumerical',24,.65,1.8,'reciprocal-wall')
        base=json.loads(json.dumps(geometry('lumerical',16,.65,1.8,'reciprocal-wall')))
        target=SimpleNamespace(platform='gpu',local_hardware_id=0,device_kind='fixture')
        provenance=dict(hardware=dict(model='fixture',driver_build='test',cuda_driver_api=123),
                        installed_numerics={'test':{'sha':'abc'}},versions={'fdtdx':'tested'},python_abi='test',
                        jax_configuration={},source_files={'worker':'abc'})
        row=dict(physical_model=base,cells=int(np.prod(base['shape'])),peak_delta_used_for_estimate_mib=1400,
                 sampled_peak_global_mib=3400,baseline_idle_used_mib=2000,allocator_peak_bytes_in_use=1000,
                 performance=dict(platform='gpu',steps=200,grid_shape=base['shape'],device_kind='fixture'),
                 numerical_source_hashes=provenance['source_files'],memory_identity=memory_identity(provenance))
        budget=dict(vendor='lumerical',ring_interface='reciprocal-wall',baseline_mesh=16,candidate_mesh=24,
                    total_mib=16384,roles={role:dict(row,role=role) for role in ('reference','device')})
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.json';path.write_text(json.dumps(budget))
            with patch.dict('os.environ',{'PHOTONICS_RING_MEMORY_PLAN':str(path)}):
                with patch('scripts.grating_cuda_identity.available_mib',return_value=14000):
                    result=fresh_admission(Path(tmp),g,target,provenance)
                self.assertEqual(result['available_at_launch_mib'],14000)
                with patch('scripts.grating_cuda_identity.available_mib',return_value=100):
                    with self.assertRaises(MemoryError):
                        fresh_admission(Path(tmp),g,target,provenance)
                self.assertIsNone(fresh_admission(Path(tmp),base,target,provenance))
                different=dict(provenance,source_files={'worker':'modified'})
                with self.assertRaises(ValueError):
                    fresh_admission(Path(tmp),g,target,different)
