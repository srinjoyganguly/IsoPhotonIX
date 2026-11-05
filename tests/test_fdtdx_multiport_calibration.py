"""Check multiport calibration arithmetic and detector geometry, without a solver."""
import copy
import unittest

import numpy as np

from backend.calibrated_geometry import multiport_model
from scripts.calibrate_fdtdx_multiport import grid, normalize, raster, reference_cache_key


class MultiportCalibrationTests(unittest.TestCase):
    def test_port_apertures_and_each_reference_match_the_actual_launch_guide(self):
        for case in ('coupler','splitter'):
            for resolution in (12,16,20):
                model = multiport_model(case)
                shape, pml, _, ports = grid(model,resolution)
                circuit = raster(model,resolution)
                self.assertEqual(circuit.shape,shape)
                np.testing.assert_array_equal(circuit,circuit[:,::-1,:])
                for i,port in enumerate(ports):
                    ix,y,z = port['start']
                    _,ny,nz = port['shape']
                    self.assertGreaterEqual(y,pml); self.assertLessEqual(y+ny,shape[1]-pml)
                    self.assertAlmostEqual(port['actual_x'],port['x'])
                    straight = raster(model,resolution,i)
                    for x in (ix,port['source_ix']):
                        part = np.s_[x,y:y+ny,z:z+nz]
                        np.testing.assert_array_equal(straight[part],circuit[part])
                        self.assertTrue(straight[part].any())

    def test_all_columns_have_outward_local_sign_and_only_reflection_subtracts_reference(self):
        model=multiport_model('splitter')
        nf,n=len(model['frequency_per_um']),len(model['ports'])
        desired=np.arange(1,n*n+1).reshape(n,n)*(.02+.017j)
        rows=[]
        for i,p in enumerate(model['ports']):
            inc=np.array([1+.1j,1-.3j,.6+.2j])*(i+1)
            into=0 if p['inward_sign']==1 else 1
            ref=np.zeros((2,nf,2),complex)
            dev=np.zeros((n,nf,2),complex)
            ref[0,:,into]=inc
            ref[0,:,1-into]=.07j*inc
            ref[1,:,into]=-.98j*inc
            for j,q in enumerate(model['ports']):
                outward=1 if q['inward_sign']==1 else 0
                dev[j,:,outward]=desired[j,i]*inc
            dev[i,:,1-into]+=ref[0,:,1-into]
            record={'source_profile_sha256':'x','mode_neff':{p['name']:[2.5]*nf},'settled':True}
            rows.append((ref,dev,copy.deepcopy(record),copy.deepcopy(record)))
        matrix,_,transmit,settled=normalize(model,rows)
        np.testing.assert_allclose(matrix,np.broadcast_to(desired,matrix.shape))
        np.testing.assert_allclose(transmit,-.98j)
        self.assertTrue(settled.all())
        rows[0][3]['settled']=False
        self.assertFalse(normalize(model,rows)[3][0])
        rows[0][3]['source_profile_sha256']='different'
        with self.assertRaisesRegex(ValueError,'Launch modes'):
            normalize(model,rows)
        rows[0][3]['source_profile_sha256']='x'
        rows[0][0][0,1,:]=0
        with self.assertRaisesRegex(ValueError,'incident amplitude'):
            normalize(model,rows)

    def test_only_physically_identical_reference_planes_across_cases_share_a_key(self):
        def plan(case,res=16):
            return {'model':multiport_model(case),'device':'gpu','actual_device':'cuda:0','device_name':'RTX',
                    'resolution':res,'after_source_cap_um_over_c':80.,'numerical_sha256':'v1','versions':{'fdtdx':'0.6.2'}}
        coupler,splitter=plan('coupler'),plan('splitter')
        self.assertEqual(reference_cache_key(coupler,2), reference_cache_key(splitter,1))
        self.assertEqual(reference_cache_key(coupler,3), reference_cache_key(splitter,2))
        self.assertNotEqual(reference_cache_key(coupler,0), reference_cache_key(coupler,1))
        self.assertNotEqual(reference_cache_key(coupler,2), reference_cache_key(plan('splitter',20),1))
        self.assertNotEqual(reference_cache_key(coupler,2), reference_cache_key(dict(splitter,device='cpu'),1))

    def test_port_sampling_rejects_invalid_aperture(self):
        with self.assertRaises(ValueError):
            multiport_model('coupler',aperture_y_um=3.)
