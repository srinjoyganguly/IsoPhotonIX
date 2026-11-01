"""Actual finite-height modal scattering checks, separate from accuracy claims."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.adapters.meep_ports3d_adapter import run


def configuration(kind='gap'):
    # Keep the test independent of FastAPI/Pydantic in the numerical runtime.
    path = Path(__file__).resolve().parents[1]/'examples'/f'ports3d-{kind}.json'
    return json.loads(path.read_text())


class MeepModalPortTests(unittest.TestCase):
    def test_unavailable_device_and_wrong_study_are_rejected_before_import(self):
        for config in ({'solver':'meep','device':'gpu'}, {'solver':'fdtdx','device':'cpu'},
                       {'solver':'meep','device':'cpu','schema_version':3,'study_type':'waveguide-ports-2d'}):
            with self.assertRaises(ValueError):
                run(config,Path('unused'))

    @unittest.skipUnless(importlib.util.find_spec('meep'), 'Requires the official Meep/MPB numerical runtime')
    def test_real_straight_gap_reciprocity_and_invalid_ringdown(self):
        with tempfile.TemporaryDirectory() as directory:
            results = {}
            retained = os.environ.get('PHOTONICS_MEEP_PORTS3D_EVIDENCE_DIR')
            def execute_or_read(config, name):
                if retained:
                    output=Path(retained)/name
                    metadata=json.loads((output/'metadata.json').read_text())
                    self.assertEqual(metadata['config'],config)
                else:
                    output=Path(directory)/name
                    metadata=run(config,output)
                return output,metadata
            for kind in ('straight','gap'):
                output,metadata=execute_or_read(configuration(kind),kind+'-final')
                self.assertTrue(metadata['sparameters_valid'])
                self.assertEqual(metadata['field_capture_stage'],'device_forward')
                self.assertEqual(len(metadata['settlement']),4)
                self.assertTrue(all(item['settled'] for item in metadata['settlement'].values()))
                with np.load(output/'sparameters.npz',allow_pickle=False) as arrays:
                    s=arrays['s'];results[kind]=s.copy()
                    self.assertEqual(s.shape,(5,2,2))
                    self.assertTrue(arrays['valid_bin'].all())
                    self.assertTrue(arrays['mode_valid'].all())
                    np.testing.assert_allclose(s,arrays['outgoing_amplitude']/arrays['incident_amplitude'][:,None,:],rtol=1e-12,atol=1e-12)
                    self.assertTrue(np.all(np.real(arrays['mode_neff'])>1))
                    self.assertLess(np.max(abs(s[:,0,1]-s[:,1,0])),.0002)
                    # Equal reflection from this mirror-symmetric geometry
                    # catches an inappropriate opposite-k Yee-grid projection.
                    self.assertLess(np.max(abs(s[:,0,0]-s[:,1,1])),.0002)
                    self.assertGreater(np.max(np.abs(np.imag(s[:,1,0]))),.1)
                with np.load(output/'arrays.npz',allow_pickle=False) as arrays:
                    self.assertEqual(arrays['fields'].shape,(3,len(arrays['z_um']),len(arrays['y_um']),len(arrays['x_um'])))
                    self.assertTrue(np.isfinite(arrays['fields']).all())
                    self.assertTrue(np.isfinite(arrays['epsilon']).all())
                    ix,iy,iz=(np.argmin(abs(arrays[axis+'_um'])) for axis in 'xyz')
                    self.assertAlmostEqual(arrays['epsilon'][iz,iy,ix],4. if kind=='straight' else 1.,places=5)
                for name in ('reference-forward','device-forward','reference-reverse','device-reverse'):
                    with np.load(output/(name+'-monitors.npz'),allow_pickle=False) as arrays:
                        self.assertTrue(np.iscomplexobj(arrays['port1_alpha']))
                        self.assertTrue(np.all(np.abs(arrays['port1_ey_anchor'])>0))
                        self.assertTrue((arrays['port1_signed_mode_neff'][:,0]>1).all())
                        self.assertTrue((arrays['port1_signed_mode_neff'][:,1]<-1).all())
            self.assertLess(np.max(abs(results['straight'][:,0,0])),1e-5)
            self.assertLess(np.max(abs(abs(results['straight'][:,1,0])**2-1)),.005)
            self.assertGreater(np.max(abs(results['gap']-results['straight'])),.05)
            invalid=configuration();invalid['max_after_sources_fs']=5.
            output,metadata=execute_or_read(invalid,'gap-timeout')
            self.assertFalse(metadata['sparameters_valid'])
            with np.load(output/'sparameters.npz',allow_pickle=False) as arrays:
                self.assertFalse(arrays['valid_bin'].any())
                self.assertTrue(np.isfinite(arrays['incident_amplitude']).all())
                self.assertTrue(np.isfinite(arrays['outgoing_amplitude']).all())


if __name__=='__main__':
    unittest.main()
