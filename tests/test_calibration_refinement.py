import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from backend.calibration_replay import diagnostic_ring, ring_rejection
from backend.calibrations import Calibration, create_bundle, grating_meshes
from scripts.build_grating_meshes import collect, BASE


class CalibrationRefinements(unittest.TestCase):
    def test_exposed_grating_meshes_have_source_and_field_evidence(self):
        catalogue=collect(BASE)
        self.assertEqual(catalogue,json.loads((BASE/'v014-grating-meshes.json').read_text()))
        self.assertFalse(grating_meshes('tidy3d','full','direct-yee'))
        for row in catalogue['qualified']:
            self.assertIn(row['resolution'],grating_meshes(row['geometry'],row['domain'],row['material']))
            config=Calibration(case='grating-3d',solver='fdtdx',device='gpu',geometry_3d=row['geometry'],
                grating_domain=row['domain'],grating_material_model=row['material'],resolution=row['resolution'])
            with TemporaryDirectory() as tmp:
                program=Path(tmp)/'program'
                create_bundle(config,program)
                arguments=json.loads((program/'manifest.json').read_text())['arguments']
                self.assertEqual('--grating-domain' in arguments,row['domain']=='mirror-y')
                self.assertTrue((program/'source/scripts/grating_mirror.py').is_file())
                if row.get('memory_profile'):
                    self.assertTrue((program/'source/docs/verification'/row['memory_profile']).is_file())
        with self.assertRaises(ValueError):
            Calibration(case='grating-3d',solver='fdtdx',device='gpu',geometry_3d='lumerical',
                        resolution=12,grating_domain='mirror-y',grating_material_model='volume-average')
        with self.assertRaises(ValueError):
            Calibration(case='coupler',solver='fdtdx',device='cpu',material_model='polygon-area')
        with self.assertRaises(ValueError):
            Calibration(case='ring-3d',solver='fdtdx',device='gpu',resolution=12,ring_interface='wall-tensor')
        with self.assertRaisesRegex(ValueError,'stability'):
            Calibration(case='ring-3d',solver='fdtdx',device='gpu',resolution=24,ring_interface='wall-tensor')

    def test_unstable_record_cannot_be_promoted_by_one_nominal_bin(self):
        proof=json.loads((BASE/'v014-ring-wall-stability.json').read_text())
        native=json.loads((BASE/proof['source_receipt']).read_text())
        self.assertEqual(native['valid_bins'],1)
        self.assertEqual(ring_rejection(native),proof['reason'])
        self.assertIsNone(ring_rejection({**native,'arrays_sha256':'another measured array'}))
        self.assertFalse(diagnostic_ring({**native,'arrays_file':'arrays.npz'},BASE))

    def test_diagnostic_export_requires_real_complete_ring_arrays_and_consistent_mask(self):
        import hashlib
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            np.savez(folder/'arrays.npz',valid_bins=[True,False],s=np.zeros((2,4),complex))
            digest=hashlib.sha256((folder/'arrays.npz').read_bytes()).hexdigest()
            report=dict(case_id='loaded-ring-3d',passed=False,measurement_complete=True,arrays_file='arrays.npz',
                        arrays_sha256=digest,valid_bins=1,total_bins=2)
            self.assertTrue(diagnostic_ring(report,folder))
            for change in ({'arrays_sha256':'changed'},{'valid_bins':2},{'measurement_complete':False},{'passed':True}):
                self.assertFalse(diagnostic_ring(report|change,folder))
