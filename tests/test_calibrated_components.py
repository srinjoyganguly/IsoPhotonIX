"""The preset, immutable source, job and offline replay contracts."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
from backend.app import app
from backend.calibrations import Calibration, create_bundle, result_zip
from backend.calibration_replay import verify_source


class CalibratedComponents(unittest.TestCase):
    def test_target_is_explicit_and_unsupported_routes_are_rejected(self):
        for values in [{'case':'ring'}, {'case':'grating','solver':'fdtdx','device':'cpu'},
                       {'case':'multimode','solver':'meep','device':'gpu'},
                       {'case':'coupler','solver':'meep','device':'cpu','output':'/tmp/path'},
                       {'case':'coupler','solver':'meep','device':'cpu','resolution':12},
                       {'case':'splitter','solver':'meep','device':'cpu','material_model':'yee-aligned'}]:
            with self.assertRaises(ValueError):
                Calibration(**values)
        model=Calibration(case='coupler',solver='fdtdx',device='gpu',material_model='yee-aligned',resolution=12)
        from backend.calibrations import specification
        self.assertEqual(model.device,'gpu')
        self.assertIn('yee-aligned',specification(model)[1])
        self.assertIn('12',specification(model)[1])

    def test_alignment_of_bundle_sources_and_reference_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=Calibration(case='multimode',solver='meep',device='cpu')
            bundle=create_bundle(config,root/'replay')
            manifest=verify_source(bundle)
            self.assertEqual(manifest['worker'],'scripts.calibrate_multimode')
            self.assertIn('source/scripts/calibrate_multimode.py',manifest['files'])
            (root/'calculation').mkdir()
            import numpy as np
            np.savez(root/'calculation/arrays.npz', values=np.array([1],dtype=int))
            (root/'calculation/report.json').write_text('{"passed": true}')
            result_zip(root,root/'test.zip')
            with zipfile.ZipFile(root/'test.zip') as archive:
                self.assertIn('reference/arrays.npz',archive.namelist())
                self.assertIn('calibration.ipynb',archive.namelist())
                self.assertIn('run.py',archive.namelist())
                self.assertEqual(json.loads(archive.read('config.json')),config.model_dump())
            (bundle/'source/scripts/calibrate_multimode.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'source or settings changed'):
                verify_source(bundle)

    def test_unavailable_gpu_request_never_submits_cpu(self):
        client=TestClient(app)
        with patch('backend.app.capabilities',return_value={'targets':[{'solver':'fdtdx','device':'gpu','available':False,'reason':'CUDA unavailable'}]}), patch('backend.app.manager.submit') as submit:
            r=client.post('/api/calibrations/jobs',json={'case':'coupler','solver':'fdtdx','device':'gpu'})
            self.assertEqual(r.status_code,409)
            submit.assert_not_called()

    def test_export_before_running_each_supported_case_and_target(self):
        client=TestClient(app)
        for preset in client.get('/api/calibrations').json():
            for route in preset['routes']:
                solver,device=route.split(':')
                config={'case':preset['case'],'solver':solver,'device':device}
                with self.subTest(config=config):
                    response=client.post('/api/reproduce/calibration',json=config)
                    self.assertEqual(response.status_code,200)
                    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                        manifest=json.loads(archive.read('manifest.json'))
                        self.assertEqual(json.loads(archive.read('config.json'))['device'],device)
                        self.assertIn('source/'+manifest['worker'].replace('.','/')+'.py',archive.namelist())
                        self.assertIn('calibration.ipynb',archive.namelist())
                        self.assertFalse(any(name.startswith('reference/') for name in archive.namelist()))

    def test_replay_preserves_unqualified_wavelengths(self):
        from backend.calibration_replay import compare
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            dirs=[Path(tmp)/name for name in ('reference','fresh')]
            for directory in dirs:
                directory.mkdir()
                np.savez(directory/'arrays.npz',samples=[.2,np.nan,.4])
                (directory/'report.json').write_text(json.dumps({'power':[.2,None,.4]}))
            self.assertTrue(compare(*dirs)['passed'])
            (dirs[1]/'report.json').write_text(json.dumps({'power':[.2,0,.4]}))
            with self.assertRaisesRegex(ValueError,'Displayed observable changed'):
                compare(*dirs)

    def test_grating_single_beam_snapshot_selects_the_vendor_source(self):
        from backend.calibrations import specification
        config=Calibration(case='grating',solver='meep',device='cpu',grating_scope='lumerical-project',resolution=40)
        self.assertEqual(specification(config)[1],['--beam-only','lumerical','--project-source','--resolution','40'])
        with self.assertRaises(ValueError):
            Calibration(case='ring',solver='meep',device='cpu',grating_scope='tidy3d-compact')

    def test_multimode_resume_rejects_other_mesh_and_corrupt_samples(self):
        from types import SimpleNamespace
        import numpy as np
        from scripts.calibrate_multimode import checkpointed_stage
        mp=SimpleNamespace(__version__='test')
        data=(np.ones((2,2,2,3),complex),np.ones((2,2,2,3)),np.ones((4,4)),(np.arange(4),np.arange(4),np.array([0])))
        settings=(48,.125,4.,1.,80.,False)
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.calibrate_multimode.measure_stage',return_value=data) as evolve:
            first=checkpointed_stage(mp,settings,0,1,True,Path(tmp))
            second=checkpointed_stage(mp,settings,0,1,True,Path(tmp))
            np.testing.assert_array_equal(first[0],second[0])
            evolve.assert_called_once()
            with self.assertRaisesRegex(ValueError,'Checkpoint'):
                checkpointed_stage(mp,(32,*settings[1:]),0,1,True,Path(tmp))
            next(Path(tmp).glob('*.npz')).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Checkpoint'):
                checkpointed_stage(mp,settings,0,1,True,Path(tmp))

    def test_catalog(self):
        props=TestClient(app).get('/api/calibrations').json()
        self.assertEqual(len(props),6)

    @unittest.skipUnless(importlib.util.find_spec('scipy'),'The independent Helmholtz solve requires SciPy.')
    def test_independent_straight_interface(self):
        import numpy as np
        from scripts.calibrate_multimode import interface_reference
        s,neff=interface_reference(1/1.55,0,1/20,7)
        np.testing.assert_allclose(abs(s)**2,np.roll(np.eye(4),2,axis=0),atol=1e-8)
        self.assertGreater(neff[1],1.44)
