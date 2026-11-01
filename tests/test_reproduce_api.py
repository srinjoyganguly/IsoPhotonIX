"""HTTP export contracts using explicit synthetic fixtures, not solver evidence."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.model import WaveguideConfig
from backend.ports import example_ports
from backend.ports3d import example_ports3d
from backend.study3d import example3d


class ReproduceAPI(unittest.TestCase):
    def test_all_executable_editor_formats_export_exact_settings_without_starting_workers(self):
        configs = [WaveguideConfig().model_dump(), example3d().model_dump(by_alias=True),
                   example_ports().model_dump(), example_ports3d().model_dump(by_alias=True)]
        client = TestClient(app)
        with patch('backend.app.manager.submit') as submit:
            for config in configs:
                with self.subTest(schema=config['schema_version']):
                    response = client.post('/api/reproduce/study', json=config)
                    self.assertEqual(response.status_code, 200, response.text[:200] if response.status_code != 200 else '')
                    self.assertEqual(response.headers['content-type'], 'application/zip')
                    self.assertIn('python-notebook.zip', response.headers['content-disposition'])
                    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                        self.assertEqual(json.loads(archive.read('config.json')), config)
                        self.assertIn('study.py', archive.namelist()); self.assertIn('study.ipynb', archive.namelist())
                        self.assertNotIn('reference/config.json', archive.namelist())
            submit.assert_not_called()

    def test_geometry_export_preserves_unsupported_simulation_drafts(self):
        geometry = example3d().geometry.model_dump(by_alias=True)
        geometry['objects'][1].update(kind='ring', radius=.8, inner_radius=.4, rotation_deg=20.)
        response = TestClient(app).post('/api/reproduce/geometry', json=geometry)
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(json.loads(archive.read('geometry.json')), geometry)
            self.assertIn('geometry.py', archive.namelist()); self.assertIn('geometry.ipynb', archive.namelist())
            self.assertNotIn('study.py', archive.namelist())
            self.assertIn(b'design geometry only', archive.read('README.md'))

    def test_invalid_target_schema_geometry_and_remote_origin_are_rejected(self):
        client = TestClient(app)
        for config in (dict(WaveguideConfig().model_dump(), device='gpu'),
                       dict(WaveguideConfig().model_dump(), schema_version=999), {'schema_version': 4}):
            self.assertEqual(client.post('/api/reproduce/study', json=config).status_code, 422)
        geometry = example3d().geometry.model_dump(by_alias=True)
        geometry['objects'][1]['kind'] = 'arbitrary-script'
        self.assertEqual(client.post('/api/reproduce/geometry', json=geometry).status_code, 422)
        self.assertEqual(client.post('/api/reproduce/study', json=WaveguideConfig().model_dump(),
                                     headers={'Origin': 'https://untrusted.example'}).status_code, 403)

    def test_completed_run_export_retains_reference_bytes_and_rejects_snapshot_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / 'saved'; output.mkdir()
            config = WaveguideConfig().model_dump()
            metadata = {'config': config, 'solver': 'meep', 'actual_device': 'CPU', 'versions': {}}
            job = {'id': 'saved', 'status': 'completed', 'config': config, 'logs': 'private fixture log'}
            (output / 'config.json').write_text(json.dumps(config))
            (output / 'metadata.json').write_text(json.dumps(metadata))
            np.savez(output / 'arrays.npz', field=np.ones((2, 2)), epsilon=np.ones((2, 2)),
                     x_um=np.arange(2.), y_um=np.arange(2.))
            (output / 'worker.log').write_text('private fixture log')
            with patch('backend.app.manager') as manager:
                manager.directory = root; manager.get.return_value = job
                client = TestClient(app)
                response = client.get('/api/jobs/saved/reproduce')
                self.assertEqual(response.status_code, 200)
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    self.assertEqual(archive.read('reference/arrays.npz'), (output / 'arrays.npz').read_bytes())
                    self.assertEqual(json.loads(archive.read('reference/config.json')), config)
                    self.assertFalse(any('worker.log' in name for name in archive.namelist()))
                    manifest = json.loads(archive.read('bundle-manifest.json'))
                    for name, record in manifest['files'].items():
                        self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), record['sha256'])
                job['status'] = 'running'
                self.assertEqual(client.get('/api/jobs/saved/reproduce').status_code, 409)
                job['status'] = 'completed'
                changed = dict(config, core_index=2.1)
                (output / 'config.json').write_text(json.dumps(changed))
                self.assertEqual(client.get('/api/jobs/saved/reproduce').status_code, 409)
                (output / 'metadata.json').write_text(json.dumps(dict(metadata, config=changed)))
                self.assertEqual(client.get('/api/jobs/saved/reproduce').status_code, 409)


if __name__ == '__main__':
    unittest.main()
