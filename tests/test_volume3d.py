import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import io

import numpy as np
from fastapi.testclient import TestClient
from backend.app import app
from backend.passport import build_passport
from backend.study3d import example3d
from backend.volume import validate_volume, volume_slice, stored_volume


def fixture():
    # An asymmetric synthetic fixture checks indexing only, never solver physics.
    z, y, x = np.indices((3, 4, 5))
    ey = 1. + x + 10*y + 100*z
    return dict(field=ey, fields=np.stack([ey+1000, ey, -ey]), epsilon=np.ones_like(ey)*2,
                x_um=np.arange(5.)-.5, y_um=np.arange(4.)-1.5, z_um=np.arange(3.)-1)


class Volume3DTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / 'run'; self.output.mkdir()
        self.data = fixture()
        np.savez_compressed(self.output/'arrays.npz', **self.data)
        self.config = example3d().model_dump()
        self.metadata = dict(config=self.config, solver='meep', actual_device='CPU', field_component='Ey', dimensions='3D')
        for filename, value in [('config.json', self.config), ('metadata.json', self.metadata)]:
            (self.output/filename).write_text(json.dumps(value))
        self.job = dict(id='run', status='completed', config=self.config, result=self.metadata, logs='fixture only')

    def tearDown(self):
        self.temp.cleanup()

    def test_every_plane_component_and_coordinate_matches_saved_samples(self):
        volume = stored_volume(self.output)
        for component, ci in [('Ex', 0), ('Ey', 1), ('Ez', 2)]:
            expected = {'xy': self.data['fields'][ci, 1], 'xz': self.data['fields'][ci, :, 1], 'yz': self.data['fields'][ci, :, :, 1]}
            for plane in expected:
                sample = volume_slice(volume, plane, 1, component)
                np.testing.assert_array_equal(sample['field'], expected[plane])
                self.assertEqual(np.shape(sample['epsilon']), expected[plane].shape)
                self.assertEqual(sample['coordinate_um'], volume[f"{sample['normal_axis']}_um"][1])
        self.assertEqual(volume_slice(volume)['coordinate_um'], 0)
        with self.assertRaises(IndexError): volume_slice(volume, 'xy', -1)
        with self.assertRaises(IndexError): volume_slice(volume, 'xy', 3)
        with self.assertRaises(ValueError): volume_slice(volume, 'yx', 0)
        with self.assertRaises(ValueError): volume_slice(volume, 'xy', 0, 'Hx')

    def test_corrupt_or_mismatched_numerical_volumes_are_rejected(self):
        for key, value in [('fields', np.zeros((2, 3, 4, 5))), ('field', np.zeros((3, 4, 5))), ('epsilon', np.ones((3, 4, 6))), ('x_um', np.zeros(5))]:
            data = fixture(); data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_volume(data)
        data = fixture(); data['fields'][0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError): validate_volume(data)
        data = fixture(); data['fields'] = data['fields'].astype(complex); data['field'] = data['field'].astype(complex)
        with self.assertRaisesRegex(ValueError, 'real numeric'): validate_volume(data)

    def test_complex_corrupt_snapshot_returns_a_visible_error_not_json_failure(self):
        data = fixture(); data['fields'] = data['fields'].astype(complex); data['field'] = data['field'].astype(complex)
        np.savez_compressed(self.output/'arrays.npz', **data)
        with patch('backend.app.manager') as manager:
            manager.directory = Path(self.temp.name); manager.get.return_value = self.job
            client = TestClient(app)
            self.assertEqual(client.get('/api/jobs/run/slice').status_code, 409)
            self.assertEqual(client.get('/api/jobs/run/export').status_code, 409)

    def test_http_volume_and_export_keep_3d_semantics_and_exact_config(self):
        client = TestClient(app)
        with patch('backend.app.manager') as manager:
            manager.directory = Path(self.temp.name); manager.get.return_value = self.job
            response = client.get('/api/jobs/run/slice?plane=xz&index=2&component=Ez')
            self.assertEqual(response.status_code, 200)
            np.testing.assert_array_equal(response.json()['field'], self.data['fields'][2, :, 2])
            self.assertEqual(client.get('/api/jobs/run/volume').json()['shape'], [3, 4, 5])
            self.assertEqual(client.get('/api/jobs/run/field').status_code, 409)
            self.assertEqual(client.get('/api/jobs/run/slice?index=100').status_code, 404)
            self.assertEqual(client.get('/api/jobs/run/slice?component=Hx').status_code, 422)
            export = client.get('/api/jobs/run/export')
            self.assertEqual(export.status_code, 200)
            with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
                self.assertEqual(json.loads(archive.read('config.json')), self.config)
                self.assertIn(b'fields[component,z,y,x]', archive.read('README.txt'))
                self.assertIn(b'final-time electric fields', archive.read('passport.html'))
            self.job['status'] = 'running'
            self.assertEqual(client.get('/api/jobs/run/volume').status_code, 409)
            self.assertEqual(client.get('/api/jobs/run/slice').status_code, 409)

    def test_3d_validation_and_unavailable_gpu_never_launch_a_worker(self):
        client = TestClient(app)
        config = self.config
        self.assertEqual(client.post('/api/validate3d', json=config).status_code, 200)
        config['solver'] = 'fdtdx'; config['device'] = 'gpu'
        with patch('backend.app.capabilities', return_value={'targets': [{'solver':'fdtdx','device':'gpu','available':False,'reason':'CUDA unavailable'}]}), patch('backend.app.manager.submit') as submit:
            self.assertEqual(client.post('/api/jobs3d', json=config).status_code, 409)
            submit.assert_not_called()
        config['geometry']['objects'][1]['kind'] = 'ring'
        with patch('backend.app.manager.submit') as submit:
            self.assertEqual(client.post('/api/jobs3d', json=config).status_code, 422)
            submit.assert_not_called()
