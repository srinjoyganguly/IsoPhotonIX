"""Validate the physical translation at the same API boundary as editor runs."""
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient

from backend.app import app
from backend.library_layout import build_design, place_component
from backend.study3d import parse_study
from backend.taper_study import example_taper


class TaperStudyTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.config = example_taper().model_dump()

    def test_editor_snapshot_leads_domain_and_source(self):
        original = build_design('taper', {'length_um': 3., 'input_width_um': .5,
            'output_width_um': 1.15, 'height_um': .3})['geometry']
        original['objects'][0]['visible'] = False
        config = {**self.config, 'geometry': original, 'resolution': 16}
        validated = self.client.post('/api/taper/validate', json=config)
        self.assertEqual(validated.status_code, 200, validated.text[:250])
        layout = validated.json()['physical_layout']
        self.assertEqual(layout['vertices'], [[p['x'], p['y']] for p in original['objects'][0]['vertices']])
        self.assertLess(-layout['domain']['x']/2 + config['pml_um'], layout['source_x'])
        self.assertLess(layout['source_x'], layout['x_in'])
        self.assertLess(layout['x_out'], layout['domain']['x']/2 - config['pml_um'])
        self.assertEqual(parse_study(config).model_dump()['geometry'], original)
        with patch('backend.app.capabilities', return_value={'targets': [
                {'solver': 'fdtdx', 'device': 'gpu', 'available': True}]}), patch('backend.app.manager.submit', return_value={'id':'sample'}) as submit:
            self.assertEqual(self.client.post('/api/taper/jobs', json={**config,'solver':'fdtdx','device':'gpu'}).status_code, 202)
            self.assertEqual(submit.call_args.args[0]['geometry'], original)
            self.assertEqual(submit.call_args.args[0]['device'], 'gpu')

    def test_ineligible_or_inconsistent_design_never_reaches_worker(self):
        base = self.config['geometry']
        invalid = []
        def altered(change):
            result = deepcopy(base)
            change(result)
            invalid.append(result)
        altered(lambda g: g['objects'][0]['vertices'][0].update(y=-.3))
        altered(lambda g: g['objects'][0].update(rotation_deg=15))
        altered(lambda g: g['components'][0]['ports'][0].update(width_um=.6))
        altered(lambda g: g['components'][0]['parameters'].update(height_um=.3))
        altered(lambda g: g['materials'][0].update(k=.1))
        altered(lambda g: g['materials'][0].update(model='tabulated-nk',wavelength_um=[1.4,1.7],n=[2.,2.],k=[0.,0.]))
        altered(lambda g: g['domain'].update(y=100.))
        invalid.extend([build_design('crossing')['geometry'], place_component(base, 'strip', fit_domain=True)['geometry']])
        with patch('backend.app.manager.submit') as submit:
            for geometry in invalid:
                with self.subTest(geometry=geometry['name']):
                    response = self.client.post('/api/taper/jobs', json={**self.config,'geometry':geometry})
                    self.assertEqual(response.status_code, 422)
            for override in ({'solver':'meep','device':'gpu'}, {'resolution':12}, {'solver':'fdtdx','device':None}):
                self.assertEqual(self.client.post('/api/taper/jobs', json={**self.config,**override}).status_code, 422)
            with patch('backend.app.capabilities',return_value={'targets':[]}):
                self.assertEqual(self.client.post('/api/taper/jobs',json={**self.config,'solver':'fdtdx','device':'gpu'}).status_code,409)
            submit.assert_not_called()

    def test_export_includes_exact_editor_geometry_and_worker_dependencies(self):
        response = self.client.post('/api/reproduce/study',json=self.config)
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            self.assertEqual(json.loads(bundle.read('config.json')),self.config)
            self.assertTrue({'study.py','study.ipynb','source/backend/taper_study.py',
                'source/backend/taper_geometry.py','source/backend/adapters/meep_taper.py',
                'source/backend/adapters/fdtdx_taper.py'} <= set(bundle.namelist()))
