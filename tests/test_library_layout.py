"""Design/API/export contracts; no synthetic electromagnetic evidence."""
import copy
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
from backend.component_library import get_catalog
from backend.geometry_draft import validate_geometry
from backend.library_layout import build_design, place_component
from backend.reproduce import geometry_bundle
from backend.study3d import example3d


class LibraryLayouts(unittest.TestCase):
    def test_every_catalog_family_builds_valid_geometry_and_unique_membership(self):
        client = TestClient(app)
        catalog = client.get('/api/library').json()
        self.assertEqual(len(catalog['components']), 27)
        with patch('backend.app.manager.submit') as submit:
            for entry in catalog['components']:
                with self.subTest(family=entry['id']):
                    response = client.post('/api/library/build', json={'component_id': entry['id']})
                    self.assertEqual(response.status_code, 200, response.text[:300])
                    result = response.json(); geometry = result['geometry']
                    self.assertEqual(geometry, validate_geometry(geometry))
                    self.assertEqual(result['component']['family'], entry['id'])
                    self.assertEqual(set(result['component']['object_ids']), {o['id'] for o in geometry['objects']})
                    self.assertEqual(result['component']['parameters'], entry['defaults'])
                    self.assertEqual(entry['simulation_support'], 'bounded-taper-fields' if entry['id'] == 'taper' else 'bounded-layout-fields')
            submit.assert_not_called()

    def test_placement_preserves_existing_design_and_requires_explicit_domain_growth(self):
        old = example3d().geometry.model_dump(by_alias=True)
        original = copy.deepcopy(old)
        placement = {'x': 20., 'y': 1., 'z': 0., 'rotation_deg': 30.}
        with self.assertRaises(ValueError): place_component(old, 'taper', placement=placement)
        self.assertEqual(old, original)
        added = place_component(old, 'taper', placement=placement, fit_domain=True)['geometry']
        self.assertEqual(added['schema_version'], 2)
        self.assertGreater(added['domain']['x'], old['domain']['x'])
        self.assertEqual(added['objects'][:len(old['objects'])], old['objects'])
        self.assertEqual(old, original)
        self.assertEqual(len({o['id'] for o in added['objects']}), len(added['objects']))

    def test_replacing_one_instance_preserves_neighbors_and_records_new_parameters(self):
        first = build_design('strip')['geometry']
        first['domain'] = {'x': 40., 'y': 20., 'z': 4.}
        together = place_component(first, 'ring_bus', placement={'x': 10., 'y': 0., 'z': 0., 'rotation_deg': 0.})['geometry']
        before = copy.deepcopy(together)
        selected = together['components'][0]
        changed = place_component(together, 'strip', parameters={'width_um': .75},
            replace_component_id=selected['id'])['geometry']
        self.assertEqual(together, before)
        unaffected = [o for o in before['objects'] if o['id'] not in selected['object_ids']]
        self.assertEqual(changed['objects'][-len(unaffected):], unaffected)
        self.assertEqual(len(changed['components']), 2)
        self.assertEqual(changed['components'][-1]['id'], selected['id'])
        self.assertEqual(changed['components'][-1]['parameters']['width_um'], .75)
        self.assertTrue(set(selected['object_ids']).isdisjoint(o['id'] for o in changed['objects']))
        with self.assertRaises(ValueError): place_component(together, 'strip', replace_component_id='stale-id')

    def test_custom_spectral_material_is_preserved_and_never_reduced_for_a_solver(self):
        material = {'id': 'custom-film', 'name': 'Measured example film', 'model': 'tabulated-nk',
                    'reference_wavelength_um': 1.55, 'wavelength_um': [1.5, 1.6],
                    'n': [2., 2.2], 'k': [.05, .15], 'source': 'Test fixture; not a measured dataset.', 'notes': 'Passive design data.'}
        result = build_design('taper', material_id='custom-film', material=material)
        geometry = result['geometry']
        self.assertEqual(geometry['materials'], [material])
        for obj in geometry['objects']:
            self.assertAlmostEqual(obj['refractive_index'], 2.1)
            self.assertEqual(obj['material_id'], 'custom-film')
        second = place_component(geometry, 'strip', material_id='custom-film', fit_domain=True)['geometry']
        self.assertEqual(second['materials'], [material])
        old_study = example3d().model_dump(by_alias=True); old_study['geometry'] = geometry
        self.assertEqual(TestClient(app).post('/api/validate3d', json=old_study).status_code, 422)

    def test_invalid_requests_and_remote_origins_cannot_build_or_start_computation(self):
        client = TestClient(app)
        for data in ({'component_id': 'unknown'}, {'component_id': 'strip', 'parameters': {'unknown': 4}},
                     {'component_id': 'strip', 'parameters': {'width_um': True}},
                     {'component_id': 'strip', 'placement': {'x': '2'}},
                     {'component_id': 'strip', 'material_id': 'unknown'}):
            self.assertEqual(client.post('/api/library/build', json=data).status_code, 422)
        self.assertEqual(client.post('/api/library/build', json={'component_id': 'strip'},
            headers={'Origin': 'https://untrusted.example'}).status_code, 403)

    def test_api_partial_regeneration_preserves_unedited_instance_settings(self):
        original = build_design('strip', {'length_um': 12},
            {'x': 10., 'y': 2., 'z': .2, 'rotation_deg': 30.}, 'silicon-reference')['geometry']
        instance = original['components'][0]
        response = TestClient(app).post('/api/library/place', json={
            'geometry': original, 'component_id': 'strip', 'replace_component_id': instance['id'],
            'parameters': {'width_um': .7}, 'placement': {'x': 9.}, 'fit_domain': True})
        self.assertEqual(response.status_code, 200, response.text[:300])
        changed = response.json()['component']
        self.assertEqual(changed['parameters'], {**instance['parameters'], 'width_um': .7})
        self.assertEqual(changed['placement'], {**instance['placement'], 'x': 9.})
        self.assertEqual(changed['material_id'], instance['material_id'])
        self.assertEqual(changed['id'], instance['id'])

    def test_exported_polygon_materials_recipes_and_generator_work_outside_checkout(self):
        geometry = build_design('nanobeam_cavity', placement={'y': 2., 'rotation_deg': 30.})['geometry']
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with zipfile.ZipFile(io.BytesIO(geometry_bundle(geometry))) as archive:
                archive.extractall(folder)
                self.assertEqual(json.loads(archive.read('geometry.json')), geometry)
                self.assertIn('design_source/component_library.py', archive.namelist())
                notebook = json.loads(archive.read('geometry.ipynb'))
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code': compile(''.join(cell['source']), cell['id'], 'exec')
            spec = importlib.util.spec_from_file_location('exported_geometry_runtime', folder/'workbench_runtime.py')
            runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
            runtime.check_bundle()
            self.assertEqual(runtime.validate_geometry(geometry), geometry)
            instance = geometry['components'][0]
            changed = runtime.regenerate_component(geometry, instance['id'], parameters={'height_um': .3},
                placement={'x': .1}, fit_domain=True)
            self.assertTrue(all(o['size']['z'] == .3 for o in changed['objects']))
            self.assertEqual(changed['components'][0]['placement'], {**instance['placement'], 'x': .1})
            self.assertEqual(geometry['components'][0]['parameters']['height_um'], .22)
            source = folder/'design_source/component_library.py'; source.write_text(source.read_text()+'\n# changed\n')
            with self.assertRaises(ValueError): runtime.check_bundle()


if __name__ == '__main__':
    unittest.main()
