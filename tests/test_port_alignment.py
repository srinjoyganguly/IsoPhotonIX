"""Port alignment must preserve each solid, its recipe and physical layout order."""
import copy
import unittest

from fastapi.testclient import TestClient

from backend.app import app
from backend.library_layout import align_ports, build_design, place_component


class PortAlignmentTests(unittest.TestCase):
    @staticmethod
    def pair():
        base = build_design('strip', {'length_um': 2., 'height_um': .35},
                            {'rotation_deg': 30., 'x': -.8, 'z': .35})['geometry']
        return place_component(base, 'directional_coupler',
                               {'length_um': 2., 'height_um': .35, 'gap_um': .25},
                               {'x': .5, 'y': 1.3, 'z': -.2, 'rotation_deg': 135.},
                               fit_domain=True)['geometry']

    def test_alignment_is_rigid_and_atomic_and_preserves_the_recipe(self):
        scene = self.pair()
        before = copy.deepcopy(scene)
        fixed, moving = scene['components']
        aligned = align_ports(scene, moving['id'], 'lower_in', fixed['id'], 'out', True)
        self.assertEqual(scene, before)
        self.assertEqual(aligned['components'][0], fixed)
        self.assertEqual(aligned['objects'][0], scene['objects'][0])
        self.assertEqual(aligned['materials'], scene['materials'])
        self.assertEqual([o['id'] for o in aligned['objects']], [o['id'] for o in scene['objects']])
        moved = aligned['components'][1]
        port = moved['ports'][0]
        target = fixed['ports'][1]
        for axis in 'xyz':
            self.assertAlmostEqual(port[axis], target[axis])
        self.assertAlmostEqual((port['angle_deg'] - target['angle_deg']) % 360, 180.)
        regenerated = place_component(aligned, moving['family'], replace_component_id=moved['id'])['geometry']
        for actual, regenerated_solid in zip(aligned['objects'], regenerated['objects']):
            for axis in 'xyz':
                self.assertAlmostEqual(actual['position'][axis], regenerated_solid['position'][axis])
            self.assertAlmostEqual(actual['rotation_deg'], regenerated_solid['rotation_deg'])
        self.assertEqual(align_ports(aligned, moving['id'], 'lower_in', fixed['id'], 'out'), aligned)

    def test_invalid_alignment_does_not_change_any_component(self):
        scene = self.pair()
        before = copy.deepcopy(scene)
        fixed, moving = scene['components']
        with self.assertRaisesRegex(ValueError, 'two different'):
            align_ports(scene, fixed['id'], 'in', fixed['id'], 'out')
        with self.assertRaisesRegex(ValueError, 'existing named'):
            align_ports(scene, moving['id'], 'invalid', fixed['id'], 'out')
        narrow = place_component(scene, 'strip', {'width_um': .25}, replace_component_id=fixed['id'])['geometry']
        with self.assertRaisesRegex(ValueError, 'widths differ'):
            align_ports(narrow, moving['id'], 'lower_in', fixed['id'], 'out', True)
        self.assertEqual(scene, before)

    def test_domain_growth_is_opt_in_and_an_existing_join_is_protected(self):
        fixed = build_design('strip', {'length_um': 3.})['geometry']
        two = place_component(fixed, 'strip', {'length_um': 3.})['geometry']
        first, second = (c['id'] for c in two['components'])
        body = dict(geometry=two, moving_component=second, moving_port='in',
                    fixed_component=first, fixed_port='out')
        client = TestClient(app)
        self.assertEqual(client.post('/api/library/align', json=body).status_code, 422)
        response = client.post('/api/library/align', json={**body, 'fit_domain': True})
        self.assertEqual(response.status_code, 200, response.text)
        joined = response.json()['geometry']
        three = place_component(joined, 'strip', {'length_um': 2.}, {'y': 2.}, fit_domain=True)['geometry']
        third = three['components'][2]['id']
        with self.assertRaisesRegex(ValueError, 'already meets'):
            align_ports(three, third, 'in', first, 'out', True)
        with self.assertRaisesRegex(ValueError, 'break an existing'):
            align_ports(three, second, 'out', third, 'in', True)
