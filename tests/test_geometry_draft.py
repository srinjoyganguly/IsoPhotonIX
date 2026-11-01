"""Shared geometry/material contract fixtures, not electromagnetic evidence."""
import copy
import json
import math
from pathlib import Path
import unittest

from backend.geometry_draft import evaluate_material, geometry_bounds, validate_geometry, validate_material

FIXTURES = json.loads((Path(__file__).parents[1] / 'frontend/geometry-fixtures.json').read_text())


class GeometryDraftTests(unittest.TestCase):
    def test_shared_valid_documents_normalize_without_mutating_input(self):
        for case in FIXTURES['valid']:
            with self.subTest(case=case['name']):
                original = copy.deepcopy(case['input'])
                actual = validate_geometry(original)
                self.assertEqual(actual, case['expected'])
                self.assertEqual(original, case['input'])
                self.assertEqual(validate_geometry(actual), actual)
                actual['objects'][0]['position']['x'] = 99
                self.assertEqual(original, case['input'])

    def test_shared_invalid_topology_material_and_component_documents_fail(self):
        for case in FIXTURES['invalid']:
            with self.subTest(case=case['name']):
                with self.assertRaises(ValueError):
                    validate_geometry(case['input'])

    def test_linear_material_interpolation_is_explicit_and_never_extrapolates(self):
        material = FIXTURES['valid'][1]['input']['materials'][0]
        self.assertEqual(evaluate_material(material, 1), {'n': 2, 'k': 0})
        self.assertEqual(evaluate_material(material, 1.5), {'n': 2.5, 'k': .1})
        self.assertEqual(evaluate_material(material, 2), {'n': 3, 'k': .2})
        for wavelength in (.5, 2.1, 0, float('nan'), float('inf'), 10**400, True, '1.5'):
            with self.subTest(wavelength=wavelength), self.assertRaises(ValueError):
                evaluate_material(material, wavelength)
        constant = FIXTURES['valid'][5]['input']['materials'][0]
        self.assertEqual(evaluate_material(constant, 10), {'n': .005, 'k': 0})
        for patch in ({'model': 'gain'}, {'n': 0}, {'n': -1}, {'n': 21}, {'k': -.01}, {'k': math.inf}, {'source': 'x' * 1025}, {'notes': 'x' * 2001}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_material(dict(constant, **patch))

    def test_offcentre_polygon_bounds_use_vertices_rotation_and_position(self):
        item = copy.deepcopy(FIXTURES['valid'][1]['input']['objects'][0])
        item.update(vertices=[{'x': 1, 'y': 1}, {'x': 3, 'y': 1}, {'x': 3, 'y': 2}, {'x': 1, 'y': 2}], holes=[],
                    position={'x': .5, 'y': -1, 'z': .3}, rotation_deg=90,
                    size={'x': 999, 'y': 999, 'z': .4})
        actual = geometry_bounds(item)
        for bound, expected in (('min', {'x': -1.5, 'y': 0, 'z': .1}), ('max', {'x': -.5, 'y': 2, 'z': .5})):
            for axis in 'xyz':
                self.assertAlmostEqual(actual[bound][axis], expected[axis])
        document = copy.deepcopy(FIXTURES['valid'][1]['input'])
        document['objects'] = [item]
        self.assertEqual(validate_geometry(document)['objects'][0]['size']['x'], 999)

    def test_v1_limit_preserved_and_v2_limit_bounded(self):
        for version, limit in ((1, 100), (2, 512)):
            document = copy.deepcopy(FIXTURES['valid'][0]['input'])
            if version == 2:
                document.update(schema_version=2, materials=[], components=[])
            prototype = document['objects'][0]
            document['objects'] = [dict(prototype, id=f'box-{i}') for i in range(limit)]
            self.assertEqual(len(validate_geometry(document)['objects']), limit)
            document['objects'].append(dict(prototype, id='too-many'))
            with self.assertRaises(ValueError):
                validate_geometry(document)

    def test_nested_holes_and_crossing_edges_are_rejected(self):
        document = copy.deepcopy(FIXTURES['valid'][1]['input'])
        solid = document['objects'][0]
        solid['vertices'] = [{'x': -3, 'y': -2}, {'x': 3, 'y': -2}, {'x': 3, 'y': 2}, {'x': -3, 'y': 2}]
        solid['holes'] = [[{'x': -1, 'y': -1}, {'x': -1, 'y': 1}, {'x': 1, 'y': 1}, {'x': 1, 'y': -1}],
                          [{'x': -.5, 'y': -.5}, {'x': -.5, 'y': .5}, {'x': .5, 'y': .5}, {'x': .5, 'y': -.5}]]
        with self.assertRaisesRegex(ValueError, 'nest'):
            validate_geometry(document)
        solid['holes'] = []
        solid['vertices'] = [{'x': 0, 'y': 0}, {'x': 2, 'y': 0}, {'x': 1, 'y': 0}, {'x': 1, 'y': 1}, {'x': 0, 'y': 1}]
        with self.assertRaises(ValueError):
            validate_geometry(document)

    def test_malformed_metadata_and_nonfinite_geometry_fail(self):
        for patch in ({'schema_version': 3}, {'domain': {'x': math.nan, 'y': 2, 'z': 2}}, {'name': 'bad\nname'}, {'components': {}}, {'materials': {}}, {'extra': 'unsupported'}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_geometry(dict(FIXTURES['valid'][1]['input'], **patch))
        for bad in (None, [], 4, 'geometry'):
            with self.assertRaises(ValueError):
                validate_geometry(bad)


if __name__ == '__main__':
    unittest.main()
