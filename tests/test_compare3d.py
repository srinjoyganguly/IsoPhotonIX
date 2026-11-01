"""Synthetic saved-data tests for comparison mathematics and compatibility gates.

These fixtures test analysis only; they are not solver or physics evidence.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.compare3d import compare_runs
from backend.study3d import example3d


class Compare3DTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        config = example3d().model_dump()
        config.update(solver='fdtdx', device='cpu')
        self.reference = dict(id='reference', status='completed', config=config)
        self.candidate = copy.deepcopy(self.reference)
        self.candidate['id'] = 'candidate'
        self.candidate['config']['device'] = 'gpu'
        z, y, x = np.indices((2, 3, 4))
        ey = (1 + x + 4*y + 12*z).astype(np.float32)
        self.arrays = dict(fields=np.stack([ey*2, ey, -ey]), field=ey,
                           epsilon=np.ones_like(ey)*4,
                           x_um=np.arange(4.), y_um=np.arange(3.), z_um=np.arange(2.))
        self.metadata = dict(dimensions='3D', field_components=['Ex', 'Ey', 'Ez'],
                             field_component='Ey', versions={'fdtdx': 'fixture', 'jax': 'fixture'},
                             source={'type': 'fixture plane', 'component': 'Ey'},
                             field_sampling='fixture common centres', material_sampling='fixture cells',
                             boundaries={'x': 'PML', 'y': 'PML', 'z': 'PML'},
                             field_units='fixture arbitrary units', array_order='fields[component,z,y,x]',
                             simulated_time_fs=60.0)
        self.save(self.reference)
        self.save(self.candidate)

    def save(self, job, arrays=None, metadata=None):
        output = self.directory / job['id']
        output.mkdir(exist_ok=True)
        saved = copy.deepcopy(self.metadata if metadata is None else metadata)
        saved['config'] = copy.deepcopy(job['config'])
        (output/'config.json').write_text(json.dumps(job['config']))
        (output/'metadata.json').write_text(json.dumps(saved))
        np.savez_compressed(output/'arrays.npz', **(self.arrays if arrays is None else arrays))

    def compare(self):
        return compare_runs(self.reference, self.candidate, self.directory)

    def test_cpu_gpu_metrics_use_saved_samples_and_reference_denominator(self):
        arrays = copy.deepcopy(self.arrays)
        arrays['fields'][0] *= 1.25
        self.save(self.candidate, arrays=arrays)
        result = self.compare()
        self.assertEqual(result['reference_target'], 'fdtdx / cpu')
        self.assertEqual(result['candidate_target'], 'fdtdx / gpu')
        self.assertAlmostEqual(result['components']['Ex']['relative_l2'], .25)
        self.assertEqual(result['components']['Ex']['max_abs_error'], 12.)
        self.assertEqual(result['components']['Ex']['reference_peak_abs'], 48.)
        self.assertEqual(result['components']['Ey']['relative_l2'], 0.)
        self.assertEqual(result['components']['Ez']['max_abs_error'], 0.)
        self.assertAlmostEqual(result['overall_relative_l2'], .5 / np.sqrt(6))
        self.assertEqual(result['commercial_agreement'], 'not_validated')
        self.assertTrue(result['epsilon_identical'])
        self.assertTrue(result['coordinate_grids_identical'])
        json.dumps(result, allow_nan=False)

    def test_presentation_changes_and_object_order_do_not_change_physics(self):
        geometry = self.candidate['config']['geometry']
        geometry['name'] = 'A new title'
        geometry['objects'].reverse()
        for obj in geometry['objects']:
            obj['visible'] = not obj['visible']
            obj['name'] = 'Presentation only'
            obj['radius'] = 123.  # Unused property for the supported box solids.
        self.save(self.candidate)
        self.assertEqual(self.compare()['overall_relative_l2'], 0.)

    def test_same_id_incomplete_and_non_3d_studies_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different completed'):
            compare_runs(self.reference, self.reference, self.directory)
        for change in (lambda j: j.update(status='running'),
                       lambda j: j['config'].update(study_type='other-study')):
            candidate = copy.deepcopy(self.candidate)
            change(candidate)
            with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, 'completed finite-height'):
                compare_runs(self.reference, candidate, self.directory)

    def test_physical_changes_are_rejected_without_rescaling(self):
        for change in (lambda c: c.update(wavelength_um=1.6),
                       lambda c: c.update(resolution=20),
                       lambda c: c.update(solver='meep'),
                       lambda c: c['geometry']['objects'][1]['size'].update(z=.625),
                       lambda c: c['geometry']['objects'][1].update(refractive_index=2.1)):
            candidate = copy.deepcopy(self.candidate)
            change(candidate['config'])
            with self.subTest(config=candidate['config']), self.assertRaisesRegex(ValueError, 'Same-study'):
                compare_runs(self.reference, candidate, self.directory)

    def test_source_versions_sampling_and_field_units_must_match_and_exist(self):
        for key in ('source', 'versions', 'field_sampling', 'material_sampling',
                    'boundaries', 'field_units', 'array_order'):
            metadata = copy.deepcopy(self.metadata)
            metadata[key] = 'different convention'
            self.save(self.candidate, metadata=metadata)
            with self.subTest(mismatch=key), self.assertRaisesRegex(ValueError, key):
                self.compare()
            metadata.pop(key)
            self.save(self.reference, metadata=metadata)
            self.save(self.candidate, metadata=metadata)
            with self.subTest(missing_from_both=key), self.assertRaisesRegex(ValueError, key):
                self.compare()
            self.save(self.reference)

    def test_different_or_invalid_times_are_rejected_and_meep_time_key_is_supported(self):
        for time in (60.001, None, '60', True, 0, float('nan'), float('inf')):
            metadata = copy.deepcopy(self.metadata)
            metadata['simulated_time_fs'] = time
            self.save(self.candidate, metadata=metadata)
            with self.subTest(time=time), self.assertRaisesRegex(ValueError, 'physical times'):
                self.compare()
        metadata = copy.deepcopy(self.metadata)
        metadata.pop('simulated_time_fs')
        metadata['actual_duration_fs'] = 60.
        self.save(self.reference, metadata=metadata)
        self.save(self.candidate, metadata=metadata)
        self.assertEqual(self.compare()['actual_time_fs'], 60.)

    def test_axes_and_actual_material_grid_must_match(self):
        for key in ('x_um', 'y_um', 'z_um', 'epsilon'):
            arrays = copy.deepcopy(self.arrays)
            arrays[key] += .01
            self.save(self.candidate, arrays=arrays)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'grids differ'):
                self.compare()

    def test_metadata_semantics_and_saved_configuration_are_checked(self):
        for key, value in (('dimensions', '2D'), ('field_components', ['Ey', 'Ex', 'Ez']),
                           ('field_component', 'Ex')):
            metadata = copy.deepcopy(self.metadata)
            metadata[key] = value
            self.save(self.candidate, metadata=metadata)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, '3D Ex/Ey/Ez'):
                self.compare()
        self.save(self.candidate)
        path = self.directory/'candidate'/'config.json'
        changed = copy.deepcopy(self.candidate['config'])
        changed['wavelength_um'] = 1.6
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'configuration does not match'):
            self.compare()

    def test_zero_reference_component_reports_undefined_relative_error_as_null(self):
        arrays = copy.deepcopy(self.arrays)
        arrays['fields'][0] = 0
        self.save(self.reference, arrays=arrays)
        self.save(self.candidate, arrays=arrays)
        result = self.compare()
        self.assertIsNone(result['components']['Ex']['relative_l2'])
        self.assertEqual(result['components']['Ex']['max_abs_error'], 0)
        self.assertEqual(result['overall_relative_l2'], 0)
        json.dumps(result, allow_nan=False)
