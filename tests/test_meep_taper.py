"""Native Meep taper geometry and straight-limit checks on actual solver arrays."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.adapters import meep3d_adapter, meep_taper
from backend.component_library import build_component
from backend.taper_geometry import taper_geometry


def study(output_width=1.25, resolution=16):
    geometry = build_component('taper', {'length_um': 3., 'height_um': .375,
                                         'input_width_um': .625, 'output_width_um': output_width})['geometry']
    # Hold every external coordinate and source fixed across geometry changes.
    geometry['domain'] = {'x': 7.5, 'y': 4., 'z': 3.5}
    return {'schema_version': 5, 'study_type': 'taper-3d', 'solver': 'meep', 'device': 'cpu',
            'geometry': geometry, 'wavelength_um': 1.55, 'pml_um': .75, 'duration_fs': 60.,
            'resolution': resolution, 'cladding_index': 1.}


def straight_block(config):
    model = taper_geometry(config)
    domain = model['domain']
    core = {'id': 'core', 'name': 'Straight block', 'kind': 'box', 'visible': True, 'rotation_deg': 0.,
            'radius': 1., 'inner_radius': .6, 'size': {'x': domain['x'], 'y': model['width_in'], 'z': model['height']},
            'position': {'x': 0., 'y': 0., 'z': 0.}, 'refractive_index': model['core_index']}
    geometry = {'schema': 'photonics-workbench-geometry', 'schema_version': 1, 'mode': 'draft-3d',
                'units': 'um', 'name': 'Matched rectangular guide', 'domain': domain, 'objects': [core]}
    return {**config, 'schema_version': 2, 'study_type': 'waveguide-3d', 'geometry': geometry}


def relative_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


class MeepTaperTests(unittest.TestCase):
    def test_other_targets_cannot_start_the_meep_taper_adapter(self):
        for config in ({'device': 'gpu'}, {'device': 'cpu', 'solver': 'fdtdx'},
                       {'device': 'cpu', 'solver': 'meep', 'schema_version': 2, 'study_type': 'waveguide-3d'}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                meep_taper.run(config, Path('unused'))

    @unittest.skipUnless(importlib.util.find_spec('meep'), 'Requires official PyMeep')
    def test_real_taper_profile_refinement_and_straight_limit(self):
        cases = {}
        with tempfile.TemporaryDirectory() as scratch:
            def simulate(config, name, adapter=meep_taper):
                folder = Path(scratch) / name
                metadata = adapter.run(config, folder)
                with np.load(folder / 'arrays.npz', allow_pickle=False) as archive:
                    data = {key: archive[key] for key in archive.files}
                self.assertEqual(metadata['actual_device'], 'CPU')
                self.assertLessEqual(abs(metadata['actual_duration_fs'] - config['duration_fs']), metadata['time_step_fs'])
                self.assertEqual(data['fields'].shape, (3, *(len(data[f'{a}_um']) for a in 'zyx')))
                self.assertTrue(np.isfinite(data['fields']).all())
                self.assertTrue(np.isfinite(data['epsilon']).all())
                self.assertTrue(all(np.any(f) for f in data['fields']))
                np.testing.assert_array_equal(data['field'], data['fields'][1])
                cases[name] = {'grid_xyz': metadata['grid_shape'], 'resolution': config['resolution'],
                               'versions': metadata['versions'], 'duration_fs': metadata['actual_duration_fs'],
                               'source_x_um': metadata['source']['x_um'], 'time_steps': metadata['time_steps'],
                               'peak_abs_ey': metadata['field_max_abs'], 'geometry': config['geometry'],
                               'array_sha256': {key: hashlib.sha256(data[key].tobytes()).hexdigest()
                                                for key in ('fields', 'epsilon')}}
                return data

            def verify_profile(config, data, name):
                model = taper_geometry(config)
                x, y, z = (data[f'{a}_um'] for a in 'xyz')
                center_y, center_z = int(np.argmin(abs(y))), int(np.argmin(abs(z)))
                epsilon = data['epsilon']
                threshold = (model['core_index']**2 + config['cladding_index']**2) / 2
                measured = []
                for position in (model['x_in'] - .25, model['x_in'] + .25, 0., model['x_out'] - .25, model['x_out'] + .25):
                    ix = int(np.argmin(abs(x - position)))
                    expected = float(np.interp(x[ix], [model['x_in'], model['x_out']], [model['width_in'], model['width_out']]))
                    width = float(np.count_nonzero(epsilon[center_z, :, ix] > threshold) / config['resolution'])
                    self.assertLessEqual(abs(expected - width), 1.5 / config['resolution'] + 1e-12)
                    measured.append({'x_um': float(x[ix]), 'expected_width_um': expected, 'measured_width_um': width})
                ix = int(np.argmin(abs(x)))
                self.assertAlmostEqual(float(epsilon[center_z, center_y, ix]), model['core_index']**2, places=4)
                height = float(np.count_nonzero(epsilon[:, center_y, ix] > threshold) / config['resolution'])
                self.assertLessEqual(abs(height - model['height']), 1.5 / config['resolution'] + 1e-12)
                self.assertAlmostEqual(float(epsilon[np.argmin(abs(z - .8)), center_y, ix]), 1., places=4)
                output_lead = (x > model['x_out'] + .1) & (x < model['domain']['x'] / 2 - config['pml_um'])
                peak = float(np.max(np.abs(data['field'][:, :, output_lead])))
                self.assertGreater(peak, 1e-6)
                cases[name].update(width_probes=measured, measured_height_um=height, output_lead_peak_abs_ey=peak)

            expanded = study()
            tapered = simulate(expanded, 'widening-taper')
            verify_profile(expanded, tapered, 'widening-taper')
            straight = study(output_width=.625)
            constant = simulate(straight, 'constant-width-taper')
            verify_profile(straight, constant, 'constant-width-taper')
            native = simulate(straight_block(straight), 'single-native-block', meep3d_adapter)
            for axis in 'xyz':
                np.testing.assert_array_equal(constant[f'{axis}_um'], native[f'{axis}_um'])
                np.testing.assert_array_equal(constant[f'{axis}_um'], tapered[f'{axis}_um'])
            material_difference = relative_l2(constant['epsilon'], native['epsilon'])
            field_difference = relative_l2(constant['fields'], native['fields'])
            self.assertLess(material_difference, .005)
            self.assertLess(field_difference, .02)
            widening_ey_difference = relative_l2(tapered['field'], constant['field'])
            self.assertGreater(widening_ey_difference, .02)
            refined = study(resolution=20)
            finer = simulate(refined, 'widening-taper-finer-grid')
            verify_profile(refined, finer, 'widening-taper-finer-grid')
        report = {'schema_version': 1, 'checked_at': datetime.now(timezone.utc).isoformat(), 'passed': True,
                  'scope': 'Real Meep CPU field and material checks for a centered linear taper and its access guides.',
                  'development_check': {'segmented_prism_and_leads_vs_uniform_block_epsilon_relative_l2': .00581317,
                                        'segmented_prism_and_leads_vs_uniform_block_epsilon_peak_difference': .793724,
                                        'correction': 'A single prism extrudes the union of the exact saved taper and access leads, removing internal subpixel interfaces.'},
                  'checks': {'relative_l2_equal_width_vs_native_block_permittivity': material_difference,
                             'relative_l2_equal_width_vs_native_block_vector_fields': field_difference,
                             'relative_l2_widening_vs_equal_width_ey': widening_ey_difference,
                             'max_allowed_width_or_height_error_grid_cells': 1.5}, 'cases': cases,
                  'interpretation': 'The two meshes check physical shape sampling. These fields use a finite ramped current sheet; they do not measure modal transmission, convergence or commercial solver agreement.',
                  'documentation': 'https://meep.readthedocs.io/en/latest/Python_User_Interface/#prism'}
        evidence = os.environ.get('PHOTONICS_MEEP_TAPER_REPORT')
        if evidence:
            Path(evidence).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    unittest.main()
