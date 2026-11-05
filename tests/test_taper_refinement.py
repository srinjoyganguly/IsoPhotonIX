"""Synthetic numerical-contract tests. Real Meep and CUDA reports live in docs/verification."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.source_snapshot import capture_implementation
from backend.taper_ports import example_taper_ports
from backend.taper_refinement import compare_taper_mesh
from tests.test_modal_ports import fixture


class TaperMeshTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.coarse = self.root / 'coarse'
        self.fine = self.root / 'fine'
        self.coarse.mkdir()
        self.fine.mkdir()
        self.archive = capture_implementation(self.coarse)
        (self.fine / 'implementation.zip').write_bytes((self.coarse / 'implementation.zip').read_bytes())

    def run_fixture(self, folder, resolution, *, angles=None, valid=None, patch_metadata=None, **config_updates):
        arrays, metadata = fixture()
        config = {**example_taper_ports().model_dump(), 'resolution': resolution, **config_updates}
        metadata.update(config=config, requested_device='cpu', device_name='CPU', implementation_snapshot=self.archive,
                        source={'mode': 'fixture', 'eig_resolution': resolution, 'source_x_um': [-2.75, 2.75]},
                        boundaries={'pml_um': 0.75}, normalization={'incident': 'measured'},
                        geometry_translation={'domain': {'x': 8, 'y': 4, 'z': 4}})
        metadata['ports']['aperture_yz_um'] = [2.5, 2.5]
        arrays['mode_neff_ports'] = np.column_stack((arrays['mode_neff'], np.full(5, 1.8 + 0j)))
        if angles is not None:
            arrays['s'][:, 1, 0] = .8 * np.exp(1j * np.deg2rad(angles))
        arrays['s'][:, 1, 1] = 0
        arrays['outgoing_amplitude'] = arrays['s'] * arrays['incident_amplitude'][:, None, :]
        if valid is not None:
            arrays['valid_bin'] = np.array(valid, dtype=bool)
            metadata['sparameters_valid'] = bool(np.all(arrays['valid_bin']))
            arrays['s'][~arrays['valid_bin']] = complex(np.nan, np.nan)
        if patch_metadata:
            patch_metadata(metadata)
        (folder / 'config.json').write_text(json.dumps(config))
        (folder / 'metadata.json').write_text(json.dumps(metadata))
        np.savez(folder / 'sparameters.npz', **arrays)

    def test_wraps_phase_intersects_bins_and_omits_zero_amplitude_phase(self):
        self.run_fixture(self.coarse, 16, angles=179, valid=[1, 1, 1, 1, 0])
        self.run_fixture(self.fine, 20, angles=-179, valid=[1, 0, 1, 1, 1])
        report = compare_taper_mesh(self.coarse, self.fine)
        self.assertEqual(report['compared_bins'], 3)
        self.assertEqual(report['jointly_valid_bins'], [True, False, True, True, False])
        phase = report['entries']['S21']['principal_phase_change_rad']
        self.assertAlmostEqual(phase[0], np.deg2rad(2))
        self.assertIsNone(phase[1])
        self.assertIsNone(phase[-1])
        self.assertLess(report['entries']['S21']['maximum_absolute_power_change'], 1e-14)
        self.assertIsNone(report['entries']['S22']['maximum_absolute_phase_change_rad'])
        self.assertGreater(report['complex_matrix_relative_l2'], 0.)
        json.dumps(report, allow_nan=False)

    def test_identical_spectra_have_zero_mesh_change(self):
        self.run_fixture(self.coarse, 16)
        self.run_fixture(self.fine, 20)
        report = compare_taper_mesh(self.coarse, self.fine)
        self.assertEqual(report['complex_matrix_relative_l2'], 0.)
        self.assertEqual(report['entries']['S21']['maximum_absolute_phase_change_rad'], 0.)

    def test_incompatible_settings_order_planes_and_empty_overlap(self):
        self.run_fixture(self.coarse, 16, valid=[0, 1, 1, 0, 0])
        cases = [
            (16, {}, 'strictly finer'),
            (20, {'decay_by': 1e-8}, 'thresholds differ'),
            (20, {'max_after_sources_fs': 500}, 'Only resolution may differ'),
            (20, {'valid': [1, 0, 0, 1, 1]}, 'No wavelength bin'),
            (20, {'patch_metadata': lambda m: m['ports'].update(port_2_x_um=1.76)}, 'reference_planes_um'),
            (20, {'patch_metadata': lambda m: m.update(device_name='changed')}, 'device_name'),
            (20, {'patch_metadata': lambda m: m.update(actual_device='cuda:0')}, 'hardware differs'),
            (20, {'patch_metadata': lambda m: m['source'].update(eig_resolution=16)}, 'mode grid'),
            (20, {'patch_metadata': lambda m: m['ports'].update(mode_gauge='different convention')}, 'port_conventions'),
            (20, {'patch_metadata': lambda m: m['ports'].update(aperture_yz_um=[2.5])}, 'finite physical aperture'),
        ]
        for res, changes, message in cases:
            with self.subTest(changes=message):
                self.run_fixture(self.fine, res, **changes)
                with self.assertRaisesRegex(ValueError, message):
                    compare_taper_mesh(self.coarse, self.fine)

    def test_rejects_unfinished_runs_or_mixed_archives(self):
        self.run_fixture(self.coarse, 16)
        self.run_fixture(self.fine, 20)
        (self.fine / 'job.json').write_text('{"status":"running"}')
        with self.assertRaisesRegex(ValueError, 'completed'):
            compare_taper_mesh(self.coarse, self.fine)
        (self.fine / 'job.json').unlink()
        with (self.fine / 'implementation.zip').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            compare_taper_mesh(self.coarse, self.fine)

    def test_saved_run_api_lists_meta_and_compares_only_completed_jobs(self):
        self.run_fixture(self.coarse, 16)
        self.run_fixture(self.fine, 20)
        jobs = {folder.name: {'id': folder.name, 'status': 'completed',
                              'config': json.loads((folder / 'config.json').read_text())}
                for folder in (self.coarse, self.fine)}
        with patch('backend.app.manager') as manager, TestClient(app) as client:
            manager.directory = self.root
            manager.jobs = jobs
            manager.get.side_effect = jobs.__getitem__
            listed = client.get('/api/taper-ports/mesh-runs').json()
            self.assertEqual([r['id'] for r in listed], ['fine', 'coarse'])
            body = {'coarse_run_id': 'coarse', 'fine_run_id': 'fine'}
            response = client.post('/api/taper-ports/compare-mesh', json=body)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['compared_bins'], 5)
            self.assertEqual(response.json()['fine_run_id'], 'fine')
            self.assertEqual(client.post('/api/taper-ports/compare-mesh',
                json={'coarse_run_id': 'fine', 'fine_run_id': 'coarse'}).status_code, 409)
            jobs['coarse']['status'] = 'running'
            self.assertEqual(client.post('/api/taper-ports/compare-mesh', json=body).status_code, 409)
            jobs['coarse']['status'] = 'completed'
            self.assertEqual(client.post('/api/taper-ports/compare-mesh',
                json={**body, 'coarse_run_id': '../coarse'}).status_code, 422)
            self.assertEqual(client.post('/api/taper-ports/compare-mesh',
                json={**body, 'fine_run_id': 'missing'}).status_code, 404)
