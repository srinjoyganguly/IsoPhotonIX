"""Export integrity and arithmetic tests; synthetic arrays are not solver evidence."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from backend.model import WaveguideConfig
from backend.ports import example_ports
from backend.ports3d import example_ports3d
from backend.reproduce import geometry_bundle, study_bundle
from backend.study3d import example3d
from backend.source_snapshot import capture_implementation


def runtime(folder):
    spec = importlib.util.spec_from_file_location('isolated_workbench_runtime', folder / 'workbench_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unpack(data, folder):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(folder)


class ReproductionExports(unittest.TestCase):
    def test_exported_alignment_verifies_its_bundled_generator_before_use(self):
        from backend.library_layout import build_design
        geometry = build_design('strip')['geometry']
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            unpack(geometry_bundle(geometry), folder)
            source = folder / 'design_source/library_layout.py'
            source.write_bytes(source.read_bytes() + b'\n# Changed after export\n')
            with self.assertRaisesRegex(ValueError, 'Bundled source or reference file changed'):
                runtime(folder).align_ports(geometry, 'moving', 'in', 'fixed', 'out')

    def test_all_study_formats_embed_editable_code_notebooks_and_complete_source(self):
        configs = [WaveguideConfig().model_dump(), example3d().model_dump(by_alias=True),
                   example_ports().model_dump(), example_ports3d().model_dump(by_alias=True)]
        for config in configs:
            with self.subTest(schema=config['schema_version']), zipfile.ZipFile(io.BytesIO(study_bundle(config))) as archive:
                self.assertEqual(json.loads(archive.read('config.json')), config)
                names = set(archive.namelist())
                self.assertIn('source/backend/worker.py', names)
                self.assertIn('source/backend/study3d.py', names)
                self.assertFalse(any(name in names for name in ('source/backend/app.py', 'source/backend/jobs.py', '.local/runtime.json')))
                self.assertFalse(any('worker.log' in name or '__pycache__' in name for name in names))
                compile(archive.read('study.py'), 'study.py', 'exec')
                notebook = json.loads(archive.read('study.ipynb'))
                self.assertEqual(notebook['nbformat'], 4)
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        compile(''.join(cell['source']), cell['id'], 'exec')
                        self.assertEqual(cell['outputs'], [])
                manifest = json.loads(archive.read('bundle-manifest.json'))
                self.assertEqual(manifest['snapshot_origin'], 'export_time')
                for name, item in manifest['files'].items():
                    self.assertEqual(item['sha256'], hashlib.sha256(archive.read(name)).hexdigest())
                self.assertNotIn(str(Path.home()), archive.read('study.py').decode())
                self.assertNotIn(str(Path.home()), archive.read('workbench_runtime.py').decode())

    def test_geometry_preserves_rings_cylinders_rotation_hidden_state_and_safe_names(self):
        geometry = example3d().geometry.model_dump(by_alias=True)
        core = geometry['objects'][1]
        core.update(kind='ring', radius=.8, inner_radius=.4, rotation_deg=32., visible=False,
                    name='Ring """; print("not code")')
        cylinder = dict(core, id='cylinder', kind='cylinder', radius=.25, visible=True)
        geometry['objects'].append(cylinder)
        with zipfile.ZipFile(io.BytesIO(geometry_bundle(geometry))) as archive:
            self.assertEqual(json.loads(archive.read('geometry.json')), geometry)
            compile(archive.read('geometry.py'), 'geometry.py', 'exec')
            notebook = json.loads(archive.read('geometry.ipynb'))
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code': compile(''.join(cell['source']), cell['id'], 'exec')
            self.assertIn('not an executable electromagnetic study', archive.read('geometry.py').decode())
        for key, value in [('radius', 50.), ('visible', 'true'), ('rotation_deg', float('nan'))]:
            bad = json.loads(json.dumps(geometry)); bad['objects'][1][key] = value
            with self.assertRaises(ValueError): geometry_bundle(bad)

    def test_saved_reference_requires_exact_configuration_and_excludes_private_files(self):
        config = WaveguideConfig().model_dump()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'config.json').write_text(json.dumps(config))
            (folder / 'metadata.json').write_text(json.dumps({'config': config, 'versions': {'numpy': np.__version__}}))
            (folder / 'worker.log').write_text('private local path and raw diagnostic log')
            (folder / '.secret').write_text('never include')
            np.savez(folder / 'arrays.npz', field=np.ones((2, 2)))
            with zipfile.ZipFile(io.BytesIO(study_bundle(config, folder))) as archive:
                self.assertIn('reference/arrays.npz', archive.namelist())
                self.assertNotIn('reference/worker.log', archive.namelist())
                self.assertNotIn('reference/.secret', archive.namelist())
                self.assertEqual(json.loads(archive.read('bundle-manifest.json'))['snapshot_origin'], 'export_time_compatibility')
            changed = dict(config, core_index=2.1)
            with self.assertRaises(ValueError): study_bundle(changed, folder)

    def test_run_time_source_archive_is_used_and_checked_without_current_code_substitution(self):
        config = WaveguideConfig().model_dump()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            record = capture_implementation(folder)
            metadata = {'config': config, 'actual_device': 'CPU', 'versions': {}, 'implementation_snapshot': record}
            (folder / 'config.json').write_text(json.dumps(config))
            (folder / 'metadata.json').write_text(json.dumps(metadata))
            np.savez(folder / 'arrays.npz', field=np.ones((2, 2)))
            with patch('backend.reproduce._source_snapshot', side_effect=AssertionError('Must use recorded source')):
                data = study_bundle(config, folder)
            with zipfile.ZipFile(io.BytesIO(data)) as archive, zipfile.ZipFile(folder / 'implementation.zip') as original:
                self.assertEqual(json.loads(archive.read('bundle-manifest.json'))['snapshot_origin'], 'run_time')
                for name in original.namelist():
                    if name.startswith('source/'):
                        self.assertEqual(archive.read(name), original.read(name))
            bundle = folder / 'bundle'; unpack(data, bundle)
            runner = runtime(bundle)
            report = runner.compare_results(folder)
            self.assertTrue(report['passed']); self.assertTrue(report['recorded_implementation_preserved'])
            (folder / 'implementation.zip').write_bytes(b'tampered')
            self.assertFalse(runner.compare_results(folder)['passed'])
            with self.assertRaises(ValueError): study_bundle(config, folder)

    def test_meep_unused_incoming_residual_has_separate_transparent_bound_only(self):
        config = example_ports3d().model_dump(by_alias=True)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); original = base / 'original'; original.mkdir()
            metadata = {'config': config, 'actual_device': 'CPU', 'versions': {'meep': 'fixture'}}
            for name, value in [('config.json', config), ('metadata.json', metadata)]: (original / name).write_text(json.dumps(value))
            np.savez(original / 'arrays.npz', field=np.ones((2, 2)))
            alpha = np.array([[1.+0j, .01+0j], [1.+0j, .01+0j]])
            np.savez(original / 'device-reverse-monitors.npz', port2_alpha=alpha, Ey=np.ones((2, 2)))
            bundle = base / 'bundle'; unpack(study_bundle(config, original), bundle)
            actual = bundle / 'actual'; actual.mkdir()
            for path in original.iterdir(): (actual / path.name).write_bytes(path.read_bytes())
            runner = runtime(bundle)
            varied = alpha.copy(); varied[0, 1] += 3.4e-8
            np.savez(actual / 'device-reverse-monitors.npz', port2_alpha=varied, Ey=np.ones((2, 2)))
            report = runner.compare_results(actual)
            self.assertTrue(report['passed']); self.assertTrue(report['physical_results_passed'])
            self.assertFalse(report['all_arrays_pointwise_passed'])
            self.assertFalse(report['arrays']['device-reverse-monitors.npz']['port2_alpha']['passed'])
            self.assertTrue(report['diagnostic_variations'][0]['amplitude_scaled_passed'])
            (actual / 'metadata.json').write_text(json.dumps(dict(metadata, versions={'meep': 'different'})))
            self.assertFalse(runner.compare_results(actual)['passed'])
            (actual / 'metadata.json').write_text(json.dumps(metadata))
            varied[0, 0] += 1e-4  # Actual outgoing coefficient must remain strict.
            np.savez(actual / 'device-reverse-monitors.npz', port2_alpha=varied, Ey=np.ones((2, 2)))
            self.assertFalse(runner.compare_results(actual)['physical_results_passed'])
            np.savez(actual / 'device-reverse-monitors.npz', port2_alpha=alpha, Ey=np.ones((2, 2)) * 1.1)
            self.assertFalse(runner.compare_results(actual)['passed'])
            np.savez(actual / 'device-reverse-monitors.npz', port2_alpha=alpha, Ey=np.ones((2, 2)), new_key=np.ones(2))
            self.assertFalse(runner.compare_results(actual)['passed'])

    def test_reproduction_records_exact_complex_samples_and_fails_real_discrepancy(self):
        config = WaveguideConfig().model_dump()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); original = base / 'original'; original.mkdir()
            metadata = {'config': config, 'actual_device': 'CPU', 'versions': {'meep': 'fixture'}}
            for name, value in [('config.json', config), ('metadata.json', metadata)]: (original / name).write_text(json.dumps(value))
            complex_values = np.array([1+2j, np.nan+0j, .5-1j])
            np.savez(original / 'arrays.npz', field=np.ones((2, 2)), x_um=np.arange(2.))
            np.savez(original / 'sparameters.npz', s=complex_values, valid_bin=np.array([True, False, True]))
            bundle = base / 'bundle'; bundle.mkdir(); unpack(study_bundle(config, original), bundle)
            actual = bundle / 'actual'; actual.mkdir()
            for path in original.iterdir(): (actual / path.name).write_bytes(path.read_bytes())
            runner = runtime(bundle)
            report = runner.compare_results(actual)
            self.assertTrue(report['passed']); self.assertTrue(report['all_arrays_exact'])
            (actual / 'metadata.json').write_text(json.dumps(dict(metadata, actual_device='cuda:0')))
            self.assertFalse(runner.compare_results(actual)['passed'])
            (actual / 'metadata.json').write_text(json.dumps(metadata))
            np.savez(actual / 'sparameters.npz', s=complex_values * 1.1, valid_bin=np.array([True, False, True]))
            report = runner.compare_results(actual)
            self.assertFalse(report['passed']); self.assertFalse(report['all_arrays_exact'])
            (actual / 'config.json').write_text(json.dumps(dict(config, core_index=2.2)))
            self.assertIsNone(runner.compare_results(actual)['passed'])

    def test_edited_code_and_reference_cannot_silently_claim_gui_reproduction(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); unpack(study_bundle(WaveguideConfig().model_dump()), folder)
            runner = runtime(folder); runner.check_bundle()
            script = folder / 'study.py'; script.write_text(script.read_text().replace("'core_index': 2.0", "'core_index': 2.1"))
            runner.check_bundle()  # Editable settings intentionally remain supported.
            source = folder / 'source/backend/worker.py'; source.write_text(source.read_text() + '\n# changed solver\n')
            with self.assertRaises(ValueError): runner.check_bundle()

    def test_gpu_or_invalid_geometry_cannot_be_silently_downgraded_in_exports(self):
        with self.assertRaises(ValueError): study_bundle(dict(WaveguideConfig().model_dump(), solver='meep', device='gpu'))
        geometry = example3d().geometry.model_dump(by_alias=True)
        geometry['objects'][1]['unexpected'] = 'ignored?'
        with self.assertRaises(ValueError): geometry_bundle(geometry)

    def test_notebook_plot_backend_does_not_leak_into_isolated_solver_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); unpack(study_bundle(WaveguideConfig().model_dump()), folder)
            runner = runtime(folder)
            with patch.dict(runner.os.environ, {'MPLBACKEND': 'module://matplotlib_inline.backend_inline'}), \
                 patch.object(runner.subprocess, 'Popen', side_effect=RuntimeError('execution deliberately intercepted')) as process:
                with self.assertRaisesRegex(RuntimeError, 'execution deliberately intercepted'):
                    runner.run(WaveguideConfig().model_dump(), output=folder / 'run')
                self.assertEqual(process.call_args.kwargs['env']['MPLBACKEND'], 'Agg')
                self.assertEqual(process.call_args.kwargs['env']['OMP_NUM_THREADS'], '2')


if __name__ == '__main__':
    unittest.main()
