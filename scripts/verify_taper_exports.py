"""Replay one real taper browser download twice outside the checkout.

Use the pinned notebook/controller Python. Set MEEP_PYTHON or FDTDX_PYTHON to
its solver interpreter. --browser-artifacts is an output folder of the real
frontend/browser-taper-smoke.cjs journey. --output must be a new directory.
"""
from __future__ import annotations

import argparse
from base64 import b64decode
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import zipfile

from verify_design_exports import execute_notebook

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def unpack_checked(archive_path, destination):
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read('bundle-manifest.json'))
        if manifest['snapshot_origin'] != 'run_time' or manifest['kind'] != 'executable-study':
            raise ValueError('Expected a completed GUI run with captured numerical source.')
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == set(manifest['files']) | {'bundle-manifest.json'}
        for name, record in manifest['files'].items():
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                raise ValueError('Unexpected bundle path.')
            data = archive.read(name)
            assert len(data) == record['size_bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
        archive.extractall(destination)
    return manifest


def array_bytes_equal(reference, candidate):
    import numpy as np

    with np.load(reference, allow_pickle=False) as left, np.load(candidate, allow_pickle=False) as right:
        return (set(left.files) == set(right.files) and
                all(left[key].tobytes() == right[key].tobytes() for key in left.files))


def comparison(directory, expected, actual_device, reference):
    result = read_json(directory/'reproduction-check.json')
    assert read_json(directory/'config.json') == expected
    metadata = read_json(directory/'metadata.json')
    assert metadata['config'] == expected and metadata['actual_device'] == actual_device
    for key in ('passed', 'all_arrays_pointwise_passed', 'physical_results_passed', 'actual_target_preserved',
                'requested_target_preserved', 'versions_match', 'recorded_implementation_preserved'):
        if result[key] is not True:
            raise ValueError('Replay verification failed for ' + key)
    records = result['arrays']['arrays.npz']
    assert {'fields', 'field', 'epsilon', 'x_um', 'y_um', 'z_um'} <= records.keys()
    assert all(value['pointwise_passed'] for value in records.values())
    return {'passed': True, 'all_arrays_exact': result['all_arrays_exact'],
            'all_array_bytes_equal': array_bytes_equal(reference/'arrays.npz', directory/'arrays.npz'),
            'all_arrays_pointwise_passed': True, 'rtol': result['rtol'], 'atol': result['atol'],
            'array_names': list(records), 'field_relative_l2': records['fields']['relative_l2'],
            'field_max_abs_error': records['fields']['max_abs_error'], 'actual_device': actual_device,
            'recorded_physical_times': result['recorded_physical_times'], 'versions_match': True,
            'implementation_preserved': True, 'source_sha256': metadata['implementation_snapshot']['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser-artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    if folder == ROOT or ROOT in folder.parents:
        parser.error('Choose a new directory outside the checkout.')
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = read_json(args.browser_artifacts/'taper-browser.json')
    assert record['passed'] and record['new_solver_jobs'] == 1 and not record['preflight']
    archive_path = args.browser_artifacts/'taper-run-python-notebook.zip'
    manifest = unpack_checked(archive_path, folder)
    config = read_json(folder/'config.json')
    reference = read_json(folder/'reference/metadata.json')
    assert config['schema_version'] == 5 and config['study_type'] == 'taper-3d'
    assert config == read_json(args.browser_artifacts/'taper-run-config.json') == reference['config']
    assert record['target'] == config['solver'] + ':' + config['device']
    actual_device = record['actual_device']
    assert reference['actual_device'] == actual_device
    solver_var = config['solver'].upper() + '_PYTHON'
    if not Path(os.environ.get(solver_var, '')).is_file():
        raise ValueError('Set ' + solver_var + ' to the installed solver interpreter.')
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(folder/'.matplotlib'))
    env.pop('PYTHONPATH', None)
    with (folder/'python.log').open('w') as log:
        subprocess.run([sys.executable, 'study.py', '--output', 'script-results'], cwd=folder, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    script_result = comparison(folder/'script-results', config, actual_device, folder/'reference')
    print(json.dumps({'target': record['target'], 'script': 'passed', 'arrays_exact': script_result['all_arrays_exact']}), flush=True)
    execute_notebook(folder, 'study.ipynb', timeout=1800)
    executed = read_json(folder/'executed-study.ipynb')
    assert all(cell['execution_count'] is not None for cell in executed['cells'] if cell['cell_type']=='code')
    plot = next(cell for cell in executed['cells'] if cell['id'] == 'plot-saved-data')
    encoded = next(o['data']['image/png'] for o in plot['outputs'] if 'image/png' in o.get('data', {}))
    image = b64decode(encoded)
    assert image.startswith(b'\x89PNG\r\n\x1a\n') and len(image)>1000
    notebook_outputs = list((folder/'output').glob('*/reproduction-check.json'))
    assert len(notebook_outputs)==1
    notebook_result = comparison(notebook_outputs[0].parent, config, actual_device, folder/'reference')
    evidence = {'passed': True, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                'source': 'Python and notebook ZIP downloaded through the real GUI',
                'target': record['target'], 'run_id': record['run_id'], 'device_name': record['device_name'],
                'archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                'config_sha256': manifest['files']['config.json']['sha256'],
                'implementation_sha256': reference['implementation_snapshot']['sha256'],
                'script': script_result, 'notebook': notebook_result, 'notebook_plot_png_bytes': len(image),
                'notebook_environment': {name: importlib.metadata.version(name) for name in ('nbclient','nbformat','ipykernel','matplotlib')},
                'wall_seconds': round(time.perf_counter() - started, 2)}
    (folder/'case-report.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'target': record['target'], 'notebook': 'passed', 'arrays_exact': notebook_result['all_arrays_exact']}))


if __name__ == '__main__':
    main()
