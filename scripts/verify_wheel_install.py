"""Exercise a built wheel in a new offline Python 3.11 environment outside the checkout."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--wheelhouse', type=Path, required=True)
    parser.add_argument('--meep-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory outside the source checkout')
    args = parser.parse_args()
    target = args.output.resolve()
    if target == ROOT or ROOT in target.parents:
        parser.error('Choose a new directory outside the checkout.')
    target.mkdir(parents=True, exist_ok=False)
    wheel = args.wheel.resolve()
    meep = args.meep_python.absolute()
    state = target / 'state'
    env = dict(os.environ, PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
               PHOTONICS_DATA_DIR=str(state), MEEP_PYTHON=str(meep), PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    env.pop('FDTDX_PYTHON', None)
    venv.EnvBuilder(with_pip=True).create(target / 'venv')
    python = target / 'venv/bin/python'
    cli = target / 'venv/bin/photonics-workbench'

    def command(argv, log, **kw):
        with (target / log).open('w') as stream:
            subprocess.run(argv, cwd=target, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=300, **kw)

    command([python, '-m', 'pip', 'install', '--no-index', '--find-links', args.wheelhouse.resolve(), wheel], 'install.log')
    command([python, '-m', 'pip', 'check'], 'dependency-check.log')
    command([cli, 'doctor', '--json'], 'doctor.json')
    capabilities = json.loads((target / 'doctor.json').read_text())
    assert any(t['solver'] == 'meep' and t['device'] == 'cpu' and t['available'] for t in capabilities['targets'])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'

    def get(route, payload=None, raw=False):
        req = Request(base + route, data=json.dumps(payload).encode() if payload is not None else None,
                      headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=65) as response:
            body = response.read()
        return body if raw else json.loads(body)

    @contextmanager
    def server(name):
        with (target / name).open('w') as log:
            proc = subprocess.Popen([cli, 'web', '--port', str(port)], cwd=target, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(100):
                    if proc.poll() is not None:
                        raise RuntimeError(f'Installed server exited. See {name}.')
                    try:
                        if get('/api/health').get('status') == 'ok':
                            break
                    except OSError:
                        time.sleep(.1)
                else:
                    raise RuntimeError('The installed service did not become ready.')
                yield
            finally:
                proc.terminate()
                proc.wait(timeout=30)

    with server('web.log'):
        html = get('/', raw=True).decode()
        assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html)
        assert any(a.endswith('.js') for a in assets) and any(a.endswith('.css') for a in assets)
        with zipfile.ZipFile(wheel) as archive:
            for asset in assets:
                assert get(asset, raw=True) == archive.read('photonics_workbench/_app/frontend/dist' + asset)
            wheel_names = set(archive.namelist())
            assert all(name.startswith('photonics_workbench/') or '.dist-info/' in name for name in wheel_names)
            assert not any(set(name.split('/')) & {'.local', 'runs', '.git', '.env', 'node_modules', '__pycache__'} for name in wheel_names)
        assert get('/api/benchmarks')['status'] == 'available'
        assert len(get('/api/library')['components']) == 20
        config = get('/api/taper/example')
        config['geometry'] = get('/api/library/build', {'component_id': 'taper', 'parameters': {
            'length_um': 3., 'height_um': .3, 'input_width_um': .5, 'output_width_um': 1.15}})['geometry']
        config.update(resolution=16, duration_fs=40.)
        assert get('/api/taper/validate', config)['valid']
        job = get('/api/taper/jobs', config)
        jid = job['id']
        for _ in range(350):
            job = get(f'/api/jobs/{jid}')
            if job['status'] in ('failed', 'cancelled', 'completed'):
                break
            time.sleep(1)
        assert job['status'] == 'completed', job.get('error')
        assert job['result']['actual_device'] == 'CPU'
        volume = get(f'/api/jobs/{jid}/volume')
        sample = get(f'/api/jobs/{jid}/slice?plane=xy&component=Ey')
        assert max(abs(value) for row in sample['field'] for value in row) > 0
        result_store = state / 'runs' / jid
        assert (result_store / 'arrays.npz').is_file()
        with zipfile.ZipFile(io.BytesIO(get(f'/api/jobs/{jid}/export', raw=True))) as result:
            assert result.read('arrays.npz') == (result_store / 'arrays.npz').read_bytes()
        with zipfile.ZipFile(io.BytesIO(get(f'/api/jobs/{jid}/reproduce', raw=True))) as exported:
            assert json.loads(exported.read('bundle-manifest.json'))['base_revision'] is None
            assert 'study.ipynb' in exported.namelist()
            exported.extractall(target / 'export')
        # The same data directory cannot be claimed concurrently by two CLI servers.
        with socket.socket() as alternate:
            alternate.bind(('127.0.0.1', 0))
            alternate_port = alternate.getsockname()[1]
        second = subprocess.run([cli, 'web', '--port', str(alternate_port)], cwd=target, env=env,
                                capture_output=True, text=True, timeout=10)
        assert second.returncode and 'already using' in second.stderr
    with server('offline-restart.log'):
        assert get(f'/api/jobs/{jid}')['status'] == 'completed'
    command([cli, 'run', result_store / 'config.json', '--output', target / 'cli-run'], 'cli-run.log')
    command([python, target / 'export/study.py', '--solver-python', meep,
             '--output', target / 'replayed'], 'replay.log')
    result = json.loads((target / 'replayed/reproduction-check.json').read_text())
    assert result['passed'] and result['all_arrays_exact']
    # Compare array contents, not NPZ container metadata, for CLI equivalence.
    command([python, '-c', 'import sys,numpy as n; a=n.load(sys.argv[1]); b=n.load(sys.argv[2]); '
             'assert set(a.files)==set(b.files) and all(n.array_equal(a[k],b[k]) for k in a.files)',
             result_store / 'arrays.npz', target / 'cli-run/arrays.npz'], 'cli-comparison.log')
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'passed': True, 'wheel': wheel.name,
              'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(), 'wheel_bytes': wheel.stat().st_size,
              'installed_from_offline_wheelhouse': True, 'pip_check': True, 'doctor_meep_ready': True,
              'served_frontend_assets': assets, 'catalog_families': 20, 'benchmark_data': True,
              'job_id': jid, 'actual_device': job['result']['actual_device'], 'field_shape_zyx': volume['shape'],
              'state_separate_from_package': True, 'exclusive_store': True, 'saved_run_after_restart': True,
              'cli_run_all_arrays_exact': True, 'standalone_python_all_arrays_exact': True,
              'solver_versions': job['result']['versions'], 'state': 'Fresh disposable user state and virtual environment.'}
    (target / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
